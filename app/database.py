"""
Database layer for storing sentiment, prices, news articles, and predictions.

NOTE: The sentiment_history table stores raw daily_sentiment (simple mean).
Cross-day decay is applied at retrieval time by sentiment_service.py

Database: Turso (libsql) — remote, persistent across deploys.
"""

import json
import os
import math
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any
import pandas as pd
import logging

try:
    import libsql_experimental as libsql  # type: ignore

    _USE_EXPERIMENTAL_LIBSQL = True
except ModuleNotFoundError:
    import libsql_client

    _USE_EXPERIMENTAL_LIBSQL = False

logger = logging.getLogger(__name__)

DATE_BETWEEN_CLAUSE = " WHERE date >= ? AND date <= ?"
DATE_FROM_CLAUSE = " WHERE date >= ?"
DATE_TO_CLAUSE = " WHERE date <= ?"
PREDICTION_COMPARE_LOOKBACK_BUFFER_DAYS = 45
PREDICTION_BOUNDS_Z_SCORE_95 = 1.96
PREDICTION_BOUNDS_DEFAULT_ERROR_STD = 0.015
COMPARE_ERROR_STD_CANDIDATES = [
    Path(__file__).resolve().parent.parent / "model_artifacts" / "error_stds_h5.json",
    Path(__file__).resolve().parent / "newModel" / "error_stds_h5.json",
]


def _parse_error_stds_payload(raw: Any) -> Dict[int, float]:
    """Normalize error std JSON payload into {horizon: std} mapping."""
    if not isinstance(raw, dict):
        return {}

    parsed: Dict[int, float] = {}
    for k, v in raw.items():
        try:
            hk = int(k)
            hv = float(v)
        except (TypeError, ValueError):
            continue
        if hk >= 1 and hv >= 0:
            parsed[hk] = hv
    return parsed


def _load_error_stds_from_file(path: Path) -> Dict[int, float]:
    """Load and parse one error std JSON file, returning empty map on failure."""
    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    return _parse_error_stds_payload(raw)


def _load_error_stds_for_compare() -> Dict[int, float]:
    """Load per-horizon error stds for compare-time bound derivation."""
    for path in COMPARE_ERROR_STD_CANDIDATES:
        parsed = _load_error_stds_from_file(path)
        if parsed:
            return parsed

    return {}


_COMPARE_ERROR_STDS = _load_error_stds_for_compare()


def _get_compare_error_std(horizon: int) -> float:
    """Return horizon std with fallback logic matching prediction behavior."""
    if horizon in _COMPARE_ERROR_STDS:
        return float(_COMPARE_ERROR_STDS[horizon])

    if _COMPARE_ERROR_STDS:
        max_known = max(_COMPARE_ERROR_STDS.keys())
        if horizon > max_known:
            return float(_COMPARE_ERROR_STDS[max_known])

    return PREDICTION_BOUNDS_DEFAULT_ERROR_STD


def _safe_float(value: Any) -> Optional[float]:
    """Convert value to float when possible, else return None."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_forecast_horizon(forecast: Dict[str, Any], idx: int) -> int:
    """Return a valid forecast horizon integer."""
    try:
        horizon = int(forecast.get("horizon") or idx)
    except (TypeError, ValueError):
        horizon = idx
    return max(1, horizon)


def _infer_forecast_return(
    forecast: Dict[str, Any],
    pred_price: Optional[float],
    current_price: float,
) -> Optional[float]:
    """Resolve forecasted return from stored value or implied price step."""
    ret = _safe_float(forecast.get("forecasted_return"))
    if ret is not None:
        return ret

    if pred_price is None or pred_price <= 0 or current_price <= 0:
        return None
    return math.log(pred_price / current_price)


def _derive_bounds_from_return(
    current_price: float,
    ret: float,
    horizon: int,
) -> tuple[float, float]:
    """Compute lower/upper bounds using compare-time error stds."""
    std = _get_compare_error_std(horizon)
    ret_lower = float(ret) - PREDICTION_BOUNDS_Z_SCORE_95 * std
    ret_upper = float(ret) + PREDICTION_BOUNDS_Z_SCORE_95 * std
    lower = max(0.01, float(current_price) * float(math.exp(ret_lower)))
    upper = max(lower, float(current_price) * float(math.exp(ret_upper)))
    return lower, upper


def _advance_forecast_price(
    current_price: float,
    pred_price: Optional[float],
    ret: Optional[float],
) -> float:
    """Advance sequential forecast baseline price for next horizon step."""
    if pred_price is not None and pred_price > 0:
        return pred_price
    if ret is not None:
        return max(0.01, float(current_price) * float(math.exp(ret)))
    return current_price


def _enrich_forecasts_with_missing_bounds(
    forecasts: List[Dict[str, Any]],
    last_price_raw: Any,
) -> List[Dict[str, Any]]:
    """Fill missing lower/upper bounds in a forecast run using model-style math."""
    if not forecasts:
        return []

    base_price = _safe_float(last_price_raw)
    if base_price is None or base_price <= 0:
        return forecasts

    enriched: List[Dict[str, Any]] = []
    current_price = float(base_price)

    for idx, item in enumerate(forecasts, start=1):
        if not isinstance(item, dict):
            continue

        forecast = dict(item)
        horizon = _resolve_forecast_horizon(forecast, idx)

        pred_price = _safe_float(forecast.get("forecasted_price"))
        ret = _infer_forecast_return(forecast, pred_price, current_price)

        lower_existing = _safe_float(forecast.get("lower_bound"))
        upper_existing = _safe_float(forecast.get("upper_bound"))
        if ret is not None and (lower_existing is None or upper_existing is None):
            lower, upper = _derive_bounds_from_return(current_price, ret, horizon)
            forecast["lower_bound"] = round(lower, 2)
            forecast["upper_bound"] = round(upper, 2)

        current_price = _advance_forecast_price(current_price, pred_price, ret)

        enriched.append(forecast)

    return enriched


class _LibsqlClientCursor:
    """Minimal DB-API-like cursor shim for libsql_client sync client."""

    def __init__(self, client):
        self._client = client
        self._rows: List[Any] = []
        self._row_index = 0
        self.description = None
        self.rowcount = 0
        self.lastrowid = None

    def execute(self, query: str, params=None):
        result = self._client.execute(query, params or ())

        self._rows = list(result) if result is not None else []
        self._row_index = 0

        columns = getattr(result, "columns", None)
        if columns:
            self.description = [(col,) for col in columns]
        else:
            self.description = None

        self.rowcount = int(getattr(result, "rows_affected", 0) or 0)
        self.lastrowid = getattr(result, "last_insert_rowid", None)
        return self

    def fetchone(self):
        if self._row_index >= len(self._rows):
            return None
        row = self._rows[self._row_index]
        self._row_index += 1
        return row

    def fetchall(self):
        if self._row_index >= len(self._rows):
            return []
        rows = self._rows[self._row_index :]
        self._row_index = len(self._rows)
        return rows


class _LibsqlClientConnection:
    """Minimal connection shim for libsql_client to match existing code paths."""

    def __init__(self, client):
        self._client = client

    def cursor(self):
        return _LibsqlClientCursor(self._client)

    def commit(self):
        # libsql_client executes statements immediately; no explicit commit needed.
        return None

    def rollback(self):
        # libsql_client has no transaction in this shim path.
        return None

    def close(self):
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            close_fn()


def get_connection():
    """Get Turso (libsql) database connection."""
    url = os.environ.get("TURSO_DATABASE_URL")
    auth_token = os.environ.get("TURSO_AUTH_TOKEN", "")

    if _USE_EXPERIMENTAL_LIBSQL:
        return libsql.connect(database=url, auth_token=auth_token)

    if url and url.startswith("libsql://"):
        url = url.replace("libsql://", "https://", 1)

    logger.warning("libsql_experimental unavailable; using libsql_client sync fallback")
    client = libsql_client.create_client_sync(url=url, auth_token=auth_token)
    return _LibsqlClientConnection(client)


def _fetchone_dict(cursor) -> Optional[Dict[str, Any]]:
    """Fetch one row as a dict using cursor column names."""
    if cursor.description is None:
        return None
    cols = [d[0] for d in cursor.description]
    row = cursor.fetchone()
    return dict(zip(cols, row)) if row else None


def _fetchall_dicts(cursor) -> List[Dict[str, Any]]:
    """Fetch all rows as dicts using cursor column names."""
    if cursor.description is None:
        return []
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def _query_to_df(conn, query: str, params=None) -> pd.DataFrame:
    """Execute a SELECT query and return results as a DataFrame."""
    cursor = conn.cursor()
    cursor.execute(query, params or ())
    if cursor.description is None:
        return pd.DataFrame()
    cols = [d[0] for d in cursor.description]
    return pd.DataFrame(cursor.fetchall(), columns=cols)


def _table_has_column(cursor, table_name: str, column_name: str) -> bool:
    """Return True if the given table already contains the target column."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    rows = cursor.fetchall() or []
    return any(len(row) > 1 and row[1] == column_name for row in rows)


def _ensure_table_column(
    cursor,
    table_name: str,
    column_name: str,
    column_sql_type: str,
) -> None:
    """Add a missing column to an existing table."""
    if _table_has_column(cursor, table_name, column_name):
        return
    cursor.execute(
        f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql_type}"
    )


def init_database() -> None:
    """Initialize the sentiment database with required tables."""
    logger.info("Initializing Turso database")

    conn = get_connection()
    cursor = conn.cursor()

    # Create sentiment history table
    # Note: daily_sentiment_decay column stores the RAW daily sentiment
    # (simple mean of article scores). Cross-day decay is applied at read time.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            daily_sentiment_decay REAL NOT NULL,
            news_volume INTEGER NOT NULL,
            log_news_volume REAL NOT NULL,
            decayed_news_volume REAL NOT NULL,
            high_news_regime INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create index on date for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_sentiment_date ON sentiment_history(date)
    """)

    # Create prices table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            price REAL NOT NULL,
            source TEXT DEFAULT 'yahoo_finance',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date)
    """)

    # Create historical prices table (dataset imports)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS historical_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            price REAL NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            volume REAL,
            change_pct REAL,
            source TEXT DEFAULT 'historical_import',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_historical_prices_date ON historical_prices(date)
    """)

    # Create historical news features table (dataset imports)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS historical_news_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            daily_sentiment_decay REAL NOT NULL,
            news_volume REAL NOT NULL,
            log_news_volume REAL NOT NULL,
            decayed_news_volume REAL NOT NULL,
            high_news_regime INTEGER NOT NULL,
            source TEXT DEFAULT 'historical_import',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_historical_news_features_date ON historical_news_features(date)
    """)

    # Convenience full-outer-date view across both historical tables.
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS historical_features_combined AS
        SELECT
            hp.date AS date,
            hp.price AS price,
            hp.open AS open,
            hp.high AS high,
            hp.low AS low,
            hp.volume AS volume,
            hp.change_pct AS change_pct,
            hnf.daily_sentiment_decay AS daily_sentiment_decay,
            hnf.news_volume AS news_volume,
            hnf.log_news_volume AS log_news_volume,
            hnf.decayed_news_volume AS decayed_news_volume,
            hnf.high_news_regime AS high_news_regime
        FROM historical_prices hp
        LEFT JOIN historical_news_features hnf ON hp.date = hnf.date
        UNION ALL
        SELECT
            hnf.date AS date,
            hp.price AS price,
            hp.open AS open,
            hp.high AS high,
            hp.low AS low,
            hp.volume AS volume,
            hp.change_pct AS change_pct,
            hnf.daily_sentiment_decay AS daily_sentiment_decay,
            hnf.news_volume AS news_volume,
            hnf.log_news_volume AS log_news_volume,
            hnf.decayed_news_volume AS decayed_news_volume,
            hnf.high_news_regime AS high_news_regime
        FROM historical_news_features hnf
        LEFT JOIN historical_prices hp ON hp.date = hnf.date
        WHERE hp.date IS NULL
    """)

    # Create news_articles table (one row per article, with per-article sentiment)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_date TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            url TEXT UNIQUE,
            image_url TEXT,
            source TEXT,
            published_at TEXT,
            sentiment_score REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    _ensure_table_column(cursor, "news_articles", "image_url", "TEXT")
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_articles_date ON news_articles(article_date)
    """)

    # Create predictions table (stores each forecast run for the active model horizon)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            generated_at TEXT NOT NULL,
            last_price_date TEXT NOT NULL,
            last_price REAL NOT NULL,
            forecasts TEXT NOT NULL,
            prediction_date TEXT,
            based_on_price_date TEXT,
            based_on_price REAL,
            forecast_day_1 REAL,
            forecast_day_2 REAL,
            forecast_day_3 REAL,
            forecast_day_4 REAL,
            forecast_day_5 REAL,
            locked_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    _ensure_table_column(cursor, "predictions", "prediction_date", "TEXT")
    _ensure_table_column(cursor, "predictions", "based_on_price_date", "TEXT")
    _ensure_table_column(cursor, "predictions", "based_on_price", "REAL")
    _ensure_table_column(cursor, "predictions", "forecast_day_1", "REAL")
    _ensure_table_column(cursor, "predictions", "forecast_day_2", "REAL")
    _ensure_table_column(cursor, "predictions", "forecast_day_3", "REAL")
    _ensure_table_column(cursor, "predictions", "forecast_day_4", "REAL")
    _ensure_table_column(cursor, "predictions", "forecast_day_5", "REAL")
    _ensure_table_column(cursor, "predictions", "locked_at", "TEXT")
    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_prediction_date
        ON predictions(prediction_date)
        """)
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_predictions_last_price_date
        ON predictions(last_price_date)
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_predictions_generated_at
        ON predictions(generated_at)
        """
    )

    # Create explanations table (stores daily explainability results)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS explanations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            explanation_date TEXT UNIQUE NOT NULL,
            prediction REAL NOT NULL,
            confidence_interval_lower REAL NOT NULL,
            confidence_interval_upper REAL NOT NULL,
            arima_contribution REAL NOT NULL,
            gru_mid_contribution REAL NOT NULL,
            gru_sent_contribution REAL NOT NULL,
            xgb_hf_contribution REAL NOT NULL,
            agreement_score REAL NOT NULL,
            confidence_level TEXT NOT NULL,
            top_shap_features TEXT NOT NULL,
            sentiment_headlines TEXT NOT NULL,
            explanation_text TEXT NOT NULL,
            model_weights TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            computation_time_seconds REAL NOT NULL,
            xai_payload TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_explanations_date ON explanations(explanation_date)
    """)

    # Migrate existing DB instances: add xai_payload column if absent
    try:
        cursor.execute("ALTER TABLE explanations ADD COLUMN xai_payload TEXT")
        conn.commit()
        logger.info("Migrated explanations table: added xai_payload column")
    except Exception:
        pass  # Column already exists — safe to ignore

    # Key-value store for persisting miscellaneous runtime state across restarts
    # (e.g. FinBERT timing metrics from the last scraper run)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS kv_store (
            key TEXT PRIMARY KEY NOT NULL,
            value TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    logger.info(
        "Database initialized successfully (sentiment, prices, articles, predictions, explanations, kv_store)"
    )


def add_sentiment(
    date_str: str,
    daily_sentiment_decay: float,
    news_volume: int,
    log_news_volume: float,
    decayed_news_volume: float,
    high_news_regime: int,
) -> bool:
    """
    Add or update sentiment data for a specific date.

    Args:
        date_str: Date in YYYY-MM-DD format
        daily_sentiment_decay: Raw daily sentiment (simple mean, no cross-day decay)
        news_volume: Number of news articles
        log_news_volume: Log-transformed volume
        decayed_news_volume: EWM-based volume estimate
        high_news_regime: Binary flag (0 or 1)

    Returns:
        True if successful
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT OR REPLACE INTO sentiment_history 
            (date, daily_sentiment_decay, news_volume, log_news_volume, 
             decayed_news_volume, high_news_regime)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                date_str,
                daily_sentiment_decay,
                news_volume,
                log_news_volume,
                decayed_news_volume,
                high_news_regime,
            ),
        )

        conn.commit()
        logger.info(f"Added sentiment for {date_str}: {daily_sentiment_decay:.4f}")
        return True

    except Exception as e:
        logger.error(f"Error adding sentiment: {e}")
        conn.rollback()
        raise

    finally:
        conn.close()


def add_bulk_sentiment(sentiment_list: List[Dict[str, Any]]) -> int:
    """
    Add multiple sentiment records at once.

    Args:
        sentiment_list: List of sentiment dictionaries

    Returns:
        Number of records added
    """
    conn = get_connection()
    cursor = conn.cursor()

    count = 0
    try:
        for record in sentiment_list:
            # Support both 'daily_sentiment' and 'daily_sentiment_decay' keys
            sentiment_value = record.get(
                "daily_sentiment_decay", record.get("daily_sentiment", 0.0)
            )

            cursor.execute(
                """
                INSERT OR REPLACE INTO sentiment_history 
                (date, daily_sentiment_decay, news_volume, log_news_volume, 
                 decayed_news_volume, high_news_regime)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    record["date"],
                    sentiment_value,
                    record["news_volume"],
                    record["log_news_volume"],
                    record["decayed_news_volume"],
                    record["high_news_regime"],
                ),
            )
            count += 1

        conn.commit()
        logger.info(f"Added {count} sentiment records")
        return count

    except Exception as e:
        logger.error(f"Error in bulk add: {e}")
        conn.rollback()
        raise

    finally:
        conn.close()


def get_sentiment_history(days: int = 60) -> pd.DataFrame:
    """
    Get sentiment history for the last N days.

    Note: Returns column as 'daily_sentiment' for feature_engineering.py
    which will apply its own alpha decay.

    Args:
        days: Number of days of history to retrieve

    Returns:
        DataFrame with sentiment data (daily_sentiment column = raw daily mean)
    """
    conn = get_connection()

    # Rename column to daily_sentiment for feature_engineering.py compatibility
    query = """
        SELECT date, daily_sentiment_decay as daily_sentiment, news_volume, log_news_volume,
               decayed_news_volume, high_news_regime
        FROM sentiment_history
        ORDER BY date DESC
        LIMIT ?
    """

    df = _query_to_df(conn, query, params=(days,))
    conn.close()

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    return df


def get_sentiment_for_dates(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get sentiment data for a specific date range.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with sentiment data
    """
    conn = get_connection()

    query = """
        SELECT date, daily_sentiment_decay as daily_sentiment, news_volume, log_news_volume,
               decayed_news_volume, high_news_regime
        FROM sentiment_history
        WHERE date >= ? AND date <= ?
        ORDER BY date
    """

    df = _query_to_df(conn, query, params=(start_date, end_date))
    conn.close()

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])

    return df


def get_all_sentiment_history() -> pd.DataFrame:
    """
    Get all historical sentiment data from 2014 onwards.

    Returns:
        DataFrame with all sentiment data ordered by date
    """
    conn = get_connection()

    query = """
        SELECT date, daily_sentiment_decay as daily_sentiment, news_volume, log_news_volume,
               decayed_news_volume, high_news_regime
        FROM sentiment_history
        ORDER BY date
    """

    df = _query_to_df(conn, query, params=())
    conn.close()

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])

    return df


def get_latest_sentiment() -> Optional[Dict[str, Any]]:
    """Get the most recent sentiment record."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT date, daily_sentiment_decay, news_volume, log_news_volume,
               decayed_news_volume, high_news_regime
        FROM sentiment_history
        ORDER BY date DESC
        LIMIT 1
    """)

    result = _fetchone_dict(cursor)
    conn.close()
    return result


def get_sentiment_count() -> int:
    """Get total number of sentiment records."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sentiment_history")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def clear_sentiment_history() -> int:
    """
    Clear all sentiment records from the database.

    Returns:
        Number of records deleted
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM sentiment_history")
    count = cursor.fetchone()[0]

    cursor.execute("DELETE FROM sentiment_history")
    conn.commit()
    conn.close()

    logger.info(f"Cleared {count} sentiment records")
    return count


# ---------------------------------------------------------------------------
# Price functions
# ---------------------------------------------------------------------------


def add_price(date_str: str, price: float, source: str = "yahoo_finance") -> bool:
    """Insert or replace a single daily price record."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT OR REPLACE INTO prices (date, price, source) VALUES (?, ?, ?)",
            (date_str, price, source),
        )
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error adding price for {date_str}: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def add_bulk_prices(price_records: List[Dict[str, Any]]) -> int:
    """
    Insert or replace multiple price records.

    Each record must have 'date' and 'price' keys.
    Optional 'source' key (defaults to 'yahoo_finance').
    """
    if not price_records:
        return 0

    conn = get_connection()
    cursor = conn.cursor()

    # Batch values into chunked multi-row INSERT statements to reduce remote DB round-trips.
    chunk_size = 200
    rows = [
        (
            rec["date"],
            float(rec["price"]),
            rec.get("source", "yahoo_finance"),
        )
        for rec in price_records
    ]

    try:
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i : i + chunk_size]
            placeholders = ", ".join(["(?, ?, ?)"] * len(chunk))
            query = (
                "INSERT OR REPLACE INTO prices (date, price, source) VALUES "
                f"{placeholders}"
            )
            params = [value for row in chunk for value in row]
            cursor.execute(query, params)

        conn.commit()
        count = len(rows)
        logger.info(f"Saved {count} price records")
        return count
    except Exception as e:
        logger.error(f"Error in bulk add prices: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def get_latest_price_date() -> Optional[str]:
    """Return the most recent date string (YYYY-MM-DD) stored in the prices table, or None."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT MAX(date) FROM prices")
        row = cursor.fetchone()
        if row and row[0]:
            return str(row[0])
        return None
    except Exception as e:
        logger.error("Error fetching latest price date: %s", e)
        return None
    finally:
        conn.close()


def get_prices(days: int = 90) -> pd.DataFrame:
    """Return the most recent N days of stored price data."""
    conn = get_connection()
    df = _query_to_df(
        conn,
        "SELECT date, price, source FROM prices ORDER BY date DESC LIMIT ?",
        params=(days,),
    )
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


def get_prices_for_date_range(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Return price rows within [start_date, end_date] from existing DB tables.

    Preference order when both tables contain the same date:
    1) prices
    2) historical_prices
    """
    conn = get_connection()

    query = """
        SELECT date, price, source, 1 AS priority
        FROM prices
        WHERE date >= ? AND date <= ?
        UNION ALL
        SELECT date, price, source, 2 AS priority
        FROM historical_prices
        WHERE date >= ? AND date <= ?
    """

    df = _query_to_df(
        conn,
        query,
        params=(start_date, end_date, start_date, end_date),
    )
    conn.close()

    if df.empty:
        return pd.DataFrame(columns=["date", "price", "source"])

    df["date"] = pd.to_datetime(df["date"])
    df = (
        df.sort_values(["date", "priority"])
        .drop_duplicates(subset=["date"], keep="first")
        .drop(columns=["priority"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    return df


def _to_float(value: Any) -> Optional[float]:
    """Convert mixed numeric text formats (%, K/M/B suffixes, commas) to float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric_value = float(value)
        return numeric_value if math.isfinite(numeric_value) else None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "-"}:
        return None

    multiplier = 1.0
    suffix = text[-1].upper()
    if suffix in {"K", "M", "B"}:
        text = text[:-1]
        multiplier = {"K": 1_000.0, "M": 1_000_000.0, "B": 1_000_000_000.0}[suffix]

    text = text.replace(",", "").replace("%", "")
    text = re.sub(r"[^0-9.+-]", "", text)

    if not text:
        return None

    try:
        numeric_value = float(text) * multiplier
        return numeric_value if math.isfinite(numeric_value) else None
    except ValueError:
        return None


def add_bulk_historical_prices(
    price_records: List[Dict[str, Any]],
    default_source: str = "historical_import",
) -> int:
    """
    Insert or replace multiple historical price records.

    Required keys per record:
    - date
    - price
    Optional keys:
    - open, high, low, volume, change_pct, source
    """
    conn = get_connection()
    cursor = conn.cursor()
    count = 0
    try:
        chunk_size = 100
        for i in range(0, len(price_records), chunk_size):
            chunk = price_records[i : i + chunk_size]
            values = []
            for rec in chunk:
                values.extend(
                    [
                        rec["date"],
                        _to_float(rec["price"]),
                        _to_float(rec.get("open")),
                        _to_float(rec.get("high")),
                        _to_float(rec.get("low")),
                        _to_float(rec.get("volume")),
                        _to_float(rec.get("change_pct")),
                        rec.get("source", default_source),
                    ]
                )

            placeholders = ", ".join(["(?, ?, ?, ?, ?, ?, ?, ?)"] * len(chunk))
            query = (
                "INSERT OR REPLACE INTO historical_prices "
                "(date, price, open, high, low, volume, change_pct, source) "
                f"VALUES {placeholders}"
            )
            cursor.execute(query, tuple(values))
            count += len(chunk)

        conn.commit()
        logger.info(f"Saved {count} historical price records")
        return count
    except Exception as e:
        logger.error(f"Error in bulk add historical prices: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def add_bulk_historical_news_features(
    feature_records: List[Dict[str, Any]],
    default_source: str = "historical_import",
) -> int:
    """
    Insert or replace multiple historical news feature rows.

    Required keys per record:
    - date, daily_sentiment_decay, news_volume, log_news_volume,
      decayed_news_volume, high_news_regime
    """
    conn = get_connection()
    cursor = conn.cursor()
    count = 0
    try:
        chunk_size = 100
        for i in range(0, len(feature_records), chunk_size):
            chunk = feature_records[i : i + chunk_size]
            values = []
            for rec in chunk:
                values.extend(
                    [
                        rec["date"],
                        _to_float(rec["daily_sentiment_decay"]),
                        _to_float(rec["news_volume"]),
                        _to_float(rec["log_news_volume"]),
                        _to_float(rec["decayed_news_volume"]),
                        int(float(rec["high_news_regime"])),
                        rec.get("source", default_source),
                    ]
                )

            placeholders = ", ".join(["(?, ?, ?, ?, ?, ?, ?)"] * len(chunk))
            query = (
                "INSERT OR REPLACE INTO historical_news_features "
                "(date, daily_sentiment_decay, news_volume, log_news_volume, "
                "decayed_news_volume, high_news_regime, source) "
                f"VALUES {placeholders}"
            )
            cursor.execute(query, tuple(values))
            count += len(chunk)

        conn.commit()
        logger.info(f"Saved {count} historical news feature records")
        return count
    except Exception as e:
        logger.error(f"Error in bulk add historical news features: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def get_historical_features_combined(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> pd.DataFrame:
    """Return joined historical price and news features over an optional date range."""
    conn = get_connection()

    base_query = """
        SELECT date, price, open, high, low, volume, change_pct,
               daily_sentiment_decay, news_volume, log_news_volume,
               decayed_news_volume, high_news_regime
        FROM historical_features_combined
    """
    params: List[Any] = []

    if start_date and end_date:
        base_query += DATE_BETWEEN_CLAUSE
        params.extend([start_date, end_date])
    elif start_date:
        base_query += DATE_FROM_CLAUSE
        params.append(start_date)
    elif end_date:
        base_query += DATE_TO_CLAUSE
        params.append(end_date)

    base_query += " ORDER BY date"

    if limit is not None:
        base_query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

    df = _query_to_df(conn, base_query, tuple(params))
    conn.close()

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])

    return df


def get_historical_features_combined_count(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> int:
    """Return total row count in combined historical view for an optional date range."""
    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT COUNT(*) FROM historical_features_combined"
    params: List[Any] = []
    if start_date and end_date:
        query += DATE_BETWEEN_CLAUSE
        params.extend([start_date, end_date])
    elif start_date:
        query += DATE_FROM_CLAUSE
        params.append(start_date)
    elif end_date:
        query += DATE_TO_CLAUSE
        params.append(end_date)

    cursor.execute(query, tuple(params))
    count = cursor.fetchone()[0]
    conn.close()
    return int(count)


def get_historical_prices(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> pd.DataFrame:
    """Return historical price dataset over an optional date range."""
    conn = get_connection()

    base_query = """
        SELECT date, price, open, high, low, volume, change_pct, source
        FROM historical_prices
    """
    params: List[Any] = []

    if start_date and end_date:
        base_query += DATE_BETWEEN_CLAUSE
        params.extend([start_date, end_date])
    elif start_date:
        base_query += DATE_FROM_CLAUSE
        params.append(start_date)
    elif end_date:
        base_query += DATE_TO_CLAUSE
        params.append(end_date)

    base_query += " ORDER BY date"

    if limit is not None:
        base_query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

    df = _query_to_df(conn, base_query, tuple(params))
    conn.close()

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])

    return df


def get_historical_prices_count(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> int:
    """Return total historical price rows for an optional date range."""
    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT COUNT(*) FROM historical_prices"
    params: List[Any] = []
    if start_date and end_date:
        query += DATE_BETWEEN_CLAUSE
        params.extend([start_date, end_date])
    elif start_date:
        query += DATE_FROM_CLAUSE
        params.append(start_date)
    elif end_date:
        query += DATE_TO_CLAUSE
        params.append(end_date)

    cursor.execute(query, tuple(params))
    count = cursor.fetchone()[0]
    conn.close()
    return int(count)


def get_historical_prices_paginated(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 500,
    offset: int = 0,
) -> tuple:
    """
    Return paginated historical prices *and* total row count in a single
    Turso round-trip using a window function.

    Returns:
        (df, total_count)
    """
    query = """
        SELECT date, price, open, high, low, volume, change_pct, source,
               COUNT(*) OVER() AS total_count
        FROM historical_prices
        WHERE (? IS NULL OR date >= ?)
          AND (? IS NULL OR date <= ?)
        ORDER BY date
        LIMIT ? OFFSET ?
    """
    params = (
        start_date,
        start_date,
        end_date,
        end_date,
        limit,
        offset,
    )

    conn = get_connection()
    df = _query_to_df(conn, query, params)
    conn.close()

    if df.empty:
        return df, 0

    total_count = int(df["total_count"].iloc[0])
    df = df.drop(columns=["total_count"])
    df["date"] = pd.to_datetime(df["date"])
    return df, total_count


def get_historical_prices_aggregated(
    granularity: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 500,
    offset: int = 0,
) -> tuple:
    """
    Aggregate historical prices at the DB level (weekly or monthly) with
    LIMIT/OFFSET pagination, returning (df, total_count) in one round-trip.

    This avoids fetching all rows into Python for server-side slicing.
    """
    fmt = "%Y-%W" if granularity == "weekly" else "%Y-%m"

    # Use CTEs to:
    # 1. rank rows within each period so we can pick first/last for open/close
    # 2. aggregate per period
    # 3. paginate + attach total count in one shot
    query = """
        WITH ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY strftime(?, date) ORDER BY date ASC)  AS rn_asc,
                   ROW_NUMBER() OVER (PARTITION BY strftime(?, date) ORDER BY date DESC) AS rn_desc
            FROM historical_prices
            WHERE (? IS NULL OR date >= ?)
              AND (? IS NULL OR date <= ?)
        ),
        agg AS (
            SELECT
                MIN(date)                                               AS date,
                MAX(CASE WHEN rn_asc  = 1 THEN open  END)              AS open,
                MAX(high)                                               AS high,
                MIN(low)                                                AS low,
                MAX(CASE WHEN rn_desc = 1 THEN price END)              AS price,
                SUM(volume)                                             AS volume,
                AVG(change_pct)                                         AS change_pct,
                MIN(source)                                             AS source
            FROM ranked
            GROUP BY strftime(?, date)
        )
        SELECT date, price, open, high, low, volume, change_pct, source,
               COUNT(*) OVER() AS total_count
        FROM agg
        ORDER BY date
        LIMIT ? OFFSET ?
    """
    params = (
        fmt,
        fmt,
        start_date,
        start_date,
        end_date,
        end_date,
        fmt,
        limit,
        offset,
    )

    conn = get_connection()
    df = _query_to_df(conn, query, params)
    conn.close()

    if df.empty:
        return df, 0

    total_count = int(df["total_count"].iloc[0])
    df = df.drop(columns=["total_count"])
    df["date"] = pd.to_datetime(df["date"])
    return df, total_count


# ---------------------------------------------------------------------------
# News article functions
# ---------------------------------------------------------------------------


def add_news_articles(article_date: str, articles: List[Dict[str, Any]]) -> int:
    """
    Store a list of news articles for a given date.

    Each article dict should contain:
        title, description, url, image_url, source, published_at, sentiment_score
    Duplicate URLs update image_url when an incoming non-empty image URL is available.

    Returns:
        Number of processed rows.
    """
    conn = get_connection()
    cursor = conn.cursor()
    count = 0
    try:
        for art in articles:
            cursor.execute(
                """
                INSERT INTO news_articles
                    (article_date, title, description, url, image_url, source, published_at, sentiment_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    image_url = CASE
                        WHEN excluded.image_url IS NOT NULL
                             AND excluded.image_url != ''
                        THEN excluded.image_url
                        ELSE news_articles.image_url
                    END
                """,
                (
                    article_date,
                    art.get("title", ""),
                    art.get("description", ""),
                    art.get("url"),
                    art.get("image_url"),
                    art.get("source", ""),
                    art.get("published_at", ""),
                    art.get("sentiment_score"),
                ),
            )
            count += 1
        conn.commit()
        logger.info(f"Processed {count} articles for {article_date}")
        return count
    except Exception as e:
        logger.error(f"Error saving articles for {article_date}: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def get_news_articles(article_date: str) -> List[Dict[str, Any]]:
    """Return all stored articles for a specific date."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, article_date, title, description, url, image_url, source, published_at, sentiment_score, created_at
        FROM news_articles WHERE article_date = ? ORDER BY id
        """,
        (article_date,),
    )
    rows = _fetchall_dicts(cursor)
    conn.close()
    return rows


def get_recent_news_articles(days: int = 7) -> List[Dict[str, Any]]:
    """Return articles from the most recent N distinct dates."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, article_date, title, description, url, image_url, source, published_at, sentiment_score
        FROM news_articles
        WHERE article_date IN (
            SELECT DISTINCT article_date FROM news_articles ORDER BY article_date DESC LIMIT ?
        )
        ORDER BY article_date DESC, id
        """,
        (days,),
    )
    rows = _fetchall_dicts(cursor)
    conn.close()
    return rows


def get_news_articles_missing_image_url(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return articles that do not yet have an image URL."""
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        SELECT id, article_date, title, description
        FROM news_articles
        WHERE (image_url IS NULL OR TRIM(image_url) = '')
    """
    params: List[Any] = []

    if start_date:
        query += " AND article_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND article_date <= ?"
        params.append(end_date)

    query += " ORDER BY article_date ASC, id ASC"

    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    cursor.execute(query, tuple(params))
    rows = _fetchall_dicts(cursor)
    conn.close()
    return rows


def update_news_article_image_url(article_id: int, image_url: str) -> bool:
    """Update image_url for a specific stored news article row."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE news_articles SET image_url = ? WHERE id = ?",
            (image_url, article_id),
        )
        conn.commit()
        return bool(cursor.rowcount)
    except Exception as e:
        logger.error(f"Error updating image_url for article id={article_id}: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def get_existing_image_urls(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> set[str]:
    """Return a set of all image URLs currently stored in the database.
    
    Args:
        start_date: Optional start date to filter (YYYY-MM-DD)
        end_date: Optional end date to filter (YYYY-MM-DD)
    
    Returns:
        Set of image URL strings (excludes NULL and empty strings)
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        query = """
            SELECT DISTINCT image_url
            FROM news_articles
            WHERE image_url IS NOT NULL AND TRIM(image_url) != ''
        """
        params: List[Any] = []
        
        if start_date and end_date:
            query += " AND article_date >= ? AND article_date <= ?"
            params = [start_date, end_date]
        elif start_date:
            query += " AND article_date >= ?"
            params = [start_date]
        elif end_date:
            query += " AND article_date <= ?"
            params = [end_date]
        
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        return {row[0] for row in rows if row and row[0]}
    except Exception as e:
        logger.error(f"Error fetching existing image URLs: {e}")
        return set()
    finally:
        conn.close()


def clear_news_article_image_urls(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> int:
    """Set image_url to NULL for all articles in the given date range.

    Returns the number of rows cleared.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        query = "UPDATE news_articles SET image_url = NULL"
        params: List[Any] = []
        if start_date and end_date:
            query += " WHERE article_date >= ? AND article_date <= ?"
            params = [start_date, end_date]
        elif start_date:
            query += " WHERE article_date >= ?"
            params = [start_date]
        elif end_date:
            query += " WHERE article_date <= ?"
            params = [end_date]
        cursor.execute(query, tuple(params))
        conn.commit()
        return int(cursor.rowcount or 0)
    except Exception as e:
        logger.error(f"Error clearing image_urls ({start_date} → {end_date}): {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Prediction functions
# ---------------------------------------------------------------------------


def add_prediction(
    generated_at: str,
    last_price_date: str,
    last_price: float,
    forecasts: List[Dict[str, Any]],
) -> int:
    """
    Persist a forecast run for the active model horizon.

    Args:
        generated_at: ISO timestamp of when the prediction was made.
        last_price_date: Date string of the last known price used.
        last_price: Last known price value.
        forecasts: List of ForecastDay dicts (serialized as JSON).

    Returns:
        Row id of the inserted record.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO predictions (generated_at, last_price_date, last_price, forecasts)
            VALUES (?, ?, ?, ?)
            """,
            (generated_at, last_price_date, last_price, json.dumps(forecasts)),
        )
        row_id = cursor.lastrowid
        conn.commit()
        logger.info(f"Saved prediction run id={row_id} generated_at={generated_at}")
        return row_id
    except Exception as e:
        logger.error(f"Error saving prediction: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_daily_prediction(
    prediction_date: str,
    based_on_price_date: str,
    based_on_price: float,
    forecasts: List[Dict[str, Any]],
    locked_at: Optional[str] = None,
) -> int:
    """
    Upsert one locked prediction row per prediction_date.

    Args:
        prediction_date: YYYY-MM-DD date key for the daily locked forecast.
        based_on_price_date: Date of official close used as model input.
        based_on_price: Closing price used as model input.
        forecasts: Forecast list from the model (expects at least 5 points).
        locked_at: Optional ISO timestamp for lock time; defaults to now.

    Returns:
        Row id of the inserted/updated prediction row.
    """
    lock_ts = locked_at or datetime.now().isoformat()
    normalized_forecasts = forecasts or []

    # Keep first 5 horizons as explicit columns for easy UI/status access.
    forecast_prices = []
    for item in normalized_forecasts[:5]:
        try:
            forecast_prices.append(float(item.get("forecasted_price")))
        except (TypeError, ValueError, AttributeError):
            forecast_prices.append(None)
    while len(forecast_prices) < 5:
        forecast_prices.append(None)

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO predictions (
                generated_at,
                last_price_date,
                last_price,
                forecasts,
                prediction_date,
                based_on_price_date,
                based_on_price,
                forecast_day_1,
                forecast_day_2,
                forecast_day_3,
                forecast_day_4,
                forecast_day_5,
                locked_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(prediction_date) DO UPDATE SET
                generated_at = excluded.generated_at,
                last_price_date = excluded.last_price_date,
                last_price = excluded.last_price,
                forecasts = excluded.forecasts,
                based_on_price_date = excluded.based_on_price_date,
                based_on_price = excluded.based_on_price,
                forecast_day_1 = excluded.forecast_day_1,
                forecast_day_2 = excluded.forecast_day_2,
                forecast_day_3 = excluded.forecast_day_3,
                forecast_day_4 = excluded.forecast_day_4,
                forecast_day_5 = excluded.forecast_day_5,
                locked_at = excluded.locked_at
            """,
            (
                lock_ts,
                based_on_price_date,
                based_on_price,
                json.dumps(normalized_forecasts),
                prediction_date,
                based_on_price_date,
                based_on_price,
                forecast_prices[0],
                forecast_prices[1],
                forecast_prices[2],
                forecast_prices[3],
                forecast_prices[4],
                lock_ts,
            ),
        )

        row_id = cursor.lastrowid
        if row_id is None:
            cursor.execute(
                "SELECT id FROM predictions WHERE prediction_date = ?",
                (prediction_date,),
            )
            row = cursor.fetchone()
            row_id = int(row[0]) if row else 0

        conn.commit()
        logger.info(
            "Upserted locked daily prediction id=%s prediction_date=%s based_on=%s",
            row_id,
            prediction_date,
            based_on_price_date,
        )
        return int(row_id or 0)
    except Exception as e:
        logger.error(f"Error upserting daily prediction: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def get_prediction_for_date(prediction_date: str) -> Optional[Dict[str, Any]]:
    """Return locked prediction row for a specific YYYY-MM-DD prediction date."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM predictions WHERE prediction_date = ? ORDER BY id DESC LIMIT 1",
        (prediction_date,),
    )
    result = _fetchone_dict(cursor)
    conn.close()

    if not result:
        return None

    try:
        result["forecasts"] = json.loads(result.get("forecasts") or "[]")
    except Exception:
        result["forecasts"] = []
    return result


def get_latest_locked_prediction() -> Optional[Dict[str, Any]]:
    """Return latest locked prediction, preferring prediction_date when available."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT *
        FROM predictions
        ORDER BY
            CASE WHEN prediction_date IS NULL OR prediction_date = '' THEN 1 ELSE 0 END,
            prediction_date DESC,
            id DESC
        LIMIT 1
        """)
    result = _fetchone_dict(cursor)
    conn.close()

    if not result:
        return None

    try:
        result["forecasts"] = json.loads(result.get("forecasts") or "[]")
    except Exception:
        result["forecasts"] = []
    return result


def get_latest_prediction() -> Optional[Dict[str, Any]]:
    """Return the most recently stored prediction run."""
    return get_latest_locked_prediction()


# ---------------------------------------------------------------------------
# FinBERT timing persistence (kv_store)
# ---------------------------------------------------------------------------

def save_finbert_timing(timing: Dict[str, Any]) -> None:
    """
    Persist FinBERT timing metrics from the last scraper run to the database.

    Stored as a JSON blob under the key 'finbert_timing' in kv_store so that
    the data survives server restarts (e.g. Render free-tier hibernation).
    """
    import json

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT OR REPLACE INTO kv_store (key, value, updated_at)
            VALUES ('finbert_timing', ?, CURRENT_TIMESTAMP)
            """,
            (json.dumps(timing),),
        )
        conn.commit()
    except Exception as e:
        logger.warning("Could not persist FinBERT timing to DB: %s", e)
    finally:
        conn.close()


def load_finbert_timing() -> Optional[Dict[str, Any]]:
    """
    Load the last persisted FinBERT timing metrics from the database.

    Returns None if no timing has been stored yet.
    """
    import json

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT value FROM kv_store WHERE key = 'finbert_timing' LIMIT 1"
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None
    except Exception as e:
        logger.warning("Could not load FinBERT timing from DB: %s", e)
        return None
    finally:
        conn.close()


def get_prediction_history(limit: int = 10) -> List[Dict[str, Any]]:
    """Return the last N prediction runs (most recent first)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
    rows = []
    for rec in _fetchall_dicts(cursor):
        rec["forecasts"] = json.loads(rec["forecasts"])
        rows.append(rec)
    conn.close()
    return rows


def _compute_quantile(values: List[float], q: float) -> float:
    """Compute a quantile from a non-empty numeric list using linear interpolation."""
    if not values:
        return 0.0

    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])

    pos = (len(sorted_vals) - 1) * q
    lower_idx = int(math.floor(pos))
    upper_idx = int(math.ceil(pos))

    if lower_idx == upper_idx:
        return float(sorted_vals[lower_idx])

    lower_val = sorted_vals[lower_idx]
    upper_val = sorted_vals[upper_idx]
    weight = pos - lower_idx
    return float(lower_val + (upper_val - lower_val) * weight)


def _collect_relative_errors_by_horizon(
    prediction_runs: pd.DataFrame,
    actual_price_by_date: Dict[date, float],
) -> Dict[int, List[float]]:
    """Build empirical signed relative errors grouped by forecast horizon."""
    errors_by_horizon: Dict[int, List[float]] = defaultdict(list)

    if prediction_runs.empty:
        return errors_by_horizon

    for _, row in prediction_runs.iterrows():
        reference_date = _parse_prediction_reference_date(
            last_price_date_raw=row.get("last_price_date"),
            generated_at_raw=row.get("generated_at"),
        )
        if reference_date is None:
            continue

        forecasts = _parse_forecasts_blob(row.get("forecasts"))
        for forecast in forecasts:
            parsed = _parse_single_forecast_observation(
                forecast=forecast,
                reference_date=reference_date,
                generated_at_raw=row.get("generated_at"),
                cutoff_date=date.today(),
            )
            if parsed is None:
                continue

            target_date, pred_entry = parsed
            actual_price = actual_price_by_date.get(target_date)
            predicted_price = float(pred_entry.get("forecasted_price", 0.0))
            horizon = int(pred_entry.get("horizon", 5))

            if actual_price is None or predicted_price <= 0:
                continue

            rel_error = (actual_price - predicted_price) / predicted_price
            errors_by_horizon[horizon].append(float(rel_error))

    return errors_by_horizon


def _calibration_pool_for_horizon(
    errors_by_horizon: Dict[int, List[float]],
    horizon: int,
    min_samples: int,
) -> List[float]:
    """Get calibration samples for a horizon, widening to neighbors and global pool if needed."""
    direct = list(errors_by_horizon.get(horizon, []))
    if len(direct) >= min_samples:
        return direct

    pooled = list(direct)
    max_h = 5
    for radius in range(1, max_h):
        left = horizon - radius
        right = horizon + radius
        if left >= 1:
            pooled.extend(errors_by_horizon.get(left, []))
        if right <= max_h:
            pooled.extend(errors_by_horizon.get(right, []))
        if len(pooled) >= min_samples:
            return pooled

    global_pool: List[float] = []
    for h in range(1, max_h + 1):
        global_pool.extend(errors_by_horizon.get(h, []))

    if global_pool:
        return global_pool

    # Final deterministic fallback if no calibration data exists at all.
    return [-0.03, -0.015, 0.0, 0.015, 0.03]


def get_latest_prediction_fan_chart(
    min_samples_per_horizon: int = 20,
) -> Dict[str, Any]:
    """
    Return fan chart data for the latest prediction run using empirical error calibration.

    Uncertainty bands are derived from historical signed relative errors
    ((actual - predicted) / predicted), grouped by horizon.
    """
    latest = get_latest_prediction()
    if not latest:
        raise ValueError("No stored prediction runs available")

    conn = get_connection()
    try:
        prediction_runs = _query_to_df(
            conn,
            """
            SELECT generated_at, last_price_date, forecasts
            FROM predictions
            ORDER BY generated_at ASC
            """,
        )
        actual_prices_df = _query_to_df(
            conn,
            """
            SELECT date, price
            FROM prices
            ORDER BY date ASC
            """,
        )
    finally:
        conn.close()

    actual_price_by_date: Dict[date, float] = {}
    if not actual_prices_df.empty:
        actual_prices_df["date"] = pd.to_datetime(actual_prices_df["date"]).dt.date
        actual_price_by_date = {
            row["date"]: float(row["price"]) for _, row in actual_prices_df.iterrows()
        }

    errors_by_horizon = _collect_relative_errors_by_horizon(
        prediction_runs=prediction_runs,
        actual_price_by_date=actual_price_by_date,
    )

    fan_points: List[Dict[str, Any]] = []
    latest_forecasts = latest.get("forecasts", []) if isinstance(latest, dict) else []

    for item in latest_forecasts:
        date_str = str(item.get("date"))
        horizon = int(item.get("horizon", 5))
        point_forecast = float(item.get("forecasted_price", 0.0))
        model_lower = item.get("lower_bound")
        model_upper = item.get("upper_bound")

        samples = _calibration_pool_for_horizon(
            errors_by_horizon=errors_by_horizon,
            horizon=horizon,
            min_samples=max(1, int(min_samples_per_horizon)),
        )

        q10 = _compute_quantile(samples, 0.10)
        q25 = _compute_quantile(samples, 0.25)
        q50 = _compute_quantile(samples, 0.50)
        q75 = _compute_quantile(samples, 0.75)
        q90 = _compute_quantile(samples, 0.90)

        def _apply(q: float) -> float:
            return round(max(0.01, point_forecast * (1.0 + q)), 2)

        fan_points.append(
            {
                "date": date_str,
                "horizon": horizon,
                "point_forecast": round(point_forecast, 2),
                "p10": _apply(q10),
                "p25": _apply(q25),
                "p50": _apply(q50),
                "p75": _apply(q75),
                "p90": _apply(q90),
                "lower_bound": (
                    round(float(model_lower), 2)
                    if model_lower is not None
                    else None
                ),
                "upper_bound": (
                    round(float(model_upper), 2)
                    if model_upper is not None
                    else None
                ),
                "sample_count": len(samples),
            }
        )

    return {
        "generated_at": str(latest.get("generated_at", "")),
        "last_price_date": str(latest.get("last_price_date", "")),
        "last_price": float(latest.get("last_price", 0.0)),
        "calibration_method": (
            "Empirical horizon-wise quantiles from historical signed relative forecast "
            "errors for p10-p90, plus model-driven 95% lower/upper bounds when present "
            "in stored forecasts."
        ),
        "fan": fan_points,
    }


def _empty_comparison_payload(cutoff_date: date) -> Dict[str, Any]:
    """Return a standard empty comparison payload."""
    return {
        "end_date": cutoff_date.strftime("%Y-%m-%d"),
        "rows": [],
        "metrics": {
            "compared_days": 0,
            "mae": None,
            "rmse": None,
            "mape": None,
        },
    }


def _parse_generated_date(generated_at_raw: Any) -> Optional[date]:
    """Parse generated_at timestamp into date, supporting trailing Z."""
    if not generated_at_raw:
        return None
    raw = str(generated_at_raw)
    try:
        return datetime.fromisoformat(raw).date()
    except ValueError:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()


def _extract_forecast_observation(
    forecast: Dict[str, Any],
    reference_date: date,
    generated_at_raw: Any,
    cutoff_date: date,
) -> Optional[tuple]:
    """Extract and validate a single forecast observation (low complexity helper)."""
    target_date_raw = forecast.get("date")
    pred_price_raw = forecast.get("forecasted_price")
    
    if target_date_raw is None or pred_price_raw is None:
        return None
    
    try:
        target_date = datetime.strptime(str(target_date_raw), "%Y-%m-%d").date()
        pred_price = float(pred_price_raw)
    except (ValueError, TypeError):
        return None
    
    if target_date > cutoff_date or reference_date > target_date:
        return None
    
    try:
        horizon = max(1, int(forecast.get("horizon", 5)))
    except (TypeError, ValueError):
        horizon = 5
    
    lower_bound = forecast.get("lower_bound")
    upper_bound = forecast.get("upper_bound")
    
    try:
        lower_bound = float(lower_bound) if lower_bound is not None else None
    except (ValueError, TypeError):
        lower_bound = None
    
    try:
        upper_bound = float(upper_bound) if upper_bound is not None else None
    except (ValueError, TypeError):
        upper_bound = None
    
    return target_date, {
        "forecasted_price": pred_price,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "horizon": horizon,
        "generated_at": str(generated_at_raw),
    }


def _process_prediction_run(
    row: Any, cutoff_date: date
) -> Optional[Dict[date, List[Dict[str, Any]]]]:
    """Process a single prediction run (extracted for lower complexity)."""
    generated_at_raw = row.get("generated_at")
    reference_date = _parse_prediction_reference_date(
        last_price_date_raw=row.get("last_price_date"),
        generated_at_raw=generated_at_raw,
    )
    if reference_date is None:
        return None

    forecasts_blob = row.get("forecasts")
    try:
        forecasts = json.loads(forecasts_blob or "[]")
        if not isinstance(forecasts, list):
            forecasts = []
    except Exception:
        forecasts = []
    
    if not forecasts:
        return None
    
    forecasts = _enrich_forecasts_with_missing_bounds(
        forecasts=forecasts,
        last_price_raw=row.get("last_price"),
    )
    
    results: Dict[date, List[Dict[str, Any]]] = defaultdict(list)
    for forecast in forecasts:
        observation = _extract_forecast_observation(
            forecast=forecast,
            reference_date=reference_date,
            generated_at_raw=generated_at_raw,
            cutoff_date=cutoff_date,
        )
        if observation is not None:
            target_date, entry = observation
            results[target_date].append(entry)
    
    return results if results else None


def _collect_predictions_by_target_date(
    prediction_runs: pd.DataFrame, cutoff_date: date
) -> Dict[date, List[Dict[str, Any]]]:
    """Collect all valid prediction observations keyed by target date."""
    per_date_predictions: Dict[date, List[Dict[str, Any]]] = defaultdict(list)

    if prediction_runs.empty:
        return dict(per_date_predictions)
    
    for _, row in prediction_runs.iterrows():
        run_predictions = _process_prediction_run(row, cutoff_date)
        if run_predictions:
            for target_date, entries in run_predictions.items():
                per_date_predictions[target_date].extend(entries)

    return dict(per_date_predictions)


def _parse_forecasts_blob(forecasts_blob: Any) -> List[Dict[str, Any]]:
    """Parse serialized forecasts JSON into a list."""
    try:
        parsed = json.loads(forecasts_blob or "[]")
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _parse_single_forecast_observation(
    forecast: Dict[str, Any],
    reference_date: date,
    generated_at_raw: Any,
    cutoff_date: date,
) -> Optional[tuple]:
    """Parse and validate one forecast item into (target_date, entry)."""
    target_date_raw = forecast.get("date")
    pred_price_raw = forecast.get("forecasted_price")
    if target_date_raw is None or pred_price_raw is None:
        return None

    try:
        target_date = datetime.strptime(str(target_date_raw), "%Y-%m-%d").date()
        pred_price = float(pred_price_raw)
    except (ValueError, TypeError):
        return None

    if target_date > cutoff_date or reference_date > target_date:
        return None

    try:
        horizon = int(forecast.get("horizon"))
    except (TypeError, ValueError):
        horizon = 5

    lower_bound = _safe_float(forecast.get("lower_bound"))
    upper_bound = _safe_float(forecast.get("upper_bound"))

    return (
        target_date,
        {
            "forecasted_price": pred_price,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "horizon": max(1, horizon),
            "generated_at": str(generated_at_raw),
        },
    )


def _parse_prediction_reference_date(
    last_price_date_raw: Any, generated_at_raw: Any
) -> Optional[date]:
    """
    Parse prediction reference date.

    Prefer last_price_date (the date the forecast starts from); fallback to generated_at.
    """
    if last_price_date_raw:
        try:
            return datetime.strptime(str(last_price_date_raw), "%Y-%m-%d").date()
        except ValueError:
            pass

    return _parse_generated_date(generated_at_raw)


def _aggregate_predictions(preds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate multiple predictions for a date using optimized single-pass computation."""
    n_preds = len(preds)
    
    # Single pass: compute all weighted values, collect prices for median, track latest
    weighted_sum = 0.0
    total_weight = 0.0
    weighted_lower_sum = 0.0
    weighted_lower_weight = 0.0
    weighted_upper_sum = 0.0
    weighted_upper_weight = 0.0
    prices = []
    latest_generated = ""
    latest_price = 0.0
    
    for pred in preds:
        horizon = pred.get("horizon", 5)
        weight = 1.0 / float(horizon)
        price = pred["forecasted_price"]
        
        # Weighted price
        weighted_sum += price * weight
        total_weight += weight
        
        # Bounds (only accumulate if present)
        lower = pred.get("lower_bound")
        if lower is not None:
            weighted_lower_sum += float(lower) * weight
            weighted_lower_weight += weight
        
        upper = pred.get("upper_bound")
        if upper is not None:
            weighted_upper_sum += float(upper) * weight
            weighted_upper_weight += weight
        
        # For median
        prices.append(price)
        
        # Track latest by generated_at
        generated_at = pred.get("generated_at", "")
        if generated_at > latest_generated:
            latest_generated = generated_at
            latest_price = price
    
    weighted_predicted = weighted_sum / total_weight if total_weight > 0 else 0.0
    weighted_lower = weighted_lower_sum / weighted_lower_weight if weighted_lower_weight > 0 else None
    weighted_upper = weighted_upper_sum / weighted_upper_weight if weighted_upper_weight > 0 else None
    
    # Compute median
    prices.sort()
    if n_preds % 2 == 1:
        median_predicted = prices[n_preds // 2]
    else:
        median_predicted = (prices[n_preds // 2 - 1] + prices[n_preds // 2]) / 2.0

    return {
        "weighted_predicted": weighted_predicted,
        "weighted_lower": weighted_lower,
        "weighted_upper": weighted_upper,
        "median_predicted": median_predicted,
        "latest_predicted": latest_price,
        "prediction_count": n_preds,
    }


def _build_comparison_rows(
    actual_df: pd.DataFrame, per_date_predictions: Dict[date, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Build day-level comparison rows for dates that have both actual and predicted values (optimized)."""
    rows: List[Dict[str, Any]] = []
    
    # Pre-aggregate all predictions to avoid repeated aggregation
    aggregated_by_date = {
        target_date: _aggregate_predictions(preds)
        for target_date, preds in per_date_predictions.items()
    }

    for _, rec in actual_df.iterrows():
        target_date = rec["date"]
        aggregated = aggregated_by_date.get(target_date)
        
        if aggregated is None:
            continue

        actual_price = float(rec["price"])
        weighted_predicted = aggregated["weighted_predicted"]
        error = actual_price - weighted_predicted
        abs_error = abs(error)
        pct_error = abs_error / actual_price * 100.0 if actual_price else None

        rows.append(
            {
                "date": target_date.strftime("%Y-%m-%d"),
                "actual_price": round(actual_price, 2),
                "predicted_price": round(weighted_predicted, 2),
                "lower_bound": (
                    round(float(aggregated["weighted_lower"]), 2)
                    if aggregated.get("weighted_lower") is not None
                    else None
                ),
                "upper_bound": (
                    round(float(aggregated["weighted_upper"]), 2)
                    if aggregated.get("weighted_upper") is not None
                    else None
                ),
                "predicted_price_median": round(aggregated["median_predicted"], 2),
                "predicted_price_latest": round(aggregated["latest_predicted"], 2),
                "prediction_count": aggregated["prediction_count"],
                "error": round(error, 2),
                "abs_error": round(abs_error, 2),
                "abs_pct_error": round(pct_error, 2) if pct_error is not None else None,
                # Preserve full-precision values so summary metrics match
                # training-time calculation style (compute first, round last).
                "_error_raw": error,
                "_abs_error_raw": abs_error,
                "_abs_pct_error_raw": pct_error,
            }
        )

    return rows


def _compute_comparison_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute MAE, RMSE, and MAPE from full-precision comparison values."""
    compared_days = len(rows)
    mae = sum(float(r.get("_abs_error_raw", r["abs_error"])) for r in rows) / compared_days
    rmse = math.sqrt(
        sum((float(r.get("_error_raw", r["error"])) ** 2) for r in rows)
        / compared_days
    )

    mape_values = [
        float(r.get("_abs_pct_error_raw", r["abs_pct_error"]))
        for r in rows
        if r.get("_abs_pct_error_raw", r["abs_pct_error"]) is not None
    ]
    mape = sum(mape_values) / len(mape_values) if mape_values else None

    return {
        "compared_days": compared_days,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4) if mape is not None else None,
    }


def _merge_supplemental_prices(
    actual_df: pd.DataFrame,
    supplemental_prices: pd.DataFrame,
    cutoff_date: "date",
    start_bound: Optional["date"],
) -> pd.DataFrame:
    """
    Merge a live-fetched price DataFrame into the DB-sourced actuals.

    DB rows always win when a date exists in both sources.  Only rows within
    [start_bound, cutoff_date] are included.
    """
    supp = supplemental_prices[["date", "price"]].copy()
    supp["date"] = pd.to_datetime(supp["date"]).dt.strftime("%Y-%m-%d")
    supp = supp[supp["date"] <= cutoff_date.strftime("%Y-%m-%d")]
    if start_bound is not None:
        supp = supp[supp["date"] >= start_bound.strftime("%Y-%m-%d")]
    if supp.empty:
        return actual_df
    if actual_df.empty:
        return supp.reset_index(drop=True)
    db_dates = set(actual_df["date"].astype(str))
    new_rows = supp[~supp["date"].isin(db_dates)]
    if new_rows.empty:
        return actual_df
    return (
        pd.concat([actual_df, new_rows], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )


def get_actual_vs_predicted_until(
    end_date: Optional[str] = None,
    start_date: Optional[str] = None,
    supplemental_prices: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Compare stored actual prices with aggregated stored predictions up to end_date.

    Multiple forecasts for the same target date are aggregated using a horizon-weighted
    mean (weight = 1 / horizon), which prioritizes shorter-horizon forecasts.

    Args:
        end_date: Optional YYYY-MM-DD end date. Defaults to today.
        start_date: Optional YYYY-MM-DD start date. Defaults to earliest available.
        supplemental_prices: Optional DataFrame with 'date' and 'price' columns from
            a live Yahoo Finance fetch.  These are merged with DB prices and fill any
            gap caused by Turso replication lag or a missed price-sync run — rows from
            the DB always win when a date exists in both sources.

    Returns:
        Dict with comparison rows and error summary metrics.
    """
    cutoff_date = (
        date.today()
        if end_date is None
        else datetime.strptime(end_date, "%Y-%m-%d").date()
    )
    start_bound = (
        datetime.strptime(start_date, "%Y-%m-%d").date()
        if start_date is not None
        else None
    )
    prediction_window_start = (
        (start_bound - timedelta(days=PREDICTION_COMPARE_LOOKBACK_BUFFER_DAYS))
        if start_bound is not None
        else None
    )

    conn = get_connection()
    try:
        # Fetch actual prices up to the requested cutoff date.
        # Union prices + historical_prices so that dates imported via
        # import_historical_data.py (historical_prices table) are also matched.
        # When a date exists in both tables, the live prices row wins.
        # Use NOT IN subquery (reliable in SQLite) rather than GROUP BY bare-column
        # priority trick which is undefined behaviour for non-aggregate columns.
        if start_bound is None:
            cutoff_str = cutoff_date.strftime("%Y-%m-%d")
            actual_df = _query_to_df(
                conn,
                """
                SELECT date, price FROM prices
                WHERE date <= ?
                UNION
                SELECT date, price FROM historical_prices
                WHERE date <= ?
                  AND date NOT IN (SELECT date FROM prices WHERE date <= ?)
                ORDER BY date ASC
                """,
                params=(cutoff_str, cutoff_str, cutoff_str),
            )
        else:
            start_str = start_bound.strftime("%Y-%m-%d")
            cutoff_str = cutoff_date.strftime("%Y-%m-%d")
            actual_df = _query_to_df(
                conn,
                """
                SELECT date, price FROM prices
                WHERE date >= ? AND date <= ?
                UNION
                SELECT date, price FROM historical_prices
                WHERE date >= ? AND date <= ?
                  AND date NOT IN (SELECT date FROM prices WHERE date >= ? AND date <= ?)
                ORDER BY date ASC
                """,
                params=(start_str, cutoff_str, start_str, cutoff_str, start_str, cutoff_str),
            )

        if prediction_window_start is None:
            prediction_runs = _query_to_df(
                conn,
                """
                SELECT generated_at, last_price_date, last_price, forecasts
                FROM predictions
                WHERE COALESCE(last_price_date, substr(generated_at, 1, 10)) <= ?
                ORDER BY generated_at ASC
                """,
                params=(cutoff_date.strftime("%Y-%m-%d"),),
            )
        else:
            prediction_runs = _query_to_df(
                conn,
                """
                                SELECT generated_at, last_price_date, last_price, forecasts
                FROM predictions
                WHERE COALESCE(last_price_date, substr(generated_at, 1, 10)) >= ?
                  AND COALESCE(last_price_date, substr(generated_at, 1, 10)) <= ?
                ORDER BY generated_at ASC
                """,
                params=(
                    prediction_window_start.strftime("%Y-%m-%d"),
                    cutoff_date.strftime("%Y-%m-%d"),
                ),
            )
    finally:
        conn.close()

    # Merge supplemental prices (e.g. live Yahoo fetch) so that dates not yet
    # committed to Turso still appear in the comparison.  DB rows always win.
    if supplemental_prices is not None and not supplemental_prices.empty:
        actual_df = _merge_supplemental_prices(
            actual_df, supplemental_prices, cutoff_date, start_bound
        )

    if actual_df.empty:
        return _empty_comparison_payload(cutoff_date)

    actual_df["date"] = pd.to_datetime(actual_df["date"]).dt.date

    per_date_predictions = _collect_predictions_by_target_date(
        prediction_runs, cutoff_date
    )
    rows = _build_comparison_rows(actual_df, per_date_predictions)

    if not rows:
        return _empty_comparison_payload(cutoff_date)

    metrics = _compute_comparison_metrics(rows)
    public_rows = [
        {k: v for k, v in row.items() if not k.startswith("_")} for row in rows
    ]

    return {
        "end_date": cutoff_date.strftime("%Y-%m-%d"),
        "rows": public_rows,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Explainability functions
# ---------------------------------------------------------------------------


def add_explanation(
    explanation_date: str,
    aggregated: Dict[str, Any],
    explanation_text: str,
    generated_at: str,
    computation_time_seconds: float,
    xai_payload: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Store a daily explainability result.

    Args:
        explanation_date: Date string (YYYY-MM-DD) when the explanation was generated.
        aggregated: Unified explanation payload with predictions, confidence, features, and model weights.
        explanation_text: Plain English narrative (3-sentence explanation).
        generated_at: ISO timestamp when computation occurred.
        computation_time_seconds: How long the computation took.
        xai_payload: Optional full dashboard-ready JSON payload.

    Returns:
        Row id of the inserted record.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO explanations (
                explanation_date, prediction, confidence_interval_lower,
                confidence_interval_upper, arima_contribution, gru_mid_contribution,
                gru_sent_contribution, xgb_hf_contribution, agreement_score,
                confidence_level, top_shap_features, sentiment_headlines,
                explanation_text, model_weights, generated_at, computation_time_seconds,
                xai_payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                explanation_date,
                aggregated["prediction"],
                aggregated["confidence_interval_lower"],
                aggregated["confidence_interval_upper"],
                aggregated["arima_contribution"],
                aggregated["gru_mid_contribution"],
                aggregated["gru_sent_contribution"],
                aggregated["xgb_hf_contribution"],
                aggregated["agreement_score"],
                aggregated["confidence_level"],
                json.dumps(aggregated["top_features"]),
                json.dumps(aggregated["sentiment_headlines"]),
                explanation_text,
                json.dumps(aggregated["model_weights"]),
                generated_at,
                computation_time_seconds,
                json.dumps(xai_payload) if xai_payload is not None else None,
            ),
        )
        row_id = cursor.lastrowid
        conn.commit()
        logger.info(f"Saved explanation for date={explanation_date} id={row_id}")
        return row_id
    except Exception as e:
        logger.error(f"Error saving explanation: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def get_explanation_for_date(explanation_date: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a stored explanation by date.

    Args:
        explanation_date: Date string (YYYY-MM-DD).

    Returns:
        Explanation dict with parsed JSON fields, or None if not found.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT * FROM explanations WHERE explanation_date = ? LIMIT 1",
            (explanation_date,),
        )
        result = _fetchone_dict(cursor)
        if result:
            # Parse JSON fields — defensively handle malformed/empty values
            for key, default in (
                ("top_shap_features", []),
                ("sentiment_headlines", []),
                ("model_weights", {}),
            ):
                raw = result.get(key)
                try:
                    result[key] = json.loads(raw) if raw else default
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        "Malformed JSON in explanations.%s for date=%s, using default",
                        key,
                        explanation_date,
                    )
                    result[key] = default

            raw_payload = result.get("xai_payload")
            try:
                result["xai_payload"] = (
                    json.loads(raw_payload)
                    if raw_payload and str(raw_payload).strip()
                    else None
                )
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Malformed xai_payload JSON for date=%s, treating as None",
                    explanation_date,
                )
                result["xai_payload"] = None
        return result
    finally:
        conn.close()


def explanation_exists_for_date(explanation_date: str) -> bool:
    """Check if an explanation already exists for a given date."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT 1 FROM explanations WHERE explanation_date = ? LIMIT 1",
            (explanation_date,),
        )
        result = cursor.fetchone()
        return result is not None
    finally:
        conn.close()


def update_explanation_xai_payload(
    explanation_date: str, xai_payload: Dict[str, Any]
) -> bool:
    """
    Update the xai_payload column for an existing explanation row.

    Returns:
        True if a row was updated, False if no row matched.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE explanations SET xai_payload = ? WHERE explanation_date = ?",
            (json.dumps(xai_payload), explanation_date),
        )
        conn.commit()
        updated = cursor.rowcount if hasattr(cursor, "rowcount") else 1
        logger.info(f"Updated xai_payload for date={explanation_date} (rows={updated})")
        return True
    except Exception as e:
        logger.error(f"Error updating xai_payload: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()
