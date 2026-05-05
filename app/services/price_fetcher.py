"""
Price fetcher service - fetches Brent oil prices from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta, timezone, time, date
from typing import Optional, Dict, Any, Literal
import logging
from threading import RLock
from time import monotonic
from zoneinfo import ZoneInfo

from app.config import BRENT_TICKER

logger = logging.getLogger(__name__)

# Python 3.10 compatibility: datetime.UTC is unavailable before 3.11.
UTC = timezone.utc

_LATEST_PRICES_CACHE_TTL_SECONDS = 120.0
_LIVE_SNAPSHOT_CACHE_TTL_SECONDS = 20.0
MARKET_OPEN_TIME_UTC = "02:00 UTC"
MARKET_CLOSE_TIME_UTC = "22:00 UTC"
_prices_cache_lock = RLock()
_prices_cache: Dict[tuple[int, str], tuple[float, pd.DataFrame]] = {}
_snapshot_cache_lock = RLock()
_snapshot_cache: Optional[tuple[float, Dict[str, Any]]] = None


def get_regular_session_window() -> Optional[Dict[str, Any]]:
    """
    Read Yahoo Finance metadata for the current regular trading period.

    Returns:
        Dict with exchange timezone and regular session start/end datetimes,
        or None if Yahoo metadata is unavailable.
    """
    try:
        ticker = yf.Ticker(BRENT_TICKER)
        meta = ticker.history_metadata or {}

        exchange_tz_name = meta.get("exchangeTimezoneName")
        regular = (meta.get("currentTradingPeriod") or {}).get("regular") or {}
        start_epoch = regular.get("start")
        end_epoch = regular.get("end")

        if not exchange_tz_name or not start_epoch or not end_epoch:
            return None

        exchange_tz = ZoneInfo(exchange_tz_name)
        return {
            "exchange_timezone": exchange_tz_name,
            "regular_start": datetime.fromtimestamp(int(start_epoch), exchange_tz),
            "regular_end": datetime.fromtimestamp(int(end_epoch), exchange_tz),
        }
    except Exception as exc:
        logger.warning("Failed to read Yahoo regular trading window: %s", exc)
        return None


def get_canonical_prediction_date(
    *,
    target_timezone: str,
    close_lock_buffer_minutes: int,
    now_target: Optional[datetime] = None,
) -> str:
    """
    Resolve a stable daily prediction key in the target timezone.

    The key advances only after Yahoo's regular session end + buffer has passed
    when converted into the target timezone.
    """
    target_tz = ZoneInfo(target_timezone)
    target_now = (
        now_target.astimezone(target_tz)
        if now_target is not None
        else datetime.now(target_tz)
    )

    session = get_regular_session_window()
    if not session:
        return target_now.strftime("%Y-%m-%d")

    regular_end = session["regular_end"]
    # Yahoo reports 23:59 ET for commodity futures (extended/overnight session).
    # Daily OHLCV bars settle at the regular close (~16:00 ET), so clamp to 16:00
    # when Yahoo returns an unreasonably late hour (>=20:00 in exchange timezone).
    _exch_tz = ZoneInfo(str(session.get("exchange_timezone", "America/New_York")))
    _re_exch = regular_end.astimezone(_exch_tz)
    if _re_exch.hour >= 20:
        regular_end = _re_exch.replace(hour=16, minute=0, second=0)
    stable_after_exchange = regular_end + timedelta(
        minutes=max(0, int(close_lock_buffer_minutes))
    )
    stable_after_target = stable_after_exchange.astimezone(target_tz)
    session_date_target: date = regular_end.astimezone(target_tz).date()

    if target_now < stable_after_target:
        prediction_date = session_date_target - timedelta(days=1)
    else:
        prediction_date = session_date_target

    return prediction_date.strftime("%Y-%m-%d")


def _get_chart_result(chart_payload: Dict[str, Any]) -> Dict[str, Any]:
    chart = chart_payload.get("chart")
    if not isinstance(chart, dict):
        raise ValueError("Invalid payload: missing 'chart' object")

    results = chart.get("result") or []
    if not results:
        error_obj = chart.get("error")
        raise ValueError(f"Invalid payload: missing chart result. error={error_obj}")

    return results[0] or {}


def _extract_quote_series(result: Dict[str, Any]) -> Dict[str, list[Any]]:
    quotes = (((result.get("indicators") or {}).get("quote") or [{}])[0]) or {}
    return {
        "timestamp": result.get("timestamp") or [],
        "open": quotes.get("open") or [],
        "high": quotes.get("high") or [],
        "low": quotes.get("low") or [],
        "close": quotes.get("close") or [],
        "volume": quotes.get("volume") or [],
    }


def _validate_quote_series(series: Dict[str, list[Any]]) -> None:
    if not series["timestamp"]:
        raise ValueError("Invalid payload: 'timestamp' is empty")

    lengths = {key: len(values) for key, values in series.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Mismatched array lengths in chart payload: {lengths}")


def _apply_missing_strategy(
    df: pd.DataFrame,
    numeric_cols: list[str],
    missing_strategy: Literal["drop", "ffill", "none"],
) -> pd.DataFrame:
    if missing_strategy == "drop":
        return df.dropna(subset=numeric_cols)
    if missing_strategy == "ffill":
        df[numeric_cols] = df[numeric_cols].ffill()
        return df.dropna(subset=numeric_cols)
    if missing_strategy == "none":
        return df

    raise ValueError("missing_strategy must be one of: 'drop', 'ffill', 'none'")


def parse_yahoo_chart_intraday(
    chart_payload: Dict[str, Any],
    local_tz: Optional[str] = "Asia/Colombo",
    missing_strategy: Literal["drop", "ffill", "none"] = "drop",
) -> pd.DataFrame:
    """
    Parse Yahoo Finance chart endpoint JSON into a structured intraday DataFrame.

    The Yahoo chart endpoint returns UNIX epoch timestamps in seconds. UNIX epoch is
    defined relative to UTC, so we first parse into timezone-aware UTC datetimes for
    consistent modeling and alignment with external UTC-aligned sources.

    We keep UTC as the modeling index and optionally add a local-time view column for
    dashboards/UI. Mixing local time directly into model pipelines can introduce DST and
    timezone-shift bugs when joining multi-source financial data.

    Args:
        chart_payload: Parsed JSON payload from Yahoo chart endpoint.
        local_tz: Optional local timezone for visualization column.
        missing_strategy: Missing value handling for OHLCV columns:
            - "drop": drop rows where any OHLCV value is missing.
            - "ffill": forward-fill OHLCV values, then drop remaining NaNs.
            - "none": keep missing values unchanged.

    Returns:
        DataFrame indexed by UTC datetime (`timestamp_utc`) with columns:
            timestamp, open, high, low, close, volume, [timestamp_local]

    Raises:
        ValueError: If payload structure is invalid or required data is missing.
    """
    result = _get_chart_result(chart_payload)
    series = _extract_quote_series(result)
    _validate_quote_series(series)

    df = pd.DataFrame(series)

    # Parse UNIX epoch seconds as timezone-aware UTC timestamps.
    df["timestamp_utc"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _apply_missing_strategy(df, numeric_cols, missing_strategy)

    if local_tz:
        # Local timezone should be presentation-only, not the canonical modeling timeline.
        df["timestamp_local"] = df["timestamp_utc"].dt.tz_convert(local_tz)

    df = df.set_index("timestamp_utc").sort_index()
    return df


def _cache_enabled() -> bool:
    """Disable in-memory caching during pytest to preserve deterministic mocks."""
    return "PYTEST_CURRENT_TEST" not in os.environ


def _get_cached_snapshot() -> Optional[Dict[str, Any]]:
    if not _cache_enabled():
        return None

    now_ts = monotonic()
    with _snapshot_cache_lock:
        if (
            _snapshot_cache
            and (now_ts - _snapshot_cache[0]) < _LIVE_SNAPSHOT_CACHE_TTL_SECONDS
        ):
            return _snapshot_cache[1].copy()
    return None


def _set_cached_snapshot(snapshot: Dict[str, Any]) -> None:
    global _snapshot_cache

    if not _cache_enabled():
        return

    with _snapshot_cache_lock:
        _snapshot_cache = (monotonic(), snapshot.copy())


def _build_intraday_snapshot(intraday: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if intraday is None or intraday.empty:
        return None

    last_row = intraday.iloc[-1]
    price = float(last_row.get("Close", 0.0))
    if price <= 0:
        return None

    last_ts = pd.to_datetime(intraday.index[-1]).tz_localize(None)
    return {
        "price": price,
        "as_of": last_ts.isoformat(),
        "as_of_date": last_ts.strftime("%Y-%m-%d"),
        "source": "yahoo_finance_intraday",
    }


def _build_fast_info_snapshot(
    fast_info: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not fast_info:
        return None

    fallback_price = fast_info.get("lastPrice") or fast_info.get(
        "regularMarketPreviousClose"
    )
    if not fallback_price:
        return None

    now_ts = datetime.now(UTC)
    return {
        "price": float(fallback_price),
        "as_of": now_ts.isoformat(),
        "as_of_date": now_ts.strftime("%Y-%m-%d"),
        "source": "yahoo_finance_fast_info",
    }


def _fetch_prices_from_db_fallback(
    lookback_days: int, end_date: Optional[datetime]
) -> pd.DataFrame:
    """
    Fallback: retrieve stored Brent oil prices from the database when Yahoo Finance
    is unavailable.  Returns a DataFrame with columns: date, price.

    Raises:
        ValueError: If the database also has no usable price data.
    """
    try:
        from app.database import get_prices_for_date_range

        if end_date is None:
            end_date = datetime.now(UTC)

        start_date = end_date - timedelta(days=lookback_days)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        prices = get_prices_for_date_range(start_str, end_str)

        if prices.empty:
            raise ValueError("No price data found in database fallback.")

        prices = prices[["date", "price"]].copy()
        prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
        prices = prices.sort_values("date").reset_index(drop=True)

        min_expected = max(1, lookback_days // 2)
        if len(prices) < min_expected:
            raise ValueError(
                f"Insufficient data in database fallback: got {len(prices)} days, "
                f"need at least {min_expected} for a {lookback_days} day window"
            )

        logger.warning(
            "Yahoo Finance unavailable — using database fallback prices "
            "(%d rows, latest date: %s)",
            len(prices),
            prices["date"].iloc[-1].strftime("%Y-%m-%d"),
        )
        return prices

    except ValueError:
        raise
    except Exception as db_exc:
        raise ValueError(
            f"Failed to fetch Brent oil prices: Yahoo Finance unavailable and "
            f"database fallback also failed: {db_exc}"
        ) from db_exc


def fetch_latest_prices(
    lookback_days: int = 90, end_date: datetime = None
) -> pd.DataFrame:
    """
    Fetch the latest Brent oil prices from Yahoo Finance.

    Args:
        lookback_days: Number of calendar days to fetch (extra buffer for weekends/holidays)
                      Default is 90 to ensure we get at least 30 valid trading days
                      after feature engineering drops initial NaN rows.
        end_date: Optional end date for fetching prices (exclusive in yfinance, so we add 1 day if needed).
                  If None, uses current date.

    Returns:
        DataFrame with columns: date, price (at least 30 trading days)

    Raises:
        ValueError: If unable to fetch sufficient price data
    """
    logger.info(f"Fetching Brent oil prices (ticker: {BRENT_TICKER})")

    cache_key = (lookback_days, end_date.date().isoformat() if end_date else "today")
    if _cache_enabled():
        now_ts = monotonic()
        with _prices_cache_lock:
            cached = _prices_cache.get(cache_key)
            if cached and (now_ts - cached[0]) < _LATEST_PRICES_CACHE_TTL_SECONDS:
                return cached[1].copy()

    if end_date is None:
        # Use UTC to avoid local DST gaps (e.g., 02:xx on DST start)
        end_date = datetime.now()

    start_date = end_date - timedelta(days=lookback_days)

    try:
        ticker = yf.Ticker(BRENT_TICKER)
        # yfinance history(end=...) is exclusive, so always pass end+1 day to
        # include today's (or the requested end date's) closing price.
        # Passing end=tomorrow also ensures yfinance never serves this request
        # from its internal LRU cache (cache_get is only triggered when
        # end_dt + 30min <= now, i.e. end is already in the past).
        start_arg = start_date.date()
        end_arg = end_date.date() + timedelta(days=1)

        df = ticker.history(start=start_arg, end=end_arg)

        if df.empty:
            raise ValueError(f"No data returned for ticker {BRENT_TICKER}")

        # Use 'Close' price
        prices = df[["Close"]].reset_index()
        prices.columns = ["date", "price"]

        # Ensure date is just date (not datetime)
        prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)

        # Sort by date
        prices = prices.sort_values("date").reset_index(drop=True)

        logger.info(f"Fetched {len(prices)} days of price data")
        logger.info(f"Date range: {prices['date'].min()} to {prices['date'].max()}")
        logger.info(f"Latest price: ${prices['price'].iloc[-1]:.2f}")

        # Validate based on requested window (allowing for weekends/holidays)
        # We expect roughly 5 trading days for every 7 calendar days
        min_expected = max(1, lookback_days // 2)
        if len(prices) < min_expected:
            raise ValueError(
                f"Insufficient data: got {len(prices)} days, need at least {min_expected} for a {lookback_days} day window"
            )

        if _cache_enabled():
            with _prices_cache_lock:
                _prices_cache[cache_key] = (monotonic(), prices.copy())

        return prices

    except Exception as e:
        logger.error(f"Error fetching prices from Yahoo Finance: {e}. Attempting database fallback.")
        return _fetch_prices_from_db_fallback(lookback_days, end_date)


def get_last_n_trading_days(prices: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """
    Get the last N trading days from a price DataFrame.

    Args:
        prices: DataFrame with 'date' and 'price' columns
        n: Number of trading days to return

    Returns:
        DataFrame with last N trading days
    """
    return prices.tail(n).reset_index(drop=True)


def get_market_status(now_utc: Optional[datetime] = None) -> dict:
    """
    Determine market status using Yahoo Finance API real-time data.

    Instead of hardcoded hours, fetches actual market state from Yahoo Finance
    which reflects the true market conditions for Brent Oil futures (BZ=F).

    Returns:
        dict with keys:
            - is_open (bool): True if market is currently open (via Yahoo Finance marketState).
            - market_state (str): "OPEN" or "CLOSED" (from Yahoo Finance).
            - message (str): Human-readable status string.
            - market_open_time (str): Typical market open time for this exchange.
            - market_close_time (str): Typical market close time for this exchange.
            - timezone_info (str): Exchange timezone reference.
    """
    try:
        # Fetch real-time market status from Yahoo Finance
        ticker = yf.Ticker(BRENT_TICKER)
        info = ticker.info

        market_state_api = info.get("marketState", "UNKNOWN")
        exchange_tz = info.get("exchangeTimezoneName", "UTC")

        # marketState can be: REGULAR, PRE, POST, CLOSED, etc.
        # Treat REGULAR as market open, everything else as closed
        is_open = market_state_api == "REGULAR"

        # Brent Oil (ICE) typical hours: 02:00-22:00 UTC (almost 24/5 trading)
        # with ~2 hour break early morning UTC
        return {
            "is_open": is_open,
            "market_state": market_state_api,
            "message": f"Market {'open' if is_open else 'closed'} ({market_state_api})",
            "market_open_time": MARKET_OPEN_TIME_UTC,
            "market_close_time": MARKET_CLOSE_TIME_UTC,
            "timezone_info": f"Exchange timezone: {exchange_tz}",
        }
    except Exception as e:
        logger.warning(
            f"Failed to fetch market status from Yahoo Finance: {e}. Using fallback logic."
        )

        # Fallback: Use deterministic trading hours if API fails
        if now_utc is None:
            now_utc = datetime.now(UTC)
        elif now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=UTC)
        else:
            now_utc = now_utc.astimezone(UTC)

        is_trading_day = now_utc.weekday() < 5
        market_open_utc = time(2, 0)
        market_close_utc = time(22, 0)
        is_within_market_hours = market_open_utc <= now_utc.time() < market_close_utc

        if is_trading_day and is_within_market_hours:
            return {
                "is_open": True,
                "market_state": "TRADING_DAY",
                "message": "Market open (trading hours - fallback)",
                "market_open_time": MARKET_OPEN_TIME_UTC,
                "market_close_time": MARKET_CLOSE_TIME_UTC,
                "timezone_info": "UTC (Brent Oil - ICE) - FALLBACK MODE",
            }

        return {
            "is_open": False,
            "market_state": "CLOSED",
            "message": "Market closed (fallback logic)",
            "market_open_time": MARKET_OPEN_TIME_UTC,
            "market_close_time": MARKET_CLOSE_TIME_UTC,
            "timezone_info": "UTC (Brent Oil - ICE) - FALLBACK MODE",
        }


def validate_price_data(prices: pd.DataFrame, min_days: int = 30) -> bool:
    """
    Validate that price data meets requirements.

    Args:
        prices: DataFrame with 'date' and 'price' columns
        min_days: Minimum number of trading days required

    Returns:
        True if valid, False otherwise
    """
    if prices is None or prices.empty:
        return False

    if len(prices) < min_days:
        return False

    if "date" not in prices.columns or "price" not in prices.columns:
        return False

    if prices["price"].isna().any():
        return False

    return True


def fetch_live_price_snapshot() -> Optional[Dict[str, Any]]:
    """
    Fetch the latest intraday Brent quote for display/forecast anchoring.

    This function does not persist intraday prices. Database persistence remains
    based on completed daily closes from fetch_latest_prices().
    """
    cached_snapshot = _get_cached_snapshot()
    if cached_snapshot is not None:
        return cached_snapshot

    try:
        ticker = yf.Ticker(BRENT_TICKER)
        intraday = ticker.history(period="1d", interval="1m", prepost=True)

        snapshot = _build_intraday_snapshot(intraday)
        if snapshot is None:
            snapshot = _build_fast_info_snapshot(getattr(ticker, "fast_info", None))

        if snapshot is not None:
            _set_cached_snapshot(snapshot)
            return snapshot
    except Exception as exc:
        logger.warning("Failed to fetch live price snapshot: %s", exc)

    return None
