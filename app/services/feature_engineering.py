"""
Feature engineering service - replicates training feature computation exactly.

CRITICAL: All feature definitions MUST match the training code exactly.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
import logging

from app.config import LAGS, VOL_WINDOWS, EMA_WINDOWS, SENT_COLS

logger = logging.getLogger(__name__)


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log prices and log returns from price data.

    Args:
        prices: DataFrame with 'date' and 'price' columns

    Returns:
        DataFrame with additional 'log_price' and 'log_return' columns
    """
    df = prices.copy()
    df["log_price"] = np.log(df["price"])
    df["log_return"] = df["log_price"].diff()
    return df


def compute_lagged_returns(df: pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
    """
    Compute lagged returns.

    Args:
        df: DataFrame with 'log_return' column
        lags: List of lag values (default: [1, 2, 3, 5, 7, 10, 14])

    Returns:
        DataFrame with added lag columns
    """
    if lags is None:
        lags = LAGS

    result = df.copy()
    for lag in lags:
        result[f"ret_lag_{lag}"] = result["log_return"].shift(lag)

    return result


def compute_volatility(df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
    """
    Compute rolling volatility (standard deviation of returns).

    Args:
        df: DataFrame with 'log_return' column
        windows: List of window sizes (default: [5, 10, 14])

    Returns:
        DataFrame with added volatility columns
    """
    if windows is None:
        windows = VOL_WINDOWS

    result = df.copy()
    for w in windows:
        result[f"vol_{w}"] = result["log_return"].rolling(w).std()

    return result


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).

    Args:
        series: Series of returns
        window: RSI window (default: 14)

    Returns:
        Series of RSI values
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / (avg_loss + 1e-8)  # Avoid division by zero
    return 100 - (100 / (1 + rs))


def compute_momentum(df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
    """
    Compute momentum indicators.

    Args:
        df: DataFrame with 'log_return' column
        windows: List of momentum windows (default: [7, 14])

    Returns:
        DataFrame with added momentum columns
    """
    if windows is None:
        windows = [7, 14]

    result = df.copy()
    for w in windows:
        result[f"momentum_{w}"] = result["log_return"] - result["log_return"].shift(w)

    return result


def compute_sentiment_emas(
    df: pd.DataFrame, sent_cols: List[str] = None, windows: List[int] = None
) -> pd.DataFrame:
    """
    Compute EMA-smoothed sentiment features.

    Args:
        df: DataFrame with sentiment columns
        sent_cols: List of sentiment column names
        windows: List of EMA windows (default: [3, 7, 14])

    Returns:
        DataFrame with added EMA columns
    """
    if sent_cols is None:
        sent_cols = [
            "daily_sentiment",
            "news_volume",
            "log_news_volume",
            "decayed_news_volume",
        ]
    if windows is None:
        windows = EMA_WINDOWS

    result = df.copy()
    for col in sent_cols:
        if col in result.columns:
            for w in windows:
                result[f"{col}_ema_{w}"] = result[col].ewm(span=w, adjust=False).mean()

    return result


def _prepare_sentiment_for_merge(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare sentiment data for merging with price features.

    Args:
        sentiment_df: Raw sentiment DataFrame

    Returns:
        Prepared sentiment DataFrame with shifted dates
    """
    sent = sentiment_df.copy()
    sent["date"] = pd.to_datetime(sent["date"])

    # Shift sentiment by 1 day (today's prediction uses yesterday's sentiment)
    sent["date"] = sent["date"] + pd.Timedelta(days=1)

    # Rename daily_sentiment to daily_sentiment_decay
    if "daily_sentiment" in sent.columns:
        sent["daily_sentiment_decay"] = sent["daily_sentiment"]

    return sent


def _add_sentiment_emas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add EMA features for sentiment columns.

    Args:
        df: DataFrame with sentiment columns

    Returns:
        DataFrame with added EMA columns
    """
    ema_cols = [
        "daily_sentiment_decay",
        "news_volume",
        "log_news_volume",
        "decayed_news_volume",
    ]

    for col in ema_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            for w in EMA_WINDOWS:
                df[f"{col}_ema_{w}"] = df[col].ewm(span=w, adjust=False).mean()

    return df


def _fill_missing_sentiment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill any missing sentiment columns with zeros.

    Args:
        df: DataFrame possibly missing some sentiment columns

    Returns:
        DataFrame with all sentiment columns present
    """
    for col in SENT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def _add_zero_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add zero-valued sentiment features when no sentiment data is available.

    Args:
        df: DataFrame with price features

    Returns:
        DataFrame with zero-valued sentiment columns
    """
    # Add base sentiment columns
    df["daily_sentiment"] = 0.0
    df["daily_sentiment_decay"] = 0.0

    for col in SENT_COLS:
        df[col] = 0.0

    # Add zero sentiment EMAs
    ema_cols = [
        "daily_sentiment_decay",
        "news_volume",
        "log_news_volume",
        "decayed_news_volume",
    ]
    for col in ema_cols:
        for w in EMA_WINDOWS:
            df[f"{col}_ema_{w}"] = 0.0

    return df


def _merge_and_process_sentiment(
    df: pd.DataFrame, sentiment_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge sentiment data with price features and compute sentiment EMAs.

    Args:
        df: DataFrame with price features
        sentiment_df: DataFrame with sentiment data

    Returns:
        DataFrame with merged sentiment features
    """
    logger.info(f"Merging {len(sentiment_df)} sentiment records")

    df["date"] = pd.to_datetime(df["date"])
    sent = _prepare_sentiment_for_merge(sentiment_df)

    # Merge
    df = df.merge(sent, on="date", how="left")

    # Fill NaN sentiment with 0
    if "daily_sentiment_decay" in df.columns:
        df["daily_sentiment_decay"] = df["daily_sentiment_decay"].fillna(0)

    # Compute EMAs for sentiment columns
    df = _add_sentiment_emas(df)

    # Fill remaining sentiment columns with 0
    df = _fill_missing_sentiment_columns(df)

    return df


def engineer_all_features(
    prices: pd.DataFrame, sentiment_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Compute all features from price data, optionally merging with sentiment.

    Args:
        prices: DataFrame with 'date' and 'price' columns
        sentiment_df: Optional DataFrame with sentiment data (already lagged)

    Returns:
        DataFrame with all engineered features
    """
    logger.info("Starting feature engineering...")

    # Compute price-based features
    df = compute_log_returns(prices)
    df = compute_lagged_returns(df)
    df = compute_volatility(df)
    df["rsi_14"] = compute_rsi(df["log_return"], 14)
    df = compute_momentum(df)

    # Handle sentiment data
    if sentiment_df is not None and not sentiment_df.empty:
        df = _merge_and_process_sentiment(df, sentiment_df)
    else:
        logger.info("No sentiment data provided, using zeros")
        df = _add_zero_sentiment_features(df)

    logger.info(f"Feature engineering complete. Shape: {df.shape}")

    return df


def get_mid_freq_features() -> List[str]:
    """
    Get list of mid-frequency feature column names.
    Must match training exactly.
    """
    return [
        "log_return",
        *[f"ret_lag_{l}" for l in LAGS],
        *[f"vol_{w}" for w in VOL_WINDOWS],
        "rsi_14",
        "momentum_7",
        "momentum_14",
    ]


def get_price_features() -> List[str]:
    """
    Get list of price feature column names for Sentiment-GRU.
    Must match training exactly - 13 features (no log_return).
    """
    # These are the exact price_features from training config
    return [
        *[f"ret_lag_{l}" for l in LAGS],  # 7 features
        *[f"vol_{w}" for w in VOL_WINDOWS],  # 3 features
        "rsi_14",
        "momentum_7",
        "momentum_14",
    ]  # Total: 13 features


def get_sentiment_features() -> List[str]:
    """
    Get list of sentiment feature column names for Sentiment-GRU.
    Must match training exactly.
    """
    # Base sentiment columns
    base_cols = [
        "daily_sentiment_decay",
        "news_volume",
        "log_news_volume",
        "decayed_news_volume",
        "high_news_regime",
    ]

    # EMA columns
    ema_cols = []
    for col in [
        "daily_sentiment_decay",
        "news_volume",
        "log_news_volume",
        "decayed_news_volume",
    ]:
        for w in EMA_WINDOWS:
            ema_cols.append(f"{col}_ema_{w}")

    return base_cols + ema_cols


def get_hf_features() -> List[str]:
    """
    Get list of high-frequency feature column names for XGBoost.
    Must match training exactly.
    """
    return [
        "log_return",
        "ret_lag_1",
        "ret_lag_2",
        "ret_lag_3",
        "vol_5",
        "daily_sentiment_decay_ema_3",
        "news_volume_ema_3",
        "high_news_regime",
    ]


def prepare_mid_features(df: pd.DataFrame, lookback: int = 21) -> np.ndarray:
    """
    Prepare mid-frequency features for GRU input.

    Args:
        df: DataFrame with all features
        lookback: Number of timesteps for sequence

    Returns:
        numpy array of shape (1, lookback, n_features)
    """
    from app.models.model_loader import model_artifacts

    feature_cols = model_artifacts.mid_features or get_mid_freq_features()

    # Get last 'lookback' rows
    data = df[feature_cols].tail(lookback)

    # Drop any rows with NaN (should be handled before this point)
    if data.isna().any().any():
        logger.warning("NaN values found in mid features, filling with 0")
        data = data.fillna(0)

    return data.values.reshape(1, lookback, -1)


def prepare_sentiment_features(
    df: pd.DataFrame, lookback: int = 21
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare price and sentiment features for Sentiment-GRU input.

    Args:
        df: DataFrame with all features
        lookback: Number of timesteps for sequence

    Returns:
        Tuple of (price_features, sentiment_features) arrays
        Each of shape (1, lookback, n_features)
    """
    from app.models.model_loader import model_artifacts

    price_cols = model_artifacts.price_features or get_price_features()
    sent_cols = model_artifacts.sentiment_features or get_sentiment_features()

    # Get last 'lookback' rows
    price_data = df[price_cols].tail(lookback).fillna(0).values
    sent_data = df[sent_cols].tail(lookback).fillna(0).values

    return (price_data.reshape(1, lookback, -1), sent_data.reshape(1, lookback, -1))


def prepare_hf_features(df: pd.DataFrame) -> np.ndarray:
    """
    Prepare high-frequency features for XGBoost input.
    Uses only the most recent day.

    Args:
        df: DataFrame with all features

    Returns:
        numpy array of shape (1, n_features)
    """
    from app.models.model_loader import model_artifacts

    feature_cols = model_artifacts.hf_features or get_hf_features()

    # Get only the last row
    data = df[feature_cols].tail(1).fillna(0).values

    return data
