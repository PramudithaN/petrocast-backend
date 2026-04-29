"""
Application configuration and constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_ARTIFACTS_DIR = BASE_DIR / "model_artifacts"

# Sentiment model path (ProsusAI/finbert downloaded from Colab)
SENTIMENT_MODEL_DIR = (
    MODEL_ARTIFACTS_DIR / "sentiment_model" / "finbert_sentiment_model"
)

# Cross-day decay parameter (matching Colab training)
# Formula: decayed[t] = sentiment[t] + exp(-LAMBDA) * decayed[t-1]
SENTIMENT_DECAY_LAMBDA = 0.3

# Model configuration defaults (runtime artifacts override these values)
LOOKBACK = 30
HORIZON = 14

# Feature definitions (must match training exactly)
LAGS = [1, 2, 3, 5, 7, 10, 14]
VOL_WINDOWS = [5, 10, 14]
EMA_WINDOWS = [3, 7, 14]

# Sentiment columns (matches Colab preprocessing)
SENT_COLS = [
    "daily_sentiment_decay",
    "news_volume",
    "log_news_volume",
    "decayed_news_volume",
    "high_news_regime",
]


# Yahoo Finance ticker for Brent Crude Oil Futures
BRENT_TICKER = "BZ=F"

# API settings
API_TITLE = "Oil Price Prediction API"
API_DESCRIPTION = (
    "Brent oil price forecasting using the active ensemble model artifacts"
)
API_VERSION = "1.0.0"

# NewsAPI configuration (loaded from environment variables)
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
NEWSDATA_KEY = os.getenv("NEWSDATA_KEY", "")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")

# Pexels fallback image settings
PEXELS_PER_PAGE = int(os.getenv("PEXELS_PER_PAGE", "15"))
PEXELS_TIMEOUT_SECONDS = int(os.getenv("PEXELS_TIMEOUT_SECONDS", "10"))

# Groq API configuration (for LLM narrative generation)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")

# XAI metadata constants
HORIZON_ACCURACY = float(
    os.getenv("HORIZON_ACCURACY", "77.3")
)  # H5 directional accuracy %
MODEL_VERSION = os.getenv("MODEL_VERSION", "v10")

# Hugging Face API configuration (for Mistral keyword extraction)
HUGGING_FACE_API_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN", "")
HUGGING_FACE_LLM_MODEL = os.getenv(
    "HUGGING_FACE_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct:fastest"
)

# Scraper API key — protects /scraper/run from unauthorized calls
# Set this in Render env vars and as SCRAPER_API_KEY GitHub secret
SCRAPER_API_KEY = os.getenv("SCRAPER_API_KEY", "")

# Turso (libsql) database credentials
TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL", "")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN", "")

# Sentiment analysis mode: 'simple' or 'finbert'
# Set to 'finbert' to use the custom ProsusAI/finbert model
SENTIMENT_MODE = os.getenv("SENTIMENT_MODE", "finbert")

# Skip FinBERT preloading (useful for deployments without HuggingFace access)
SKIP_FINBERT_PRELOAD = os.getenv("SKIP_FINBERT_PRELOAD", "false").lower() == "true"

# Hugging Face model revision pin for secure/reproducible from_pretrained downloads.
# Prefer a commit SHA in production; defaults to "main" for compatibility.
FINBERT_MODEL_REVISION = os.getenv("FINBERT_MODEL_REVISION", "main")

# Prediction API performance controls
PREDICT_CACHE_TTL_SECONDS = float(os.getenv("PREDICT_CACHE_TTL_SECONDS", "45"))
PREDICTION_PRECOMPUTE_ENABLED = (
    os.getenv("PREDICTION_PRECOMPUTE_ENABLED", "true").lower() == "true"
)
PREDICTION_PRECOMPUTE_INTERVAL_SECONDS = int(
    os.getenv("PREDICTION_PRECOMPUTE_INTERVAL_SECONDS", "900")
)
PREDICTION_LOCK_SCHEDULE_ENABLED = (
    os.getenv("PREDICTION_LOCK_SCHEDULE_ENABLED", "true").lower() == "true"
)
# Defaults target a post-close lock in local timezone (Asia/Colombo):
# ICE Brent session ends at 03:30 LKT; run at 08:00 LKT (4.5h after close).
PREDICTION_LOCK_SCHEDULE_HOUR = int(os.getenv("PREDICTION_LOCK_SCHEDULE_HOUR", "8"))
PREDICTION_LOCK_SCHEDULE_MINUTE = int(
    os.getenv("PREDICTION_LOCK_SCHEDULE_MINUTE", "0")
)
PREDICTION_LOCK_SCHEDULE_TIMEZONE = os.getenv(
    "PREDICTION_LOCK_SCHEDULE_TIMEZONE", "Asia/Colombo"
)
PREDICTION_CLOSE_LOCK_BUFFER_MINUTES = int(
    os.getenv("PREDICTION_CLOSE_LOCK_BUFFER_MINUTES", "20")
)

# ---------------------------------------------------------------------------
# Daily price sync scheduler
# ---------------------------------------------------------------------------
# Runs independently of the prediction scheduler to keep the prices table
# current every day.  Default: 01:00 UTC (after most market closes).
PRICE_SYNC_SCHEDULE_ENABLED = (
    os.getenv("PRICE_SYNC_SCHEDULE_ENABLED", "true").lower() == "true"
)
PRICE_SYNC_SCHEDULE_HOUR = int(os.getenv("PRICE_SYNC_SCHEDULE_HOUR", "1"))
PRICE_SYNC_SCHEDULE_MINUTE = int(os.getenv("PRICE_SYNC_SCHEDULE_MINUTE", "0"))
PRICE_SYNC_SCHEDULE_TIMEZONE = os.getenv("PRICE_SYNC_SCHEDULE_TIMEZONE", "UTC")

# Explainability scheduler configuration
# Should run safely after the prediction lock job (PREDICTION_LOCK_SCHEDULE_HOUR:MINUTE).
EXPLAINABILITY_SCHEDULE_TIMEZONE = os.getenv(
    "EXPLAINABILITY_SCHEDULE_TIMEZONE", "America/New_York"
)
EXPLAINABILITY_SCHEDULE_HOUR = int(os.getenv("EXPLAINABILITY_SCHEDULE_HOUR", "6"))
EXPLAINABILITY_SCHEDULE_MINUTE = int(os.getenv("EXPLAINABILITY_SCHEDULE_MINUTE", "30"))
EXPLAINABILITY_SCHEDULE_RETRY_HOUR = int(
    os.getenv("EXPLAINABILITY_SCHEDULE_RETRY_HOUR", "8")
)
EXPLAINABILITY_SCHEDULE_RETRY_MINUTE = int(
    os.getenv("EXPLAINABILITY_SCHEDULE_RETRY_MINUTE", "0")
)

# News API performance controls
NEWS_CACHE_TTL_SECONDS = float(os.getenv("NEWS_CACHE_TTL_SECONDS", "300"))

# Web scraper configuration
SCRAPER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
