"""
News scraping and sentiment pipeline.

The daily scrape is triggered externally (GitHub Actions cron → POST /scraper/run).
Can also be triggered manually via the API or run_scraper_now().

Pipeline per run:
1. Scrape articles from all configured sources
2. Compute sentiment features via FinBERT
3. Store results in the SQLite database
4. Apply sentiment decay if no articles found
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

_last_run: Optional[Dict[str, Any]] = None


def _run_daily_scrape(target_date: str = None) -> Dict[str, Any]:
    """
    Execute the full scraping + sentiment pipeline for a single day.

    Args:
        target_date: YYYY-MM-DD string. Defaults to yesterday.

    Returns:
        Summary dict with article count, sentiment value, status.
    """
    global _last_run

    from app.services.news_scraper import scrape_all_sources
    from app.services.news_fetcher import compute_sentiment_features_with_articles
    from app.services.sentiment_service import sentiment_service
    from app.database import add_news_articles

    if target_date is None:
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info("[Scheduler] Starting daily scrape")
    result = {
        "date": target_date,
        "started_at": datetime.now().isoformat(),
        "status": "running",
        "articles_found": 0,
        "sentiment_value": None,
        "decay_applied": False,
        "error": None,
    }

    try:
        # Step 1: Scrape articles
        articles = scrape_all_sources(target_date=target_date)
        result["articles_found"] = len(articles)

        if articles:
            # Step 2: Compute sentiment from scraped articles (also get per-article details)
            features, enriched_articles = compute_sentiment_features_with_articles(
                articles
            )
            result["sentiment_value"] = features["daily_sentiment_decay"]

            # Step 3a: Store per-article data in database
            add_news_articles(target_date, enriched_articles)

            # Step 3b: Store aggregated sentiment in database
            sentiment_service.add_daily_sentiment(
                date_str=target_date,
                daily_sentiment=features["daily_sentiment_decay"],
                news_volume=features["news_volume"],
                log_news_volume=features["log_news_volume"],
                decayed_news_volume=features["decayed_news_volume"],
                high_news_regime=features["high_news_regime"],
            )
            logger.info(
                "[Scheduler] Stored sentiment: %.4f from %d articles",
                features["daily_sentiment_decay"],
                len(articles),
            )
        else:
            # Step 2b: No articles → apply sentiment decay
            logger.warning("[Scheduler] No articles found, applying sentiment decay")
            decay_result = sentiment_service.apply_no_news_decay(target_date)
            result["decay_applied"] = True
            result["sentiment_value"] = decay_result.get("decayed_sentiment", 0.0)
            logger.info("[Scheduler] Applied decay: %.4f", result["sentiment_value"])

        result["status"] = "success"

    except Exception as e:
        logger.error("[Scheduler] Scrape pipeline failed", exc_info=True)
        result["status"] = "error"
        result["error"] = str(e)

    result["completed_at"] = datetime.now().isoformat()

    # Attach FinBERT timing snapshot from the most recent run
    try:
        from app.services.finbert_analyzer import get_finbert_timing
        result["finbert_timing"] = get_finbert_timing()
    except Exception:
        result["finbert_timing"] = None

    _last_run = result
    return result


def run_scraper_now(target_date: str = None) -> Dict[str, Any]:
    """
    Manually trigger a scrape run. Callable from API endpoints.

    Args:
        target_date: YYYY-MM-DD. Defaults to yesterday.

    Returns:
        Run result dict.
    """
    return _run_daily_scrape(target_date=target_date)


def _check_existing_dates(all_dates: List[str]) -> set:
    """
    Check which dates already have sentiment data.

    Args:
        all_dates: List of date strings to check

    Returns:
        Set of date strings that already exist in database
    """
    from app.database import get_sentiment_for_dates
    import pandas as pd

    if not all_dates:
        return set()

    existing_df = get_sentiment_for_dates(all_dates[0], all_dates[-1])
    if existing_df.empty:
        return set()

    return set(pd.to_datetime(existing_df["date"]).dt.strftime("%Y-%m-%d"))


def _process_date_with_articles(date_str: str, articles: List[Dict]) -> Dict[str, Any]:
    """
    Process a date that has articles by computing sentiment.

    Args:
        date_str: Date string (YYYY-MM-DD)
        articles: List of article dictionaries

    Returns:
        Result dictionary with status and sentiment info
    """
    from app.services.news_fetcher import compute_sentiment_features
    from app.services.sentiment_service import sentiment_service

    try:
        features = compute_sentiment_features(articles)
        sentiment_service.add_daily_sentiment(
            date_str=date_str,
            daily_sentiment=features["daily_sentiment_decay"],
            news_volume=features["news_volume"],
            log_news_volume=features["log_news_volume"],
            decayed_news_volume=features["decayed_news_volume"],
            high_news_regime=features["high_news_regime"],
        )
        return {
            "status": "filled",
            "articles": len(articles),
            "sentiment": features["daily_sentiment_decay"],
        }
    except Exception as e:
        logger.error("[Backfill] Failed to process date", exc_info=True)
        return {"status": "error", "error": str(e)}


def _process_date_without_articles(date_str: str) -> Dict[str, Any]:
    """
    Process a date with no articles by applying sentiment decay.

    Args:
        date_str: Date string (YYYY-MM-DD)

    Returns:
        Result dictionary with status and sentiment info
    """
    from app.services.sentiment_service import sentiment_service

    try:
        decay_result = sentiment_service.apply_no_news_decay(date_str)
        return {
            "status": "decayed",
            "sentiment": decay_result.get("decayed_sentiment", 0.0),
        }
    except Exception as e:
        logger.error("[Backfill] Decay failed", exc_info=True)
        return {"status": "error", "error": str(e)}


def _compute_backfill_summary(
    per_date_results: Dict[str, Dict], started_at: str, days_back: int
) -> Dict[str, Any]:
    """
    Compute summary statistics for backfill operation.

    Args:
        per_date_results: Dictionary of per-date results
        started_at: ISO timestamp when backfill started
        days_back: Number of days that were backfilled

    Returns:
        Summary dictionary
    """
    filled = sum(1 for v in per_date_results.values() if v["status"] == "filled")
    decayed = sum(1 for v in per_date_results.values() if v["status"] == "decayed")
    skipped = sum(1 for v in per_date_results.values() if v["status"] == "skipped")
    errors = sum(1 for v in per_date_results.values() if v["status"] == "error")

    return {
        "started_at": started_at,
        "completed_at": datetime.now().isoformat(),
        "days_back": days_back,
        "days_filled": filled,
        "days_decayed": decayed,
        "days_skipped": skipped,
        "days_errored": errors,
        "details": per_date_results,
    }


def backfill_history(
    days_back: int = 30, max_pages_per_site: int = 15
) -> Dict[str, Any]:
    """
    Backfill the last N days of sentiment history by paginating through
    site archives. Designed to be called once after fresh deployment.

    Workflow:
    1. Crawl up to max_pages_per_site from each of the 4 news sites
    2. Group scraped articles by date
    3. For each date with articles → compute sentiment via FinBERT → store
    4. For each date without articles → apply sentiment decay from previous day
    5. Process dates in chronological order so decay chains work correctly

    Args:
        days_back: Number of days to backfill (default 30).
        max_pages_per_site: How many pages to crawl per site (default 15).

    Returns:
        Summary dict with per-date results.
    """
    from app.services.news_scraper import scrape_all_sources_multiday

    logger.info(
        "[Backfill] Starting backfill: %d days, max %d pages/site",
        days_back,
        max_pages_per_site,
    )
    started_at = datetime.now().isoformat()

    # Step 1: Bulk-scrape articles across all pages/sites
    articles_by_date = scrape_all_sources_multiday(
        days_back=days_back,
        max_pages_per_site=max_pages_per_site,
    )

    # Step 2: Build the full list of target dates (chronological order)
    today = datetime.now().date()
    all_dates = sorted(
        [
            (today - timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(days_back, 0, -1)
        ]
    )

    # Step 3: Check which dates already have data
    existing_dates = _check_existing_dates(all_dates)

    # Step 4: Process each date chronologically
    per_date_results = {}
    for date_str in all_dates:
        if date_str in existing_dates:
            per_date_results[date_str] = {
                "status": "skipped",
                "reason": "already_exists",
            }
            continue

        articles = articles_by_date.get(date_str, [])
        if articles:
            per_date_results[date_str] = _process_date_with_articles(date_str, articles)
        else:
            per_date_results[date_str] = _process_date_without_articles(date_str)

    # Summary stats
    summary = _compute_backfill_summary(per_date_results, started_at, days_back)

    logger.info(
        "[Backfill] Complete: %d filled, %d decayed, %d skipped, %d errors",
        summary["days_filled"],
        summary["days_decayed"],
        summary["days_skipped"],
        summary["days_errored"],
    )
    return summary


def get_scheduler_status() -> Dict[str, Any]:
    """Return the cron-based scheduler status and last run info."""
    return {
        "scheduler_mode": "render_cron",
        "schedule": "daily at 02:00 UTC (via Render Cron Job)",
        "last_run": _last_run,
    }
