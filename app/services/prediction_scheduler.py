"""
Daily locked prediction scheduler.

Runs once after market close and stores exactly one locked forecast row per day.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import (
    PREDICTION_CLOSE_LOCK_BUFFER_MINUTES,
    PREDICTION_LOCK_SCHEDULE_ENABLED,
    PREDICTION_LOCK_SCHEDULE_HOUR,
    PREDICTION_LOCK_SCHEDULE_MINUTE,
    PREDICTION_LOCK_SCHEDULE_TIMEZONE,
)
from app.database import add_bulk_prices, upsert_daily_prediction, get_prediction_for_date, get_prices
from app.services.prediction import prediction_service
from app.services.price_fetcher import (
    fetch_latest_prices,
    get_regular_session_window,
    get_canonical_prediction_date,
)

logger = logging.getLogger(__name__)

_scheduler: Optional[AsyncIOScheduler] = None


def _trigger_explainability_after_lock(prediction_date: str) -> dict:
    """
    Attempt explainability generation immediately after locking today's prediction.

    This keeps explanation availability aligned with the locked forecast lifecycle.
    Failures here must not fail the prediction lock itself.
    """
    try:
        from app.services.explainability import explainability_service

        result = explainability_service.run_daily_job()
        result_date = str(result.get("date") or "")

        if result_date and result_date != prediction_date:
            logger.warning(
                "Explainability generated for date=%s but lock job date=%s",
                result_date,
                prediction_date,
            )

        logger.info("Post-lock explainability result: %s", result)
        return result
    except Exception as exc:
        logger.error(
            "Post-lock explainability trigger failed for prediction_date=%s: %s",
            prediction_date,
            exc,
            exc_info=True,
        )
        return {"status": "failed", "error": str(exc), "date": prediction_date}


def _to_yyyymmdd(ts: pd.Timestamp) -> str:
    return pd.to_datetime(ts).strftime("%Y-%m-%d")


def _resolve_previous_trading_close(
    prices: pd.DataFrame, prediction_date: str
) -> tuple[str, float]:
    """
    Select the latest available trading close strictly before prediction_date.
    """
    if prices.empty:
        raise ValueError("No price data available to compute locked prediction")

    working = prices[["date", "price"]].copy()
    working["date"] = pd.to_datetime(working["date"]).dt.tz_localize(None)
    working = working.sort_values("date").reset_index(drop=True)

    prediction_dt = pd.to_datetime(prediction_date)
    candidates = working[working["date"] < prediction_dt]
    if candidates.empty:
        # Fallback to latest available row if historical window is too narrow.
        row = working.iloc[-1]
    else:
        row = candidates.iloc[-1]

    return _to_yyyymmdd(row["date"]), round(float(row["price"]), 2)


def _resolve_stable_close(
    prices: pd.DataFrame, local_now: datetime
) -> tuple[str, float]:
    """
    Select the latest stable daily close based on Yahoo session timing.

    Rule:
    - If current regular session has NOT ended + buffer, use previous trading close.
    - If session ended + buffer, use latest close up to the session date.
    - If Yahoo session metadata is unavailable, fallback to previous-day rule.
    """
    if prices.empty:
        raise ValueError("No price data available to compute locked prediction")

    working = prices[["date", "price"]].copy()
    working["date"] = pd.to_datetime(working["date"]).dt.tz_localize(None)
    working = working.sort_values("date").reset_index(drop=True)

    session = get_regular_session_window()
    if not session:
        prediction_date = local_now.strftime("%Y-%m-%d")
        return _resolve_previous_trading_close(
            prices=working, prediction_date=prediction_date
        )

    exchange_tz = ZoneInfo(str(session["exchange_timezone"]))
    now_exchange = local_now.astimezone(exchange_tz)
    regular_end = session["regular_end"]
    stable_after = regular_end + timedelta(
        minutes=max(0, int(PREDICTION_CLOSE_LOCK_BUFFER_MINUTES))
    )
    session_date = pd.Timestamp(regular_end.date())

    if now_exchange < stable_after:
        candidates = working[working["date"] < session_date]
        selection_reason = "pre_close_or_buffer"
    else:
        candidates = working[working["date"] <= session_date]
        selection_reason = "post_close_buffer"

    if candidates.empty:
        row = working.iloc[-1]
        selection_reason = f"{selection_reason}_fallback_latest"
    else:
        row = candidates.iloc[-1]

    selected_date = _to_yyyymmdd(row["date"])
    selected_price = round(float(row["price"]), 2)
    logger.info(
        "Stable close selection: %s date=%s price=%.2f exchange_now=%s regular_end=%s buffer_min=%s",
        selection_reason,
        selected_date,
        selected_price,
        now_exchange.isoformat(),
        regular_end.isoformat(),
        PREDICTION_CLOSE_LOCK_BUFFER_MINUTES,
    )
    return selected_date, selected_price


def run_daily_prediction_job(now_local: Optional[datetime] = None) -> dict:
    """
    Generate and upsert the single locked forecast row for the current local day.

    Returns:
        Summary dict containing lock metadata and persistence outcome.
    """
    tz = ZoneInfo(PREDICTION_LOCK_SCHEDULE_TIMEZONE)
    local_now = now_local.astimezone(tz) if now_local is not None else datetime.now(tz)
    prediction_date = get_canonical_prediction_date(
        target_timezone=PREDICTION_LOCK_SCHEDULE_TIMEZONE,
        close_lock_buffer_minutes=PREDICTION_CLOSE_LOCK_BUFFER_MINUTES,
        now_target=local_now,
    )

    logger.info(
        "Daily locked prediction job started for prediction_date=%s", prediction_date
    )

    prices = fetch_latest_prices(
        lookback_days=180, end_date=local_now.replace(tzinfo=None)
    )

    # Persist the freshly-fetched prices so the compare endpoint always has
    # up-to-date actuals (the prices table is NOT updated elsewhere).
    try:
        price_records = [
            {"date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"), "price": float(row["price"])}
            for _, row in prices.iterrows()
        ]
        saved = add_bulk_prices(price_records)
        logger.info("Persisted %d price records to prices table", saved)
    except Exception as _price_save_exc:
        logger.error("Failed to persist prices to DB: %s", _price_save_exc, exc_info=True)

    based_on_price_date, based_on_price = _resolve_stable_close(
        prices=prices,
        local_now=local_now,
    )

    # Model receives historical prices up to and including the based-on close date.
    model_prices = prices.copy()
    model_prices["date"] = pd.to_datetime(model_prices["date"]).dt.tz_localize(None)
    model_prices = model_prices[
        model_prices["date"] <= pd.to_datetime(based_on_price_date)
    ]
    model_prices = model_prices.sort_values("date").reset_index(drop=True)

    if model_prices.empty:
        raise ValueError("No model input prices available for locked daily prediction")

    forecasts = prediction_service.predict(prices=model_prices)

    # Keep API response dates aligned to the based-on close (first forecast = next business day).
    current_date = pd.to_datetime(based_on_price_date)
    aligned_forecasts = []
    for forecast in forecasts:
        current_date += pd.Timedelta(days=1)
        while current_date.weekday() >= 5:
            current_date += pd.Timedelta(days=1)

        aligned_forecasts.append(
            {
                **forecast,
                "date": current_date.strftime("%Y-%m-%d"),
            }
        )

    locked_at = datetime.now().isoformat()
    row_id = upsert_daily_prediction(
        prediction_date=prediction_date,
        based_on_price_date=based_on_price_date,
        based_on_price=based_on_price,
        forecasts=aligned_forecasts,
        locked_at=locked_at,
    )

    result = {
        "status": "success",
        "prediction_date": prediction_date,
        "based_on_price_date": based_on_price_date,
        "based_on_price": based_on_price,
        "locked_at": locked_at,
        "row_id": row_id,
        "forecast_points": len(aligned_forecasts),
    }

    # Keep explanation generation in the same daily flow so `/explain`
    # can serve today's prediction rationale shortly after lock.
    result["explainability"] = _trigger_explainability_after_lock(prediction_date)

    logger.info("Daily locked prediction job completed: %s", result)
    return result


def init_prediction_scheduler() -> Optional[AsyncIOScheduler]:
    """Initialize and start the daily locked prediction scheduler."""
    global _scheduler

    if not PREDICTION_LOCK_SCHEDULE_ENABLED:
        logger.info("Locked prediction scheduler disabled by config")
        return None

    if _scheduler is not None and _scheduler.running:
        logger.info("Locked prediction scheduler already running")
        return _scheduler

    tz = ZoneInfo(PREDICTION_LOCK_SCHEDULE_TIMEZONE)
    _scheduler = AsyncIOScheduler(timezone=tz)

    _scheduler.add_job(
        run_daily_prediction_job,
        CronTrigger(
            hour=PREDICTION_LOCK_SCHEDULE_HOUR,
            minute=PREDICTION_LOCK_SCHEDULE_MINUTE,
            timezone=tz,
        ),
        id="daily_locked_prediction",
        name="Daily Locked Oil Forecast",
        replace_existing=True,
    )

    _scheduler.start()
    logger.info(
        "Locked prediction scheduler started at %02d:%02d %s",
        PREDICTION_LOCK_SCHEDULE_HOUR,
        PREDICTION_LOCK_SCHEDULE_MINUTE,
        PREDICTION_LOCK_SCHEDULE_TIMEZONE,
    )
    return _scheduler


def shutdown_prediction_scheduler() -> None:
    """Shutdown locked prediction scheduler if running."""
    global _scheduler

    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=True)
        logger.info("Locked prediction scheduler shut down")
    _scheduler = None


def trigger_prediction_job_now() -> dict:
    """Manual trigger helper for admin/testing usage."""
    try:
        return run_daily_prediction_job()
    except Exception as exc:
        logger.error("Manual daily prediction trigger failed: %s", exc, exc_info=True)
        return {"status": "failed", "error": str(exc)}


def _backfill_one_day(pred_date_str: str, prices_df: pd.DataFrame) -> str:
    """
    Generate and store a locked prediction for a single past business day.

    Args:
        pred_date_str: YYYY-MM-DD prediction date to backfill.
        prices_df: Full price DataFrame (tz-naive, sorted ascending).

    Returns:
        "saved", "skipped", or "no_prices".
    """
    if get_prediction_for_date(pred_date_str):
        return "skipped"

    prices_before = prices_df[prices_df["date"] < pd.Timestamp(pred_date_str)]
    if prices_before.empty:
        logger.warning("Backfill: no prices before %s — skipping", pred_date_str)
        return "no_prices"

    based_on_row = prices_before.iloc[-1]
    based_on_date = pd.Timestamp(based_on_row["date"]).strftime("%Y-%m-%d")
    based_on_price = float(based_on_row["price"])

    model_prices = prices_df[prices_df["date"] <= pd.Timestamp(based_on_date)].copy()
    model_prices = model_prices.sort_values("date").reset_index(drop=True)
    if model_prices.empty:
        return "no_prices"

    forecasts = prediction_service.predict(prices=model_prices)

    current_date = pd.to_datetime(based_on_date)
    aligned: list = []
    for forecast in forecasts:
        current_date += pd.Timedelta(days=1)
        while current_date.weekday() >= 5:
            current_date += pd.Timedelta(days=1)
        aligned.append({**forecast, "date": current_date.strftime("%Y-%m-%d")})

    upsert_daily_prediction(
        prediction_date=pred_date_str,
        based_on_price_date=based_on_date,
        based_on_price=based_on_price,
        forecasts=aligned,
        locked_at=datetime.now().isoformat(),
    )
    logger.info("Backfill: saved prediction for %s based_on=%s", pred_date_str, based_on_date)
    return "saved"


def backfill_missing_locked_predictions(max_days_back: int = 14) -> dict:
    """
    Retroactively generate locked predictions for each business day in the past
    max_days_back days that has stored price data but no locked prediction.

    This repairs gaps caused by the scheduler missing runs (e.g., HF Space sleeping).
    Predictions for those days are generated using prices from the DB up to each
    day's previous trading close, so the compare view can show accurate comparisons.

    Returns:
        Summary dict with backfilled, skipped, and error counts.
    """
    tz = ZoneInfo(PREDICTION_LOCK_SCHEDULE_TIMEZONE)
    today = datetime.now(tz).date()

    candidates = []
    check = today - timedelta(days=1)
    cutoff = today - timedelta(days=max_days_back)
    while check >= cutoff:
        if check.weekday() < 5:
            candidates.append(check)
        check -= timedelta(days=1)

    if not candidates:
        return {"status": "nothing_to_backfill", "backfilled": 0, "skipped": 0, "errors": []}

    prices_df = get_prices(days=max_days_back + 60)
    if prices_df.empty:
        return {"status": "no_prices", "backfilled": 0, "skipped": 0, "errors": []}

    prices_df["date"] = pd.to_datetime(prices_df["date"]).dt.tz_localize(None)
    prices_df = prices_df.sort_values("date").reset_index(drop=True)

    backfilled = 0
    skipped = 0
    errors: list = []

    for pred_date in candidates:
        pred_date_str = pred_date.strftime("%Y-%m-%d")
        try:
            result = _backfill_one_day(pred_date_str, prices_df)
            if result == "saved":
                backfilled += 1
            elif result == "skipped":
                skipped += 1
        except Exception as exc:
            logger.error("Backfill: prediction failed for %s: %s", pred_date_str, exc, exc_info=True)
            errors.append({"date": pred_date_str, "error": str(exc)})

    logger.info(
        "Prediction backfill complete: backfilled=%d skipped=%d errors=%d",
        backfilled, skipped, len(errors),
    )
    return {
        "status": "completed",
        "backfilled": backfilled,
        "skipped": skipped,
        "errors": errors,
    }
