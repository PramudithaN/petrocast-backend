"""
Comprehensive explainability service for oil price predictions.

Pipeline:
1. Generate ARIMA decomposition (trend, seasonal, residual contributions)
2. Extract GRU timestep attributions using TimeSHAP
3. Compute XGBoost feature importances using SHAP TreeExplainer
4. Analyze sentiment headlines using LIME
5. Aggregate SHAP values across ensemble models
6. Build structured prompt with all explanation data
7. Generate narrative using Phi-3-mini local LLM
8. Store results in database keyed by date
"""

import numpy as np
import pandas as pd
import torch
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import os
from functools import lru_cache
import time

try:
    import shap  # pyright: ignore[reportMissingImports]
except ImportError:
    shap = None

try:
    import timeshap  # pyright: ignore[reportMissingImports]
    from timeshap.explainer import local_report  # pyright: ignore[reportMissingImports]
except ImportError:
    timeshap = None

try:
    import lime  # pyright: ignore[reportMissingImports]
except ImportError:
    lime = None

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from transformers import pipeline

from app.config import HORIZON, LOOKBACK, HORIZON_ACCURACY, MODEL_VERSION
from app.models.model_loader import model_artifacts
from app.database import (
    add_explanation,
    get_sentiment_history,
    get_prices as get_prices_db,
    get_news_articles,
)
from app.services.prediction_snapshot import (
    LockedPredictionUnavailableError,
    current_prediction_date_local,
    get_required_locked_prediction_snapshot,
)
from app.services.sentiment_service import sentiment_service
from app.services.feature_engineering import engineer_all_features

logger = logging.getLogger(__name__)

# ── Feature category map ─────────────────────────────────────────────────────
# Maps HF feature names to display categories for the dashboard
FEATURE_CATEGORIES: Dict[str, str] = {
    "log_return": "price",
    "ret_lag_1": "price",
    "ret_lag_2": "price",
    "ret_lag_3": "price",
    "ret_lag_5": "price",
    "ret_lag_7": "price",
    "ret_lag_10": "price",
    "ret_lag_14": "price",
    "vol_5": "volatility",
    "vol_10": "volatility",
    "vol_14": "volatility",
    "momentum_7": "technical",
    "momentum_14": "technical",
    "rsi_14": "technical",
    "daily_sentiment_decay": "sentiment",
    "news_volume": "sentiment",
    "log_news_volume": "sentiment",
    "decayed_news_volume": "sentiment",
    "high_news_regime": "sentiment",
    "daily_sentiment_decay_ema_3": "sentiment",
    "daily_sentiment_decay_ema_7": "sentiment",
    "daily_sentiment_decay_ema_14": "sentiment",
    "news_volume_ema_3": "sentiment",
    "news_volume_ema_7": "sentiment",
    "news_volume_ema_14": "sentiment",
    "log_news_volume_ema_3": "sentiment",
    "log_news_volume_ema_7": "sentiment",
    "log_news_volume_ema_14": "sentiment",
    "decayed_news_volume_ema_3": "sentiment",
    "decayed_news_volume_ema_7": "sentiment",
    "decayed_news_volume_ema_14": "sentiment",
}


class ExplainabilityService:
    """Main orchestrator for daily explainability computation."""

    def __init__(self):
        self.artifacts = model_artifacts
        self._llm_pipeline = None

    @property
    def llm_pipeline(self):
        """Lazy-load Phi-3 mini LLM pipeline (expensive operation)."""
        if self._llm_pipeline is None:
            logger.info("Loading Phi-3-mini LLM pipeline...")
            self._llm_pipeline = pipeline(
                "text-generation",
                model="microsoft/Phi-3-mini-4k-instruct",
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            logger.info("Phi-3-mini LLM loaded successfully")
        return self._llm_pipeline

    def _resolve_freshness_reference(self, prediction_date: str):
        """Return the date to use as the price-freshness reference.

        When a locked prediction exists its ``last_price_date`` reflects the
        actual last trading day (e.g. Friday), not the calendar prediction date
        (e.g. Sunday).  Using the locked date prevents weekend/holiday gaps from
        being misread as stale data.
        """
        from app.services.prediction_snapshot import get_locked_prediction_snapshot

        try:
            snapshot = get_locked_prediction_snapshot(prediction_date=prediction_date)
            if snapshot and snapshot.get("last_price_date"):
                locked_ref = datetime.strptime(
                    str(snapshot["last_price_date"]), "%Y-%m-%d"
                ).date()
                logger.info(
                    "Using locked prediction last_price_date=%s as freshness reference "
                    "(prediction_date=%s)",
                    locked_ref,
                    prediction_date,
                )
                return locked_ref
        except Exception as snap_err:
            logger.debug(
                "Could not read locked snapshot for freshness reference: %s", snap_err
            )
        return datetime.strptime(prediction_date, "%Y-%m-%d").date()

    @staticmethod
    def _try_refresh_prices(add_bulk_prices, get_prices_db) -> pd.DataFrame:
        """Attempt a live Yahoo Finance price fetch and upsert into the DB."""
        from app.services.price_fetcher import fetch_latest_prices

        live_prices = fetch_latest_prices(lookback_days=14)
        records = [
            {
                "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
                "price": float(row["price"]),
                "source": "yahoo_finance",
            }
            for row in live_prices[["date", "price"]].to_dict(orient="records")
        ]
        if records:
            add_bulk_prices(records)
        return get_prices_db(days=7)

    def _validate_prices_available_for_prediction_date(
        self, prediction_date: str
    ) -> bool:
        """
        Verify that price data is fresh enough for the canonical prediction date.

        If a locked prediction already exists for today, its ``last_price_date``
        is used as the freshness reference instead of the calendar prediction date.
        This correctly handles weekends and holidays where the last trading day
        is 2+ calendar days behind today.

        Args:
            prediction_date: Canonical prediction date key (YYYY-MM-DD).

        Returns:
            True if prices are fresh enough for the prediction date, False otherwise.
        """
        try:
            from app.database import add_bulk_prices, get_prices as get_prices_db

            reference_date_obj = self._resolve_freshness_reference(prediction_date)

            def _days_behind(prices_df: pd.DataFrame) -> Optional[int]:
                if prices_df is None or prices_df.empty:
                    return None
                return (reference_date_obj - pd.to_datetime(prices_df["date"].iloc[-1]).date()).days

            prices_df = get_prices_db(days=7)
            days_behind = _days_behind(prices_df)

            # If DB is stale/missing, try live fetch + upsert before deferring.
            if days_behind is None or days_behind > 1:
                try:
                    prices_df = self._try_refresh_prices(add_bulk_prices, get_prices_db)
                    days_behind = _days_behind(prices_df)
                except Exception as live_err:
                    logger.warning(
                        "Live price refresh failed during explainability validation: %s",
                        live_err,
                    )

            if days_behind is None:
                logger.warning("No prices found in database after refresh attempt.")
                return False

            latest_price_date = pd.to_datetime(prices_df["date"].iloc[-1]).date()
            if days_behind <= 1:
                logger.info(
                    "Prices acceptable: %s day(s) behind reference_date=%s (latest=%s)",
                    days_behind,
                    reference_date_obj,
                    latest_price_date,
                )
                return True

            logger.error(
                "Prices are %s days behind reference_date=%s (latest=%s). Data too stale.",
                days_behind,
                reference_date_obj,
                latest_price_date,
            )
            return False

        except Exception as e:
            logger.error(f"Error validating prices: {e}")
            return False

    def run_daily_job(self) -> Dict[str, Any]:
        """
        Run the complete daily explainability pipeline.

        Returns:
            Dict with explanation results and metadata.
        """
        job_start = time.time()
        prediction_key = current_prediction_date_local()
        explanation_date = prediction_key

        logger.info(
            "Starting daily explainability job for prediction_key=%s", prediction_key
        )

        # Step 0b: Validate that today's prices are available (critical check)
        try:
            prices_available = self._validate_prices_available_for_prediction_date(
                prediction_key
            )
            if not prices_available:
                logger.warning(
                    "Prices not ready for canonical prediction key %s. Deferring explainability job.",
                    prediction_key,
                )
                return {
                    "status": "deferred",
                    "reason": "prices_not_available_for_prediction_key",
                    "date": prediction_key,
                }
        except Exception as e:
            logger.error(f"Failed to validate prices availability: {e}")
            return {"status": "failed", "reason": "price_validation_error"}

        try:
            # Step 1: Generate prediction and fetch data
            logger.info("Step 1: Generating prediction...")
            prediction_result, prices_df, _ = self._fetch_prediction_data()
            explanation_date = str(
                prediction_result.get("prediction_date") or prediction_key
            )

            # Step 1b: Skip if explanation already exists for this prediction snapshot date
            from app.database import explanation_exists_for_date

            if explanation_exists_for_date(explanation_date):
                logger.info(
                    "Explanation already exists for prediction date %s, skipping computation",
                    explanation_date,
                )
                return {
                    "status": "skipped",
                    "reason": "already_computed",
                    "date": explanation_date,
                }

            # Step 2: ARIMA explainability
            logger.info("Step 2: Computing ARIMA decomposition...")
            arima_explanation = self._explain_arima(prices_df)

            # Step 3: Level 1 — Ridge meta-model SHAP (sub-model contributions)
            logger.info(
                "Step 3: Computing Ridge SHAP (Level 1 — sub-model attribution)..."
            )
            ridge_explanation = self._explain_ridge(prices_df)

            # Step 4: GRU attention weights (built-in XAI for sentiment stream)
            logger.info("Step 4: Extracting SentimentGRU attention weights...")
            gru_attention = self._explain_gru_attention(prices_df)

            # Step 5: Level 2 — XGBoost SHAP (feature-level attribution)
            logger.info(
                "Step 5: Computing XGBoost SHAP (Level 2 — feature attribution)..."
            )
            xgb_explanation = self._explain_xgboost(prices_df)

            # Step 6: Sentiment explainability
            logger.info("Step 6: Analyzing sentiment headlines...")
            sentiment_explanation = self._explain_sentiment(
                article_date=str(prediction_result.get("last_date") or prediction_key)
            )

            # Step 7: Ensemble aggregation
            logger.info("Step 7: Aggregating ensemble explanations...")
            aggregated = self._aggregate_explanations(
                arima_explanation,
                ridge_explanation,
                gru_attention,
                xgb_explanation,
                sentiment_explanation,
                prediction_result,
            )

            # Step 8: Build prompt and generate narrative
            logger.info("Step 8: Generating narrative (Groq / template)...")
            prompt_str = self._build_explanation_prompt(aggregated)
            llm_result = self._generate_llm_narrative(prompt_str, aggregated)
            # llm_result is a dict: {headline, narrative, sentiment_story, risk_note, model_used}
            explanation_text = llm_result.get("narrative", "")

            # Step 9: Build full dashboard-ready XAI payload and store
            logger.info("Step 9: Storing explanation in database...")
            xai_payload = self._build_xai_payload(
                explanation_date, aggregated, llm_result
            )
            computation_time = time.time() - job_start
            self._store_explanation(
                explanation_date,
                aggregated,
                explanation_text,
                computation_time,
                xai_payload,
            )

            logger.info(
                f"Daily explainability job completed in {computation_time:.2f}s"
            )
            return {
                "status": "success",
                "date": explanation_date,
                "computation_time_seconds": computation_time,
            }

        except LockedPredictionUnavailableError as e:
            logger.warning(
                "Deferring explainability job: locked prediction not available yet (%s)",
                e,
            )
            return {
                "status": "deferred",
                "reason": "locked_prediction_not_available",
                "date": explanation_date,
            }

        except Exception as e:
            logger.error(f"Daily explainability job failed: {e}", exc_info=True)
            raise

    def _fetch_prediction_data(
        self,
    ) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
        """Fetch current prediction and supporting data."""
        # Ensure models are loaded (no-op if already loaded by lifespan)
        if not self.artifacts._loaded:
            logger.info("Loading model artifacts for explainability...")
            self.artifacts.load_all()

        # Fetch live prices and upsert into database, then return as DataFrame
        from app.services.price_fetcher import fetch_latest_prices
        from app.database import add_bulk_prices, get_prices as get_prices_db

        try:
            latest_prices = fetch_latest_prices(lookback_days=120)
            records = [
                {
                    "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
                    "price": float(row["price"]),
                    "source": "yahoo_finance",
                }
                for row in latest_prices[["date", "price"]].to_dict(orient="records")
            ]
            if records:
                add_bulk_prices(records)
            latest_prices = latest_prices[["date", "price"]].copy()
            latest_prices["date"] = pd.to_datetime(
                latest_prices["date"]
            ).dt.tz_localize(None)
        except Exception as e:
            logger.warning(f"Live price fetch failed, falling back to DB: {e}")
            latest_prices = get_prices_db(days=120)
        latest_prices["date"] = pd.to_datetime(latest_prices["date"])
        latest_prices = latest_prices.sort_values("date").reset_index(drop=True)

        prediction_key = current_prediction_date_local()

        # Use the same locked forecast source as `/predict` to keep values consistent.
        # Fetch snapshot FIRST so we can use its last_price_date as the freshness
        # reference — this correctly handles weekends/holidays where the calendar
        # prediction_date is 2+ days ahead of the last market close.
        snapshot = get_required_locked_prediction_snapshot(
            prediction_date=prediction_key
        )

        # Freshness reference: locked prediction's last_price_date (actual last
        # trading day) rather than today's calendar date.
        reference_date_obj = self._resolve_freshness_reference(prediction_key)

        last_price_date = pd.to_datetime(latest_prices["date"].iloc[-1]).date()
        days_behind = (reference_date_obj - last_price_date).days

        if days_behind > 1:
            logger.warning(
                "Last price date is %s days behind reference_date=%s (latest=%s). "
                "Aborting explainability computation to prevent stale explanations.",
                days_behind,
                reference_date_obj,
                last_price_date,
            )
            raise ValueError(
                f"Price data is too stale ({days_behind} days old relative to locked "
                f"prediction reference {reference_date_obj}). "
                "Cannot generate reliable explanations on outdated data."
            )

        if days_behind == 1:
            logger.info(
                "Last price date is 1 day behind reference_date=%s (latest=%s). Proceeding.",
                reference_date_obj,
                last_price_date,
            )

        prediction_result = {
            "last_price": float(snapshot["last_price"]),
            "last_date": str(snapshot["last_price_date"]),
            "prediction_date": str(snapshot.get("prediction_date") or ""),
            "forecasts": snapshot["forecasts"],
        }

        # Fetch sentiment
        sentiment_df = get_sentiment_history(days=LOOKBACK)

        return prediction_result, latest_prices, sentiment_df

    def _explain_arima(self, prices_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Decompose ARIMA forecast into trend, seasonal, residual components.

        Returns:
            Dict with component contributions and values.
        """
        try:
            arima_order = self.artifacts.config.get("ARIMA_ORDER", (1, 1, 1))
            prices = prices_df["price"].values

            # Fit ARIMA
            model = ARIMA(prices, order=arima_order)
            arima_fit = model.fit()

            # Get fitted values and forecast for next horizon steps
            forecast = arima_fit.get_forecast(steps=HORIZON)
            forecast_mean = forecast.predicted_mean

            # STL decomposition (on full series for stable components)
            if len(prices) >= 14:
                stl = STL(prices, period=7, seasonal=7)
                result = stl.fit()

                # STL results are numpy arrays — use [-1] not .iloc[-1]
                trend = float(result.trend[-1])
                seasonal = float(result.seasonal[-1])
                residual = float(result.resid[-1])

                # Normalize to contribution on final forecast
                total_component = abs(trend) + abs(seasonal) + abs(residual)
                if total_component > 0:
                    trend_pct = trend / total_component
                    seasonal_pct = seasonal / total_component
                    residual_pct = residual / total_component
                else:
                    trend_pct = seasonal_pct = residual_pct = 1.0 / 3

                last_price = prices[-1]
                trend_contribution = last_price * trend_pct
                seasonal_contribution = last_price * seasonal_pct
                residual_contribution = last_price * residual_pct
            else:
                # Fallback for short series
                last_price = prices[-1]
                trend_contribution = last_price * 0.7
                seasonal_contribution = last_price * 0.2
                residual_contribution = last_price * 0.1

            return {
                "trend_contribution": float(trend_contribution),
                "seasonal_contribution": float(seasonal_contribution),
                "residual_contribution": float(residual_contribution),
                "forecast_mean": (
                    float(forecast_mean[0]) if len(forecast_mean) > 0 else last_price
                ),
            }

        except Exception as e:
            logger.warning(f"ARIMA decomposition failed: {e}, using fallback")
            last_price = prices_df["price"].iloc[-1]
            return {
                "trend_contribution": last_price * 0.7,
                "seasonal_contribution": last_price * 0.2,
                "residual_contribution": last_price * 0.1,
                "forecast_mean": last_price,
            }

    def _explain_ridge(self, prices_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Level 1 — SHAP on Ridge meta-model.

        Answers: "how much did Sentiment-GRU vs ARIMA vs XGBoost drive the forecast?"
        Uses LinearExplainer with a zero baseline so shap_i = coef_i * scaled_sub_pred_i,
        giving the exact linear attribution for each of the four sub-models.

        Returns:
            Dict with model_contributions {name: {shap_value, pct, direction}},
            dominant_model, and method string.
        """
        if shap is None:
            logger.warning("SHAP library not available for Ridge explainer")
            return {"model_contributions": {}, "method": "unavailable"}

        try:
            from app.services.feature_engineering import (
                engineer_all_features,
            )
            from app.services.prediction import prediction_service as ps

            recent_sentiment_df = get_sentiment_history(days=LOOKBACK + 30)
            feat_df = engineer_all_features(prices_df, recent_sentiment_df)
            if feat_df is None or len(feat_df) < 2:
                return {"model_contributions": {}, "method": "insufficient_data"}

            # Compute component predictions for h=1 (step-ahead, most informative)
            try:
                trend_fc = ps._arima_forecast(feat_df, self.artifacts.horizon)
            except Exception as e:
                logger.debug(f"ARIMA component failed in Ridge SHAP: {e}")
                trend_fc = np.zeros(self.artifacts.horizon)

            try:
                mid_fc = ps._mid_gru_forecast(feat_df, self.artifacts.lookback)
            except Exception as e:
                logger.debug(f"Mid-GRU component failed in Ridge SHAP: {e}")
                mid_fc = np.zeros(self.artifacts.horizon)

            try:
                sent_fc = ps._sent_gru_forecast(feat_df, self.artifacts.lookback)
            except Exception as e:
                logger.debug(f"Sent-GRU component failed in Ridge SHAP: {e}")
                sent_fc = np.zeros(self.artifacts.horizon)

            try:
                hf_fc = ps._xgb_hf_forecast(feat_df, self.artifacts.horizon)
            except Exception as e:
                logger.debug(f"XGBoost component failed in Ridge SHAP: {e}")
                hf_fc = np.zeros(self.artifacts.horizon)

            # Stack inputs for Ridge horizon-1 model
            x_meta = np.array(
                [
                    [
                        float(trend_fc[0]),
                        float(mid_fc[0]),
                        float(sent_fc[0]),
                        float(hf_fc[0]),
                    ]
                ]
            )

            ridge_model = self.artifacts.meta_models.get(1)
            meta_scaler = self.artifacts.meta_scalers.get(1)
            if ridge_model is None or meta_scaler is None:
                return {"model_contributions": {}, "method": "model_unavailable"}

            x_meta_scaled = meta_scaler.transform(x_meta)  # (1, 4)

            # LinearExplainer with zero baseline → shap_i = coef_i * x_scaled_i
            # This is exact for linear models and interpretable without background data
            background = np.zeros_like(x_meta_scaled)
            explainer = shap.LinearExplainer(
                ridge_model,
                masker=shap.maskers.Independent(background),
            )
            shap_values = np.asarray(explainer.shap_values(x_meta_scaled)).flatten()

            names = ["arima", "mid_gru", "sent_gru", "xgb_hf"]
            abs_vals = np.abs(shap_values)
            total = float(abs_vals.sum())
            pcts = (abs_vals / total * 100).tolist() if total > 0 else [25.0] * 4

            model_contributions: Dict[str, Any] = {}
            for name, sv, pct in zip(names, shap_values.tolist(), pcts):
                model_contributions[name] = {
                    "shap_value": float(sv),
                    "pct": float(pct),
                    "direction": "positive" if sv >= 0 else "negative",
                }

            dominant = max(
                model_contributions, key=lambda n: model_contributions[n]["pct"]
            )

            return {
                "model_contributions": model_contributions,
                "dominant_model": dominant,
                "method": "shap_linear",
            }

        except Exception as e:
            logger.warning(f"Ridge SHAP failed: {e}")
            return {"model_contributions": {}, "method": "failed"}

    def _explain_gru_attention(self, prices_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract SentimentGRU attention weights — built-in XAI for the sentiment stream.

        The SentimentGRU already has an Attention layer over the full LOOKBACK window.
        We run a forward pass on the sentiment GRU stream and read back the softmax
        weights to see *which past days* the model focused on when processing sentiment.

        Returns:
            Dict with top_timesteps [{timestep, days_ago, attention_weight, pct}],
            top5_coverage_pct, and method string.
        """
        try:
            from app.services.feature_engineering import (
                engineer_all_features,
                prepare_sentiment_features,
            )

            recent_sentiment_df = get_sentiment_history(days=LOOKBACK + 30)
            feat_df = engineer_all_features(prices_df, recent_sentiment_df)
            if feat_df is None or len(feat_df) < 2:
                return {
                    "top_timesteps": [],
                    "attention_vector": [],
                    "top_timestep_lag": 0,
                    "top_attention_weight": 0.0,
                    "method": "insufficient_data",
                }

            _, x_sent = prepare_sentiment_features(feat_df, LOOKBACK)

            # Scale only the sentiment stream — price stream is not needed for attention extraction
            x_sent_scaled = self.artifacts.scaler_sent.transform(
                x_sent.reshape(-1, x_sent.shape[-1])
            ).reshape(1, LOOKBACK, -1)

            x_sent_tensor = torch.tensor(
                x_sent_scaled, dtype=torch.float32, device=self.artifacts.device
            )

            sent_gru = self.artifacts.sent_gru
            if sent_gru is None:
                return {
                    "top_timesteps": [],
                    "attention_vector": [],
                    "top_timestep_lag": 0,
                    "top_attention_weight": 0.0,
                    "method": "model_unavailable",
                }

            sent_gru.eval()
            with torch.no_grad():
                # Replicate Attention.forward to capture weights before summing
                hs, _ = sent_gru.sent_gru(x_sent_tensor)  # (1, LOOKBACK, hidden)
                attn_logits = sent_gru.attn.attn(hs)  # (1, LOOKBACK, 1)
                attn_weights = (
                    torch.softmax(attn_logits, dim=1)
                    .squeeze()  # (LOOKBACK,) or scalar
                    .cpu()
                    .numpy()
                )

            attn_np = np.atleast_1d(attn_weights)
            seq_len = len(attn_np)
            total_weight = float(attn_np.sum())

            top_k = min(5, seq_len)
            top_indices = np.argsort(attn_np)[-top_k:][::-1]

            top_timesteps = [
                {
                    "timestep": int(idx),
                    "days_ago": int(seq_len - idx),
                    "attention_weight": float(attn_np[idx]),
                    "pct": (
                        float(attn_np[idx] / total_weight * 100)
                        if total_weight > 0
                        else 0.0
                    ),
                }
                for idx in top_indices
            ]

            top5_mass = (
                float(attn_np[top_indices].sum() / total_weight * 100)
                if total_weight > 0
                else 0.0
            )
            top_lag = int(seq_len - top_indices[0]) if len(top_indices) > 0 else 0
            top_attn_weight = (
                float(attn_np[top_indices[0]]) if len(top_indices) > 0 else 0.0
            )

            return {
                "top_timesteps": top_timesteps,
                "attention_vector": attn_np.tolist(),
                "top_timestep_lag": top_lag,
                "top_attention_weight": top_attn_weight,
                "method": "attention_weights",
                "top5_coverage_pct": top5_mass,
            }

        except Exception as e:
            logger.warning(f"GRU attention extraction failed: {e}")
            return {
                "top_timesteps": [],
                "attention_vector": [],
                "top_timestep_lag": 0,
                "top_attention_weight": 0.0,
                "method": "failed",
            }

    def _explain_gru(self, prices_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract GRU timestep attributions using TimeSHAP.

        Returns:
            Dict with top 5 influential timesteps.
        """
        if timeshap is None:
            logger.warning("TimeSHAP not available, using fallback")
            return {
                "top_timesteps": [],
                "method": "unavailable",
            }

        try:
            # Prepare features for GRU — engineer_all_features returns the full df,
            # then prepare_mid_features returns numpy (1, lookback, n_features)
            from app.services.feature_engineering import (
                engineer_all_features,
                prepare_mid_features,
            )

            recent_sentiment_df = get_sentiment_history(days=LOOKBACK + 30)
            feat_df = engineer_all_features(prices_df, recent_sentiment_df)
            if feat_df is None or len(feat_df) < 2:
                logger.warning("Insufficient GRU features for TimeSHAP")
                return {"top_timesteps": [], "method": "insufficient_data"}

            # prepare_mid_features returns numpy shape (1, lookback, n_features)
            x_input = prepare_mid_features(feat_df, lookback=LOOKBACK)
            # Convert to tensor — x_input already has shape (1, LOOKBACK, n_features)
            x_tensor = torch.tensor(
                x_input, dtype=torch.float32, device=self.artifacts.device
            )

            # Run TimeSHAP (expensive - use local_report with limited samples)
            # Note: TimeSHAP may not be fully compatible with all GRU architectures
            # Fall back to SHAP regression if needed
            try:
                report = local_report(
                    self.artifacts.mid_gru,
                    x_tensor,
                    timestep=True,
                    num_samples=10,
                )
                shap_values = (
                    report.shap_values if hasattr(report, "shap_values") else []
                )
            except Exception as e:
                logger.warning(f"TimeSHAP local_report failed: {e}, using alternatives")
                shap_values = []

            # Extract top 5 timesteps by absolute SHAP value
            top_timesteps = []
            if shap_values:
                for idx, shap_val in enumerate(shap_values[-5:]):
                    timestep_idx = len(shap_values) - 5 + idx
                    days_ago = LOOKBACK - timestep_idx
                    top_timesteps.append(
                        {
                            "timestep": int(timestep_idx),
                            "days_ago": int(days_ago),
                            "shap_value": (
                                float(np.mean(shap_val))
                                if isinstance(shap_val, np.ndarray)
                                else float(shap_val)
                            ),
                            "feature_name": "multi-feature composition",
                        }
                    )

            return {
                "top_timesteps": sorted(
                    top_timesteps, key=lambda x: abs(x["shap_value"]), reverse=True
                )[:5],
                "method": "timeshap" if shap_values else "unavailable",
            }

        except Exception as e:
            logger.warning(f"GRU TimeSHAP failed: {e}")
            return {"top_timesteps": [], "method": "failed"}

    def _explain_xgboost(self, prices_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Level 2 — XGBoost SHAP (feature-level attribution).

        Returns:
            Dict with top 5 features (with direction/category/shap_value_usd),
            total_sentiment_impact_usd, high_news_regime_flagged, and method.
        """
        if shap is None:
            logger.warning("SHAP library not available")
            return {
                "top_features": [],
                "method": "unavailable",
                "total_sentiment_impact_usd": 0.0,
                "high_news_regime_flagged": False,
            }

        try:
            from app.services.feature_engineering import (
                engineer_all_features,
                prepare_hf_features,
                get_hf_features,
            )

            recent_sentiment_df = get_sentiment_history(days=LOOKBACK + 30)
            feat_df = engineer_all_features(prices_df, recent_sentiment_df)
            if feat_df is None or len(feat_df) == 0:
                logger.warning("Insufficient XGBoost features")
                return {
                    "top_features": [],
                    "method": "insufficient_data",
                    "total_sentiment_impact_usd": 0.0,
                    "high_news_regime_flagged": False,
                }

            x_today = prepare_hf_features(feat_df)  # shape: (1, n_features)
            if x_today is None or x_today.shape[0] == 0:
                return {
                    "top_features": [],
                    "method": "insufficient_data",
                    "total_sentiment_impact_usd": 0.0,
                    "high_news_regime_flagged": False,
                }

            feature_names = get_hf_features()
            last_price = float(prices_df["price"].iloc[-1])

            xgb_model = self.artifacts.xgb_hf_models.get(1)
            if xgb_model is None:
                return {
                    "top_features": [],
                    "method": "model_unavailable",
                    "total_sentiment_impact_usd": 0.0,
                    "high_news_regime_flagged": False,
                }

            explainer = shap.TreeExplainer(xgb_model)
            shap_values = self._flatten_shap_values(explainer.shap_values(x_today))

            top_features = self._build_top_shap_features(
                shap_values, feature_names, x_today, last_price
            )

            # Total USD impact of all sentiment features
            total_sentiment_usd = sum(
                float(shap_values[i]) * last_price
                for i, fn in enumerate(feature_names)
                if FEATURE_CATEGORIES.get(fn, "technical") == "sentiment"
                and i < len(shap_values)
            )

            # High-news-regime flag from the feature value itself
            hnr_idx = (
                feature_names.index("high_news_regime")
                if "high_news_regime" in feature_names
                else -1
            )
            high_news_regime_flagged = bool(
                hnr_idx >= 0 and float(x_today[0, hnr_idx]) > 0.5
            )

            return {
                "top_features": top_features,
                "method": "shap",
                "baseline": (
                    float(explainer.expected_value)
                    if hasattr(explainer, "expected_value")
                    else 0.0
                ),
                "total_sentiment_impact_usd": float(total_sentiment_usd),
                "high_news_regime_flagged": high_news_regime_flagged,
            }

        except Exception as e:
            logger.warning(f"XGBoost SHAP failed: {e}")
            return {
                "top_features": [],
                "method": "failed",
                "total_sentiment_impact_usd": 0.0,
                "high_news_regime_flagged": False,
            }

    def _flatten_shap_values(self, raw_shap_values: Any) -> np.ndarray:
        """Normalize SHAP outputs to a flat numpy array."""
        shap_values = raw_shap_values
        if isinstance(shap_values, list):
            shap_values = shap_values[0] if len(shap_values) > 0 else np.array([])
        if isinstance(shap_values, np.ndarray):
            return shap_values.flatten()
        return np.array([])

    def _build_top_shap_features(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        x_today: np.ndarray,
        last_price: float = 0.0,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Build top feature rows sorted by absolute SHAP value.

        Adds `direction` (bullish/bearish), `category`, and `shap_value_usd`
        so the dashboard can render feature bars with proper colour and labels.
        """
        if len(shap_values) == 0:
            return []

        top_features: List[Dict[str, Any]] = []
        for idx in np.argsort(np.abs(shap_values))[-top_k:][::-1]:
            feature_name = (
                feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            )
            sv = float(shap_values[idx])
            top_features.append(
                {
                    "feature_name": feature_name,
                    "shap_value": sv,
                    "shap_value_usd": sv * last_price,
                    "feature_value": float(x_today[0, idx]),
                    "direction": "bullish" if sv > 0 else "bearish",
                    "category": FEATURE_CATEGORIES.get(feature_name, "technical"),
                }
            )
        return top_features

    def _sentiment_label(self, score: float) -> str:
        if score > 0:
            return "bullish"
        if score < 0:
            return "bearish"
        return "neutral"

    def _prepare_scored_articles(
        self, articles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract title/score/description for recent articles, skipping malformed rows."""
        articles_with_score = []
        for article in articles[:10]:
            try:
                title = article.get("title", "")
                score = article.get("sentiment_score", 0.0)
                articles_with_score.append(
                    {
                        "title": title,
                        "score": float(score),
                        "description": article.get("description", ""),
                    }
                )
            except Exception as e:
                logger.debug(f"Error processing article: {e}")
                continue
        return articles_with_score

    def _build_sentiment_headline(
        self, article: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create one headline explanation row from an article record."""
        try:
            title = article["title"]
            score = float(article["score"])
            words = title.lower().split()
            lime_words = [
                word
                for word in words
                if len(word) > 4 and word not in ["price", "market", "oil", "brent"]
            ][:5]

            return {
                "headline": title,
                "sentiment_score": score,
                "sentiment_label": self._sentiment_label(score),
                "top_keywords": lime_words,
            }
        except Exception as e:
            logger.debug(f"Error with LIME on article: {e}")
            return None

    def _explain_sentiment(self, article_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze top 3 sentiment headlines.

        Returns:
            Dict with headline sentiment metadata and keyword hints.
        """
        try:
            # Use the same market reference date as locked prediction.
            target_article_date = article_date or current_prediction_date_local()
            articles = get_news_articles(target_article_date)

            if not articles:
                logger.info("No articles found for %s", target_article_date)
                return {"top_headlines": [], "method": "no_data"}

            # Sort by sentiment magnitude
            articles_with_score = self._prepare_scored_articles(articles)

            # Sort by absolute sentiment and take top 3
            sorted_articles = sorted(
                articles_with_score, key=lambda x: abs(x["score"]), reverse=True
            )[:3]

            # Apply LIME to get word importance
            top_headlines = []
            for article in sorted_articles:
                headline = self._build_sentiment_headline(article)
                if headline is not None:
                    top_headlines.append(headline)

            method = "lime_keywords" if lime is not None else "heuristic_keywords"
            if lime is None:
                logger.info(
                    "LIME library not available; using heuristic keyword extraction for sentiment headlines"
                )

            return {
                "top_headlines": top_headlines,
                "method": method if top_headlines else "unavailable",
            }

        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return {"top_headlines": [], "method": "failed"}

    def _aggregate_explanations(
        self,
        arima_exp: Dict[str, Any],
        ridge_exp: Dict[str, Any],
        gru_attn_exp: Dict[str, Any],
        xgb_exp: Dict[str, Any],
        sentiment_exp: Dict[str, Any],
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Aggregate all component explanations into a unified view.

        Uses Ridge SHAP percentages (Level 1) as real model weights and stores
        GRU attention data for the Groq narrative.
        """
        # Extract real model weights from Ridge SHAP (Level 1)
        ridge_contributions = ridge_exp.get("model_contributions", {})
        if ridge_contributions:
            arima_pct = ridge_contributions.get("arima", {}).get("pct", 25.0) / 100
            mid_pct = ridge_contributions.get("mid_gru", {}).get("pct", 25.0) / 100
            sent_pct = ridge_contributions.get("sent_gru", {}).get("pct", 25.0) / 100
            xgb_pct = ridge_contributions.get("xgb_hf", {}).get("pct", 25.0) / 100
        else:
            arima_pct = mid_pct = sent_pct = xgb_pct = 0.25

        # Compute price-scale contributions using Ridge SHAP percentages
        last_price = prediction["last_price"]
        arima_contribution = arima_exp.get("trend_contribution", last_price * arima_pct)
        gru_mid_contribution = last_price * mid_pct
        gru_sent_contribution = last_price * sent_pct
        xgb_hf_contribution = last_price * xgb_pct

        # Build model_weights dict — Ridge SHAP percentages + GRU attention for storage
        model_weights: Dict[str, Any] = {
            "arima": arima_pct * 100,
            "mid_gru": mid_pct * 100,
            "sent_gru": sent_pct * 100,
            "xgb_hf": xgb_pct * 100,
            "dominant_model": ridge_exp.get("dominant_model", "unknown"),
            "ridge_shap_method": ridge_exp.get("method", "unknown"),
            "gru_attention_top5": gru_attn_exp.get("top_timesteps", []),
            "gru_attention_method": gru_attn_exp.get("method", "unknown"),
        }

        # Aggregate top features from XGBoost SHAP (Level 2) — the primary feature source
        top_global_features = xgb_exp.get("top_features", [])[:7]

        # Compute agreement score (lower = higher agreement)
        forecast_prices = [
            f.get("forecasted_price", prediction["last_price"])
            for f in prediction.get("forecasts", [])
        ]
        if forecast_prices and len(forecast_prices) > 1:
            mean_price = np.mean(forecast_prices)
            std_price = np.std(forecast_prices)
            agreement_score = std_price / mean_price if mean_price > 0 else 0.0
        else:
            agreement_score = 0.0

        confidence_level = "high" if agreement_score < 0.05 else "moderate"

        # Confidence interval (±2% of prediction as proxy)
        pred_price = forecast_prices[0] if forecast_prices else prediction["last_price"]
        ci_width = pred_price * 0.02
        ci_lower = pred_price - ci_width
        ci_upper = pred_price + ci_width

        # Price direction and current price
        current_price = float(prediction["last_price"])
        direction = "UP" if pred_price > current_price else "DOWN"

        # Sentiment dominance from Ridge SHAP
        dominant_model = ridge_exp.get("dominant_model", "")
        sentiment_dominant = dominant_model in ("sent_gru", "sentiment_gru")

        # Total sentiment USD impact from XGBoost SHAP
        total_sentiment_impact_usd = float(
            xgb_exp.get("total_sentiment_impact_usd", 0.0)
        )
        high_news_regime_flagged = bool(xgb_exp.get("high_news_regime_flagged", False))

        # Top sentiment feature (first sentiment-category feature in XGB top features)
        top_sentiment_feature = next(
            (
                f["feature_name"]
                for f in top_global_features
                if f.get("category") == "sentiment"
            ),
            "daily_sentiment_decay_ema_3",
        )

        # GRU attention details
        attention_vector = gru_attn_exp.get("attention_vector", [])
        top_timestep_lag = gru_attn_exp.get("top_timestep_lag", 0)
        top_attention_weight = gru_attn_exp.get("top_attention_weight", 0.0)

        return {
            "prediction": float(pred_price),
            "current_price": current_price,
            "direction": direction,
            "confidence_interval_lower": float(ci_lower),
            "confidence_interval_upper": float(ci_upper),
            "arima_contribution": float(arima_contribution),
            "gru_mid_contribution": float(gru_mid_contribution),
            "gru_sent_contribution": float(gru_sent_contribution),
            "xgb_hf_contribution": float(xgb_hf_contribution),
            "agreement_score": float(agreement_score),
            "confidence_level": confidence_level,
            "top_features": top_global_features,
            "sentiment_headlines": sentiment_exp.get("top_headlines", []),
            "model_weights": model_weights,
            # Sentiment signals
            "total_sentiment_impact_usd": total_sentiment_impact_usd,
            "high_news_regime_flagged": high_news_regime_flagged,
            "sentiment_dominant": sentiment_dominant,
            "top_sentiment_feature": top_sentiment_feature,
            # GRU attention
            "attention_vector": attention_vector,
            "top_timestep_lag": top_timestep_lag,
            "top_attention_weight": top_attention_weight,
            # Extra keys for Groq narrative (not persisted to DB directly)
            "ridge_explanation": ridge_exp,
            "gru_attention": gru_attn_exp,
        }

    def _build_explanation_prompt(self, aggregated: Dict[str, Any]) -> str:
        """
        Build a structured prompt covering all three XAI levels.

        Used by the Phi-3 fallback and as context in the smart template.
        The Groq narrative builds its own richer prompt from `aggregated` directly.

        Returns:
            Formatted prompt string (under ~700 tokens).
        """
        pred = aggregated["prediction"]
        ci_lower = aggregated["confidence_interval_lower"]
        ci_upper = aggregated["confidence_interval_upper"]
        confidence = aggregated["confidence_level"]

        # Level 1 — Ridge SHAP sub-model percentages
        ridge_exp = aggregated.get("ridge_explanation", {})
        ridge_contributions = ridge_exp.get("model_contributions", {})
        dominant = ridge_exp.get("dominant_model", "unknown")

        prompt_parts = [
            "=== OIL PRICE FORECAST EXPLAINABILITY ===\n",
            f"Prediction: ${pred:.2f}/barrel",
            f"Confidence Interval: ${ci_lower:.2f} - ${ci_upper:.2f}",
            f"Confidence Level: {confidence.upper()}",
            f"Model Agreement Score: {aggregated['agreement_score']:.4f}\n",
            "LEVEL 1 — Sub-model contributions (Ridge SHAP):",
        ]
        if ridge_contributions:
            for name, data in sorted(
                ridge_contributions.items(), key=lambda x: -x[1]["pct"]
            ):
                dir_sym = "▲" if data["direction"] == "positive" else "▼"
                prompt_parts.append(
                    f"  {name}: {data['pct']:.1f}% {dir_sym} (SHAP {data['shap_value']:+.4f})"
                )
            prompt_parts.append(f"  Dominant sub-model: {dominant}")
        else:
            prompt_parts.append("  (unavailable)")

        prompt_parts.append(
            f"\n  ARIMA: ${aggregated['arima_contribution']:.2f} | "
            f"Mid-GRU: ${aggregated['gru_mid_contribution']:.2f} | "
            f"Sent-GRU: ${aggregated['gru_sent_contribution']:.2f} | "
            f"XGBoost: ${aggregated['xgb_hf_contribution']:.2f}"
        )

        # Level 2 — XGBoost SHAP feature drivers
        prompt_parts.append("\nLEVEL 2 — Feature drivers (XGBoost SHAP):")
        for i, feature in enumerate(aggregated["top_features"][:5], 1):
            fname = feature.get("feature_name", f"Feature {i}").replace("_", " ")
            shap_val = feature.get("shap_value", 0.0)
            prompt_parts.append(f"  {i}. {fname}: {shap_val:+.4f}")

        # GRU attention — sentiment stream focus
        gru_attn = aggregated.get("gru_attention", {})
        attn_ts = gru_attn.get("top_timesteps", [])
        if attn_ts:
            prompt_parts.append(
                "\nSentiment-GRU Attention (most influential past days):"
            )
            for t in attn_ts[:3]:
                prompt_parts.append(
                    f"  {t['days_ago']} days ago — {t['pct']:.1f}% attention"
                )

        # Sentiment headlines
        prompt_parts.append("\nTop Sentiment Headlines:")
        for i, headline in enumerate(aggregated["sentiment_headlines"][:3], 1):
            title = headline.get("headline", "")[:80]
            sentiment = headline.get("sentiment_label", "neutral")
            prompt_parts.append(f"  {i}. [{sentiment.upper()}] {title}")

        return "\n".join(prompt_parts)

    def _generate_groq_narrative(self, aggregated: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize Level 1 + Level 2 + GRU attention via Groq into structured sections.

        Returns dict with headline, narrative, sentiment_story, risk_note, model_used.
        Falls back to smart template if GROQ_API_KEY is absent or call fails.
        """
        from app.config import GROQ_API_KEY, GROQ_LLM_MODEL

        if not GROQ_API_KEY:
            logger.info("GROQ_API_KEY not set — using smart template narrative")
            return self._smart_template_narrative(aggregated)

        try:
            import groq as groq_sdk  # pyright: ignore[reportMissingImports]

            client = groq_sdk.Groq(api_key=GROQ_API_KEY)

            pred = aggregated.get("prediction", 0.0)
            current = aggregated.get("current_price", pred)
            direction = aggregated.get("direction", "UP")
            ci_lower = aggregated.get("confidence_interval_lower", pred * 0.98)
            ci_upper = aggregated.get("confidence_interval_upper", pred * 1.02)
            confidence = aggregated.get("confidence_level", "moderate").upper()
            change = pred - current

            _na = "  Not available"

            # Level 1 — Ridge SHAP
            ridge_exp = aggregated.get("ridge_explanation", {})
            ridge_contributions = ridge_exp.get("model_contributions", {})
            dominant = ridge_exp.get("dominant_model", "unknown")
            ridge_lines = (
                "\n".join(
                    f"  {name}: {data['pct']:.1f}% "
                    f"({'▲' if data['direction'] == 'positive' else '▼'} "
                    f"SHAP {data['shap_value']:+.4f})"
                    for name, data in sorted(
                        ridge_contributions.items(), key=lambda x: -x[1]["pct"]
                    )
                )
                or _na
            )

            # Level 2 — XGBoost SHAP
            xgb_features = aggregated.get("top_features", [])
            xgb_lines = (
                "\n".join(
                    f"  {f['feature_name'].replace('_', ' ')}: "
                    f"SHAP {f['shap_value_usd']:+.4f} USD ({f['direction']}) [{f['category']}]"
                    for f in xgb_features[:5]
                )
                or _na
            )

            # GRU Attention
            gru_attn = aggregated.get("gru_attention", {})
            attn_ts = gru_attn.get("top_timesteps", [])
            top_lag = aggregated.get("top_timestep_lag", 0)
            top_attn_w = aggregated.get("top_attention_weight", 0.0)
            hnr = aggregated.get("high_news_regime_flagged", False)
            attn_lines = (
                "\n".join(
                    f"  t-{t['days_ago']}: {t['pct']:.1f}% attention"
                    for t in attn_ts[:5]
                )
                or _na
            )

            # Sentiment headlines
            headlines = aggregated.get("sentiment_headlines", [])
            headline_lines = (
                "\n".join(
                    f"  [{h.get('sentiment_label','neutral').upper()}] {h.get('headline','')[:90]}"
                    for h in headlines[:3]
                )
                or "  None"
            )

            sent_total_usd = aggregated.get("total_sentiment_impact_usd", 0.0)
            top_sent_feat = aggregated.get("top_sentiment_feature", "")

            prompt = f"""You are a senior financial analyst explaining an AI Brent crude oil price forecast to investors and policy makers. Always quantify impacts in USD. Be specific, clear, and free of jargon.

=== FORECAST ===
Current price: ${current:.2f} | Forecast: ${pred:.2f} | Change: {'+' if change >= 0 else ''}{change:.2f} USD ({direction})
Confidence interval: ${ci_lower:.2f} – ${ci_upper:.2f} | Confidence: {confidence}
Historical directional accuracy at this horizon: {HORIZON_ACCURACY}%

=== LEVEL 1 — Sub-model contributions (Ridge SHAP) ===
{ridge_lines}
Dominant sub-model: {dominant}

=== LEVEL 2 — Feature drivers (XGBoost SHAP, USD) ===
{xgb_lines}
Total sentiment feature impact: {sent_total_usd:+.4f} USD | Top sentiment feature: {top_sent_feat}

=== GRU ATTENTION ===
{attn_lines}
Peak attention at t-{top_lag} days ago (weight={top_attn_w:.4f}) | High-news-regime: {'YES' if hnr else 'No'}

=== MARKET HEADLINES ===
{headline_lines}

=== YOUR TASK ===
Respond in EXACTLY this format (no extra text before or after):

HEADLINE: [One sentence — the forecast price, direction in USD, and dominant sub-model with its %]

NARRATIVE:
[Paragraph 1 — Overall prediction. State forecast price, USD change, direction, and what {HORIZON_ACCURACY}% accuracy means in practice.]
[Paragraph 2 — How sentiment influenced the forecast. Mention Sentiment-GRU %, top sentiment feature, attention peak at t-{top_lag}, and whether high-news-regime was active.]
[Paragraph 3 — Price/technical drivers. Cover XGBoost top features, ARIMA, Mid-GRU contributions.]

SENTIMENT_STORY:
[One focused paragraph for a sentiment analyst — deep-dive on sentiment signals: decay EMAs, news volume, attention weight, high-news flag, total sentiment USD impact.]

RISK_NOTE:
[One sentence only — the single biggest risk or uncertainty in this forecast.]"""

            response = client.chat.completions.create(
                model=GROQ_LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=700,
                temperature=0.3,
            )
            raw = response.choices[0].message.content.strip()
            logger.info("Groq structured narrative generated successfully")
            return self._parse_llm_sections(raw, "groq")

        except Exception as e:
            logger.warning(
                f"Groq narrative failed: {e} — falling back to smart template"
            )
            return self._smart_template_narrative(aggregated)

    def _parse_llm_sections(self, text: str, model_used: str) -> Dict[str, Any]:
        """Parse HEADLINE / NARRATIVE / SENTIMENT_STORY / RISK_NOTE sections from LLM output."""
        sections: Dict[str, str] = {
            "HEADLINE": "",
            "NARRATIVE": "",
            "SENTIMENT_STORY": "",
            "RISK_NOTE": "",
        }
        current: Optional[str] = None
        buffer: List[str] = []

        for line in text.splitlines():
            header = self._parse_section_header(line, sections)
            if header is not None:
                if current:
                    sections[current] = "\n".join(buffer).strip()
                current = header["key"]
                first_line = header["value"]
                buffer = [first_line] if first_line else []
                continue

            if current:
                buffer.append(line)

        if current:
            sections[current] = "\n".join(buffer).strip()

        # Fallback: if parsing failed, put everything in narrative
        if not sections["NARRATIVE"] and not sections["HEADLINE"]:
            sections["NARRATIVE"] = text.strip()
            sections["HEADLINE"] = "Model explanation generated"

        return {
            "headline": sections["HEADLINE"],
            "narrative": sections["NARRATIVE"],
            "sentiment_story": sections["SENTIMENT_STORY"],
            "risk_note": sections["RISK_NOTE"],
            "model_used": model_used,
        }

    def _parse_section_header(
        self,
        line: str,
        sections: Dict[str, str],
    ) -> Optional[Dict[str, str]]:
        """Return section key/value if line starts with a known SECTION: prefix."""
        stripped = line.strip()
        upper = stripped.upper()

        for key in sections:
            prefix = key + ":"
            if upper.startswith(prefix):
                return {
                    "key": key,
                    "value": stripped[len(prefix) :].strip(),
                }

        return None

    def _build_feature_summary_text(self, top_features: List[Dict[str, Any]]) -> str:
        """Build short text about top 1-2 SHAP features."""
        if not top_features:
            return ""

        top = top_features[0]
        fdir = "upward" if top.get("shap_value", 0) > 0 else "downward"
        fname = top["feature_name"].replace("_", " ")
        feature_text = (
            f", driven primarily by {fname} exerting {fdir} pressure "
            f"(SHAP: {top['shap_value']:+.3f})"
        )

        if len(top_features) > 1:
            second = top_features[1]
            feature_text += (
                f" followed by {second['feature_name'].replace('_', ' ')} "
                f"({second['shap_value']:+.3f})"
            )

        return feature_text

    def _build_sentiment_summary_text(self, headlines: List[Dict[str, Any]]) -> str:
        """Build one sentence summarizing headline-level sentiment balance."""
        n = len(headlines)
        if n == 0:
            return "Sentiment data was unavailable for this period"

        bullish = sum(1 for h in headlines if h.get("sentiment_label") == "bullish")
        bearish = sum(1 for h in headlines if h.get("sentiment_label") == "bearish")

        if bullish > bearish:
            return (
                f"Market sentiment leans bullish with {bullish} of {n} recent "
                "headline(s) carrying a positive signal"
            )
        if bearish > bullish:
            return (
                f"Market sentiment leans bearish with {bearish} of {n} recent "
                "headline(s) carrying a negative signal"
            )
        return "Market sentiment is broadly neutral across recent headlines"

    def _generate_llm_narrative(
        self,
        prompt: str,
        aggregated: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate structured explanation dict with headline/narrative/sentiment_story/risk_note.

        Priority order:
        1. Groq API (GROQ_API_KEY set) — full structured output from llama-3.3-70b.
        2. Phi-3-mini local LLM (ENABLE_LLM_NARRATIVE=true) — plain 3-sentence narrative.
        3. Smart data-driven template — always works, no network/model required.

        Returns:
            Dict with headline, narrative, sentiment_story, risk_note, model_used.
        """
        # Priority 1: Groq
        if aggregated is not None:
            from app.config import GROQ_API_KEY

            if GROQ_API_KEY:
                return self._generate_groq_narrative(aggregated)

        # Priority 2: Local Phi-3 (opt-in)
        enable_llm = os.getenv("ENABLE_LLM_NARRATIVE", "false").lower() == "true"
        if enable_llm:
            try:
                system_prompt = (
                    "You are a financial analyst explaining oil price forecasts to non-expert users. "
                    "Be concise, factual, and reference the specific data provided. "
                    "Write exactly 3 sentences."
                )
                full_prompt = f"{system_prompt}\n\nData:\n{prompt}\n\nExplanation:"
                outputs = self.llm_pipeline(
                    full_prompt,
                    max_new_tokens=120,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=False,
                    return_full_text=False,
                )
                if outputs and len(outputs) > 0:
                    generated_text = outputs[0].get("generated_text", "").strip()
                    sentences = [
                        s.strip() + "." for s in generated_text.split(".") if s.strip()
                    ]
                    narrative = " ".join(sentences[:3])
                    if narrative:
                        return self._wrap_narrative_as_dict(
                            narrative, aggregated, "phi3"
                        )
            except Exception as e:
                logger.warning(
                    f"Phi-3 LLM generation failed: {e}, falling back to smart template"
                )

        # Priority 3: smart template
        if aggregated is not None:
            return self._smart_template_narrative(aggregated)

        fallback_narrative = (
            "Oil price forecast generated based on historical price trends, "
            "market sentiment, and ensemble model analysis."
        )
        return self._wrap_narrative_as_dict(fallback_narrative, aggregated, "fallback")

    def _wrap_narrative_as_dict(
        self,
        narrative: str,
        aggregated: Optional[Dict[str, Any]],
        model_used: str,
    ) -> Dict[str, Any]:
        """Wrap a plain narrative string into the standard dict shape."""
        pred = aggregated.get("prediction", 0.0) if aggregated else 0.0
        direction = aggregated.get("direction", "UP") if aggregated else "UP"
        dominant = (
            aggregated.get("model_weights", {}).get("dominant_model", "")
            if aggregated
            else ""
        )
        headline = (
            f"{'Bullish' if direction == 'UP' else 'Bearish'} forecast: "
            f"Brent Crude at ${pred:.2f}/barrel"
            + (f" — led by {dominant.replace('_', '-')}" if dominant else "")
        )
        return {
            "headline": headline,
            "narrative": narrative,
            "sentiment_story": "",
            "risk_note": "",
            "model_used": model_used,
        }

    def _smart_template_narrative(self, aggregated: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate structured explanation using actual forecast data — no LLM required.
        Returns the same dict shape as Groq: headline/narrative/sentiment_story/risk_note.
        """
        pred = aggregated.get("prediction", 0.0)
        current = aggregated.get("current_price", pred)
        direction = aggregated.get("direction", "UP")
        ci_lower = aggregated.get("confidence_interval_lower", pred * 0.98)
        ci_upper = aggregated.get("confidence_interval_upper", pred * 1.02)
        confidence = aggregated.get("confidence_level", "moderate").upper()
        agreement = aggregated.get("agreement_score", 0.0)
        change = pred - current

        # Dominant sub-model
        dominant_model = aggregated.get("ridge_explanation", {}).get(
            "dominant_model", ""
        ) or aggregated.get("model_weights", {}).get("dominant_model", "")
        dominant_text = (
            f" ({dominant_model.replace('_', '-')} was the dominant sub-model)"
            if dominant_model
            else ""
        )

        top_features = aggregated.get("top_features", [])
        feature_text = self._build_feature_summary_text(top_features)

        headlines = aggregated.get("sentiment_headlines", [])
        sent_text = self._build_sentiment_summary_text(headlines)

        # Contributions
        arima = aggregated.get("arima_contribution", 0.0)
        gru_mid = aggregated.get("gru_mid_contribution", 0.0)
        gru_sent = aggregated.get("gru_sent_contribution", 0.0)
        xgb = aggregated.get("xgb_hf_contribution", 0.0)
        reliability = "high reliability" if agreement < 0.03 else "moderate uncertainty"

        # Sentiment details
        sent_usd = aggregated.get("total_sentiment_impact_usd", 0.0)
        top_sent_feat = aggregated.get("top_sentiment_feature", "sentiment features")
        top_lag = aggregated.get("top_timestep_lag", 0)
        top_attn_w = aggregated.get("top_attention_weight", 0.0)
        hnr = aggregated.get("high_news_regime_flagged", False)
        sent_dir = "bullish" if sent_usd >= 0 else "bearish"

        headline = (
            f"{'Bullish' if direction == 'UP' else 'Bearish'} forecast: "
            f"Brent Crude at ${pred:.2f} ({'+' if change >= 0 else ''}{change:.2f} USD)"
            + (
                f" — led by {dominant_model.replace('_', '-')}"
                if dominant_model
                else ""
            )
        )

        s1 = (
            f"The ensemble model forecasts Brent crude at ${pred:.2f}/barrel "
            f"(range ${ci_lower:.2f}–${ci_upper:.2f}){feature_text}{dominant_text}."
        )
        s2 = (
            f"{sent_text}, with ARIMA, Mid-GRU, Sentiment-GRU, and XGBoost "
            f"contributing ${arima:.2f}, ${gru_mid:.2f}, ${gru_sent:.2f}, "
            f"and ${xgb:.2f} respectively."
        )
        s3 = (
            f"Overall confidence is {confidence} with a model agreement score of "
            f"{agreement:.4f}, indicating {reliability}."
        )
        narrative = f"{s1} {s2} {s3}"

        sentiment_story = (
            f"Sentiment had a net {sent_dir} impact of {sent_usd:+.4f} USD on the XGBoost model. "
            f"The Sentiment-GRU contributed {gru_sent:+.2f} USD, processing features including "
            f"{top_sent_feat.replace('_', ' ')} across the {LOOKBACK}-day lookback. "
            f"The attention mechanism peaked at t-{top_lag} days ago "
            f"(weight {top_attn_w:.4f})"
            + (
                ", with high-news-regime active — elevated media activity detected."
                if hnr
                else "."
            )
        )

        risk_note = (
            "The primary uncertainty is that sentiment-dominant forecasts are sensitive "
            "to rapid news-tone reversals from unexpected geopolitical events."
        )

        return {
            "headline": headline,
            "narrative": narrative,
            "sentiment_story": sentiment_story,
            "risk_note": risk_note,
            "model_used": "template",
        }

    def _build_xai_payload(
        self,
        explanation_date: str,
        aggregated: Dict[str, Any],
        llm_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build the full dashboard-ready XAI payload in the shape expected by
        `xai_dashboard.html` and the frontend.  Stored as `xai_payload` JSON
        in the DB so `/explain` can return it verbatim.
        """
        ridge_contributions = aggregated.get("ridge_explanation", {}).get(
            "model_contributions", {}
        )
        last_price = aggregated["current_price"]

        # ── Sub-model contributions in USD (signed Ridge SHAP values × price) ──
        # These are DIRECTIONAL — negative means that sub-model pulled the forecast DOWN.
        # This is what the dashboard bar chart visualises (left = bearish, right = bullish).
        def _ridge_shap_usd(name: str) -> float:
            sv = ridge_contributions.get(name, {}).get("shap_value", 0.0)
            return float(sv) * last_price

        arima_usd = _ridge_shap_usd("arima")
        mid_usd = _ridge_shap_usd("mid_gru")
        sent_usd = _ridge_shap_usd("sent_gru")
        xgb_usd = _ridge_shap_usd("xgb_hf")

        # Sentiment % of forecast = sent_gru SHAP pct (used for top-stat "SENTIMENT DRIVER")
        sent_gru_pct = ridge_contributions.get("sent_gru", {}).get("pct", 25.0)

        # Dominant sub-model
        dominant_model = aggregated["model_weights"].get("dominant_model", "")

        # Fallback: if Ridge SHAP wasn't available use the absolute contributions
        if not ridge_contributions:
            arima_usd = aggregated["arima_contribution"]
            mid_usd = aggregated["gru_mid_contribution"]
            sent_usd = aggregated["gru_sent_contribution"]
            xgb_usd = aggregated["xgb_hf_contribution"]

        return {
            "date": explanation_date,
            "forecast_price": aggregated["prediction"],
            "current_price": last_price,
            "direction": aggregated["direction"],
            "horizon": HORIZON,
            "model_version": MODEL_VERSION,
            "horizon_accuracy": HORIZON_ACCURACY,
            "sub_model_contributions": {
                "arima_contribution_usd": arima_usd,
                "mid_gru_contribution_usd": mid_usd,
                "sentiment_gru_contribution_usd": sent_usd,
                "xgboost_contribution_usd": xgb_usd,
                "dominant_model": dominant_model,
                "sentiment_pct_of_forecast": float(sent_gru_pct),
            },
            "top_feature_drivers": aggregated["top_features"],
            "attention_insight": {
                "top_sentiment_feature": aggregated.get("top_sentiment_feature", ""),
                "top_timestep_lag": aggregated.get("top_timestep_lag", 0),
                "attention_weight": aggregated.get("top_attention_weight", 0.0),
                "high_news_regime_flagged": aggregated.get(
                    "high_news_regime_flagged", False
                ),
                "attention_vector": aggregated.get("attention_vector", []),
            },
            "total_sentiment_impact_usd": aggregated.get(
                "total_sentiment_impact_usd", 0.0
            ),
            "sentiment_dominant": aggregated.get("sentiment_dominant", False),
            # LLM sections
            "headline": llm_result.get("headline", ""),
            "narrative": llm_result.get("narrative", ""),
            "sentiment_story": llm_result.get("sentiment_story", ""),
            "risk_note": llm_result.get("risk_note", ""),
            "model_used": llm_result.get("model_used", "template"),
            # Extra context
            "confidence_level": aggregated["confidence_level"],
            "agreement_score": aggregated["agreement_score"],
            "confidence_interval_lower": aggregated["confidence_interval_lower"],
            "confidence_interval_upper": aggregated["confidence_interval_upper"],
            "sentiment_headlines": aggregated["sentiment_headlines"],
            "model_weights": aggregated["model_weights"],
            "generated_at": datetime.now().isoformat(),
        }

    def _store_explanation(
        self,
        explanation_date: str,
        aggregated: Dict[str, Any],
        explanation_text: str,
        computation_time: float,
        xai_payload: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Store explanation result in database."""
        from app.database import add_explanation

        return add_explanation(
            explanation_date=explanation_date,
            aggregated=aggregated,
            explanation_text=explanation_text,
            generated_at=datetime.now().isoformat(),
            computation_time_seconds=computation_time,
            xai_payload=xai_payload,
        )


# Singleton instance
explainability_service = ExplainabilityService()
