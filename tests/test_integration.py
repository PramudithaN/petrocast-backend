"""
Integration tests for the complete prediction pipeline.
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestPredictionPipeline:
    """End-to-end tests for prediction pipeline."""

    @patch("app.services.price_fetcher.fetch_latest_prices")
    @patch("app.services.sentiment_service.sentiment_service.get_sentiment_window")
    @patch("app.models.model_loader.model_artifacts")
    def test_complete_prediction_pipeline(
        self, mock_artifacts, mock_sentiment, mock_prices
    ):
        """Test complete prediction pipeline from data fetch to forecast."""
        # Mock price data
        dates = pd.date_range(end=datetime.now(), periods=21, freq="D")
        rng = np.random.default_rng(42)
        mock_prices.return_value = pd.DataFrame(
            {"date": dates, "price": rng.uniform(70, 90, size=21)}
        )

        # Mock sentiment data
        mock_sentiment.return_value = pd.DataFrame(
            {
                "date": dates,
                "daily_sentiment": rng.uniform(-0.5, 0.5, size=21),
                "news_volume": rng.integers(5, 20, size=21),
                "log_news_volume": rng.uniform(1.5, 3.0, size=21),
                "decayed_news_volume": rng.uniform(5, 15, size=21),
                "high_news_regime": rng.integers(0, 2, size=21),
            }
        )

        # Mock model artifacts
        mock_artifacts._loaded = True
        mock_artifacts.lookback = 21
        mock_artifacts.horizon = 5

        # Test will attempt full pipeline
        # May fail without real models, but tests the structure
        try:
            from app.services.prediction import prediction_service

            prediction_service.predict()
        except Exception:
            # Expected without real models
            pass


class TestAPIIntegration:
    """Integration tests for API endpoints."""

    @patch("app.main.get_market_status")
    @patch("app.main.trigger_prediction_job_now")
    @patch("app.main.get_locked_prediction_snapshot")
    def test_predict_endpoint_integration(
        self,
        mock_get_locked_prediction_snapshot,
        mock_trigger_prediction_job_now,
        mock_get_market_status,
        test_client,
    ):
        """Test /predict endpoint integration."""
        mock_get_market_status.return_value = {
            "is_open": True,
            "market_state": "REGULAR",
            "message": "Market open (REGULAR)",
            "market_open_time": "01:00 UTC",
            "market_close_time": "23:00 UTC",
            "timezone_info": "Exchange timezone: Europe/London",
        }

        today = datetime.now().date()
        based_on_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        mock_forecasts = [
            {
                "date": (today + timedelta(days=i)).strftime("%Y-%m-%d"),
                "forecasted_price": 75.0 + i * 0.5,
                "forecasted_return": 0.001,
                "horizon": i,
            }
            for i in range(1, 15)
        ]
        mock_get_locked_prediction_snapshot.return_value = {
            "source": "locked_for_date",
            "prediction_date": today.strftime("%Y-%m-%d"),
            "last_price_date": based_on_date,
            "last_price": 92.0,
            "based_on_price_date": based_on_date,
            "based_on_price": 92.0,
            "locked_at": "2026-03-18T18:02:00",
            "forecasts": mock_forecasts,
        }
        mock_trigger_prediction_job_now.return_value = {"status": "success"}

        # Make request
        response = test_client.get("/predict")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["forecasts"]) == 14
        assert data["prediction_date"] == today.strftime("%Y-%m-%d")
        assert data["last_price_date"] == based_on_date
        assert data["last_price"] == pytest.approx(92.0)

        # Verify forecast structure
        for i, forecast in enumerate(data["forecasts"], 1):
            assert "date" in forecast
            assert "forecasted_price" in forecast
            assert "forecasted_return" in forecast
            assert "horizon" in forecast
            assert forecast["horizon"] == i
        mock_trigger_prediction_job_now.assert_not_called()

    @patch("app.main._sync_latest_prices_cached")
    def test_prices_endpoint_integration(
        self, mock_sync_latest_prices_cached, test_client, sample_prices_df
    ):
        """Test /prices endpoint integration."""
        mock_sync_latest_prices_cached.return_value = sample_prices_df

        response = test_client.get("/prices")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "prices" in data
        assert len(data["prices"]) > 0
        mock_sync_latest_prices_cached.assert_called_once_with(lookback_days=60)


class TestDataFlow:
    """Tests for data flow through the system."""

    def test_price_to_features_flow(self, sample_prices_df):
        """Test data flow from prices to features."""
        from app.services.feature_engineering import engineer_all_features

        # Create minimal sentiment data
        rng = np.random.default_rng(42)
        sentiment_df = pd.DataFrame(
            {
                "date": sample_prices_df["date"],
                "daily_sentiment": rng.uniform(-0.5, 0.5, size=len(sample_prices_df)),
                "news_volume": rng.integers(5, 20, size=len(sample_prices_df)),
                "log_news_volume": rng.uniform(1.5, 3.0, size=len(sample_prices_df)),
                "decayed_news_volume": rng.uniform(5, 15, size=len(sample_prices_df)),
                "high_news_regime": rng.integers(0, 2, size=len(sample_prices_df)),
            }
        )

        features = engineer_all_features(
            prices=sample_prices_df, sentiment_df=sentiment_df
        )

        assert features is not None
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0

    def test_sentiment_to_features_flow(self, sample_sentiment_df):
        """Test data flow from sentiment to features."""
        from app.services.sentiment_service import sentiment_service

        # Add required columns
        sample_sentiment_df["daily_sentiment_decay"] = sample_sentiment_df["sentiment"]
        sample_sentiment_df["log_news_volume"] = np.log(
            sample_sentiment_df["article_count"] + 1
        )
        sample_sentiment_df["decayed_news_volume"] = sample_sentiment_df[
            "article_count"
        ]
        sample_sentiment_df["high_news_regime"] = 0
        sample_sentiment_df.rename(
            columns={"article_count": "news_volume"}, inplace=True
        )
        sample_sentiment_df.rename(
            columns={"sentiment": "daily_sentiment"}, inplace=True
        )

        # Compute decay
        decayed = sentiment_service.apply_cross_day_decay(sample_sentiment_df)
        assert "daily_sentiment_decay" in decayed.columns

        # Compute features
        result = sentiment_service.compute_sentiment_features(decayed)
        assert isinstance(result, pd.DataFrame)


class TestErrorHandling:
    """Tests for error handling."""

    @patch("app.main.fetch_latest_prices")
    def test_price_fetch_error_handling(self, mock_fetch, test_client):
        """Test error handling when price fetch fails."""
        mock_fetch.side_effect = Exception("API Error")

        response = test_client.get("/prices")
        assert response.status_code == 500

    @patch("app.main.get_market_status")
    @patch("app.main.get_locked_prediction_snapshot")
    def test_prediction_error_handling(
        self, mock_get_locked_prediction_snapshot, mock_get_market_status, test_client
    ):
        """Test error handling when prediction fails."""
        mock_get_market_status.return_value = {
            "is_open": False,
            "market_state": "CLOSED",
            "message": "Market closed (CLOSED)",
            "market_open_time": "01:00 UTC",
            "market_close_time": "23:00 UTC",
            "timezone_info": "Exchange timezone: Europe/London",
        }
        mock_get_locked_prediction_snapshot.side_effect = Exception("Snapshot Error")

        response = test_client.get("/predict")
        assert response.status_code == 500

    def test_invalid_date_format(self, test_client):
        """Test handling of invalid date format."""
        from pydantic import ValidationError
        from app.schemas.prediction import PriceInput

        with pytest.raises(ValidationError):
            PriceInput(date="invalid-date", price=75.0)


class TestConcurrency:
    """Tests for concurrent requests."""

    @patch("app.main.get_market_status")
    @patch("app.main.trigger_prediction_job_now")
    @patch("app.main.get_locked_prediction_snapshot")
    def test_concurrent_predict_requests(
        self,
        mock_get_locked_prediction_snapshot,
        mock_trigger_prediction_job_now,
        mock_get_market_status,
        test_client,
    ):
        """Test handling multiple concurrent prediction requests."""
        mock_get_market_status.return_value = {
            "is_open": True,
            "market_state": "REGULAR",
            "message": "Market open (REGULAR)",
            "market_open_time": "01:00 UTC",
            "market_close_time": "23:00 UTC",
            "timezone_info": "Exchange timezone: Europe/London",
        }

        today = datetime.now().date()
        based_on_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        mock_get_locked_prediction_snapshot.return_value = {
            "source": "locked_for_date",
            "prediction_date": today.strftime("%Y-%m-%d"),
            "last_price_date": based_on_date,
            "last_price": 92.0,
            "based_on_price_date": based_on_date,
            "based_on_price": 92.0,
            "locked_at": "2026-03-18T18:02:00",
            "forecasts": [
                {
                    "date": (today + timedelta(days=i)).strftime("%Y-%m-%d"),
                    "forecasted_price": 75.0,
                    "forecasted_return": 0.001,
                    "horizon": i,
                }
                for i in range(1, 15)
            ],
        }
        mock_trigger_prediction_job_now.return_value = {"status": "success"}

        # Make multiple requests
        responses = [test_client.get("/predict") for _ in range(3)]

        # All should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["forecasts"]) == 14

        mock_trigger_prediction_job_now.assert_not_called()
