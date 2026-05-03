"""
Tests for API endpoints.
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, test_client):
        """Test health check returns correct status with market information."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        # Market status fields
        assert "is_market_open" in data
        assert "market_state" not in data
        assert "market_status_message" not in data
        assert "market_open_time" in data
        assert "market_close_time" in data
        assert "timezone_info" in data
        # Verify market hours are correct for Brent Oil
        assert data["market_open_time"] == "02:00 UTC"
        assert data["market_close_time"] == "22:00 UTC"
        # Timezone info is now from Yahoo Finance (real exchange timezone)
        assert "Exchange timezone" in data["timezone_info"]


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_endpoint(self, test_client):
        """Test root endpoint returns API info."""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data
        assert "health" in data
        assert "predict" in data


class TestPricesEndpoint:
    """Tests for prices endpoint."""

    @patch("app.main._sync_latest_prices")
    def test_get_prices_success(self, mock_sync_prices, test_client, sample_prices_df):
        """Test successful price fetch."""
        mock_sync_prices.return_value = sample_prices_df

        response = test_client.get("/prices")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "ticker" in data
        assert "data_points" in data
        assert "prices" in data
        assert len(data["prices"]) > 0
        mock_sync_prices.assert_called_once_with(lookback_days=60)

    @patch("app.main._sync_latest_prices")
    def test_get_prices_error(self, mock_sync_prices, test_client):
        """Test price fetch handles errors."""
        mock_sync_prices.side_effect = Exception("API Error")

        response = test_client.get("/prices")
        assert response.status_code == 500


class TestNewsEndpoint:
    """Tests for news endpoint."""

    @patch("app.main.get_recent_news_articles")
    def test_get_recent_news_success(self, mock_get_recent_news, test_client):
        """Test recent news retrieval for frontend consumption."""
        mock_get_recent_news.return_value = [
            {
                "id": 1,
                "article_date": "2026-03-16",
                "title": "Oil prices edge higher",
                "description": "Brent crude rose on supply concerns.",
                "url": "https://example.com/oil-1",
                "image_url": "https://images.pexels.com/photos/1/sample.jpg",
                "source": "Reuters",
                "published_at": "2026-03-16T09:30:00",
                "sentiment_score": 0.27,
            }
        ]

        response = test_client.get("/news?days=3")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total_records"] == 1
        assert data["days"] == 3
        assert data["latest_article_date"] == "2026-03-16"
        assert data["articles"][0]["title"] == "Oil prices edge higher"
        assert (
            data["articles"][0]["image_url"]
            == "https://images.pexels.com/photos/1/sample.jpg"
        )
        mock_get_recent_news.assert_called_once_with(days=3)

    @patch("app.main.get_news_articles")
    def test_get_news_by_date_success(self, mock_get_news_articles, test_client):
        """Test exact-date article retrieval."""
        mock_get_news_articles.return_value = [
            {
                "id": 2,
                "article_date": "2026-03-15",
                "title": "OPEC output steady",
                "description": "Production remained flat this week.",
                "url": "https://example.com/oil-2",
                "image_url": "https://images.pexels.com/photos/2/sample.jpg",
                "source": "Bloomberg",
                "published_at": "2026-03-15T07:00:00",
                "sentiment_score": -0.05,
                "created_at": "2026-03-15T08:00:00",
            }
        ]

        response = test_client.get("/news?article_date=2026-03-15")
        assert response.status_code == 200
        data = response.json()
        assert data["requested_date"] == "2026-03-15"
        assert data["days"] == 1
        assert data["articles"][0]["source"] == "Bloomberg"
        assert (
            data["articles"][0]["image_url"]
            == "https://images.pexels.com/photos/2/sample.jpg"
        )
        mock_get_news_articles.assert_called_once_with("2026-03-15")

    def test_get_news_invalid_date(self, test_client):
        """Test invalid article date validation."""
        response = test_client.get("/news?article_date=2026-02-30")
        assert response.status_code == 400

    @patch("app.main.get_recent_news_articles")
    def test_get_news_server_error(self, mock_get_recent_news, test_client):
        """Test news endpoint handles storage errors."""
        mock_get_recent_news.side_effect = Exception("DB Error")

        response = test_client.get("/news")
        assert response.status_code == 500


class TestSentimentOverviewEndpoint:
    """Tests for sentiment overview endpoint."""

    @patch("app.main.sentiment_service.get_frontend_sentiment_overview")
    def test_sentiment_overview_success(self, mock_overview, test_client):
        """Endpoint should return frontend-ready sentiment payload."""
        mock_overview.return_value = {
            "success": True,
            "meta": {
                "requested_days": 30,
                "actual_records": 2,
                "start_date": "2026-03-15",
                "end_date": "2026-03-16",
                "decay_lambda": 0.3,
                "decay_factor": 0.7408,
                "decay_formula": "decayed[t] = raw_sentiment[t] + exp(-lambda) * decayed[t-1]",
                "ema_windows": [3, 7, 14],
            },
            "summary": {
                "latest_raw_sentiment": 0.22,
                "latest_decayed_sentiment": 0.36,
                "average_raw_sentiment": 0.18,
                "average_decayed_sentiment": 0.29,
                "average_news_volume": 24.5,
                "high_news_regime_days": 1,
                "positive_days": 2,
                "negative_days": 0,
                "neutral_days": 0,
                "latest_trend": "bullish",
            },
            "timeline": [
                {
                    "date": "2026-03-15",
                    "raw_daily_sentiment": 0.14,
                    "cross_day_decayed_sentiment": 0.14,
                    "sentiment_change_vs_prev_day": 0.0,
                    "decayed_sentiment_change_vs_prev_day": 0.0,
                    "news_volume": 19,
                    "log_news_volume": 2.94,
                    "decayed_news_volume": 16.2,
                    "high_news_regime": False,
                    "ema": {
                        "daily_sentiment_decay_ema_3": 0.14,
                        "news_volume_ema_3": 19.0,
                        "log_news_volume_ema_3": 2.94,
                        "decayed_news_volume_ema_3": 16.2,
                    },
                }
            ],
        }

        response = test_client.get("/sentiment/overview?days=30")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["meta"]["decay_lambda"] == pytest.approx(0.3)
        assert data["summary"]["latest_trend"] == "bullish"
        assert len(data["timeline"]) == 1
        mock_overview.assert_called_once()
        call_kwargs = mock_overview.call_args[1]
        assert call_kwargs["days"] == 30
        assert call_kwargs["end_date"] is None
        assert call_kwargs["include_all_history"] is False

    @patch("app.main.sentiment_service.get_frontend_sentiment_overview")
    def test_sentiment_overview_with_all_history(self, mock_overview, test_client):
        """Endpoint should support include_all_history parameter."""
        mock_overview.return_value = {
            "success": True,
            "meta": {
                "requested_days": 60,
                "actual_records": 4000,
                "start_date": "2014-01-01",
                "end_date": "2025-12-31",
                "decay_lambda": 0.3,
                "decay_factor": 0.7408,
                "decay_formula": "decayed[t] = raw_sentiment[t] + exp(-lambda) * decayed[t-1]",
                "ema_windows": [3, 7, 14],
            },
            "summary": {
                "latest_raw_sentiment": 0.22,
                "latest_decayed_sentiment": 0.36,
                "average_raw_sentiment": 0.18,
                "average_decayed_sentiment": 0.29,
                "average_news_volume": 24.5,
                "high_news_regime_days": 1,
                "positive_days": 2,
                "negative_days": 0,
                "neutral_days": 0,
                "latest_trend": "bullish",
            },
            "timeline": [],
        }

        response = test_client.get("/sentiment/overview?include_all_history=true")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["meta"]["start_date"] == "2014-01-01"
        assert data["meta"]["end_date"] == "2025-12-31"
        mock_overview.assert_called_once()

    @patch("app.main.sentiment_service.get_frontend_sentiment_overview")
    def test_sentiment_overview_with_date_range(self, mock_overview, test_client):
        """Endpoint should support start_date and end_date parameters."""
        mock_overview.return_value = {
            "success": True,
            "meta": {
                "requested_days": 60,
                "actual_records": 4000,
                "start_date": "2014-01-01",
                "end_date": "2025-12-31",
                "decay_lambda": 0.3,
                "decay_factor": 0.7408,
                "decay_formula": "decayed[t] = raw_sentiment[t] + exp(-lambda) * decayed[t-1]",
                "ema_windows": [3, 7, 14],
            },
            "summary": {
                "latest_raw_sentiment": 0.22,
                "latest_decayed_sentiment": 0.36,
                "average_raw_sentiment": 0.18,
                "average_decayed_sentiment": 0.29,
                "average_news_volume": 24.5,
                "high_news_regime_days": 1,
                "positive_days": 2,
                "negative_days": 0,
                "neutral_days": 0,
                "latest_trend": "bullish",
            },
            "timeline": [],
        }

        response = test_client.get("/sentiment/overview?start_date=2014-01-01&end_date=2025-12-31&include_headlines=false")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["meta"]["start_date"] == "2014-01-01"
        assert data["meta"]["end_date"] == "2025-12-31"
        mock_overview.assert_called_once()

    def test_sentiment_overview_invalid_end_date(self, test_client):
        """Invalid end_date should return 400."""
        response = test_client.get("/sentiment/overview?end_date=2026-02-30")
        assert response.status_code == 400


class TestPredictEndpoint:
    """Tests for prediction endpoint."""

    @patch("app.main.get_market_status")
    @patch("app.main.trigger_prediction_job_now")
    @patch("app.main.get_locked_prediction_snapshot")
    def test_predict_success(
        self,
        mock_get_locked_prediction_snapshot,
        mock_trigger_prediction_job_now,
        mock_get_market_status,
        test_client,
    ):
        """Test /predict returns today's locked forecast from database."""
        mock_get_market_status.return_value = {
            "is_open": True,
            "market_state": "REGULAR",
            "message": "Market open (REGULAR)",
            "market_open_time": "01:00 UTC",
            "market_close_time": "23:00 UTC",
            "timezone_info": "Exchange timezone: Europe/London",
        }

        locked_forecasts = [
            {
                "date": (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"),
                "forecasted_price": 75.0 + i * 0.5,
                "forecasted_return": 0.001 * i,
                "horizon": i,
            }
            for i in range(1, 6)
        ]
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
            "forecasts": locked_forecasts,
        }
        mock_trigger_prediction_job_now.return_value = {"status": "success"}

        response = test_client.get("/predict")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "last_price" in data
        assert "forecasts" in data
        assert len(data["forecasts"]) == 14
        assert data["last_price"] == pytest.approx(92.0)
        assert data["last_price_date"] == based_on_date
        assert data["is_market_open"] is True
        assert data["market_open_time"] == "01:00 UTC"
        assert data["market_close_time"] == "23:00 UTC"
        assert data["timezone_info"] == "Exchange timezone: Europe/London"
        mock_trigger_prediction_job_now.assert_not_called()

    @patch("app.main.get_market_status")
    @patch("app.main.trigger_prediction_job_now")
    @patch("app.main.get_locked_prediction_snapshot")
    def test_predict_refreshes_when_snapshot_missing(
        self,
        mock_get_locked_prediction_snapshot,
        mock_trigger_prediction_job_now,
        mock_get_market_status,
        test_client,
    ):
        """If today's snapshot is missing, endpoint triggers refresh and retries."""
        mock_get_market_status.return_value = {
            "is_open": False,
            "market_state": "CLOSED",
            "message": "Market closed (CLOSED)",
            "market_open_time": "01:00 UTC",
            "market_close_time": "23:00 UTC",
            "timezone_info": "Exchange timezone: Europe/London",
        }

        today = datetime.now().date()
        based_on_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        refreshed_snapshot = {
            "source": "locked_for_date",
            "prediction_date": today.strftime("%Y-%m-%d"),
            "last_price_date": based_on_date,
            "last_price": 85.5,
            "based_on_price_date": based_on_date,
            "based_on_price": 85.5,
            "locked_at": "2026-03-20T18:00:00",
            "forecasts": [
                {
                    "date": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
                    "forecasted_price": 86.0,
                    "forecasted_return": 0.01,
                    "horizon": 1,
                }
            ],
        }
        mock_get_locked_prediction_snapshot.side_effect = [None, refreshed_snapshot]
        mock_trigger_prediction_job_now.return_value = {"status": "success"}

        response = test_client.get("/predict")
        assert response.status_code == 200
        body = response.json()
        assert body["prediction_date"] == today.strftime("%Y-%m-%d")
        assert body["last_price_date"] == based_on_date
        assert body["last_price"] == pytest.approx(85.5)
        assert mock_get_locked_prediction_snapshot.call_count == 2
        mock_trigger_prediction_job_now.assert_called_once()

    @patch("app.main.get_market_status")
    @patch("app.main.trigger_prediction_job_now")
    @patch("app.main.get_locked_prediction_snapshot")
    def test_predict_returns_503_when_no_locked_forecast(
        self,
        mock_get_locked_prediction_snapshot,
        mock_trigger_prediction_job_now,
        mock_get_market_status,
        test_client,
    ):
        """When no stored locked rows exist, endpoint should return 503."""
        mock_get_market_status.return_value = {
            "is_open": False,
            "market_state": "CLOSED",
            "message": "Market closed (CLOSED)",
            "market_open_time": "01:00 UTC",
            "market_close_time": "23:00 UTC",
            "timezone_info": "Exchange timezone: Europe/London",
        }
        mock_get_locked_prediction_snapshot.side_effect = [None, None]
        mock_trigger_prediction_job_now.return_value = {
            "status": "failed",
            "error": "no_data",
        }

        response = test_client.get("/predict")
        assert response.status_code == 503


class TestUploadPredictionEndpoints:
    """Tests for upload-based prediction endpoints."""

    @staticmethod
    def _excel_bytes_from_df(df: pd.DataFrame) -> bytes:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        return buffer.getvalue()

    def test_upload_template_download(self, test_client):
        """Template endpoint should return downloadable xlsx content."""
        response = test_client.get("/predict/upload-excel/template")
        assert response.status_code == 200
        assert (
            response.headers["content-type"]
            == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        assert (
            "attachment; filename=oil_price_upload_template.xlsx"
            in response.headers.get("content-disposition", "")
        )
        assert response.content[:2] == b"PK"

    @patch("app.main.get_market_status")
    @patch("app.main.run_prediction_from_uploaded_excel")
    def test_upload_predict_success(
        self, mock_upload_predict, mock_get_market_status, test_client
    ):
        """Upload endpoint should return model payload for valid excel uploads."""
        mock_upload_predict.return_value = {
            "data_source": "Uploaded Excel + Database Backfill + Sentiment History",
            "last_price_date": "2026-03-18",
            "last_price": 84.12,
            "forecasts": [
                {
                    "date": "2026-03-19",
                    "forecasted_price": 84.8,
                    "forecasted_return": 0.01,
                    "horizon": 1,
                }
            ],
            "upload_window": {
                "lookback_days": 21,
                "window_start": "2026-02-17",
                "window_end": "2026-03-18",
                "uploaded_rows_used": 27,
                "filled_from_database": 3,
                "filled_by_carry": 0,
            },
            "resolved_price_window": [
                {"date": "2026-03-18", "price": 84.12, "source": "uploaded"}
            ],
        }
        mock_get_market_status.return_value = {
            "is_open": False,
            "market_state": "CLOSED",
            "message": "Market closed (CLOSED)",
            "market_open_time": "02:00 UTC",
            "market_close_time": "22:00 UTC",
            "timezone_info": "Exchange timezone: UTC",
        }

        valid_df = pd.DataFrame(
            {
                "date": pd.date_range(end=datetime.now(), periods=30, freq="D"),
                "price": np.linspace(80, 85, 30),
            }
        )
        payload = self._excel_bytes_from_df(valid_df)

        response = test_client.post(
            "/predict/upload-excel",
            files={
                "file": (
                    "prices.xlsx",
                    payload,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["last_price"] == pytest.approx(84.12)
        assert len(body["forecasts"]) == 1
        assert body["is_market_open"] is False
        assert body["market_open_time"] == "02:00 UTC"
        assert body["market_close_time"] == "22:00 UTC"
        assert body["timezone_info"] == "Exchange timezone: UTC"

    def test_upload_predict_missing_columns(self, test_client):
        """Upload should fail when required date/price columns are missing."""
        invalid_df = pd.DataFrame(
            {
                "day_number": [1, 2, 3],
                "close_value": [80.0, 81.0, 82.0],
            }
        )
        payload = self._excel_bytes_from_df(invalid_df)

        response = test_client.post(
            "/predict/upload-excel",
            files={
                "file": (
                    "bad_columns.xlsx",
                    payload,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )

        assert response.status_code == 400
        assert "Excel must include date and price columns" in response.json()["detail"]

    @patch("app.main.run_prediction_from_uploaded_excel")
    def test_upload_predict_insufficient_data(self, mock_upload_predict, test_client):
        """Upload should return 400 when lookback window cannot be built."""
        mock_upload_predict.side_effect = ValueError(
            "Insufficient data to build full lookback window"
        )

        valid_df = pd.DataFrame(
            {
                "date": pd.date_range(end=datetime.now(), periods=3, freq="D"),
                "price": [80.0, 81.0, 82.0],
            }
        )
        payload = self._excel_bytes_from_df(valid_df)

        response = test_client.post(
            "/predict/upload-excel",
            files={
                "file": (
                    "small.xlsx",
                    payload,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )

        assert response.status_code == 400
        assert "Insufficient data" in response.json()["detail"]

    def test_upload_predict_empty_file(self, test_client):
        """Upload should reject empty files."""
        response = test_client.post(
            "/predict/upload-excel",
            files={
                "file": (
                    "empty.xlsx",
                    b"",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "Uploaded file is empty"

    def test_upload_predict_invalid_price_text(self, test_client):
        """Upload should reject non-numeric price values with clear row errors."""
        invalid_df = pd.DataFrame(
            {
                "date": ["2026-03-18"],
                "price": ["abc"],
            }
        )
        payload = self._excel_bytes_from_df(invalid_df)

        response = test_client.post(
            "/predict/upload-excel",
            files={
                "file": (
                    "bad_price.xlsx",
                    payload,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "Upload validation failed" in detail
        assert "price must be a numeric value greater than 0" in detail

    def test_upload_predict_invalid_date_format(self, test_client):
        """Upload should reject rows where date is not YYYY-MM-DD."""
        invalid_df = pd.DataFrame(
            {
                "date": ["03/18/2026"],
                "price": [82.35],
            }
        )
        payload = self._excel_bytes_from_df(invalid_df)

        response = test_client.post(
            "/predict/upload-excel",
            files={
                "file": (
                    "bad_date.xlsx",
                    payload,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "Upload validation failed" in detail
        assert "date must be in YYYY-MM-DD format" in detail

    def test_upload_predict_more_than_lookback_rows(self, test_client):
        """Upload should reject files containing more than 21 rows."""
        too_many_df = pd.DataFrame(
            {
                "date": pd.date_range(
                    end=datetime.now(), periods=22, freq="D"
                ).strftime("%Y-%m-%d"),
                "price": np.linspace(80, 85, 22),
            }
        )
        payload = self._excel_bytes_from_df(too_many_df)

        response = test_client.post(
            "/predict/upload-excel",
            files={
                "file": (
                    "too_many.xlsx",
                    payload,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )

        assert response.status_code == 400
        assert "at most 21 days" in response.json()["detail"]

    @patch("app.main.run_prediction_from_uploaded_excel")
    def test_upload_predict_allows_some_missing_prices(
        self,
        mock_upload_predict,
        test_client,
    ):
        """Upload should accept files when some price rows are missing."""
        mock_upload_predict.return_value = {
            "data_source": "Uploaded Excel + Database Backfill + Sentiment History",
            "last_price_date": "2026-03-18",
            "last_price": 84.12,
            "forecasts": [
                {
                    "date": "2026-03-19",
                    "forecasted_price": 84.8,
                    "forecasted_return": 0.01,
                    "horizon": 1,
                }
            ],
            "upload_window": {
                "lookback_days": 21,
                "window_start": "2026-02-17",
                "window_end": "2026-03-18",
                "uploaded_rows_used": 2,
                "filled_from_database": 28,
                "filled_by_carry": 0,
            },
            "resolved_price_window": [
                {"date": "2026-03-18", "price": 84.12, "source": "uploaded"}
            ],
        }

        df = pd.DataFrame(
            {
                "date": ["2026-03-16", "2026-03-17", "2026-03-18"],
                "price": [82.1, "", 84.12],
            }
        )
        payload = self._excel_bytes_from_df(df)

        response = test_client.post(
            "/predict/upload-excel",
            files={
                "file": (
                    "some_missing_prices.xlsx",
                    payload,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )

        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_upload_predict_rejects_all_missing_prices(self, test_client):
        """Upload should reject files when every price value is missing."""
        df = pd.DataFrame(
            {
                "date": ["2026-03-16", "2026-03-17", "2026-03-18"],
                "price": ["", None, np.nan],
            }
        )
        payload = self._excel_bytes_from_df(df)

        response = test_client.post(
            "/predict/upload-excel",
            files={
                "file": (
                    "all_missing_prices.xlsx",
                    payload,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )

        assert response.status_code == 400
        assert "All price values are missing" in response.json()["detail"]


class TestModelInfoEndpoint:
    """Tests for model info endpoint."""

    @patch("app.main.sentiment_service.get_latest_info")
    def test_model_info(self, mock_sentiment, test_client):
        """Test model info endpoint."""
        mock_sentiment.return_value = {
            "total_records": 100,
            "latest_date": "2026-03-01",
        }

        response = test_client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "lookback" in data
        assert "horizon" in data
        assert "device" in data
        assert "models_loaded" in data
        assert "sentiment_data" in data
        assert "components" in data


class TestScraperEndpoints:
    """Tests for scraper endpoints."""

    @patch("app.main.get_scheduler_status")
    def test_scraper_status(self, mock_status, test_client):
        """Test scraper status endpoint."""
        mock_status.return_value = {"enabled": False, "running": False}

        response = test_client.get("/scraper/status")
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data

    @patch("app.main.run_scraper_now")
    def test_scraper_run_success(self, mock_run, test_client):
        """Test manual scraper run."""
        mock_run.return_value = {"status": "success", "articles_scraped": 10}

        response = test_client.post("/scraper/run")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    @patch("app.main.run_scraper_now")
    def test_scraper_run_error(self, mock_run, test_client):
        """Test scraper run handles errors."""
        mock_run.side_effect = Exception("Scraper error")

        response = test_client.post("/scraper/run")
        assert response.status_code == 500

    @patch("app.main.backfill_history")
    def test_scraper_backfill_success(self, mock_backfill, test_client):
        """Test scraper backfill."""
        mock_backfill.return_value = {"status": "success", "days_filled": 30}

        response = test_client.post("/scraper/backfill?days_back=30&max_pages=15")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    @patch("app.main.backfill_history")
    def test_scraper_backfill_error(self, mock_backfill, test_client):
        """Test scraper backfill handles errors."""
        mock_backfill.side_effect = Exception("Backfill error")

        response = test_client.post("/scraper/backfill")
        assert response.status_code == 500
