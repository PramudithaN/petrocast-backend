"""
Model artifact loader - loads all trained models at startup.
"""

import json
import numpy as np
import torch
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from app.config import MODEL_ARTIFACTS_DIR, HORIZON
from app.models.gru_models import MidFreqGRU, SentimentGRU

logger = logging.getLogger(__name__)


class ModelArtifacts:
    """
    Container for all loaded model artifacts.
    Loaded once at application startup.
    """

    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.artifacts_dir = MODEL_ARTIFACTS_DIR
        self._is_h5_bundle = False
        self.scaler_mid = None
        self.scaler_price = None
        self.scaler_sent = None
        self.meta_models: Dict[int, Any] = {}
        self.meta_scalers: Dict[int, Any] = {}
        self.xgb_hf_models: Dict[int, Any] = {}
        self.error_stds: Dict[int, float] = {}
        self.arima_model = None
        self.mid_gru: Optional[MidFreqGRU] = None
        self.sent_gru: Optional[SentimentGRU] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False

    def load_all(self) -> None:
        """Load all model artifacts from disk."""
        if self._loaded:
            logger.info("Models already loaded, skipping...")
            return

        self.artifacts_dir, self._is_h5_bundle = self._resolve_artifact_layout()
        logger.info(f"Loading model artifacts from {self.artifacts_dir}")
        logger.info(f"Artifact profile: {'h5' if self._is_h5_bundle else 'legacy'}")
        logger.info(f"Using device: {self.device}")

        # Load configuration
        self._load_config()

        # Load scalers
        self._load_scalers()

        # Load meta-ensemble models
        self._load_meta_models()

        # Load XGBoost HF models
        self._load_xgb_models()

        # Load per-horizon forecast error standard deviations (optional)
        self._load_error_stds()

        # Load ARIMA artifact/order
        self._load_arima_artifacts()

        # Load PyTorch GRU models
        self._load_gru_models()

        self._loaded = True
        logger.info("All model artifacts loaded successfully!")

    def _resolve_artifact_layout(self) -> tuple[Path, bool]:
        """Resolve whether we are loading new H5 bundle artifacts or legacy files."""
        h5_config = MODEL_ARTIFACTS_DIR / "model_config_h5.json"
        if h5_config.exists():
            return MODEL_ARTIFACTS_DIR, True
        return MODEL_ARTIFACTS_DIR, False

    def _load_config(self) -> None:
        """Load configuration from H5 JSON or legacy pickle config."""
        if self._is_h5_bundle:
            config_path = self.artifacts_dir / "model_config_h5.json"
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            config_path = self.artifacts_dir / "config.pkl"
            self.config = joblib.load(config_path)

        logger.info(
            "Config loaded: LOOKBACK=%s, HORIZON=%s, ARIMA_ORDER=%s",
            self.lookback,
            self.horizon,
            self.arima_order,
        )

    def _load_scalers(self) -> None:
        """Load all sklearn scalers."""
        if self._is_h5_bundle:
            self.scaler_mid = joblib.load(self.artifacts_dir / "scaler_mid_h5.pkl")
            self.scaler_price = joblib.load(self.artifacts_dir / "scaler_price_h5.pkl")
            self.scaler_sent = joblib.load(self.artifacts_dir / "scaler_sent_h5.pkl")
        else:
            self.scaler_mid = joblib.load(self.artifacts_dir / "scaler_mid.pkl")
            self.scaler_price = joblib.load(self.artifacts_dir / "scaler_price.pkl")
            self.scaler_sent = joblib.load(self.artifacts_dir / "scaler_sent.pkl")

        # Check for scaler_mid mismatch (known issue from training notebook)
        # The mid-GRU expects 14 features, but scaler_mid may have been saved with fewer
        logger.info(
            f"Scalers loaded - mid: {self.scaler_mid.n_features_in_}, "
            f"price: {self.scaler_price.n_features_in_}, "
            f"sent: {self.scaler_sent.n_features_in_}"
        )

    def _load_meta_models(self) -> None:
        """Load Ridge meta-ensemble models and their scalers."""
        if self._is_h5_bundle:
            self.meta_models = joblib.load(self.artifacts_dir / "meta_models_h5.pkl")
            self.meta_scalers = joblib.load(self.artifacts_dir / "meta_scalers_h5.pkl")
        else:
            self.meta_models = joblib.load(self.artifacts_dir / "meta_models.pkl")
            self.meta_scalers = joblib.load(self.artifacts_dir / "meta_scalers.pkl")
        logger.info(f"Meta models loaded for horizons: {list(self.meta_models.keys())}")

    def _load_xgb_models(self) -> None:
        """Load XGBoost high-frequency models."""
        if self._is_h5_bundle:
            self.xgb_hf_models = joblib.load(self.artifacts_dir / "xgb_models_h5.pkl")
        else:
            self.xgb_hf_models = joblib.load(self.artifacts_dir / "xgb_hf_models.pkl")
        logger.info(
            f"XGBoost HF models loaded for horizons: {list(self.xgb_hf_models.keys())}"
        )

    def _load_error_stds(self) -> None:
        """Load optional horizon-wise forecast error standard deviations."""
        self.error_stds = {}

        if not self._is_h5_bundle:
            return

        error_stds_path = self.artifacts_dir / "error_stds_h5.json"
        if not error_stds_path.exists():
            logger.info("No error_stds_h5.json found; forecast bounds will use fallback")
            return

        with open(error_stds_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        parsed: Dict[int, float] = {}
        for key, value in raw.items():
            try:
                horizon = int(key)
                std_value = float(value)
            except (TypeError, ValueError):
                continue

            if horizon >= 1 and std_value > 0:
                parsed[horizon] = std_value

        self.error_stds = parsed
        logger.info("Loaded error stds for horizons: %s", sorted(self.error_stds.keys()))

    def _load_arima_artifacts(self) -> None:
        """Load saved ARIMA model (if available) and ARIMA order."""
        self.arima_model = None

        if self._is_h5_bundle:
            arima_model_path = self.artifacts_dir / "arima_model_h5.pkl"
            arima_order_path = self.artifacts_dir / "arima_order_h5.pkl"

            if arima_model_path.exists():
                self.arima_model = joblib.load(arima_model_path)
                logger.info("ARIMA model loaded")

            if arima_order_path.exists() and "arima_order" not in self.config:
                order_data = joblib.load(arima_order_path)
                best_order = (
                    order_data.get("best_order")
                    if isinstance(order_data, dict)
                    else None
                )
                if best_order is not None:
                    self.config["arima_order"] = list(best_order)
        else:
            arima_model_path = self.artifacts_dir / "arima_model.pkl"
            if arima_model_path.exists():
                self.arima_model = joblib.load(arima_model_path)
                logger.info("ARIMA model loaded")

    def _load_gru_models(self) -> None:
        """Load PyTorch GRU models."""
        from sklearn.preprocessing import StandardScaler

        horizon = self.horizon

        # Get feature dimensions from scalers
        n_price_features = self.scaler_price.n_features_in_
        n_sent_features = self.scaler_sent.n_features_in_

        # For mid-GRU, check the actual model weight dimensions
        mid_gru_path = (
            self.artifacts_dir / "mid_gru_best_h5.pt"
            if self._is_h5_bundle
            else self.artifacts_dir / "mid_gru.pt"
        )
        mid_state = torch.load(mid_gru_path, map_location="cpu", weights_only=True)
        n_mid_features = mid_state["gru.weight_ih_l0"].shape[1]

        logger.info(
            f"Feature dimensions - mid: {n_mid_features}, "
            f"price: {n_price_features}, sent: {n_sent_features}"
        )

        # Handle scaler_mid mismatch
        if self.scaler_mid.n_features_in_ != n_mid_features:
            logger.warning(
                f"scaler_mid dimension mismatch: saved={self.scaler_mid.n_features_in_}, "
                f"model expects={n_mid_features}. Creating new StandardScaler."
            )
            # Create a new scaler that will use identity transform
            # (mean=0, std=1 for all features - no scaling)
            self.scaler_mid = StandardScaler()
            # Fit with dummy data to set the right dimensions
            dummy_data = np.zeros((1, n_mid_features))
            self.scaler_mid.fit(dummy_data)
            # Set to identity transform (no actual scaling)
            self.scaler_mid.mean_ = np.zeros(n_mid_features)
            self.scaler_mid.scale_ = np.ones(n_mid_features)
            self.scaler_mid.var_ = np.ones(n_mid_features)
            logger.info("Created identity StandardScaler for mid features")

        # Load Mid-frequency GRU
        self.mid_gru = MidFreqGRU(
            n_features=n_mid_features, hidden_size=64, dropout=0.3, horizon=horizon
        ).to(self.device)

        self.mid_gru.load_state_dict(mid_state)
        self.mid_gru.eval()
        logger.info("Mid-frequency GRU loaded")

        # Load Sentiment GRU
        self.sent_gru = SentimentGRU(
            n_price=n_price_features,
            n_sent=n_sent_features,
            hidden=64,
            dropout=0.3,
            horizon=horizon,
        ).to(self.device)

        sent_gru_path = (
            self.artifacts_dir / "sentiment_gru_best_h5.pt"
            if self._is_h5_bundle
            else self.artifacts_dir / "sent_gru.pt"
        )
        self.sent_gru.load_state_dict(
            torch.load(sent_gru_path, map_location=self.device, weights_only=True)
        )
        self.sent_gru.eval()
        logger.info("Sentiment GRU loaded")

    @property
    def lookback(self) -> int:
        return int(self.config.get("lookback", self.config.get("LOOKBACK", 21)))

    @property
    def horizon(self) -> int:
        return int(self.config.get("horizon", self.config.get("HORIZON", 5)))

    @property
    def arima_order(self) -> tuple:
        value = self.config.get(
            "arima_order", self.config.get("ARIMA_ORDER", (1, 0, 1))
        )
        return tuple(value)

    @property
    def price_features(self) -> list:
        return self.config.get("sent_price_cols", self.config.get("price_features", []))

    @property
    def sentiment_features(self) -> list:
        return self.config.get(
            "sent_sent_cols", self.config.get("sentiment_features", [])
        )

    @property
    def mid_features(self) -> list:
        return self.config.get("mid_feat_cols", self.config.get("mid_features", []))

    @property
    def hf_features(self) -> list:
        return self.config.get("hf_feat_cols", self.config.get("hf_features", []))


# Global singleton instance
model_artifacts = ModelArtifacts()
