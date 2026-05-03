"""
PyTorch model architectures - MUST match training code exactly.
"""

import torch
import torch.nn as nn


class MidFreqGRU(nn.Module):
    """
    Mid-frequency GRU model for price pattern prediction.
    Uses price-derived features only.

    Architecture (from training):
    - GRU: input_size=n_features, hidden_size=64, num_layers=1
    - Dropout: 0.3
    - Linear: 64 -> HORIZON (5)
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        dropout: float = 0.3,
        horizon: int = 5,
    ):
        super().__init__()
        self.horizon = horizon

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            (batch, horizon) - predictions for each horizon
        """
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Take last timestep
        out = self.dropout(out)
        return self.fc(out)


class Attention(nn.Module):
    """
    Attention mechanism for sentiment stream in SentimentGRU.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            (batch, hidden_dim) - attention-weighted sum
        """
        weights = torch.softmax(self.attn(x), dim=1)
        return (weights * x).sum(dim=1)


class SentimentGRU(nn.Module):
    """
    Dual-stream GRU with attention for sentiment-aware prediction.

    Architecture (from training):
    - Price stream: GRU(n_price, 64)
    - Sentiment stream: GRU(n_sent, 64) + Attention
    - Fusion: Concatenate + Dropout + Linear(128 -> HORIZON)
    """

    def __init__(
        self,
        n_price: int,
        n_sent: int,
        hidden: int = 64,
        dropout: float = 0.3,
        horizon: int = 5,
    ):
        super().__init__()
        self.horizon = horizon

        # Dual GRU streams
        self.price_gru = nn.GRU(n_price, hidden, batch_first=True)
        self.sent_gru = nn.GRU(n_sent, hidden, batch_first=True)

        # Attention for sentiment
        self.attn = Attention(hidden)

        # Output layer
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, horizon))

    def forward(self, xp: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xp: (batch, seq_len, n_price) - price features
            xs: (batch, seq_len, n_sent) - sentiment features
        Returns:
            (batch, horizon) - predictions for each horizon
        """
        # Price stream - use final hidden state
        _, hp = self.price_gru(xp)

        # Sentiment stream - use attention over all timesteps
        hs, _ = self.sent_gru(xs)
        hs_att = self.attn(hs)

        # Concatenate and predict
        h = torch.cat([hp.squeeze(0), hs_att], dim=1)
        return self.fc(h)
