import math
from typing import Literal

import torch
from torch import nn

from ..layer.linear import Linear
from ..layer.utils import ema_weights


class AutoregressiveForecastHead(nn.Module):
    """Forecast head for autoregressive time series prediction using residual/delta learning.

    Predicts the CHANGE (delta) from the last timestep rather than absolute values.
    This makes learning smoother and more stable in autoregressive loops.

    [B, S, hidden_size] -> [B, 1, feature_num] (delta prediction)

    Args:
        hidden_size: Input hidden dimension
        feature_num: Number of features to predict
        lookback_horizon: Length of lookback sequence
        aggregation: Method to aggregate sequence ('mean', 'ema', 'last')
        ema_decay: Decay rate for EMA aggregation
        delta_scale: Scaling factor for delta predictions (default: 0.1)
        dtype: Data type for parameters ('float32', 'float16', 'bfloat16')
    """

    def __init__(self,
                 hidden_size: int,
                 lookback_horizon: int,
                 feature_num: int,
                 aggregation: Literal['mean', 'ema', 'last'],
                 ema_decay: float,
                 delta_scale: float = 0.1,
                 learnable_delta_scale: bool = False,
                 dtype: torch.dtype = torch.float32):
        super().__init__()

        if learnable_delta_scale:
            # Initialize delta_scale as a learnable parameter
            # Use log space to ensure positivity and better gradient flow
            self.log_delta_scale = nn.Parameter(
                torch.log(torch.tensor(delta_scale, dtype=dtype)))
        else:
            self.delta_scale = delta_scale

        # Predict delta (change) rather than absolute value
        self.forecast_delta = Linear(in_features=hidden_size,
                                     out_features=feature_num, bias=True, dtype=dtype)

        # Initialize to output near-zero deltas for stability
        with torch.no_grad():
            # Scale based on lookback to account for autoregressive depth
            scale = 1.0 / math.sqrt(lookback_horizon)
            self.forecast_delta.w.mul_(scale)
            self.forecast_delta.b.zero_()  # type: ignore

        if aggregation == "ema":
            ema_w = ema_weights(
                decay=ema_decay, length=lookback_horizon, dtype=dtype).view(1, -1, 1)
            self.register_buffer("ema_weights", ema_w, persistent=False)
        self.aggregation = aggregation

    def forward(self, x: torch.Tensor, last_value: torch.Tensor) -> torch.Tensor:
        """Predict next value as: last_value + delta_scale * predicted_delta

        Args:
            x: Hidden states [B, S, hidden_size]
            last_value: Last known value [B, 1, feature_num]

        Returns:
            Next prediction [B, 1, feature_num]
        """
        # Aggregate hidden states
        if self.aggregation == 'mean':
            agg = x.mean(dim=1)
        elif self.aggregation == 'ema':
            # Ensure ema_weights is on the same device as input
            ema_w = self.ema_weights
            ema_weights = ema_w.to(device=x.device, dtype=x.dtype)
            agg = (x * ema_weights).sum(dim=1)  # type: ignore
        else:
            agg = x[:, -1, :]

        # Predict delta (change from last value)
        delta = self.forecast_delta(agg).unsqueeze(1)  # [B, 1, feature_num]

        # Apply delta to last value with scaling
        if hasattr(self, 'log_delta_scale'):
            # Use learnable parameter (exponentiate to ensure positivity)
            delta_scale = torch.exp(self.log_delta_scale)
        else:
            # Use fixed parameter
            delta_scale = self.delta_scale

        next_value = last_value + delta_scale * delta

        return next_value


class TransformerDecoderHead(nn.Module):
    def __init__(self, hidden_dim, forecast_horizon, num_layers=2):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, 1)
        self.pos_encoding = PositionalEncoding(hidden_dim)

    def forward(self, encoder_output):  # encoder_output: [B, S, H]
        # Create target sequence queries
        batch_size = encoder_output.size(0)
        tgt = torch.zeros(batch_size, self.forecast_horizon,
                          encoder_output.size(-1))
        tgt = self.pos_encoding(tgt.transpose(0, 1))  # [N, B, H]

        memory = encoder_output.transpose(0, 1)  # [S, B, H]

        # Autoregressive decoding with causal masking
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, tgt_mask=self.causal_mask)

        forecasts = self.output_proj(tgt.transpose(0, 1))  # [B, N, 1]
        return forecasts.squeeze(-1)  # [B, N]

    def causal_mask(self, size):
        return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)


class MultiHorizonHead(nn.Module):
    def __init__(self, hidden_dim, horizons=[1, 3, 7, 14, 28]):
        super().__init__()
        self.horizons = horizons
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, h)
            ) for h in horizons
        ])

    def forward(self, context):
        # Use different aggregation for different horizons
        context_pooled = context.mean(dim=1)  # Global context

        forecasts = []
        for i, head in enumerate(self.heads):
            if self.horizons[i] <= 5:  # Short-term: use recent context
                local_ctx = context[:, -5:, :].mean(dim=1)
                forecast = head(local_ctx)
            else:  # Long-term: use global context
                forecast = head(context_pooled)
            forecasts.append(forecast)

        return torch.cat(forecasts, dim=-1)  # [B, total_horizon]


class TimeSeriesHead(nn.Module):
    def __init__(self, hidden_dim, forecast_horizon):
        super().__init__()
        self.attention_pool = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.short_term_proj = nn.Linear(hidden_dim, min(7, forecast_horizon))
        self.long_term_proj = nn.Linear(hidden_dim, forecast_horizon)

    def forward(self, context):  # [B, S, H]
        # Adaptive pooling
        query = context.mean(dim=1, keepdim=True).transpose(0, 1)  # [1, B, H]
        context_t = context.transpose(0, 1)  # [S, B, H]
        pooled, _ = self.attention_pool(query, context_t, context_t)
        pooled = pooled.transpose(0, 1).squeeze(1)  # [B, H]

        # Multi-scale forecasting
        short_term = self.short_term_proj(context[:, -3:, :].mean(dim=1))
        long_term = self.long_term_proj(pooled)

        # Combine (handle horizon mismatch)
        return self.combine_forecasts(short_term, long_term)
