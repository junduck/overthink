import math
from typing import Literal

import torch
from torch import nn

from .linear import Linear


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
                 dtype: torch.dtype = torch.float32):
        super().__init__()

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
            pos = torch.arange(lookback_horizon - 1, -1, -1).float()
            pos_w = torch.exp(-ema_decay * pos)
            ema_weights = (pos_w / pos_w.sum()).view(1, -1, 1)  # [1, S, 1]
            self.register_buffer("ema_weights", ema_weights, persistent=False)
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
        next_value = last_value + self.delta_scale * delta

        return next_value
