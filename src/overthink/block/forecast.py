import math
from typing import Literal

import torch
from torch import nn

from overthink.layer import Linear
from overthink.layer.utils import ema_weights, ema


class AutoregressiveHead(nn.Module):
    """Forecast head for autoregressive time series prediction using residual/delta learning.

    [B, S, hidden_size] -> [B, 1, feature_num]
    """

    def __init__(
        self,
        hidden_size: int,
        lookback_horizon: int,
        feature_num: int,
        aggregation: Literal["mean", "ema", "last"],
        ema_period: int = 0,
        delta_scale: float = 0.1,
        learnable_delta_scale: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.delta_proj = Linear(
            in_features=hidden_size, out_features=feature_num, bias=True, dtype=dtype
        )

        if learnable_delta_scale:
            # Initialize delta_scale as a learnable parameter
            # Use log space to ensure positivity and better gradient flow
            self.log_delta_scale = nn.Parameter(
                torch.log(torch.tensor(delta_scale, dtype=dtype))
            )
        else:
            self.delta_scale = delta_scale
        with torch.no_grad():
            # Scale based on lookback to account for autoregressive depth
            scale = 1.0 / math.sqrt(lookback_horizon)
            self.delta_proj.w.mul_(scale)
            self.delta_proj.b.zero_()  # type: ignore

        self.lookback = lookback_horizon
        self.ema_period = ema_period
        if aggregation == "ema":
            ema_w = ema_weights(
                period=ema_period, length=lookback_horizon, dtype=dtype
            ).view(1, -1, 1)
            self.register_buffer("ema_weights", ema_w, persistent=False)
        self.aggregation = aggregation

    def forward(self, x: torch.Tensor, last: torch.Tensor):
        """
        x: [B, S, hidden_size]
        last: [B, 1, feature_num]
        """
        # Aggregate hidden states
        if self.aggregation == "mean":
            agg = x.mean(dim=1)
        elif self.aggregation == "ema":
            # Check if S matches lookback
            if x.size(1) == self.lookback:
                agg = (x * self.ema_weights).sum(dim=1)  # type: ignore
            else:
                # we have different sequence length during inference, calculate ema in-place
                agg = ema(x, dim=1, period=self.ema_period)
        else:
            agg = x[:, -1, :]

        # Predict delta (change from last value)
        delta = self.delta_proj(agg).unsqueeze(1)  # [B, 1, feature_num]

        # Apply delta to last value with scaling
        if hasattr(self, "log_delta_scale"):
            # Use learnable parameter (exponentiate to ensure positivity)
            delta_scale = torch.exp(self.log_delta_scale)
        else:
            # Use fixed parameter
            delta_scale = self.delta_scale

        next_value = last + delta_scale * delta

        return next_value


class DirectForecastHead(nn.Module):
    """Forecast head for direct multi-step time series prediction.

    Used as a benchmark for AutoregressiveHead.

    [B, S, hidden_size] -> [B, forecast_horizon, feature_num]
    """

    def __init__(
        self,
        hidden_size: int,
        lookback_horizon: int,
        feature_num: int,
        forecast_horizon: int,
        aggregation: Literal["mean", "ema", "last"],
        ema_period: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.forecast_horizon = forecast_horizon
        self.feature_num = feature_num
        self.lookback = lookback_horizon
        self.ema_period = ema_period

        # Predict all future steps at once
        self.forecast_proj = Linear(
            in_features=hidden_size,
            out_features=forecast_horizon * feature_num,
            bias=True,
            dtype=dtype,
        )

        # Initialize with small weights for stability
        with torch.no_grad():
            scale = 1.0 / math.sqrt(lookback_horizon)
            self.forecast_proj.w.mul_(scale)
            self.forecast_proj.b.zero_()  # type: ignore

        if aggregation == "ema":
            ema_w = ema_weights(
                period=ema_period, length=lookback_horizon, dtype=dtype
            ).view(1, -1, 1)
            self.register_buffer("ema_weights", ema_w, persistent=False)
        self.aggregation = aggregation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, S, hidden_size] -> [B, forecast_horizon, feature_num]
        """
        # Aggregate hidden states
        if self.aggregation == "mean":
            agg = x.mean(dim=1)  # [B, hidden_size]
        elif self.aggregation == "ema":
            # Check if S matches lookback
            if x.size(1) == self.lookback:
                agg = (x * self.ema_weights).sum(dim=1)  # type: ignore
            else:
                # we have different sequence length during inference, calculate ema in-place
                agg = ema(x, dim=1, period=self.ema_period)
        else:  # 'last'
            agg = x[:, -1, :]  # [B, hidden_size]

        # Predict all future steps
        # [B, forecast_horizon * feature_num]
        forecast = self.forecast_proj(agg)

        # Reshape to [B, forecast_horizon, feature_num]
        forecast = forecast.view(-1, self.forecast_horizon, self.feature_num)

        return forecast
