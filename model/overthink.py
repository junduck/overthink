import math
from typing import Tuple

import torch
from torch import nn
from einops import repeat

from block.trans_block import TransBlock
from block.trans_stack import TransStack
from layer.forecast import AutoregressiveForecastHead
from layer.linear import Linear
from layer.rope import RoPE
from layer.swiglu import SwiGLU

from .model_config import ModelConfig


class OverthinkModel(nn.Module):
    """Overthink Model for time series forecasting: fast, overfitting using hierarchical reasoning.

    Traders say it's a hunch. They are just overfitting.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.rope = RoPE(dim=config.hidden_size // config.head_num,
                         max_seq_len=config.rope_max_seq_len,
                         theta=config.rope_theta)

        # Use SwiGLU for richer feature mixing
        self.input_proj_linear = Linear(
            in_features=config.feature_num,
            out_features=config.hidden_size,
            bias=True
        )
        self.input_feat_mixing = SwiGLU(
            hidden_size=config.hidden_size,
            expansion_factor=config.expansion_factor
        )
        self.input_scale = 1. / math.sqrt(config.hidden_size)

        self.high_freq_reasoning = TransStack(
            layer_num=config.hidden_layer_num,
            hidden_size=config.hidden_size,
            head_num=config.head_num,
            causal=config.use_causal,
            expansion_factor=config.expansion_factor,
            eps=config.rms_eps,
            rope=self.rope if config.use_rope else None,
        )

        self.low_freq_reasoning = TransBlock(
            hidden_size=config.hidden_size,
            head_num=config.head_num,
            causal=False,  # non-causal for bi-dir aggr
            expansion_factor=config.expansion_factor,
            eps=config.rms_eps,
            rope=self.rope if config.use_rope else None,
        )

        self.forecast_head = AutoregressiveForecastHead(
            hidden_size=config.hidden_size,
            lookback_horizon=config.lookback_horizon,
            feature_num=config.feature_num,
            aggregation=config.forecast_aggregation,
            ema_decay=config.forecast_ema_decay,
            delta_scale=config.forecast_residual_scale,
        )

        self.high_freq_init = nn.Parameter(
            torch.zeros(config.hidden_size))
        self.low_freq_init = nn.Parameter(
            torch.zeros(config.hidden_size))

        # TODO: support ACT
        if False:
            self.q_head = Linear(in_features=config.hidden_size,
                                 out_features=1, bias=True)
            with torch.no_grad():
                self.q_head.w.zero_()
                if self.q_head.b is not None:
                    self.q_head.b.fill_(-10.0)

    def input_projection(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Project input sequence to hidden dimension with SwiGLU for feature mixing.

        Args:
            input_seq: Input time series [B, S, feature_num]

        Returns:
            Projected tensor [B, S, hidden_size]
        """
        # Project from feature_num to hidden_size
        x = self.input_proj_linear(input_seq)  # [B, S, hidden_size]
        # Apply SwiGLU for non-linear feature mixing
        x = self.input_feat_mixing(x)  # [B, S, hidden_size]
        # Normalize by sqrt(hidden_size) to prevent explosion
        x = x * self.input_scale
        return x

    def reasoning(self,
                  high_freq: torch.Tensor,
                  low_freq: torch.Tensor,
                  residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one reasoning step with high and low frequency inputs.

        Args:
            high_freq: High frequency reasoning state [B, S, D]
            low_freq: Low frequency reasoning state [B, S, D]
            residual: Residual connection tensor [B, S, D]

        Returns:
            Updated high frequency and low frequency tensors.
        """
        with torch.no_grad():
            for _ in range(self.config.low_freq_step - 1):
                res_h = low_freq + residual
                for _ in range(self.config.high_freq_step):
                    high_freq = self.high_freq_reasoning(high_freq, res_h)
                low_freq = self.low_freq_reasoning(low_freq + high_freq)

        res_h = low_freq + residual
        for _ in range(self.config.high_freq_step):
            high_freq = self.high_freq_reasoning(high_freq, res_h)
        low_freq = self.low_freq_reasoning(low_freq + high_freq)

        # TODO: support ACT, conditionally .detach()
        return high_freq, low_freq

    def forward(self,
                input_seq: torch.Tensor,
                ) -> torch.Tensor:
        """Autoregressive forward pass for multi-step forecasting with residual predictions.

        Args:
            input_seq: Historical time series [B, lookback_horizon, feature_num]

        Returns:
            Forecasted time series [B, forecast_horizon, feature_num]
        """
        batch_size = input_seq.size(0)
        seq_len = self.config.lookback_horizon

        # Store all predictions
        predictions = []

        # Current input sequence for the rolling window
        current_seq = input_seq  # [B, S, feature_num]

        # Autoregressive loop: generate forecast_horizon steps
        for _ in range(self.config.forecast_horizon):
            # Project current sequence
            input_proj = self.input_projection(current_seq)  # [B, S, D]

            # Initialize reasoning states
            hf_state = repeat(self.high_freq_init, 'd -> b s d',
                              b=batch_size, s=seq_len)
            lf_state = repeat(self.low_freq_init, 'd -> b s d',
                              b=batch_size, s=seq_len)

            # Perform reasoning
            hf_state, lf_state = self.reasoning(
                high_freq=hf_state,
                low_freq=lf_state,
                residual=input_proj
            )

            # Generate next step prediction using residual/delta learning
            # Predict: next_value = last_value + delta_scale * predicted_delta
            last_value = current_seq[:, -1:, :]  # [B, 1, feature_num]
            next_pred = self.forecast_head(
                lf_state, last_value)  # [B, 1, feature_num]
            predictions.append(next_pred)

            # Update sequence for next iteration: slide window
            # Remove oldest timestep and append new prediction
            current_seq = torch.cat([current_seq[:, 1:, :], next_pred], dim=1)

        # Stack all predictions: [B, forecast_horizon, feature_num]
        forecast = torch.cat(predictions, dim=1)
        return forecast
