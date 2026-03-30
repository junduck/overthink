import math

import torch
from torch import nn

from overthink.block import TransBlock, TransStack
from overthink.layer import Linear, RoPE, SwiGLU
from overthink.layer.utils import get_torch_dtype

from .simple_config import SimpleConfig


class OverthinkSimple(nn.Module):
    """Simplified Overthink model: deep reasoning without FiLM or fixed lookback.

    Key differences from OverthinkModel:
    - No FiLM conditioning (can be added externally if needed)
    - No fixed lookback_horizon: accepts variable-length sequences like standard decoder-only models
    - Simpler forecast head: projects hidden state directly to next token
    - Maintains the hierarchical reasoning loop (high/low frequency reasoning)

    The reasoning loop iteratively refines hidden states through alternating:
    - High-frequency (local) reasoning via temporal_mix: refines local patterns
    - Low-frequency (global) reasoning via attention: aggregates global context

    Usage:
        # Single-step prediction
        next_token = model(input_tokens)  # [B, S, F] -> [B, 1, F]

        # Autoregressive generation
        for _ in range(horizon):
            next_token = model(current_seq[:, -1:, :])
            current_seq = torch.cat([current_seq, next_token], dim=1)
    """

    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.config = config
        dtype = get_torch_dtype(config.model_dtype)

        self.rope = RoPE(
            dim=config.hidden_size // config.head_num,
            max_seq_len=config.rope_max_seq_len,
            theta=config.rope_theta,
            dtype=dtype,
        )

        self.input_proj = Linear(
            in_features=config.feature_num,
            out_features=config.hidden_size,
            bias=True,
            dtype=dtype,
        )
        self.input_mixer = SwiGLU(
            hidden_size=config.hidden_size,
            expansion_factor=config.expansion_factor,
            dtype=dtype,
        )
        self.input_scale = 1.0 / math.sqrt(config.hidden_size)

        self.temporal_mix = TransStack(
            layer_num=config.hidden_layer_num,
            hidden_size=config.hidden_size,
            head_num=config.head_num,
            query_grp=config.query_group,
            dropout=config.attn_dropout,
            causal=config.use_causal,
            expansion_factor=config.expansion_factor,
            eps=config.rms_eps,
            rope=self.rope if config.use_rope else None,
            dtype=dtype,
        )

        self.attention = TransBlock(
            hidden_size=config.hidden_size,
            head_num=config.head_num,
            query_grp=config.query_group,
            dropout=config.attn_dropout,
            causal=config.use_causal,
            expansion_factor=config.expansion_factor,
            eps=config.rms_eps,
            rope=self.rope if config.use_rope else None,
            dtype=dtype,
        )

        self.forecast_head = Linear(
            in_features=config.hidden_size,
            out_features=config.feature_num,
            bias=True,
            dtype=dtype,
        )

        with torch.no_grad():
            self.forecast_head.w.mul_(0.01)
            if self.forecast_head.b is not None:
                self.forecast_head.b.zero_()

    def reason(self, state: torch.Tensor) -> torch.Tensor:
        """Iteratively refine hidden state through hierarchical reasoning.

        Args:
            state: Hidden state [B, S, D]

        Returns:
            Refined hidden state [B, S, D]
        """
        local = state.clone()
        globa = state.clone()
        with torch.no_grad():
            for _ in range(self.config.global_reason_step - 1):
                res = globa + state
                for _ in range(self.config.local_reason_step):
                    local = self.temporal_mix(local, res)
                globa = self.attention(globa + local)

        res = globa + state
        for _ in range(self.config.local_reason_step):
            state = self.temporal_mix(local, res)
        state = self.attention(globa + local)

        return state

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Forward pass: process input sequence and predict next values.

        Args:
            input_seq: Input tokens [B, S, feature_num]. Can be any sequence length.

        Returns:
            Predicted next values [B, 1, feature_num]
        """
        x = self.input_proj(input_seq)
        x = self.input_mixer(x)
        x = x * self.input_scale

        state = self.reason(x)

        delta = self.forecast_head(state[:, -1:, :])
        last_token = input_seq[:, -1:, :]
        output = last_token + 0.1 * delta

        return output

    def autoregressive_generate(
        self,
        input_seq: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        """Generate future values autoregressively.

        Args:
            input_seq: Initial sequence [B, S, feature_num]
            horizon: Number of steps to generate

        Returns:
            Full sequence including inputs and generated values [B, S + horizon, feature_num]
        """
        current_seq = input_seq

        for _ in range(horizon):
            next_val = self.forward(current_seq)
            current_seq = torch.cat([current_seq, next_val], dim=1)

        return current_seq
