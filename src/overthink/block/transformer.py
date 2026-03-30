from typing import Optional

import torch
from torch import nn

from overthink.layer import Attention, GQAttention, RoPE, SwiGLU
from overthink.layer.utils import rms_norm


class TransBlock(nn.Module):
    """Standard transformer: Self-attention + MLP pre-norm

    [B, S, Hidden] -> [B, S, Hidden]

    Args:
        hidden_size: Hidden dimension size
        head_num: Number of attention heads
        query_grp: Number of grouped query
        causal: Whether to use causal masking in attention
        expansion_factor: MLP expansion factor for intermediate dimension
        eps: Epsilon for RMS normalization
        rope: Optional RoPE module to use in attention
        dtype: Data type for parameters ('float32', 'float16', 'bfloat16')
    """

    def __init__(
        self,
        hidden_size: int,
        head_num: int,
        query_grp: int,
        dropout: float,
        causal: bool,
        expansion_factor: float,
        eps: float,
        rope: Optional[RoPE] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.eps = eps

        head_dim = hidden_size // head_num
        if query_grp == 0:
            self.self_attn = Attention(
                hidden_size=hidden_size,
                head_dim=head_dim,
                head_num=head_num,
                dropout=dropout,
                causal=causal,
                rope=rope,
                dtype=dtype,
            )
        else:
            self.self_attn = GQAttention(
                hidden_size=hidden_size,
                head_dim=head_dim,
                head_num=head_num,
                ngrp=query_grp,
                dropout=dropout,
                causal=causal,
                rope=rope,
                dtype=dtype,
            )

        self.mlp = SwiGLU(
            hidden_size=hidden_size, expansion_factor=expansion_factor, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(rms_norm(x, eps=self.eps))
        x = x + self.mlp(rms_norm(x, eps=self.eps))
        return x


class TransStack(nn.Module):
    """A stack of transformer layers with residual connection.

    [B, S, Hidden] -> [B, S, Hidden]

    Args:
        layer_num: Number of transformer layers in the block
        hidden_size: Hidden dimension size
        head_num: Number of attention heads
        query_grp: Number of grouped query
        causal: Whether to use causal masking in attention
        expansion_factor: MLP expansion factor for intermediate dimension
        eps: Epsilon for RMS normalization
        rope: Optional RoPE module to use in attention
        dtype: Data type for parameters ('float32', 'float16', 'bfloat16')
    """

    def __init__(
        self,
        layer_num: int,
        hidden_size: int,
        head_num: int,
        query_grp: int,
        dropout: float,
        causal: bool,
        expansion_factor: float,
        eps: float,
        rope: Optional[RoPE] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransBlock(
                    hidden_size=hidden_size,
                    head_num=head_num,
                    query_grp=query_grp,
                    dropout=dropout,
                    causal=causal,
                    expansion_factor=expansion_factor,
                    eps=eps,
                    rope=rope,
                    dtype=dtype,
                )
                for _ in range(layer_num)
            ]
        )

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden = x + residual
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden
