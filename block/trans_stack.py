from typing import Optional

import torch
from torch import nn

from layer.rope import RoPE

from .trans_block import TransBlock


class TransStack(nn.Module):
    """A stack of transformer layers with residual connection.

    [B, S, Hidden] -> [B, S, Hidden]

    Args:
        layer_num: Number of transformer layers in the block
        hidden_size: Hidden dimension size
        head_num: Number of attention heads
        causal: Whether to use causal masking in attention
        expansion_factor: MLP expansion factor for intermediate dimension
        eps: Epsilon for RMS normalization
        rope: Optional RoPE module to use in attention
    """

    def __init__(self,
                 layer_num: int,
                 hidden_size: int,
                 head_num: int,
                 causal: bool,
                 expansion_factor: float,
                 eps: float,
                 rope: Optional[RoPE] = None,):
        super().__init__()
        self.layers = nn.ModuleList([
            TransBlock(
                hidden_size=hidden_size,
                head_num=head_num,
                causal=causal,
                expansion_factor=expansion_factor,
                eps=eps,
                rope=rope,
            ) for _ in range(layer_num)
        ])

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden = x + residual
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden
