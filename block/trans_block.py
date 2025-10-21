from typing import Optional

import torch
from torch import nn

from layer.attention import Attention
from layer.rope import RoPE
from layer.swiglu import SwiGLU
from layer.utils import rms_norm


class TransBlock(nn.Module):
    """Standard transformer: Self-attention + MLP pre-norm

    [B, S, Hidden] -> [B, S, Hidden]

    Args:
        hidden_size: Hidden dimension size
        head_num: Number of attention heads
        causal: Whether to use causal masking in attention
        expansion_factor: MLP expansion factor for intermediate dimension
        eps: Epsilon for RMS normalization
        rope: Optional RoPE module to use in attention
    """

    def __init__(self,
                 hidden_size: int,
                 head_num: int,
                 causal: bool,
                 expansion_factor: float,
                 eps: float,
                 rope: Optional[RoPE] = None,):
        super().__init__()

        self.eps = eps

        head_dim = hidden_size // head_num
        self.self_attn = Attention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            head_num=head_num,
            causal=causal,
            rope=rope
        )
        self.mlp = SwiGLU(hidden_size=hidden_size,
                          expansion_factor=expansion_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(rms_norm(x, eps=self.eps))
        x = x + self.mlp(rms_norm(x, eps=self.eps))
        return x
