from typing import Optional

import einops
import torch
from torch import nn

from .linear import Linear
from .rope import RoPE


class Attention(nn.Module):
    """Multi-head attention.

    Args:
        hidden_size: Input/output hidden dimension
        head_num: Number of attention heads
        head_dim: Dimension per attention head
        causal: Whether to apply causal masking (default: False)
        rope: Optional RoPE module to apply. Pass a shared RoPE instance
              across layers to avoid redundant cache creation (default: None)
        dtype: Data type for parameters ('float32', 'float16', 'bfloat16')
    """

    def __init__(
        self,
        hidden_size: int,
        head_num: int,
        head_dim: int,
        dropout: float = 0.0,
        causal: bool = False,
        rope: Optional[RoPE] = None,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()

        output_size = head_dim * head_num
        self.head_dim = head_dim
        self.head_num = head_num
        self.dropout = dropout
        self.causal = causal
        self.rope_module = rope

        self.qkv = Linear(in_features=hidden_size,
                          out_features=output_size * 3, bias=False, dtype=dtype)
        self.out = Linear(in_features=output_size,
                          out_features=hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = einops.rearrange(
            qkv, 'b s (three h d) -> three b s h d', three=3, h=self.head_num, d=self.head_dim)

        # Apply RoPE if enabled on q and k (shape: b s h d)
        if self.rope_module is not None:
            q, k = self.rope_module(q, k)

        # rearrange for multi-head attention
        q = einops.rearrange(q, 'b s h d -> b h s d')
        k = einops.rearrange(k, 'b s h d -> b h s d')
        v = einops.rearrange(v, 'b s h d -> b h s d')
        score = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout, is_causal=self.causal)
        return self.out(einops.rearrange(score, 'b h s d -> b s (h d)'))
