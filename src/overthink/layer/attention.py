from typing import Literal, Optional

import einops
import torch
from torch import nn
from torch.nn import functional as F

from .linear import Linear
from .rope import RoPE


class Attention(nn.Module):
    """Multi-head attention

    Args:
        hidden_size: Input hidden dimension
        head_num: Number of attention heads
        head_dim: Dimension per attention head
        dropout: Dropout probability (default: 0.0)
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
        score = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout, is_causal=self.causal)
        return self.out(einops.rearrange(score, 'b h s d -> b s (h d)'))


class LinearAttention(nn.Module):
    """Linear attention mechanism

    Args:
        hidden_size: Input hidden dimension
        head_num: Number of attention heads
        head_dim: Dimension per attention head
        dropout: Dropout probability (default: 0.0)
        eps: Epsilon for numerical stability (default: 1e-6)

        dtype: Data type for parameters ('float32', 'float16', 'bfloat16')
    """

    def __init__(
        self,
        hidden_size: int,
        head_num: int,
        head_dim: int,
        dropout: float = 0.0,
        eps: float = 1e-6,
        feature_map: Literal["elu", "relu"] = "elu",
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()

        output_size = head_dim * head_num
        self.head_dim = head_dim
        self.head_num = head_num
        self.dropout = dropout
        self.eps = eps
        self.feature_map = feature_map

        self.qkv = Linear(in_features=hidden_size,
                          out_features=output_size * 3, bias=False, dtype=dtype)
        self.out = Linear(in_features=output_size,
                          out_features=hidden_size, bias=False, dtype=dtype)

    def _do_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_map == "elu":
            return F.elu(x) + 1
        elif self.feature_map == "relu":
            return F.relu(x)
        else:
            raise ValueError(
                f"Unsupported feature map: {self.feature_map}. Supported: 'elu', 'relu'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = einops.rearrange(
            qkv, 'b s (three h d) -> three b s h d', three=3, h=self.head_num, d=self.head_dim)

        # Apply feature mapping (ELU+1) to q and k only
        q = self._do_feature_map(q)
        k = self._do_feature_map(k)

        # Rearrange for multi-head attention computation
        q = einops.rearrange(q, 'b s h d -> b h s d')
        k = einops.rearrange(k, 'b s h d -> b h s d')
        v = einops.rearrange(v, 'b s h d -> b h s d')

        kv = torch.einsum('bhsd,bhse->bhde', k, v)  # [B, H, D, D]
        k_sum = k.sum(dim=2, keepdim=True)  # [B, H, 1, D]

        # Compute linear attention: q @ kv / q @ k_sum
        qkv = torch.einsum('bhsd,bhde->bhse', q, kv)  # [B, H, S, D]
        qk_sum = torch.einsum('bhsd,bhsd->bhs', q, k_sum).clamp_(min=self.eps)

        attn_output = qkv / qk_sum.unsqueeze(-1)  # [B, H, S, D]

        # Apply dropout if needed
        if self.dropout > 0:
            attn_output = F.dropout(
                attn_output, p=self.dropout, training=self.training)

        # Rearrange back to original shape and project to output
        attn_output = einops.rearrange(attn_output, 'b h s d -> b s (h d)')
        return self.out(attn_output)


class GQAttention(nn.Module):
    """Grouped Query Attention

    Args:
        hidden_size: Input hidden dimension
        head_num: Number of attention heads (query heads)
        head_dim: Dimension per attention head
        ngrp: Number of groups (key/value heads). Must be <= head_num
        dropout: Dropout probability (default: 0.0)
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
        ngrp: int,
        dropout: float = 0.0,
        causal: bool = False,
        rope: Optional[RoPE] = None,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()

        if ngrp > head_num:
            raise ValueError(
                f"Number of groups ({ngrp}) cannot be greater than number of heads ({head_num})")
        if head_num % ngrp != 0:
            raise ValueError(
                f"Number of heads ({head_num}) must be divisible by number of groups ({ngrp})")

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_dim = head_dim
        self.ngrp = ngrp
        self.dropout = dropout
        self.causal = causal
        self.rope_module = rope

        # Query projection: [hidden_size -> head_num * head_dim]
        self.q_proj = Linear(in_features=hidden_size,
                             out_features=head_dim * head_num, bias=False, dtype=dtype)

        # Key and Value projections: [hidden_size -> ngrp * head_dim]
        self.k_proj = Linear(in_features=hidden_size,
                             out_features=head_dim * ngrp, bias=False, dtype=dtype)
        self.v_proj = Linear(in_features=hidden_size,
                             out_features=head_dim * ngrp, bias=False, dtype=dtype)

        # Output projection: [head_num * head_dim -> hidden_size]
        self.out = Linear(in_features=head_dim * head_num,
                          out_features=hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)  # [B, S, head_num * head_dim]
        k = self.k_proj(x)  # [B, S, ngrp * head_dim]
        v = self.v_proj(x)  # [B, S, ngrp * head_dim]

        # rearrange for multi-head attention
        q = einops.rearrange(q, 'b s (h h2) -> b s h h2',
                             h=self.head_num, h2=self.head_dim)
        k = einops.rearrange(k, 'b s (g h) -> b s g h',
                             g=self.ngrp, h=self.head_dim)
        v = einops.rearrange(v, 'b s (g h) -> b s g h',
                             g=self.ngrp, h=self.head_dim)

        # Apply RoPE if enabled on q and k
        if self.rope_module is not None:
            q, k = self.rope_module(q, k)

        # Rearrange for flash attention
        heads_per_group = self.head_num // self.ngrp
        q = einops.rearrange(q, 'b s h d -> b h s d')  # [B, head, S, head]
        k = einops.rearrange(k, 'b s h d -> b h s d')  # [B, ngrp, S, head_dim]
        v = einops.rearrange(v, 'b s h d -> b h s d')  # [B, ngrp, S, head_dim]
        k = einops.repeat(k, 'b g s d -> b (g h) s d', h=heads_per_group)
        v = einops.repeat(v, 'b g s d -> b (g h) s d', h=heads_per_group)

        score = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout, is_causal=self.causal)

        return self.out(einops.rearrange(score, 'b h s d -> b s (h d)'))
