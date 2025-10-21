import torch
from torch import nn
from typing import Tuple


class RoPE(nn.Module):
    """Rotary Position Embedding (RoPE).

    Applies rotary position embeddings to query and key tensors.
    Based on "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    (https://arxiv.org/abs/2104.09864)

    Args:
        dim: Dimension of each attention head
        max_seq_len: Maximum sequence length (default: 8192)
        theta: Base for the exponential decay (default: 10000)
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequency tensor: shape (dim // 2,)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin cache for efficiency
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for the given sequence length."""
        # Create position indices: shape (seq_len,)
        inv_freq = self.inv_freq
        assert isinstance(inv_freq, torch.Tensor)
        t = torch.arange(seq_len, dtype=inv_freq.dtype)

        # Compute frequencies: shape (seq_len, dim // 2)
        freqs = torch.outer(t, inv_freq)

        # Concatenate to match full dimension: shape (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)

        # Cache cos and sin: shape (1, seq_len, 1, dim)
        self.register_buffer("cos_cached", emb.cos()[
                             None, :, None, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[
                             None, :, None, :], persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Rotated tensor of same shape
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embedding to query and key tensors.

        Args:
            q: Query tensor of shape (batch, seq_len, head_num, head_dim)
            k: Key tensor of shape (batch, seq_len, head_num, head_dim)

        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes as input
        """
        seq_len = q.shape[1]

        # Extend cache if sequence is longer than cached
        cos_cached = self.cos_cached
        sin_cached = self.sin_cached
        assert isinstance(cos_cached, torch.Tensor) and isinstance(
            sin_cached, torch.Tensor)

        if seq_len > cos_cached.shape[1]:
            self._build_cache(seq_len)
            cos_cached = self.cos_cached
            sin_cached = self.sin_cached
            assert isinstance(cos_cached, torch.Tensor) and isinstance(
                sin_cached, torch.Tensor)

        # Get cos/sin for current sequence length: shape (1, seq_len, 1, dim)
        cos = cos_cached[:, :seq_len, :, :]
        sin = sin_cached[:, :seq_len, :, :]

        # Apply rotation: x * cos + rotate_half(x) * sin
        q_embed = q * cos + self._rotate_half(q) * sin
        k_embed = k * cos + self._rotate_half(k) * sin

        return q_embed, k_embed
