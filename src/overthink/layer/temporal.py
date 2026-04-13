import math

import torch
from torch import nn

from overthink.layer.linear import Linear


class TemporalEmbedding(nn.Module):
    """Learnable temporal embeddings for financial time series.

    Extracts 5 calendar features and maps each to a dense vector,
    then sums them and projects to hidden_size. Following Kronos's design.

    Args:
        hidden_size: Output dimension
        embed_dim: Per-feature embedding dimension (before projection)
        dtype: Parameter data type
    """

    def __init__(
        self,
        hidden_size: int,
        embed_dim: int = 64,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.minute_embed = nn.Embedding(1440, embed_dim, dtype=dtype)
        self.hour_embed = nn.Embedding(24, embed_dim, dtype=dtype)
        self.dow_embed = nn.Embedding(7, embed_dim, dtype=dtype)
        self.dom_embed = nn.Embedding(31, embed_dim, dtype=dtype)
        self.month_embed = nn.Embedding(12, embed_dim, dtype=dtype)

        self.proj = Linear(embed_dim, hidden_size, bias=False, dtype=dtype)
        self.scale = math.sqrt(embed_dim)

        for emb in [
            self.minute_embed,
            self.hour_embed,
            self.dow_embed,
            self.dom_embed,
            self.month_embed,
        ]:
            nn.init.normal_(emb.weight, mean=0, std=embed_dim**-0.5)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Compute temporal embeddings.

        Args:
            timestamps: [B, S, 5] long tensor with columns:
                minute_of_day (0-1439), hour_of_day (0-23),
                day_of_week (0-6), day_of_month (0-30), month (0-11)

        Returns:
            [B, S, hidden_size] temporal embeddings
        """
        e = (
            self.minute_embed(timestamps[..., 0])
            + self.hour_embed(timestamps[..., 1])
            + self.dow_embed(timestamps[..., 2])
            + self.dom_embed(timestamps[..., 3])
            + self.month_embed(timestamps[..., 4])
        )
        return self.proj(e * self.scale)
