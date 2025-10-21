import torch
from torch import nn
from .utils import trunc_normal


class Embed(nn.Module):
    """Embedding layer with truncated normal initialization.

    Args:
        embed_num: Vocabulary size (number of embeddings)
        embed_dim: Embedding dimension
        std: Standard deviation for truncated normal initialization
    """

    def __init__(self, embed_num: int, embed_dim: int, std: float):
        super().__init__()
        self.w = nn.Parameter(trunc_normal(
            torch.empty(embed_num, embed_dim), std=std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(x, self.w.to(x.dtype))
