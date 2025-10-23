import torch
from torch import nn
from .utils import trunc_normal


class Embed(nn.Module):
    """Embedding layer with truncated normal initialization.

    Args:
        embed_num: Vocabulary size (number of embeddings)
        embed_dim: Embedding dimension
        std: Standard deviation for truncated normal initialization
        dtype: Data type for parameters ('float32', 'float16', 'bfloat16')
    """

    def __init__(
        self,
        embed_num: int,
        embed_dim: int,
        std: float,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()

        self.w = nn.Parameter(trunc_normal(
            torch.empty(embed_num, embed_dim, dtype=dtype), std=std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure weights are on the same device and dtype as input
        w = self.w.to(device=x.device, dtype=x.dtype)
        return torch.nn.functional.embedding(x, w)
