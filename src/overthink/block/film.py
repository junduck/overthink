from typing import Tuple
import torch
from torch import nn

from overthink.layer import Linear


class FiLMBlock(nn.Module):
    def __init__(
        self,
        film_dim: int,
        film_hidden_size: int,
        model_hidden_size: int,
        film_dropout: float = 0.0,
    ):
        super().__init__()

        self.film_mlp = nn.Sequential(
            Linear(in_features=film_dim, out_features=film_hidden_size, bias=True),
            nn.GELU(),
            nn.Dropout(film_dropout),  # only dropout if too dense
            Linear(
                in_features=film_hidden_size,
                out_features=model_hidden_size * 2,
                bias=False,
            ),
        )

    def forward(self, film_cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [B, film_dim] -> [B, model_hidden_size] x2 (gamma, beta)
        """
        params = self.film_mlp(film_cond)
        gamma, beta = params.chunk(2, dim=-1)
        return gamma, beta
