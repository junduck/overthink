import math
import torch
from torch import nn

from overthink.layer import SwiGLU, Linear
from overthink.layer.utils import rms_norm


class FeatureMixBlock(nn.Module):
    """
    [B, S, F] -> [B, S, Hidden]
    """

    def __init__(
        self,
        feature_num: int,
        hidden_size: int,
        expansion_factor: float,
        eps: float,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.eps = eps
        self.proj = Linear(
            in_features=feature_num, out_features=hidden_size, bias=True, dtype=dtype
        )
        self.mix = SwiGLU(
            hidden_size=hidden_size, expansion_factor=expansion_factor, dtype=dtype
        )
        self.scale = 1.0 / math.sqrt(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x + self.mix(rms_norm(x, eps=self.eps))
        x = x * self.scale
        return x
