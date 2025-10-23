import math

import torch
from torch import nn

from .utils import trunc_normal


class Linear(nn.Module):
    """Linear transformation with truncated normal initialization.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        bias: Whether to include bias term (default: True)
        dtype: Data type for parameters ('float32', 'float16', 'bfloat16')
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()

        self.w = nn.Parameter(trunc_normal(
            torch.empty(out_features, in_features, dtype=dtype),
            std=1./math.sqrt(in_features)))
        self.b = nn.Parameter(torch.zeros(
            out_features, dtype=dtype)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure weights are on the same device and dtype as input
        w = self.w.to(device=x.device, dtype=x.dtype)
        b = self.b.to(device=x.device,
                      dtype=x.dtype) if self.b is not None else None
        return torch.nn.functional.linear(x, w, b)
