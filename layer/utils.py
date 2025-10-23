import math
import torch
from torch import nn


def trunc_normal(tensor: torch.Tensor, std: float = 1., lower: float = -2., upper: float = 2.) -> torch.Tensor:
    """Initialize a tensor with values drawn from a truncated normal distribution.
    Args:
        tensor (torch.Tensor): The tensor to be initialized.
        std (float, optional): Standard deviation of the normal distribution. Default is 1.
        lower (float, optional): Lower bound for truncation. Default is -2.
        upper (float, optional): Upper bound for truncation. Default is 2.
    Returns:
        torch.Tensor: The initialized tensor.
    """
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / \
                math.sqrt(1 - (upper * pdf_u - lower * pdf_l) /
                          z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor


def rms_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Functional RMSNorm without learnable parameters."""
    orig = x.dtype
    x = x.float()
    var = x.square().mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return x.to(orig)


class RMSNorm(nn.Module):
    """RMSNorm with learnable scale parameter (like LLaMA)."""

    def __init__(self, hidden_size: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.w = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.w * x).to(orig)


class RevIN(nn.Module):
    """Reversible Instance Normalization (RevIN) module."""

    def __init__(self, feature_num: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_w = nn.Parameter(torch.ones(feature_num))
            self.affine_b = nn.Parameter(torch.zeros(feature_num))

    def _denorm(self, x: torch.Tensor, m: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_b) / self.affine_w
        x = x * std + m
        return x

    def forward(self, x: torch.Tensor, denorm: bool = False) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True) + self.eps)
        if denorm:
            x = self._denorm(x, mean, std)
        else:
            x = (x - mean) / std
            if self.affine:
                x = x * self.affine_w + self.affine_b
        return x
