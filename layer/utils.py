import math
from typing import Literal

import torch


def get_torch_dtype(dtype_str: Literal['float32', 'float16', 'bfloat16']) -> torch.dtype:
    """Convert dtype string to torch dtype.

    Args:
        dtype_str: String representation of dtype

    Returns:
        torch.dtype corresponding to the string
    """
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    return dtype_map[dtype_str]


def trunc_normal(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0) -> torch.Tensor:
    """Initialize tensor with truncated normal distribution.
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
    m2 = x.square().mean(-1, keepdim=True)
    x = x * torch.rsqrt(m2 + eps)
    return x.to(orig)


def ema_weights(decay: float, length: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Generate EMA weights

    Args:
        decay: Decay rate for EMA (0 < decay < 1). Higher values give more weight to recent timesteps.
        length: Length of the sequence (number of timesteps).
        dtype: Data type for the weights (default: torch.float32).

    Returns:
        EMA weights tensor of shape [length] that sums to 1.

    Example:
        >>> weights = ema_weights(decay=0.1, length=5)
        >>> print(weights.shape)  # torch.Size([5])
        >>> print(weights.sum())  # tensor(1.)
    """
    if not 0 < decay < 1:
        raise ValueError(f"Decay must be between 0 and 1, got {decay}")
    if length <= 0:
        raise ValueError(f"Length must be positive, got {length}")

    decay_factors = torch.cumprod(
        torch.full((length,), 1.0 - decay, dtype=torch.float32),
        dim=0
    )
    # Reverse to give more weight to recent timesteps
    decay_factors = torch.flip(decay_factors, [0])
    # Normalize to sum to 1
    w = (decay_factors / decay_factors.sum()).view(-1).to(dtype=dtype)

    return w


def ema(x: torch.Tensor, dim: int, alpha: float) -> torch.Tensor:
    """Compute Exponential Moving Average (EMA) along a specified dimension."""
    m = torch.empty_like(x)
    m.select(dim, 0).copy_(x.select(dim, 0))
    for i in range(1, x.size(dim)):
        prev = m.select(dim, i - 1)
        cur = x.select(dim, i)
        m.select(dim, i).copy_(prev + alpha * (cur - prev))
    return m
