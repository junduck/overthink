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


def ema_weights(period: int, length: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Generate EMA weights

    Args:
        period: EMA period (e.g., 10 for 10-period EMA). Higher values = more smoothing.
        length: Length of the sequence (number of timesteps).
        dtype: Data type for the weights (default: torch.float32).

    Returns:
        EMA weights tensor of shape [length] that sums to 1.
        More recent timesteps have higher weights.

    Example:
        >>> weights = ema_weights(period=10, length=5)
        >>> print(weights.shape)  # torch.Size([5])
        >>> print(weights.sum())  # tensor(1.)
    """
    if period <= 0:
        raise ValueError(f"Period must be positive, got {period}")
    if length <= 0:
        raise ValueError(f"Length must be positive, got {length}")

    # Financial convention: alpha = 2 / (period + 1)
    alpha = 2.0 / (period + 1)

    # Generate decay factors: (1-alpha)^0, (1-alpha)^1, (1-alpha)^2, ...
    indices = torch.arange(length, dtype=torch.float32)
    decay_factors = (1.0 - alpha) ** indices

    # Reverse to give more weight to recent timesteps
    decay_factors = torch.flip(decay_factors, [0])

    # Normalize to sum to 1
    w = (decay_factors / decay_factors.sum()).to(dtype=dtype)

    return w


def ema(x: torch.Tensor, dim: int, period: int) -> torch.Tensor:
    """Compute final EMA value along a specified dimension.

    Args:
        x: Input tensor
        dim: Dimension along which to compute EMA
        period: EMA period (e.g., 10 for 10-period EMA). Higher values = more smoothing.
                Uses financial convention: alpha = 2 / (period + 1) ~ responsiveness SMA(period)

    Returns:
        Final EMA value with dimension `dim` having size 1.

    Example:
        >>> x = torch.randn(2, 100, 512)  # [B, S, Hidden]
        >>> result = ema(x, dim=1, period=10)  # [B, 1, Hidden]
    """
    if period <= 0:
        raise ValueError(f"Period must be positive, got {period}")

    # Financial convention: alpha = 2 / (period + 1)
    alpha = 2.0 / (period + 1)

    result = x.select(dim, 0).clone()
    for i in range(1, x.size(dim)):
        cur = x.select(dim, i)
        # EMA[t] = alpha * x[t] + (1 - alpha) * EMA[t-1]
        result.add_(cur - result, alpha=alpha)

    return result.unsqueeze(dim)
