import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm with learnable scale parameter (like LLaMA)."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.eps = eps
        self.w = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m2 = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(m2 + self.eps)
        return self.w * x


class RevIN(nn.Module):
    """Reversible Instance Normalization (RevIN) module.

    RevIN normalizes instances (samples) independently and can reverse the normalization
    during inference. This is particularly useful for time series forecasting.
    """

    def __init__(self, feature_num: int, affine: bool, eps: float, dtype: torch.dtype):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.feature_num = feature_num

        # Store statistics for denormalization
        self.register_buffer('stored_mean', torch.zeros(
            1, 1, feature_num, dtype=dtype))
        self.register_buffer('stored_std', torch.ones(
            1, 1, feature_num, dtype=dtype))
        self._stats_initialized = False

        if affine:
            self.affine_w = nn.Parameter(torch.ones(feature_num, dtype=dtype))
            self.affine_b = nn.Parameter(torch.zeros(feature_num, dtype=dtype))

    def _denorm(self, x: torch.Tensor, m: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Denormalize using stored statistics."""
        x = x * std + m
        if self.affine:
            affine_w = self.affine_w.view(1, 1, -1)
            affine_b = self.affine_b.view(1, 1, -1)
            x = (x - affine_b) / affine_w
        return x

    def forward(self, x: torch.Tensor, denorm: bool = False) -> torch.Tensor:
        """
        Forward pass for RevIN.

        Args:
            x: Input tensor [B, S, feature_num] or [B, feature_num]
            denorm: If True, perform denormalization, otherwise normalize

        Returns:
            Normalized or denormalized tensor
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, feature_num]
            squeeze_output = True
        else:
            squeeze_output = False

        if denorm:
            # Use stored statistics for denormalization
            if self._stats_initialized:
                m = getattr(self, 'stored_mean')
                std = getattr(self, 'stored_std')
                x = self._denorm(x, m, std)
            else:
                raise ValueError("No stored statistics available for denormalization. "
                                 "Apply normalization first.")
        else:
            # Normalize and store statistics
            m = x.mean(dim=1, keepdim=True)
            std = torch.sqrt(
                x.var(dim=1, keepdim=True, unbiased=False) + self.eps)

            # Store statistics for later denormalization
            setattr(self, 'stored_mean', m.detach())
            setattr(self, 'stored_std', std.detach())
            self._stats_initialized = True

            x = (x - m) / std
            if self.affine:
                affine_w = self.affine_w.view(1, 1, -1)
                affine_b = self.affine_b.view(1, 1, -1)
                x = x * affine_w + affine_b

        # Restore original shape if needed
        if squeeze_output:
            x = x.squeeze(1)

        return x
