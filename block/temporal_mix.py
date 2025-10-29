import torch
from torch import nn
from einops import rearrange

from layer import SwiGLU
from layer.utils import rms_norm


class TemporalMixBlock(nn.Module):
    """
    [B, S, Hidden] -> [B, S, Hidden]
    """

    def __init__(self,
                 time_horizon: int,
                 expansion_factor: float,
                 eps: float,
                 dtype: torch.dtype = torch.float32):
        super().__init__()

        self.eps = eps
        self.mix = SwiGLU(hidden_size=time_horizon,
                          expansion_factor=expansion_factor,
                          dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, S, Hidden] we need [B, Hidden, S] for temporal mixing
        x = rearrange(x, 'b s h -> b h s')
        x = x + self.mix(rms_norm(x, eps=self.eps))
        x = rearrange(x, 'b h s -> b s h')
        return x


class TemporalMixStack(nn.Module):
    """
    [B, S, Hidden] -> [B, S, Hidden]

    Args:
        layer_num: Number of TemporalMix layers in the block
        time_horizon: Time dimension size
        expansion_factor: MLP expansion factor for intermediate dimension
        eps: Epsilon for RMS normalization
        dtype: Data type for parameters ('float32', 'float16', 'bfloat16')
    """

    def __init__(self,
                 layer_num: int,
                 hidden_size: int,
                 time_horizon: int,
                 expansion_factor: float,
                 eps: float,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.layers = nn.ModuleList([
            TemporalMixBlock(
                time_horizon=time_horizon,
                expansion_factor=expansion_factor,
                eps=eps,
                dtype=dtype,
            ) for _ in range(layer_num)
        ])
        self.eps = eps
        self.mlp = SwiGLU(hidden_size=hidden_size,
                          expansion_factor=expansion_factor,
                          dtype=dtype)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden = x + residual
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = hidden + self.mlp(rms_norm(hidden, eps=self.eps))
        return hidden
