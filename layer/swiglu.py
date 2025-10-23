import torch
from torch import nn

from .linear import Linear


class SwiGLU(nn.Module):
    """SwiGLU activation function for MLPs.

    Applies gated linear unit with SiLU activation: SiLU(gate) * up, then projects down.

    Args:
        hidden_size: Input/output hidden dimension
        expansion_factor: Factor to expand hidden size for intermediate layer (default: 4.0)
        dtype: Data type for parameters ('float32', 'float16', 'bfloat16')
    """

    def __init__(
        self,
        hidden_size: int,
        expansion_factor: float = 4.0,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        I = (int((2. / 3) * expansion_factor * hidden_size) + 255) // 256 * 256
        # [B, S, I * 2]
        self.gate_up = Linear(in_features=hidden_size,
                              out_features=I * 2, bias=False, dtype=dtype)
        # [B, S, hidden_size]
        self.down = Linear(in_features=I,
                           out_features=hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(torch.nn.functional.silu(gate) * up)
