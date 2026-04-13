from typing import Optional

import torch
from torch import nn

from overthink.block.transformer import TransStack
from overthink.layer import RoPE


class ReasoningBlock(nn.Module):
    """Iterative reasoning block with local/global hierarchical mixing.

    Two scratch states (local, global) refined by a single shared TransStack:

        global_mixed = global + x
        local  = stack(local + global_mixed)
        global = stack(local + global)

    Both start at zero. The loop strengthens the signal through repeated
    application of the same stack. No persistent state needed.

    Args:
        hidden_size: Hidden dimension size
        stack_depth: Number of transformer layers in the stack
        local_steps: Number of local reasoning steps per global step
        global_steps: Number of global reasoning iterations
        head_num: Number of attention heads
        query_grp: Number of query groups for GQA (0 = standard MHA)
        dropout: Dropout rate for attention
        causal: Whether to use causal masking
        expansion_factor: MLP expansion factor
        eps: Epsilon for RMS normalization
        rope: Optional RoPE module
        dtype: Parameter data type
    """

    def __init__(
        self,
        hidden_size: int,
        stack_depth: int = 3,
        local_steps: int = 2,
        global_steps: int = 2,
        head_num: int = 4,
        query_grp: int = 0,
        dropout: float = 0.0,
        causal: bool = True,
        expansion_factor: float = 2.0,
        eps: float = 1e-5,
        rope: Optional[RoPE] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.local_steps = local_steps
        self.global_steps = global_steps

        self.stack = TransStack(
            layer_num=stack_depth,
            hidden_size=hidden_size,
            head_num=head_num,
            query_grp=query_grp,
            dropout=dropout,
            causal=causal,
            expansion_factor=expansion_factor,
            eps=eps,
            rope=rope,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run reasoning loop.

        Args:
            x: Input signal [B, S, H]

        Returns:
            Refined state [B, S, H]
        """
        local = torch.zeros_like(x)
        global_ = torch.zeros_like(x)

        with torch.no_grad():
            for _ in range(self.global_steps - 1):
                global_mixed = global_ + x
                for _ in range(self.local_steps):
                    local = self.stack(local + global_mixed)
                global_ = self.stack(local + global_)

        global_mixed = global_ + x
        for _ in range(self.local_steps):
            local = self.stack(local + global_mixed)
        global_ = self.stack(local + global_)

        return global_
