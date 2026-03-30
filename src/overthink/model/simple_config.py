from typing import Literal, Optional
from pydantic import BaseModel, Field


class SimpleConfig(BaseModel):
    """Configuration for OverthinkSimple model."""

    model_dtype: Literal["float32", "float16", "bfloat16"] = Field(
        default="float32",
        description="Data type for model parameters and computations",
    )

    # Input/Output
    feature_num: int = Field(description="Number of input/output features")

    # Model architecture
    hidden_size: int = Field(description="Hidden dimension size")
    head_num: int = Field(description="Number of attention heads")
    query_group: int = Field(
        default=0,
        description="Number of query groups for Grouped Query Attention (0 = standard MHA)",
    )
    hidden_layer_num: int = Field(description="Number of transformer layers")
    expansion_factor: float = Field(
        default=4.0,
        description="MLP expansion factor for intermediate dimension",
    )

    # Reasoning loop
    local_reason_step: int = Field(
        description="Number of local reasoning steps per global reasoning step"
    )
    global_reason_step: int = Field(description="Number of global reasoning steps")

    # Attention
    use_causal: bool = Field(
        default=True,
        description="Whether to use causal masking in attention",
    )
    use_rope: bool = Field(
        default=True,
        description="Whether to use Rotary Position Embeddings",
    )
    attn_dropout: float = Field(
        default=0.0,
        description="Dropout rate for attention weights",
    )
    rms_eps: float = Field(
        default=1e-5,
        description="Epsilon for RMS normalization",
    )

    # RoPE
    rope_theta: float = Field(
        default=10000.0,
        description="Base theta value for RoPE",
    )
    rope_max_seq_len: int = Field(
        default=2048,
        description="Maximum sequence length for RoPE",
    )
