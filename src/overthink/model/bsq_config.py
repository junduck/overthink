from typing import Literal

from pydantic import BaseModel, Field


class BSQConfig(BaseModel):
    """Configuration for OverthinkBSQ model.

    Operates on discrete (s1, s2) token pairs from Kronos BSQ tokenizer.
    Prediction is classification over vocab_size (like an LLM), not
    continuous regression.
    """

    model_dtype: Literal["float32", "float16", "bfloat16"] = Field(
        default="float32",
        description="Data type for model parameters and computations",
    )

    vocab_size: int = Field(
        default=1024,
        description="Subtoken vocabulary size (2^bits). 1024 for 10-bit BSQ.",
    )

    hidden_size: int = Field(default=128, description="Hidden dimension size")
    head_num: int = Field(default=4, description="Number of attention heads")
    query_group: int = Field(
        default=0,
        description="Query groups for GQA (0 = standard MHA)",
    )
    stack_depth: int = Field(
        default=3,
        description="Number of transformer layers in reasoning stack",
    )
    expansion_factor: float = Field(
        default=2.0,
        description="MLP expansion factor (SwiGLU intermediate dim = hidden_size * this)",
    )

    local_steps: int = Field(
        default=2,
        description="Number of local reasoning steps per global step",
    )
    global_steps: int = Field(
        default=2,
        description="Number of global reasoning iterations",
    )

    use_causal: bool = Field(
        default=True,
        description="Causal masking in attention",
    )
    use_rope: bool = Field(
        default=True,
        description="Rotary Position Embeddings",
    )
    use_temporal: bool = Field(
        default=True,
        description="Add temporal embeddings (minute, hour, dow, dom, month)",
    )
    temporal_embed_dim: int = Field(
        default=64,
        description="Per-feature embedding dimension for temporal features",
    )
    attn_dropout: float = Field(
        default=0.0,
        description="Dropout rate for attention weights",
    )
    rms_eps: float = Field(
        default=1e-5,
        description="Epsilon for RMS normalization",
    )

    rope_theta: float = Field(
        default=10000.0,
        description="Base theta value for RoPE",
    )
    rope_max_seq_len: int = Field(
        default=2048,
        description="Maximum sequence length for RoPE",
    )


class BSQDataConfig(BaseModel):
    """Data pipeline configuration."""

    data_dir: str = Field(description="Root directory with {code}/{date}.pq layout")
    tokenized_dir: str | None = Field(
        default=None,
        description="Pre-tokenized data directory. If set, training uses BSQPreTokenizedDataset.",
    )
    tokenizer: str = Field(
        default="NeoQuasar/Kronos-Tokenizer-base",
        description="HuggingFace model ID or local path for Kronos tokenizer",
    )
    min_bars: int = Field(default=30, description="Minimum bars per day to include")
    val_cutoff: str | None = Field(
        default=None,
        description="Date (YYYY-MM-DD) for train/val split. Dates >= are val.",
    )


class BSQTrainConfig(BaseModel):
    """Training hyperparameters."""

    batch_size: int = Field(default=32)
    lr: float = Field(default=3e-4)
    weight_decay: float = Field(default=0.01)
    epochs: int = Field(default=10)
    grad_clip: float = Field(default=1.0, description="Max gradient norm")
    num_workers: int = Field(default=0)
    val_max_batches: int = Field(
        default=200,
        description="Max batches for val evaluation (None = full)",
    )
    ckpt_path: str = Field(default="checkpoints/bsq")
    dry_run: bool = Field(default=False)


class BSQRunConfig(BaseModel):
    """Top-level config for a training run. Matches JSON layout:

    {
        "data": { ... },
        "train": { ... },
        "model": { ... }
    }
    """

    data: BSQDataConfig
    train: BSQTrainConfig = Field(default_factory=BSQTrainConfig)
    model: BSQConfig = Field(default_factory=BSQConfig)
