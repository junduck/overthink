from typing import Literal

from pydantic import BaseModel, Field


class SIGRegTokenizerConfig(BaseModel):
    """Configuration for SIGRegTokenizer.

    Replaces the 5 BSQ hyperparameters (beta, gamma0, gamma, zeta, group_size)
    with a single SIGReg weight lambda. No per-window z-score normalization
    is needed — SIGReg forces the encoder's latent space to N(0, I), making
    sign() quantization produce uniform codes by construction.
    """

    model_dtype: Literal["float32", "float16", "bfloat16"] = Field(
        default="float32",
        description="Data type for model parameters and computations",
    )

    d_in: int = Field(default=6, description="Input feature dimension (OHLCVA)")
    d_model: int = Field(default=256, description="Model hidden dimension")
    n_heads: int = Field(default=4, description="Number of attention heads")
    ff_dim: int = Field(default=512, description="Feed-forward dimension")
    n_enc_layers: int = Field(default=3, description="Number of encoder transformer layers")
    n_dec_layers: int = Field(default=3, description="Number of decoder transformer layers")
    ffn_dropout_p: float = Field(default=0.0, description="FFN dropout probability")
    attn_dropout_p: float = Field(default=0.0, description="Attention dropout probability")
    resid_dropout_p: float = Field(default=0.0, description="Residual dropout probability")

    encoder_window: int = Field(
        default=64,
        description="Sliding window size for causal encoder. Each bar sees the previous W-1 bars.",
    )

    s1_bits: int = Field(default=10, description="Bits for coarse subtoken (vocab = 2^s1_bits)")
    s2_bits: int = Field(default=10, description="Bits for fine subtoken (vocab = 2^s2_bits)")

    sigreg_lambda: float = Field(
        default=0.05,
        description="Weight for SIGReg loss (replaces beta, gamma0, gamma, zeta)",
    )
    sigreg_num_slices: int = Field(
        default=256,
        description="Number of random 1D projections for SIGReg",
    )
    sigreg_t_max: float = Field(
        default=5.0,
        description="Integration range for Epps-Pulley test",
    )
    sigreg_n_points: int = Field(
        default=17,
        description="Number of quadrature points for Epps-Pulley test",
    )

    @property
    def codebook_dim(self) -> int:
        return self.s1_bits + self.s2_bits

    @property
    def vocab_size(self) -> int:
        return 2**self.s1_bits


class SIGRegTokenizerTrainConfig(BaseModel):
    """Training hyperparameters for SIGRegTokenizer."""

    batch_size: int = Field(default=32)
    lr: float = Field(default=2e-4)
    weight_decay: float = Field(default=0.1)
    betas: tuple[float, float] = Field(default=(0.9, 0.95))
    epochs: int = Field(default=30)
    grad_clip: float = Field(default=2.0, description="Max gradient norm")
    warmup_pct: float = Field(default=0.03, description="Fraction of steps for LR warmup")
    num_workers: int = Field(default=0)
    ckpt_path: str = Field(default="checkpoints/sigreg_tokenizer")
    dry_run: bool = Field(default=False)
    max_seq_len: int = Field(default=512, description="Maximum sequence length for sliding window")


class SIGRegTokenizerRunConfig(BaseModel):
    """Top-level config for a tokenizer training run."""

    data: "SIGRegTokenizerDataConfig"
    train: SIGRegTokenizerTrainConfig = Field(default_factory=SIGRegTokenizerTrainConfig)
    model: SIGRegTokenizerConfig = Field(default_factory=SIGRegTokenizerConfig)


class SIGRegTokenizerDataConfig(BaseModel):
    """Data pipeline configuration for SIGReg tokenizer training."""

    sqlite_dir: str | None = Field(
        default=None,
        description="Directory with per-code SQLite DBs ({code}.db)",
    )
    window_size: int = Field(default=200, description="Sliding window length for training samples")
    stride: int = Field(default=30, description="Sliding window stride")
    min_bars: int = Field(default=30, description="Minimum bars per day to include")
    max_codes: int | None = Field(
        default=None,
        description="Subsample this many codes (None = all)",
    )


SIGRegTokenizerRunConfig.model_rebuild()
