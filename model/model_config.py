from typing import Literal
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    # General
    model_dtype: Literal['float32', 'float16', 'bfloat16'] = Field(
        default='float32',
        description="Data type for model parameters and computations")
    teacher_forcing: bool = Field(
        default=False,
        description="Whether to use teacher forcing during training"
    )
    teacher_forcing_ratio: float = Field(
        default=0.6,
        description="Probability of enforcing teacher forcing"
    )

    # Time series
    feature_num: int = Field(
        description="Number of input/output features")
    lookback_horizon: int = Field(
        description="Number of past time steps to consider")
    forecast_horizon: int = Field(
        description="Number of future time steps to predict")
    input_mixing_dropout: float = Field(
        default=0.1,
        description="Dropout rate for input feature mixing")
    decoder_only: bool = Field(
        default=True,
        description="Whether to use a decoder-only architecture")

    # Feature-wise Linear Modulation
    use_film: bool = Field(
        default=False,
        description="Whether to use FiLM conditioning"
    )
    film_feature_num: int | None = Field(
        default=None,
        description="Number of features for FiLM conditioning"
    )
    film_hidden_size: int | None = Field(
        default=None,
        description="Hidden size for FiLM MLP"
    )
    film_dropout: float | None = Field(
        default=None,
        description="Dropout rate for FiLM MLP"
    )

    @model_validator(mode="after")
    def validate_film_config(self):
        if self.use_film:
            if self.film_feature_num is None:
                raise ValueError(
                    "film_feature_num must be set when use_film is True.")
            if self.film_hidden_size is None:
                raise ValueError(
                    "film_hidden_size must be set when use_film is True.")
            if self.film_dropout is None:
                raise ValueError(
                    "film_dropout must be set when use_film is True.")
        return self

    # Batch
    batch_size: int = Field(description="Batch size for training")

    # Overthink Model
    high_freq_step: int = Field(
        description="Number of high frequency reasoning steps per low frequency step")
    low_freq_step: int = Field(
        description="Number of low frequency reasoning steps")
    hidden_layer_num: int = Field(
        description="Number of hidden layers in the model")

    # Transformer
    hidden_size: int = Field(
        description="Hidden dimension size for transformer blocks")
    head_num: int = Field(
        description="Number of attention heads in transformer blocks")
    use_causal: bool = Field(
        description="Whether to use causal masking in attention")
    use_rope: bool = Field(
        description="Whether to use RoPE in attention layers")
    expansion_factor: float = Field(
        default=4.0,
        description="MLP expansion factor for intermediate dimension")
    attn_dropout: float = Field(
        default=0.0,
        description="Dropout rate for attention weights")

    # RoPE
    rope_theta: float = Field(
        default=10000.0,
        description="Base theta value for RoPE")
    rope_max_seq_len: int = Field(
        default=2048,
        description="Maximum sequence length for RoPE")

    # Normalisation
    rms_eps: float = Field(
        default=1e-5,
        description="Epsilon for RMS normalization")

    # Forecast
    forecast_aggregation: Literal['mean', 'ema', 'last'] = Field(
        default='mean',
        description="Aggregation method for multi-step forecast outputs")
    forecast_ema_period: int = Field(
        default=0,
        description="EMA period for smoothing forecast outputs")
    forecast_residual_scale: float = Field(
        default=0.05,
        description="Scaling factor for residual connection in forecast head")
    learnable_forecast_residual_scale: bool = Field(
        default=False,
        description="Whether to make the forecast residual scale a learnable parameter")
