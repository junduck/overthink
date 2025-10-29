from typing import Literal
from pydantic import BaseModel, Field


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
    forecast_ema_decay: float = Field(
        default=0.1,
        description="EMA decay rate for smoothing forecast outputs")
    forecast_residual_scale: float = Field(
        default=0.05,
        description="Scaling factor for residual connection in forecast head")
    learnable_forecast_residual_scale: bool = Field(
        default=False,
        description="Whether to make the forecast residual scale a learnable parameter")
