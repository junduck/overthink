# Overthink: A Hierarchical Reasoning Framework for Time Series Forecasting

## Quick Start

1. **Install [uv](https://github.com/astral-sh/uv):**

```sh
pip install uv
```

2. **Run the example script using uv:**

```sh
uv run example.py
```

This will install dependencies and execute `example.py` in a single step.

## Model Overview

Overthink implements a hierachical reasoning model for time series autoregressive forecasting.

## Architecture Flowchart

```mermaid
flowchart TD
    Start([Input: Historical Time Series<br/>B × lookback_horizon × feature_num]) --> InputProj[Input Projection<br/>Linear + SwiGLU]

    InputProj --> InitStates[Initialize Reasoning States<br/>High Freq & Low Freq]

    InitStates --> LoopStart{Low Frequency<br/>Iteration<br/>low_freq_step times}

    LoopStart -->|Iterate| ResH[res_h = low_freq + residual]

    ResH --> HFLoop{High Frequency<br/>Iteration<br/>high_freq_step times}

    HFLoop -->|Iterate| HFReasoning[High Freq Reasoning<br/>TransStack: Causal Attention]
    HFReasoning -->|Update high_freq| HFLoop

    HFLoop -->|Complete| LFReasoning[Low Freq Reasoning<br/>TransBlock: Bi-directional Attention<br/>Input: low_freq + high_freq]

    LFReasoning -->|Update low_freq| LoopStart

    LoopStart -->|Complete| Forecast[Forecast Head<br/>Autoregressive Prediction<br/>next = last_value + δ]

    Forecast --> Prediction[Next Step Prediction<br/>B × 1 × feature_num]

    Prediction --> UpdateSeq[Update Sequence<br/>Slide window: remove oldest,<br/>append prediction]

    UpdateSeq --> ARLoop{Autoregressive<br/>Loop<br/>forecast_horizon<br/>steps?}

    ARLoop -->|Continue| InputProj
    ARLoop -->|Complete| Output([Output: Forecasted Series<br/>B × forecast_horizon × feature_num])

    style Start fill:#e1f5ff
    style Output fill:#e1ffe1
    style HFReasoning fill:#fff4e1
    style LFReasoning fill:#ffe1f5
    style Forecast fill:#f5e1ff
```

### Key Highlights

1. Feature mixing.
2. Efficient hierarchical reasoning for high and low frequency components.
3. Can be trained with multi-scale trend following loss for strong trend forecasting performance.

## TODO

1. Add active computatation time support.
    - Not sure if ACT is beneficial for this model at the moment.
2. Adapt personal training pipeline for large scale experiments.
    - Full market portfolio builder.
    - Automatic train -> eval -> backtest -> report pipeline.
3. TBD
