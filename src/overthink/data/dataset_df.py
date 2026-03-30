from typing import Optional
import polars as pl


import polars as pl
import numpy as np
import torch


def generate_train_data(
    train_data: pl.DataFrame,
    feature_columns: list,
    lookback_horizon: int = 40,
    forecast_horizon: int = 10,
    target_columns: Optional[list] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate train_X and train_Y from train_data using sliding window approach.

    Args:
        train_data: DataFrame containing the time series data with Code, row_num, and feature columns
        feature_columns: List of column names to use as features
        lookback_horizon: Number of time steps to look back for input sequence
        forecast_horizon: Number of time steps to forecast ahead
        target_columns: List of column names to predict (defaults to feature_columns[:3] for High, Low, VWAP)

    Returns:
        train_X: Input sequences [num_samples, lookback_horizon, feature_num]
        train_Y: Target sequences [num_samples, forecast_horizon, target_num]
    """
    if target_columns is None:
        # Default to predicting High, Low, VWAP (first 3 features)
        target_columns = feature_columns[:3]

    # Get unique codes
    codes = train_data['Code'].unique()

    # Initialize lists to store sequences
    X_sequences = []
    Y_sequences = []

    # Process each code separately to maintain time series continuity
    for code in codes:
        # Filter data for current code and sort by row_num
        code_data = train_data.filter(pl.col('Code') == code).sort('row_num')

        # Extract features as numpy array
        features = code_data.select(feature_columns).to_numpy()
        targets = code_data.select(target_columns).to_numpy()

        # Calculate maximum starting index for sliding window
        max_start_idx = len(code_data) - lookback_horizon - \
            forecast_horizon + 1

        # Generate sliding windows if we have enough data points
        if max_start_idx > 0:
            for start_idx in range(max_start_idx):
                # Input sequence: lookback_horizon time steps
                X_seq = features[start_idx:start_idx + lookback_horizon]

                # Target sequence: forecast_horizon time steps
                Y_seq = targets[start_idx + lookback_horizon:start_idx +
                                lookback_horizon + forecast_horizon]

                X_sequences.append(X_seq)
                Y_sequences.append(Y_seq)

    # Convert to numpy arrays and then to tensors
    X_array = np.array(X_sequences)
    Y_array = np.array(Y_sequences)

    # Convert to PyTorch tensors
    train_X = torch.FloatTensor(X_array)
    train_Y = torch.FloatTensor(Y_array)

    return train_X, train_Y


def generate_train_data_with_row_num(
    train_data: pl.DataFrame,
    feature_columns: list,
    lookback_horizon: int = 40,
    forecast_horizon: int = 10,
    target_columns: list = None,
    overlap_horizon: int = 30
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate train_X and train_Y from train_data using overlapping windows with row_num indexing.
    This version creates overlapping windows to maximize training samples.

    Args:
        train_data: DataFrame containing the time series data with Code, row_num, and feature columns
        feature_columns: List of column names to use as features
        lookback_horizon: Number of time steps to look back for input sequence
        forecast_horizon: Number of time steps to forecast ahead
        target_columns: List of column names to predict (defaults to feature_columns[:3] for High, Low, VWAP)
        overlap_horizon: Number of time steps to overlap between consecutive samples

    Returns:
        train_X: Input sequences [num_samples, lookback_horizon, feature_num]
        train_Y: Target sequences [num_samples, forecast_horizon, target_num]
    """
    if target_columns is None:
        # Default to predicting High, Low, VWAP (first 3 features)
        target_columns = feature_columns[:3]

    # Get unique codes
    codes = train_data['Code'].unique()

    # Initialize lists to store sequences
    X_sequences = []
    Y_sequences = []

    # Process each code separately to maintain time series continuity
    for code in codes:
        # Filter data for current code and sort by row_num
        code_data = train_data.filter(pl.col('Code') == code).sort('row_num')

        # Extract features as numpy array
        features = code_data.select(feature_columns).to_numpy()
        targets = code_data.select(target_columns).to_numpy()
        row_nums = code_data['row_num'].to_numpy()

        # Calculate maximum starting index for sliding window
        max_start_idx = len(code_data) - lookback_horizon - \
            forecast_horizon + 1

        # Generate sliding windows with overlap if we have enough data points
        if max_start_idx > 0:
            # Use overlap_horizon to determine step size
            step_size = lookback_horizon - overlap_horizon

            # Generate windows
            for start_idx in range(0, max_start_idx, step_size):
                # Ensure we don't go beyond the data
                if start_idx + lookback_horizon + forecast_horizon <= len(code_data):
                    # Input sequence: lookback_horizon time steps
                    X_seq = features[start_idx:start_idx + lookback_horizon]

                    # Target sequence: forecast_horizon time steps
                    Y_seq = targets[start_idx + lookback_horizon:start_idx +
                                    lookback_horizon + forecast_horizon]

                    X_sequences.append(X_seq)
                    Y_sequences.append(Y_seq)

    # Convert to numpy arrays and then to tensors
    X_array = np.array(X_sequences)
    Y_array = np.array(Y_sequences)

    # Convert to PyTorch tensors
    train_X = torch.FloatTensor(X_array)
    train_Y = torch.FloatTensor(Y_array)

    return train_X, train_Y
