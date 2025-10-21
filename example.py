"""
Example script demonstrating the Overthink time series forecasting model.

This example:
1. Creates synthetic time series data (sine wave with noise)
2. Trains the model on the data
3. Makes predictions and plots results
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from model.model_config import ModelConfig
from model.overthink import OverthinkModel


def generate_synthetic_data(
    num_samples: int = 1000,
    lookback_horizon: int = 50,
    forecast_horizon: int = 20,
    feature_num: int = 1,
    noise_level: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic time series data (sine wave with noise).

    Args:
        num_samples: Number of training samples to generate
        lookback_horizon: Length of historical sequence
        forecast_horizon: Length of forecast sequence
        feature_num: Number of features
        noise_level: Standard deviation of Gaussian noise

    Returns:
        X: Historical sequences [num_samples, lookback_horizon, feature_num]
        y: Future sequences [num_samples, forecast_horizon, feature_num]
    """
    total_length = num_samples + lookback_horizon + forecast_horizon

    # Generate time steps
    t = np.linspace(0, 4 * np.pi * (total_length / 100), total_length)

    # Create sine wave with multiple frequencies
    data = np.sin(t) + 0.5 * np.sin(2 * t) + 0.3 * np.sin(0.5 * t)

    # Add noise
    data += np.random.normal(0, noise_level, size=data.shape)

    # Create sliding windows
    X_list = []
    y_list = []

    for i in range(num_samples):
        # Historical window
        x = data[i:i + lookback_horizon]
        # Future window
        y = data[i + lookback_horizon:i + lookback_horizon + forecast_horizon]

        X_list.append(x)
        y_list.append(y)

    # Convert to tensors and add feature dimension
    # [N, lookback_horizon, 1]
    X = torch.FloatTensor(np.array(X_list)).unsqueeze(-1)
    # [N, forecast_horizon, 1]
    y = torch.FloatTensor(np.array(y_list)).unsqueeze(-1)

    # Replicate to feature_num if needed
    if feature_num > 1:
        X = X.repeat(1, 1, feature_num)
        y = y.repeat(1, 1, feature_num)

    return X, y


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 32,
) -> list[float]:
    """Train the model on synthetic data.

    Args:
        model: The OverthinkModel instance
        X_train: Training inputs [N, lookback_horizon, feature_num]
        y_train: Training targets [N, forecast_horizon, feature_num]
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training

    Returns:
        List of training losses
    """
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    losses = []
    num_samples = X_train.size(0)

    # Training loop with tqdm progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")

    for epoch in epoch_pbar:
        epoch_loss = 0.0
        num_batches = 0

        # Shuffle data
        perm = torch.randperm(num_samples, device=X_train.device)
        X_train = X_train[perm]
        y_train = y_train[perm]

        # Mini-batch training with progress bar
        batch_pbar = tqdm(
            range(0, num_samples, batch_size),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=False,
            unit="batch"
        )

        for i in batch_pbar:
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]

            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch_X)

            # Debug first batch of first epoch
            if epoch == 0 and i == 0:
                print("\n[DEBUG] First batch diagnostics:")
                print(
                    f"  Input range: [{batch_X.min():.4f}, {batch_X.max():.4f}]")
                print(
                    f"  Input mean/std: {batch_X.mean():.4f} / {batch_X.std():.4f}")
                print(
                    f"  Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
                print(
                    f"  Predictions mean/std: {predictions.mean():.4f} / {predictions.std():.4f}")
                print(
                    f"  Targets range: [{batch_y.min():.4f}, {batch_y.max():.4f}]")
                print(
                    f"  Has NaN in predictions: {torch.isnan(predictions).any()}")
                print(
                    f"  Has Inf in predictions: {torch.isinf(predictions).any()}")

            # Compute loss
            loss = criterion(predictions, batch_y)

            # Check for NaN/Inf
            if not torch.isfinite(loss):
                print(
                    f"\nWarning: Loss became {loss.item()} at epoch {epoch + 1}, batch {i // batch_size}")
                print(
                    f"Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
                print(
                    f"Targets range: [{batch_y.min():.4f}, {batch_y.max():.4f}]")
                print(
                    f"  Has NaN in predictions: {torch.isnan(predictions).any()}")
                print(
                    f"  Has Inf in predictions: {torch.isinf(predictions).any()}")
                raise ValueError("Loss is NaN or Inf - training stopped")

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # Update batch progress bar
            batch_pbar.set_postfix({"batch_loss": f"{loss.item():.6f}"})

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        # Update epoch progress bar
        epoch_pbar.set_postfix({
            "loss": f"{avg_loss:.6f}",
            "best": f"{min(losses):.6f}"
        })

    return losses


def plot_results(
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    predictions: torch.Tensor,
    losses: list[float],
    save_path: str = "forecast_results.png"
):
    """Plot training loss and forecast results.

    Args:
        X_test: Test input sequences [N, lookback_horizon, feature_num]
        y_test: Test target sequences [N, forecast_horizon, feature_num]
        predictions: Model predictions [N, forecast_horizon, feature_num]
        losses: Training losses over epochs
        save_path: Path to save the plot
    """
    _, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Training loss
    axes[0].plot(losses, linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("Training Loss Over Time")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Forecast visualization (first test sample)
    lookback_horizon = X_test.size(1)
    forecast_horizon = y_test.size(1)

    # Use first sample and first feature
    historical = X_test[0, :, 0].detach().cpu().numpy()
    ground_truth = y_test[0, :, 0].detach().cpu().numpy()
    forecast = predictions[0, :, 0].detach().cpu().numpy()

    # Create time axis
    hist_time = np.arange(lookback_horizon)
    future_time = np.arange(
        lookback_horizon, lookback_horizon + forecast_horizon)

    axes[1].plot(hist_time, historical, 'b-',
                 linewidth=2, label='Historical Data')
    axes[1].plot(future_time, ground_truth, 'g-',
                 linewidth=2, label='Ground Truth')
    axes[1].plot(future_time, forecast, 'r--', linewidth=2, label='Forecast')
    axes[1].axvline(x=lookback_horizon, color='k', linestyle=':', alpha=0.5)
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Value")
    axes[1].set_title("Time Series Forecast (First Test Sample)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 60)
    print("Overthink Time Series Forecasting Example")
    print("=" * 60)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Configuration - Small and fast for debugging
    config = ModelConfig(
        feature_num=1,
        lookback_horizon=40,
        forecast_horizon=10,
        batch_size=16,
        high_freq_step=3,
        low_freq_step=2,
        hidden_layer_num=2,
        hidden_size=32,
        head_num=2,
        use_causal=True,
        use_rope=True,
        expansion_factor=2.0,  # Reduced from 4.0
        forecast_aggregation='mean',  # Simpler than ema
        forecast_ema_decay=0.1,
    )

    print("\n1. Generating synthetic data...")
    print(f"   - Lookback horizon: {config.lookback_horizon}")
    print(f"   - Forecast horizon: {config.forecast_horizon}")
    print(f"   - Features: {config.feature_num}")

    # Generate data - Reduced for faster debugging
    X_train, y_train = generate_synthetic_data(
        num_samples=200,      # Reduced from 800
        lookback_horizon=config.lookback_horizon,
        forecast_horizon=config.forecast_horizon,
        feature_num=config.feature_num,
        noise_level=0.05,     # Reduced noise
    )

    X_test, y_test = generate_synthetic_data(
        num_samples=20,       # Reduced from 100
        lookback_horizon=config.lookback_horizon,
        forecast_horizon=config.forecast_horizon,
        feature_num=config.feature_num,
        noise_level=0.05,     # Reduced noise
    )

    print(f"   - Training samples: {X_train.size(0)}")
    print(f"   - Test samples: {X_test.size(0)}")

    # Check data before normalization
    print("\n   - Before normalization:")
    print(f"     X_train range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"     X_train mean/std: {X_train.mean():.4f} / {X_train.std():.4f}")
    print(
        f"     Has NaN/Inf: {torch.isnan(X_train).any()} / {torch.isinf(X_train).any()}")

    # Normalize data to prevent NaN during training
    print("\n   - Normalizing data...")
    train_mean = X_train.mean()
    train_std = X_train.std()

    # Check for zero std
    if train_std < 1e-6:
        print(f"     WARNING: Standard deviation is too small: {train_std}")
        train_std = torch.tensor(1.0)

    X_train_norm = (X_train - train_mean) / (train_std + 1e-8)
    y_train_norm = (y_train - train_mean) / (train_std + 1e-8)
    X_test_norm = (X_test - train_mean) / (train_std + 1e-8)
    y_test_norm = (y_test - train_mean) / (train_std + 1e-8)
    print(f"     Mean: {train_mean:.4f}, Std: {train_std:.4f}")

    # Check after normalization
    print("\n   - After normalization:")
    print(
        f"     X_train range: [{X_train_norm.min():.4f}, {X_train_norm.max():.4f}]")
    print(
        f"     X_train mean/std: {X_train_norm.mean():.4f} / {X_train_norm.std():.4f}")
    print(
        f"     Has NaN/Inf: {torch.isnan(X_train_norm).any()} / {torch.isinf(X_train_norm).any()}")

    # Use normalized versions
    X_train = X_train_norm
    y_train = y_train_norm
    X_test = X_test_norm
    y_test = y_test_norm

    # Initialize model
    print("\n2. Initializing model...")
    model = OverthinkModel(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   - Total parameters: {num_params:,}")

    # Determine device and move model
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"   - Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"   - Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print(f"   - Using CPU device")

    model = model.to(device)
    print(f"   - Model moved to {device}")

    # Train model - Fewer epochs for faster debugging
    print("\n3. Training model...")
    # Move training data to device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    print(f"   - Training data moved to {device}")

    losses = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        num_epochs=20,        # Reduced from 100
        learning_rate=0.0001,  # Reduced learning rate to prevent NaN
        batch_size=config.batch_size,
    )

    # Evaluate on test set
    print("\n4. Evaluating on test set...")
    model.eval()
    # Move test data to device
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    with torch.no_grad():
        test_predictions = model(X_test)
        test_loss = nn.MSELoss()(test_predictions, y_test)
        print(f"   - Test MSE: {test_loss.item():.6f}")

    # Plot results
    print("\n5. Generating plots...")
    # Move tensors back to CPU for plotting
    plot_results(
        X_test=X_test.cpu(),
        y_test=y_test.cpu(),
        predictions=test_predictions.cpu(),
        losses=losses,
        save_path="forecast_results.png"
    )

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
