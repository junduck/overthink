"""
Example script demonstrating the Overthink time series forecasting model with FiLM conditioning.

This example:
1. Creates a noisy time series function with varying volatility
2. Calculates upper and lower bounds in the lookback_horizon period as FiLM features
3. Trains the model with FiLM conditioning
4. Evaluates and plots results showing the benefit of FiLM features
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from model.model_config import ModelConfig
from model.overthink import OverthinkModel


def generate_noisy_function(
    num_samples: int = 1000,
    lookback_horizon: int = 40,
    forecast_horizon: int = 10,
    feature_num: int = 1,
    base_noise_level: float = 0.1,
    volatility_change_freq: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a noisy time series function with changing volatility.

    Args:
        num_samples: Number of training samples to generate
        lookback_horizon: Length of historical sequence
        forecast_horizon: Length of forecast sequence
        feature_num: Number of features
        base_noise_level: Base standard deviation of Gaussian noise
        volatility_change_freq: Frequency of volatility changes

    Returns:
        X: Historical sequences [num_samples, lookback_horizon, feature_num]
        y: Future sequences [num_samples, forecast_horizon, feature_num]
        film_features: FiLM features [num_samples, 2] (upper_bound, lower_bound)
    """
    total_length = num_samples + lookback_horizon + forecast_horizon

    # Generate time steps
    t = np.linspace(0, 4 * np.pi * (total_length / 100), total_length)

    # Create base signal with multiple frequencies
    signal = np.sin(t) + 0.5 * np.sin(2 * t) + 0.3 * np.sin(0.5 * t)

    # Create time-varying volatility (noise level)
    volatility = base_noise_level * \
        (1 + 0.5 * np.sin(volatility_change_freq * t))

    # Add time-varying noise
    noise = np.random.normal(0, 1, size=signal.shape) * volatility
    data = signal + noise

    # Create sliding windows
    X_list = []
    y_list = []
    film_features_list = []

    for i in range(num_samples):
        # Historical window
        x = data[i:i + lookback_horizon]
        # Future window
        y = data[i + lookback_horizon:i + lookback_horizon + forecast_horizon]

        # Calculate FiLM features: upper and lower bounds in the lookback window
        window_data = data[i:i + lookback_horizon]

        # Calculate bounds using percentiles (more robust to outliers)
        # upper_bound = np.percentile(window_data, 90)  # 90th percentile
        # lower_bound = np.percentile(window_data, 10)  # 10th percentile

        # Alternative: use mean ± std
        mean_val = np.mean(window_data)
        std_val = np.std(window_data)
        upper_bound = mean_val + 2 * std_val
        lower_bound = mean_val - 2 * std_val

        film_features = np.array([upper_bound, lower_bound])

        X_list.append(x)
        y_list.append(y)
        film_features_list.append(film_features)

    # Convert to tensors and add feature dimension
    # [N, lookback_horizon, 1]
    X = torch.FloatTensor(np.array(X_list)).unsqueeze(-1)
    # [N, forecast_horizon, 1]
    y = torch.FloatTensor(np.array(y_list)).unsqueeze(-1)
    # [N, 2] - FiLM features (upper_bound, lower_bound)
    film_features = torch.FloatTensor(np.array(film_features_list))

    # Replicate to feature_num if needed
    if feature_num > 1:
        X = X.repeat(1, 1, feature_num)
        y = y.repeat(1, 1, feature_num)

    return X, y, film_features


def print_layer_params(model, indent=0):
    """Recursively print parameter count for each layer in a model.

    Args:
        model: The PyTorch model or module
        indent: Current indentation level for nested modules
    """
    prefix = "  " * indent
    total_params = 0

    # Get all named modules and parameters
    for name, module in model.named_modules():
        # Skip the root model itself to avoid duplication
        if name == "":
            continue

        # Count parameters in this module
        module_params = sum(p.numel() for p in module.parameters())
        if module_params > 0:
            # Calculate full name path
            full_name = name
            print(f"{prefix}{full_name}: {module_params:,} parameters")
            total_params += module_params

    # Print total for this level
    if indent == 0:
        root_params = sum(p.numel() for p in model.parameters())
        print(f"\n{prefix}Total: {root_params:,} parameters")


def linear_teacher_forcing_ratio(epoch, total_epochs, initial_ratio=0.8, final_ratio=0.1):
    """Calculate teacher forcing ratio based on training progress."""
    # Linear decay from initial_ratio to final_ratio
    progress = epoch / total_epochs
    ratio = initial_ratio + (final_ratio - initial_ratio) * progress
    return max(final_ratio, ratio)


def train_model_with_film(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    film_train: torch.Tensor,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 32,
) -> list[float]:
    """Train the model on synthetic data with FiLM features.

    Args:
        model: The OverthinkModel instance
        X_train: Training inputs [N, lookback_horizon, feature_num]
        y_train: Training targets [N, forecast_horizon, feature_num]
        film_train: FiLM features [N, 2] (upper_bound, lower_bound)
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
    epoch_pbar = tqdm(range(num_epochs),
                      desc="Training with FiLM", unit="epoch")

    for epoch in epoch_pbar:
        epoch_loss = 0.0
        num_batches = 0

        # Shuffle data
        perm = torch.randperm(num_samples, device=X_train.device)
        X_train = X_train[perm]
        y_train = y_train[perm]
        film_train = film_train[perm]

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
            batch_film = film_train[i:i + batch_size]

            # Forward pass with FiLM features
            optimizer.zero_grad()
            tf_ratio = linear_teacher_forcing_ratio(epoch, num_epochs)
            predictions = model(
                input_seq=batch_X,
                film_features=batch_film,
                target_seq=batch_y,
                tf_ratio_overwrite=tf_ratio
            )

            # Debug first batch of first epoch
            if epoch == 0 and i == 0:
                print("\n[DEBUG] First batch diagnostics with FiLM:")
                print(
                    f"  Input range: [{batch_X.min():.4f}, {batch_X.max():.4f}]")
                print(
                    f"  FiLM features range: [{batch_film.min():.4f}, {batch_film.max():.4f}]")
                print(f"  FiLM features mean: {batch_film.mean(dim=0)}")
                print(
                    f"  Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
                print(
                    f"  Targets range: [{batch_y.min():.4f}, {batch_y.max():.4f}]")
                print(
                    f"  Has NaN in predictions: {torch.isnan(predictions).any()}")

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


def plot_film_results(
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    film_test: torch.Tensor,
    predictions: torch.Tensor,
    losses: list[float],
    save_path: str = "film_forecast_results.png"
):
    """Plot training loss and forecast results with FiLM features.

    Args:
        X_test: Test input sequences [N, lookback_horizon, feature_num]
        y_test: Test target sequences [N, forecast_horizon, feature_num]
        film_test: Test FiLM features [N, 2] (upper_bound, lower_bound)
        predictions: Model predictions [N, forecast_horizon, feature_num]
        losses: Training losses over epochs
        save_path: Path to save the plot
    """
    _, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Plot 1: Training loss
    axes[0].plot(losses, linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("Training Loss Over Time (with FiLM)")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Forecast visualization (first test sample)
    lookback_horizon = X_test.size(1)
    forecast_horizon = y_test.size(1)

    # Use first sample and first feature
    historical = X_test[0, :, 0].detach().cpu().numpy()
    ground_truth = y_test[0, :, 0].detach().cpu().numpy()
    forecast = predictions[0, :, 0].detach().cpu().numpy()

    # Get FiLM bounds for visualization
    upper_bound = film_test[0, 0].item()
    lower_bound = film_test[0, 1].item()

    # Create time axis
    hist_time = np.arange(lookback_horizon)
    future_time = np.arange(
        lookback_horizon, lookback_horizon + forecast_horizon)

    axes[1].plot(hist_time, historical, 'b-',
                 linewidth=2, label='Historical Data')
    axes[1].plot(future_time, ground_truth, 'g-',
                 linewidth=2, label='Ground Truth')
    axes[1].plot(future_time, forecast, 'r--', linewidth=2, label='Forecast')
    axes[1].axhline(y=upper_bound, color='orange', linestyle=':',
                    alpha=0.7, label='Upper Bound (FiLM)')
    axes[1].axhline(y=lower_bound, color='purple', linestyle=':',
                    alpha=0.7, label='Lower Bound (FiLM)')
    axes[1].axvline(x=lookback_horizon, color='k', linestyle=':', alpha=0.5)
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Value")
    axes[1].set_title(
        "Time Series Forecast with FiLM Conditioning (First Test Sample)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: FiLM feature distribution
    upper_bounds = film_test[:, 0].detach().cpu().numpy()
    lower_bounds = film_test[:, 1].detach().cpu().numpy()

    axes[2].hist(upper_bounds, bins=30, alpha=0.7,
                 label='Upper Bounds', color='orange')
    axes[2].hist(lower_bounds, bins=30, alpha=0.7,
                 label='Lower Bounds', color='purple')
    axes[2].set_xlabel("Value")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Distribution of FiLM Features (Upper/Lower Bounds)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 60)
    print("Overthink Time Series Forecasting Example with FiLM")
    print("=" * 60)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Configuration with FiLM enabled
    config = ModelConfig(
        feature_num=1,
        lookback_horizon=40,
        forecast_horizon=20,
        batch_size=128,
        high_freq_step=3,
        low_freq_step=2,
        hidden_layer_num=2,
        hidden_size=32,
        head_num=4,
        use_causal=True,
        use_rope=True,
        expansion_factor=2.0,
        forecast_aggregation='ema',
        forecast_ema_period=5,
        learnable_forecast_residual_scale=True,
        model_dtype='bfloat16',

        # FiLM configuration
        use_film=True,
        film_feature_num=2,  # Upper bound and lower bound
        film_hidden_size=32,
        film_dropout=0.,
    )

    print("\n1. Generating noisy time series data with FiLM features...")
    print(f"   - Lookback horizon: {config.lookback_horizon}")
    print(f"   - Forecast horizon: {config.forecast_horizon}")
    print(f"   - Features: {config.feature_num}")
    print(
        f"   - FiLM features: {config.film_feature_num} (upper bound, lower bound)")

    # Generate data with FiLM features
    X_train, y_train, film_train = generate_noisy_function(
        num_samples=200,
        lookback_horizon=config.lookback_horizon,
        forecast_horizon=config.forecast_horizon,
        feature_num=config.feature_num,
        base_noise_level=0.1,
        volatility_change_freq=0.05,
    )

    X_test, y_test, film_test = generate_noisy_function(
        num_samples=40,
        lookback_horizon=config.lookback_horizon,
        forecast_horizon=config.forecast_horizon,
        feature_num=config.feature_num,
        base_noise_level=0.1,
        volatility_change_freq=0.05,
    )

    print(f"   - Training samples: {X_train.size(0)}")
    print(f"   - Test samples: {X_test.size(0)}")
    print(f"   - FiLM features shape: {film_train.shape}")

    # Check data before normalization
    print("\n   - Before normalization:")
    print(f"     X_train range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(
        f"     Film features range: [{film_train.min():.4f}, {film_train.max():.4f}]")

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

    # Also normalize FiLM features using the same statistics
    film_train_norm = (film_train - train_mean) / (train_std + 1e-8)
    film_test_norm = (film_test - train_mean) / (train_std + 1e-8)

    print(f"     Mean: {train_mean:.4f}, Std: {train_std:.4f}")

    # Use normalized versions
    X_train = X_train_norm
    y_train = y_train_norm
    X_test = X_test_norm
    y_test = y_test_norm
    film_train = film_train_norm
    film_test = film_test_norm

    # Initialize model
    print("\n2. Initializing model with FiLM...")
    model = OverthinkModel(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   - Total parameters: {num_params:,}")
    print("\n   - Parameter count per layer:")
    print_layer_params(model)
    print(f"   - FiLM enabled: {config.use_film}")

    # Determine device and move model
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"   - Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("   - Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("   - Using CPU device")

    model = model.to(device)
    print(f"   - Model moved to {device}")

    # Train model
    print("\n3. Training model with FiLM features...")
    # Move training data to device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    film_train = film_train.to(device)
    print(f"   - Training data moved to {device}")

    losses = train_model_with_film(
        model=model,
        X_train=X_train,
        y_train=y_train,
        film_train=film_train,
        num_epochs=50,
        learning_rate=0.001,
        batch_size=config.batch_size,
    )

    # Evaluate on test set
    print("\n4. Evaluating on test set...")
    model.eval()
    # Move test data to device
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    film_test = film_test.to(device)

    with torch.no_grad():
        test_predictions = model(X_test, film_features=film_test)
        test_loss = nn.MSELoss()(test_predictions, y_test)
        print(f"   - Test MSE: {test_loss.item():.6f}")

    # Plot results
    print("\n5. Generating plots...")
    # Move tensors back to CPU for plotting
    plot_film_results(
        X_test=X_test.cpu(),
        y_test=y_test.cpu(),
        film_test=film_test.cpu(),
        predictions=test_predictions.cpu(),
        losses=losses,
        save_path="film_forecast_results.png"
    )

    print("\n" + "=" * 60)
    print("FiLM example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
