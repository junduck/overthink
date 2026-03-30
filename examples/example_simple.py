"""Example script demonstrating OverthinkSimple for time series forecasting."""

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from overthink.model import SimpleConfig, OverthinkSimple


def generate_sine_data(
    num_samples: int = 1000,
    seq_len: int = 50,
    horizon: int = 10,
    noise_level: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic sine wave data with sliding windows."""
    total_length = num_samples + seq_len + horizon

    t = np.linspace(0, 4 * np.pi * (total_length / 100), total_length)
    data = np.sin(t) + 0.5 * np.sin(2 * t) + 0.3 * np.sin(0.5 * t)
    data += np.random.normal(0, noise_level, size=data.shape)

    X_list, y_list = [], []
    for i in range(num_samples):
        X_list.append(data[i : i + seq_len])
        y_list.append(data[i + seq_len : i + seq_len + horizon])

    X = torch.FloatTensor(np.array(X_list)).unsqueeze(-1)
    y = torch.FloatTensor(np.array(y_list)).unsqueeze(-1)

    return X, y


def main():
    print("=" * 60)
    print("OverthinkSimple Time Series Forecasting Example")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    config = SimpleConfig(
        feature_num=1,
        hidden_size=32,
        head_num=2,
        hidden_layer_num=2,
        local_reason_step=2,
        global_reason_step=2,
        expansion_factor=2.0,
        use_causal=True,
        use_rope=True,
    )
    print(
        f"\nModel config: {sum(p.numel() for p in OverthinkSimple(config).parameters()):,} params"
    )

    print("\n1. Generating data...")
    X_train, y_train = generate_sine_data(num_samples=200, seq_len=50, horizon=10)
    X_test, y_test = generate_sine_data(num_samples=50, seq_len=50, horizon=10)

    # Normalize data
    train_mean = X_train.mean()
    train_std = X_train.std()
    X_train = (X_train - train_mean) / (train_std + 1e-8)
    y_train = (y_train - train_mean) / (train_std + 1e-8)
    X_test = (X_test - train_mean) / (train_std + 1e-8)
    y_test = (y_test - train_mean) / (train_std + 1e-8)

    print(f"   Train: {X_train.shape} -> {y_train.shape}")
    print(f"   Test:  {X_test.shape} -> {y_test.shape}")

    print("\n2. Training...")
    model = OverthinkSimple(config)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    horizon = 10
    num_epochs = 30
    tf_ratio = 0.8  # Teacher forcing ratio

    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        perm = torch.randperm(X_train.size(0))
        X_train = X_train[perm]
        y_train = y_train[perm]

        epoch_loss = 0.0
        for i in range(0, X_train.size(0), 64):
            loss = model.train_step(
                X_train[i : i + 32], y_train[i : i + 32], optimizer, tf_ratio
            )
            epoch_loss += loss

        tf_ratio = max(0.1, tf_ratio * 0.99)

        if (epoch + 1) % 10 == 0:
            print(
                f"   Epoch {epoch + 1}: loss = {epoch_loss:.6f}, tf_ratio = {tf_ratio:.2f}"
            )

    print("\n3. Evaluating...")
    model.eval()
    with torch.no_grad():
        test_preds = model.autoregressive_generate(X_test[:4], horizon=10)
        test_loss = torch.nn.functional.mse_loss(test_preds[:, -10:, :], y_test[:4])
        print(f"   Test MSE: {test_loss.item():.6f}")

    print("\n4. Plotting results...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    idx = 0
    lookback = X_test.size(1)
    horizon = y_test.size(1)

    axes[0].plot(range(lookback), X_test[idx, :, 0], "b-", label="Historical")
    axes[0].plot(
        range(lookback, lookback + horizon),
        y_test[idx, :, 0],
        "g-",
        label="Ground Truth",
    )
    axes[0].plot(
        range(lookback, lookback + horizon),
        test_preds[idx, -horizon:, 0],
        "r--",
        label="Prediction",
    )
    axes[0].axvline(x=lookback, color="k", linestyle=":", alpha=0.5)
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Value")
    axes[0].set_title("Time Series Forecast")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y_test[:4, :, 0], test_preds[:4, -horizon:, 0], alpha=0.5)
    axes[1].plot([-2, 2], [-2, 2], "r--", label="Perfect prediction")
    axes[1].set_xlabel("Ground Truth")
    axes[1].set_ylabel("Prediction")
    axes[1].set_title("Prediction vs Ground Truth")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("simple_forecast_results.png", dpi=150)
    print("   Saved to simple_forecast_results.png")
    plt.show()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
