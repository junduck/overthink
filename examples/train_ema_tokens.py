"""Train OverthinkSimple on daily EMA token data.

Usage:
    python train_ema_tokens.py --config config.json
    python train_ema_tokens.py --dry-run  # Generate default config and exit
    python train_ema_tokens.py --generate-config my_config.json
"""

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np
import sqlite3
import torch
from pydantic import BaseModel, Field
from torch import nn
from torch.utils.data import DataLoader, Dataset

from overthink.model import OverthinkSimple, SimpleConfig


class DataConfig(BaseModel):
    """Configuration for data preparation."""

    db_path: str = Field(description="Path to SQLite database")
    context_len: int = Field(
        default=20, description="Number of input context tokens")
    pred_horizon: int = Field(
        default=2, description="Number of tokens to predict")
    min_tokens: int = Field(default=22, description="Minimum tokens per stock")
    overlap: bool = Field(
        default=True, description="Generate overlapping sequences")
    clip_percentile: float = Field(
        default=99.0, description="Percentile for clipping")
    train_ratio: float = Field(
        default=0.8, description="Fraction of stocks for training"
    )
    val_ratio: float = Field(
        default=0.1, description="Fraction of stocks for validation"
    )
    batch_strategy: Literal["single_stock", "mixed_stock"] = Field(
        default="mixed_stock", description="Batch composition strategy"
    )
    batch_size: int = Field(default=32, description="Batch size")


class TrainConfig(BaseModel):
    """Configuration for training."""

    lr: float = Field(default=1e-3, description="Learning rate")
    epochs: int = Field(default=100, description="Number of training epochs")
    tf_ratio_start: float = Field(
        default=0.8, description="Initial teacher forcing ratio"
    )
    tf_ratio_end: float = Field(
        default=0.1, description="Final teacher forcing ratio")
    loss_weights: list[float] = Field(
        default=[1.0, 0.3], description="Weight for each prediction step"
    )
    ckpt_path: str = Field(
        default="checkpoints", description="Checkpoint save directory"
    )
    dry_run: bool = Field(
        default=False, description="Print config and exit without training"
    )


class ModelConfig(BaseModel):
    """Configuration for model architecture."""

    hidden_size: int = Field(default=64, description="Hidden dimension size")
    head_num: int = Field(default=4, description="Number of attention heads")
    query_group: int = Field(
        default=0, description="Query groups for GQA (0 = MHA)")
    hidden_layer_num: int = Field(
        default=3, description="Number of transformer layers")
    expansion_factor: float = Field(
        default=2.0, description="MLP expansion factor")
    local_reason_step: int = Field(
        default=2, description="Local reasoning steps")
    global_reason_step: int = Field(
        default=2, description="Global reasoning steps")
    use_causal: bool = Field(default=True, description="Use causal attention")
    use_rope: bool = Field(
        default=True, description="Use rotary position embeddings")
    rope_max_seq_len: int = Field(
        default=512, description="Max sequence length for RoPE"
    )


class ExperimentConfig(BaseModel):
    """Combined experiment configuration."""

    data: DataConfig
    train: TrainConfig
    model: ModelConfig

    def save(self, path: str | Path) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def create_default_config() -> ExperimentConfig:
    """Create default experiment configuration."""
    return ExperimentConfig(
        data=DataConfig(
            db_path="example_data/ts_daily_token.db",
            context_len=20,
            pred_horizon=2,
            min_tokens=22,
            overlap=True,
            clip_percentile=99.0,
            train_ratio=0.8,
            val_ratio=0.1,
            batch_strategy="mixed_stock",
            batch_size=32,
        ),
        train=TrainConfig(
            lr=1e-3,
            epochs=50,
            tf_ratio_start=0.8,
            tf_ratio_end=0.1,
            loss_weights=[1.0, 0.3],
            ckpt_path="checkpoints",
            dry_run=False,
        ),
        model=ModelConfig(
            hidden_size=64,
            head_num=4,
            query_group=0,
            hidden_layer_num=3,
            expansion_factor=2.0,
            local_reason_step=2,
            global_reason_step=2,
            use_causal=True,
            use_rope=True,
            rope_max_seq_len=512,
        ),
    )


class PreprocessedData:
    """Container for preprocessed data."""

    def __init__(
        self,
        sequences: list[np.ndarray],
        ts_codes: list[str],
        clip_len: float,
        clip_str: float,
        mean: np.ndarray,
        std: np.ndarray,
    ):
        self.sequences = sequences
        self.ts_codes = ts_codes
        self.clip_len = clip_len
        self.clip_str = clip_str
        self.mean = mean
        self.std = std


def load_and_preprocess(config: DataConfig) -> PreprocessedData:
    """Load data from SQLite and preprocess tokens."""

    conn = sqlite3.connect(config.db_path)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT ts_code, length, strength FROM ema ORDER BY ts_code, start_idx"
    )
    rows = cursor.fetchall()
    conn.close()

    ts_code_prev = None
    current_seq = []
    sequences = []
    ts_codes = []

    for ts_code, length, strength in rows:
        if ts_code != ts_code_prev:
            if current_seq:
                sequences.append(np.array(current_seq, dtype=np.float32))
                ts_codes.append(ts_code_prev)
            current_seq = []
            ts_code_prev = ts_code
        current_seq.append([length, strength])

    if current_seq:
        sequences.append(np.array(current_seq, dtype=np.float32))
        ts_codes.append(ts_code_prev)

    all_lengths = np.concatenate([seq[:, 0] for seq in sequences])
    all_strengths = np.concatenate([seq[:, 1] for seq in sequences])

    log_lengths = np.log1p(all_lengths)
    log_strengths = np.log1p(all_strengths)

    clip_len = float(np.percentile(log_lengths, config.clip_percentile))
    clip_str = float(np.percentile(log_strengths, config.clip_percentile))

    processed_sequences = []
    for seq in sequences:
        log_len = np.log1p(seq[:, 0])
        log_str = np.log1p(seq[:, 1])
        log_len = np.clip(log_len, None, clip_len)
        log_str = np.clip(log_str, None, clip_str)
        processed_sequences.append(np.stack([log_len, log_str], axis=-1))

    all_data = np.concatenate(processed_sequences, axis=0)
    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0)

    normalized_sequences = []
    for seq in processed_sequences:
        normalized = (seq - mean) / (std + 1e-8)
        normalized_sequences.append(normalized)

    valid_indices = [
        i for i, seq in enumerate(normalized_sequences) if len(seq) >= config.min_tokens
    ]
    normalized_sequences = [normalized_sequences[i] for i in valid_indices]
    ts_codes = [ts_codes[i] for i in valid_indices]

    print(f"Loaded {len(normalized_sequences)} valid sequences")
    print(f"Clip thresholds: length={clip_len:.4f}, strength={clip_str:.4f}")
    print(f"Mean: {mean}, Std: {std}")

    return PreprocessedData(
        sequences=normalized_sequences,
        ts_codes=ts_codes,
        clip_len=clip_len,
        clip_str=clip_str,
        mean=mean,
        std=std,
    )


def split_by_stock(
    data: PreprocessedData, config: DataConfig
) -> tuple[PreprocessedData, PreprocessedData, PreprocessedData]:
    """Split data by stock codes."""

    n_stocks = len(data.ts_codes)
    indices = np.arange(n_stocks)
    np.random.shuffle(indices)

    n_train = int(n_stocks * config.train_ratio)
    n_val = int(n_stocks * config.val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train: n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def make_split(idx_list):
        return PreprocessedData(
            sequences=[data.sequences[i] for i in idx_list],
            ts_codes=[data.ts_codes[i] for i in idx_list],
            clip_len=data.clip_len,
            clip_str=data.clip_str,
            mean=data.mean,
            std=data.std,
        )

    return make_split(train_idx), make_split(val_idx), make_split(test_idx)


class EMASequenceDataset(Dataset):
    """Dataset for EMA token sequences."""

    def __init__(
        self,
        data: PreprocessedData,
        config: DataConfig,
    ):
        self.sequences = data.sequences
        self.ts_codes = data.ts_codes
        self.context_len = config.context_len
        self.pred_horizon = config.pred_horizon
        self.overlap = config.overlap
        self.samples = []

        stock_start_indices = []
        current_idx = 0

        for seq_idx, seq in enumerate(self.sequences):
            n_tokens = len(seq)
            min_len = self.context_len + self.pred_horizon

            if n_tokens < min_len:
                continue

            if self.overlap:
                n_samples = n_tokens - min_len + 1
                for i in range(n_samples):
                    self.samples.append((seq_idx, i))
            else:
                n_samples = n_tokens // min_len
                for i in range(n_samples):
                    start = i * min_len
                    self.samples.append((seq_idx, start))

            if config.batch_strategy == "single_stock":
                stock_start_indices.append((current_idx, len(self.samples)))
                current_idx = len(self.samples)

        self.stock_start_indices = stock_start_indices

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_idx, start = self.samples[idx]
        seq = self.sequences[seq_idx]

        input_seq = seq[start: start + self.context_len]
        target_seq = seq[
            start + self.context_len: start + self.context_len + self.pred_horizon
        ]

        return torch.from_numpy(input_seq), torch.from_numpy(target_seq)


def make_data_loaders(
    train_data: PreprocessedData,
    val_data: PreprocessedData,
    test_data: PreprocessedData,
    config: DataConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for train, validation, and test."""

    train_dataset = EMASequenceDataset(train_data, config)
    val_dataset = EMASequenceDataset(val_data, config)
    test_dataset = EMASequenceDataset(test_data, config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def compute_metrics(
    model: OverthinkSimple,
    dataloader: DataLoader,
    device: torch.device,
    loss_weights: list[float],
) -> dict[str, float]:
    """Compute evaluation metrics."""

    model.eval()

    all_lengths_pred: list[np.ndarray] = []
    all_lengths_target: list[np.ndarray] = []
    all_strengths_pred: list[np.ndarray] = []
    all_strengths_target: list[np.ndarray] = []

    step_losses = [0.0] * len(loss_weights)
    total_samples = 0

    with torch.no_grad():
        for input_seq, target_seq in dataloader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            current = input_seq
            preds: list[torch.Tensor] = []

            for t in range(target_seq.size(1)):
                pred = model(current)
                preds.append(pred)
                current = torch.cat([current, pred], dim=1)

            preds_tensor = torch.cat(preds, dim=1)

            for t, w in enumerate(loss_weights):
                step_losses[t] += (
                    w
                    * nn.functional.mse_loss(
                        preds_tensor[:, t, :], target_seq[:, t, :]
                    ).item()
                )

            all_lengths_pred.append(preds_tensor[:, :, 0].cpu().numpy())
            all_lengths_target.append(target_seq[:, :, 0].cpu().numpy())
            all_strengths_pred.append(preds_tensor[:, :, 1].cpu().numpy())
            all_strengths_target.append(target_seq[:, :, 1].cpu().numpy())

            total_samples += input_seq.size(0)

    lengths_pred = np.concatenate(all_lengths_pred, axis=0)
    lengths_target = np.concatenate(all_lengths_target, axis=0)
    strengths_pred = np.concatenate(all_strengths_pred, axis=0)
    strengths_target = np.concatenate(all_strengths_target, axis=0)

    mse_length = float(np.mean((lengths_pred - lengths_target) ** 2))
    mse_strength = float(np.mean((strengths_pred - strengths_target) ** 2))

    mape_length = float(
        np.mean(np.abs(lengths_pred - lengths_target) /
                (np.abs(lengths_target) + 1e-8))
    )
    mape_strength = float(
        np.mean(
            np.abs(strengths_pred - strengths_target)
            / (np.abs(strengths_target) + 1e-8)
        )
    )

    flat_strengths = strengths_target.flatten()
    top20_pct = np.percentile(np.abs(flat_strengths), 80)
    top20_mask = np.abs(strengths_target) >= top20_pct
    top20_error = float(
        np.mean((strengths_pred[top20_mask] -
                strengths_target[top20_mask]) ** 2)
    )

    metrics: dict[str, float] = {
        "mse_length": mse_length,
        "mse_strength": mse_strength,
        "mape_length": mape_length,
        "mape_strength": mape_strength,
        "top20_strength_mse": top20_error,
    }
    for t in range(len(step_losses)):
        metrics[f"loss_step{t + 1}"] = step_losses[t] / total_samples

    return metrics


def train(
    model: OverthinkSimple,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
) -> None:
    """Train the model."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    tf_decay = (config.tf_ratio_start - config.tf_ratio_end) / config.epochs

    Path(config.ckpt_path).mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        model.train()
        tf_ratio = max(config.tf_ratio_end,
                       config.tf_ratio_start - tf_decay * epoch)

        epoch_loss = 0.0
        n_batches = 0

        for input_seq, target_seq in train_loader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            loss = model.train_step(
                input_seq,
                target_seq,
                optimizer,
                tf_ratio=tf_ratio,
                loss_weights=config.loss_weights,
            )
            epoch_loss += loss
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        val_metrics = compute_metrics(
            model, val_loader, device, config.loss_weights)
        val_loss = sum(
            val_metrics[f"loss_step{t + 1}"] for t in range(len(config.loss_weights))
        )

        print(
            f"Epoch {epoch + 1}/{config.epochs} - "
            f"Train Loss: {avg_train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
            f"TF Ratio: {tf_ratio:.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                Path(config.ckpt_path) / "best_model.pt",
            )

    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")


def print_config(
    data_config: DataConfig,
    train_config: TrainConfig,
    model_config: ModelConfig,
) -> None:
    """Print all configurations for dry run."""

    print("\n" + "=" * 60)
    print("DATA CONFIGURATION")
    print("=" * 60)
    print(f"  db_path:         {data_config.db_path}")
    print(f"  context_len:     {data_config.context_len}")
    print(f"  pred_horizon:    {data_config.pred_horizon}")
    print(f"  min_tokens:      {data_config.min_tokens}")
    print(f"  overlap:         {data_config.overlap}")
    print(f"  clip_percentile: {data_config.clip_percentile}")
    print(f"  train_ratio:     {data_config.train_ratio}")
    print(f"  val_ratio:       {data_config.val_ratio}")
    print(f"  batch_strategy:  {data_config.batch_strategy}")
    print(f"  batch_size:      {data_config.batch_size}")

    print("\n" + "=" * 60)
    print("TRAIN CONFIGURATION")
    print("=" * 60)
    print(f"  lr:              {train_config.lr}")
    print(f"  epochs:          {train_config.epochs}")
    print(f"  tf_ratio_start:  {train_config.tf_ratio_start}")
    print(f"  tf_ratio_end:    {train_config.tf_ratio_end}")
    print(f"  loss_weights:     {train_config.loss_weights}")
    print(f"  ckpt_path:       {train_config.ckpt_path}")
    print(f"  dry_run:         {train_config.dry_run}")

    print("\n" + "=" * 60)
    print("MODEL CONFIGURATION")
    print("=" * 60)
    print(f"  hidden_size:       {model_config.hidden_size}")
    print(f"  head_num:          {model_config.head_num}")
    print(f"  query_group:       {model_config.query_group}")
    print(f"  hidden_layer_num:  {model_config.hidden_layer_num}")
    print(f"  expansion_factor:  {model_config.expansion_factor}")
    print(f"  local_reason_step: {model_config.local_reason_step}")
    print(f"  global_reason_step:{model_config.global_reason_step}")
    print(f"  use_causal:        {model_config.use_causal}")
    print(f"  use_rope:          {model_config.use_rope}")
    print(f"  rope_max_seq_len:  {model_config.rope_max_seq_len}")


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Train OverthinkSimple on EMA token data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--generate-config",
        type=str,
        default=None,
        help="Generate default config file and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without training",
    )
    args = parser.parse_args()

    if args.generate_config:
        config = create_default_config()
        config.save(args.generate_config)
        print(f"Default config saved to {args.generate_config}")
        return

    if args.config:
        config = ExperimentConfig.load(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = create_default_config()
        print("Using default config")

    if args.dry_run:
        config.train.dry_run = True

    print_config(config.data, config.train, config.model)

    if config.train.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - Exiting without training")
        print("=" * 60)
        return

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    print("\n" + "=" * 60)
    print("Loading and preprocessing data...")
    print("=" * 60)
    data = load_and_preprocess(config.data)

    print("\n" + "=" * 60)
    print("Splitting data by stock...")
    print("=" * 60)
    train_data, val_data, test_data = split_by_stock(data, config.data)
    print(f"Train: {len(train_data.sequences)} stocks")
    print(f"Val: {len(val_data.sequences)} stocks")
    print(f"Test: {len(test_data.sequences)} stocks")

    print("\n" + "=" * 60)
    print("Creating data loaders...")
    print("=" * 60)
    train_loader, val_loader, test_loader = make_data_loaders(
        train_data, val_data, test_data, config.data
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    print("\n" + "=" * 60)
    print("Creating model...")
    print("=" * 60)
    simple_config = SimpleConfig(
        feature_num=2,
        hidden_size=config.model.hidden_size,
        head_num=config.model.head_num,
        query_group=config.model.query_group,
        hidden_layer_num=config.model.hidden_layer_num,
        expansion_factor=config.model.expansion_factor,
        local_reason_step=config.model.local_reason_step,
        global_reason_step=config.model.global_reason_step,
        use_causal=config.model.use_causal,
        use_rope=config.model.use_rope,
        rope_max_seq_len=config.model.rope_max_seq_len,
    )
    model = OverthinkSimple(simple_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    Path(config.train.ckpt_path).mkdir(parents=True, exist_ok=True)
    config.save(Path(config.train.ckpt_path) / "config.json")

    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    train(model, train_loader, val_loader, config.train, device)

    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)
    checkpoint = torch.load(
        Path(config.train.ckpt_path) / "best_model.pt", weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = compute_metrics(
        model, test_loader, device, config.train.loss_weights
    )

    print("Test Results:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.6f}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
