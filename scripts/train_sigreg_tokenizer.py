"""Train SIGRegTokenizer on raw OHLCV data from SQLite.

Usage:
    uv run python scripts/train_sigreg_tokenizer.py --sqlite-dir /Volumes/CN_book/cn_1m_ohlcv_sqlite
    uv run python scripts/train_sigreg_tokenizer.py --config scripts/configs/sigreg_tokenizer.json
    uv run python scripts/train_sigreg_tokenizer.py --config scripts/configs/sigreg_tokenizer.json --dry-run
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from overthink.data.dataset_window import SlidingWindowDataset
from overthink.model.sigreg_tokenizer import SIGRegTokenizer
from overthink.model.sigreg_tokenizer_config import (
    SIGRegTokenizerRunConfig,
    SIGRegTokenizerDataConfig,
)


def collate_ohlcv(batch):
    """Stack fixed-size windows."""
    return torch.stack(batch)


def train_one_epoch(
    model: SIGRegTokenizer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: OneCycleLR,
    device: torch.device,
    sigreg_lambda: float,
    grad_clip: float = 2.0,
    report_every: int = 500,
) -> dict:
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_sigreg = 0.0
    n_batches = 0

    for ohlcv in loader:
        ohlcv = ohlcv.to(device)

        (recon_s1, recon_full), sigreg_loss, _, _ = model(ohlcv)
        recon_loss = F.mse_loss(recon_s1, ohlcv) + F.mse_loss(recon_full, ohlcv)
        loss = (recon_loss + sigreg_lambda * sigreg_loss) / 2

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_sigreg += sigreg_loss.item()
        n_batches += 1

        if n_batches % report_every == 0:
            avg_l = total_loss / n_batches
            avg_r = total_recon / n_batches
            avg_s = total_sigreg / n_batches
            lr = scheduler.get_last_lr()[0]
            print(
                f"\r  step {n_batches} loss={avg_l:.4f} "
                f"recon={avg_r:.4f} sigreg={avg_s:.2f} lr={lr:.2e}",
                end="",
                flush=True,
            )

    n = max(n_batches, 1)
    return {
        "train_loss": total_loss / n,
        "train_recon": total_recon / n,
        "train_sigreg": total_sigreg / n,
    }


@torch.no_grad()
def epoch_report(model: SIGRegTokenizer, loader: DataLoader, device: torch.device, max_batches: int = 50) -> dict:
    """Quick diagnostic: codebook utilization, latent stats from a few batches."""
    model.eval()
    all_s1 = []
    all_s2 = []
    latent_sum = None
    latent_sq_sum = None
    n_elements = 0

    for i, ohlcv in enumerate(loader):
        if i >= max_batches:
            break
        ohlcv = ohlcv.to(device)

        s1, s2 = model.encode(ohlcv, half=True)
        all_s1.append(s1.cpu())
        all_s2.append(s2.cpu())

        z = model._encode_latent(ohlcv)
        flat = z.reshape(-1, z.size(-1))
        n_elements += flat.size(0)
        if latent_sum is None:
            latent_sum = flat.sum(0).cpu()
            latent_sq_sum = flat.square().sum(0).cpu()
        else:
            latent_sum += flat.sum(0).cpu()
            latent_sq_sum += flat.square().sum(0).cpu()

    s1_cat = torch.cat(all_s1)
    s2_cat = torch.cat(all_s2)
    mean = latent_sum / n_elements
    var = latent_sq_sum / n_elements - mean.square()
    std = var.clamp(min=0).sqrt()

    return {
        "unique_s1": len(s1_cat.unique()),
        "unique_s2": len(s2_cat.unique()),
        "s1_utilization": f"{len(s1_cat.unique()) / 1024 * 100:.1f}%",
        "s2_utilization": f"{len(s2_cat.unique()) / 1024 * 100:.1f}%",
        "latent_mean_abs": f"{mean.abs().mean().item():.4f}",
        "latent_std_avg": f"{std.mean().item():.4f}",
        "latent_std_range": f"[{std.min().item():.2f}, {std.max().item():.2f}]",
    }


def main():
    parser = argparse.ArgumentParser(description="Train SIGRegTokenizer")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--sqlite-dir", type=str, default="")
    parser.add_argument("--max-codes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--ckpt-path", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            raw = json.load(f)
        cfg = SIGRegTokenizerRunConfig.model_validate(raw)
    elif args.sqlite_dir:
        cfg = SIGRegTokenizerRunConfig(
            data=SIGRegTokenizerDataConfig(sqlite_dir=args.sqlite_dir),
        )
    else:
        parser.error("Either --config or --sqlite-dir is required")

    if args.sqlite_dir:
        cfg.data.sqlite_dir = args.sqlite_dir
    if args.max_codes is not None:
        cfg.data.max_codes = args.max_codes
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.ckpt_path:
        cfg.train.ckpt_path = args.ckpt_path
    if args.dry_run:
        cfg.train.dry_run = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("SIGRegTokenizer Training")
    print("=" * 60)
    print(f"  sqlite_dir:     {cfg.data.sqlite_dir}")
    print(f"  window_size:    {cfg.data.window_size}")
    print(f"  stride:         {cfg.data.stride}")
    print(f"  max_codes:      {cfg.data.max_codes or 'all'}")
    print(f"  encoder_window: {cfg.model.encoder_window}")
    print(f"  d_model:        {cfg.model.d_model}")
    print(f"  n_enc_layers:   {cfg.model.n_enc_layers}")
    print(f"  n_dec_layers:   {cfg.model.n_dec_layers}")
    print(f"  sigreg_lambda:  {cfg.model.sigreg_lambda}")
    print(f"  batch_size:     {cfg.train.batch_size}")
    print(f"  lr:             {cfg.train.lr}")
    print(f"  epochs:         {cfg.train.epochs}")
    print(f"  device:         {device}")

    model = SIGRegTokenizer(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params:   {n_params:,}")

    if cfg.train.dry_run:
        print("\n--- Dry Run ---")
        ds = SlidingWindowDataset(
            cfg.data.sqlite_dir,
            window_size=cfg.data.window_size,
            stride=cfg.data.stride,
            max_codes=3,
        )
        sample = next(iter(ds))
        print(f"  Sample shape:   {sample.shape}")
        sample = sample.unsqueeze(0).to(device)
        (recon_s1, recon_full), sigreg_loss, quantized, indices = model(sample)
        print(f"  recon_s1:       {recon_s1.shape}")
        print(f"  recon_full:     {recon_full.shape}")
        print(f"  sigreg_loss:    {sigreg_loss.item():.4f}")
        print(f"  quantized:      {quantized.shape} range=[{quantized.min():.1f}, {quantized.max():.1f}]")
        s1, s2 = model.encode(sample, half=True)
        print(f"  s1_ids:         {s1.shape} unique={len(s1.unique())}")
        print(f"  s2_ids:         {s2.shape} unique={len(s2.unique())}")
        print("\nDry run complete.")
        return

    ds = SlidingWindowDataset(
        cfg.data.sqlite_dir,
        window_size=cfg.data.window_size,
        stride=cfg.data.stride,
        max_codes=cfg.data.max_codes,
    )

    loader = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        collate_fn=collate_ohlcv,
        num_workers=cfg.train.num_workers,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=cfg.train.betas,
    )

    # Estimate steps: count windows from a sample of codes
    # Per code: ~475K bars, window=200 stride=30 => ~15827 windows, aligned to batch_size
    import sqlite3
    sample_dbs = ds._db_files[:min(10, len(ds._db_files))]
    windows_per_code = []
    for db_path in sample_dbs:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
        conn.close()
        n_windows = (rows - cfg.data.window_size) // cfg.data.stride + 1
        n_aligned = (n_windows // cfg.train.batch_size) * cfg.train.batch_size
        windows_per_code.append(n_aligned)
    avg_windows = sum(windows_per_code) / len(windows_per_code)
    total_windows = int(avg_windows * len(ds._db_files))
    steps_per_epoch = total_windows // cfg.train.batch_size
    total_steps = steps_per_epoch * cfg.train.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.train.lr,
        total_steps=total_steps,
        pct_start=cfg.train.warmup_pct,
        div_factor=10,
    )

    print(f"  steps/epoch:    {steps_per_epoch}")
    print(f"  total_steps:    {total_steps}")
    print()

    ckpt_dir = Path(cfg.train.ckpt_path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(cfg.model_dump(), f, indent=2)

    best_loss = float("inf")
    history = []

    for epoch in range(cfg.train.epochs):
        t0 = time.time()
        metrics = train_one_epoch(
            model, loader, optimizer, scheduler, device,
            cfg.model.sigreg_lambda, cfg.train.grad_clip,
        )
        elapsed = time.time() - t0

        epoch_metrics = {"epoch": epoch + 1, **metrics, "time_s": round(elapsed, 1)}
        print(
            f"\nEpoch {epoch + 1}/{cfg.train.epochs}  "
            f"loss={metrics['train_loss']:.4f}  "
            f"recon={metrics['train_recon']:.4f}  "
            f"sigreg={metrics['train_sigreg']:.2f}  "
            f"time={elapsed:.0f}s",
        )

        diag = epoch_report(model, loader, device)
        print(
            f"  codebook: s1={diag['s1_utilization']} s2={diag['s2_utilization']}  "
            f"latent: |μ|={diag['latent_mean_abs']} σ={diag['latent_std_avg']} {diag['latent_std_range']}",
            end="",
        )

        if metrics["train_loss"] < best_loss:
            best_loss = metrics["train_loss"]
            torch.save(
                {"config": cfg.model.model_dump(), "model_state_dict": model.state_dict(), "epoch": epoch},
                ckpt_dir / "best_model.pt",
            )
            print("  *best*", end="")

        print()
        torch.save(
            {"config": cfg.model.model_dump(), "model_state_dict": model.state_dict(), "epoch": epoch},
            ckpt_dir / "latest.pt",
        )
        history.append(epoch_metrics)
        with open(ckpt_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
