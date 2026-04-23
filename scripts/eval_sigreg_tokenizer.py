"""Evaluate a trained SIGRegTokenizer.

Measures:
  1. Reconstruction MSE (recon_s1, recon_full)
  2. SIGReg loss (how close to N(0,I))
  3. Codebook utilization (unique s1/s2 tokens)
  4. Token stability (same bar at different positions)
  5. Latent distribution stats (mean, std, kurtosis per dim)

Usage:
    uv run python scripts/eval_sigreg_tokenizer.py checkpoints/sigreg_tokenizer/best_model.pt
    uv run python scripts/eval_sigreg_tokenizer.py checkpoints/sigreg_tokenizer/best_model.pt --max-codes 50
"""

import argparse

import torch
import torch.nn.functional as F

from overthink.data.dataset_window import SlidingWindowDataset
from overthink.model.sigreg_tokenizer import SIGRegTokenizer
from overthink.model.sigreg_tokenizer_config import SIGRegTokenizerConfig
from torch.utils.data import DataLoader


def collate_ohlcv(batch):
    return torch.stack(batch)


@torch.no_grad()
def evaluate(model, loader, device, max_batches=500):
    model.eval()
    total_recon_s1 = 0.0
    total_recon_full = 0.0
    total_sigreg = 0.0
    all_s1 = []
    all_s2 = []
    all_latent_means = []
    all_latent_stds = []
    n = 0

    for ohlcv in loader:
        if n >= max_batches:
            break
        ohlcv = ohlcv.to(device)

        (recon_s1, recon_full), sigreg_loss, quantized, _ = model(ohlcv)

        total_recon_s1 += F.mse_loss(recon_s1, ohlcv).item()
        total_recon_full += F.mse_loss(recon_full, ohlcv).item()
        total_sigreg += sigreg_loss.item()
        n += 1

        s1_ids, s2_ids = model.encode(ohlcv, half=True)
        all_s1.append(s1_ids.cpu())
        all_s2.append(s2_ids.cpu())

        if len(all_latent_means) < 10:
            z = model._encode_latent(ohlcv)
            all_latent_means.append(z.mean(dim=[0, 1]).cpu())
            all_latent_stds.append(z.std(dim=[0, 1]).cpu())

    n = max(n, 1)
    s1_cat = torch.cat(all_s1)
    s2_cat = torch.cat(all_s2)
    mean_of_means = torch.stack(all_latent_means).mean(0)
    mean_of_stds = torch.stack(all_latent_stds).mean(0)

    return {
        "recon_mse_s1": total_recon_s1 / n,
        "recon_mse_full": total_recon_full / n,
        "sigreg": total_sigreg / n,
        "unique_s1": len(s1_cat.unique()),
        "unique_s2": len(s2_cat.unique()),
        "total_positions": s1_cat.numel(),
        "s1_utilization": len(s1_cat.unique()) / 1024,
        "s2_utilization": len(s2_cat.unique()) / 1024,
        "latent_mean_abs_max": mean_of_means.abs().max().item(),
        "latent_mean_abs_avg": mean_of_means.abs().mean().item(),
        "latent_std_avg": mean_of_stds.mean().item(),
        "latent_std_min": mean_of_stds.min().item(),
        "latent_std_max": mean_of_stds.max().item(),
    }


@torch.no_grad()
def test_stability(model, device, window_size=200):
    model.eval()
    bar = torch.tensor([10.5, 10.6, 10.4, 10.55, 1000.0, 10500.0])

    a = torch.randn(1, 500, 6, device=device)
    a[0, 100] = bar
    s1_a, s2_a = model.encode(a, half=True)

    b = torch.randn(1, 800, 6, device=device)
    b[0, 300] = bar
    s1_b, s2_b = model.encode(b, half=True)

    same_s1 = (s1_a[0, 100] == s1_b[0, 300]).item()
    same_s2 = (s2_a[0, 100] == s2_b[0, 300]).item()
    return same_s1, same_s2


def main():
    parser = argparse.ArgumentParser(description="Evaluate SIGRegTokenizer")
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint")
    parser.add_argument("--sqlite-dir", type=str, default="/Volumes/CN_book/cn_1m_ohlcv_sqlite")
    parser.add_argument("--max-codes", type=int, default=50)
    parser.add_argument("--max-batches", type=int, default=500)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, weights_only=True)
    cfg = SIGRegTokenizerConfig.model_validate(ckpt["config"])
    model = SIGRegTokenizer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Epoch: {ckpt.get('epoch', '?')}")
    print(f"Device: {device}")
    print()

    ds = SlidingWindowDataset(
        args.sqlite_dir,
        window_size=cfg.encoder_window,
        stride=30,
        batch_size=128,
        max_codes=args.max_codes,
    )
    loader = DataLoader(ds, batch_size=128, collate_fn=collate_ohlcv)

    metrics = evaluate(model, loader, device, args.max_batches)
    same_s1, same_s2 = test_stability(model, device, cfg.encoder_window)

    print("=" * 60)
    print("Reconstruction")
    print(f"  MSE (coarse s1 only):  {metrics['recon_mse_s1']:.6f}")
    print(f"  MSE (full s1+s2):      {metrics['recon_mse_full']:.6f}")

    print("\nSIGReg (lower = closer to N(0,I), 0 = perfect)")
    print(f"  SIGReg loss:           {metrics['sigreg']:.2f}")

    print("\nCodebook Utilization (target: >90%)")
    print(f"  Unique s1: {metrics['unique_s1']}/1024 ({metrics['s1_utilization']*100:.1f}%)")
    print(f"  Unique s2: {metrics['unique_s2']}/1024 ({metrics['s2_utilization']*100:.1f}%)")

    print("\nLatent Distribution (target: mean≈0, std≈1)")
    print(f"  Mean |μ|_max:          {metrics['latent_mean_abs_max']:.4f}")
    print(f"  Mean |μ|_avg:          {metrics['latent_mean_abs_avg']:.4f}")
    print(f"  Std avg:               {metrics['latent_std_avg']:.4f}")
    print(f"  Std range:             [{metrics['latent_std_min']:.4f}, {metrics['latent_std_max']:.4f}]")

    print("\nToken Stability (same bar, different position)")
    print(f"  s1 stable: {same_s1}")
    print(f"  s2 stable: {same_s2}")
    print("=" * 60)

    print("\nSuccess Criteria:")
    checks = [
        ("Codebook s1 > 80%", metrics["s1_utilization"] > 0.8),
        ("Codebook s2 > 80%", metrics["s2_utilization"] > 0.8),
        ("Latent mean close to 0", metrics["latent_mean_abs_avg"] < 0.5),
        ("Latent std close to 1", abs(metrics["latent_std_avg"] - 1.0) < 0.5),
    ]
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")


if __name__ == "__main__":
    main()
