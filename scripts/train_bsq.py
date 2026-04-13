"""Train OverthinkBSQ with online tokenization pipeline.

Pipeline:
    1. Dataset yields raw OHLCV + timestamps (cross-day context)
    2. Training loop normalizes + tokenizes online (frozen Kronos)
    3. Model predicts next-token via CE loss on today's portion
    4. After each epoch: eval on held-out val split with full metrics

Usage:
    # With config file
    uv run python scripts/train_bsq.py --config scripts/configs/bsq_small.json

    # With CLI overrides (data_dir must be provided if not in config)
    uv run python scripts/train_bsq.py \
        --config scripts/configs/bsq_small.json \
        --data-dir /Volumes/CN_book/cn_1m_ohlcv_per_code/ \
        --epochs 3

    # Dry run (validate pipeline end-to-end)
    uv run python scripts/train_bsq.py \
        --config scripts/configs/bsq_small.json \
        --data-dir /Volumes/CN_book/cn_1m_ohlcv_per_code/ \
        --dry-run
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from overthink.data.dataset_online import BSQOnlineDataset
from overthink.data.tokenize import OnlineTokenizer
from overthink.eval import (
    compute_perplexity,
    compute_topk_accuracy,
)
from overthink.model.bsq_config import BSQRunConfig
from overthink.model.overthink_bsq import OverthinkBSQ


def load_config(args: argparse.Namespace) -> BSQRunConfig:
    """Load config from JSON file, apply CLI overrides."""
    if args.config:
        with open(args.config) as f:
            raw = json.load(f)
        cfg = BSQRunConfig.model_validate(raw)
    else:
        cfg = BSQRunConfig(
            data={"data_dir": args.data_dir},
        )

    if args.data_dir:
        cfg.data.data_dir = args.data_dir
    if args.tokenizer:
        cfg.data.tokenizer = args.tokenizer
    if args.val_cutoff:
        cfg.data.val_cutoff = args.val_cutoff
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.ckpt_path:
        cfg.train.ckpt_path = args.ckpt_path
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    if args.val_max_batches is not None:
        cfg.train.val_max_batches = args.val_max_batches
    if args.dry_run:
        cfg.train.dry_run = True
    if args.hidden_size is not None:
        cfg.model.hidden_size = args.hidden_size
    if args.heads is not None:
        cfg.model.head_num = args.heads
    if args.stack_depth is not None:
        cfg.model.stack_depth = args.stack_depth
    if args.local_steps is not None:
        cfg.model.local_steps = args.local_steps
    if args.global_steps is not None:
        cfg.model.global_steps = args.global_steps

    return cfg


def collate_variable_length(batch):
    """Pad variable-length samples to the same sequence length."""
    ohlcvs, timestamps, masks = zip(*batch)

    max_len = max(o.size(0) for o in ohlcvs)
    B = len(batch)
    H = ohlcvs[0].size(1)

    padded_ohlcv = torch.zeros(B, max_len, H)
    padded_ts = torch.zeros(B, max_len, 5, dtype=torch.long)
    padded_mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, (ohlcv, ts, mask) in enumerate(zip(ohlcvs, timestamps, masks)):
        L = ohlcv.size(0)
        padded_ohlcv[i, :L] = ohlcv
        padded_ts[i, :L] = ts
        padded_mask[i, :L] = mask

    return padded_ohlcv, padded_ts, padded_mask


def train(
    model: OverthinkBSQ,
    tokenizer: OnlineTokenizer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """One epoch of training."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    V = model.config.vocab_size

    pbar = tqdm(loader, desc="Training", leave=False)
    for ohlcv, timestamps, loss_mask in pbar:
        ohlcv = ohlcv.to(device)
        timestamps = timestamps.to(device)
        loss_mask = loss_mask.to(device)

        s1_ids, s2_ids = tokenizer.tokenize(ohlcv)

        ts_input = timestamps[:, :-1]
        s1_input = s1_ids[:, :-1]
        s2_input = s2_ids[:, :-1]
        s1_target = s1_ids[:, 1:]
        s2_target = s2_ids[:, 1:]
        valid_mask = loss_mask[:, 1:]

        s1_logits, s2_logits = model(s1_input, s2_input, ts_input)

        s1_logits_flat = s1_logits.reshape(-1, V)
        s2_logits_flat = s2_logits.reshape(-1, V)
        s1_target_flat = s1_target.reshape(-1)
        s2_target_flat = s2_target.reshape(-1)
        valid_flat = valid_mask.reshape(-1)

        loss_s1 = F.cross_entropy(s1_logits_flat, s1_target_flat, reduction="none")
        loss_s2 = F.cross_entropy(s2_logits_flat, s2_target_flat, reduction="none")
        loss = (loss_s1[valid_flat] + loss_s2[valid_flat]).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: OverthinkBSQ,
    tokenizer: OnlineTokenizer,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict:
    """Full evaluation: CE loss, perplexity, top-k accuracy.

    Args:
        max_batches: Limit eval to N batches (None = full dataset).

    Returns:
        Dict with all metrics.
    """
    model.eval()
    V = model.config.vocab_size
    total_loss = 0.0
    n_batches = 0

    all_s1_logits = []
    all_s2_logits = []
    all_s1_target = []
    all_s2_target = []
    all_valid_mask = []

    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for ohlcv, timestamps, loss_mask in pbar:
        if max_batches is not None and n_batches >= max_batches:
            break

        ohlcv = ohlcv.to(device)
        timestamps = timestamps.to(device)
        loss_mask = loss_mask.to(device)

        s1_ids, s2_ids, _mean, _std = tokenizer.tokenize_with_stats(ohlcv)

        ts_input = timestamps[:, :-1]
        s1_input = s1_ids[:, :-1]
        s2_input = s2_ids[:, :-1]
        s1_target = s1_ids[:, 1:]
        s2_target = s2_ids[:, 1:]
        valid_mask = loss_mask[:, 1:]

        s1_logits, s2_logits = model(s1_input, s2_input, ts_input)

        s1_logits_flat = s1_logits.reshape(-1, V)
        s2_logits_flat = s2_logits.reshape(-1, V)
        s1_target_flat = s1_target.reshape(-1)
        s2_target_flat = s2_target.reshape(-1)
        valid_flat = valid_mask.reshape(-1)

        loss_s1 = F.cross_entropy(s1_logits_flat, s1_target_flat, reduction="none")
        loss_s2 = F.cross_entropy(s2_logits_flat, s2_target_flat, reduction="none")
        loss = (loss_s1[valid_flat] + loss_s2[valid_flat]).mean()
        total_loss += loss.item()
        n_batches += 1

        all_s1_logits.append(s1_logits.cpu())
        all_s2_logits.append(s2_logits.cpu())
        all_s1_target.append(s1_target.cpu())
        all_s2_target.append(s2_target.cpu())
        all_valid_mask.append(valid_mask.cpu())

    avg_loss = total_loss / max(n_batches, 1)

    if n_batches == 0:
        return {"val_loss": float("inf")}

    cat_s1_logits = torch.cat(all_s1_logits, dim=0)
    cat_s2_logits = torch.cat(all_s2_logits, dim=0)
    cat_s1_target = torch.cat(all_s1_target, dim=0)
    cat_s2_target = torch.cat(all_s2_target, dim=0)
    cat_valid = torch.cat(all_valid_mask, dim=0)

    ppl = compute_perplexity(
        cat_s1_logits, cat_s2_logits, cat_s1_target, cat_s2_target, cat_valid
    )
    topk = compute_topk_accuracy(
        cat_s1_logits, cat_s2_logits, cat_s1_target, cat_s2_target, cat_valid, k=10
    )

    return {
        "val_loss": avg_loss,
        "perplexity": ppl,
        **topk,
    }


def main():
    parser = argparse.ArgumentParser(description="Train OverthinkBSQ (online pipeline)")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--tokenizer", type=str, default="")
    parser.add_argument("--val-cutoff", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--ckpt-path", type=str, default="")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--val-max-batches", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--stack-depth", type=int, default=None)
    parser.add_argument("--local-steps", type=int, default=None)
    parser.add_argument("--global-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.config and not args.data_dir:
        parser.error("Either --config or --data-dir is required")

    cfg = load_config(args)

    print("=" * 60)
    print("OverthinkBSQ Training (Online Pipeline)")
    print("=" * 60)
    print(f"  data_dir:       {cfg.data.data_dir}")
    print(f"  batch_size:     {cfg.train.batch_size}")
    print(f"  hidden_size:    {cfg.model.hidden_size}")
    print(f"  heads:          {cfg.model.head_num}")
    print(f"  stack_depth:    {cfg.model.stack_depth}")
    print(f"  local_steps:    {cfg.model.local_steps}")
    print(f"  global_steps:   {cfg.model.global_steps}")
    print(f"  val_cutoff:     {cfg.data.val_cutoff}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device:         {device}")

    if cfg.data.val_cutoff:
        train_dataset = BSQOnlineDataset(
            cfg.data.data_dir, split="train", val_cutoff=cfg.data.val_cutoff
        )
        val_dataset = BSQOnlineDataset(
            cfg.data.data_dir, split="val", val_cutoff=cfg.data.val_cutoff
        )
    else:
        train_dataset = BSQOnlineDataset(cfg.data.data_dir)
        val_dataset = None

    model = OverthinkBSQ(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params:   {n_params:,}")

    if cfg.train.dry_run:
        print("\n--- Dry Run ---")
        sample = next(iter(train_dataset))
        ohlcv, ts, mask = sample
        print(f"  Sample shapes:  ohlcv={ohlcv.shape} ts={ts.shape} mask={mask.shape}")
        print(f"  Context bars:   {(~mask).sum()} today bars: {mask.sum()}")

        ohlcv_batch = ohlcv.unsqueeze(0).to(device)
        tok = OnlineTokenizer(cfg.data.tokenizer, device=str(device))
        s1, s2, mean, std = tok.tokenize_with_stats(ohlcv_batch)
        print(f"  Token shapes:   s1={s1.shape} s2={s2.shape}")

        ts_batch = ts.unsqueeze(0).to(device)
        s1_logits, s2_logits = model(s1, s2, ts_batch)
        print(f"  Output shapes:  s1_logits={s1_logits.shape}")

        decoded = tok.decode(s1, s2, mean, std)
        print(f"  Decoded shape:  {decoded.shape}")

        if val_dataset is not None:
            print("\n  Val dataset check:")
            val_sample = next(iter(val_dataset))
            print(f"  Val sample:     ohlcv={val_sample[0].shape}")

        print("\nDry run complete.")
        return

    tokenizer = OnlineTokenizer(cfg.data.tokenizer, device=str(device))
    print(f"  Tokenizer:      {cfg.data.tokenizer}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    ckpt_dir = Path(cfg.train.ckpt_path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = cfg.model_dump()
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(resolved_config, f, indent=2)
    print(f"  Config saved:   {ckpt_dir / 'config.json'}")

    best_val_loss = float("inf")
    history = []

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        collate_fn=collate_variable_length,
        num_workers=cfg.train.num_workers,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.train.batch_size,
            collate_fn=collate_variable_length,
            num_workers=cfg.train.num_workers,
        )

    for epoch in range(cfg.train.epochs):
        t0 = time.time()
        train_loss = train(
            model, tokenizer, train_loader, optimizer, device, cfg.train.grad_clip
        )
        elapsed = time.time() - t0

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "time_s": round(elapsed, 1),
        }

        print(
            f"Epoch {epoch + 1}/{cfg.train.epochs}  train_loss={train_loss:.4f}  "
            f"time={elapsed:.0f}s",
            end="",
        )

        if val_loader is not None:
            val_metrics = evaluate(
                model,
                tokenizer,
                val_loader,
                device,
                max_batches=cfg.train.val_max_batches,
            )
            epoch_metrics.update(val_metrics)
            print(
                f"  val_loss={val_metrics['val_loss']:.4f}  "
                f"ppl={val_metrics['perplexity']:.1f}  "
                f"top10={val_metrics['mean_top10']:.3f}",
                end="",
            )

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                torch.save(
                    {
                        "config": cfg.model.model_dump(),
                        "model_state_dict": model.state_dict(),
                        "train_loss": train_loss,
                        "val_metrics": val_metrics,
                        "epoch": epoch,
                    },
                    ckpt_dir / "best_model.pt",
                )
                print(f"  *best*", end="")

        print()

        torch.save(
            {
                "config": cfg.model.model_dump(),
                "model_state_dict": model.state_dict(),
                "train_loss": train_loss,
                "epoch": epoch,
            },
            ckpt_dir / "latest.pt",
        )

        history.append(epoch_metrics)
        with open(ckpt_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
