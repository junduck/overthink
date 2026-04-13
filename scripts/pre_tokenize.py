"""Pre-tokenize raw OHLCV data using frozen Kronos BSQ tokenizer.

Reads raw .pq files from {data_dir}/{code}/{date}.pq, builds cross-day
(yesterday+today) windows, normalizes and tokenizes on GPU, saves per-code
.pt files to {output_dir}/{code}.pt.

Usage:
    uv run python scripts/pre_tokenize.py --config scripts/configs/bsq_local.json
    uv run python scripts/pre_tokenize.py --config scripts/configs/bsq_local.json --resume
"""

import argparse
import json
import time
from pathlib import Path

import torch

from overthink.data.dataset_online import (
    _safe_read,
    df_to_tensors,
)
from overthink.data.tokenize import OnlineTokenizer
from overthink.model.bsq_config import BSQDataConfig


def _get_codes(data_dir: Path) -> list[str]:
    return sorted(d.name for d in data_dir.iterdir() if d.is_dir())


def _get_dates(code_dir: Path) -> list[str]:
    return sorted(p.stem for p in code_dir.glob("*.pq") if not p.name.startswith("."))


def _collect_raw_samples(
    data_dir: Path, code: str, min_bars: int
) -> list[tuple[torch.Tensor, torch.Tensor, int, int, str]]:
    """Read all dates for a code, build cross-day windows.

    Returns list of (ohlcv, timestamps, prev_len, today_len, date).
    """
    code_dir = data_dir / code
    dates = _get_dates(code_dir)

    samples = []
    prev_ohlcv = None
    prev_ts = None

    for date in dates:
        path = code_dir / f"{date}.pq"
        df = _safe_read(path)
        if df is None or len(df) < min_bars:
            prev_ohlcv = None
            continue

        today_ohlcv, today_ts = df_to_tensors(df)

        if prev_ohlcv is not None:
            ohlcv = torch.cat([prev_ohlcv, today_ohlcv], dim=0)
            timestamps = torch.cat([prev_ts, today_ts], dim=0)
            samples.append((ohlcv, timestamps, len(prev_ohlcv), len(today_ohlcv), date))

        prev_ohlcv = today_ohlcv
        prev_ts = today_ts

    return samples


def _tokenize_code(
    tokenizer: OnlineTokenizer,
    data_dir: Path,
    code: str,
    out_file: Path,
    min_bars: int,
    batch_size: int,
    device: torch.device,
):
    raw = _collect_raw_samples(data_dir, code, min_bars)
    if not raw:
        return

    all_s1 = []
    all_s2 = []
    all_ts = []
    all_prev_lens = []
    all_today_lens = []
    all_dates = []

    for i in range(0, len(raw), batch_size):
        chunk = raw[i : i + batch_size]
        max_len = max(s[0].size(0) for s in chunk)
        B = len(chunk)

        padded = torch.zeros(B, max_len, 6)
        for j, (ohlcv, _ts, _pl, _tl, _d) in enumerate(chunk):
            padded[j, : ohlcv.size(0)] = ohlcv

        padded = padded.to(device)
        s1, s2 = tokenizer.tokenize(padded)

        all_s1.append(s1.cpu())
        all_s2.append(s2.cpu())

        for ohlcv, ts, pl, tl, d in chunk:
            pad_len = max_len - ohlcv.size(0)
            if pad_len > 0:
                ts = torch.cat([ts, torch.zeros(pad_len, 5, dtype=torch.long)])
            all_ts.append(ts)
            all_prev_lens.append(pl)
            all_today_lens.append(tl)
            all_dates.append(d)

    torch.save(
        {
            "s1_ids": torch.cat(all_s1, dim=0).to(torch.int16),
            "s2_ids": torch.cat(all_s2, dim=0).to(torch.int16),
            "timestamps": torch.stack(all_ts).to(torch.int16),
            "prev_lens": torch.tensor(all_prev_lens, dtype=torch.int16),
            "today_lens": torch.tensor(all_today_lens, dtype=torch.int16),
            "dates": all_dates,
        },
        out_file,
    )


def pre_tokenize(
    data_cfg: BSQDataConfig,
    output_dir: str,
    tokenizer_batch_size: int = 2048,
    resume: bool = False,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(data_cfg.data_dir)

    print(f"Device: {device}")
    print(f"Data dir: {data_cfg.data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Tokenizer batch size: {tokenizer_batch_size}")
    print(f"Resume: {resume}")

    tokenizer = OnlineTokenizer(data_cfg.tokenizer, device=str(device))

    codes = _get_codes(data_dir)
    print(f"Total codes: {len(codes)}")

    done = 0
    skipped = 0
    failed = 0
    t_start = time.time()

    for ci, code in enumerate(codes):
        out_file = output_path / f"{code}.pt"
        if resume and out_file.exists():
            skipped += 1
            continue

        try:
            _tokenize_code(
                tokenizer,
                data_dir,
                code,
                out_file,
                data_cfg.min_bars,
                tokenizer_batch_size,
                device,
            )
            done += 1
        except Exception as e:
            print(f"\n  ERROR {code}: {e}")
            failed += 1
            if out_file.exists():
                out_file.unlink()

        if (ci + 1) % 100 == 0 or ci == len(codes) - 1:
            elapsed = time.time() - t_start
            rate = (ci + 1) / elapsed
            print(
                f"  [{ci + 1}/{len(codes)}] "
                f"done={done} skipped={skipped} failed={failed} "
                f"rate={rate:.1f} codes/s"
            )

    elapsed = time.time() - t_start
    print(
        f"\nDone. {done} tokenized, {skipped} skipped, {failed} failed "
        f"in {elapsed:.0f}s"
    )


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize OHLCV data")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--tokenizer", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.config:
        with open(args.config, encoding="utf-8") as f:
            raw = json.load(f)
        cfg = BSQDataConfig.model_validate(raw.get("data", {}))
    elif args.data_dir:
        cfg = BSQDataConfig(data_dir=args.data_dir)
    else:
        parser.error("Either --config or --data-dir required")

    if args.tokenizer:
        cfg.tokenizer = args.tokenizer

    output_dir = args.output_dir or str(Path(cfg.data_dir).parent / "tokenized")

    pre_tokenize(cfg, output_dir, args.batch_size, args.resume)


if __name__ == "__main__":
    main()
