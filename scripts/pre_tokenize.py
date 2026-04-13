"""Pre-tokenize raw OHLCV data using frozen Kronos BSQ tokenizer.

Loads per-date whole-market files from {market_dir}/{date}-ohlcv-1m.pq,
groups by code, builds cross-day windows, normalizes and tokenizes on GPU.
Saves per-code .pt files to {output_dir}/{code}.pt.

Approach: load one year of data at a time into memory (~1s per year),
partition by code, accumulate samples across years, then tokenize all
samples for each code in large GPU batches.

Usage:
    uv run python scripts/pre_tokenize.py --config scripts/configs/bsq_local.json
    uv run python scripts/pre_tokenize.py --config scripts/configs/bsq_local.json --resume
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import polars as pl
import torch

from overthink.data.dataset_online import OHLCV_COLS, extract_timestamps
from overthink.data.tokenize import OnlineTokenizer
from overthink.model.bsq_config import BSQDataConfig


def _extract_date(filename: str) -> str:
    """Extract date string from filename like '2018-01-02-ohlcv-1m.pq'."""
    return filename[:10]


def _get_years(market_dir: Path) -> list[str]:
    """Get unique years from market data filenames."""
    years = set()
    for f in os.listdir(market_dir):
        if f.endswith(".pq") and not f.startswith("."):
            years.add(f[:4])
    return sorted(years)


def _load_year(market_dir: Path, year: str) -> dict[str, pl.DataFrame]:
    """Load one year of market data, return {code: DataFrame}."""
    files = sorted(
        str(market_dir / f)
        for f in os.listdir(market_dir)
        if f.startswith(year) and f.endswith(".pq") and not f.startswith(".")
    )
    if not files:
        return {}

    df = pl.read_parquet(files)
    partitioned = df.partition_by("Code", as_dict=True)
    return {k[0]: v for k, v in partitioned.items()}


def _build_windows_for_year(
    code_dfs: dict[str, pl.DataFrame],
    min_bars: int,
) -> dict[str, list[tuple]]:
    """Build cross-day windows for all codes from one year of data.

    Returns {code: [(ohlcv, timestamps, prev_len, today_len, date), ...]}
    """
    result = defaultdict(list)

    for code, df in code_dfs.items():
        df = df.sort("TimeInterval")
        df = df.with_columns(df["TimeInterval"].dt.date().alias("_date"))

        prev_ohlcv = None
        prev_ts = None

        for day_df in df.partition_by("_date"):
            day_df = day_df.sort("TimeInterval")
            date_str = str(day_df["_date"][0])

            if len(day_df) < min_bars:
                prev_ohlcv = None
                continue

            ohlcv = torch.tensor(day_df[OHLCV_COLS].to_numpy(), dtype=torch.float32)
            ts = extract_timestamps(day_df["TimeInterval"])

            if prev_ohlcv is not None:
                combined_ohlcv = torch.cat([prev_ohlcv, ohlcv], dim=0)
                combined_ts = torch.cat([prev_ts, ts], dim=0)
                result[code].append(
                    (combined_ohlcv, combined_ts, len(prev_ohlcv), len(ohlcv), date_str)
                )

            prev_ohlcv = ohlcv
            prev_ts = ts

    return result


def _tokenize_code(
    tokenizer: OnlineTokenizer,
    samples: list[tuple],
    out_file: Path,
    batch_size: int,
    device: torch.device,
):
    """Tokenize all samples for one code and save."""
    if not samples:
        return

    all_s1 = []
    all_s2 = []
    all_ts = []
    all_prev_lens = []
    all_today_lens = []
    all_dates = []

    for i in range(0, len(samples), batch_size):
        chunk = samples[i : i + batch_size]
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

    if data_cfg.market_dir:
        market_dir = Path(data_cfg.market_dir)
    else:
        market_dir = Path(data_cfg.data_dir)

    print(f"Device: {device}")
    print(f"Market dir: {market_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Tokenizer batch size: {tokenizer_batch_size}")
    print(f"Resume: {resume}")

    tokenizer = OnlineTokenizer(data_cfg.tokenizer, device=str(device))

    years = _get_years(market_dir)
    print(f"Years: {years}")

    accumulated: dict[str, list[tuple]] = defaultdict(list)
    done = 0
    skipped = 0
    failed = 0
    t_start = time.time()

    for yi, year in enumerate(years):
        t0 = time.time()
        print(f"\n--- Loading {year} ({yi + 1}/{len(years)}) ---")

        code_dfs = _load_year(market_dir, year)
        if not code_dfs:
            print(f"  No data for {year}")
            continue
        print(f"  Loaded {len(code_dfs)} codes in {time.time() - t0:.1f}s")

        windows = _build_windows_for_year(code_dfs, data_cfg.min_bars)
        print(
            f"  Built {sum(len(v) for v in windows.values())} windows across {len(windows)} codes in {time.time() - t0:.1f}s"
        )

        for code, new_samples in windows.items():
            accumulated[code].extend(new_samples)

    print(f"\n--- Tokenizing {len(accumulated)} codes ---")

    codes = sorted(accumulated.keys())
    for ci, code in enumerate(codes):
        out_file = output_path / f"{code}.pt"
        if resume and out_file.exists():
            skipped += 1
            continue

        try:
            _tokenize_code(
                tokenizer,
                accumulated[code],
                out_file,
                tokenizer_batch_size,
                device,
            )
            done += 1
        except Exception as e:
            print(f"\n  ERROR {code}: {e}")
            failed += 1
            if out_file.exists():
                out_file.unlink()

        if (ci + 1) % 500 == 0 or ci == len(codes) - 1:
            elapsed = time.time() - t_start
            print(
                f"  [{ci + 1}/{len(codes)}] "
                f"done={done} skipped={skipped} failed={failed} "
                f"elapsed={elapsed:.0f}s"
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
