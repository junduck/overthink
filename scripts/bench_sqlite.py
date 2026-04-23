"""Benchmark iteration speed from per-code SQLite databases.

Loads one code's entire data into memory (numpy), iterates with
cross-day lookback simulating training, then moves to next code.

Usage:
    uv run python scripts/bench_sqlite.py --db-dir /path/to/sqlite
    uv run python scripts/bench_sqlite.py --db-dir /path/to/sqlite --max-codes 100
"""

import argparse
import sqlite3
import time
from pathlib import Path

import numpy as np
import torch

_LOAD_QUERY = (
    "SELECT date_epoch, minute_of_day, hour, dow, dom, month, "
    "open, high, low, close, volume, turnover "
    "FROM bars ORDER BY time_idx"
)

COL_TS = 5  # first 5 non-epoch cols are temporal: minute_of_day..month
COL_OHLCV = 6  # last 6 cols: open..turnover


def load_code(db_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load all bars for one code into memory.

    Returns:
        (temporal [N, 5], ohlcv [N, 6], date_epochs [N]) or None.
    """
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.execute("PRAGMA journal_mode = WAL")
    rows = conn.execute(_LOAD_QUERY).fetchall()
    conn.close()
    if not rows:
        return None
    arr = np.array(rows)
    date_epochs = arr[:, 0].astype(np.int32)
    unique_epochs = np.unique(date_epochs)
    temporal = arr[:, 1 : 1 + COL_TS].astype(np.int64)
    ohlcv = arr[:, 1 + COL_TS :].astype(np.float32)
    return temporal, ohlcv, date_epochs, unique_epochs


def iter_code(
    temporal: np.ndarray,
    ohlcv: np.ndarray,
    date_epochs: np.ndarray,
    unique_epochs: np.ndarray,
    min_bars: int = 30,
):
    """Yield (ohlcv, timestamps, loss_mask) samples with cross-day lookback."""
    prev_ohlcv: torch.Tensor | None = None
    prev_ts: torch.Tensor | None = None

    for epoch in unique_epochs:
        mask = date_epochs == epoch
        day_ts = torch.tensor(temporal[mask], dtype=torch.long)
        day_ohlcv = torch.tensor(ohlcv[mask], dtype=torch.float32)
        n = len(day_ohlcv)
        if n < min_bars:
            prev_ohlcv = None
            continue

        if prev_ohlcv is not None:
            yield (
                torch.cat([prev_ohlcv, day_ohlcv]),
                torch.cat([prev_ts, day_ts]),
                torch.cat(
                    [
                        torch.zeros(len(prev_ohlcv), dtype=torch.bool),
                        torch.ones(n, dtype=torch.bool),
                    ]
                ),
            )

        prev_ohlcv = day_ohlcv
        prev_ts = day_ts


def main() -> None:
    """Run SQLite iteration benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark SQLite dataset iteration")
    parser.add_argument("--db-dir", required=True, help="SQLite DB directory")
    parser.add_argument("--max-codes", type=int, default=0, help="0 = all codes")
    parser.add_argument("--min-bars", type=int, default=30)
    args = parser.parse_args()

    db_dir = Path(args.db_dir)
    db_files = sorted(
        p
        for p in db_dir.glob("*.db")
        if not p.name.startswith(".") and p.stat().st_size > 0
    )
    if args.max_codes > 0:
        db_files = db_files[: args.max_codes]

    print(f"[bench] {len(db_files)} codes to iterate")

    t0 = time.time()
    total_samples = 0
    total_bars = 0

    for i, db_path in enumerate(db_files):
        loaded = load_code(db_path)
        if loaded is None:
            continue
        temporal, ohlcv, date_epochs, unique_epochs = loaded
        n_bars = len(ohlcv)
        code_samples = 0

        for _ in iter_code(temporal, ohlcv, date_epochs, unique_epochs, args.min_bars):
            code_samples += 1

        total_samples += code_samples
        total_bars += n_bars

        if (i + 1) % 500 == 0 or i < 3:
            elapsed = time.time() - t0
            print(
                f"  [{i + 1}/{len(db_files)}] {db_path.stem}: "
                f"{n_bars} bars, {code_samples} samples "
                f"(cumulative {elapsed:.1f}s)"
            )

    elapsed = time.time() - t0
    print(f"\n[bench] Done in {elapsed:.2f}s")
    print(
        f"  {total_bars:,} bars, {total_samples:,} samples across {len(db_files)} codes"
    )
    print(f"  {total_samples / max(elapsed, 0.001):,.0f} samples/s")
    print(f"  {total_bars / max(elapsed, 0.001):,.0f} bars/s")


if __name__ == "__main__":
    main()
