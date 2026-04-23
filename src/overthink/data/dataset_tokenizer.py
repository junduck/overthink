"""Dataset for SIGReg tokenizer training on market-wide parquet files.

Loads per-date files from {market_dir}/{date}-ohlcv-1m.pq containing all
stocks for one trading day. Builds cross-day windows (prev_day + today) per
code. Yields raw OHLCV tensors — no normalization.

Designed to be fast on GPU workstations by pre-loading a configurable subset
of codes into memory.
"""

import random
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import IterableDataset

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume", "Turnover"]


def extract_timestamps(ts_col: pl.Series) -> torch.Tensor:
    """Extract 5 temporal features from a polars datetime column.

    Returns:
        [L, 5] long tensor: (minute_of_day, hour_of_day, day_of_week, day_of_month, month)
    """
    dt = ts_col.dt
    hour = dt.hour().cast(pl.Int32)
    minute = dt.minute().cast(pl.Int32)
    minute_of_day = hour * 60 + minute
    return torch.stack(
        [
            torch.tensor(minute_of_day.to_list(), dtype=torch.long),
            torch.tensor(hour.to_list(), dtype=torch.long),
            torch.tensor(dt.weekday().cast(pl.Int32).to_list(), dtype=torch.long) - 1,
            torch.tensor(dt.day().cast(pl.Int32).to_list(), dtype=torch.long) - 1,
            torch.tensor(dt.month().cast(pl.Int32).to_list(), dtype=torch.long) - 1,
        ],
        dim=-1,
    )


def df_to_tensors(df: pl.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert OHLCV polars DataFrame to (ohlcv, timestamps) tensors."""
    df = df.sort("TimeInterval")
    ohlcv = torch.tensor(df[OHLCV_COLS].to_numpy(), dtype=torch.float32)
    ts = extract_timestamps(df["TimeInterval"])
    return ohlcv, ts


class TokenizerDataset(IterableDataset):
    """Streaming dataset for tokenizer training from market-wide parquet files.

    Loads daily files, partitions by code, builds prev_day+today windows.
    Optionally subsamples codes for faster iteration.

    Args:
        market_dir: Directory with {date}-ohlcv-1m.pq files.
        min_bars: Minimum bars per day to include.
        max_codes: If set, randomly subsample this many codes.
        split: 'train', 'val', or 'all'.
        val_cutoff: Date string for train/val split.
    """

    def __init__(
        self,
        market_dir: str,
        min_bars: int = 30,
        max_codes: int | None = None,
        max_dates: int | None = None,
        split: str = "all",
        val_cutoff: str | None = None,
    ):
        if split in ("train", "val") and val_cutoff is None:
            raise ValueError("val_cutoff required when split is 'train' or 'val'")
        self.market_dir = Path(market_dir)
        self.min_bars = min_bars
        self.max_codes = max_codes
        self.split = split
        self.val_cutoff = val_cutoff

        self._files = sorted(
            f
            for f in self.market_dir.glob("*.pq")
            if not f.name.startswith(".")
        )
        if max_dates:
            self._files = self._files[:max_dates]

        self._codes = self._discover_codes()
        if max_codes and len(self._codes) > max_codes:
            random.shuffle(self._codes)
            self._codes = sorted(self._codes[:max_codes])

        print(
            f"[TokenizerDataset] {len(self._codes)} codes, "
            f"{len(self._files)} dates, split={split}"
        )

    def _discover_codes(self) -> list[str]:
        """Read the first available file to get code list."""
        for f in self._files[:3]:
            df = pl.read_parquet(f, columns=["Code"])
            return sorted(df["Code"].unique().to_list())
        return []

    def _files_for_split(self) -> list[Path]:
        if self.split == "train":
            return [f for f in self._files if f.stem[:10] < self.val_cutoff]
        elif self.split == "val":
            return [f for f in self._files if f.stem[:10] >= self.val_cutoff]
        return list(self._files)

    def __iter__(self):
        codes = list(self._codes)
        random.shuffle(codes)
        files = self._files_for_split()

        prev_day: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        for f in files:
            path = f
            df = pl.read_parquet(path)
            if df.is_empty():
                prev_day.clear()
                continue

            partitioned = df.partition_by("Code", as_dict=True)

            for code in codes:
                key = (code,)
                if key not in partitioned:
                    if code in prev_day:
                        del prev_day[code]
                    continue

                code_df = partitioned[key].sort("TimeInterval")
                if len(code_df) < self.min_bars:
                    if code in prev_day:
                        del prev_day[code]
                    continue

                today_ohlcv, today_ts = df_to_tensors(code_df)

                if code in prev_day:
                    prev_ohlcv, prev_ts = prev_day[code]
                    ohlcv = torch.cat([prev_ohlcv, today_ohlcv], dim=0)
                    timestamps = torch.cat([prev_ts, today_ts], dim=0)
                    yield ohlcv, timestamps

                prev_day[code] = (today_ohlcv, today_ts)
