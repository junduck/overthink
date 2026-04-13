"""Online dataset for BSQ training with cross-day context.

Loads raw OHLCV .pq files organized as {data_dir}/{code}/{date}.pq.
Each sample = (yesterday full day) + (today full day), yielding raw bars
and temporal features. Tokenization happens in the training loop.

Uses polars for fast .pq reads. Iterates by code in sequential date order
(shuffled across codes), keeping a one-day lookback buffer so each file
is read exactly once.
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
    minute_of_day = dt.hour() * 60 + dt.minute()
    return torch.stack(
        [
            torch.tensor(minute_of_day.to_list(), dtype=torch.long),
            torch.tensor(dt.hour().to_list(), dtype=torch.long),
            torch.tensor(dt.weekday().to_list(), dtype=torch.long) - 1,
            torch.tensor(dt.day().to_list(), dtype=torch.long) - 1,
            torch.tensor(dt.month().to_list(), dtype=torch.long) - 1,
        ],
        dim=-1,
    )


def df_to_tensors(df: pl.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert OHLCV polars DataFrame to (ohlcv, timestamps) tensors."""
    df = df.sort("TimeInterval")
    ohlcv = torch.tensor(df[OHLCV_COLS].to_numpy(), dtype=torch.float32)
    ts = extract_timestamps(df["TimeInterval"])
    return ohlcv, ts


def _safe_read(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        return None
    df = pl.read_parquet(path)
    return df if not df.is_empty() else None


class BSQOnlineDataset(IterableDataset):
    """Streaming dataset that yields raw (ohlcv, timestamps, loss_mask) samples.

    Each sample = (prev_day + today) for one stock on one date.
    loss_mask is True for today's portion, False for prev_day (context only).

    Iterates codes in random order, dates in sequential order within each
    code. Keeps a one-day lookback buffer so each .pq file is read exactly
    once per epoch.

    Args:
        data_dir: Root directory with per-code layout ({code}/{date}.pq).
        min_bars: Minimum bars required in today's data.
        split: 'train', 'val', or 'all'. If 'train'/'val', dates are split
            at val_cutoff. 'val' uses dates >= val_cutoff, 'train' uses dates
            before val_cutoff. 'all' uses all dates.
        val_cutoff: Date string (YYYY-MM-DD) for train/val split. Dates >=
            this are val. Required when split is 'train' or 'val'.
    """

    def __init__(
        self,
        data_dir: str,
        min_bars: int = 30,
        split: str = "all",
        val_cutoff: str | None = None,
    ):
        if split in ("train", "val") and val_cutoff is None:
            raise ValueError("val_cutoff required when split is 'train' or 'val'")
        self.data_dir = Path(data_dir)
        self.min_bars = min_bars
        self.split = split
        self.val_cutoff = val_cutoff
        self._codes: list[str] = sorted(
            d.name for d in self.data_dir.iterdir() if d.is_dir()
        )
        print(
            f"[BSQOnlineDataset] {len(self._codes)} codes, split={split}, val_cutoff={val_cutoff}"
        )

    def __len__(self) -> int:
        return -1

    def _dates_for(self, code: str) -> list[str]:
        code_dir = self.data_dir / code
        dates = sorted(p.stem for p in code_dir.glob("*.pq"))
        if self.split == "train":
            return [d for d in dates if d < self.val_cutoff]
        elif self.split == "val":
            return [d for d in dates if d >= self.val_cutoff]
        return dates

    def _iter_code(self, code: str):
        """Yield samples for one code with sequential lookback."""
        dates = self._dates_for(code)
        prev_ohlcv: torch.Tensor | None = None
        prev_ts: torch.Tensor | None = None

        for date in dates:
            path = self.data_dir / code / f"{date}.pq"
            df = _safe_read(path)
            if df is None or len(df) < self.min_bars:
                prev_ohlcv = None
                continue

            today_ohlcv, today_ts = df_to_tensors(df)

            if prev_ohlcv is not None:
                ohlcv = torch.cat([prev_ohlcv, today_ohlcv], dim=0)
                timestamps = torch.cat([prev_ts, today_ts], dim=0)
                loss_mask = torch.cat(
                    [
                        torch.zeros(len(prev_ohlcv), dtype=torch.bool),
                        torch.ones(len(today_ohlcv), dtype=torch.bool),
                    ]
                )
                yield ohlcv, timestamps, loss_mask

            prev_ohlcv = today_ohlcv
            prev_ts = today_ts

    def __iter__(self):
        codes = list(self._codes)
        random.shuffle(codes)
        for code in codes:
            yield from self._iter_code(code)
