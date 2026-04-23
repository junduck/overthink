"""SQLite-backed dataset for BSQ training with cross-day context.

Reads per-code SQLite databases ({db_dir}/{code}.db), each containing
1-minute OHLCV bars with pre-computed temporal features. Yields
(ohlcv, timestamps, loss_mask) samples for online tokenization.

Same interface as BSQOnlineDataset but backed by SQLite for better
IO performance and inode efficiency.
"""

import random
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

_BARS_QUERY = (
    "SELECT minute_of_day, hour, dow, dom, month, "
    "open, high, low, close, volume, turnover "
    "FROM bars WHERE date_epoch = ? ORDER BY time_idx"
)


class BSQSQLiteDataset(IterableDataset):
    """Streaming dataset from per-code SQLite databases.

    Each sample = (prev_day + today) for one stock on one trading day.
    loss_mask is True for today's portion, False for prev_day context.

    Args:
        db_dir: Directory with {code}.db SQLite databases.
        min_bars: Minimum bars required in today's data.
        split: 'train', 'val', or 'all'.
        val_cutoff: Date string (YYYY-MM-DD) for train/val split.
    """

    def __init__(
        self,
        db_dir: str,
        min_bars: int = 30,
        split: str = "all",
        val_cutoff: str | None = None,
    ):
        if split in ("train", "val") and val_cutoff is None:
            raise ValueError("val_cutoff required when split is 'train' or 'val'")
        self.db_dir = Path(db_dir)
        self.min_bars = min_bars
        self.split = split
        self._val_epoch = (
            (date.fromisoformat(val_cutoff) - date(1970, 1, 1)).days
            if val_cutoff
            else 0
        )
        self._codes: list[str] = sorted(
            p.stem for p in self.db_dir.glob("*.db") if not p.name.startswith(".")
        )
        print(
            f"[BSQSQLiteDataset] {len(self._codes)} codes, "
            f"split={split}, val_cutoff={val_cutoff}"
        )

    def _get_dates(self, conn: sqlite3.Connection) -> list[int]:
        """Get distinct date_epoch values filtered by split."""
        rows = conn.execute(
            "SELECT DISTINCT date_epoch FROM bars ORDER BY date_epoch"
        ).fetchall()
        epochs = [r[0] for r in rows]
        if self.split == "train":
            return [e for e in epochs if e < self._val_epoch]
        if self.split == "val":
            return [e for e in epochs if e >= self._val_epoch]
        return epochs

    def _get_day_bars(
        self, conn: sqlite3.Connection, date_epoch: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Fetch one day's bars as (ohlcv [L,6], timestamps [L,5]) tensors."""
        rows = conn.execute(_BARS_QUERY, (date_epoch,)).fetchall()
        if not rows:
            return None
        arr = np.array(rows)
        timestamps = torch.tensor(arr[:, :5], dtype=torch.long)
        ohlcv = torch.tensor(arr[:, 5:], dtype=torch.float32)
        return ohlcv, timestamps

    def _iter_code(self, code: str):
        """Yield samples for one code with cross-day lookback."""
        db_path = self.db_dir / f"{code}.db"
        if not db_path.exists():
            return

        conn = sqlite3.connect(str(db_path), isolation_level=None)
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            date_epochs = self._get_dates(conn)
            prev_ohlcv: torch.Tensor | None = None
            prev_ts: torch.Tensor | None = None

            for epoch in date_epochs:
                result = self._get_day_bars(conn, epoch)
                if result is None:
                    prev_ohlcv = None
                    continue
                ohlcv, ts = result
                if len(ohlcv) < self.min_bars:
                    prev_ohlcv = None
                    continue

                if prev_ohlcv is not None:
                    yield (
                        torch.cat([prev_ohlcv, ohlcv]),
                        torch.cat([prev_ts, ts]),
                        torch.cat(
                            [
                                torch.zeros(len(prev_ohlcv), dtype=torch.bool),
                                torch.ones(len(ohlcv), dtype=torch.bool),
                            ]
                        ),
                    )

                prev_ohlcv = ohlcv
                prev_ts = ts
        finally:
            conn.close()

    def __iter__(self):
        codes = list(self._codes)
        random.shuffle(codes)
        for code in codes:
            yield from self._iter_code(code)
