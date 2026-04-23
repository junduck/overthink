"""Dataset for SIGReg tokenizer training from per-code SQLite databases.

Loads all bars for each code from {sqlite_dir}/{code}.db, then extracts
sliding windows of size `window_size` with stride `stride`.

Each sample is a raw [W, 6] OHLCV tensor — no normalization.

Windows from the same code are grouped and padded to align with batch_size,
so no batch ever mixes bars from different codes.

Args:
    sqlite_dir: Directory with {code}.db files.
    window_size: Sliding window length (default 200).
    stride: Sliding window stride (default 30).
    batch_size: DataLoader batch size, used to align code boundaries (default 128).
    max_codes: If set, randomly subsample this many codes.
    val_cutoff: Date string (YYYY-MM-DD) for train/val split on date_epoch.
    split: 'train', 'val', or 'all'.
"""

import random
import sqlite3
from pathlib import Path

import torch
from torch.utils.data import IterableDataset

_OHLCV_COLS = "open, high, low, close, volume, turnover"


class SlidingWindowDataset(IterableDataset):
    """Streaming dataset: per-code SQLite → sliding windows of raw OHLCV.

    Windows from each code are truncated to a multiple of batch_size so that
    no batch contains windows from different codes.
    """

    def __init__(
        self,
        sqlite_dir: str,
        window_size: int = 200,
        stride: int = 30,
        batch_size: int = 128,
        max_codes: int | None = None,
        split: str = "all",
        val_cutoff: str | None = None,
    ):
        self.sqlite_dir = Path(sqlite_dir)
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.split = split
        self.val_cutoff = val_cutoff

        self._db_files = sorted(
            f
            for f in self.sqlite_dir.glob("*.db")
            if not f.name.startswith(".")
        )

        if max_codes and len(self._db_files) > max_codes:
            random.shuffle(self._db_files)
            self._db_files = sorted(self._db_files[:max_codes])

        print(
            f"[SlidingWindowDataset] {len(self._db_files)} codes, "
            f"window={window_size}, stride={stride}, batch_size={batch_size}, split={split}"
        )

    def _load_code(self, db_path: Path) -> torch.Tensor | None:
        """Load all OHLCV bars for one code. Returns [N, 6] or None."""
        conn = sqlite3.connect(str(db_path))

        if self.split != "all" and self.val_cutoff:
            cutoff_epoch = self._date_to_epoch(self.val_cutoff)
            if self.split == "train":
                where = f"WHERE date_epoch < {cutoff_epoch}"
            else:
                where = f"WHERE date_epoch >= {cutoff_epoch}"
        else:
            where = ""

        rows = conn.execute(f"SELECT {_OHLCV_COLS} FROM bars {where}").fetchall()
        conn.close()

        if len(rows) < self.window_size:
            return None

        return torch.tensor([list(r) for r in rows], dtype=torch.float32)

    @staticmethod
    def _date_to_epoch(date_str: str) -> int:
        """Convert YYYY-MM-DD to approximate date_epoch for filtering."""
        from datetime import datetime
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.timestamp() / 86400)

    def _windows(self, data: torch.Tensor):
        """Yield sliding windows from [N, 6] tensor, aligned to batch_size."""
        N = data.size(0)
        W = self.window_size
        S = self.stride
        n_windows = (N - W) // S + 1
        n_aligned = (n_windows // self.batch_size) * self.batch_size
        for i in range(n_aligned):
            start = i * S
            yield data[start : start + W]

    def __iter__(self):
        db_files = list(self._db_files)
        random.shuffle(db_files)
        for db_path in db_files:
            data = self._load_code(db_path)
            if data is not None:
                yield from self._windows(data)
