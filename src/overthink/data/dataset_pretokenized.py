"""Pre-tokenized dataset for BSQ training.

Loads per-code .pt files produced by pre_tokenize.py. Each file contains
all tokenized samples for one stock. The dataset streams one code at a time,
yielding individual (s1_ids, s2_ids, timestamps, loss_mask) samples.

Much faster than the online pipeline since tokenization is done offline.
"""

import random
from pathlib import Path

import torch
from torch.utils.data import IterableDataset


class BSQPreTokenizedDataset(IterableDataset):
    """Streaming dataset from pre-tokenized .pt files.

    Each .pt file (one per stock) contains:
        s1_ids:      [N, max_len] int16
        s2_ids:      [N, max_len] int16
        timestamps:  [N, max_len, 5] int16
        prev_lens:   [N] int16
        today_lens:  [N] int16
        dates:       list[str]

    Args:
        tokenized_dir: Directory with {code}.pt files.
        split: 'train', 'val', or 'all'.
        val_cutoff: Date string (YYYY-MM-DD) for train/val split.
    """

    def __init__(
        self,
        tokenized_dir: str,
        split: str = "all",
        val_cutoff: str | None = None,
    ):
        if split in ("train", "val") and val_cutoff is None:
            raise ValueError("val_cutoff required when split is 'train' or 'val'")
        self.tokenized_dir = Path(tokenized_dir)
        self.split = split
        self.val_cutoff = val_cutoff
        self._files: list[Path] = sorted(
            p for p in self.tokenized_dir.glob("*.pt") if not p.name.startswith(".")
        )
        print(
            f"[BSQPreTokenizedDataset] {len(self._files)} files, "
            f"split={split}, val_cutoff={val_cutoff}"
        )

    def _iter_file(self, path: Path):
        """Load one code's .pt file, yield individual samples."""
        data = torch.load(path, weights_only=True)
        s1_ids = data["s1_ids"].long()
        s2_ids = data["s2_ids"].long()
        timestamps = data["timestamps"].long()
        prev_lens = data["prev_lens"].long()
        today_lens = data["today_lens"].long()
        dates = data["dates"]

        for i in range(len(dates)):
            if self.split == "train" and dates[i] >= self.val_cutoff:
                continue
            elif self.split == "val" and dates[i] < self.val_cutoff:
                continue

            pl = prev_lens[i].item()
            tl = today_lens[i].item()
            seq_len = pl + tl

            loss_mask = torch.cat(
                [
                    torch.zeros(pl, dtype=torch.bool),
                    torch.ones(tl, dtype=torch.bool),
                ]
            )

            yield (
                s1_ids[i, :seq_len],
                s2_ids[i, :seq_len],
                timestamps[i, :seq_len],
                loss_mask,
            )

    def __iter__(self):
        files = list(self._files)
        random.shuffle(files)
        for f in files:
            yield from self._iter_file(f)
