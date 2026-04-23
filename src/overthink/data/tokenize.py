"""Online tokenization: normalize + frozen Kronos BSQ tokenizer.

Handles instance-level z-score normalization and tokenization in a single
callable, designed to run inside the training loop on GPU.

Also provides SIGRegOnlineTokenizer which wraps SIGRegTokenizer — no
z-score normalization needed since SIGReg handles distribution matching
in latent space.
"""

import sys
from pathlib import Path

import torch


def _load_kronos_tokenizer(tokenizer_path: str):
    """Load KronosTokenizer from vendored submodule."""
    kronos_root = str(Path(__file__).resolve().parents[3] / "extern" / "kronos")
    if kronos_root not in sys.path:
        sys.path.insert(0, kronos_root)
    from model.kronos import KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    tokenizer.eval()
    return tokenizer


class OnlineTokenizer:
    """Frozen Kronos tokenizer + instance-level z-score normalization.

    Args:
        tokenizer_path: HuggingFace model ID or local path.
        device: torch device.
        clip: Z-score clip range (default 5.0).
    """

    def __init__(
        self,
        tokenizer_path: str = "NeoQuasar/Kronos-Tokenizer-base",
        device: str = "cpu",
        clip: float = 5.0,
    ):
        self.device = device
        self.clip = clip
        self.tokenizer = _load_kronos_tokenizer(tokenizer_path)
        self.tokenizer = self.tokenizer.to(device)

    @torch.no_grad()
    def tokenize(self, ohlcv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize + tokenize OHLCV data.

        Args:
            ohlcv: [B, S, 6] raw OHLCV values

        Returns:
            (s1_ids, s2_ids) each [B, S] integer token IDs
        """
        mean = ohlcv.mean(dim=1, keepdim=True)
        std = ohlcv.std(dim=1, keepdim=True).clamp(min=1e-5)
        normalized = (ohlcv - mean) / std
        normalized = normalized.clamp(-self.clip, self.clip)

        s1_ids, s2_ids = self.tokenizer.encode(normalized, half=True)
        return s1_ids, s2_ids

    @torch.no_grad()
    def tokenize_with_stats(
        self, ohlcv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize + tokenize, also return normalization stats.

        Returns:
            (s1_ids, s2_ids, mean, std) — stats are [B, 1, 6] for denormalization.
        """
        mean = ohlcv.mean(dim=1, keepdim=True)
        std = ohlcv.std(dim=1, keepdim=True).clamp(min=1e-5)
        normalized = (ohlcv - mean) / std
        normalized = normalized.clamp(-self.clip, self.clip)

        s1_ids, s2_ids = self.tokenizer.encode(normalized, half=True)
        return s1_ids, s2_ids, mean, std

    @torch.no_grad()
    def decode(
        self,
        s1_ids: torch.Tensor,
        s2_ids: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """Decode tokens back to raw OHLCV space.

        Args:
            s1_ids: [B, S] coarse subtoken IDs
            s2_ids: [B, S] fine subtoken IDs
            mean: [B, 1, 6] normalization mean
            std: [B, 1, 6] normalization std

        Returns:
            [B, S, 6] reconstructed OHLCV in original scale
        """
        reconstructed = self.tokenizer.decode((s1_ids, s2_ids), half=True)
        return reconstructed * std + mean


class SIGRegOnlineTokenizer:
    """Frozen SIGRegTokenizer — no z-score normalization.

    SIGReg forces encoder latents to N(0, I), so raw input is tokenized
    directly. Same raw candle always gets the same token.

    Args:
        tokenizer: A trained SIGRegTokenizer instance.
        device: torch device.
    """

    def __init__(self, tokenizer, device: str = "cpu"):
        self.device = device
        self.tokenizer = tokenizer.to(device)
        self.tokenizer.eval()

    @torch.no_grad()
    def tokenize(self, ohlcv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize raw OHLCV data (no normalization).

        Args:
            ohlcv: [B, S, 6] raw OHLCV values.

        Returns:
            (s1_ids, s2_ids) each [B, S] integer token IDs.
        """
        s1_ids, s2_ids = self.tokenizer.encode(ohlcv, half=True)
        return s1_ids, s2_ids

    @torch.no_grad()
    def decode(
        self,
        s1_ids: torch.Tensor,
        s2_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Decode tokens back to OHLCV space.

        Args:
            s1_ids: [B, S] coarse subtoken IDs.
            s2_ids: [B, S] fine subtoken IDs.

        Returns:
            [B, S, 6] reconstructed OHLCV.
        """
        return self.tokenizer.decode((s1_ids, s2_ids), half=True)
