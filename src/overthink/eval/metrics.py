"""Evaluation metrics for OverthinkBSQ.

Metrics:
    - Perplexity: exp(CE loss) on held-out data
    - Top-k accuracy: fraction of true tokens in model's top-k predictions
    - Directional accuracy: does predicted close direction match actual
    - Return correlation: Pearson correlation between predicted and actual returns
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_perplexity(
    s1_logits: torch.Tensor,
    s2_logits: torch.Tensor,
    s1_target: torch.Tensor,
    s2_target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> float:
    """Compute perplexity from logits and targets.

    Args:
        s1_logits: [B, S, V]
        s2_logits: [B, S, V]
        s1_target: [B, S]
        s2_target: [B, S]
        valid_mask: [B, S] boolean

    Returns:
        Perplexity (float)
    """
    _B, _S, V = s1_logits.shape
    flat = valid_mask.reshape(-1)

    ce_s1 = F.cross_entropy(
        s1_logits.reshape(-1, V), s1_target.reshape(-1), reduction="none"
    )
    ce_s2 = F.cross_entropy(
        s2_logits.reshape(-1, V), s2_target.reshape(-1), reduction="none"
    )
    avg_ce = (ce_s1[flat] + ce_s2[flat]).mean()

    return torch.exp(avg_ce).item()


@torch.no_grad()
def compute_topk_accuracy(
    s1_logits: torch.Tensor,
    s2_logits: torch.Tensor,
    s1_target: torch.Tensor,
    s2_target: torch.Tensor,
    valid_mask: torch.Tensor,
    k: int = 10,
) -> dict[str, float]:
    """Compute top-k accuracy for both subtoken heads.

    Returns:
        Dict with s1_top{k}, s2_top{k}, mean_top{k}
    """
    flat_mask = valid_mask.reshape(-1)

    s1_top = s1_logits.reshape(-1, s1_logits.size(-1)).topk(k, dim=-1).indices
    s1_correct = (s1_top == s1_target.reshape(-1).unsqueeze(-1)).any(dim=-1)
    s1_acc = s1_correct[flat_mask].float().mean().item()

    s2_top = s2_logits.reshape(-1, s2_logits.size(-1)).topk(k, dim=-1).indices
    s2_correct = (s2_top == s2_target.reshape(-1).unsqueeze(-1)).any(dim=-1)
    s2_acc = s2_correct[flat_mask].float().mean().item()

    return {
        f"s1_top{k}": s1_acc,
        f"s2_top{k}": s2_acc,
        f"mean_top{k}": (s1_acc + s2_acc) / 2,
    }


@torch.no_grad()
def compute_directional_accuracy(
    pred_ohlcv: torch.Tensor,
    actual_ohlcv: torch.Tensor,
    valid_mask: torch.Tensor,
) -> float:
    """Directional accuracy on close price changes.

    For each position, compare sign of predicted vs actual close-to-close return.

    Args:
        pred_ohlcv: [B, S, 6] predicted OHLCV (col 3 = close)
        actual_ohlcv: [B, S, 6] actual OHLCV
        valid_mask: [B, S] boolean

    Returns:
        Fraction of positions where predicted direction matches actual.
    """
    pred_ret = pred_ohlcv[:, 1:, 3] - pred_ohlcv[:, :-1, 3]
    actual_ret = actual_ohlcv[:, 1:, 3] - actual_ohlcv[:, :-1, 3]

    mask = valid_mask[:, 1:]
    if mask.sum() == 0:
        return 0.0

    same_sign = ((pred_ret * actual_ret) > 0).float()
    return same_sign[mask].mean().item()


@torch.no_grad()
def compute_return_correlation(
    pred_ohlcv: torch.Tensor,
    actual_ohlcv: torch.Tensor,
    valid_mask: torch.Tensor,
) -> float:
    """Pearson correlation between predicted and actual close-to-close returns.

    Args:
        pred_ohlcv: [B, S, 6] predicted OHLCV (col 3 = close)
        actual_ohlcv: [B, S, 6] actual OHLCV
        valid_mask: [B, S] boolean

    Returns:
        Pearson r (float), or 0.0 if undefined.
    """
    pred_ret = (pred_ohlcv[:, 1:, 3] - pred_ohlcv[:, :-1, 3]).reshape(-1)
    actual_ret = (actual_ohlcv[:, 1:, 3] - actual_ohlcv[:, :-1, 3]).reshape(-1)
    mask = valid_mask[:, 1:].reshape(-1)

    pred_ret = pred_ret[mask]
    actual_ret = actual_ret[mask]

    if len(pred_ret) < 2:
        return 0.0

    pred_centered = pred_ret - pred_ret.mean()
    actual_centered = actual_ret - actual_ret.mean()

    denom = pred_centered.norm() * actual_centered.norm()
    if denom < 1e-8:
        return 0.0

    return (pred_centered @ actual_centered / denom).item()
