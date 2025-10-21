from typing import Literal

import torch
from torch import nn


class MultiScaleTrendLoss(nn.Module):
    def __init__(self, alphas: list[float], weights: list[float], reduction: Literal["mean", "sum", "none"] = "mean"):
        super().__init__()
        self.alphas = alphas
        self.weights = weights
        self.reduction = reduction

    def _ema(self, x: torch.Tensor, dim: int, alpha: float) -> torch.Tensor:
        """Compute Exponential Moving Average (EMA) along a specified dimension."""
        ema = torch.empty_like(x)
        ema.select(dim, 0).copy_(x.select(dim, 0))
        for i in range(1, x.size(dim)):
            prev = ema.select(dim, i - 1)
            cur = x.select(dim, i)
            ema.select(dim, i).copy_(alpha * cur + (1 - alpha) * prev)
        return ema

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for alpha, weight in zip(self.alphas, self.weights):
            pred_ema = self._ema(pred, dim=1, alpha=alpha)
            target_ema = self._ema(target, dim=1, alpha=alpha)
            mse = (pred_ema - target_ema).pow(2).mean(dim=1)        # [B]
            losses.append(weight * mse)
        loss = torch.stack(losses, dim=0).sum(dim=0)                # [B]
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class MultiScaleTrendDirectionLoss(nn.Module):
    def __init__(self, alphas: list[float], weights: list[float], reduction: Literal["mean", "sum", "none"] = "mean"):
        super().__init__()
        self.alphas = alphas
        self.weights = weights
        self.reduction = reduction

    def _ema(self, x: torch.Tensor, dim: int, alpha: float) -> torch.Tensor:
        """Compute Exponential Moving Average (EMA) along a specified dimension."""
        ema = torch.empty_like(x)
        ema.select(dim, 0).copy_(x.select(dim, 0))
        for i in range(1, x.size(dim)):
            prev = ema.select(dim, i - 1)
            cur = x.select(dim, i)
            ema.select(dim, i).copy_(alpha * cur + (1 - alpha) * prev)
        return ema

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss on direction mismatches."""
        losses = []
        for alpha, weight in zip(self.alphas, self.weights):
            pred_ema = self._ema(pred, dim=1, alpha=alpha)      # [B, S]
            target_ema = self._ema(target, dim=1, alpha=alpha)  # [B, S]

            # First differences
            pred_delta = pred_ema[:, 1:] - pred_ema[:, :-1]        # [B, S-1]
            target_delta = target_ema[:, 1:] - target_ema[:, :-1]  # [B, S-1]

            pred_sign = torch.sign(pred_delta)
            target_sign = torch.sign(target_delta)

            # Direction mismatch mask (True where signs differ)
            mismatch = pred_sign != target_sign                   # [B, S-1]

            # Squared error per timestep (broadcast compatible)
            # Use centered error of EMA values at current timestep index (1..S-1)
            step_err = (pred_ema[:, 1:] - target_ema[:, 1:]).pow(2)  # [B, S-1]

            masked_err = torch.where(mismatch, step_err, torch.zeros_like(
                step_err))  # zero if same direction

            # Mean over time axis; if no mismatches, mean of all zeros => 0.
            scale_loss = masked_err.mean(dim=1)  # [B]
            losses.append(weight * scale_loss)

        loss = torch.stack(losses, dim=0).sum(dim=0)  # [B]
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
