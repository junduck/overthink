from typing import Literal

import torch
from torch import nn


class MultiScaleTrendLoss(nn.Module):
    def __init__(self, alphas: list[float], weights: list[float], reduction: Literal["mean", "sum", "none"] = "mean"):
        if (len(alphas) != len(weights)):
            raise ValueError("alphas and weights must have the same length.")
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
    def __init__(self,
                 alphas: list[float],
                 weights: list[float],
                 reduction: Literal["mean", "sum", "none"],
                 loss_type: Literal["hinge", "huber", "cosine", "softsign"],
                 huber_margin: float,
                 softness: float,
                 norm_eps: float):
        if (len(alphas) != len(weights)):
            raise ValueError("alphas and weights must have the same length.")
        super().__init__()

        self.alphas = alphas
        self.weights = weights
        self.reduction = reduction
        self.loss_type = loss_type
        self.huber_margin = huber_margin
        self.softness = softness
        self.norm_eps = norm_eps

    def _ema(self, x: torch.Tensor, dim: int, alpha: float) -> torch.Tensor:
        """Compute Exponential Moving Average (EMA) along a specified dimension."""
        ema = torch.empty_like(x)
        ema.select(dim, 0).copy_(x.select(dim, 0))
        for i in range(1, x.size(dim)):
            prev = ema.select(dim, i - 1)
            cur = x.select(dim, i)
            ema.select(dim, i).copy_(alpha * cur + (1 - alpha) * prev)
        return ema

    def _hinge_loss(self,
                    pred_delta: torch.Tensor,      # [B, S-1]
                    target_delta: torch.Tensor,    # [B, S-1]
                    ) -> torch.Tensor:
        dir_agreement = pred_delta * target_delta
        scale_loss = torch.clamp(-dir_agreement, min=0).mean(dim=1)  # [B]
        return scale_loss  # [B]

    def _huber_loss(self,
                    pred_delta: torch.Tensor,      # [B, S-1]
                    target_delta: torch.Tensor,    # [B, S-1]
                    ) -> torch.Tensor:
        dir_agreement = pred_delta * target_delta
        step_err = (pred_delta - target_delta).abs()
        scale_loss = torch.where(
            dir_agreement < 0,
            step_err,
            torch.clamp(self.huber_margin - dir_agreement,
                        min=0).pow(2) / (2 * self.huber_margin)
        ).mean(dim=1)  # [B]
        return scale_loss  # [B]

    def _cosine_loss(self,
                     pred_delta: torch.Tensor,      # [B, S-1]
                     target_delta: torch.Tensor,    # [B, S-1]
                     ) -> torch.Tensor:
        pred_delta_norm = pred_delta / (pred_delta.abs() + self.norm_eps)
        target_delta_norm = target_delta / (target_delta.abs() + self.norm_eps)
        # Cosine similarity: (-1, 1) -> scale loss: (0, 2)
        cos_sim = (pred_delta_norm * target_delta_norm).mean(dim=1)
        scale_loss = 1 - cos_sim
        return scale_loss

    def _softsign_loss(self,
                       pred_delta: torch.Tensor,      # [B, S-1]
                       target_delta: torch.Tensor,    # [B, S-1]
                       ) -> torch.Tensor:
        # Soft sign agreement using tanh
        step_err = (pred_delta - target_delta).abs()
        soft_agreement = torch.tanh(
            pred_delta / self.softness) * torch.tanh(target_delta / self.softness)
        scale_loss = ((1 - soft_agreement) * step_err).mean(dim=1)
        return scale_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss on direction mismatches.

        Args:
            pred: [B, S] predicted values
            target: [B, S] target values

        Returns:
            scalar if reduction='mean'/'sum', [B] if reduction='none'
        """
        losses = []  # will hold [B] tensors
        for alpha, weight in zip(self.alphas, self.weights):
            pred_ema = self._ema(pred, dim=1, alpha=alpha)      # [B, S]
            target_ema = self._ema(target, dim=1, alpha=alpha)  # [B, S]

            # First differences
            pred_delta = pred_ema[:, 1:] - pred_ema[:, :-1]        # [B, S-1]
            target_delta = target_ema[:, 1:] - target_ema[:, :-1]  # [B, S-1]

            if self.loss_type == "hinge":
                scale_loss = self._hinge_loss(pred_delta, target_delta)
            elif self.loss_type == "huber":
                scale_loss = self._huber_loss(pred_delta, target_delta)
            elif self.loss_type == "cosine":
                scale_loss = self._cosine_loss(pred_delta, target_delta)
            elif self.loss_type == "softsign":
                scale_loss = self._softsign_loss(pred_delta, target_delta)
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")
            losses.append(weight * scale_loss)  # [B]

        loss = torch.stack(losses, dim=0).sum(dim=0)  # [num_scales, B] -> [B]
        if self.reduction == "mean":
            return loss.mean()  # scalar
        if self.reduction == "sum":
            return loss.sum()   # scalar
        return loss  # [B]
