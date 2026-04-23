"""Sketched Isotropic Gaussian Regularization (SIGReg).

From LeJEPA (Balestriero & LeCun, 2025, arXiv:2511.08544).

Forces a batch of embeddings toward N(0, I) by:
  1. Projecting onto M random 1D directions (slicing)
  2. Computing the Epps-Pulley characteristic-function test on each slice
  3. Averaging the test statistics into a single differentiable loss

When SIGReg is active, sign() quantization yields uniformly distributed
binary codes by construction — no entropy or commitment regularization
needed.
"""

import torch
from torch import nn

try:
    from torch.distributed._functional_collectives import all_reduce as _all_reduce

    def _maybe_all_reduce(x: torch.Tensor, op: str = "avg") -> torch.Tensor:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return _all_reduce(x, op.lower(), torch.distributed.group.WORLD)
        return x
except ImportError:

    def _maybe_all_reduce(x: torch.Tensor, op: str = "avg") -> torch.Tensor:
        return x


class EppsPulley1D(nn.Module):
    """Epps-Pulley goodness-of-fit test against N(0, 1) via characteristic functions.

    Statistic:
        T = N * integral_0^{t_max} |phi_hat(t) - phi(t)|^2 w(t) dt

    where phi_hat is the empirical CF, phi(t) = exp(-t^2/2), and we exploit
    symmetry so only t >= 0 are computed with doubled weights.

    Args:
        t_max: Upper integration limit.
        n_points: Number of trapezoidal quadrature points (odd recommended).
    """

    def __init__(self, t_max: float = 5.0, n_points: int = 17):
        super().__init__()
        assert n_points % 2 == 1, "n_points must be odd"
        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        dt = t_max / (n_points - 1)
        phi = t.square().mul(0.5).neg().exp()
        weights = torch.full((n_points,), 2 * dt)
        weights[0] = dt
        weights[-1] = dt
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Epps-Pulley statistic.

        Args:
            x: [N] univariate samples.

        Returns:
            Scalar test statistic.
        """
        x_t = x.unsqueeze(-1) * self.t
        cos_mean = torch.cos(x_t).mean(0)
        sin_mean = torch.sin(x_t).mean(0)
        cos_mean = _maybe_all_reduce(cos_mean)
        sin_mean = _maybe_all_reduce(sin_mean)
        err = (cos_mean - self.phi).square() + sin_mean.square()
        N = x.numel()
        ws = (
            torch.distributed.get_world_size()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 1
        )
        return (err @ self.weights) * N * ws


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization.

    Projects embeddings z in R^D onto M random unit directions, applies
    Epps-Pulley1D to each, averages.  Minimizing this loss forces z ~ N(0, I).

    Args:
        embed_dim: Dimensionality D of the embedding space.
        num_slices: Number M of random 1D projections per step.
        t_max: Integration limit for Epps-Pulley.
        n_points: Quadrature points for Epps-Pulley.
    """

    def __init__(
        self,
        embed_dim: int,
        num_slices: int = 256,
        t_max: float = 5.0,
        n_points: int = 17,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_slices = num_slices
        self.ep = EppsPulley1D(t_max=t_max, n_points=n_points)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self._gen: torch.Generator | None = None
        self._gen_device: torch.device | None = None

    def _get_generator(self, device: torch.device, seed: int) -> torch.Generator:
        if self._gen is None or self._gen_device != device:
            self._gen = torch.Generator(device=device)
            self._gen_device = device
        self._gen.manual_seed(seed)
        return self._gen

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute SIGReg loss.

        Args:
            z: [..., D] embeddings, typically [B, T, D].

        Returns:
            Scalar loss (mean Epps-Pulley statistic across slices).
        """
        z_flat = z.reshape(-1, self.embed_dim)
        N, D = z_flat.shape

        with torch.no_grad():
            seed = self.global_step.item()
            g = self._get_generator(z.device, seed)
            A = torch.randn(D, self.num_slices, device=z.device, generator=g)
            A = A / A.norm(p=2, dim=0, keepdim=True)
            self.global_step.add_(1)

        projected = z_flat @ A
        stats = torch.stack(
            [self.ep(projected[:, s]) for s in range(self.num_slices)]
        )
        return stats.mean()
