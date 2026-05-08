"""Distributions for Toy 3 (embodiment-aware diffusion on a learned graph manifold).

Both data and limiting distributions sample (q, z_e) jointly:
    z_e ∼ Uniform[z_min, z_max]            (per sample, frozen during diffusion)
    q   ∼ N(μ_q, σ_q² I)                    (chart Gaussian)
    x   = (q, F_φ(q, z_e), z_e)             (lifted to learned manifold M_φ(z_e))

Each acts on x in the augmented form  (B, n_q + n_p + n_z) = (B, 5).
"""
from __future__ import annotations

import torch
from torch import Tensor

from smcdp.manifolds import EmbodimentGraphManifold


class _EmbodimentChartGauss:
    """Joint-sampler: z_e ~ Uniform, q ~ N(μ_q, σ²I), x = (q, F(q, z_e), z_e)."""

    def __init__(
        self,
        manifold: EmbodimentGraphManifold,
        mean_q: list[float] | torch.Tensor,
        scale_q: float,
        z_e_range: tuple[float, float] = (0.0, 0.3),
    ):
        self.manifold = manifold
        if isinstance(mean_q, list):
            mean_q = torch.tensor(mean_q)
        assert mean_q.numel() == manifold.n_q
        self.mean_q = mean_q
        self.scale_q = float(scale_q)
        self.z_lo, self.z_hi = z_e_range

    def _sample_z(self, n: int, device, dtype) -> Tensor:
        return self.z_lo + (self.z_hi - self.z_lo) * torch.rand(
            n, self.manifold.n_z, device=device, dtype=dtype
        )

    def sample(self, n: int, device=None, dtype=torch.float32, z_e=None) -> Tensor:
        mu = self.mean_q.to(device=device, dtype=dtype)
        q = mu + self.scale_q * torch.randn(n, self.manifold.n_q, device=device, dtype=dtype)
        if z_e is None:
            z_e = self._sample_z(n, device, dtype)
        return self.manifold.make_x(q, z_e)


class ChartGaussianOnEmbodiment(_EmbodimentChartGauss):
    """Data distribution for Toy 3 — chart-Gaussian in q + uniform in z_e."""

    pass


class WrappedNormalEmbodiment(_EmbodimentChartGauss):
    """Limiting distribution for Toy-3 Langevin SDE.

    Same joint chart-Gaussian × uniform-z_e as the data distribution, but with
    its own (typically wider) scale γ_lim, used to define the SDE's potential
    U(x) = ½ ‖q − μ_q‖² / γ² + ½ log det G(q, z_e).

    grad_U returns the Riemannian gradient lifted to the (q, p) tangent block;
    the z_e component is zero (z_e is frozen, not part of the SDE drift).
    """

    def __init__(
        self,
        manifold: EmbodimentGraphManifold,
        mean_q,
        scale: float = 1.0,
        z_e_range: tuple[float, float] = (0.0, 0.3),
    ):
        super().__init__(manifold, mean_q=mean_q, scale_q=scale, z_e_range=z_e_range)
        self.scale = float(scale)              # γ for the wrapped Gaussian potential

    def _grad_log_det_G_chart(self, q: Tensor, z: Tensor) -> Tensor:
        """∂_q [½ log det G(q, z_e)]  via autograd at fixed z."""
        def _half_logdet(q_s: Tensor, z_s: Tensor) -> Tensor:
            G = self.manifold.G(q_s.unsqueeze(0), z_s.unsqueeze(0))[0]
            return 0.5 * torch.linalg.slogdet(G).logabsdet
        return torch.func.vmap(torch.func.jacrev(_half_logdet, argnums=0))(q, z)

    def grad_U(self, x: Tensor) -> Tensor:
        n_q, n_p = self.manifold.n_q, self.manifold.n_p
        q = x[..., :n_q]
        z = x[..., n_q + n_p :]
        mu = self.mean_q.to(device=q.device, dtype=q.dtype)
        diff = (q - mu) / (self.scale ** 2)

        if self.manifold.metric == "riemannian":
            chart_grad_logdet = self._grad_log_det_G_chart(q, z)
            grad_q = diff + chart_grad_logdet
            L = self.manifold.G_chol(q, z)
            riem_grad_chart = torch.cholesky_solve(grad_q.unsqueeze(-1), L).squeeze(-1)
        else:
            riem_grad_chart = diff

        return self.manifold.lift_chart_to_tangent(x, riem_grad_chart)
