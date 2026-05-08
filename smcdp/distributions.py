"""Data distributions and SDE limiting distributions on Riemannian manifolds.

Two roles:
  (1) Data distributions: source of training samples (ChartGaussianOnGraph,
      WrappedNormalS1, WrappedMixtureS1).
  (2) SDE limiting (reference) distributions: define the potential U whose
      gradient drives the Langevin SDE drift.  Each such distribution exposes
        .sample(n, device, dtype) -> (n, ambient_dim) on M
        .grad_U(x) -> (B, ambient_dim) tangent at x        (∇_M U(x) ∈ T_xM)
      This matches RSGM's `WrapNormDistribution` / `UniformDistribution` API.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from smcdp.manifolds import Sphere1, GraphManifold


class WrappedNormalS1:
    """Wrapped Gaussian on S^1 with chart mean μ (rad) and scale γ.

    Sample θ ~ N(μ, γ²), then wrap to (cos θ, sin θ) ∈ S^1.
    """

    def __init__(self, mean_angle: float = 0.0, scale: float = 0.5):
        self.mean = float(mean_angle)
        self.scale = float(scale)
        self.manifold = Sphere1()

    def sample(self, n: int, device=None, dtype=torch.float32) -> Tensor:
        theta = self.mean + self.scale * torch.randn(n, device=device, dtype=dtype)
        return self.manifold.from_angle(theta)


class WrappedMixtureS1:
    """Equally weighted mixture of K wrapped Gaussians on S^1.

    Default: 3 modes at angles {0, 2π/3, −2π/3}, scale 0.3 — a multi-modal
    target useful for stress-testing the model's mode-coverage.
    """

    def __init__(self, mean_angles: list[float] | None = None, scale: float = 0.3):
        if mean_angles is None:
            mean_angles = [0.0, 2.0 * math.pi / 3.0, -2.0 * math.pi / 3.0]
        self.means = torch.tensor(mean_angles)
        self.scale = float(scale)
        self.manifold = Sphere1()

    def sample(self, n: int, device=None, dtype=torch.float32) -> Tensor:
        means = self.means.to(device=device, dtype=dtype)
        idx = torch.randint(0, means.numel(), (n,), device=device)
        mu = means[idx]
        theta = mu + self.scale * torch.randn(n, device=device, dtype=dtype)
        return self.manifold.from_angle(theta)

    def marginal_score(self, x: Tensor, t: Tensor, schedule, n_wraps: int = 8) -> Tensor:
        """Analytic ground-truth Riemannian score ∇_M log p_t(x), in ambient form.

        For Brownian SDE on S^1 with rescaled time τ(t) = ∫_0^t β(s) ds, the marginal
        at time t is the convolution of the data mixture with the heat kernel:
            p_t(θ) = (1/K) Σ_k WrappedNormal(θ − μ_k ; σ²_data + τ(t))
        This is the exact target the DSM/score-matching objective should converge to.

        x : (B, 2) on S^1            t : (B,)
        schedule: BrownianSDE schedule providing integral(t).
        """
        sphere = self.manifold
        theta = sphere.to_angle(x)                                          # (B,)
        tau = schedule.integral(t)                                          # (B,)
        sigma2 = self.scale ** 2 + tau                                       # (B,) total variance

        means = self.means.to(theta.device, dtype=theta.dtype)              # (K,)
        wraps = torch.arange(-n_wraps, n_wraps + 1,
                             device=theta.device, dtype=theta.dtype)        # (W,)

        # shift[b, k, w] = θ_b − μ_k + 2π·w
        shifts = (theta.view(-1, 1, 1) - means.view(1, -1, 1)
                  + 2 * math.pi * wraps.view(1, 1, -1))                     # (B, K, W)
        s2 = sigma2.view(-1, 1, 1)                                           # (B, 1, 1)
        log_dens = -0.5 * shifts ** 2 / s2                                  # ignore (2πσ²)^{-1/2}, cancels in dp/p

        # use logsumexp for numerical stability
        L = log_dens.reshape(theta.shape[0], -1)                             # (B, K*W)
        L_max = L.amax(dim=-1, keepdim=True)
        weights = torch.exp(L - L_max)                                       # (B, K*W)
        d_factors = (-shifts / s2).reshape(theta.shape[0], -1)               # (B, K*W)

        p_unnorm = weights.sum(-1)                                           # (B,)
        dp_unnorm = (weights * d_factors).sum(-1)                            # (B,)
        score_chart = dp_unnorm / p_unnorm.clamp(min=1e-30)                  # (B,)

        return sphere.lift_chart_to_tangent(x, score_chart.unsqueeze(-1))    # (B, 2)


class WrappedNormalS1Marginal:
    """Helper that exposes WrappedNormalS1's analytic marginal score (single-mode case)."""

    def __init__(self, base: "WrappedNormalS1"):
        self.base = base

    def marginal_score(self, x: Tensor, t: Tensor, schedule, n_wraps: int = 8) -> Tensor:
        # Treat as 1-mode mixture for code reuse.
        proxy = WrappedMixtureS1(mean_angles=[self.base.mean], scale=self.base.scale)
        return proxy.marginal_score(x, t, schedule, n_wraps=n_wraps)


# -----------------------------------------------------------------------------
# Distributions on graph manifolds (Toy 2)
# -----------------------------------------------------------------------------


class ChartGaussianOnGraph:
    """Single-mode Gaussian in the chart q-space, lifted to a graph manifold via H.

    Data x = (q, F(q)) with q ∼ N(μ_q, σ²_q I).  Used as the data distribution for
    Toy 2 (graph manifold + analytic FK) — analogous to WrappedNormalS1 on Sphere1.
    """

    def __init__(self, manifold, mean_q: list[float] | torch.Tensor,
                 scale: float = 0.2):
        self.manifold = manifold
        if isinstance(mean_q, list):
            mean_q = torch.tensor(mean_q)
        self.mean_q = mean_q
        self.scale = float(scale)
        assert mean_q.numel() == manifold.n_q

    def sample(self, n: int, device=None, dtype=torch.float32) -> torch.Tensor:
        mean = self.mean_q.to(device=device, dtype=dtype)
        q = mean + self.scale * torch.randn(n, self.manifold.n_q, device=device, dtype=dtype)
        return self.manifold.H(q)


# =============================================================================
# SDE limiting distributions (RSGM-faithful API: sample + grad_U)
# =============================================================================


class UniformOnManifold:
    """Uniform measure on a compact manifold M (RSGM `UniformDistribution`).

    Used as the limiting distribution for Brownian SDE in the compact case
    (Sphere1, T^d, SO(3), ...).  Yields ∇_M U ≡ 0  →  zero drift.
    """

    def __init__(self, manifold):
        self.manifold = manifold

    def sample(self, n: int, device=None, dtype=torch.float32) -> Tensor:
        return self.manifold.random_uniform(n, device=device, dtype=dtype)

    def grad_U(self, x: Tensor) -> Tensor:
        return torch.zeros_like(x)


class WrappedNormalSphere1:
    """Wrapped Gaussian on S^1 — exact (G ≡ 1, so log det correction vanishes).

    Sample :  θ ~ N(μ, γ²),  return (cos θ, sin θ) ∈ S^1
    Grad U :  ∇_M U(x) = -log_x(μ) / γ²,   where U(x) = ½ d_M(x, μ)²/γ²

    On S^1 the geodesic Log is closed-form (Sphere1.log) and the volume form
    has constant √det G ≡ 1, so the wrapped Gaussian potential is just the
    squared-distance term — no log-det correction needed.
    """

    def __init__(self, mean_angle: float = 0.0, scale: float = 0.5):
        self.mean_angle = float(mean_angle)
        self.scale = float(scale)
        self.manifold = Sphere1()

    def sample(self, n: int, device=None, dtype=torch.float32) -> Tensor:
        theta = self.mean_angle + self.scale * torch.randn(n, device=device, dtype=dtype)
        return self.manifold.from_angle(theta)

    def grad_U(self, x: Tensor) -> Tensor:
        mu = self.manifold.from_angle(
            torch.tensor(self.mean_angle, device=x.device, dtype=x.dtype)
        ).expand_as(x)
        # ∇_M U = -log_x(μ) / γ²  (gradient of ½ d²(x, μ)/γ² on a Riemannian manifold)
        return -self.manifold.log(x, mu) / (self.scale ** 2)


class WrappedNormalGraph:
    """Exp-wrapped Gaussian on a graph manifold M = {(q, F(q))}.

    Construction (RSGM §3.1, exp-wrapped option;  Idea_formulation §4.1 default):
        Sample  v_q ~ N(0, γ² I)  in T_μM (chart at μ),  then  x = exp_μ(v_q),
        where exp_μ here is the graph retraction  v_q ↦ (μ_q + v_q, F(μ_q + v_q)).

    Resulting Riemannian density on M (w.r.t. dvol_M = √det G dq):
        p_R(x) ∝ N(q − μ_q ; 0, γ² I) / √det G(q)
    Equivalently, U(x) = -log p_R(x) (up to const) is
        U(x) = ½ ‖q − μ_q‖² / γ² + ½ log det G(q)
    The `+ ½ log det G(q)` is the manifold's `logdetexp` correction (matches
    RSGM's `WrapNormDistribution.U`).  Without it the chart marginal would be
    distorted by √det G.

    Riemannian gradient (lifted to ambient):
        ∇_M U(x) = J_H(q) · G⁻¹(q) · (∂_q U)
                 = J_H(q) · G⁻¹(q) · [(q − μ_q)/γ² + ½ ∂_q log det G(q)]

    The gradient of log det G is computed via autograd (jacrev on a scalar fn
    of q), so subclasses don't need closed forms.
    """

    def __init__(
        self,
        manifold: GraphManifold,
        mean_q: list[float] | torch.Tensor,
        scale: float = 0.5,
    ):
        self.manifold = manifold
        if isinstance(mean_q, list):
            mean_q = torch.tensor(mean_q)
        assert mean_q.numel() == manifold.n_q
        self.mean_q = mean_q
        self.scale = float(scale)

    def sample(self, n: int, device=None, dtype=torch.float32) -> Tensor:
        mu = self.mean_q.to(device=device, dtype=dtype)
        v_q = self.scale * torch.randn(n, self.manifold.n_q, device=device, dtype=dtype)
        return self.manifold.H(mu + v_q)

    def _grad_log_det_G_chart(self, q: Tensor) -> Tensor:
        """∂_q [½ log det G(q)]  via autograd, batched."""
        def _half_log_det_G(q_single: Tensor) -> Tensor:
            G = self.manifold.G(q_single.unsqueeze(0))[0]                  # (n_q, n_q)
            sign, logabsdet = torch.linalg.slogdet(G)
            return 0.5 * logabsdet                                          # scalar
        # vmap over the batch
        return torch.func.vmap(torch.func.jacrev(_half_log_det_G))(q)      # (B, n_q)

    def grad_U(self, x: Tensor) -> Tensor:
        """∇_M U(x) consistent with the manifold's metric mode.

        Both modes target the same chart marginal N(μ_q, γ² I) for the SDE
        invariant — the difference is whether the metric correction is applied:

          'riemannian'      : full induced-metric form
            U_R(q) = ½‖q-μ‖²/γ² + ½ log det G(q)
            ∂_q U_R = (q-μ)/γ² + ½ ∂_q log det G(q)
            ∇_M U_R = G⁻¹ · ∂_q U_R   (Riemannian gradient in chart)

          'chart_euclidean' : pretend G ≡ I (Idea §3.3 toy approximation)
            U_E(q) = ½‖q-μ‖²/γ²
            ∇ U_E = (q-μ)/γ²          (no log-det correction, no G⁻¹)
        """
        n_q = self.manifold.n_q
        q = x[..., :n_q]
        mu = self.mean_q.to(device=q.device, dtype=q.dtype)
        diff = (q - mu) / (self.scale ** 2)

        if self.manifold.metric == "riemannian":
            chart_grad_logdet = self._grad_log_det_G_chart(q)
            grad_q = diff + chart_grad_logdet
            L = self.manifold.G_chol(q)
            riem_grad_chart = torch.cholesky_solve(grad_q.unsqueeze(-1), L).squeeze(-1)
        else:                                                                # chart_euclidean
            riem_grad_chart = diff

        return self.manifold.lift_chart_to_tangent(x, riem_grad_chart)
