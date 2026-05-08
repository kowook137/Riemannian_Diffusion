"""Forward SDEs on Riemannian manifolds — RSGM-faithful structure.

Mirrors riemannian-score-sde/riemannian_score_sde/sde.py:
    SDE base  ─►  Langevin (drift = -½ β ∇_M U)  ─►  Brownian (limiting=Uniform → ∇U=0)

Mathematical contract (RSGM §3.1, Idea_formulation §4.1):
    Forward SDE :  dX_t = -½ β(t) ∇_M U(X_t) dt + √β(t) dB^M_t,    X_0 ∼ p_data
    Reverse SDE :  dY_τ = {-b(Y_τ) + β(t) ∇_M log p_t(Y_τ)} dτ + √β(t) dB̃^M_τ
                                                                  (Theorem 1)

The potential U is determined by the limiting distribution attached to the SDE:
    UniformOnManifold       (compact M):     ∇_M U ≡ 0   →  Brownian (target uniform)
    WrappedNormal(μ, γ):    (any M):         ∇_M U = ∇_M [½ d_M(·,μ)²/γ² + ½ log det G]
                                                  →  Langevin targeting wrapped Gaussian
                                              (RSGM §3.1, exp-wrapped option;
                                              Idea §4.1 "bounded workspace default")
"""
from __future__ import annotations

import torch
from torch import Tensor

from smcdp.manifolds import Manifold


class LinearBetaSchedule:
    """β(t) = β_0 + (β_f − β_0) · u,  u = (t − t_0)/(t_f − t_0).

    Provides
      - β(t)
      - I(t) = ∫_{t_0}^t β(s) ds          (rescaled Brownian time τ)
      - σ_proxy(t) = √(1 − exp(−I(t)))    (Euclidean OU std proxy for std_trick)
    """

    def __init__(self, beta_0: float = 0.001, beta_f: float = 2.0,
                 t0: float = 0.0, tf: float = 1.0):
        assert tf > t0
        self.beta_0 = beta_0
        self.beta_f = beta_f
        self.t0 = t0
        self.tf = tf

    def beta(self, t: Tensor) -> Tensor:
        u = (t - self.t0) / (self.tf - self.t0)
        return self.beta_0 + u * (self.beta_f - self.beta_0)

    def integral(self, t: Tensor) -> Tensor:
        u = (t - self.t0) / (self.tf - self.t0)
        return (self.tf - self.t0) * (self.beta_0 * u + 0.5 * (self.beta_f - self.beta_0) * u ** 2)

    def proxy_std(self, t: Tensor) -> Tensor:
        I = self.integral(t)
        return torch.sqrt(1.0 - torch.exp(-I))


class LangevinSDE:
    """Langevin dynamics on M  —  general parent class (RSGM `Langevin`).

        dX_t = -½ β(t) · ∇_M U(X_t) dt + √β(t) · dB^M_t

    The potential U is supplied by the `limiting` distribution via its `grad_U`
    method.  The SDE's stationary distribution (under suitable smoothness) is
    proportional to e^{-U(x)} w.r.t. the Riemannian volume form on M.
    """

    def __init__(
        self,
        manifold: Manifold,
        schedule: LinearBetaSchedule,
        limiting,
    ):
        self.manifold = manifold
        self.schedule = schedule
        self.limiting = limiting              # must expose .grad_U(x) and .sample(n,...)

    @property
    def t0(self) -> float:
        return self.schedule.t0

    @property
    def tf(self) -> float:
        return self.schedule.tf

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        # ∇_M U(x) is a tangent vector at x (see WrappedNormal.grad_U).
        gradU = self.limiting.grad_U(x)
        beta = self.schedule.beta(t)
        return -0.5 * beta.unsqueeze(-1) * gradU

    def diffusion(self, t: Tensor) -> Tensor:
        return torch.sqrt(self.schedule.beta(t))

    def sample_limiting(self, n: int, device=None, dtype=torch.float32) -> Tensor:
        return self.limiting.sample(n, device=device, dtype=dtype)

    def marginal_sample(self, x_0: Tensor, t: Tensor, n_grw_steps: int = 100) -> Tensor:
        """Sample x_t ∼ p_t(· | x_0) by forward GRW (general; no analytic for Langevin)."""
        from smcdp.sampling import forward_grw
        return forward_grw(self, x_0, t, n_grw_steps)


class BrownianSDE(LangevinSDE):
    """Special case of Langevin with `limiting = UniformOnManifold` (∇U ≡ 0).

    Forward SDE reduces to dX_t = √β(t) dB^M_t — Brownian motion on M with
    time-varying diffusion.  Stationary = uniform on M.  Requires M compact.
    For compact manifolds whose analytic heat-kernel is implemented (e.g.
    Sphere1.marginal_sample), the marginal sampler short-circuits GRW.
    """

    def __init__(self, manifold: Manifold, schedule: LinearBetaSchedule, limiting=None):
        if limiting is None:
            from smcdp.distributions import UniformOnManifold
            limiting = UniformOnManifold(manifold)
        super().__init__(manifold, schedule, limiting)

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        return torch.zeros_like(x)

    def marginal_sample(self, x_0: Tensor, t: Tensor, n_grw_steps: int = 100) -> Tensor:
        if self.manifold.has_analytic_marginal:
            return self.manifold.marginal_sample(x_0, t, self.schedule)
        return super().marginal_sample(x_0, t, n_grw_steps)
