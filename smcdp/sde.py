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

    def proxy_std(self, t: Tensor, mode: str = "ou") -> Tensor:
        """Proxy std for std_trick / loss weighting.

        - ``mode="ou"`` (default, backward-compat): √(1 − exp(−I(t))), the
          stationary OU std for VP-SDE. Correct calibration when forward has
          a Langevin drift toward an OU stationary.
        - ``mode="brownian"``: √I(t), the pure-Brownian marginal std (no
          drift). Correct calibration for Method A (drift OFF), where the
          forward marginal cov is τ_brown(t)·G^{-1}, so std scale is √τ_brown.
        """
        I = self.integral(t)
        if mode == "brownian":
            return torch.sqrt(I.clamp(min=1e-12))
        if mode != "ou":
            raise ValueError(f"unknown proxy_std mode {mode!r}")
        return torch.sqrt(1.0 - torch.exp(-I))

    # ===================================================================
    # joint_limit_extension v5.1 — closed-form VP-OU transition helpers
    # (joint_limit_extension.tex §8.1, §8.2)
    # ===================================================================
    # Forward OU on chart u:
    #     du_r = -½ β(r) u_r dr + √β(r) Ḡ_Q^{-1/2} dW_r
    # Closed-form transition (constant diffusion → Gaussian):
    #     p_{r|0}(u_r | u_0) = N( α(r) · u_0 , σ²(r) · Ḡ_Q^{-1} )
    #     α(r) := exp(-τ(r)/2)             (decay of mean)
    #     σ²(r) := 1 - exp(-τ(r))          (variance growth)
    #     τ(r)  := ∫_0^r β(s) ds = self.integral(r)
    # Exact Euclidean score:
    #     ∇_{u_r} log p_{r|0}(u_r|u_0) = -Ḡ_Q · (u_r - α(r) u_0) / σ²(r)
    # σ(r) = √σ²(r) is identical to `proxy_std(r, "ou")` — kept as a named alias.

    def tau(self, t: Tensor) -> Tensor:
        """Cumulative β-integral τ(r) := ∫_0^r β(s) ds (alias for `integral`)."""
        return self.integral(t)

    def alpha(self, t: Tensor) -> Tensor:
        """OU mean-decay factor α(r) = exp(-τ(r)/2)  ∈ (0, 1].  α(0) = 1."""
        return torch.exp(-0.5 * self.integral(t))

    def sigma2(self, t: Tensor) -> Tensor:
        """OU marginal variance scalar σ²(r) = 1 − exp(−τ(r))  ∈ [0, 1).
        Full marginal covariance of u_r | u_0 is σ²(r) · Ḡ_Q^{-1}.
        σ²(0) = 0 (no noise); σ²(K) → 1 as τ(K) → ∞ (stationary limit)."""
        return 1.0 - torch.exp(-self.integral(t))

    def sigma(self, t: Tensor) -> Tensor:
        """σ(r) = √σ²(r) = √(1 − exp(−τ(r))).  Identical to proxy_std(t,'ou')."""
        return torch.sqrt((1.0 - torch.exp(-self.integral(t))).clamp(min=1e-12))


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
