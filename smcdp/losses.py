"""Score-matching losses on Riemannian manifolds.

Toy 1 uses Denoising Score Matching with the Varadhan asymptotic
(RSGM Table 2, ℓ_{t|0} DSM, Varadhan row):

    ℓ(θ) = E_{t, x_0, x_t} ‖ s_θ(t, x_t) − Log_{x_t}(x_0) / t ‖²_{x_t}

This is the only DSM form available when the heat kernel of M is unknown
(e.g. learned graph manifolds in Toy 2/3), so we use it from the start.
On S^1 it coincides numerically with the Sturm–Liouville and closed-form
heat-kernel forms because G ≡ 1.
"""
from __future__ import annotations

import torch
from torch import Tensor

from smcdp.sde import BrownianSDE


def dsm_varadhan_loss(
    score_fn,                          # callable (x: (B, d), t: (B,)) -> (B, d) ∈ T_xM
    sde: BrownianSDE,
    x_0: Tensor,                       # (B, d) on M
    eps: float = 1e-3,
    weight: str = "sigma2",            # "sigma2" | "beta" | "none"
    target: str = "varadhan",          # "varadhan" | "exact"
    n_grw_steps: int = 100,            # used only if no analytic marginal
) -> Tensor:
    """Single-batch DSM loss (RSGM Table 2).

    ℓ(θ) = E_{t, x_0, x_t} [ w(t) · ‖ s_θ(t, x_t) − target(x_t, x_0, t) ‖²_{x_t} ]

    Target choices:
      - "varadhan": Log_{x_t}(x_0) / t           (asymptotically valid as t→0;
                                                  RSGM `dsmv` default)
      - "exact":    ∇_{x_t} log p_{t|0}(x_t|x_0) (exact Sturm–Liouville heat-kernel
                                                  gradient; requires manifold to
                                                  implement `heat_kernel_score`)

    Weighting choices:
      - "sigma2": w(t) = σ_proxy(t)²  (RSGM `like_w=False`)
      - "beta":   w(t) = β(t)         (RSGM `like_w=True`)
      - "none":   w(t) = 1
    """
    B = x_0.shape[0]
    device, dtype = x_0.device, x_0.dtype
    schedule = sde.schedule

    # t ~ U[ε, t_f]
    t = eps + (schedule.tf - eps) * torch.rand(B, device=device, dtype=dtype)

    # forward marginal x_t ∼ p_t(· | x_0)
    x_t = sde.marginal_sample(x_0, t, n_grw_steps=n_grw_steps)

    if target == "varadhan":
        # Varadhan asymptotic divides by Brownian (rescaled) time τ(t) = ∫_0^t β(s) ds,
        # NOT by SDE forward time t.  This matches RSGM's `varhadan_exp` (delta_t =
        # rescale_t(t) − rescale_t(s)).  Using t here would silently distort the
        # target by a factor τ/t that ranges over orders of magnitude across t.
        tau = schedule.integral(t).clamp(min=1e-12)
        tgt = sde.manifold.log(x_t, x_0) / tau.unsqueeze(-1)
    elif target == "exact":
        if not hasattr(sde.manifold, "heat_kernel_score"):
            raise ValueError(
                f"target='exact' requires manifold to implement heat_kernel_score; "
                f"{type(sde.manifold).__name__} does not"
            )
        tau = schedule.integral(t)                                     # (B,)
        tgt = sde.manifold.heat_kernel_score(x_t, x_0, tau)            # (B, d) ∈ T_{x_t}M
    else:
        raise ValueError(f"unknown target '{target}'")

    score = score_fn(x_t, t)                                           # (B, d) ∈ T_{x_t}M
    diff = score - tgt
    sq = sde.manifold.squared_norm(x_t, diff)                          # (B,)

    if weight == "sigma2":
        w = schedule.proxy_std(t) ** 2
    elif weight == "beta":
        w = schedule.beta(t)
    elif weight == "none":
        w = torch.ones_like(t)
    else:
        raise ValueError(f"unknown weight '{weight}'")
    return (w * sq).mean()
