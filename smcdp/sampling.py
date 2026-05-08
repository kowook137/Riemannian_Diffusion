"""Geodesic Random Walk samplers (forward and reverse).

Implements RSGM Algorithm 1: at each step, draw tangent Gaussian noise,
take an Euler–Maruyama step in the tangent space, and exp-map onto M.
This is *intrinsic* — by construction every iterate lies on M (up to the
manifold's exp/retraction accuracy), so there is no extrinsic projection
step and no accumulated projection error.

General Itô SDE on M:
    Forward :  dX_t = b(X_t) dt + σ(t) dB^M_t
    Reverse :  dY_τ = (-b + σ² · s_θ) dτ + σ(t) dB̃^M_τ                 (Thm 1)
    Prob. flow ODE :  dY_τ = (-b + ½ σ² · s_θ) dτ                       (deterministic)

Generalized to any LangevinSDE — the drift `sde.drift(x, t)` carries the
forward drift (b = 0 for Brownian, b = -½ β ∇U for Langevin).
"""
from __future__ import annotations

import torch
from torch import Tensor

from smcdp.sde import LangevinSDE


@torch.no_grad()
def forward_grw(sde: LangevinSDE, x_0: Tensor, t_target: Tensor, n_steps: int) -> Tensor:
    """Forward GRW integrating dX_t = b(X) dt + σ(t) dB^M_t from t_0 to t_target.

    Each sample marches with its own dt = (t_target − t_0)/n_steps so the batch
    finishes exactly at the per-sample target time.  Drift is taken from
    `sde.drift(x, t)` so this works for both Brownian (b=0) and Langevin (b≠0).
    """
    schedule = sde.schedule
    dt = (t_target - schedule.t0) / n_steps                          # (B,)
    x = x_0
    for k in range(n_steps):
        t_k = schedule.t0 + k * dt                                   # (B,)
        b_fwd = sde.drift(x, t_k)                                    # (B, d) ∈ T_xM
        sigma = sde.diffusion(t_k)                                   # (B,)
        z = sde.manifold.random_normal_tangent(x)                    # (B, d) ∈ T_xM
        v = b_fwd * dt.unsqueeze(-1) + \
            sigma.unsqueeze(-1) * z * dt.abs().sqrt().unsqueeze(-1)
        x = sde.manifold.exp(x, v)
    return x


@torch.no_grad()
def reverse_grw(
    sde: LangevinSDE,
    score_fn,                      # callable (x: Tensor[B,d], t: Tensor[B]) -> Tensor[B,d] ∈ T_xM
    x_T: Tensor,
    n_steps: int,
    eps: float = 1e-3,
    return_history: bool = False,
):
    """Reverse-time GRW (RSGM Theorem 1).

    Reverse drift in forward-in-τ formulation:
        b_rev(x, t) = -b_fwd(x, t) + σ²(t) · score(x, t)
    For Brownian (b_fwd = 0):                 b_rev = β · score
    For Langevin (b_fwd = -½ β ∇U):           b_rev = ½ β ∇U + β · score
    """
    schedule = sde.schedule
    B = x_T.shape[0]
    device, dtype = x_T.device, x_T.dtype

    ts = torch.linspace(schedule.tf, schedule.t0 + eps,
                        n_steps + 1, device=device, dtype=dtype)
    history = [] if return_history else None
    x = x_T
    for k in range(n_steps):
        t_k = ts[k].expand(B)
        dtau = (ts[k] - ts[k + 1])                                   # > 0
        beta = schedule.beta(t_k)                                    # (B,)
        b_fwd = sde.drift(x, t_k)                                    # (B, d) ∈ T_xM
        score = score_fn(x, t_k)                                     # (B, d) ∈ T_xM
        z = sde.manifold.random_normal_tangent(x)                    # (B, d) ∈ T_xM

        reverse_drift = -b_fwd + beta.unsqueeze(-1) * score
        diffusion = beta.sqrt().unsqueeze(-1) * z

        v = reverse_drift * dtau + diffusion * dtau.sqrt()
        x = sde.manifold.exp(x, v)
        if return_history:
            history.append(x)

    if return_history:
        return x, torch.stack(history, dim=0), ts
    return x


@torch.no_grad()
def reverse_ode(
    sde: LangevinSDE,
    score_fn,
    x_T: Tensor,
    n_steps: int,
    eps: float = 1e-3,
):
    """Probability flow ODE counterpart of reverse_grw (deterministic).

    Same marginals as reverse SDE, integrated forward-in-τ as
        dY_τ = (-b_fwd + ½ σ² · score) dτ
    via Euler steps with manifold exp.  Useful diagnostic for whether
    reverse-SDE noise injection is degrading sharpness.
    """
    schedule = sde.schedule
    B = x_T.shape[0]
    device, dtype = x_T.device, x_T.dtype
    ts = torch.linspace(schedule.tf, schedule.t0 + eps,
                        n_steps + 1, device=device, dtype=dtype)
    x = x_T
    for k in range(n_steps):
        t_k = ts[k].expand(B)
        dtau = (ts[k] - ts[k + 1])
        beta = schedule.beta(t_k)
        b_fwd = sde.drift(x, t_k)
        score = score_fn(x, t_k)
        v = (-b_fwd + 0.5 * beta.unsqueeze(-1) * score) * dtau
        x = sde.manifold.exp(x, v)
    return x
