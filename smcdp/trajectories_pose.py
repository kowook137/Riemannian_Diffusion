"""Pose-aware trajectory diffusion module — extension.tex Sections 6, 7, 8.

Mirrors `smcdp/trajectories.py` for the pose-extended (SE(3)) framework.
Key differences:
  - State storage:  τ ∈ R^{B × (H+1) × ambient_dim_pose=15}  (q, q_R, p, z_e)
  - Score:          chart-form  s^q ∈ R^{B × (H+1) × n_q=7}
  - DSM target:     chart form  a*_pose per extension.tex Eq. (35), using
                    Log_SE(3)( T_φ(q_r)^{-1} T_φ(q_0) )  and the W-weighted
                    pseudoinverse — clean and avoids point/tangent dim mismatch.
  - Sampling:       chart-coord noise N(0, G_pose^{-1}), retraction via
                    manifold.exp (which lifts q+δq through T_phi).

Components:
  - TrajectoryScoreNetUNetPose
        Reads τ (B, H+1, 15), slices q-block + z_e, runs ConditionalUnet1D,
        returns chart score (B, H+1, n_q=7).  goal_cond_dim accommodates
        T_start ⊕ T_target = 14-dim conditioning.
  - traj_dsm_pose_loss
        Chart-G DSM loss per Eq. (37): (s^q − a*)^⊤ G_pose (s^q − a*).
  - traj_forward_grw_pose / traj_reverse_grw_pose
        Riemannian random-walk forward/reverse on the pose manifold's chart.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from smcdp.manifolds_pose import EmbodimentPoseGraphManifold
from smcdp.lie_se3 import (
    log_relative_Rp,
    Rp_to_pose7,
)


class PoseLangevinSDE:
    """Lightweight SDE container for the pose pipeline.

    The chart-form pose pipeline computes its own limiting drift in
    `traj_reverse_grw_pose`, so a full LangevinSDE (with `limiting.grad_U`
    returning ambient-tangent form) is unnecessary.  This wrapper just holds
    `manifold` and `schedule`, matching the attribute access pattern of
    `LangevinSDE` used by the DSM loss + sampler.
    """

    def __init__(self, manifold: EmbodimentPoseGraphManifold, schedule):
        self.manifold = manifold
        self.schedule = schedule

    @property
    def t0(self) -> float:
        return self.schedule.t0

    @property
    def tf(self) -> float:
        return self.schedule.tf


# =====================================================================
# Score net (chart-form output)
# =====================================================================


class TrajectoryScoreNetUNetPose(nn.Module):
    """Pose-aware trajectory score net (extension of TrajectoryScoreNetUNet).

    Input τ ∈ R^{B × H1 × ambient_dim_pose}.  Slices q-block (chart) and z_e
    block, runs a ConditionalUnet1D, and returns the chart score
    s^q ∈ R^{B × H1 × n_q}.  The chart-form loss (`traj_dsm_pose_loss`) and
    chart-form sampler (`traj_reverse_grw_pose`) consume this directly.

    Conditioning:
        global cond  =  z_e (frozen, n_z) ⊕ goal_cond (T_start ⊕ T_target ∈ R^14)
        channel cond  ⇒  cond is broadcast across H+1 axis as input channels.
    """

    def __init__(
        self,
        manifold: EmbodimentPoseGraphManifold,
        H: int,
        down_dims: tuple = (128, 256, 512),
        diffusion_step_embed_dim: int = 256,
        n_groups: int = 8,
        kernel_size: int = 3,
        cond_predict_scale: bool = False,
        t_scale: float = 1000.0,
        goal_cond_dim: int = 0,
        cond_injection: str = "global",
    ):
        super().__init__()
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        self.manifold = manifold
        self.H = int(H)
        H1 = self.H + 1
        downsample_factor = 2 ** (len(down_dims) - 1)
        if H1 % downsample_factor != 0:
            raise ValueError(
                f"H+1 ({H1}) must be divisible by 2^(len(down_dims)-1) = "
                f"{downsample_factor}."
            )
        self.t_scale = float(t_scale)
        n_z = getattr(manifold, "n_z", 0)
        self.has_embodiment = bool(n_z > 0)
        self.n_z = int(n_z)
        self.n_q = manifold.n_q
        self.goal_cond_dim = int(goal_cond_dim)
        # Pose storage block size (4 quaternion + 3 position) — used to slice
        # past the pose-block to reach z_e.
        self.n_p_storage = 7
        if cond_injection not in ("global", "channel"):
            raise ValueError(f"cond_injection must be 'global' or 'channel'")
        self.cond_injection = cond_injection

        if self.cond_injection == "global":
            global_dim = (self.n_z if self.has_embodiment else 0) + self.goal_cond_dim
            input_dim_unet = self.n_q
        else:
            global_dim = 0
            input_dim_unet = self.n_q + self.goal_cond_dim + (self.n_z if self.has_embodiment else 0)

        self.unet = ConditionalUnet1D(
            input_dim=input_dim_unet,
            global_cond_dim=(global_dim if global_dim > 0 else None),
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

    def forward(self, tau: Tensor, t: Tensor, goal_cond: Tensor | None = None) -> Tensor:
        """tau: (B, H+1, ambient_dim_pose)  →  s_chart: (B, H+1, n_q)."""
        B, H1, _ = tau.shape
        q_traj = tau[..., : self.n_q]                                         # (B, H+1, n_q)
        if self.has_embodiment:
            z_traj = tau[..., self.n_q + self.n_p_storage:]                    # (B, H+1, n_z)
            z_global = z_traj[:, 0, :]
        else:
            z_traj = None
            z_global = None

        if self.goal_cond_dim > 0 and goal_cond is None:
            raise ValueError(f"goal_cond_dim={self.goal_cond_dim} but goal_cond not provided")
        if goal_cond is not None and goal_cond.shape != (B, self.goal_cond_dim):
            raise ValueError(
                f"goal_cond shape {tuple(goal_cond.shape)} != ({B}, {self.goal_cond_dim})"
            )

        t_scaled = t * self.t_scale

        if self.cond_injection == "global":
            if self.goal_cond_dim > 0:
                full_cond = (torch.cat([z_global, goal_cond], dim=-1)
                              if z_global is not None else goal_cond)
            else:
                full_cond = z_global
            s_chart = self.unet(q_traj, t_scaled, global_cond=full_cond)
        else:
            channel_inputs = [q_traj]
            if self.goal_cond_dim > 0:
                channel_inputs.append(goal_cond.unsqueeze(1).expand(-1, H1, -1))
            if self.has_embodiment:
                channel_inputs.append(z_traj)
            x_input = torch.cat(channel_inputs, dim=-1)
            out_full = self.unet(x_input, t_scaled, global_cond=None)
            s_chart = out_full[..., : self.n_q]
        return s_chart


# =====================================================================
# Forward / reverse Riemannian random walk on the pose manifold (chart form)
# =====================================================================


def _per_timestep_G_chol(manifold: EmbodimentPoseGraphManifold, q: Tensor, z: Tensor) -> Tensor:
    """G_pose Cholesky per (B, H+1) timestep, batched."""
    B, H1, n_q = q.shape
    q_flat = q.reshape(B * H1, n_q)
    z_flat = z.reshape(B * H1, -1)
    L_flat = manifold.G_pose_chol(q_flat, z_flat)                              # (B*H1, n_q, n_q)
    return L_flat.reshape(B, H1, n_q, n_q)


def _grad_log_det_G_pose_chart(
    manifold: EmbodimentPoseGraphManifold,
    q: Tensor, z: Tensor,
) -> Tensor:
    """∂_q (½ log det G_pose(q, z)) via plain autograd.

    Required for the volume-correction term in extension.tex Eq. (17):
        b^q = −(1/2γ²) G⁻¹ (q − μ_q) − (1/4) G⁻¹ ∇_q log det G_pose

    Implementation note: for `LearnedSelfModelFranka7DoFPose`, the hybrid
    `jacobian_pose` uses inner `torch.func.vmap(jacrev(...))` whose
    second-order autograd is unstable / NaN-prone.  We therefore use the
    analytic FK only (`Franka7DoFPose.jacobian_pose`) for the volume
    correction even when a learned residual is present — a benign
    small-residual approximation since ξ_φ ~ 1 cm / 1°, so
    G_a ≈ G_pose to leading order and ∇log det G_a ≈ ∇log det G_pose.
    """
    from smcdp.manifolds_pose import Franka7DoFPose

    with torch.enable_grad():
        q_leaf = q.detach().clone().requires_grad_(True)
        # Compute G using the *analytic* J_pose only (skip residual).
        # This is plain autograd (no vmap), so propagates through
        # pytorch_kinematics' chain.jacobian cleanly.
        if isinstance(manifold, Franka7DoFPose):
            Jp = Franka7DoFPose.jacobian_pose(manifold, q_leaf, z)
        else:
            # Generic fallback: full hybrid Jacobian (may NaN for
            # learned-residual manifolds).
            Jp = manifold.jacobian_pose(q_leaf, z)
        W = manifold._W_diag(q_leaf).unsqueeze(-1)                              # (6, 1)
        eye = torch.eye(manifold.n_q, device=q.device, dtype=q.dtype)
        G = eye + Jp.transpose(-1, -2) @ (W * Jp)
        half_logdet = 0.5 * torch.linalg.slogdet(G).logabsdet                  # (B,)
        grad = torch.autograd.grad(
            half_logdet.sum(), q_leaf,
            create_graph=False, retain_graph=False,
        )[0]
    grad = grad.detach()
    # Guard against any residual NaN (extreme conditioning, etc.)
    if torch.isnan(grad).any() or torch.isinf(grad).any():
        return torch.zeros_like(q)
    return grad


def traj_forward_grw_pose(
    sde,
    tau_0: Tensor,
    r: Tensor,                                                                  # (B,) diffusion times
    n_steps: int = 20,
) -> Tensor:
    """Forward GRW on M_φ^pose(z_e)^{H+1} in chart form, retracting via H_φ^pose.

    Brownian-on-M with diffusion √β (extension.tex Sec. 5):
        dX_r = √β(r) dB^M_r       (we omit Langevin drift; score net learns the
                                    drift correction implicitly via DSM)
    Discretized chart update per substep:
        δq ~ √(β(r) · Δr) · ξ_q,  ξ_q ~ N(0, G_pose^{-1})
    so accumulated variance ≈ ∫β(s) ds = τ_brown(r), matching the DSM target's
    1/τ_brown rescaling.
    """
    manifold = sde.manifold
    schedule = sde.schedule
    B, H1, d = tau_0.shape
    n_q = manifold.n_q
    n_z = manifold.n_z

    q = tau_0[..., :n_q].clone()                                                # (B, H+1, n_q)
    z = tau_0[..., n_q + 7:].clone()                                            # (B, H+1, n_z)

    # Discretise (0 → r) per sample
    dr = (r / n_steps).view(B, 1, 1)                                            # (B, 1, 1)
    for k in range(n_steps):
        # Mid-step time for β coefficient (Stratonovich-ish; OK for small Δr)
        r_k = (k + 0.5) / n_steps * r                                            # (B,)
        beta_k = schedule.beta(r_k).view(B, 1, 1)                                # (B, 1, 1)
        L = _per_timestep_G_chol(manifold, q, z)                                # (B, H+1, n_q, n_q)
        eps = torch.randn_like(q)
        a = torch.linalg.solve_triangular(
            L.transpose(-1, -2), eps.unsqueeze(-1), upper=True
        ).squeeze(-1)                                                           # (B, H+1, n_q)
        # √(β · Δr) · a — matches position-only `traj_forward_grw` convention.
        q = q + (beta_k * dr).sqrt() * a

    q_flat = q.reshape(B * H1, n_q)
    z_flat = z.reshape(B * H1, n_z)
    x_flat = manifold.make_x(q_flat, z_flat)
    return x_flat.reshape(B, H1, manifold.ambient_dim)


# =====================================================================
# Chart-form DSM loss (extension.tex Eq. (37), block-decomposed Eq. (35))
# =====================================================================


def _dsm_chart_target_pose(
    manifold: EmbodimentPoseGraphManifold,
    q_r: Tensor, q_0: Tensor, z: Tensor,
) -> Tensor:
    """a*_pose(q_r, x_0) ∈ R^{n_q}  per extension.tex Eq. (35):

        a* = G_pose^{-1} · [ (q_0 − q_r) + J_pose^⊤ W · Log_SE(3)( T_φ(q_r)^{-1} T_φ(q_0) ) ]
    """
    R_r, p_r = manifold.T_phi_Rp(q_r, z)
    R_0, p_0 = manifold.T_phi_Rp(q_0, z)
    xi = log_relative_Rp(R_r, p_r, R_0, p_0)                                    # (B, 6)
    Jp = manifold.jacobian_pose(q_r, z)                                         # (B, 6, n_q)
    W = manifold._W_diag(q_r)                                                   # (6,)
    rhs = (q_0 - q_r) + (
        Jp.transpose(-1, -2) @ (W.unsqueeze(-1) * xi.unsqueeze(-1))
    ).squeeze(-1)                                                               # (B, n_q)
    L = manifold.G_pose_chol(q_r, z)                                            # (B, n_q, n_q)
    a_star = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)             # (B, n_q)
    return a_star


def traj_dsm_pose_loss(
    score_fn,                                                                   # (τ, t, goal_cond=None) → (B, H+1, n_q)  CHART score
    sde,
    tau_0: Tensor,
    eps: float = 1e-3,
    weight: str = "sigma2",
    n_grw_steps: int = 20,
    goal_cond: Optional[Tensor] = None,
    cond_drop_prob: float = 0.0,
    endpoint_weight: float = 1.0,
) -> Tensor:
    """ℒ_pose = E_{r, τ_0, τ_r} [ w(r) · Σ_h (s^q_h − a*_pose,h)^⊤ G_pose,h (s^q_h − a*_pose,h) / τ²_brown ]

    Chart-G DSM loss (extension.tex Eq. (37)) with the Varadhan-asymptotic
    1/τ_brown scaling:  target  =  a*_pose / τ_brown.

    `score_fn` must return chart score s^q (B, H+1, n_q), NOT lifted.
    """
    manifold: EmbodimentPoseGraphManifold = sde.manifold
    schedule = sde.schedule
    B, H1, _ = tau_0.shape
    n_q = manifold.n_q
    device, dtype = tau_0.device, tau_0.dtype

    # Sample diffusion time r ∈ [eps, tf]
    r = eps + (schedule.tf - eps) * torch.rand(B, device=device, dtype=dtype)
    tau_r = traj_forward_grw_pose(sde, tau_0, r, n_grw_steps)                   # (B, H+1, ambient_dim_pose)

    # CFG cond dropout
    if goal_cond is not None and cond_drop_prob > 0.0:
        drop_mask = torch.rand(B, device=device) < cond_drop_prob
        if drop_mask.any():
            goal_cond = goal_cond.clone()
            goal_cond[drop_mask] = 0.0

    # Brownian rescaled time τ = ∫_0^r β(s) ds, broadcast across timesteps
    tau_brown = schedule.integral(r).clamp(min=1e-12)                            # (B,)

    # Compute chart target a*_pose per (B*H1) sample
    q_r = tau_r[..., :n_q]
    q_0 = tau_0[..., :n_q]
    z = tau_r[..., n_q + 7:]                                                    # (B, H+1, n_z)
    q_r_flat = q_r.reshape(B * H1, n_q)
    q_0_flat = q_0.reshape(B * H1, n_q)
    z_flat = z.reshape(B * H1, -1)
    a_star_flat = _dsm_chart_target_pose(manifold, q_r_flat, q_0_flat, z_flat)
    a_star = a_star_flat.reshape(B, H1, n_q)

    # Score (chart) and difference
    if goal_cond is not None:
        s_chart = score_fn(tau_r, r, goal_cond=goal_cond)
    else:
        s_chart = score_fn(tau_r, r)
    target = a_star / tau_brown.view(B, 1, 1)                                   # (B, H+1, n_q)
    diff = s_chart - target                                                      # (B, H+1, n_q)

    # G_pose-weighted squared norm per timestep:  diff^T G_pose diff
    Jp_flat = manifold.jacobian_pose(q_r_flat, z_flat)                          # (B*H1, 6, n_q)
    W = manifold._W_diag(q_r_flat).unsqueeze(0).unsqueeze(-1)                   # (1, 6, 1)
    G_flat = torch.eye(n_q, device=device, dtype=dtype).expand(B * H1, n_q, n_q) \
        + Jp_flat.transpose(-1, -2) @ (W * Jp_flat)                             # (B*H1, n_q, n_q)
    diff_flat = diff.reshape(B * H1, n_q)
    sq_per_pt = (diff_flat.unsqueeze(-2) @ G_flat @ diff_flat.unsqueeze(-1)
                  ).squeeze(-1).squeeze(-1)                                     # (B*H1,)
    sq_per_pt_h = sq_per_pt.reshape(B, H1)

    if endpoint_weight != 1.0:
        h_weights = torch.ones(H1, device=device, dtype=dtype)
        h_weights[-1] = float(endpoint_weight)
        sq_per_traj = (sq_per_pt_h * h_weights).sum(-1)
    else:
        sq_per_traj = sq_per_pt_h.sum(-1)

    if weight == "sigma2":
        w = schedule.proxy_std(r) ** 2
    elif weight == "beta":
        w = schedule.beta(r)
    elif weight == "none":
        w = torch.ones_like(r)
    else:
        raise ValueError(f"unknown weight '{weight}'")
    return (w * sq_per_traj).mean()


# =====================================================================
# Reverse-time GRW sampler on the pose manifold (chart form)
# =====================================================================


@torch.no_grad()
def traj_reverse_grw_pose(
    sde,
    score_fn,                                                                   # (τ, t, goal_cond=None) → (B, H+1, n_q) CHART
    n_samples: int,
    H: int,
    n_steps: int = 200,
    goal_cond: Optional[Tensor] = None,
    z_e: Optional[Tensor] = None,                                               # (B, n_z)
    limiting_q_mean: Optional[Tensor] = None,                                   # (n_q,)
    limiting_scale: float = 0.6,
    eps: float = 2e-4,
    device=None, dtype=torch.float32,
    score_postprocess=None,                                                     # optional post-net rescaling
) -> Tensor:
    """Reverse-time GRW on M_φ^pose^{H+1}, chart form.

    Initial sample: q_h ~ N(μ_q, γ² G_pose(μ_q, z_e)^{-1}), lifted via H_φ^pose
    (extension.tex Sec. 6 reference distribution).

    Reverse Euler step (per timestep h):
        μ^q_h = −b^q(q_{h, k}) + s^q_h(τ_k)
        δq_h  = Δr · μ^q_h + √Δr · ξ_h^q,    ξ_h^q ~ N(0, G_pose^{-1})
        q_{h, k+1} = q_{h, k} + δq_h
        x_{h, k+1} = H_φ^pose(q_{h, k+1}, z_e).
    """
    manifold: EmbodimentPoseGraphManifold = sde.manifold
    schedule = sde.schedule
    n_q = manifold.n_q
    n_z = manifold.n_z
    H1 = H + 1
    B = n_samples

    if z_e is None:
        z_e = torch.zeros(B, n_z, device=device, dtype=dtype)
    z_e = z_e.to(device=device, dtype=dtype)
    if z_e.shape[0] != B:
        raise ValueError(f"z_e batch ({z_e.shape[0]}) ≠ B ({B})")
    z_traj = z_e.unsqueeze(1).expand(B, H1, n_z).contiguous()                   # (B, H+1, n_z)

    if limiting_q_mean is None:
        mu_q = torch.zeros(n_q, device=device, dtype=dtype)
    else:
        mu_q = limiting_q_mean.to(device=device, dtype=dtype)

    # Initial chart-Gaussian sample
    mu_q_flat = mu_q.expand(B * H1, n_q)
    z_flat = z_traj.reshape(B * H1, n_z)
    L_mu = manifold.G_pose_chol(mu_q_flat, z_flat)                              # (B*H1, n_q, n_q)
    eps_init = torch.randn(B, H1, n_q, device=device, dtype=dtype)
    a_init = torch.linalg.solve_triangular(
        L_mu.transpose(-1, -2),
        eps_init.reshape(B * H1, n_q, 1),
        upper=True,
    ).squeeze(-1)                                                               # (B*H1, n_q)
    q = mu_q_flat + limiting_scale * a_init
    q = q.reshape(B, H1, n_q)
    x = manifold.make_x(q.reshape(B * H1, n_q), z_flat).reshape(B, H1, manifold.ambient_dim)

    # Discretise reverse time r: tf → eps
    r_grid = torch.linspace(schedule.tf, eps, n_steps + 1, device=device, dtype=dtype)
    for k in range(n_steps):
        r_k = r_grid[k]
        r_kp1 = r_grid[k + 1]
        dr = (r_k - r_kp1).abs()                                                # positive
        beta_k = schedule.beta(r_k.expand(B)).view(B, 1, 1)                     # (B, 1, 1)

        # Score (chart, B, H+1, n_q)
        if goal_cond is not None:
            s = score_fn(x, r_k.expand(B), goal_cond=goal_cond)
        else:
            s = score_fn(x, r_k.expand(B))
        if score_postprocess is not None:
            s = score_postprocess(s, r_k)

        # Limiting drift (chart, extension.tex Eq. (17)):
        #   b^q = −(1/2γ²) G^{-1} (q − μ)  −  (1/4) G^{-1} ∇_q log det G_pose
        # Both terms included for full Riemannian-volume-correct sampling.
        gamma2 = limiting_scale ** 2
        q_flat = q.reshape(B * H1, n_q)
        L = manifold.G_pose_chol(q_flat, z_flat)
        rhs = (q_flat - mu_q_flat).unsqueeze(-1)
        Ginv_diff = torch.cholesky_solve(rhs, L).squeeze(-1)                    # (B*H1, n_q)
        b_q_term1 = -(0.5 / gamma2) * Ginv_diff                                 # −(1/2γ²) G⁻¹(q−μ)

        # Volume correction term: −(1/4) G⁻¹ ∇log det G  =  −½ G⁻¹ ∇(½ log det G).
        try:
            grad_half_logdet = _grad_log_det_G_pose_chart(
                manifold, q_flat, z_flat,
            )                                                                   # (B*H1, n_q)
            Ginv_grad_logdet = torch.cholesky_solve(
                grad_half_logdet.unsqueeze(-1), L,
            ).squeeze(-1)
            b_q_term2 = -0.5 * Ginv_grad_logdet                                 # −(1/4) G⁻¹ ∇log det G
        except RuntimeError:
            # Fallback: skip volume correction (e.g. if autograd through
            # pytorch_kinematics fails for some manifold variant).
            b_q_term2 = torch.zeros_like(b_q_term1)
        b_q = (b_q_term1 + b_q_term2).reshape(B, H1, n_q)

        # Anderson reverse SDE: dY = [-b_fwd + β·score] dr̄ + √β dB^M
        # (extension.tex Sec. 5; matches position-only `traj_reverse_grw` convention).
        reverse_drift = -b_q + beta_k * s                                       # (B, H+1, n_q)

        # Tangent noise ξ_q ~ N(0, G^{-1})
        eps_step = torch.randn(B, H1, n_q, device=device, dtype=dtype)
        a_step = torch.linalg.solve_triangular(
            L.transpose(-1, -2), eps_step.reshape(B * H1, n_q, 1), upper=True,
        ).squeeze(-1).reshape(B, H1, n_q)

        delta_q = reverse_drift * dr + (beta_k * dr).sqrt() * a_step
        q = q + delta_q
        x = manifold.make_x(q.reshape(B * H1, n_q), z_flat).reshape(B, H1, manifold.ambient_dim)

    return x


class TrajectoryScaledScoreFnPose(nn.Module):
    """std_trick wrapper for the pose-aware chart-output score net.

    Mirrors `TrajectoryScaledScoreFn` but works in chart space (output
    (B, H+1, n_q)).  No residual_trick (would require a chart-form b_fwd).
    """

    def __init__(self, net: TrajectoryScoreNetUNetPose, sde, std_trick: bool = True):
        super().__init__()
        self.net = net
        self.sde = sde
        self.std_trick = std_trick

    def forward(self, tau: Tensor, t: Tensor, goal_cond: Tensor | None = None) -> Tensor:
        out = self.net(tau, t, goal_cond=goal_cond)
        if self.std_trick:
            B = tau.shape[0]
            sigma = self.sde.schedule.proxy_std(t).clamp(min=1e-6).view(B, 1, 1)
            out = out / sigma
        return out
