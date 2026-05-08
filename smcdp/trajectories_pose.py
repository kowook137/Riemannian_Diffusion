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
    """Lightweight Langevin SDE container for the pose pipeline.

    Holds the manifold + schedule + Langevin limiting parameters
    (μ_q, γ).  The chart-form pose forward and reverse GRW both use these
    via internal helpers (no separate limiting-distribution class).

    Two drift potential conventions are supported (selected by
    ``confining_kappa``):

    Legacy (default, ``confining_kappa = 0``):
        U^pose(q)  =  (1/2γ²) ‖q − μ_q‖²  +  ½ log det G_pose(q)
        Stationary chart density ∝ e^{−U^pose} √det G  =  e^{−‖q−μ‖²/(2γ²)}
        i.e. an isotropic chart-Gaussian, which DOES NOT match the sampling
        initialization q_K ~ N(μ_q, γ² G_pose(μ_q)^{-1}) (extension.tex Sec. 6).

    Fix 3 / Option B (``confining_kappa > 0``, noise_stationary_fix.md Sec. 2.3):
        U^pose(q; μ_q)  =  (1/2γ²) (q − μ_q)^⊤ Ĝ (q − μ_q)  +  κ U_box(q)
        where Ĝ := G_pose(μ_q) is the FIXED anchor metric (computed once per
        forward / reverse call) and U_box is the soft joint-range confining
        potential.  Stationary chart density ∝ e^{−U^pose} √det G.  Near
        q ≈ μ_q (sampling initialization region), G(q) ≈ Ĝ so √det G is
        approximately constant and the stationary reduces to
            N(μ_q, γ² Ĝ^{-1}) × box,
        exactly matching the sampling initialization.

    extension.tex Eq. (17) form is preserved in both:  b^q = −½ G⁻¹ ∇_q U^pose.
    """

    def __init__(
        self,
        manifold: EmbodimentPoseGraphManifold,
        schedule,
        limiting_q_mean: Tensor | None = None,
        limiting_scale: float = 0.6,
        forward_langevin_drift: bool = False,
        confining_kappa: float = 0.0,
        confining_epsilon_frac: float = 0.05,
    ):
        """
        Args:
            forward_langevin_drift: if True, forward GRW includes the Langevin
                limiting drift -½ β G^{-1} ∇U (extension.tex Eq. 15-17).
                Default False because with W_p = σ_p^{-2} ≫ 1 (e.g. 10⁴), the
                drift on the null-space of J_pose can push q out of valid
                Franka joint range, producing NaN in pytorch_kinematics →
                non-PD G_pose.  Pure Brownian forward (default) is the
                Varadhan-regime DSM standard and is empirically stable.
                The reverse GRW always uses Langevin drift (sampling-time
                only, manifold-bounded).
            confining_kappa: strength of the soft confining box potential
                (Fix 3, noise_stationary_fix.md Sec. 2.3).  Set > 0 to enable
                anchor-metric Option B: replaces the legacy V = ½γ⁻²‖q−μ‖²
                + ½ log det G with the anchor-metric form
                V = (1/2γ²)(q−μ)^T Ĝ (q−μ) + κ U_box, where Ĝ = G(μ) is fixed
                per call.  Recommended value: κ ∈ [10², 10⁴].
            confining_epsilon_frac: ε_box = epsilon_frac · (q_max − q_min)
                margin inside Franka joint range.  Default 5% (per doc).
                Only relevant when ``confining_kappa > 0``.
        """
        self.manifold = manifold
        self.schedule = schedule
        if limiting_q_mean is not None:
            self.limiting_q_mean = torch.as_tensor(limiting_q_mean, dtype=torch.float32)
        else:
            self.limiting_q_mean = None
        self.limiting_scale = float(limiting_scale)
        self.forward_langevin_drift = bool(forward_langevin_drift)
        self.confining_kappa = float(confining_kappa)
        self.confining_epsilon_frac = float(confining_epsilon_frac)

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


def _grad_U_box_chart(
    q: Tensor,                                                                  # (..., n_q)
    q_lower: Tensor,                                                            # (n_q,)
    q_upper: Tensor,                                                            # (n_q,)
    epsilon: Tensor,                                                            # (n_q,) or scalar
    kappa: float,
) -> Tensor:
    """∇_q [κ Σ_i (ReLU(q_i − q_max + ε)² + ReLU(q_min + ε − q_i)²)].

    Closed-form (no autograd).  Active only near boundary; zero deep inside the
    Franka joint range.  Per noise_stationary_fix.md Sec. 2.3.6, ε is set to
    a 5%-of-range margin so the potential activates BEFORE q hits the FK-NaN
    region.
    """
    if kappa == 0.0:
        return torch.zeros_like(q)
    # Broadcast (n_q,) limits / ε across leading dims of q.
    upper_excess = (q - (q_upper - epsilon)).clamp(min=0.0)
    lower_excess = ((q_lower + epsilon) - q).clamp(min=0.0)
    return 2.0 * kappa * (upper_excess - lower_excess)                           # (..., n_q)


def _anchor_drift_potential_grad(
    sde,
    q: Tensor,                                                                   # (B, ..., n_q)
    z: Tensor,                                                                   # (B, ..., n_z)
    anchor_G: Tensor | None,                                                     # (B, n_q, n_q) or None
) -> Tensor:
    """Compute ∇_q U^pose for the forward / reverse drift, in chart form.

    Returns the same leading shape as ``q``.  Selects legacy vs Fix 3 (Option
    B) form based on ``sde.confining_kappa``:

      - κ = 0 (legacy):  ∇U = (q − μ)/γ² + ∇(½ log det G(q))
      - κ > 0 (Fix 3):   ∇U = γ⁻² Ĝ_b (q − μ) + ∇U_box(q),  Ĝ_b = anchor_G[b]

    For Fix 3, ``anchor_G`` must be provided as a per-batch tensor of shape
    (B, n_q, n_q) — Ĝ_b = G(μ_q, z_e[b]) is sample-specific because z_e
    varies per sample (e.g. Franka tool extension).  For legacy, it is ignored.
    """
    manifold: EmbodimentPoseGraphManifold = sde.manifold
    n_q = manifold.n_q
    mu_q = sde.limiting_q_mean.to(device=q.device, dtype=q.dtype)
    gamma2 = sde.limiting_scale ** 2

    if sde.confining_kappa > 0.0:
        # Fix 3 / Option B (per-batch Ĝ)
        if anchor_G is None:
            raise ValueError("Fix 3 enabled but anchor_G was not provided.")
        if anchor_G.dim() != 3:
            raise ValueError(
                f"anchor_G must have shape (B, n_q, n_q); got {tuple(anchor_G.shape)}"
            )
        diff = q - mu_q                                                          # (B, ..., n_q)
        # Per-batch quadratic gradient: anchor_quad[b, ..., i] = Ĝ_b[i, j] · diff[b, ..., j].
        anchor_quad = torch.einsum("bij,b...j->b...i", anchor_G, diff) / gamma2  # (B, ..., n_q)
        # Joint range box potential.
        q_lower, q_upper = manifold.joint_limits(device=q.device, dtype=q.dtype) # (n_q,) each
        eps_box = sde.confining_epsilon_frac * (q_upper - q_lower)               # (n_q,)
        box_grad = _grad_U_box_chart(q, q_lower, q_upper, eps_box,
                                       sde.confining_kappa)                      # (B, ..., n_q)
        return anchor_quad + box_grad
    else:
        # Legacy form: ∇U = (q − μ)/γ² + ∇(½ log det G(q))
        chart_grad_quad = (q - mu_q) / gamma2                                    # (B, ..., n_q)
        leading = q.shape[:-1]
        q_flat = q.reshape(-1, n_q)
        z_flat = z.reshape(-1, z.shape[-1])
        chart_grad_logdet = _grad_log_det_G_pose_chart(
            manifold, q_flat, z_flat,
        ).reshape(*leading, n_q)
        return chart_grad_quad + chart_grad_logdet


def _compute_anchor_G(sde, z_e: Tensor) -> Tensor | None:
    """Compute per-batch Ĝ_b = G_pose(μ_q, z_e[b]) once for Fix 3.

    Args:
        z_e: (B, n_z) per-sample embodiment context.  μ_q is shared across the
            batch (single anchor in q-space), but Ĝ varies per b because z_e
            enters G_pose through the kinematic Jacobian (e.g. Franka tool
            extension).  Per-batch Ĝ is the principled form (no
            shared-z approximation); cost is B×O(n_q² · jacobian eval), which
            is small for n_q = 7.

    Returns:
        anchor_G: (B, n_q, n_q), detached.  None if Fix 3 is OFF.
    """
    if sde.confining_kappa <= 0.0:
        return None
    if sde.limiting_q_mean is None:
        raise ValueError("Fix 3 requires limiting_q_mean (anchor μ_q) to be set.")
    manifold = sde.manifold
    n_q = manifold.n_q
    if z_e.dim() != 2:
        raise ValueError(
            f"z_e must have shape (B, n_z); got {tuple(z_e.shape)}"
        )
    B = z_e.shape[0]
    mu_q = sde.limiting_q_mean.to(device=z_e.device, dtype=z_e.dtype)
    mu_q_b = mu_q.unsqueeze(0).expand(B, n_q)                                    # (B, n_q)
    return manifold.G_pose(mu_q_b, z_e).detach()                                 # (B, n_q, n_q)


def traj_forward_grw_pose(
    sde,
    tau_0: Tensor,
    r: Tensor,                                                                  # (B,) diffusion times
    n_steps: int = 20,
) -> Tensor:
    """Forward GRW on M_φ^pose(z_e)^{H+1} in chart form, retracting via H_φ^pose.

    Langevin SDE (extension.tex Sec. 5, Eq. 15-17):
        dX_r = b(X_r) dr + √β(r) dB^M_r,    b = −½ ∇_M U^pose
        b^q   = −½ G⁻¹ ∇_q U^pose
              = −(1/2γ²) G⁻¹ (q − μ_q) − (1/4) G⁻¹ ∇_q log det G_pose

    Discretized chart update per substep (matches position-only convention,
    with β scaling applied to BOTH drift and noise per VP-style Langevin):
        δq = β · b^q · Δr   +   √(β · Δr) · ξ_q,    ξ_q ~ N(0, G^{-1})

    If `sde.limiting_q_mean` is None, drift is omitted (pure Brownian-on-M).
    """
    manifold = sde.manifold
    schedule = sde.schedule
    B, H1, d = tau_0.shape
    n_q = manifold.n_q
    n_z = manifold.n_z

    q = tau_0[..., :n_q].clone()                                                # (B, H+1, n_q)
    z = tau_0[..., n_q + 7:].clone()                                            # (B, H+1, n_z)

    has_drift = (sde.forward_langevin_drift
                  and sde.limiting_q_mean is not None)
    # Fix 3 per-batch anchor metric: Ĝ_b = G(μ_q, z_e[b, 0]) — z_e is shared
    # across timesteps within a sample, so we pick the first timestep.
    anchor_G = (_compute_anchor_G(sde, z[:, 0, :])                              # (B, n_q, n_q)
                if (has_drift and sde.confining_kappa > 0.0) else None)

    dr = (r / n_steps).view(B, 1, 1)                                            # (B, 1, 1)
    for k in range(n_steps):
        r_k = (k + 0.5) / n_steps * r                                           # (B,)
        beta_k = schedule.beta(r_k).view(B, 1, 1)                               # (B, 1, 1)
        L = _per_timestep_G_chol(manifold, q, z)                                # (B, H+1, n_q, n_q)

        if has_drift:
            # Chart drift b^q = −½ G⁻¹ ∇_q U^pose (form selected by Fix 3 flag).
            chart_grad_U = _anchor_drift_potential_grad(sde, q, z, anchor_G)    # (B, H+1, n_q)
            G_inv_grad_U = torch.cholesky_solve(
                chart_grad_U.reshape(B * H1, n_q, 1), L.reshape(B * H1, n_q, n_q)
            ).squeeze(-1).reshape(B, H1, n_q)
            b_q = -0.5 * G_inv_grad_U                                           # (B, H+1, n_q)
        else:
            b_q = 0.0

        eps = torch.randn_like(q)
        a = torch.linalg.solve_triangular(
            L.transpose(-1, -2), eps.unsqueeze(-1), upper=True
        ).squeeze(-1)                                                           # (B, H+1, n_q)
        q = q + beta_k * b_q * dr + (beta_k * dr).sqrt() * a

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
# Reward guidance helpers (extension.tex Sec. 9, Eq. 41-48)
# =====================================================================
#
# All guidance is computed in CHART form (G_pose-Riemannian gradient),
# matching the chart-form score output of `TrajectoryScoreNetUNetPose`.
# Autograd is used throughout per extension.tex Sec. 9.3 implementation
# note (avoids the J_l^{-T} factor from Appendix A).


def _pose_anchor_guidance_chart(
    tau: Tensor,                                                                # (B, H+1, ambient_dim_pose)
    T_anchor_Rp: tuple[Tensor, Tensor],                                         # (R_anchor (B,3,3), p_anchor (B,3))
    manifold: EmbodimentPoseGraphManifold,
    alpha_p: float,
    alpha_R: float,
    h_indices: list[int] | None = None,
) -> Tensor:
    """Pose-anchor reward gradient (chart-form, Riemannian) per extension.tex Eq. (42-45).

        R_anchor(q_h) = −( α_p ‖e_ρ‖² + α_R ‖e_ω‖² ),
            (e_ρ, e_ω) = Log_SE(3)( T_φ(q_h, z_e)^{-1} T_anchor )

        guidance_chart_h = G_pose(q_h)^{-1} · ∂_q R_anchor(q_h)
                         = −G_pose^{-1} · ∂_q [α_p ‖e_ρ‖² + α_R ‖e_ω‖²]   (autograd)

    Applied at timesteps in `h_indices` (default [H] — endpoint only).
    Returns (B, H+1, n_q), zeros at non-anchor timesteps.
    """
    B, H1, _ = tau.shape
    n_q = manifold.n_q
    out = torch.zeros(B, H1, n_q, device=tau.device, dtype=tau.dtype)
    R_a, p_a = T_anchor_Rp
    if h_indices is None:
        h_indices = [H1 - 1]

    for h in h_indices:
        q_h = tau[:, h, :n_q]
        z_h = tau[:, h, n_q + 7:]
        with torch.enable_grad():
            q_leaf = q_h.detach().clone().requires_grad_(True)
            R_phi, p_phi = manifold.T_phi_Rp(q_leaf, z_h)
            xi = log_relative_Rp(R_phi, p_phi, R_a, p_a)                        # (B, 6)
            err_sq = (alpha_p * (xi[..., :3] ** 2).sum(-1)
                       + alpha_R * (xi[..., 3:] ** 2).sum(-1))                  # (B,)
            chart_grad = torch.autograd.grad(
                err_sq.sum(), q_leaf,
                create_graph=False, retain_graph=False,
            )[0].detach()
        L = manifold.G_pose_chol(q_h, z_h)
        G_inv_grad = torch.cholesky_solve(
            chart_grad.unsqueeze(-1), L,
        ).squeeze(-1)                                                           # (B, n_q)
        # ∇_M R_anchor = −G^{-1} ∂_q err_sq  (R_anchor = −err_sq)
        out[:, h] = -G_inv_grad
    return out


def _smoothness_guidance_chart(
    tau: Tensor,
    manifold: EmbodimentPoseGraphManifold,
    alpha_vel: float,
    alpha_acc: float,
) -> Tensor:
    """Trajectory smoothness reward gradient (chart-form, Riemannian).

    Mirrors position-only `_smoothness_guidance` with the same ½-prefixed penalty
    convention (so closed-form chart gradients use small integer coefficients):
        R_vel(τ) = −½ Σ_{h=0..H-1} ‖q_{h+1} − q_h‖²
        R_acc(τ) = −½ Σ_{h=1..H-1} ‖q_{h+1} − 2 q_h + q_{h-1}‖²

        guidance_h = G_pose(q_h)^{-1} · ∂_q (α_v R_vel + α_a R_acc)
                   = −G_pose^{-1} · (α_v ∂_q ½‖vel‖² + α_a ∂_q ½‖acc‖²)

    Closed-form chart gradients (no autograd) per timestep, then G^{-1}.
    Returns chart-form (B, H+1, n_q).
    """
    B, H1, _ = tau.shape
    n_q = manifold.n_q
    if alpha_vel == 0.0 and alpha_acc == 0.0:
        return torch.zeros(B, H1, n_q, device=tau.device, dtype=tau.dtype)

    q = tau[..., :n_q]                                                          # (B, H+1, n_q)
    z = tau[..., n_q + 7:]

    # ∂(½ Σ ‖q_{h+1}−q_h‖²)/∂q_h  closed-form (matches position-only convention)
    grad_vel = torch.zeros_like(q)
    grad_vel[:, 0, :]  = -(q[:, 1, :] - q[:, 0, :])
    grad_vel[:, -1, :] =  (q[:, -1, :] - q[:, -2, :])
    if H1 > 2:
        grad_vel[:, 1:-1, :] = 2 * q[:, 1:-1, :] - q[:, 2:, :] - q[:, :-2, :]

    # ∂(½ Σ ‖q_{h+1}−2q_h+q_{h-1}‖²)/∂q_h  via shifted differences (position-only style)
    grad_acc = torch.zeros_like(q)
    if H1 >= 3:
        r = q[:, 2:, :] - 2 * q[:, 1:-1, :] + q[:, :-2, :]                       # (B, H-1, n_q)
        grad_acc[:, 1:-1, :] += -2.0 * r
        grad_acc[:, 0, :]    += +1.0 * r[:, 0, :]
        grad_acc[:, -1, :]   += +1.0 * r[:, -1, :]
        if H1 > 3:
            grad_acc[:, 2:, :]  += +1.0 * r
            grad_acc[:, :-2, :] += +1.0 * r

    chart_grad = alpha_vel * grad_vel + alpha_acc * grad_acc                    # (B, H+1, n_q)

    # Per-timestep G^{-1} (descent direction: −G^{-1} chart_grad = ∇_M R_smooth)
    q_flat = q.reshape(B * H1, n_q)
    z_flat = z.reshape(B * H1, -1)
    L = manifold.G_pose_chol(q_flat, z_flat)
    G_inv_grad = torch.cholesky_solve(
        chart_grad.reshape(B * H1, n_q, 1), L,
    ).squeeze(-1).reshape(B, H1, n_q)
    return -G_inv_grad                                                          # ∇_M R_smooth


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
    # ---- Stage 6' reward guidance (extension.tex Sec. 9) ----
    T_start_Rp: Optional[tuple[Tensor, Tensor]] = None,                         # for start anchor
    start_alpha_p: float = 0.0,
    start_alpha_R: float = 0.0,
    start_h_indices: Optional[list[int]] = None,                                # default: [0]
    T_target_Rp: Optional[tuple[Tensor, Tensor]] = None,                        # for goal anchor
    goal_alpha_p: float = 0.0,
    goal_alpha_R: float = 0.0,
    goal_h_indices: Optional[list[int]] = None,                                 # default: [H]
    smoothness_alpha_vel: float = 0.0,
    smoothness_alpha_acc: float = 0.0,
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

    # Sync the SDE's μ_q to the kwarg so Fix 3 / drift helpers see the right
    # anchor (the SDE constructor copy may live on CPU at fp32; reverse runs
    # may use a different device/dtype).
    sde.limiting_q_mean = mu_q
    sde.limiting_scale = float(limiting_scale)

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

    # Fix 3 per-batch anchor metric: Ĝ_b = G(μ_q, z_e[b]).
    anchor_G = (_compute_anchor_G(sde, z_e)                                     # (B, n_q, n_q)
                if sde.confining_kappa > 0.0 else None)

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

        # ---- Stage 6' reward guidance (extension.tex Sec. 9) ----
        # Pose start-anchor:  pulls q_h such that T_φ(q_h, z_e) ≈ T_start
        if T_start_Rp is not None and (start_alpha_p > 0.0 or start_alpha_R > 0.0):
            s_h = start_h_indices if start_h_indices is not None else [0]
            s = s + _pose_anchor_guidance_chart(
                x, T_start_Rp, manifold,
                alpha_p=start_alpha_p, alpha_R=start_alpha_R,
                h_indices=s_h,
            )
        # Pose goal-anchor:  pulls q_h such that T_φ(q_h, z_e) ≈ T_target
        if T_target_Rp is not None and (goal_alpha_p > 0.0 or goal_alpha_R > 0.0):
            g_h = goal_h_indices if goal_h_indices is not None else [H1 - 1]
            s = s + _pose_anchor_guidance_chart(
                x, T_target_Rp, manifold,
                alpha_p=goal_alpha_p, alpha_R=goal_alpha_R,
                h_indices=g_h,
            )
        # Trajectory smoothness:  vel + acc penalties
        if smoothness_alpha_vel > 0.0 or smoothness_alpha_acc > 0.0:
            s = s + _smoothness_guidance_chart(
                x, manifold,
                alpha_vel=smoothness_alpha_vel, alpha_acc=smoothness_alpha_acc,
            )

        # Limiting drift (chart, extension.tex Eq. (17)):
        #   Legacy:  b^q = −(1/2γ²) G⁻¹(q − μ)  −  (1/4) G⁻¹ ∇log det G
        #   Fix 3 :  b^q = −½ G⁻¹ [γ⁻² Ĝ(q − μ)  +  ∇U_box]
        # Form is selected inside `_anchor_drift_potential_grad` based on
        # `sde.confining_kappa`.
        q_flat = q.reshape(B * H1, n_q)
        L = manifold.G_pose_chol(q_flat, z_flat)
        try:
            chart_grad_U = _anchor_drift_potential_grad(
                sde, q, z_traj, anchor_G,
            )                                                                   # (B, H+1, n_q)
            Ginv_grad_U = torch.cholesky_solve(
                chart_grad_U.reshape(B * H1, n_q, 1), L,
            ).squeeze(-1).reshape(B, H1, n_q)
            b_q = -0.5 * Ginv_grad_U
        except RuntimeError:
            # Fallback: skip the log-det-G correction term (autograd through
            # pytorch_kinematics has rarely been seen to fail on extreme
            # configurations).  Keeps the quadratic-anchor part.
            mu_q_full = mu_q.view(1, 1, n_q)
            gamma2 = limiting_scale ** 2
            rhs = (q - mu_q_full).reshape(B * H1, n_q, 1)
            Ginv_diff = torch.cholesky_solve(rhs, L).squeeze(-1)
            b_q = -(0.5 / gamma2) * Ginv_diff.reshape(B, H1, n_q)

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
