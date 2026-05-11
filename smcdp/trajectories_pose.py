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
        limiting_scale: float | None = None,
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
        # limiting_scale is the σ_K reverse-init scale; legacy default 0.6, but
        # Method A leaves it as None and the reverse sampler auto-calibrates to
        # √τ_brown(K).  When None we still need a fallback for any forward-drift
        # legacy path; use the auto-calibrated value.
        if limiting_scale is None:
            tf_t = torch.tensor(schedule.tf, dtype=torch.float32)
            self.limiting_scale = float(schedule.integral(tf_t).clamp(min=1e-12).sqrt().item())
        else:
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

    v4.1 chart-aware behavior (joint_limit_extension §7.1):
        When `manifold` is a BoundedChartPoseManifold wrapper, the chart slot
        of state stores `u` (not `q`), and `manifold.G_pose_chol`,
        `manifold.make_x` automatically dispatch to the chart-aware overrides
        — the noise covariance becomes `G_Q^A^{-1}` and retraction becomes
        `Retr^Q_{(u,T)}(δu) = (u + δu, T_φ(ψ(u + δu), z_e))`.  No code change
        needed in this function for v4.1; the only spec deviation is the
        legacy Langevin drift path (used only if `forward_langevin_drift=True`),
        which is not part of Method A and not validated under bounded chart.
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
    """a*_pose(q_r, x_0) ∈ R^{n_q}  per extension.tex Eq. (35) / joint_limit_extension §9:

    v4 (unbounded chart, chart slot = q):
        a*_pose = G_pose^{-1} · [(q_0 − q_r) + J_pose^⊤ W · Log_SE(3)(T_φ(q_r)^{-1} T_φ(q_0))]

    v4.1 (bounded chart wrapper, chart slot = u — automatic via overrides):
        a*_Q    = G_Q^A^{-1} · [(u_0 − u_r) + (J^Q)^⊤ W · Log_SE(3)(T_φ(ψ(u_r))^{-1} T_φ(ψ(u_0)))]

    The function code is identical for both — when manifold is wrapped with
    BoundedChartPoseManifold, `T_phi_Rp(u, z)` returns T_φ(ψ(u), z) (overridden),
    `jacobian_pose(u, z)` returns J^Q = J_pose · D_ψ (overridden), and
    `G_pose_chol(u, z)` returns chol(G_Q^A) (overridden), giving the v4.1
    formula automatically.  See joint_limit_extension §9 for the structural
    bias caveat (chart-Euclidean displacement near boundary).
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
    proxy_std_mode: str = "ou",
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
        w = schedule.proxy_std(r, mode=proxy_std_mode) ** 2
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
    *,
    chart_form: str = "q",
) -> Tensor:
    """Trajectory smoothness reward gradient (chart-form, Riemannian).

    Two penalty conventions, selected by ``chart_form`` (joint_limit_extension
    v4.1 §12.3):

    chart_form="q" (DEFAULT, v4.1 spec):
        Operates on the *physical* joint configuration q = ψ(u). Equivalent
        to v4 when ψ is the identity (i.e., manifold not wrapped or wrapped
        with IdentityChart):
            R_vel(τ) = −½ Σ_{h=0..H-1} ‖ψ(u_{h+1}) − ψ(u_h)‖²
            R_acc(τ) = −½ Σ_{h=1..H-1} ‖ψ(u_{h+1}) − 2 ψ(u_h) + ψ(u_{h-1})‖²
        Rationale: physical interpretation, baseline parity (BC/DP report
        q-chart smoothness), gradient auto-damping near boundary via D_ψ.

    chart_form="u" (alternative):
        Operates on the chart coordinate u directly:
            R_vel^u = −½ Σ ‖u_{h+1} − u_h‖²,  R_acc^u = −½ Σ ‖u_{h+1} − 2 u_h + u_{h-1}‖²
        Use only when chart-Euclidean smoothness is explicitly desired.

    Riemannian gradient: ∇_M R = G_pose(u)^{-1} · ∂_u R   (autograd through ψ).

    Implementation: autograd is used uniformly to handle both forms with
    chart-aware ψ chain rule.  Numerical equivalence to v4 closed-form
    gradients holds when ψ = identity (i.e., manifold has no `chart` attr or
    chart is IdentityChart).
    """
    if chart_form not in ("q", "u"):
        raise ValueError(f"chart_form must be 'q' or 'u', got {chart_form!r}")
    B, H1, _ = tau.shape
    n_q = manifold.n_q
    if alpha_vel == 0.0 and alpha_acc == 0.0:
        return torch.zeros(B, H1, n_q, device=tau.device, dtype=tau.dtype)

    chart_slot = tau[..., :n_q]                                                 # u (or q in v4 unwrapped)
    z = tau[..., n_q + 7:]

    # Resolve which "physical" coordinate to penalize.  When chart_form="q"
    # we compose with ψ; when manifold lacks a `.chart` attribute (i.e., v4
    # unwrapped where chart slot IS physical q), ψ is implicitly identity.
    has_chart = hasattr(manifold, "chart")

    with torch.enable_grad():
        u_leaf = chart_slot.detach().clone().requires_grad_(True)
        if chart_form == "q" and has_chart:
            # Apply ψ for v4.1 q-chart smoothness (autograd handles D_ψ chain rule)
            q_phys = manifold.chart.psi(u_leaf)
        else:
            # chart_form="u" OR (chart_form="q" AND manifold unwrapped, ψ=identity)
            q_phys = u_leaf

        # R_vel = −½ Σ ‖q_{h+1}−q_h‖²  →  loss_vel = ½ Σ ‖q_{h+1}−q_h‖²
        diff_vel = q_phys[:, 1:, :] - q_phys[:, :-1, :]                          # (B, H, n_q)
        loss_vel = 0.5 * (diff_vel ** 2).sum()
        if H1 >= 3:
            diff_acc = q_phys[:, 2:, :] - 2 * q_phys[:, 1:-1, :] + q_phys[:, :-2, :]
            loss_acc = 0.5 * (diff_acc ** 2).sum()
        else:
            loss_acc = q_phys.new_zeros(())

        total = alpha_vel * loss_vel + alpha_acc * loss_acc                      # scalar
        chart_grad = torch.autograd.grad(total, u_leaf)[0].detach()              # ∂R/∂u

    # Per-timestep G^{-1} (descent direction: −G^{-1} chart_grad = ∇_M R_smooth)
    chart_slot_flat = chart_slot.reshape(B * H1, n_q)
    z_flat = z.reshape(B * H1, -1)
    L = manifold.G_pose_chol(chart_slot_flat, z_flat)
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
    limiting_q_mean: Optional[Tensor] = None,                                   # legacy: single μ_q (n_q,)
    q_init: Optional[Tensor] = None,                                            # Method A: per-traj anchor (B, n_q)
    limiting_scale: Optional[float] = None,                                     # if None → √τ_brown(K) auto-calibrated
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
    smoothness_chart_form: str = "q",                                            # v4.1 §12.3 default
) -> Tensor:
    """Reverse-time GRW on M_φ^pose^{H+1}, chart form.

    Initial sample: q_h ~ N(μ_q, γ² G_pose(μ_q, z_e)^{-1}), lifted via H_φ^pose
    (extension.tex Sec. 6 reference distribution).

    Reverse Euler step (per timestep h):
        μ^q_h = −b^q(q_{h, k}) + s^q_h(τ_k)
        δq_h  = Δr · μ^q_h + √Δr · ξ_h^q,    ξ_h^q ~ N(0, G_pose^{-1})
        q_{h, k+1} = q_{h, k} + δq_h
        x_{h, k+1} = H_φ^pose(q_{h, k+1}, z_e).

    v4.1 chart-aware behavior (joint_limit_extension §10):
        When `manifold` is a BoundedChartPoseManifold wrapper, the variable
        `q` in this function semantically holds `u` (the chart coordinate),
        and `q_init` should be passed as `u_init = ψ⁻¹(q_init)` by the caller.
        The retraction `manifold.make_x(u, z)` automatically computes
        `H_φ^Q(u, z) = (ψ(u), T_φ(ψ(u), z), z)`, satisfying joint-limit
        feasibility by construction (§13: viol(τ) = 0).  Smoothness reward
        defaults to q-chart per spec §12.3 (override via `smoothness_chart_form`).
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

    # ---- Anchor selection: Method A (per-traj q_init) vs legacy (single μ_q) ----
    use_per_traj_anchor = q_init is not None
    if use_per_traj_anchor:
        # Method A: q_init is per-batch (B, n_q), broadcast across timesteps.
        if q_init.shape != (B, n_q):
            raise ValueError(
                f"q_init shape {tuple(q_init.shape)} != ({B}, {n_q})"
            )
        anchor_q = q_init.to(device=device, dtype=dtype).unsqueeze(1).expand(B, H1, n_q).contiguous()
        mu_q = None  # not used when per-traj anchor active
    else:
        if limiting_q_mean is None:
            mu_q = torch.zeros(n_q, device=device, dtype=dtype)
        else:
            mu_q = limiting_q_mean.to(device=device, dtype=dtype)
        anchor_q = mu_q.view(1, 1, n_q).expand(B, H1, n_q).contiguous()
        # Sync SDE's μ_q (legacy drift helpers read from it on the right device/dtype).
        sde.limiting_q_mean = mu_q

    # ---- σ_K auto-calibration: √τ_brown(K) for Method A drift-free forward ----
    if limiting_scale is None:
        tf_t = torch.tensor(schedule.tf, device=device, dtype=dtype)
        sigma_K = float(schedule.integral(tf_t).clamp(min=1e-12).sqrt().item())
    else:
        sigma_K = float(limiting_scale)
    sde.limiting_scale = sigma_K

    # ---- Initial chart-Gaussian sample at the anchor ----
    z_flat = z_traj.reshape(B * H1, n_z)
    anchor_q_flat = anchor_q.reshape(B * H1, n_q)
    L_anchor = manifold.G_pose_chol(anchor_q_flat, z_flat)                      # (B*H1, n_q, n_q)
    eps_init = torch.randn(B, H1, n_q, device=device, dtype=dtype)
    a_init = torch.linalg.solve_triangular(
        L_anchor.transpose(-1, -2),
        eps_init.reshape(B * H1, n_q, 1),
        upper=True,
    ).squeeze(-1)                                                               # (B*H1, n_q)
    q = anchor_q_flat + sigma_K * a_init
    q = q.reshape(B, H1, n_q)
    x = manifold.make_x(q.reshape(B * H1, n_q), z_flat).reshape(B, H1, manifold.ambient_dim)

    # Fix 3 per-batch anchor metric: Ĝ_b = G(μ_q, z_e[b]).  None for Method A
    # (no anchor metric in pure-Brownian forward).
    anchor_G = (_compute_anchor_G(sde, z_e)                                     # (B, n_q, n_q)
                if (sde.confining_kappa > 0.0 and not use_per_traj_anchor) else None)

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
        # Trajectory smoothness:  vel + acc penalties (chart_form per v4.1 §12.3)
        if smoothness_alpha_vel > 0.0 or smoothness_alpha_acc > 0.0:
            s = s + _smoothness_guidance_chart(
                x, manifold,
                alpha_vel=smoothness_alpha_vel, alpha_acc=smoothness_alpha_acc,
                chart_form=smoothness_chart_form,
            )

        # Limiting drift (chart, Anderson reverse formula).  Method A: when the
        # forward SDE has NO drift (sde.forward_langevin_drift == False AND
        # confining_kappa == 0), the reverse drift's b_fwd term is also 0 — only
        # the score remains.  This restores forward/reverse consistency that the
        # legacy code path (which always computed an OU-style b_q) silently broke.
        q_flat = q.reshape(B * H1, n_q)
        L = manifold.G_pose_chol(q_flat, z_flat)
        if (not getattr(sde, "forward_langevin_drift", False)) and (sde.confining_kappa == 0.0):
            # Method A: pure Brownian forward → reverse b_fwd = 0
            b_q = torch.zeros(B, H1, n_q, device=device, dtype=dtype)
        else:
            #   Legacy:  b^q = −(1/2γ²) G⁻¹(q − μ)  −  (1/4) G⁻¹ ∇log det G
            #   Fix 3 :  b^q = −½ G⁻¹ [γ⁻² Ĝ(q − μ)  +  ∇U_box]
            try:
                chart_grad_U = _anchor_drift_potential_grad(
                    sde, q, z_traj, anchor_G,
                )                                                               # (B, H+1, n_q)
                Ginv_grad_U = torch.cholesky_solve(
                    chart_grad_U.reshape(B * H1, n_q, 1), L,
                ).squeeze(-1).reshape(B, H1, n_q)
                b_q = -0.5 * Ginv_grad_U
            except RuntimeError:
                if mu_q is None:
                    b_q = torch.zeros(B, H1, n_q, device=device, dtype=dtype)
                else:
                    mu_q_full = mu_q.view(1, 1, n_q)
                    gamma2 = sigma_K ** 2
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

    Args:
        proxy_std_mode: passed to `schedule.proxy_std(t, mode=...)`.  Use
            ``"brownian"`` (√I) for Method A drift-free forward; ``"ou"`` for
            legacy VP-SDE convention.
    """

    def __init__(self, net: TrajectoryScoreNetUNetPose, sde, std_trick: bool = True,
                  proxy_std_mode: str = "ou"):
        super().__init__()
        self.net = net
        self.sde = sde
        self.std_trick = std_trick
        self.proxy_std_mode = str(proxy_std_mode)

    def forward(self, tau: Tensor, t: Tensor, goal_cond: Tensor | None = None) -> Tensor:
        out = self.net(tau, t, goal_cond=goal_cond)
        if self.std_trick:
            B = tau.shape[0]
            sigma = self.sde.schedule.proxy_std(t, mode=self.proxy_std_mode).clamp(min=1e-6).view(B, 1, 1)
            out = out / sigma
        return out


# =====================================================================
# joint_limit_extension v5.1 — chart-space OU SDE (closed-form transition)
# =====================================================================
#
# Implements joint_limit_extension.tex v5.1:
#     Forward SDE (chart u):  du_r = -½ β(r) u_r dr + √β(r) Ḡ_Q^{-1/2} dW_r
#     Closed-form transition: p_{r|0}(u_r | u_0) = N(α(r) u_0, σ²(r) Ḡ_Q^{-1})
#         α(r) = exp(-τ(r)/2),  σ²(r) = 1 - exp(-τ(r)),  τ(r) = ∫_0^r β ds
#     Exact Euclidean score:  ∇_{u_r} log p_{r|0} = -Ḡ_Q (u_r - α u_0) / σ²(r)
#     Stationary p_∞ = N(0, Ḡ_Q^{-1})   ← IK-free reference (spec §11.2)
#     Reverse SDE (Convention 1, Δr < 0):
#         du_r = [-½ β u_r - β Ḡ_Q^{-1} s_θ^u] dr + √β Ḡ_Q^{-1/2} d\bar W_r
#
# This module is the replacement for the v4.1 Method-A drift-free Brownian
# pipeline above.  Score-network architecture (TrajectoryScoreNetUNetPose)
# and chart-aware overrides (BoundedChartPoseManifold) are unchanged — the
# only network-input change required by v5.1 is the absence of any
# q_init / IK-derived anchor, which is already the case for the existing
# net (input is [u, r, h/H, z_e, T_start, T_target] — see spec §9).


class PoseChartOUSDE:
    """v5.1 chart-space OU SDE container (joint_limit_extension.tex §8).

    Forward SDE on chart u ∈ R^{n_q}:
        du_r = -½ β(r) u_r dr + √β(r) Ḡ_Q^{-1/2} dW_r,   u_0 ~ p_data
    Closed-form transition:
        p_{r|0}(u_r | u_0) = N(α(r) u_0, σ²(r) Ḡ_Q^{-1})
        α(r) = exp(-τ(r)/2),  σ²(r) = 1 - exp(-τ(r)),  τ(r) = ∫_0^r β(s) ds
    Stationary (IK-free reference, spec §11.2):
        p_∞ = N(0, Ḡ_Q^{-1})

    The reference metric Ḡ_Q is a *constant* SPD matrix — independent of u, z_e,
    c, and sample.  Three options per spec §7.2:
        - "identity":  Ḡ_Q = I_{n_q}                                   (default)
        - "origin":    Ḡ_Q = G_Q(u=0, z_e_bar)  (fixed embodiment z_e_bar)
        - "data_mean": Ḡ_Q = E_{(u_0, z_e) ~ p_data}[G_Q(u_0, z_e)]
    Only "identity" is wired here; the other modes raise NotImplementedError
    (hooks reserved for v5.1 ablations).
    """

    def __init__(
        self,
        manifold: EmbodimentPoseGraphManifold,
        schedule,
        *,
        gbar_mode: str = "identity",
    ):
        self.manifold = manifold
        self.schedule = schedule
        if gbar_mode not in ("identity", "origin", "data_mean"):
            raise ValueError(f"unknown gbar_mode '{gbar_mode}'")
        if gbar_mode != "identity":
            raise NotImplementedError(
                f"gbar_mode='{gbar_mode}' not yet wired (v5.1 ablation hook); "
                f"use 'identity' for the spec default."
            )
        self.gbar_mode = str(gbar_mode)

    @property
    def t0(self) -> float:
        return self.schedule.t0

    @property
    def tf(self) -> float:
        return self.schedule.tf

    @property
    def n_q(self) -> int:
        return self.manifold.n_q

    # ------------------ Ḡ_Q applies (vectorized over leading dims) ------------------

    def gbar_apply(self, x: Tensor) -> Tensor:
        """Compute (Ḡ_Q · x)  along the last axis.  Identity for default mode
        (Ḡ_Q = I_{n_q})  →  no-op."""
        if self.gbar_mode == "identity":
            return x
        raise NotImplementedError

    def gbar_inv_apply(self, x: Tensor) -> Tensor:
        """Compute (Ḡ_Q^{-1} · x)  along the last axis.  Identity default."""
        if self.gbar_mode == "identity":
            return x
        raise NotImplementedError

    def gbar_inv_sqrt_apply(self, x: Tensor) -> Tensor:
        """Compute (Ḡ_Q^{-1/2} · x)  along the last axis.  Identity default."""
        if self.gbar_mode == "identity":
            return x
        raise NotImplementedError


# ---------------------------------------------------------------------
# Forward sampler — closed-form OU transition (spec §8.1, §11)
# ---------------------------------------------------------------------


def traj_forward_ou_chart_pose(
    sde: PoseChartOUSDE,
    tau_0: Tensor,                                                              # (B, H+1, ambient_dim_pose)
    r: Tensor,                                                                  # (B,) diffusion times
) -> Tensor:
    """Closed-form OU forward sample on chart u (spec §8.1).

    For each batch element b and timestep h, draws
        u_{h, r_b}  =  α(r_b) · u_{h, 0}  +  σ(r_b) · Ḡ_Q^{-1/2} · ε_{b, h},
        ε_{b, h}  ~  N(0, I_{n_q})
    so that  u_{h, r_b} | u_{h, 0}  ~  N(α u_0, σ² Ḡ_Q^{-1})  exactly  (spec §8.1.2).

    Returns the lifted trajectory in ambient form via graph retraction
    `manifold.make_x(u_r, z) = (u_r, T̃_φ(u_r, z), z)`, i.e. (B, H+1, ambient_dim).

    No iterative discretization, no Varadhan approximation, no drift term —
    the constant-coefficient OU admits an exact Gaussian transition.
    """
    manifold = sde.manifold
    schedule = sde.schedule
    n_q = manifold.n_q
    B, H1, _ = tau_0.shape

    u_0 = tau_0[..., :n_q]                                                       # (B, H+1, n_q)
    z_traj = tau_0[..., n_q + 7:]                                                # (B, H+1, n_z) frozen

    alpha_r = schedule.alpha(r).view(B, 1, 1)                                    # (B, 1, 1)
    sigma_r = schedule.sigma(r).view(B, 1, 1)                                    # (B, 1, 1)
    eps = torch.randn_like(u_0)                                                  # (B, H+1, n_q)
    noise = sde.gbar_inv_sqrt_apply(eps)                                         # Ḡ_Q^{-1/2} ε
    u_r = alpha_r * u_0 + sigma_r * noise                                        # (B, H+1, n_q)

    z_flat = z_traj.reshape(B * H1, -1)
    u_r_flat = u_r.reshape(B * H1, n_q)
    x_flat = manifold.make_x(u_r_flat, z_flat)
    return x_flat.reshape(B, H1, manifold.ambient_dim)


# ---------------------------------------------------------------------
# Score-matching loss — exact OU target, optional pose-consistency reg
# (spec §10)
# ---------------------------------------------------------------------


def traj_ou_score_loss_pose(
    score_fn,                                                                   # (τ, t, goal_cond=None) → (B, H+1, n_q)  Euclidean chart score
    sde: PoseChartOUSDE,
    tau_0: Tensor,
    *,
    eps: float = 1e-3,
    weight: str = "sigma4",                                                     # spec §10 default w(r) = σ²(r)²
    metric: str = "I",                                                          # spec §10: M ∈ {I, G^{-1}, G}
    goal_cond: Optional[Tensor] = None,
    cond_drop_prob: float = 0.0,
    endpoint_weight: float = 1.0,
    _shared: Optional[dict] = None,                                             # internal: share r, tau_r with pose-reg
) -> Tensor:
    """Exact OU score-matching loss (joint_limit_extension v5.1 §10):

        L_score = E_{r, u_0, u_r} [ w(r) · Σ_h ‖s_θ^u_h - s*_h‖²_M ]
        s*(u_r, u_0; r)  =  - Ḡ_Q · (u_r - α(r) u_0) / σ²(r)         (spec §10 boxed)

    Default M = I_{n_q}, w(r) = σ²(r)² (SNR-aware, spec §10).
    The chart target s* is the *exact* Euclidean OU score; no Varadhan
    approximation (spec §10 "exact, not Varadhan-approximated").
    """
    manifold = sde.manifold
    schedule = sde.schedule
    n_q = manifold.n_q
    B, H1, _ = tau_0.shape
    device, dtype = tau_0.device, tau_0.dtype

    # 1. Sample diffusion time r ∈ [eps, tf]  — or reuse from _shared
    if _shared is not None and "r" in _shared:
        r = _shared["r"]
        tau_r = _shared["tau_r"]
    else:
        r = eps + (schedule.tf - eps) * torch.rand(B, device=device, dtype=dtype)
        tau_r = traj_forward_ou_chart_pose(sde, tau_0, r)
        if _shared is not None:
            _shared["r"] = r
            _shared["tau_r"] = tau_r

    # 2. CFG cond dropout
    if goal_cond is not None and cond_drop_prob > 0.0:
        drop_mask = torch.rand(B, device=device) < cond_drop_prob
        if drop_mask.any():
            goal_cond = goal_cond.clone()
            goal_cond[drop_mask] = 0.0

    # 3. Exact OU score target  s* = -Ḡ_Q (u_r - α u_0) / σ²(r)
    u_0 = tau_0[..., :n_q]
    u_r = tau_r[..., :n_q]
    alpha_r = schedule.alpha(r).view(B, 1, 1)
    sigma2_r = schedule.sigma2(r).clamp(min=1e-12).view(B, 1, 1)
    residual = u_r - alpha_r * u_0                                              # (B, H+1, n_q)
    target = -sde.gbar_apply(residual) / sigma2_r                               # (B, H+1, n_q)
    if _shared is not None:
        _shared["target"] = target

    # 4. Network Euclidean chart score
    if goal_cond is not None:
        s_chart = score_fn(tau_r, r, goal_cond=goal_cond)
    else:
        s_chart = score_fn(tau_r, r)
    if _shared is not None:
        _shared["s_chart"] = s_chart

    diff = s_chart - target                                                     # (B, H+1, n_q)

    # 5. Weighting metric M
    z = tau_r[..., n_q + 7:]
    if metric == "I":
        sq_per_pt_h = (diff * diff).sum(-1)                                     # (B, H+1)
    elif metric == "G_inv":
        u_r_flat = u_r.reshape(B * H1, n_q)
        z_flat = z.reshape(B * H1, -1)
        L = manifold.G_pose_chol(u_r_flat, z_flat)                              # (B*H1, n_q, n_q)
        diff_flat = diff.reshape(B * H1, n_q, 1)
        a = torch.linalg.solve_triangular(L, diff_flat, upper=False).squeeze(-1)
        sq_per_pt_h = a.pow(2).sum(-1).reshape(B, H1)
    elif metric == "G":
        u_r_flat = u_r.reshape(B * H1, n_q)
        z_flat = z.reshape(B * H1, -1)
        G = manifold.G_pose(u_r_flat, z_flat)                                   # (B*H1, n_q, n_q)
        diff_flat = diff.reshape(B * H1, n_q)
        sq_per_pt_h = (diff_flat.unsqueeze(-2) @ G @ diff_flat.unsqueeze(-1)
                       ).squeeze(-1).squeeze(-1).reshape(B, H1)
    else:
        raise ValueError(f"metric must be 'I' | 'G_inv' | 'G', got {metric!r}")

    if endpoint_weight != 1.0:
        h_weights = torch.ones(H1, device=device, dtype=dtype)
        h_weights[-1] = float(endpoint_weight)
        sq_per_traj = (sq_per_pt_h * h_weights).sum(-1)                         # (B,)
    else:
        sq_per_traj = sq_per_pt_h.sum(-1)

    # 6. SNR-aware weighting w(r)
    if weight == "sigma4":
        w = schedule.sigma2(r).pow(2)                                           # spec §10 default
    elif weight == "sigma2":
        w = schedule.sigma2(r)
    elif weight == "beta":
        w = schedule.beta(r)
    elif weight == "none":
        w = torch.ones_like(r)
    else:
        raise ValueError(f"unknown weight '{weight}'")

    return (w * sq_per_traj).mean()


def traj_pose_consistency_loss(
    score_fn,
    sde: PoseChartOUSDE,
    tau_0: Tensor,
    *,
    eps: float = 1e-3,
    tau_cutoff: float = 0.5,                                                    # spec §10 default
    goal_cond: Optional[Tensor] = None,
    _shared: Optional[dict] = None,                                             # internal: share r, tau_r, s_chart
) -> Tensor:
    """Auxiliary pose-geometric consistency regularizer (v5.1 §10).

        L_pose = E_{r, u_0, u_r} [ λ_p(r) · Σ_h ‖J^Q(u_{h,r}, z_e) · s_θ^u_h
                                              - (1/τ(r)) Log_SE3(T̃_φ(u_{h,r})^{-1} T̃_φ(u_{h,0}))‖²_W ]
        λ_p(r) = 1[τ(r) < τ_cutoff],   τ_cutoff ≈ 0.5      (down-weights large-r where Varadhan breaks)

    The pose component is NOT an exact transition score (since T̃_φ is a
    deterministic function of u, so the u-OU does NOT induce a closed-form
    Gaussian on T).  This term is retained as a Varadhan-style consistency
    prior, valid in the small-r regime — see spec §10 "Validity regime".

    Added to L_score with factor μ_pose (default 0).
    """
    manifold = sde.manifold
    schedule = sde.schedule
    n_q = manifold.n_q
    B, H1, _ = tau_0.shape
    device, dtype = tau_0.device, tau_0.dtype

    # Reuse r / tau_r / s_chart from L_score path when available.
    if _shared is not None and "tau_r" in _shared:
        r = _shared["r"]
        tau_r_x = _shared["tau_r"]
        s_chart = _shared.get("s_chart")
    else:
        r = eps + (schedule.tf - eps) * torch.rand(B, device=device, dtype=dtype)
        tau_r_x = traj_forward_ou_chart_pose(sde, tau_0, r)
        s_chart = None

    if s_chart is None:
        if goal_cond is not None:
            s_chart = score_fn(tau_r_x, r, goal_cond=goal_cond)
        else:
            s_chart = score_fn(tau_r_x, r)

    u_r = tau_r_x[..., :n_q]
    u_0 = tau_0[..., :n_q]
    z = tau_r_x[..., n_q + 7:]

    u_r_flat = u_r.reshape(B * H1, n_q)
    u_0_flat = u_0.reshape(B * H1, n_q)
    z_flat = z.reshape(B * H1, -1)
    s_flat = s_chart.reshape(B * H1, n_q)

    # J^Q at u_r  ∈ R^{(B*H1) × 6 × n_q} ;  body-frame on bounded chart.
    Jq_flat = manifold.jacobian_pose(u_r_flat, z_flat)
    Js_flat = (Jq_flat @ s_flat.unsqueeze(-1)).squeeze(-1)                       # (B*H1, 6)

    # Pose-tangent displacement from r to 0:  Log_SE(3)( T̃_φ(u_r)^{-1} T̃_φ(u_0) )
    R_r, p_r = manifold.T_phi_Rp(u_r_flat, z_flat)
    R_0, p_0 = manifold.T_phi_Rp(u_0_flat, z_flat)
    xi = log_relative_Rp(R_r, p_r, R_0, p_0)                                    # (B*H1, 6)

    tau_brown = schedule.tau(r).clamp(min=1e-12)                                # (B,)
    tau_per_h = tau_brown.view(B, 1).expand(B, H1).reshape(B * H1, 1)            # (B*H1, 1)
    target_pose = xi / tau_per_h                                                # (B*H1, 6)

    W = manifold._W_diag(u_r_flat)                                              # (6,)
    diff = Js_flat - target_pose                                                # (B*H1, 6)
    sq_per_pt = (diff * diff * W.unsqueeze(0)).sum(-1).reshape(B, H1)           # (B, H+1)
    sq_per_traj = sq_per_pt.sum(-1)                                             # (B,)

    # Indicator down-weight  λ_p(r) = 1[τ(r) < τ_cutoff]
    lam_p = (tau_brown < tau_cutoff).to(dtype=dtype)                            # (B,)
    return (lam_p * sq_per_traj).mean()


def traj_total_loss_v51_pose(
    score_fn,
    sde: PoseChartOUSDE,
    tau_0: Tensor,
    *,
    eps: float = 1e-3,
    weight: str = "sigma4",
    metric: str = "I",
    goal_cond: Optional[Tensor] = None,
    cond_drop_prob: float = 0.0,
    endpoint_weight: float = 1.0,
    mu_pose: float = 0.0,
    tau_cutoff: float = 0.5,
) -> Tensor:
    """v5.1 total loss: L = L_score + μ_pose · L_pose   (spec §10).

    When μ_pose == 0 (default), L_pose is not computed (clean exact OU score
    matching baseline).  When μ_pose > 0, the r-sample, u_r noise, and
    network score are shared between the two terms for efficiency — both
    losses observe the SAME forward sample so their gradients are coherent.
    """
    if mu_pose == 0.0:
        return traj_ou_score_loss_pose(
            score_fn, sde, tau_0, eps=eps, weight=weight, metric=metric,
            goal_cond=goal_cond, cond_drop_prob=cond_drop_prob,
            endpoint_weight=endpoint_weight,
        )

    shared: dict = {}
    L_score = traj_ou_score_loss_pose(
        score_fn, sde, tau_0, eps=eps, weight=weight, metric=metric,
        goal_cond=goal_cond, cond_drop_prob=cond_drop_prob,
        endpoint_weight=endpoint_weight, _shared=shared,
    )
    L_pose = traj_pose_consistency_loss(
        score_fn, sde, tau_0, eps=eps, tau_cutoff=tau_cutoff,
        goal_cond=goal_cond, _shared=shared,
    )
    return L_score + mu_pose * L_pose


# ---------------------------------------------------------------------
# Reverse sampler — Convention-1 reverse OU + graph retraction (spec §11)
# ---------------------------------------------------------------------


@torch.no_grad()
def traj_reverse_ou_chart_pose(
    sde: PoseChartOUSDE,
    score_fn,                                                                   # (τ, t, goal_cond=None) → (B, H+1, n_q)
    n_samples: int,
    H: int,
    n_steps: int = 200,
    goal_cond: Optional[Tensor] = None,
    z_e: Optional[Tensor] = None,                                               # (B, n_z)
    eps: float = 2e-4,
    device=None, dtype=torch.float32,
    score_postprocess=None,
    # ---- reward guidance (spec §11.4 + §12) ----
    T_start_Rp: Optional[tuple[Tensor, Tensor]] = None,
    start_alpha_p: float = 0.0,
    start_alpha_R: float = 0.0,
    start_h_indices: Optional[list[int]] = None,
    T_target_Rp: Optional[tuple[Tensor, Tensor]] = None,
    goal_alpha_p: float = 0.0,
    goal_alpha_R: float = 0.0,
    goal_h_indices: Optional[list[int]] = None,
    smoothness_alpha_vel: float = 0.0,
    smoothness_alpha_acc: float = 0.0,
    smoothness_chart_form: str = "q",
) -> Tensor:
    """Reverse-time chart-space OU sampler with graph retraction (v5.1 §11).

    Reference distribution (IK-FREE — spec §11.2):
        u_h^{(K)}  ~  N(0, Ḡ_Q^{-1})         (data-independent, conditioning-independent)
        x_h^{(K)}  =  H̃_φ^{b-pose}(u_h^{(K)}, z_e) = (u_h, T̃_φ(u_h, z_e), z_e)

    Reverse SDE (Convention 1: forward time r, Δr < 0 — spec §11.3):
        du_r = [-½ β(r) u_r - β(r) Ḡ_Q^{-1} s_θ^u(u_r, r, c, z_e)] dr
               + √β(r) Ḡ_Q^{-1/2} d\bar W_r

    Discrete Euler-Maruyama with forward-positive dr := |Δr|  > 0:
        u_{k+1}  =  u_k  +  [½ β(r_k) u_k + β(r_k) Ḡ_Q^{-1} (s_θ + m_k G_Q^{-1} ∇R)] · dr
                       +  √(β(r_k) · dr) · Ḡ_Q^{-1/2} · ξ_k

    (Sign verification, spec §11.3.4: drift in `-½ β u · Δr` with Δr < 0 gives
    `+½ β u · dr` — the reverse OU mirror pushing u away from origin, mirroring
    the forward attraction; score term `-β Ḡ^{-1} s · Δr` gives `+β Ḡ^{-1} s · dr`,
    pointing toward higher-density regions.  Signs are consistent.)

    No IK seed, no q_warm, no per-trajectory anchor — multimodality is captured
    by the score network alone, conditional on (T_start, T_target, z_e).
    """
    manifold = sde.manifold
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
    z_traj = z_e.unsqueeze(1).expand(B, H1, n_z).contiguous()
    z_flat = z_traj.reshape(B * H1, n_z)

    # ---- IK-FREE initial sample  u ~ N(0, Ḡ_Q^{-1})  (spec §11.2) ----
    eps_init = torch.randn(B, H1, n_q, device=device, dtype=dtype)
    u = sde.gbar_inv_sqrt_apply(eps_init)                                       # (B, H+1, n_q)
    x = manifold.make_x(
        u.reshape(B * H1, n_q), z_flat,
    ).reshape(B, H1, manifold.ambient_dim)

    # Reverse-time grid: r descends from tf to eps
    r_grid = torch.linspace(schedule.tf, eps, n_steps + 1, device=device, dtype=dtype)

    for k in range(n_steps):
        r_k = r_grid[k]
        r_kp1 = r_grid[k + 1]
        dr = (r_k - r_kp1).abs()                                                # > 0 (forward-positive |Δr|)
        beta_k = schedule.beta(r_k.expand(B)).view(B, 1, 1)                     # (B, 1, 1)

        # ---- Network Euclidean chart score ----
        if goal_cond is not None:
            s = score_fn(x, r_k.expand(B), goal_cond=goal_cond)
        else:
            s = score_fn(x, r_k.expand(B))
        if score_postprocess is not None:
            s = score_postprocess(s, r_k)

        # ---- Reward guidance correction (spec §11.4, §12) ----
        # The helpers already produce  -G_Q^{-1} ∂_u R  (the natural-gradient-
        # preconditioned ascent direction on  -R) — they are added directly to
        # the score in the same convention as the v4.1 path.  Outer Ḡ_Q^{-1}
        # is applied below.
        if T_start_Rp is not None and (start_alpha_p > 0.0 or start_alpha_R > 0.0):
            s_h = start_h_indices if start_h_indices is not None else [0]
            s = s + _pose_anchor_guidance_chart(
                x, T_start_Rp, manifold,
                alpha_p=start_alpha_p, alpha_R=start_alpha_R,
                h_indices=s_h,
            )
        if T_target_Rp is not None and (goal_alpha_p > 0.0 or goal_alpha_R > 0.0):
            g_h = goal_h_indices if goal_h_indices is not None else [H1 - 1]
            s = s + _pose_anchor_guidance_chart(
                x, T_target_Rp, manifold,
                alpha_p=goal_alpha_p, alpha_R=goal_alpha_R,
                h_indices=g_h,
            )
        if smoothness_alpha_vel > 0.0 or smoothness_alpha_acc > 0.0:
            s = s + _smoothness_guidance_chart(
                x, manifold,
                alpha_vel=smoothness_alpha_vel,
                alpha_acc=smoothness_alpha_acc,
                chart_form=smoothness_chart_form,
            )

        # ---- Ḡ_Q^{-1} (score + guidance) ----
        gbar_inv_s = sde.gbar_inv_apply(s)                                      # (B, H+1, n_q)

        # ---- Forward-positive-dr drift:  ½ β u + β · Ḡ_Q^{-1} · s_total ----
        drift = 0.5 * beta_k * u + beta_k * gbar_inv_s                          # (B, H+1, n_q)

        # ---- Noise:  √(β · dr) · Ḡ_Q^{-1/2} · ξ ----
        xi = torch.randn(B, H1, n_q, device=device, dtype=dtype)
        noise = (beta_k * dr).sqrt() * sde.gbar_inv_sqrt_apply(xi)              # (B, H+1, n_q)

        u = u + drift * dr + noise

        # ---- Graph retract:  x = (u, T̃_φ(ψ(u), z), z) ----
        x = manifold.make_x(
            u.reshape(B * H1, n_q), z_flat,
        ).reshape(B, H1, manifold.ambient_dim)

    return x
