"""Pose-aware bimodal trajectory demos for Franka 7-DoF.

Pose extension of `demo_gen.py`'s `FrankaBimodalReachingDemo`:
target conditioning is a full SE(3) pose (R_target, p_target), not just position.

Pose-IK uses damped least-squares with body-frame error twist:
    e = Log_SE(3)(T(q)^{-1} T_target) ∈ R^6
    q ← q + α · J_pose^+ · e    + α_null · (I − J^+ J)(q_rest − q)
where J_pose^+ = J^⊤ (J J^⊤ + λ² I_6)^{-1} is the DLS pseudo-inverse.

Same EE pose target with two distinct rest postures gives bimodal demos
(extension of position-only redundancy).

Returns from `sample(n)`:
    x         : (n, H+1, ambient_dim_pose=15)   on M_φ^pose(z_e)
    branch_A  : (n,)                            ground-truth mode label
    z_e       : (n, 1)
    T_target  : (n, 7)                          (q_R, p) storage form
    T_start   : (n, 7)
"""
from __future__ import annotations

import torch
from torch import Tensor

from smcdp.lie_se3 import (
    log_relative_Rp, exp_SO3, R_to_quat, Rp_to_pose7,
)
from smcdp.manifolds_pose import Franka7DoFPose


def _dls_ik_pose_step(
    arm: Franka7DoFPose,
    q: Tensor,                                                                # (B, 7)
    z_e: Tensor,                                                              # (B, 1)
    T_target_Rp: tuple[Tensor, Tensor],                                       # (R_t, p_t)
    q_rest: Tensor,                                                           # (B, 7)
    alpha: float = 0.5,
    alpha_null: float = 0.3,
    lam: float = 0.05,
    clamp_to_limits: bool = False,
    clamp_margin_frac: float = 0.001,
) -> Tensor:
    """One DLS pose-IK iteration: body-frame twist + null-space rest bias.

    Uses the analytic Franka body-frame Jacobian (no learned residual) for
    deterministic, fast IK.

    `clamp_to_limits` (Experiment_plan.md §2.2 Tier 1/2): if True, clip the
    post-step q to (q_min + δ, q_max - δ) where δ = clamp_margin_frac · q_range.
    This emulates a real-robot joint-limit-aware IK controller and ensures
    boundary-active demo trajectories stay strictly feasible.  Default False
    preserves Tier 0 (control) behavior — IK output may exceed limits when
    seeds are aggressive, which is the failure mode we observed for v1 attempts.
    """
    R_q, p_q = arm.T_phi_Rp(q, z_e)                                           # (B, 3, 3), (B, 3)
    e = log_relative_Rp(R_q, p_q, *T_target_Rp)                               # (B, 6) body-frame twist
    J = arm.jacobian_pose(q, z_e)                                             # (B, 6, 7)
    JJt = J @ J.transpose(-1, -2)                                              # (B, 6, 6)
    eye6 = torch.eye(6, device=q.device, dtype=q.dtype)
    Jpinv = J.transpose(-1, -2) @ torch.linalg.inv(JJt + (lam ** 2) * eye6)   # (B, 7, 6)
    d_main = (Jpinv @ e.unsqueeze(-1)).squeeze(-1)                             # (B, 7)
    eye7 = torch.eye(7, device=q.device, dtype=q.dtype)
    null_proj = eye7 - Jpinv @ J                                               # (B, 7, 7)
    d_null = (null_proj @ (q_rest - q).unsqueeze(-1)).squeeze(-1)              # (B, 7)
    q_next = q + alpha * d_main + alpha_null * d_null
    if clamp_to_limits:
        q_lo, q_hi = arm.joint_limits(device=q.device, dtype=q.dtype)         # (n_q,) each
        q_range = q_hi - q_lo
        delta = clamp_margin_frac * q_range
        q_next = torch.minimum(torch.maximum(q_next, q_lo + delta), q_hi - delta)
    return q_next


class FrankaBimodalReachingDemoPose:
    """Pose-target bimodal Franka demo generator (extension of FrankaBimodalReachingDemo).

    Trajectory generation:
      1. Sample p_start, p_end ~ Uniform(p_box).
      2. Sample R_target = R_anchor · exp_SO3(ω),  ω_i ~ Uniform[-30°, +30°].
         R_start = R_anchor (constant gripper-down by default).
      3. SE(3) waypoint interpolation: p(h) linear in p, R(h) via Slerp from
         R_start to R_target  (using log_SO3 trick).
      4. Sample mode ∈ {A, B} with P(A) = `branch_p_A`.
      5. Sample z_e ~ Uniform(z_range)  (frozen across trajectory).
      6. q(0) = q_rest_mode + jitter_q · ε.
      7. For h = 1..H, warm-start from q(h-1) and run `n_ik_steps` of
         pose-DLS-IK with null-space bias toward q_rest_mode.
      8. Lift each timestep via `manifold.make_x(q, z_e)` (pose manifold).

    NOTE: The bimodal mode signal is in the JOINT trajectory (different
    rest postures for A and B), not the EE pose trajectory — same target
    pose, different paths, same kinematic redundancy story as position-only.
    """

    def __init__(
        self,
        manifold: Franka7DoFPose,
        ik_arm: Franka7DoFPose,
        H: int,
        q_rest_A,
        q_rest_B,
        p_box_lo=(0.40, -0.10, 0.40),
        p_box_hi=(0.55,  0.10, 0.55),
        z_e_range: tuple[float, float] = (0.05, 0.15),
        branch_p_A: float = 0.5,
        jitter_q: float = 0.05,
        n_ik_steps: int = 8,
        ik_alpha: float = 0.5,
        ik_alpha_null: float = 0.3,
        ik_lam: float = 0.05,
        # Anchor orientation (gripper pointing down — standard Franka tool frame).
        R_anchor_axis_angle: tuple[float, float, float] = (3.14159265, 0.0, 0.0),
        # Target rotation perturbation: each axis-angle component ~ Uniform[-perturb, +perturb] (rad).
        target_perturb_rad: float = 0.5235987756,                              # 30° in radians
        # Experiment_plan.md §2.2 Tier 1/2: clamp IK output to (q_min+δ, q_max-δ)
        # to keep boundary-active demos strictly feasible.  Default False
        # preserves Tier 0 (control) behavior.
        ik_clamp_to_limits: bool = False,
        ik_clamp_margin_frac: float = 0.001,                                   # δ = 0.1% of q_range
    ):
        assert getattr(manifold, "n_q", 0) == 7
        assert getattr(manifold, "n_z", 0) == 1
        self.manifold = manifold
        self.ik_arm = ik_arm
        self.H = int(H)
        self.q_rest_A = torch.as_tensor(list(q_rest_A), dtype=torch.float32)
        self.q_rest_B = torch.as_tensor(list(q_rest_B), dtype=torch.float32)
        assert self.q_rest_A.numel() == 7 and self.q_rest_B.numel() == 7
        self.p_box_lo = torch.as_tensor(list(p_box_lo), dtype=torch.float32)
        self.p_box_hi = torch.as_tensor(list(p_box_hi), dtype=torch.float32)
        self.z_lo, self.z_hi = z_e_range
        self.branch_p_A = float(branch_p_A)
        self.jitter_q = float(jitter_q)
        self.n_ik_steps = int(n_ik_steps)
        self.ik_clamp_to_limits = bool(ik_clamp_to_limits)
        self.ik_clamp_margin_frac = float(ik_clamp_margin_frac)
        self.ik_alpha = float(ik_alpha)
        self.ik_alpha_null = float(ik_alpha_null)
        self.ik_lam = float(ik_lam)
        self.R_anchor_aa = torch.as_tensor(list(R_anchor_axis_angle), dtype=torch.float32)
        self.target_perturb_rad = float(target_perturb_rad)

    @torch.no_grad()
    def sample(
        self, n: int, device=None, dtype=torch.float32,
        T_target: Tensor | None = None,                                       # (n, 7) storage form
        T_start: Tensor | None = None,
    ):
        """Returns (x, branch_A, z_e, T_target_used, T_start_used).

        x            : (n, H+1, manifold.ambient_dim) ∈ M_φ^pose
        branch_A     : (n,) bool
        z_e          : (n, 1)
        T_target/T_start_used : (n, 7) storage form (q_R, p).

        v4.1 chart-aware behavior (joint_limit_extension §10.3):
            IK produces *physical* q ∈ (q_min, q_max).  When `self.manifold`
            is a `BoundedChartPoseManifold`, the chart slot of x stores the
            chart coordinate u = ψ⁻¹(q) (with η-clip safety, init-only per
            spec §10.3) — NOT q.  The T-block is identical in both cases
            since T_φ(ψ(u), z) = T_φ(q, z) when u = ψ⁻¹(q).  When manifold is
            unwrapped (v4) or wraps an IdentityChart, ψ is the identity and
            the chart slot stores q directly (backward compat).
        """
        H1 = self.H + 1
        rest_A = self.q_rest_A.to(device=device, dtype=dtype)
        rest_B = self.q_rest_B.to(device=device, dtype=dtype)
        lo = self.p_box_lo.to(device=device, dtype=dtype)
        hi = self.p_box_hi.to(device=device, dtype=dtype)
        R_anchor_aa = self.R_anchor_aa.to(device=device, dtype=dtype)
        R_anchor = exp_SO3(R_anchor_aa.expand(n, 3))                          # (n, 3, 3)

        # 1. EE position waypoints — endpoints in p_box
        if T_start is None:
            p_start = lo + (hi - lo) * torch.rand(n, 3, device=device, dtype=dtype)
            R_start = R_anchor
        else:
            T_start = T_start.to(device=device, dtype=dtype)
            from smcdp.lie_se3 import quat_to_R
            R_start = quat_to_R(T_start[..., :4])
            p_start = T_start[..., 4:]
        if T_target is None:
            p_end = lo + (hi - lo) * torch.rand(n, 3, device=device, dtype=dtype)
            omega = (torch.rand(n, 3, device=device, dtype=dtype) * 2 - 1) * self.target_perturb_rad
            R_end = R_anchor @ exp_SO3(omega)                                 # (n, 3, 3)
        else:
            T_target = T_target.to(device=device, dtype=dtype)
            from smcdp.lie_se3 import quat_to_R
            R_end = quat_to_R(T_target[..., :4])
            p_end = T_target[..., 4:]

        # 2. SE(3) interpolation: linear p, Slerp R
        s = torch.linspace(0, 1, H1, device=device, dtype=dtype).view(1, H1, 1)
        p_traj = p_start.unsqueeze(1) + s * (p_end - p_start).unsqueeze(1)    # (n, H+1, 3)
        # R_traj_h = R_start · exp_SO3(s_h · log_SO3(R_start^T R_end))
        from smcdp.lie_se3 import log_SO3
        R_rel = R_start.transpose(-1, -2) @ R_end                              # (n, 3, 3)
        omega_rel = log_SO3(R_rel)                                             # (n, 3)
        # s_h · omega_rel for each h
        s_flat = s.squeeze(-1)                                                 # (1, H+1)
        omega_h = omega_rel.unsqueeze(1) * s_flat.unsqueeze(-1)                # (n, H+1, 3)
        R_h = R_start.unsqueeze(1) @ exp_SO3(omega_h.reshape(-1, 3)).reshape(n, H1, 3, 3)

        # 3. mode + z_e per trajectory
        branch_A = torch.rand(n, device=device) < self.branch_p_A              # (n,) bool
        z_e = self.z_lo + (self.z_hi - self.z_lo) * torch.rand(
            n, 1, device=device, dtype=dtype
        )
        q_rest = torch.where(branch_A.unsqueeze(-1), rest_A, rest_B)            # (n, 7)

        # 4. Initial q(0): rest + jitter
        q_curr = q_rest + self.jitter_q * torch.randn(n, 7, device=device, dtype=dtype)

        # 5. Sequential pose-IK across timesteps
        q_traj_list = []
        for h in range(H1):
            R_h_b = R_h[:, h, :, :]                                            # (n, 3, 3)
            p_h_b = p_traj[:, h, :]                                            # (n, 3)
            for _ in range(self.n_ik_steps):
                q_curr = _dls_ik_pose_step(
                    self.ik_arm, q_curr, z_e, (R_h_b, p_h_b), q_rest,
                    alpha=self.ik_alpha, alpha_null=self.ik_alpha_null,
                    lam=self.ik_lam,
                    clamp_to_limits=self.ik_clamp_to_limits,
                    clamp_margin_frac=self.ik_clamp_margin_frac,
                )
            q_traj_list.append(q_curr)
        q_traj = torch.stack(q_traj_list, dim=1)                                # (n, H+1, 7)

        # 6. Lift to ambient via the (possibly learned) pose manifold.
        #
        # v4.1: when the manifold has a `.chart` attribute (i.e., it is a
        # BoundedChartPoseManifold), convert the IK-produced physical q to
        # chart coordinate u = ψ⁻¹(q) before make_x.  The η-clip in psi_inv
        # (eps=1e-3) guards against IK output sitting exactly at the joint
        # limit (where atanh diverges); spec §10.3 designates this as
        # initialization-only safety, not sampling-time enforcement.  Under
        # v4 (unwrapped or IdentityChart), this is a no-op (ψ⁻¹ = identity).
        z_traj = z_e.unsqueeze(1).expand(n, H1, 1).contiguous()
        q_flat = q_traj.reshape(-1, 7)
        z_flat = z_traj.reshape(-1, 1)
        if hasattr(self.manifold, "chart"):
            chart_slot_flat = self.manifold.chart.psi_inv(q_flat, eps=1e-3)
        else:
            chart_slot_flat = q_flat                                          # v4: chart slot IS physical q
        x_flat = self.manifold.make_x(chart_slot_flat, z_flat)
        x = x_flat.reshape(n, H1, self.manifold.ambient_dim)

        # 7. Pack T_target and T_start in storage form (q_R, p).
        # Use *realized* endpoints T_φ(q_0, z_e) and T_φ(q_H, z_e), NOT the
        # IK targets, so that the conditioning (T_start, T_target) is
        # exactly consistent with the trajectory endpoints.  Critical for
        # Tier 2: when IK clamps q to limits, the realized pose deviates from
        # the IK target, and using the IK target as conditioning would teach
        # the score net an inconsistent map.  For Tier 0 (no clamp + IK
        # converges) this is byte-equivalent within IK tolerance.
        #
        # CHART-AWARE: when manifold is wrapped (BoundedChartPoseManifold),
        # `manifold.T_phi_Rp` expects the chart slot u and applies psi(u)
        # internally → passing physical q here would double-apply psi.
        # Use the underlying base manifold's T_phi_Rp for physical q.
        physical_arm = getattr(self.manifold, "base", self.manifold)
        R_phi_0, p_phi_0 = physical_arm.T_phi_Rp(q_traj[:, 0, :], z_e)
        R_phi_H, p_phi_H = physical_arm.T_phi_Rp(q_traj[:, -1, :], z_e)
        T_start_used  = torch.cat([R_to_quat(R_phi_0), p_phi_0], dim=-1)        # (n, 7)
        T_target_used = torch.cat([R_to_quat(R_phi_H), p_phi_H], dim=-1)        # (n, 7)
        return x, branch_A, z_e, T_target_used, T_start_used

    def sample_x(self, n: int, device=None, dtype=torch.float32) -> Tensor:
        x, *_ = self.sample(n, device=device, dtype=dtype)
        return x


class FrankaUniformJointDemoPose:
    """Stage-1 pose self-model training data:  uniform joint-config samples
    + ground-truth pose (via `TrueFrankaCompliancePose`).

    Mirrors `FrankaUniformJointDemo` but emits (q, z_e, T_true_Rp).
    """

    def __init__(
        self,
        arm: Franka7DoFPose,
        truth,                                                                 # TrueFrankaCompliancePose
        z_e_range: tuple[float, float] = (0.05, 0.15),
    ):
        self.arm = arm
        self.truth = truth
        self.z_lo, self.z_hi = z_e_range

    @torch.no_grad()
    def sample(self, n: int, device=None, dtype=torch.float32):
        lower = self.arm.q_lower.to(device=device, dtype=dtype)
        upper = self.arm.q_upper.to(device=device, dtype=dtype)
        margin = self.arm.joint_limit_margin_frac * (upper - lower)
        lo = lower + margin
        hi = upper - margin
        q = lo + (hi - lo) * torch.rand(n, 7, device=device, dtype=dtype)
        z_e = self.z_lo + (self.z_hi - self.z_lo) * torch.rand(
            n, 1, device=device, dtype=dtype
        )
        R_t, p_t = self.truth.T_true_Rp(q, z_e)
        return q, z_e, (R_t, p_t)
