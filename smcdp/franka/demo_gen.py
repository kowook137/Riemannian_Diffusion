"""Bimodal trajectory demos for Franka 7-DoF (discrete IK branches).

Idea_formulation §15.1:  redundant kinematics (7-DoF arm × 3-DoF EE position)
with `same end-effector target → multi-modal joint-trajectory solutions`.

We construct two distinct postures (mode A and mode B) by damped-least-squares
IK with null-space bias toward two rest configurations:

    q_{k+1} = q_k + α · J^+ · (p_target − p(q_k, z_e))
                  + α_null · (I − J^+ J) · (q_rest − q_k)

where J^+ = J^T (J J^T + λ²I)^{-1} is the DLS pseudo-inverse and the second
term projects the rest-bias onto the null space of the task Jacobian.  Same
EE waypoint trajectory ⇒ same p(t); different rest postures ⇒ distinct
q-trajectories that all reach the same EE position cluster.

This is the 7-DoF analogue of the planar 3-link bimodal IK (elbow-up / elbow-
down) in toy3p5: discrete branches of the IK manifold rather than continuous
null-space variation.
"""
from __future__ import annotations

import torch
from torch import Tensor

from smcdp.manifolds import Franka7DoF


def _dls_ik_step(
    arm: Franka7DoF,
    q: Tensor,                  # (B, 7)
    z_e: Tensor,                # (B, 1)
    p_target: Tensor,           # (B, 3)
    q_rest: Tensor,             # (B, 7)
    alpha: float = 0.5,
    alpha_null: float = 0.3,
    lam: float = 0.05,
) -> Tensor:
    """One DLS-IK iteration with null-space rest-posture bias."""
    p = arm.F(q, z_e)                                                # (B, 3)
    J = arm.jacobian_F(q, z_e)                                       # (B, 3, 7)
    err = (p_target - p).unsqueeze(-1)                                # (B, 3, 1)
    JJt = J @ J.transpose(-1, -2)                                     # (B, 3, 3)
    eye3 = torch.eye(3, device=q.device, dtype=q.dtype)
    Jpinv = J.transpose(-1, -2) @ torch.linalg.inv(JJt + (lam ** 2) * eye3)
    d_main = (Jpinv @ err).squeeze(-1)                                # (B, 7)
    eye7 = torch.eye(7, device=q.device, dtype=q.dtype)
    null_proj = eye7 - Jpinv @ J                                      # (B, 7, 7)
    d_null = (null_proj @ (q_rest - q).unsqueeze(-1)).squeeze(-1)     # (B, 7)
    return q + alpha * d_main + alpha_null * d_null


class FrankaBimodalReachingDemo:
    """Bimodal Franka demo generator with two posture modes A and B.

    Each demo:
      1. Sample EE waypoints  p(0), p(1), …, p(H)  by linear interpolation
         between random p_start, p_end ∼ Uniform(p_box).
      2. Sample mode ∈ {A, B}  with  P(A) = `branch_p_A`.
      3. Sample z_e ∼ Uniform(z_range)  (frozen across trajectory).
      4. q(0) = q_rest_mode + jitter_q · ε.
      5. For h = 1..H, warm-start from q(h−1) and run `n_ik_steps` of
         DLS IK with null-space bias toward q_rest_mode.
      6. Lift each timestep via `manifold.make_x(q, z_e)`.

    Returns from `sample(n)`:
        x         : (n, H+1, ambient_dim)  on the manifold
        branch_A  : (n,) bool — ground-truth mode label
        z_e       : (n, 1)    — per-trajectory embodiment
        p_target  : (n, 3)    — end-of-trajectory EE waypoint (used as
                                §15.1 goal-conditional input at sample time)
        p_start   : (n, 3)    — start-of-trajectory EE waypoint (used as
                                start-anchor cond + analytic start guidance)
    """

    def __init__(
        self,
        manifold,                     # Franka7DoF or LearnedSelfModelFranka7DoF
        ik_arm: Franka7DoF,           # use ANALYTIC arm for IK (faster, deterministic)
        H: int,
        q_rest_A,
        q_rest_B,
        p_box_lo=(0.40, -0.10, 0.40),
        p_box_hi=(0.55,  0.10, 0.55),
        z_e_range: tuple[float, float] = (0.05, 0.15),
        branch_p_A: float = 0.5,
        jitter_q: float = 0.05,
        n_ik_steps: int = 5,
        ik_alpha: float = 0.5,
        ik_alpha_null: float = 0.3,
        ik_lam: float = 0.05,
    ):
        # n_z must be 1 for the EmbodimentGraphManifold contract used here.
        assert getattr(manifold, "n_q", 0) == 7
        assert getattr(manifold, "n_p", 0) == 3
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
        self.ik_alpha = float(ik_alpha)
        self.ik_alpha_null = float(ik_alpha_null)
        self.ik_lam = float(ik_lam)

    @torch.no_grad()
    def sample(self, n: int, device=None, dtype=torch.float32, p_target=None, p_start=None):
        """Return (x, branch_A, z_e, p_end, p_start_used).

        If `p_target` (n, 3) is provided, use it as the END EE waypoint.
        If `p_start` (n, 3) is provided, use it as the START EE waypoint.
        Otherwise sampled uniformly in p_box."""
        H1 = self.H + 1
        rest_A = self.q_rest_A.to(device=device, dtype=dtype)
        rest_B = self.q_rest_B.to(device=device, dtype=dtype)
        lo = self.p_box_lo.to(device=device, dtype=dtype)
        hi = self.p_box_hi.to(device=device, dtype=dtype)

        # 1. EE waypoints — endpoints in p_box
        if p_start is None:
            p_start_used = lo + (hi - lo) * torch.rand(n, 3, device=device, dtype=dtype)
        else:
            p_start_used = p_start.to(device=device, dtype=dtype)
        if p_target is None:
            p_end = lo + (hi - lo) * torch.rand(n, 3, device=device, dtype=dtype)
        else:
            p_end = p_target.to(device=device, dtype=dtype)
        s = torch.linspace(0, 1, H1, device=device, dtype=dtype).view(1, H1, 1)
        p_traj = p_start_used.unsqueeze(1) + s * (p_end - p_start_used).unsqueeze(1)  # (n, H+1, 3)

        # 2. mode + z_e per trajectory
        branch_A = torch.rand(n, device=device) < self.branch_p_A                  # (n,) bool
        z_e = self.z_lo + (self.z_hi - self.z_lo) * torch.rand(
            n, 1, device=device, dtype=dtype
        )                                                                          # (n, 1)
        # per-trajectory rest posture (B, 7)
        q_rest = torch.where(branch_A.unsqueeze(-1), rest_A, rest_B)               # (n, 7)

        # 3. Initial q(0): rest + jitter (works whether or not p_start is provided)
        # Note: `p_start` here refers to the BIND in this scope used below
        q_curr = q_rest + self.jitter_q * torch.randn(n, 7, device=device, dtype=dtype)
        del p_start  # avoid scope shadowing of the input arg below

        # 4. Sequential IK across timesteps (warm-started)
        q_traj_list = []
        for h in range(H1):
            target_h = p_traj[:, h, :]                                             # (n, 3)
            for _ in range(self.n_ik_steps):
                q_curr = _dls_ik_step(
                    self.ik_arm, q_curr, z_e, target_h, q_rest,
                    alpha=self.ik_alpha, alpha_null=self.ik_alpha_null,
                    lam=self.ik_lam,
                )
            q_traj_list.append(q_curr)
        q_traj = torch.stack(q_traj_list, dim=1)                                   # (n, H+1, 7)

        # 5. Lift to ambient via the (possibly learned) manifold
        z_traj = z_e.unsqueeze(1).expand(n, H1, 1).contiguous()
        q_flat = q_traj.reshape(-1, 7)
        z_flat = z_traj.reshape(-1, 1)
        x_flat = self.manifold.make_x(q_flat, z_flat)
        x = x_flat.reshape(n, H1, self.manifold.ambient_dim)
        return x, branch_A, z_e, p_end, p_start_used

    def sample_x(self, n: int, device=None, dtype=torch.float32) -> Tensor:
        x, *_ = self.sample(n, device=device, dtype=dtype)
        return x


class FrankaUniformJointDemo:
    """Stage-1 self-model training data:  uniform joint-config samples.

    Per Idea §7.2: Stage 1 needs (q, z_e, p_true) pairs.  The simplest sufficient
    distribution is uniform within (margin-shrunken) joint limits + uniform z_e —
    this matches the chart-uniform sampling Franka7DoF.random_uniform provides
    but ALSO returns the ground-truth EE position via TrueFrankaCompliance.

    For an embodiment-aware Stage-1 (see Idea §3, §7.3), z_e is sampled freely
    within the calibration range so Δ_φ learns the joint × tool dependence.
    """

    def __init__(
        self,
        arm: Franka7DoF,
        truth,                           # TrueFrankaCompliance
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
        p_true = self.truth.p_true(q, z_e)
        return q, z_e, p_true
