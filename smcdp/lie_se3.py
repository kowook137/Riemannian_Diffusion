"""SE(3) Lie group utilities for the pose-extended SMCDP framework.

All operations follow the conventions established in `extension.tex`:
  - Tangent vectors at T ∈ SE(3) are expressed in BODY-FRAME (right-trivialized)
    coordinates: dot(T) = T · ξ^∧, with ξ = (ρ, ω) ∈ R^3 × R^3.
  - so(3) is identified with R^3 via the hat map  [·]_×.
  - se(3) is identified with R^6 = R^3 (linear) × R^3 (angular), in that order.
  - Storage representation for trajectories: T = (q_R, p) ∈ R^7, with
    quaternion q_R = (x, y, z, w) (real part LAST, matching `roma`).

The utilities are batched (any leading dims), autograd-safe (Taylor expansion
near θ → 0 to avoid 0/0), and pure-PyTorch.  Quaternion ↔ rotation-matrix
conversions delegate to `roma` (well-tested, batched, autograd).

References:
  - Barfoot, "State Estimation for Robotics" (Cambridge), Ch. 7.
  - Sola, Deray, Atchuthan, "A micro Lie theory for state estimation in robotics".
"""
from __future__ import annotations

import torch
from torch import Tensor

import roma


# ---------------------------------------------------------------------------
# Numerical-safe small-angle coefficients
# ---------------------------------------------------------------------------
#
# Returned shapes match the input theta.  Each function is autograd-safe
# (torch.where branches are both NaN-free).  We use Taylor-series expansion
# for theta below TAYLOR_EPS and the closed-form expression elsewhere.

TAYLOR_EPS = 1e-3                                                            # threshold below which we switch to Taylor


def _sinc(theta: Tensor) -> Tensor:
    """sin(θ)/θ, smooth at θ=0 with value 1.0."""
    safe = torch.where(theta.abs() < TAYLOR_EPS, torch.ones_like(theta), theta)
    big = torch.sin(safe) / safe
    series = 1.0 - theta**2 / 6.0 + theta**4 / 120.0
    return torch.where(theta.abs() < TAYLOR_EPS, series, big)


def _one_minus_cos_over_theta2(theta: Tensor) -> Tensor:
    """(1 − cos θ)/θ² , smooth at θ=0 with value 1/2."""
    safe = torch.where(theta.abs() < TAYLOR_EPS, torch.ones_like(theta), theta)
    big = (1.0 - torch.cos(safe)) / safe**2
    series = 0.5 - theta**2 / 24.0 + theta**4 / 720.0
    return torch.where(theta.abs() < TAYLOR_EPS, series, big)


def _theta_minus_sin_over_theta3(theta: Tensor) -> Tensor:
    """(θ − sin θ)/θ³ , smooth at θ=0 with value 1/6."""
    safe = torch.where(theta.abs() < TAYLOR_EPS, torch.ones_like(theta), theta)
    big = (safe - torch.sin(safe)) / safe**3
    series = 1.0 / 6.0 - theta**2 / 120.0 + theta**4 / 5040.0
    return torch.where(theta.abs() < TAYLOR_EPS, series, big)


def _Jl_inv_coeff(theta: Tensor) -> Tensor:
    """Coefficient of [ω]²_× in J_l^{-1}(ω):

        c(θ) = 1/θ² − (1 + cos θ) / (2 θ sin θ)

    Smooth at θ=0 with value 1/12.
    """
    safe = torch.where(theta.abs() < TAYLOR_EPS, torch.ones_like(theta), theta)
    big = 1.0 / safe**2 - (1.0 + torch.cos(safe)) / (2.0 * safe * torch.sin(safe))
    series = 1.0 / 12.0 + theta**2 / 720.0 + theta**4 / 30240.0
    return torch.where(theta.abs() < TAYLOR_EPS, series, big)


# ---------------------------------------------------------------------------
# so(3) hat / vee
# ---------------------------------------------------------------------------


def hat_so3(omega: Tensor) -> Tensor:
    """ω ∈ R^3 → [ω]_× ∈ R^{3×3}.  Batched: any leading dims preserved."""
    o0, o1, o2 = omega[..., 0], omega[..., 1], omega[..., 2]
    zero = torch.zeros_like(o0)
    row0 = torch.stack([zero,  -o2,    o1], dim=-1)
    row1 = torch.stack([  o2, zero,   -o0], dim=-1)
    row2 = torch.stack([ -o1,   o0,  zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)                           # (..., 3, 3)


def vee_so3(M: Tensor) -> Tensor:
    """[ω]_× ∈ R^{3×3} → ω ∈ R^3.  Inverse of hat_so3 (assumes M skew)."""
    return torch.stack([M[..., 2, 1], M[..., 0, 2], M[..., 1, 0]], dim=-1)


# ---------------------------------------------------------------------------
# SO(3) exp / log
# ---------------------------------------------------------------------------


def exp_SO3(omega: Tensor) -> Tensor:
    """Rodrigues' formula:  ω ∈ R^3 → R = exp([ω]_×) ∈ SO(3).

    Returns rotation matrix (..., 3, 3).
    """
    theta = omega.norm(dim=-1, keepdim=True).clamp(min=0.0)                  # (..., 1)
    K = hat_so3(omega)                                                       # (..., 3, 3)
    A = _sinc(theta).unsqueeze(-1)                                           # (..., 1, 1)
    B = _one_minus_cos_over_theta2(theta).unsqueeze(-1)                      # (..., 1, 1)
    I = torch.eye(3, dtype=omega.dtype, device=omega.device).expand_as(K)
    R = I + A * K + B * (K @ K)
    return R


def log_SO3(R: Tensor) -> Tensor:
    """Inverse of exp_SO3: R ∈ SO(3) → ω ∈ R^3 with ‖ω‖ ∈ [0, π].

    Numerically robust at θ = 0 (returns 0) and at θ = π (uses symmetric
    branch from R + R^T).  Uses atan2(sin θ, cos θ) instead of acos+clamp
    to preserve gradient flow through R = I (where cos θ = 1 exactly).
    """
    R_minus_RT = R - R.transpose(-1, -2)
    omega_skew_part = vee_so3(R_minus_RT) * 0.5                              # (..., 3); ‖·‖ = |sin θ|
    sin_theta_abs = omega_skew_part.norm(dim=-1, keepdim=True)               # (..., 1)
    cos_theta = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1.0) * 0.5
                  ).unsqueeze(-1)                                            # (..., 1)
    theta = torch.atan2(sin_theta_abs, cos_theta)                            # (..., 1) ∈ [0, π]

    # General case: ω = θ · n̂, with n̂ = (vee((R-R^T)/2)) / sin θ.
    # Compute  coeff = θ / sin θ  with safe Taylor branch at small θ.
    safe_sin = torch.where(theta.abs() < TAYLOR_EPS,
                            torch.ones_like(sin_theta_abs), sin_theta_abs)
    coeff_big = theta / safe_sin                                             # (..., 1)
    coeff_series = 1.0 + theta**2 / 6.0 + 7.0 * theta**4 / 360.0             # (..., 1)
    coeff = torch.where(theta.abs() < TAYLOR_EPS, coeff_series, coeff_big)
    omega_general = omega_skew_part * coeff                                  # (..., 3)

    # Edge case θ ≈ π: the skew-part formula degenerates (sin θ → 0).
    # Always compute the alternative branch and select via torch.where so the
    # path is vmap-compatible (no data-dependent control flow).
    near_pi = theta > torch.pi - 0.05                                        # (..., 1)
    diag = torch.diagonal(R, dim1=-2, dim2=-1)                               # (..., 3)
    omega_sq = ((diag + 1.0) * 0.5).clamp(min=0.0)                           # (..., 3)
    omega_abs = torch.sqrt(omega_sq + 1e-20)                                 # (..., 3)
    argmax = omega_abs.argmax(dim=-1, keepdim=True)                          # (..., 1)
    sym_part = 0.5 * (R + R.transpose(-1, -2))                               # (..., 3, 3)
    row = torch.gather(sym_part, -2,
                        argmax.unsqueeze(-1).expand(*argmax.shape, 3)
                        ).squeeze(-2)                                        # (..., 3)
    sign_vec = torch.sign(row + 1e-20)                                       # (..., 3)
    argmax_3 = argmax.expand_as(sign_vec)
    idx_range = torch.arange(3, device=R.device).expand_as(argmax_3)
    idx_mask = idx_range == argmax_3
    sign_vec = torch.where(idx_mask, torch.ones_like(sign_vec), sign_vec)
    omega_pi = theta * omega_abs * sign_vec                                  # (..., 3)

    return torch.where(near_pi, omega_pi, omega_general)


# ---------------------------------------------------------------------------
# SO(3) left Jacobian J_l(ω) and inverse
# ---------------------------------------------------------------------------


def J_l_SO3(omega: Tensor) -> Tensor:
    """Left Jacobian on SO(3):
        J_l(ω) = I + (1−cos θ)/θ² [ω]_× + (θ−sin θ)/θ³ [ω]_×²

    Smooth at θ=0 (J_l(0) = I).  Returns (..., 3, 3).
    """
    theta = omega.norm(dim=-1, keepdim=True)
    K = hat_so3(omega)
    B = _one_minus_cos_over_theta2(theta).unsqueeze(-1)
    C = _theta_minus_sin_over_theta3(theta).unsqueeze(-1)
    I = torch.eye(3, dtype=omega.dtype, device=omega.device).expand_as(K)
    return I + B * K + C * (K @ K)


def J_l_inv_SO3(omega: Tensor) -> Tensor:
    """Inverse left Jacobian on SO(3):
        J_l^{-1}(ω) = I − ½ [ω]_× + (1/θ² − (1+cos θ)/(2θ sin θ)) [ω]_×²

    Smooth at θ=0 (returns I).  Returns (..., 3, 3).
    """
    theta = omega.norm(dim=-1, keepdim=True)
    K = hat_so3(omega)
    coeff = _Jl_inv_coeff(theta).unsqueeze(-1)
    I = torch.eye(3, dtype=omega.dtype, device=omega.device).expand_as(K)
    return I - 0.5 * K + coeff * (K @ K)


# ---------------------------------------------------------------------------
# se(3) exp / log via (R, p) representation
# ---------------------------------------------------------------------------


def exp_SE3(xi: Tensor) -> tuple[Tensor, Tensor]:
    """ξ = (ρ, ω) ∈ R^6 → (R, p) ∈ SE(3).

    R = exp_SO3(ω);   p = J_l(ω) · ρ.

    Body-frame trivialization: a ξ in the Lie algebra maps to a finite SE(3)
    transform via the matrix exponential of ξ^∧.

    Returns (R, p) with shapes (..., 3, 3) and (..., 3).
    """
    rho = xi[..., :3]                                                        # (..., 3)
    omega = xi[..., 3:]                                                      # (..., 3)
    R = exp_SO3(omega)
    Vrho = (J_l_SO3(omega) @ rho.unsqueeze(-1)).squeeze(-1)                  # (..., 3)
    return R, Vrho


def log_SE3(R: Tensor, p: Tensor) -> Tensor:
    """(R, p) ∈ SE(3) → ξ = (ρ, ω) ∈ R^6.

    ω = log_SO3(R);   ρ = J_l(ω)^{-1} · p.

    Returns ξ shape (..., 6).
    """
    omega = log_SO3(R)                                                       # (..., 3)
    rho = (J_l_inv_SO3(omega) @ p.unsqueeze(-1)).squeeze(-1)                 # (..., 3)
    return torch.cat([rho, omega], dim=-1)


# ---------------------------------------------------------------------------
# Quaternion utilities (storage form: (x, y, z, w), w last; matches roma)
# ---------------------------------------------------------------------------


def quat_to_R(q: Tensor) -> Tensor:
    """Unit quaternion (x, y, z, w) → rotation matrix R ∈ SO(3).

    Delegates to roma.unitquat_to_rotmat.  Caller is responsible for
    pre-normalising q to unit norm if needed.
    """
    return roma.unitquat_to_rotmat(q)


def R_to_quat(R: Tensor) -> Tensor:
    """Rotation matrix R ∈ SO(3) → unit quaternion (x, y, z, w).

    Delegates to roma.rotmat_to_unitquat.  Output is in the same hemisphere
    convention as roma (no canonicalization here).
    """
    return roma.rotmat_to_unitquat(R)


def quat_inverse(q: Tensor) -> Tensor:
    """Unit-quaternion inverse: (x, y, z, w) → (-x, -y, -z, w).
    Equivalently the conjugate, since unit quaternions satisfy q^{-1} = q*.
    """
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)


def quat_mul(q1: Tensor, q2: Tensor) -> Tensor:
    """Hamilton product of two quaternions, (x, y, z, w) convention.

    (q1 * q2) such that R(q1 * q2) = R(q1) · R(q2).
    """
    return roma.quat_product(q1, q2)


def hemisphere_consistency(qs: Tensor) -> Tensor:
    """Force a sequence of unit quaternions to live in a consistent hemisphere
    along the last sequence axis (axis = -2 is timestep).

    `qs` has shape (..., H+1, 4); for each h ≥ 1, if  q_h^T q_{h-1} < 0  flip
    q_h ← -q_h.  This preserves the rotation it represents (double-cover) and
    eliminates arbitrary sign jumps in stored trajectories — required by
    extension.tex Sec. 6 (rotation representation remark).

    Performed sequentially (cumulative sign).  Returns same shape.
    """
    if qs.shape[-2] < 2:
        return qs
    out = [qs[..., 0, :]]
    for h in range(1, qs.shape[-2]):
        prev = out[-1]
        cur = qs[..., h, :]
        dot = (prev * cur).sum(dim=-1, keepdim=True)
        cur = torch.where(dot < 0, -cur, cur)
        out.append(cur)
    return torch.stack(out, dim=-2)


# ---------------------------------------------------------------------------
# Pose storage form (q_R, p) ∈ R^7  ↔  (R, p) tuple
# ---------------------------------------------------------------------------


def pose7_to_Rp(T7: Tensor) -> tuple[Tensor, Tensor]:
    """T = (q_R, p) ∈ R^7  →  (R, p) ∈ SO(3) × R^3."""
    q = T7[..., :4]
    p = T7[..., 4:]
    return quat_to_R(q), p


def Rp_to_pose7(R: Tensor, p: Tensor) -> Tensor:
    """(R, p) → (q_R, p) ∈ R^7."""
    q = R_to_quat(R)
    return torch.cat([q, p], dim=-1)


def pose_compose(T1: Tensor, T2: Tensor) -> Tensor:
    """Compose two poses in storage form: T1 · T2.  (storage = (q_R, p) ∈ R^7)

    NOTE: This uses quat_to_R inside, which (via roma) calls operations that
    are NOT vmap-compatible.  For autograd-critical paths (Jacobian of T_φ,
    DSM target, guidance reward), use the (R, p)-tuple variants
    `compose_Rp / inverse_Rp / log_relative_Rp` below.  Storage form should
    only be used at network input/output and ckpt save/load boundaries.
    """
    q1, p1 = T1[..., :4], T1[..., 4:]
    q2, p2 = T2[..., :4], T2[..., 4:]
    q12 = quat_mul(q1, q2)
    R1 = quat_to_R(q1)
    p12 = (R1 @ p2.unsqueeze(-1)).squeeze(-1) + p1
    return torch.cat([q12, p12], dim=-1)


def pose_inverse(T: Tensor) -> Tensor:
    """Inverse on storage form: (R, p)^{-1} = (R^T, -R^T p).  See note in
    `pose_compose` regarding vmap compatibility."""
    q, p = T[..., :4], T[..., 4:]
    q_inv = quat_inverse(q)
    R_inv = quat_to_R(q_inv)
    p_inv = -(R_inv @ p.unsqueeze(-1)).squeeze(-1)
    return torch.cat([q_inv, p_inv], dim=-1)


def pose_log(T: Tensor) -> Tensor:
    """Log map on storage form: (q_R, p) ∈ R^7 → ξ = (ρ, ω) ∈ R^6."""
    q, p = T[..., :4], T[..., 4:]
    R = quat_to_R(q)
    return log_SE3(R, p)


def pose_exp(xi: Tensor) -> Tensor:
    """Exp map: ξ ∈ R^6 → storage form (q_R, p) ∈ R^7."""
    R, p = exp_SE3(xi)
    return Rp_to_pose7(R, p)


def pose_log_relative(T_a: Tensor, T_b: Tensor) -> Tensor:
    """ξ = log_SE3( T_a^{-1} · T_b ) on storage form.  Storage variant; see
    `log_relative_Rp` for vmap-safe (R, p)-form."""
    return pose_log(pose_compose(pose_inverse(T_a), T_b))


# ---------------------------------------------------------------------------
# (R, p)-tuple form — vmap-safe, autograd-critical paths
# ---------------------------------------------------------------------------
#
# These operate on (R, p) directly and avoid R↔quat conversions, so they
# compose with torch.func.vmap and torch.func.jacrev cleanly.  Used inside
# the manifold (J_F, lift, retraction), the DSM target, and reward gradients.


def compose_Rp(R1: Tensor, p1: Tensor, R2: Tensor, p2: Tensor) -> tuple[Tensor, Tensor]:
    """(R1, p1) · (R2, p2) = (R1 R2, R1 p2 + p1).  Returns (R, p)."""
    R12 = R1 @ R2
    p12 = (R1 @ p2.unsqueeze(-1)).squeeze(-1) + p1
    return R12, p12


def inverse_Rp(R: Tensor, p: Tensor) -> tuple[Tensor, Tensor]:
    """(R, p)^{-1} = (R^T, -R^T p).  Returns (R, p)."""
    R_inv = R.transpose(-1, -2)
    p_inv = -(R_inv @ p.unsqueeze(-1)).squeeze(-1)
    return R_inv, p_inv


def log_relative_Rp(R_a: Tensor, p_a: Tensor, R_b: Tensor, p_b: Tensor) -> Tensor:
    """ξ = log_SE3( (R_a, p_a)^{-1} · (R_b, p_b) ) ∈ R^6.

    Body-frame error twist of (R_b, p_b) w.r.t. (R_a, p_a).  No quat ops →
    vmap-safe.  Use this for DSM target, reward gradients, and Jacobian
    computations.
    """
    R_a_inv, p_a_inv = inverse_Rp(R_a, p_a)
    R_rel, p_rel = compose_Rp(R_a_inv, p_a_inv, R_b, p_b)
    return log_SE3(R_rel, p_rel)


# ---------------------------------------------------------------------------
# Geometric Jacobian utilities — body-frame, by autograd through pose_log
# ---------------------------------------------------------------------------
#
# For a forward map  T: R^{n_q} → SE(3),  q ↦ T(q),  the body-frame geometric
# Jacobian J_pose(q) ∈ R^{6 × n_q} is defined by  dot(T) = T · (J_pose · q̇)^∧.
#
# In closed form using only `T` (not its component R, p separately):
#     J_pose = ∂_q ξ(q + δq, q)  evaluated at δq = 0,
#   where  ξ(q', q) = log_SE3( T(q)^{-1} · T(q') ).
#
# This matches body-frame trivialization:  varying q' near q introduces a
# small twist, and  log_SE3(T(q)^{-1} · T(q + δq))  picks out exactly that
# body-frame twist (with vanishing right-Jacobian factor at δq = 0).
#
# We provide an autograd-based helper here.  Subclasses of the new
# EmbodimentPoseGraphManifold may override with a closed-form expression
# (e.g. pytorch_kinematics body-frame Jacobian + tool-offset cross term).


def adjoint_Rp(R: Tensor, p: Tensor) -> Tensor:
    """SE(3) Adjoint matrix Ad_T ∈ R^{6×6} for T = (R, p):

        Ad_T = [[ R,    [p]_× R ],
                [ 0_3,  R       ]]

    Acts on body-frame twists (ρ, ω) ∈ se(3): Ad_T · ξ transforms a twist
    expressed in the T-body frame to a twist expressed in T's parent frame.
    Returns shape (..., 6, 6).
    """
    P_R = hat_so3(p) @ R                                                 # (..., 3, 3)
    zero = torch.zeros_like(R)
    top = torch.cat([R, P_R], dim=-1)                                    # (..., 3, 6)
    bot = torch.cat([zero, R], dim=-1)                                   # (..., 3, 6)
    return torch.cat([top, bot], dim=-2)                                 # (..., 6, 6)


def adjoint_inverse_Rp(R: Tensor, p: Tensor) -> Tensor:
    """SE(3) Adjoint of T^{-1} for T = (R, p):

        Ad_{T^{-1}} = [[ R^⊤,    −R^⊤ [p]_× ],
                       [ 0_3,    R^⊤        ]]

    Equivalently, this is (Ad_T)^{-1}.  Used to map a body-frame twist of T
    into the body-frame of T's parent (e.g. composing T = T_a · T_d, the
    body-twist of T_a transforms via Ad_{T_d^{-1}} to T's body frame).
    Returns shape (..., 6, 6).
    """
    Rt = R.transpose(-1, -2)
    upper_right = -(Rt @ hat_so3(p))                                     # (..., 3, 3)
    zero = torch.zeros_like(R)
    top = torch.cat([Rt, upper_right], dim=-1)
    bot = torch.cat([zero, Rt], dim=-1)
    return torch.cat([top, bot], dim=-2)


def jacobian_pose_autograd(T_fn_Rp, q: Tensor) -> Tensor:
    """Body-frame geometric Jacobian via autograd through log_SE3.

    T_fn_Rp : callable  q (B, n_q) -> (R, p)  with R: (B, 3, 3), p: (B, 3).
    q       : (B, n_q)   chart-coordinate batch.

    Returns J_pose ∈ R^{B × 6 × n_q}.

    Implementation: we build  ξ(q') = log_SE3((R(q), p(q))^{-1} · (R(q'), p(q')))
    with q DETACHED as the base, and differentiate ξ w.r.t. q' at q' = q.
    At q' = q,  ξ = 0  and the right-Jacobian factor in d(ξ)/d(δq) equals
    identity, so the result is the body-frame velocity Jacobian J_pose.

    Operations stay in (R, p) form throughout (no quaternion round-trip), so
    this composes correctly with torch.func.vmap.

    NOTE: per-sample base poses (R_base_inv, p_base_inv) are passed through
    vmap's `in_dims` — closure-captured batched tensors are NOT auto-mapped
    by vmap and would leak the full batch into each per-sample call.
    """
    q_base = q.detach()
    R_base, p_base = T_fn_Rp(q_base)                                         # (B, 3, 3), (B, 3)
    R_base_inv, p_base_inv = inverse_Rp(R_base, p_base)
    R_base_inv = R_base_inv.detach()
    p_base_inv = p_base_inv.detach()

    def xi_per_sample(q_var_s: Tensor, R_inv_s: Tensor, p_inv_s: Tensor) -> Tensor:
        R_var, p_var = T_fn_Rp(q_var_s.unsqueeze(0))
        R_var = R_var.squeeze(0)                                             # (3, 3)
        p_var = p_var.squeeze(0)                                             # (3,)
        R_rel = R_inv_s @ R_var
        p_rel = (R_inv_s @ p_var.unsqueeze(-1)).squeeze(-1) + p_inv_s
        return log_SE3(R_rel, p_rel)                                         # (6,)

    J = torch.func.vmap(
        torch.func.jacrev(xi_per_sample, argnums=0),
        in_dims=(0, 0, 0),
    )(q, R_base_inv, p_base_inv)                                             # (B, 6, n_q)
    return J
