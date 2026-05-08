"""Unit tests for smcdp.lie_se3 — SE(3) Lie group utilities.

Verifies:
  - hat / vee inverse
  - exp / log roundtrip on SO(3), SE(3) (small + large angles)
  - left-Jacobian identities (J_l(0) = I, J_l · J_l^{-1} = I)
  - quaternion ↔ rotation matrix consistency
  - pose composition / inverse / log-relative correctness
  - hemisphere consistency
  - autograd geometric Jacobian (body-frame velocity twist)

All checks at machine-precision (allclose with atol=1e-5, except where noted).

Run:  python -m tests.test_lie_se3   (or)   pytest tests/test_lie_se3.py
"""
from __future__ import annotations

import math

import torch

from smcdp.lie_se3 import (
    hat_so3, vee_so3,
    exp_SO3, log_SO3,
    J_l_SO3, J_l_inv_SO3,
    exp_SE3, log_SE3,
    quat_to_R, R_to_quat, quat_inverse, quat_mul,
    hemisphere_consistency,
    pose_compose, pose_inverse, pose_log, pose_exp, pose_log_relative,
    Rp_to_pose7, pose7_to_Rp,
    jacobian_pose_autograd,
)


torch.manual_seed(0)


def _assert_close(a, b, atol, msg):
    diff = (a - b).abs().max().item()
    assert diff < atol, f"{msg}: max diff {diff:.3e} > atol {atol:.3e}"
    print(f"  ✓ {msg}: max diff {diff:.3e}")


def test_hat_vee():
    print("\n[hat / vee]")
    omega = torch.randn(8, 3, dtype=torch.float64)
    K = hat_so3(omega)
    _assert_close(K + K.transpose(-1, -2), torch.zeros_like(K), 1e-12,
                  "hat_so3(ω) is skew-symmetric")
    _assert_close(vee_so3(K), omega, 1e-12, "vee(hat(ω)) = ω")


def test_so3_exp_log_roundtrip():
    print("\n[SO(3) exp/log roundtrip]")
    # Small rotations
    omega_small = torch.randn(16, 3, dtype=torch.float64) * 1e-4
    R_small = exp_SO3(omega_small)
    omega_back = log_SO3(R_small)
    _assert_close(omega_back, omega_small, 1e-10,
                  "log(exp(ω)) = ω  (small ω, |ω| ~ 1e-4)")

    # Mid-range
    omega_mid = torch.randn(16, 3, dtype=torch.float64)
    R_mid = exp_SO3(omega_mid)
    omega_back = log_SO3(R_mid)
    # Note: log returns ω with ‖ω‖ ∈ [0, π]; if input was ‖ω‖ > π we get the
    # equivalent rep.  We test only inputs with ‖ω‖ < π.
    keep = omega_mid.norm(dim=-1) < math.pi - 0.1
    _assert_close(omega_back[keep], omega_mid[keep], 1e-10,
                  "log(exp(ω)) = ω  (mid-range, ‖ω‖ < π)")

    # R = I (identity) → log = 0
    R_I = torch.eye(3, dtype=torch.float64).expand(8, 3, 3)
    _assert_close(log_SO3(R_I), torch.zeros(8, 3, dtype=torch.float64), 1e-12,
                  "log(I) = 0")

    # Round-trip the other way: exp(log(R)) = R
    omega_test = torch.randn(16, 3, dtype=torch.float64)
    R_test = exp_SO3(omega_test)
    R_back = exp_SO3(log_SO3(R_test))
    _assert_close(R_back, R_test, 1e-10, "exp(log(R)) = R")


def test_so3_left_jacobian():
    print("\n[SO(3) left Jacobian]")
    # J_l(0) = I
    J0 = J_l_SO3(torch.zeros(8, 3, dtype=torch.float64))
    I = torch.eye(3, dtype=torch.float64).expand(8, 3, 3)
    _assert_close(J0, I, 1e-10, "J_l(0) = I")

    Jinv0 = J_l_inv_SO3(torch.zeros(8, 3, dtype=torch.float64))
    _assert_close(Jinv0, I, 1e-10, "J_l_inv(0) = I")

    # J_l · J_l^{-1} = I (general)
    omega = torch.randn(16, 3, dtype=torch.float64) * 0.7
    J = J_l_SO3(omega)
    Jinv = J_l_inv_SO3(omega)
    _assert_close(J @ Jinv,
                  torch.eye(3, dtype=torch.float64).expand_as(J), 1e-10,
                  "J_l · J_l^{-1} = I")

    # Small omega (ensure Taylor branch)
    omega_small = torch.randn(8, 3, dtype=torch.float64) * 1e-4
    J_s = J_l_SO3(omega_small)
    Jinv_s = J_l_inv_SO3(omega_small)
    _assert_close(J_s @ Jinv_s,
                  torch.eye(3, dtype=torch.float64).expand_as(J_s), 1e-10,
                  "J_l · J_l^{-1} = I  (small ω, Taylor branch)")


def test_se3_exp_log_roundtrip():
    print("\n[SE(3) exp/log roundtrip]")
    xi = torch.randn(16, 6, dtype=torch.float64)
    # Limit angle to < π so log is unique
    omega_norm = xi[..., 3:].norm(dim=-1, keepdim=True)
    scale = (math.pi - 0.1) / omega_norm.clamp(min=1e-3)
    xi[..., 3:] = xi[..., 3:] * torch.where(omega_norm > math.pi - 0.1, scale,
                                              torch.ones_like(scale))

    R, p = exp_SE3(xi)
    xi_back = log_SE3(R, p)
    _assert_close(xi_back, xi, 1e-9, "log(exp(ξ)) = ξ  (general)")

    # Small ξ
    xi_small = torch.randn(8, 6, dtype=torch.float64) * 1e-5
    R_s, p_s = exp_SE3(xi_small)
    xi_back = log_SE3(R_s, p_s)
    _assert_close(xi_back, xi_small, 1e-10, "log(exp(ξ)) = ξ  (small ξ)")


def test_quaternion_consistency():
    print("\n[quaternion ↔ R consistency]")
    R = exp_SO3(torch.randn(16, 3, dtype=torch.float64))
    q = R_to_quat(R)
    _assert_close(q.norm(dim=-1), torch.ones(16, dtype=torch.float64), 1e-12,
                  "‖R_to_quat(R)‖ = 1")
    R_back = quat_to_R(q)
    _assert_close(R_back, R, 1e-10, "quat_to_R(R_to_quat(R)) = R")

    # quat composition matches matrix composition
    R1 = exp_SO3(torch.randn(8, 3, dtype=torch.float64))
    R2 = exp_SO3(torch.randn(8, 3, dtype=torch.float64))
    q1 = R_to_quat(R1)
    q2 = R_to_quat(R2)
    R12_via_mat = R1 @ R2
    R12_via_quat = quat_to_R(quat_mul(q1, q2))
    _assert_close(R12_via_quat, R12_via_mat, 1e-10,
                  "quat_mul(q1, q2) consistent with R1 @ R2")

    # quat inverse
    qinv = quat_inverse(q1)
    qprod = quat_mul(q1, qinv)
    # Should be identity quaternion (0, 0, 0, 1) — or its negation (sign-ambiguous).
    target = torch.tensor([0., 0., 0., 1.], dtype=torch.float64).expand_as(qprod)
    err = (qprod - target).abs().sum(dim=-1)
    err_neg = (qprod + target).abs().sum(dim=-1)
    err_min = torch.minimum(err, err_neg)
    assert err_min.max().item() < 1e-10, "quat inverse roundtrip failed"
    print(f"  ✓ quat_mul(q, quat_inverse(q)) = ±(0,0,0,1)")


def test_pose_compose_inverse():
    print("\n[pose composition + inverse]")
    # Random poses in storage form
    R1 = exp_SO3(torch.randn(8, 3, dtype=torch.float64))
    p1 = torch.randn(8, 3, dtype=torch.float64)
    R2 = exp_SO3(torch.randn(8, 3, dtype=torch.float64))
    p2 = torch.randn(8, 3, dtype=torch.float64)
    T1 = Rp_to_pose7(R1, p1)
    T2 = Rp_to_pose7(R2, p2)

    # Compose
    T12 = pose_compose(T1, T2)
    R12_check = R1 @ R2
    p12_check = (R1 @ p2.unsqueeze(-1)).squeeze(-1) + p1
    R12, p12 = pose7_to_Rp(T12)
    _assert_close(R12, R12_check, 1e-10, "pose_compose: R1 R2")
    _assert_close(p12, p12_check, 1e-10, "pose_compose: R1 p2 + p1")

    # Inverse
    T1_inv = pose_inverse(T1)
    T_id = pose_compose(T1, T1_inv)
    R_id, p_id = pose7_to_Rp(T_id)
    _assert_close(R_id,
                  torch.eye(3, dtype=torch.float64).expand_as(R_id), 1e-10,
                  "pose_compose(T, T^{-1}) → R = I")
    _assert_close(p_id, torch.zeros_like(p_id), 1e-10,
                  "pose_compose(T, T^{-1}) → p = 0")

    # log(T · T) = log(T) + log(T) only if T_1 commutes with T_2.
    # log(T · T^{-1}) = 0  is the more useful check.
    xi_zero = pose_log(T_id)
    _assert_close(xi_zero, torch.zeros(8, 6, dtype=torch.float64), 1e-10,
                  "pose_log(T · T^{-1}) = 0")


def test_pose_log_relative():
    print("\n[pose_log_relative]")
    # ξ = log(T_a^{-1} T_b) is the body-frame error twist of b w.r.t. a.
    xi_truth = torch.randn(16, 6, dtype=torch.float64) * 0.3                 # small twist
    R_a = exp_SO3(torch.randn(16, 3, dtype=torch.float64) * 0.5)
    p_a = torch.randn(16, 3, dtype=torch.float64) * 0.3
    T_a = Rp_to_pose7(R_a, p_a)

    # T_b = T_a · exp(ξ_truth)
    T_delta = pose_exp(xi_truth)
    T_b = pose_compose(T_a, T_delta)

    xi_back = pose_log_relative(T_a, T_b)
    _assert_close(xi_back, xi_truth, 1e-10,
                  "pose_log_relative(T_a, T_a · exp(ξ)) = ξ")


def test_hemisphere_consistency():
    print("\n[hemisphere consistency]")
    H1 = 16
    omegas = torch.randn(2, H1, 3, dtype=torch.float64) * 0.05
    Rs = exp_SO3(omegas)
    qs = R_to_quat(Rs)                                                       # (2, H1, 4)
    # Random sign flips along time
    flip_signs = torch.where(torch.rand(2, H1, 1) < 0.5,
                              -torch.ones(2, H1, 1, dtype=torch.float64),
                              torch.ones(2, H1, 1, dtype=torch.float64))
    qs_flipped = qs * flip_signs

    qs_consistent = hemisphere_consistency(qs_flipped)

    # All neighbouring dot products should be ≥ 0
    dots = (qs_consistent[:, 1:] * qs_consistent[:, :-1]).sum(dim=-1)
    assert (dots >= -1e-10).all(), "hemisphere consistency failed"
    print(f"  ✓ all neighbour dot products ≥ 0 (min={dots.min():.3e})")

    # Resulting rotations unchanged (double-cover)
    Rs_consistent = quat_to_R(qs_consistent)
    _assert_close(Rs_consistent, Rs, 1e-10,
                  "hemisphere flips do not change rotations")


def test_autograd_pose_jacobian():
    """Check: for a small SO(3)-rotation forward map T(q) = (exp_SO3([0,0,1]·q[0]),
    [q[1], q[2], q[3]]), the body-frame Jacobian matches a manually-derived one.
    """
    print("\n[autograd body-frame Jacobian]")

    def T_fn_Rp(q):                                                           # q: (B, 4)
        zeros = torch.zeros_like(q[..., :1])
        omega = torch.cat([zeros, zeros, q[..., :1]], dim=-1)                 # (..., 3)
        R = exp_SO3(omega)
        p = q[..., 1:]
        return R, p

    q = torch.randn(8, 4, dtype=torch.float64) * 0.3
    J = jacobian_pose_autograd(T_fn_Rp, q)                                   # (B, 6, 4)

    # Manual: body-frame velocity twist for δq at this q:
    #   - δq[0]: rotation around body-z; for R_z(q[0]) the body-frame ω is (0,0,1).
    #   - δq[1:4]: world-frame translation → body-frame ρ = R^T · e_i.
    R, _ = T_fn_Rp(q)
    Rt = R.transpose(-1, -2)
    expected = torch.zeros_like(J)
    expected[..., 5, 0] = 1.0                                                 # ω-z col 0
    expected[..., :3, 1:4] = Rt                                               # ρ cols 1-3 = R^T

    _assert_close(J, expected, 1e-9,
                  "jacobian_pose_autograd matches manual closed-form")


def test_autograd_pose_jacobian_finite_diff():
    """Finite-difference cross-check for a non-trivial T_fn (where I don't
    trust my own derivation): T(q) maps R^5 → SE(3) by composing rotations
    around different body axes and a translation that depends nonlinearly on q.
    """
    print("\n[autograd body-frame Jacobian — finite-difference cross-check]")

    def T_fn_Rp(q):                                                           # q: (B, 5)
        # ω(q) = (q[0], 0.5·q[1], q[2]·sin(q[3]))   p(q) = (q[3]², q[4], q[0]·q[2])
        omega = torch.stack([q[..., 0],
                              0.5 * q[..., 1],
                              q[..., 2] * torch.sin(q[..., 3])], dim=-1)
        R = exp_SO3(omega)
        p = torch.stack([q[..., 3] ** 2,
                          q[..., 4],
                          q[..., 0] * q[..., 2]], dim=-1)
        return R, p

    q = torch.randn(4, 5, dtype=torch.float64) * 0.4
    J_auto = jacobian_pose_autograd(T_fn_Rp, q)                              # (4, 6, 5)

    # FD reference: ξ_j(δ) = log_SE3( T(q)^{-1} · T(q + δ·e_j) ),  J ≈ ∂ξ/∂δ
    R_base, p_base = T_fn_Rp(q)
    Rt = R_base.transpose(-1, -2)
    p_inv = -(Rt @ p_base.unsqueeze(-1)).squeeze(-1)
    eps = 1e-6
    J_fd = torch.zeros_like(J_auto)
    for j in range(q.shape[-1]):
        e = torch.zeros_like(q); e[..., j] = eps
        R_p, p_p = T_fn_Rp(q + e)
        R_m, p_m = T_fn_Rp(q - e)
        R_rel_p = Rt @ R_p
        p_rel_p = (Rt @ p_p.unsqueeze(-1)).squeeze(-1) + p_inv
        R_rel_m = Rt @ R_m
        p_rel_m = (Rt @ p_m.unsqueeze(-1)).squeeze(-1) + p_inv
        xi_p = log_SE3(R_rel_p, p_rel_p)
        xi_m = log_SE3(R_rel_m, p_rel_m)
        J_fd[..., j] = (xi_p - xi_m) / (2 * eps)

    _assert_close(J_auto, J_fd, 1e-6,
                  "autograd Jacobian matches finite difference (n_q=5)")


def test_position_only_consistency():
    """Position-only special case: ω_φ ≡ 0 → exp_SE3((ρ, 0)) = (I, ρ)."""
    print("\n[position-only specialization]")
    rho = torch.randn(8, 3, dtype=torch.float64)
    xi = torch.cat([rho, torch.zeros_like(rho)], dim=-1)
    R, p = exp_SE3(xi)
    _assert_close(R,
                  torch.eye(3, dtype=torch.float64).expand_as(R), 1e-12,
                  "exp_SE3((ρ, 0)) → R = I")
    _assert_close(p, rho, 1e-12, "exp_SE3((ρ, 0)) → p = ρ")


def main():
    test_hat_vee()
    test_so3_exp_log_roundtrip()
    test_so3_left_jacobian()
    test_se3_exp_log_roundtrip()
    test_quaternion_consistency()
    test_pose_compose_inverse()
    test_pose_log_relative()
    test_hemisphere_consistency()
    test_autograd_pose_jacobian()
    test_autograd_pose_jacobian_finite_diff()
    test_position_only_consistency()
    print("\nAll SE(3) Lie group unit tests passed.")


if __name__ == "__main__":
    main()
