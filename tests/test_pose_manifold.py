"""Sanity tests for the pose-extended embodiment graph manifold (Phase 3).

Verifies (per extension.tex Sections 2-4):
    S1.  Retraction exactness:  g_φ(Retr_x(v)) = 0 (manifold adherence by construction).
    S2.  On-manifold tangent:  J_g · J_H^pose = 0  (numerical via finite difference of g_φ).
    S3.  G_pose symmetric, positive definite, and matches I + J_pose^⊤ W J_pose.
    S4.  Lift idempotence:  proj_to_tangent(lift(a)) = lift(a).
    S5.  log/exp roundtrip on M_φ:  exp_x(log_x(y)) = y for nearby (x, y).
    S6.  Position-only fallback:  setting ω_φ ≡ 0 (output_omega=False) recovers
         a manifold whose pose's R-block matches the analytic Franka FK.
    S7.  Closed-form analytic J_pose vs autograd reference (analytic-only).

All tests at machine precision unless noted.
Run:  python -m tests.test_pose_manifold
"""
from __future__ import annotations

import math

import pybullet_data
import torch

from smcdp.lie_se3 import (
    log_SE3, log_relative_Rp, quat_to_R, R_to_quat,
    Rp_to_pose7,
)
from smcdp.manifolds_pose import EmbodimentPoseGraphManifold, Franka7DoFPose
from smcdp.franka.self_model_pose import (
    PoseResidualMLP, LearnedSelfModelFranka7DoFPose, pose_self_model_loss,
)


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"
torch.manual_seed(0)


def _assert_close(a, b, atol, msg):
    diff = (a - b).abs().max().item()
    assert diff < atol, f"{msg}: max diff {diff:.3e} > atol {atol:.3e}"
    print(f"  ✓ {msg}: max diff {diff:.3e}")


def _make_analytic_manifold(dtype=torch.float64):
    m = Franka7DoFPose(urdf_path=URDF, sigma_p=0.01, sigma_R=0.1)
    m.chain = m.chain.to(dtype=dtype)
    m._chain_state = (None, dtype)
    return m


def _make_learned_manifold(seed=0, dtype=torch.float64, output_omega=True):
    torch.manual_seed(seed)
    net = PoseResidualMLP(
        n_q=7, n_z=1, hidden=64, n_layers=3,
        final_init_scale=0.05,                                            # bigger than default → meaningful residual
        output_omega=output_omega,
    ).to(dtype=dtype)
    # Push residual a bit harder so it's distinguishable from zero.
    with torch.no_grad():
        for p in net.net[-1].parameters():
            p.mul_(20.0)
    m = LearnedSelfModelFranka7DoFPose(
        residual_net=net, urdf_path=URDF,
        sigma_p=0.01, sigma_R=0.1,
    )
    m.chain = m.chain.to(dtype=dtype)
    m._chain_state = (None, dtype)
    return m


def _sample_chart(m, B=8, dtype=torch.float64):
    lower, upper = m.joint_limits(dtype=dtype)
    margin = 0.2 * (upper - lower)
    lo = lower + margin
    hi = upper - margin
    q = lo + (hi - lo) * torch.rand(B, 7, dtype=dtype)
    z = torch.rand(B, 1, dtype=dtype) * m.tool_z_max
    return q, z


def test_S1_retraction_exactness():
    print("\n[S1] retraction exactness on M_φ^pose")
    m = _make_learned_manifold()
    q, z = _sample_chart(m, B=8)
    x = m.make_x(q, z)
    # On-manifold check
    g0 = m.constraint(x)
    # Numerical floor ~3e-8 from pytorch_kinematics' non-orthogonality in
    # float64 (internal axis-angle composition); tolerated.
    _assert_close(g0, torch.zeros_like(g0), 1e-6, "g_φ(x) = 0 for x = H(q, z)")

    # Retract by random chart-vector and re-check
    a = torch.randn(8, 7, dtype=q.dtype) * 0.05
    v = m.lift_chart_to_tangent(x, a)
    x1 = m.exp(x, v)
    g1 = m.constraint(x1)
    _assert_close(g1, torch.zeros_like(g1), 1e-6, "g_φ(Retr_x(v)) = 0")


def test_S2_on_manifold_tangent_J_g_J_H():
    print("\n[S2] on-manifold tangent identity  J_g · J_H^pose = 0")
    m = _make_learned_manifold()
    q, z = _sample_chart(m, B=4)
    x = m.make_x(q, z)
    a = torch.randn(4, 7, dtype=q.dtype) * 0.05
    # Lifted tangent
    v = m.lift_chart_to_tangent(x, a)                                    # (B, n_q + 6 + n_z)
    # Verify on-manifold tangent: g_φ(x) = 0 already (S1).  Test the
    # linearisation: g_φ(x + ε v) − g_φ(x) ≈ ε · (J_g · v).  But J_g · J_H · a
    # should be 0 on-manifold per extension.tex Eq. (12).  We test:
    # finite-diff (g(x + ε v) − g(x − ε v)) / (2ε)  should ≈ 0.
    eps = 1e-4
    # Construct x + ε v in trivialized form: q-block uses ε v_q, pose-block
    # uses exp_SE3(ε v_xi) right-multiplied to T_φ(q, z).
    v_q = v[..., :7]
    v_xi = v[..., 7:13]
    q_p = q + eps * v_q
    # We reconstruct the perturbed x WITHOUT calling make_x (which would lift
    # back to M); instead we use the on-manifold T_φ(q, z) right-multiplied by
    # exp(ε v_xi) and concatenate.  The aim is to test J_g linearised at x.
    R_phi, p_phi = m.T_phi_Rp(q, z)
    from smcdp.lie_se3 import exp_SE3, compose_Rp
    R_d, p_d = exp_SE3(eps * v_xi)
    R_perturbed, p_perturbed = compose_Rp(R_phi, p_phi, R_d, p_d)
    q_R_perturbed = R_to_quat(R_perturbed)
    x_p = torch.cat([q_p, q_R_perturbed, p_perturbed, z], dim=-1)
    # And the - perturbation
    R_dm, p_dm = exp_SE3(-eps * v_xi)
    R_pm, p_pm = compose_Rp(R_phi, p_phi, R_dm, p_dm)
    q_R_pm = R_to_quat(R_pm)
    x_m = torch.cat([q - eps * v_q, q_R_pm, p_pm, z], dim=-1)

    g_p = m.constraint(x_p)
    g_m = m.constraint(x_m)
    Jg_v = (g_p - g_m) / (2 * eps)                                       # FD of J_g · v
    _assert_close(Jg_v, torch.zeros_like(Jg_v), 1e-6,
                  "J_g · J_H^pose · a ≈ 0  (FD,  ε=1e-4)")


def test_S3_G_pose_properties():
    print("\n[S3] G_pose properties (symmetric, PD, formula)")
    m = _make_learned_manifold()
    q, z = _sample_chart(m, B=4)
    G = m.G_pose(q, z)
    _assert_close(G, G.transpose(-1, -2), 1e-10, "G_pose symmetric")
    eigs = torch.linalg.eigvalsh(G)
    assert eigs.min().item() > 1e-8, f"G_pose not PD: min eig {eigs.min():.3e}"
    print(f"  ✓ G_pose positive definite (min eig {eigs.min():.3e}, max eig {eigs.max():.3e})")

    # Re-derive: G = I + J^T W J
    Jp = m.jacobian_pose(q, z)
    W = m._W_diag(q).unsqueeze(-1)                                       # (6, 1)
    eye = torch.eye(7, dtype=q.dtype)
    G_check = eye + Jp.transpose(-1, -2) @ (W * Jp)
    _assert_close(G, G_check, 1e-10, "G_pose = I + J_pose^T W J_pose")


def test_S4_lift_idempotence():
    print("\n[S4] proj_to_tangent ∘ lift = lift")
    m = _make_learned_manifold()
    q, z = _sample_chart(m, B=4)
    x = m.make_x(q, z)
    a = torch.randn(4, 7, dtype=q.dtype) * 0.1
    v = m.lift_chart_to_tangent(x, a)
    v_proj = m.proj_to_tangent(x, v)
    _assert_close(v_proj, v, 1e-9, "proj(lift(a)) = lift(a)")


def test_S5_log_exp_roundtrip():
    print("\n[S5] log/exp roundtrip on M  (small distance)")
    m = _make_learned_manifold()
    q, z = _sample_chart(m, B=4)
    x = m.make_x(q, z)
    delta_q = torch.randn(4, 7, dtype=q.dtype) * 0.01
    y = m.make_x(q + delta_q, z)
    v = m.log(x, y)
    x_reto = m.exp(x, v)
    # Compare on the q-chart (storage-form quaternions can have hemisphere flips)
    q_x_reto, _, _, _ = m.split_x(x_reto)
    q_y, _, _, _ = m.split_x(y)
    _assert_close(q_x_reto, q_y, 1e-12, "exp(log(y)) chart matches y")


def test_S6_position_only_fallback():
    print("\n[S6] position-only fallback (output_omega=False)")
    m = _make_learned_manifold(output_omega=False)
    q, z = _sample_chart(m, B=4)
    R, p = m.T_phi_Rp(q, z)
    # With ω_φ ≡ 0, R_φ should match the analytic R_hand.
    # Recover analytic via parent class
    R_analytic, p_analytic = Franka7DoFPose.T_phi_Rp(m, q, z)
    _assert_close(R, R_analytic, 1e-12,
                  "ω_φ ≡ 0 ⇒ R_φ = R_analytic")
    # And p_φ = p_analytic + R_analytic · ρ_φ (extension.tex Sec. 1)
    xi = m.residual_net(q, z)
    rho = xi[..., :3]
    p_expected = p_analytic + (R_analytic @ rho.unsqueeze(-1)).squeeze(-1)
    _assert_close(p, p_expected, 1e-12,
                  "p_φ = p_analytic + R_analytic · ρ_φ")


def test_S7_analytic_jacobian_vs_autograd():
    print("\n[S7] closed-form J_pose vs autograd (analytic-only Franka)")
    m = _make_analytic_manifold()
    q, z = _sample_chart(m, B=4)
    Jp_cf = m.jacobian_pose(q, z)                                        # closed-form (Franka7DoFPose)

    # Reference: finite-diff body-frame twist.
    eps = 1e-6
    R_b, p_b = m.T_phi_Rp(q, z)
    Rt_b = R_b.transpose(-1, -2)
    p_inv = -(Rt_b @ p_b.unsqueeze(-1)).squeeze(-1)
    Jp_fd = torch.zeros_like(Jp_cf)
    for j in range(7):
        e = torch.zeros_like(q); e[..., j] = eps
        R_p, p_p = m.T_phi_Rp(q + e, z)
        R_m, p_m = m.T_phi_Rp(q - e, z)
        from smcdp.lie_se3 import compose_Rp
        Rrp, prp = compose_Rp(Rt_b, p_inv, R_p, p_p)
        Rrm, prm = compose_Rp(Rt_b, p_inv, R_m, p_m)
        xi_p = log_SE3(Rrp, prp)
        xi_m = log_SE3(Rrm, prm)
        Jp_fd[..., j] = (xi_p - xi_m) / (2 * eps)

    _assert_close(Jp_cf, Jp_fd, 1e-5,
                  "closed-form J_pose ≈ FD reference  (Franka analytic)")


def test_S7b_learned_jacobian_vs_FD():
    print("\n[S7b] learned-model hybrid J_pose vs finite difference")
    m = _make_learned_manifold()
    q, z = _sample_chart(m, B=4)
    Jp_hybrid = m.jacobian_pose(q, z)                                    # hybrid (Ad_inv·J_a + J_d)

    eps = 1e-6
    R_b, p_b = m.T_phi_Rp(q, z)
    Rt_b = R_b.transpose(-1, -2)
    p_inv = -(Rt_b @ p_b.unsqueeze(-1)).squeeze(-1)
    Jp_fd = torch.zeros_like(Jp_hybrid)
    for j in range(7):
        e = torch.zeros_like(q); e[..., j] = eps
        R_p, p_p = m.T_phi_Rp(q + e, z)
        R_m, p_m = m.T_phi_Rp(q - e, z)
        from smcdp.lie_se3 import compose_Rp
        Rrp, prp = compose_Rp(Rt_b, p_inv, R_p, p_p)
        Rrm, prm = compose_Rp(Rt_b, p_inv, R_m, p_m)
        xi_p = log_SE3(Rrp, prp)
        xi_m = log_SE3(Rrm, prm)
        Jp_fd[..., j] = (xi_p - xi_m) / (2 * eps)

    _assert_close(Jp_hybrid, Jp_fd, 1e-5,
                  "hybrid J_pose ≈ FD reference  (Franka + learned ξ_φ)")


def test_S8_pose_self_model_loss_runs():
    print("\n[S8] pose_self_model_loss forward + backward sanity")
    torch.manual_seed(0)
    net = PoseResidualMLP(n_q=7, n_z=1, hidden=64, n_layers=3,
                           final_init_scale=1e-2).to(dtype=torch.float64)
    m = _make_analytic_manifold()
    q, z = _sample_chart(m, B=16)
    R_t, p_t = m.T_phi_Rp(q, z)                                          # use analytic FK as "truth"
    # Add a small synthetic perturbation so the loss is non-zero
    R_perturb = exp_SE3 = None
    from smcdp.lie_se3 import exp_SE3 as _exp, compose_Rp
    perturb = torch.randn(16, 6, dtype=q.dtype) * 0.01
    R_d, p_d = _exp(perturb)
    R_t, p_t = compose_Rp(R_t, p_t, R_d, p_d)

    loss, info = pose_self_model_loss(
        net, q, z, (R_t, p_t),
        fk_analytic_Rp=lambda q_, z_: m.T_phi_Rp(q_, z_),
        w_p=1.0, w_R=1.0, beta_p=1e-3, beta_R=1e-3,
    )
    print(f"  loss = {loss.item():.4f}, info = {info}")
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(g.norm().item() < 1e6 for g in grads), \
        "no parameter grads or grad explosion"
    print(f"  ✓ {len(grads)} parameter grads computed; max ‖∇‖ = "
          f"{max(g.norm().item() for g in grads):.3e}")


def main():
    test_S1_retraction_exactness()
    test_S2_on_manifold_tangent_J_g_J_H()
    test_S3_G_pose_properties()
    test_S4_lift_idempotence()
    test_S5_log_exp_roundtrip()
    test_S6_position_only_fallback()
    test_S7_analytic_jacobian_vs_autograd()
    test_S7b_learned_jacobian_vs_FD()
    test_S8_pose_self_model_loss_runs()
    print("\nAll pose-manifold sanity tests passed.")


if __name__ == "__main__":
    main()
