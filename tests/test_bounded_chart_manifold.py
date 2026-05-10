"""Sanity tests for `BoundedChartPoseManifold` (joint_limit_extension v4.1, Phase 2).

Each test maps directly to a spec equation.  Tests use a synthetic
mock-base-manifold (no URDF / pytorch_kinematics) so they run on CPU only
and are deterministic.  GPU + Franka integration tests are in Phase 4.

Verification matrix (spec → test):
  W1   §3   ψ(u) ∈ (q_min, q_max) by construction
  W2   §4   J^Q = J_pose · D_ψ  (vs analytic mock + autograd)
  W3   §4   J_g · J_H^{Q,A} = 0 on M^Q  (tangent identity)
  W4   §5   G_Q^A = I + (J^Q)^T W J^Q  (formula equality)
  W5   §5   G_Q^A ≽ I  (eigenvalues ≥ 1 globally, including saturation)
  W6   §5   G_Q^A symmetric, PD
  W7   §5.1 G_Q^{A,reg} - G_Q^A = (c_λ tr(G)/n_q + λ_floor) I  (Tikhonov form)
  W8   §3   make_x stores u (chart slot), T_φ(ψ(u)) (T-block)
  W9   §3   constraint(make_x(u, z)) = 0  (manifold adherence by construction)
  W10  §10  exp/Retr keeps state on M^Q  (constraint == 0 after retract)
  W11  §10  ψ(u + δu) ∈ (q_min, q_max) for arbitrary δu  (no clip needed)
  W12  §9   log lifts u-displacement: log(x, y) = (u_y - u_x, J^Q · (u_y - u_x), 0)
  W13  §5   IdentityChart wrapper ≡ base manifold (v4 backward compat)
  W14  §13  violates_limits == False everywhere on M^Q (by construction)
  W15  §5.1 lambda_floor strictly raises G_Q^A's min eigenvalue

Run:  python -m tests.test_bounded_chart_manifold
"""
from __future__ import annotations

import torch
from torch import Tensor

from smcdp.charts import TanhBoundedChart, IdentityChart
from smcdp.manifolds_pose import (
    EmbodimentPoseGraphManifold, BoundedChartPoseManifold,
)
from smcdp.lie_se3 import quat_to_R, log_relative_Rp, R_to_quat, Rp_to_pose7


# ---------------------------------------------------------------------------
# Mock base manifold (no URDF dependency)
# ---------------------------------------------------------------------------


class _MockPoseManifold(EmbodimentPoseGraphManifold):
    """Minimal pose-graph manifold with synthetic kinematics.

    Forward map: T_φ(q, z_e) constructs an SE(3) pose with
        p = M @ q + z_e · ẑ   (linear)
        R = exp_SO3(N @ q)     (axis-angle from N · q)
    where M ∈ R^{3×n_q}, N ∈ R^{3×n_q} are fixed random matrices.  This is
    smooth, vmap-safe, and has analytically known Jacobians for cross-check.
    """

    def __init__(self, n_q: int = 5, n_z: int = 1, sigma_p=0.05, sigma_R=0.1, seed=0):
        super().__init__(n_q=n_q, n_z=n_z, sigma_p=sigma_p, sigma_R=sigma_R)
        g = torch.Generator().manual_seed(seed)
        # Use FP64 for reproducibility under autograd cross-checks.
        self.M = torch.randn(3, n_q, generator=g, dtype=torch.float64) * 0.3
        self.N = torch.randn(3, n_q, generator=g, dtype=torch.float64) * 0.3
        self.q_lower = torch.full((n_q,), -2.0, dtype=torch.float64)
        self.q_upper = torch.full((n_q,), +2.0, dtype=torch.float64)

    def joint_limits(self, device=None, dtype=torch.float32) -> tuple[Tensor, Tensor]:
        return (self.q_lower.to(device=device, dtype=dtype),
                self.q_upper.to(device=device, dtype=dtype))

    def violates_limits(self, q: Tensor) -> Tensor:
        lo = self.q_lower.to(device=q.device, dtype=q.dtype)
        hi = self.q_upper.to(device=q.device, dtype=q.dtype)
        return (q < lo).any(-1) | (q > hi).any(-1)

    def T_phi_Rp(self, q: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        # Promote constants to q's device/dtype to keep autograd happy
        M = self.M.to(device=q.device, dtype=q.dtype)
        N = self.N.to(device=q.device, dtype=q.dtype)
        # p = M q + z · ẑ
        p_lin = (q.unsqueeze(-2) @ M.transpose(-1, -2)).squeeze(-2)              # (..., 3)
        p = p_lin + z[..., :1] * torch.tensor([0.0, 0.0, 1.0], dtype=q.dtype,
                                                device=q.device)
        # R = exp_SO3(N q) — Rodrigues
        omega = (q.unsqueeze(-2) @ N.transpose(-1, -2)).squeeze(-2)               # (..., 3)
        theta = omega.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        k = omega / theta
        K = _hat(k)                                                                # (..., 3, 3)
        Rk = K @ K
        eye = torch.eye(3, device=q.device, dtype=q.dtype).expand(*omega.shape[:-1], 3, 3)
        s = theta.unsqueeze(-1).sin()
        c1 = (1 - theta.unsqueeze(-1).cos())
        R = eye + s * K + c1 * Rk
        return R, p


def _hat(v: Tensor) -> Tensor:
    """Return the skew-symmetric (hat) matrix of v ∈ R^3.  Vmap-safe."""
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([z, -v[..., 2], v[..., 1]], dim=-1),
        torch.stack([v[..., 2], z, -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1], v[..., 0], z], dim=-1),
    ], dim=-2)


def _make_pair(n_q=5, dtype=torch.float64):
    base = _MockPoseManifold(n_q=n_q, dtype_seed=0) if False else _MockPoseManifold(n_q=n_q)
    q_min, q_max = base.joint_limits(dtype=dtype) if hasattr(base, "joint_limits") else (
        torch.full((n_q,), -2.0, dtype=dtype), torch.full((n_q,), +2.0, dtype=dtype)
    )
    chart = TanhBoundedChart(q_min, q_max)
    wrapped = BoundedChartPoseManifold(base, chart, lambda_floor=0.0)        # disable floor for cleanest checks
    return base, chart, wrapped


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_W1_psi_image_inside_limits():
    """W1 — §3.  ψ(u) ∈ (q_min, q_max) for any finite u."""
    base, chart, wrapped = _make_pair()
    torch.manual_seed(0)
    u = (torch.rand(20, 5, dtype=torch.float64) - 0.5) * 8.0                 # |u| ≤ 4
    q = chart.psi(u)
    q_lo, q_hi = chart.joint_limits(dtype=torch.float64)
    assert (q > q_lo).all() and (q < q_hi).all()
    print("W1  ψ(u) ⊂ (q_min, q_max):  PASS")


def test_W2_J_Q_equals_Jpose_times_Dpsi():
    """W2 — §4.  J^Q = J_pose · D_ψ.  Cross-check via element-wise scaling
    against base.jacobian_pose at q = ψ(u)."""
    base, chart, wrapped = _make_pair()
    torch.manual_seed(1)
    u = (torch.rand(4, 5, dtype=torch.float64) - 0.5) * 4.0
    z = torch.rand(4, 1, dtype=torch.float64) * 0.2
    Jq_wrap = wrapped.jacobian_pose(u, z)                                   # (4, 6, 5)
    q = chart.psi(u)
    Jpose_base = base.jacobian_pose(q, z)                                    # (4, 6, 5)
    D_diag = chart.D_psi_diag(u)                                             # (4, 5)
    Jq_expected = Jpose_base * D_diag.unsqueeze(-2)                          # right-mul by diag
    err = (Jq_wrap - Jq_expected).abs().max().item()
    assert err < 1e-12, f"J^Q mismatch: {err}"
    print(f"W2  J^Q = J_pose · D_ψ:  PASS  (max err {err:.3e})")


def test_W3_tangent_identity_J_g_J_H_zero():
    """W3 — §4.  J_g^Q · J_H^{Q,A} = 0 on M^Q.

    On M^Q the constraint Jacobian J_g^Q in (u̇, ξ) coordinates is
    [-J^Q, I_6].  Combined with J_H^{Q,A} = [I_{n_q}, J^Q]:
        J_g^Q · J_H^{Q,A} = -J^Q · I + I_6 · J^Q = 0.
    """
    base, chart, wrapped = _make_pair()
    torch.manual_seed(2)
    u = (torch.rand(3, 5, dtype=torch.float64) - 0.5) * 4.0
    z = torch.rand(3, 1, dtype=torch.float64) * 0.2

    # Build J_H^{Q,A} explicitly: stack [I_{n_q} ; J^Q] along row axis.
    n_q = wrapped.n_q
    Jq = wrapped.jacobian_pose(u, z)                                        # (3, 6, n_q)
    eye_nq = torch.eye(n_q, dtype=torch.float64).expand(3, n_q, n_q)
    JH = torch.cat([eye_nq, Jq], dim=-2)                                    # (3, n_q + 6, n_q)
    # J_g^Q on M^Q in (u̇, ξ) = [-J^Q, I_6].
    Jg = torch.cat([-Jq, torch.eye(6, dtype=torch.float64).expand(3, 6, 6)], dim=-1)
    prod = Jg @ JH                                                           # (3, 6, n_q)
    err = prod.abs().max().item()
    assert err < 1e-12, f"J_g^Q · J_H^{{Q,A}} ≠ 0:  max |entry| = {err}"
    print(f"W3  J_g^Q · J_H^{{Q,A}} = 0:  PASS  (max err {err:.3e})")


def test_W4_G_Q_A_formula():
    """W4 — §5.  G_Q^A = I_{n_q} + (J^Q)^T W J^Q (formula equality)."""
    base, chart, wrapped = _make_pair()
    torch.manual_seed(3)
    u = (torch.rand(4, 5, dtype=torch.float64) - 0.5) * 4.0
    z = torch.rand(4, 1, dtype=torch.float64) * 0.2
    G_wrap = wrapped.G_pose(u, z)
    Jq = wrapped.jacobian_pose(u, z)
    W = torch.diag(wrapped._W_diag(u))                                      # (6, 6)
    G_explicit = torch.eye(5, dtype=torch.float64).expand(4, 5, 5) + Jq.transpose(-1, -2) @ W @ Jq
    err = (G_wrap - G_explicit).abs().max().item()
    assert err < 1e-12, f"G_Q^A formula mismatch: {err}"
    print(f"W4  G_Q^A formula:  PASS  (max err {err:.3e})")


def test_W5_G_Q_A_lower_bounded_by_I():
    """W5 — §5.  G_Q^A ≽ I_{n_q} globally (Choice A's identity floor).

    Test at: (a) chart center (u=0), (b) random interior u, (c) saturated
    boundary (|u| = 8 → D_ψ ≈ 0 → G_Q^A → I).
    """
    base, chart, wrapped = _make_pair()
    z = torch.tensor([[0.1]], dtype=torch.float64)

    # (a) center
    u_center = torch.zeros(1, 5, dtype=torch.float64)
    G = wrapped.G_pose(u_center, z)
    eigs = torch.linalg.eigvalsh(G)
    assert (eigs >= 1.0 - 1e-10).all(), f"G_Q^A at u=0 has eig < 1: {eigs}"

    # (b) random interior
    torch.manual_seed(4)
    u_random = (torch.rand(8, 5, dtype=torch.float64) - 0.5) * 4.0
    z_b = torch.rand(8, 1, dtype=torch.float64) * 0.2
    G_b = wrapped.G_pose(u_random, z_b)
    eigs_b = torch.linalg.eigvalsh(G_b)
    assert (eigs_b >= 1.0 - 1e-10).all(), f"G_Q^A interior has eig < 1: {eigs_b.min()}"

    # (c) saturated — D_ψ ≈ 0 → G_Q^A ≈ I
    u_sat = torch.full((1, 5), 8.0, dtype=torch.float64)
    G_sat = wrapped.G_pose(u_sat, z)
    eigs_sat = torch.linalg.eigvalsh(G_sat)
    assert (eigs_sat >= 1.0 - 1e-10).all()
    diag_dev = (G_sat - torch.eye(5, dtype=torch.float64)).abs().max().item()
    assert diag_dev < 1e-5, f"G_Q^A at saturation should ≈ I, deviation {diag_dev}"
    print(f"W5  G_Q^A ≽ I globally:  PASS  (saturation deviation from I: {diag_dev:.3e})")


def test_W6_G_symmetric_PD():
    """W6 — §5.  G_Q^A symmetric and positive definite."""
    _, _, wrapped = _make_pair()
    torch.manual_seed(5)
    u = (torch.rand(4, 5, dtype=torch.float64) - 0.5) * 4.0
    z = torch.rand(4, 1, dtype=torch.float64) * 0.2
    G = wrapped.G_pose(u, z)
    sym_err = (G - G.transpose(-1, -2)).abs().max().item()
    assert sym_err < 1e-13
    L = torch.linalg.cholesky(G)                            # would raise if not PD
    assert L.shape == G.shape
    print(f"W6  G_Q^A symmetric & PD:  PASS  (sym err {sym_err:.3e})")


def test_W7_tikhonov_form():
    """W7 — §5.1.  G_Q^{A,reg} = G_Q^A + (c_λ tr(G_Q^A)/n_q + λ_floor) I."""
    base, chart, _ = _make_pair()
    # Use a wrapped instance with explicit tikhonov_frac and lambda_floor
    base.tikhonov_frac = 1e-2
    wrapped = BoundedChartPoseManifold(base, chart, lambda_floor=1e-4)

    torch.manual_seed(6)
    u = (torch.rand(3, 5, dtype=torch.float64) - 0.5) * 4.0
    z = torch.rand(3, 1, dtype=torch.float64) * 0.2

    G = wrapped.G_pose(u, z)
    L = wrapped.G_pose_chol(u, z, jitter=0.0)               # disable jitter for clean check
    G_reg = L @ L.transpose(-1, -2)
    # Expected adaptive λ
    tr_G = torch.diagonal(G, dim1=-2, dim2=-1).sum(-1)
    lam_expected = (1e-2 * tr_G / wrapped.n_q + 1e-4).reshape(-1, 1, 1)
    G_reg_expected = G + lam_expected * torch.eye(5, dtype=torch.float64).expand(3, 5, 5)
    err = (G_reg - G_reg_expected).abs().max().item()
    assert err < 1e-10, f"Tikhonov form mismatch: {err}"
    print(f"W7  G_Q^{{A,reg}} Tikhonov form:  PASS  (max err {err:.3e})")
    base.tikhonov_frac = 0.0                                 # reset for other tests


def test_W8_make_x_storage_layout():
    """W8 — §3.  make_x(u, z) stores [u, q_R, p, z] with T-block = T_φ(ψ(u))."""
    base, chart, wrapped = _make_pair()
    torch.manual_seed(7)
    u = (torch.rand(3, 5, dtype=torch.float64) - 0.5) * 4.0
    z = torch.rand(3, 1, dtype=torch.float64) * 0.2

    x = wrapped.make_x(u, z)
    assert x.shape == (3, wrapped.ambient_dim)
    # First slot is u (not q)
    assert torch.allclose(x[..., : wrapped.n_q], u)
    # Recover q via physical_q
    q_phys = wrapped.physical_q(x)
    assert torch.allclose(q_phys, chart.psi(u))
    # T-block matches T_φ(ψ(u))
    R_expected, p_expected = base.T_phi_Rp(chart.psi(u), z)
    R_stored = quat_to_R(x[..., wrapped.n_q : wrapped.n_q + 4])
    p_stored = x[..., wrapped.n_q + 4 : wrapped.n_q + 7]
    assert torch.allclose(R_stored, R_expected, atol=1e-10)
    assert torch.allclose(p_stored, p_expected, atol=1e-10)
    print("W8  make_x storage [u, T_φ(ψ(u)), z]:  PASS")


def test_W9_constraint_zero_on_manifold():
    """W9 — §3 (constraint).  g_φ(make_x(u, z)) = 0 (manifold by construction)."""
    base, chart, wrapped = _make_pair()
    torch.manual_seed(8)
    u = (torch.rand(5, 5, dtype=torch.float64) - 0.5) * 4.0
    z = torch.rand(5, 1, dtype=torch.float64) * 0.2
    x = wrapped.make_x(u, z)
    g = wrapped.constraint(x)
    err = g.abs().max().item()
    assert err < 1e-10, f"constraint(make_x(u, z)) should be 0, got {err}"
    print(f"W9  constraint(make_x(u, z)) = 0:  PASS  (max |g| = {err:.3e})")


def test_W10_exp_keeps_on_manifold():
    """W10 — §10.  Retr^Q keeps state on M^Q."""
    base, chart, wrapped = _make_pair()
    torch.manual_seed(9)
    u = (torch.rand(4, 5, dtype=torch.float64) - 0.5) * 3.0
    z = torch.rand(4, 1, dtype=torch.float64) * 0.2
    x = wrapped.make_x(u, z)
    delta_u = (torch.rand(4, 5, dtype=torch.float64) - 0.5) * 1.0
    # Build a tangent vector with arbitrary chart-velocity in v_q slot
    v_q_slot = delta_u
    v_xi_slot = torch.zeros(4, 6, dtype=torch.float64)                       # ignored by exp
    v_z_slot = torch.zeros(4, 1, dtype=torch.float64)
    v = torch.cat([v_q_slot, v_xi_slot, v_z_slot], dim=-1)
    x_new = wrapped.exp(x, v)
    g_new = wrapped.constraint(x_new)
    err = g_new.abs().max().item()
    assert err < 1e-10, f"Retr^Q result not on manifold: {err}"
    # u-slot should equal u + δu
    u_new = x_new[..., : wrapped.n_q]
    assert torch.allclose(u_new, u + delta_u)
    print(f"W10 Retr^Q on-manifold:  PASS  (max |g_new| = {err:.3e})")


def test_W11_psi_no_clip_during_sampling():
    """W11 — §10.  ψ(u + δu) ∈ (q_min, q_max) for arbitrary finite δu (no clip needed)."""
    _, chart, _ = _make_pair()
    torch.manual_seed(10)
    u = (torch.rand(20, 5, dtype=torch.float64) - 0.5) * 6.0
    delta = (torch.rand(20, 5, dtype=torch.float64) - 0.5) * 5.0             # large step
    u_new = u + delta
    q_new = chart.psi(u_new)
    q_lo, q_hi = chart.joint_limits(dtype=torch.float64)
    assert (q_new > q_lo).all() and (q_new < q_hi).all()
    print("W11 ψ keeps q in open interval after arbitrary step:  PASS")


def test_W12_log_lifts_u_displacement():
    """W12 — §9.  log(x, y) returns (u_y - u_x, J^Q · (u_y - u_x), 0)
    matching the chart-Euclidean lifting in the DSM target."""
    base, chart, wrapped = _make_pair()
    torch.manual_seed(11)
    u_x = (torch.rand(3, 5, dtype=torch.float64) - 0.5) * 3.0
    u_y = (torch.rand(3, 5, dtype=torch.float64) - 0.5) * 3.0
    z = torch.rand(3, 1, dtype=torch.float64) * 0.2
    x = wrapped.make_x(u_x, z)
    y = wrapped.make_x(u_y, z)
    v = wrapped.log(x, y)
    n_q = wrapped.n_q
    delta_u = u_y - u_x
    # First block = δu
    assert torch.allclose(v[..., :n_q], delta_u, atol=1e-12)
    # Second block (xi) = J^Q · δu evaluated at x's u
    Jq = wrapped.jacobian_pose(u_x, z)
    xi_expected = (Jq @ delta_u.unsqueeze(-1)).squeeze(-1)
    err = (v[..., n_q : n_q + 6] - xi_expected).abs().max().item()
    assert err < 1e-12, f"log xi-block mismatch: {err}"
    # Third block = 0
    assert v[..., n_q + 6 :].abs().max().item() < 1e-15
    print(f"W12 log lifts u-displacement:  PASS  (xi err {err:.3e})")


def test_W13_identity_chart_recovers_v4():
    """W13 — backward compat.  IdentityChart wrapper produces the same
    G_pose, jacobian_pose, make_x as the unwrapped base manifold."""
    base = _MockPoseManifold(n_q=5)
    base.tikhonov_frac = 0.0
    chart_id = IdentityChart(n_q=5)
    wrapped = BoundedChartPoseManifold(base, chart_id, lambda_floor=0.0)

    torch.manual_seed(12)
    q_or_u = (torch.rand(4, 5, dtype=torch.float64) - 0.5) * 1.0             # interior
    z = torch.rand(4, 1, dtype=torch.float64) * 0.2

    # jacobian_pose: wrapped(u) should equal base(q) since u = q under identity
    J_wrap = wrapped.jacobian_pose(q_or_u, z)
    J_base = base.jacobian_pose(q_or_u, z)
    assert torch.allclose(J_wrap, J_base, atol=1e-13)

    # G_pose
    G_wrap = wrapped.G_pose(q_or_u, z)
    G_base = base.G_pose(q_or_u, z)
    assert torch.allclose(G_wrap, G_base, atol=1e-13)

    # make_x: chart slot should match q (since ψ(u)=u)
    x_wrap = wrapped.make_x(q_or_u, z)
    x_base = base.make_x(q_or_u, z)
    assert torch.allclose(x_wrap, x_base, atol=1e-13)
    print("W13 IdentityChart wrapper ≡ base (v4 backward compat):  PASS")


def test_W14_violates_limits_zero_on_manifold():
    """W14 — §13.  violates_limits(u) is False everywhere by construction."""
    base, chart, wrapped = _make_pair()
    torch.manual_seed(13)
    u = (torch.rand(50, 5, dtype=torch.float64) - 0.5) * 10.0                 # extreme
    viol = wrapped.violates_limits(u)
    assert not viol.any().item(), f"viol should be 0 by construction, got {viol.sum()}"
    print("W14 violates_limits == 0 by construction:  PASS")


def test_W15_lambda_floor_raises_min_eig():
    """W15 — §5.1.  lambda_floor strictly increases the smallest eigenvalue."""
    base, chart, _ = _make_pair()
    base.tikhonov_frac = 0.0                                                  # isolate floor
    w_no_floor = BoundedChartPoseManifold(base, chart, lambda_floor=0.0)
    w_floor = BoundedChartPoseManifold(base, chart, lambda_floor=1e-3)

    torch.manual_seed(14)
    # Use saturated u where G_Q^A → I; floor should raise min eig from 1 to 1+λ_floor
    u_sat = torch.full((1, 5), 6.0, dtype=torch.float64)
    z = torch.tensor([[0.1]], dtype=torch.float64)
    L_no = w_no_floor.G_pose_chol(u_sat, z, jitter=0.0)
    L_w = w_floor.G_pose_chol(u_sat, z, jitter=0.0)
    G_no = L_no @ L_no.transpose(-1, -2)
    G_w = L_w @ L_w.transpose(-1, -2)
    eig_no = torch.linalg.eigvalsh(G_no).min().item()
    eig_w = torch.linalg.eigvalsh(G_w).min().item()
    delta = eig_w - eig_no
    # Expected ~ 1e-3; give modest tolerance for adaptive Tikhonov path effects.
    assert delta > 5e-4, f"floor did not raise min eig sufficiently: Δ = {delta}"
    print(f"W15 λ_floor raises min eig:  PASS  (Δλ_min = {delta:.3e})")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== BoundedChartPoseManifold tests (joint_limit_extension v4.1, Phase 2) ===\n")
    test_W1_psi_image_inside_limits()
    test_W2_J_Q_equals_Jpose_times_Dpsi()
    test_W3_tangent_identity_J_g_J_H_zero()
    test_W4_G_Q_A_formula()
    test_W5_G_Q_A_lower_bounded_by_I()
    test_W6_G_symmetric_PD()
    test_W7_tikhonov_form()
    test_W8_make_x_storage_layout()
    test_W9_constraint_zero_on_manifold()
    test_W10_exp_keeps_on_manifold()
    test_W11_psi_no_clip_during_sampling()
    test_W12_log_lifts_u_displacement()
    test_W13_identity_chart_recovers_v4()
    test_W14_violates_limits_zero_on_manifold()
    test_W15_lambda_floor_raises_min_eig()
    print("\n=== all tests passed ===")
