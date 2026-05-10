"""Phase 4 integration tests for trajectories under v4.1 bounded chart.

Verifies trajectory-level functions (forward GRW, reverse GRW, DSM target,
smoothness guidance) operate correctly when the manifold is wrapped with a
BoundedChartPoseManifold.

Each test maps directly to a spec equation:
  T1  §9         a*_Q = G_Q^A^{-1} [(u_0 - u_r) + (J^Q)^T W Log_SE3(...)]
  T2  §7.1       Forward GRW preserves manifold adherence (g_φ = 0)
  T3  §10        Reverse GRW retraction enforces feasibility throughout
  T4  §12.3      Smoothness q-chart (default) ≠ u-chart (alt) under non-trivial ψ
  T5  §12.3      Smoothness q-chart with IdentityChart equals q-chart with
                 unwrapped manifold (v4 backward compat)
  T6  §10.2      Reference-distribution init covariance: q_h ~ N(u^init,
                 τ_brown(K) · G_Q^A^{-1}(u^init))
  T7  §13        After full reverse GRW, viol(τ) == 0 by construction

Tests use a synthetic _MockPoseManifold (no URDF) so they run on CPU.
"""
from __future__ import annotations

import torch

from smcdp.charts import TanhBoundedChart, IdentityChart
from smcdp.manifolds_pose import BoundedChartPoseManifold
from smcdp.trajectories_pose import (
    PoseLangevinSDE,
    _dsm_chart_target_pose,
    _smoothness_guidance_chart,
    traj_forward_grw_pose,
    traj_reverse_grw_pose,
)
from tests.test_bounded_chart_manifold import _MockPoseManifold


class _LinearBetaSchedule:
    """Minimal β-schedule mock: β(r) = β_0 + r(β_f - β_0).

    integral(r) = β_0 r + ½ (β_f - β_0) r².  proxy_std(t, mode) supports
    'brownian' (√I) and 'ou' (√(1 - e^{-I})).
    """

    def __init__(self, beta_0=1e-3, beta_f=4.0, tf=1.0):
        self.beta_0 = float(beta_0)
        self.beta_f = float(beta_f)
        self.tf = float(tf)
        self.t0 = 0.0

    def beta(self, r):
        return self.beta_0 + (self.beta_f - self.beta_0) * r

    def integral(self, r):
        return self.beta_0 * r + 0.5 * (self.beta_f - self.beta_0) * (r ** 2)

    def proxy_std(self, t, mode="brownian"):
        I = self.integral(t)
        if mode == "brownian":
            return torch.sqrt(I.clamp(min=1e-12))
        return torch.sqrt(1.0 - torch.exp(-I))


def _make_setup(n_q=5, dtype=torch.float64, bounded=True):
    """Build a base manifold + (optionally bounded) wrapper + SDE."""
    base = _MockPoseManifold(n_q=n_q)
    if bounded:
        q_min, q_max = base.joint_limits(dtype=dtype)
        chart = TanhBoundedChart(q_min, q_max)
        manifold = BoundedChartPoseManifold(base, chart, lambda_floor=0.0)
    else:
        manifold = base                                         # v4 unwrapped
    schedule = _LinearBetaSchedule(beta_0=1e-3, beta_f=4.0, tf=1.0)
    sde = PoseLangevinSDE(
        manifold=manifold, schedule=schedule,
        limiting_q_mean=None,                                   # Method A: no fixed anchor
        limiting_scale=None,                                    # auto-calibrate to √τ_brown(K)
        forward_langevin_drift=False,
        confining_kappa=0.0,
    )
    return base, manifold, sde


# ---------------------------------------------------------------------------
# T1  §9   a*_Q matches the Choice A weighted least-squares formula
# ---------------------------------------------------------------------------


def test_T1_dsm_target_matches_spec_a_star_Q():
    base, manifold, _ = _make_setup(bounded=True)
    chart = manifold.chart
    torch.manual_seed(0)
    B = 3
    n_q = manifold.n_q
    u_0 = (torch.rand(B, n_q, dtype=torch.float64) - 0.5) * 2.0
    u_r = (torch.rand(B, n_q, dtype=torch.float64) - 0.5) * 2.0
    z = torch.rand(B, 1, dtype=torch.float64) * 0.2

    a_star = _dsm_chart_target_pose(manifold, u_r, u_0, z)

    # Manually rebuild the spec §9 formula.  The function applies a default
    # jitter=1e-4 inside G_pose_chol for numerical safety; we mirror it here
    # to verify the formula structure exactly (rather than the bare unregularized
    # solve).  This corresponds to the regularized G_Q^{A,reg} per §5.1 with
    # tikhonov_frac=0, lambda_floor=0, jitter=1e-4.
    from smcdp.lie_se3 import log_relative_Rp
    R_r, p_r = base.T_phi_Rp(chart.psi(u_r), z)
    R_0, p_0 = base.T_phi_Rp(chart.psi(u_0), z)
    xi = log_relative_Rp(R_r, p_r, R_0, p_0)                    # (B, 6)
    Jq = manifold.jacobian_pose(u_r, z)                          # (B, 6, n_q)  — already J^Q
    W = manifold._W_diag(u_r)                                    # (6,)
    rhs = (u_0 - u_r) + (Jq.transpose(-1, -2) @ (W.unsqueeze(-1) * xi.unsqueeze(-1))).squeeze(-1)
    eye = torch.eye(manifold.n_q, dtype=torch.float64).expand(B, manifold.n_q, manifold.n_q)
    G_reg = manifold.G_pose(u_r, z) + 1e-4 * eye                 # match default jitter
    a_expected = torch.linalg.solve(G_reg, rhs.unsqueeze(-1)).squeeze(-1)

    err = (a_star - a_expected).abs().max().item()
    assert err < 1e-10, f"a*_Q mismatch: {err}"
    print(f"T1  a*_Q matches spec §9 formula (with §5.1 regularization):  PASS  (max err {err:.3e})")


# ---------------------------------------------------------------------------
# T2  §7.1   Forward GRW preserves manifold adherence
# ---------------------------------------------------------------------------


def test_T2_forward_grw_keeps_manifold():
    base, manifold, sde = _make_setup(bounded=True)
    n_q = manifold.n_q
    n_z = manifold.n_z

    torch.manual_seed(1)
    B, H1 = 2, 4
    u_0 = (torch.rand(B, H1, n_q, dtype=torch.float64) - 0.5) * 1.5
    z_0 = torch.rand(B, H1, n_z, dtype=torch.float64) * 0.2

    # Make a valid x_0 ∈ M^Q
    tau_0 = manifold.make_x(u_0.reshape(-1, n_q), z_0.reshape(-1, n_z)).reshape(B, H1, manifold.ambient_dim)
    g0 = manifold.constraint(tau_0).abs().max().item()
    assert g0 < 1e-10, f"x_0 not on manifold: {g0}"

    r = torch.rand(B, dtype=torch.float64) * 0.5 + 0.1
    tau_r = traj_forward_grw_pose(sde, tau_0, r, n_steps=10)
    g_r = manifold.constraint(tau_r).abs().max().item()
    assert g_r < 1e-10, f"forward GRW broke manifold adherence: {g_r}"

    # Joint feasibility by construction — psi(u_r) ∈ (q_min, q_max)
    u_r = tau_r[..., :n_q]
    q_phys = manifold.physical_q(tau_r)
    q_lo, q_hi = manifold.joint_limits(dtype=torch.float64)
    assert (q_phys > q_lo).all() and (q_phys < q_hi).all()
    print(f"T2  forward GRW: manifold adherence + joint feasibility:  PASS  (|g| {g_r:.3e})")


# ---------------------------------------------------------------------------
# T3  §10  Reverse GRW retraction enforces feasibility throughout
# ---------------------------------------------------------------------------


def test_T3_reverse_grw_stays_feasible():
    base, manifold, sde = _make_setup(bounded=True)
    n_q = manifold.n_q
    n_z = manifold.n_z

    # Trivial score (identity) and goal_cond (None).
    def trivial_score(tau, t, goal_cond=None):
        return torch.zeros(tau.shape[0], tau.shape[1], n_q, device=tau.device, dtype=tau.dtype)

    torch.manual_seed(2)
    B = 4
    z_e = torch.rand(B, n_z, dtype=torch.float64) * 0.2
    u_init = (torch.rand(B, n_q, dtype=torch.float64) - 0.5) * 1.0           # interior

    tau_out = traj_reverse_grw_pose(
        sde, trivial_score, n_samples=B, H=3, n_steps=5,
        goal_cond=None, z_e=z_e,
        q_init=u_init,                                                       # u-chart anchor when bounded
        device=u_init.device, dtype=torch.float64,
    )

    # 1) Manifold adherence
    g_out = manifold.constraint(tau_out).abs().max().item()
    assert g_out < 1e-10, f"reverse GRW broke manifold adherence: {g_out}"

    # 2) Joint feasibility (spec §13: viol = 0 by construction)
    q_phys = manifold.physical_q(tau_out)
    q_lo, q_hi = manifold.joint_limits(dtype=torch.float64)
    assert (q_phys > q_lo).all() and (q_phys < q_hi).all()

    # 3) violates_limits == 0 (chart-level wrapper sanity check)
    u_out = tau_out[..., :n_q]
    viol = manifold.violates_limits(u_out)
    assert not viol.any().item()

    print(f"T3  reverse GRW: feasibility by construction:  PASS  (|g| {g_out:.3e})")


# ---------------------------------------------------------------------------
# T4  §12.3  q-chart smoothness ≠ u-chart smoothness under non-trivial ψ
# ---------------------------------------------------------------------------


def test_T4_smoothness_q_vs_u_chart_distinct():
    base, manifold, _ = _make_setup(bounded=True)
    n_q = manifold.n_q
    n_z = manifold.n_z

    torch.manual_seed(3)
    B, H1 = 2, 5
    # Place trajectory near boundary so D_ψ varies (else q ≈ u and smoothness equal)
    u = (torch.randn(B, H1, n_q, dtype=torch.float64)) * 1.5 + 1.0           # offset toward +u
    z = torch.rand(B, H1, n_z, dtype=torch.float64) * 0.2
    tau = manifold.make_x(u.reshape(-1, n_q), z.reshape(-1, n_z)).reshape(B, H1, manifold.ambient_dim)

    g_q = _smoothness_guidance_chart(tau, manifold, alpha_vel=1.0, alpha_acc=0.5, chart_form="q")
    g_u = _smoothness_guidance_chart(tau, manifold, alpha_vel=1.0, alpha_acc=0.5, chart_form="u")
    diff = (g_q - g_u).abs().max().item()
    assert diff > 0.01, (
        f"q-chart and u-chart smoothness should differ noticeably under non-trivial ψ; "
        f"max diff = {diff}"
    )
    print(f"T4  q-chart smoothness ≠ u-chart smoothness:  PASS  (max diff {diff:.3e})")


# ---------------------------------------------------------------------------
# T5  §12.3  q-chart smoothness with IdentityChart matches v4 (unwrapped)
# ---------------------------------------------------------------------------


def test_T5_smoothness_identity_chart_matches_v4():
    """With ψ = identity (IdentityChart wrapper or unwrapped), q-chart and
    u-chart smoothness coincide, and both match v4 closed-form behavior."""
    base = _MockPoseManifold(n_q=5)
    manifold_id = BoundedChartPoseManifold(base, IdentityChart(n_q=5), lambda_floor=0.0)

    torch.manual_seed(4)
    B, H1 = 2, 5
    n_q, n_z = 5, 1
    chart_slot = torch.randn(B, H1, n_q, dtype=torch.float64) * 0.5
    z = torch.rand(B, H1, n_z, dtype=torch.float64) * 0.2
    tau_id = manifold_id.make_x(
        chart_slot.reshape(-1, n_q), z.reshape(-1, n_z)
    ).reshape(B, H1, manifold_id.ambient_dim)

    g_q = _smoothness_guidance_chart(tau_id, manifold_id, alpha_vel=1.0, alpha_acc=0.3, chart_form="q")
    g_u = _smoothness_guidance_chart(tau_id, manifold_id, alpha_vel=1.0, alpha_acc=0.3, chart_form="u")
    err = (g_q - g_u).abs().max().item()
    assert err < 1e-12, f"With IdentityChart, q-chart and u-chart should coincide; got diff {err}"

    # Also check against unwrapped base manifold (v4 path)
    tau_base = base.make_x(chart_slot.reshape(-1, n_q), z.reshape(-1, n_z)).reshape(
        B, H1, base.ambient_dim
    )
    g_base = _smoothness_guidance_chart(tau_base, base, alpha_vel=1.0, alpha_acc=0.3, chart_form="q")
    err_base = (g_q - g_base).abs().max().item()
    assert err_base < 1e-12, f"IdentityChart wrapper should match unwrapped: {err_base}"
    print(f"T5  IdentityChart smoothness ≡ v4 unwrapped:  PASS  (err {err:.3e}, vs base {err_base:.3e})")


# ---------------------------------------------------------------------------
# T6  §10.2  Reference distribution covariance matches spec
# ---------------------------------------------------------------------------


def test_T6_reference_distribution_covariance():
    """After reverse GRW init (k=0), q_h ~ N(u^init, σ_K² G_Q^A(u^init)^{-1})."""
    base, manifold, sde = _make_setup(bounded=True)
    n_q = manifold.n_q
    n_z = manifold.n_z

    # Use a small batch of identical u_init, monte-carlo estimate the empirical
    # covariance, compare to σ_K² G_Q^A^{-1}(u_init).
    torch.manual_seed(5)
    u_init_value = torch.zeros(n_q, dtype=torch.float64)                     # at chart center
    z_e_value = torch.tensor([0.1], dtype=torch.float64)
    N = 4000

    # Manually replicate the reverse GRW init step (no model, just init)
    u_init_b = u_init_value.unsqueeze(0).expand(N, -1).contiguous()
    z_e_b = z_e_value.unsqueeze(0).expand(N, -1).contiguous()
    H1 = 1
    z_traj = z_e_b.unsqueeze(1).expand(N, H1, n_z).contiguous()
    z_flat = z_traj.reshape(N * H1, n_z)
    anchor_q_flat = u_init_b.unsqueeze(1).expand(N, H1, n_q).contiguous().reshape(N * H1, n_q)

    L_anchor = manifold.G_pose_chol(anchor_q_flat, z_flat)
    eps_init = torch.randn(N, H1, n_q, dtype=torch.float64)
    a_init = torch.linalg.solve_triangular(
        L_anchor.transpose(-1, -2), eps_init.reshape(N * H1, n_q, 1), upper=True,
    ).squeeze(-1)

    sigma_K = float((sde.schedule.integral(torch.tensor(sde.schedule.tf, dtype=torch.float64))
                     ).clamp(min=1e-12).sqrt().item())
    u_K = anchor_q_flat + sigma_K * a_init                                   # (N, n_q)

    # Empirical covariance (N samples)
    u_diff = u_K - u_init_value.unsqueeze(0)
    emp_cov = (u_diff.unsqueeze(-1) @ u_diff.unsqueeze(-2)).mean(0)          # (n_q, n_q)

    # Expected covariance: σ_K² G_Q^A^{-1}(u_init)
    G = manifold.G_pose(u_init_value.unsqueeze(0), z_e_value.unsqueeze(0)).squeeze(0)
    G_inv = torch.linalg.inv(G)
    expected_cov = sigma_K ** 2 * G_inv

    rel_err = (emp_cov - expected_cov).abs().max().item() / expected_cov.abs().max().item()
    # Monte Carlo noise floor with N=4000 is around 3% per matrix entry
    assert rel_err < 0.10, f"reference distribution cov mismatch: {rel_err:.3f}"
    print(f"T6  reference dist N(u_init, σ_K² G_Q^A^{{-1}}):  PASS  (rel err {rel_err:.3%})")


# ---------------------------------------------------------------------------
# T7  §13  After reverse GRW, viol(τ) = 0 by construction
# ---------------------------------------------------------------------------


def test_T7_post_sampling_viol_zero():
    base, manifold, sde = _make_setup(bounded=True)
    n_q = manifold.n_q
    n_z = manifold.n_z

    def random_score(tau, t, goal_cond=None):
        # large random score to stress the sampler
        return torch.randn(tau.shape[0], tau.shape[1], n_q,
                           device=tau.device, dtype=tau.dtype) * 5.0

    torch.manual_seed(6)
    B = 8
    z_e = torch.rand(B, n_z, dtype=torch.float64) * 0.2
    u_init = (torch.rand(B, n_q, dtype=torch.float64) - 0.5) * 1.0

    tau_out = traj_reverse_grw_pose(
        sde, random_score, n_samples=B, H=3, n_steps=20,
        goal_cond=None, z_e=z_e, q_init=u_init,
        device=u_init.device, dtype=torch.float64,
    )

    q_phys = manifold.physical_q(tau_out)
    q_lo, q_hi = manifold.joint_limits(dtype=torch.float64)
    n_viol = ((q_phys < q_lo).any(-1) | (q_phys > q_hi).any(-1)).sum().item()
    assert n_viol == 0, f"viol(τ) should be 0 by construction, got {n_viol}"
    print(f"T7  post-sampling viol = 0 by construction (random score, |s| ~ 5):  PASS")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Trajectories under bounded chart (joint_limit_extension v4.1, Phase 4) ===\n")
    test_T1_dsm_target_matches_spec_a_star_Q()
    test_T2_forward_grw_keeps_manifold()
    test_T3_reverse_grw_stays_feasible()
    test_T4_smoothness_q_vs_u_chart_distinct()
    test_T5_smoothness_identity_chart_matches_v4()
    test_T6_reference_distribution_covariance()
    test_T7_post_sampling_viol_zero()
    print("\n=== all tests passed ===")
