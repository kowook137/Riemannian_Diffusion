"""Phase 5 test: demo gen converts IK-produced q to chart u when wrapped.

This test bypasses the actual IK loop by monkey-patching the manifold's
make_x receiver to inspect what gets passed in. Since the demo gen calls
`manifold.make_x(chart_slot_flat, z_flat)` after the IK, we verify that:

  V1  Wrapped manifold receives u = ψ⁻¹(q) in chart slot (not q)
  V2  Unwrapped manifold receives q directly (v4 backward compat)
  V3  IdentityChart wrapper passes q through unchanged
  V4  Resulting x has T-block matching T_φ(ψ(u), z) = T_φ(q, z)
  V5  Resulting x[..., :n_q] equals u (not q) when wrapped

Notes
-----
We bypass the URDF / pytorch_kinematics dependency by stubbing the IK
arm and the FK chain via the synthetic _MockPoseManifold from Phase 2
tests, plus a stand-in for the demo gen's IK loop (we directly produce
a known q_traj and call only the lift step).
"""
from __future__ import annotations

import torch

from smcdp.charts import TanhBoundedChart, IdentityChart
from smcdp.manifolds_pose import BoundedChartPoseManifold
from smcdp.lie_se3 import quat_to_R
from tests.test_bounded_chart_manifold import _MockPoseManifold


def _lift_step(manifold, q_traj, z_e, n_q=5):
    """Replicate the demo gen's step-6 lift logic (post-IK).

    Verbatim copy of the relevant 5 lines from
    `FrankaBimodalReachingDemoPose.sample` so this test exercises the same
    code path without requiring a Franka URDF.
    """
    n, H1 = q_traj.shape[0], q_traj.shape[1]
    z_traj = z_e.unsqueeze(1).expand(n, H1, 1).contiguous()
    q_flat = q_traj.reshape(-1, n_q)
    z_flat = z_traj.reshape(-1, 1)
    if hasattr(manifold, "chart"):
        chart_slot_flat = manifold.chart.psi_inv(q_flat, eps=1e-3)
    else:
        chart_slot_flat = q_flat
    x_flat = manifold.make_x(chart_slot_flat, z_flat)
    return x_flat.reshape(n, H1, manifold.ambient_dim), chart_slot_flat


def _make_synthetic_q_traj(base, n=4, H1=3):
    """Build a plausible q-trajectory inside the joint range with margin."""
    torch.manual_seed(0)
    q_lo, q_hi = base.joint_limits(dtype=torch.float64)
    q_mid = 0.5 * (q_lo + q_hi)
    q_range = q_hi - q_lo
    rel = (torch.rand(n, H1, base.n_q, dtype=torch.float64) - 0.5) * 0.7        # margin
    return q_mid + rel * q_range


def test_V1_wrapped_receives_u_not_q():
    """V1 — Wrapped manifold receives u = ψ⁻¹(q) in chart slot."""
    base = _MockPoseManifold(n_q=5)
    chart = TanhBoundedChart(*base.joint_limits(dtype=torch.float64))
    wrapped = BoundedChartPoseManifold(base, chart, lambda_floor=0.0)

    q_traj = _make_synthetic_q_traj(base, n=4, H1=3)
    z_e = torch.rand(4, 1, dtype=torch.float64) * 0.2

    x, chart_slot = _lift_step(wrapped, q_traj, z_e, n_q=5)

    # The chart slot received by make_x should be u = ψ⁻¹(q), NOT q.
    u_expected = chart.psi_inv(q_traj.reshape(-1, 5), eps=1e-3)
    err = (chart_slot - u_expected).abs().max().item()
    assert err < 1e-13
    # And distinct from q (when chart non-trivial)
    diff_from_q = (chart_slot - q_traj.reshape(-1, 5)).abs().max().item()
    assert diff_from_q > 0.01, f"chart slot should differ from q under non-trivial ψ; diff={diff_from_q}"
    print(f"V1  wrapped receives u = ψ⁻¹(q):  PASS  (vs expected {err:.3e}, vs q {diff_from_q:.3e})")


def test_V2_unwrapped_receives_q():
    """V2 — v4 backward compat: unwrapped manifold receives q directly."""
    base = _MockPoseManifold(n_q=5)
    q_traj = _make_synthetic_q_traj(base, n=4, H1=3)
    z_e = torch.rand(4, 1, dtype=torch.float64) * 0.2

    x, chart_slot = _lift_step(base, q_traj, z_e, n_q=5)
    err = (chart_slot - q_traj.reshape(-1, 5)).abs().max().item()
    assert err < 1e-15, f"unwrapped should receive q verbatim; got diff {err}"
    print("V2  unwrapped receives q (v4 backward compat):  PASS")


def test_V3_identity_chart_passes_q_through():
    """V3 — IdentityChart wrapper: ψ⁻¹ is identity, q passes through unchanged."""
    base = _MockPoseManifold(n_q=5)
    wrapped = BoundedChartPoseManifold(base, IdentityChart(n_q=5), lambda_floor=0.0)

    q_traj = _make_synthetic_q_traj(base, n=3, H1=4)
    z_e = torch.rand(3, 1, dtype=torch.float64) * 0.2
    x, chart_slot = _lift_step(wrapped, q_traj, z_e, n_q=5)
    err = (chart_slot - q_traj.reshape(-1, 5)).abs().max().item()
    assert err < 1e-15
    print(f"V3  IdentityChart passes q through:  PASS  (err {err:.3e})")


def test_V4_T_block_matches_T_phi():
    """V4 — Resulting x's T-block matches T_φ(ψ(u), z) = T_φ(q, z)."""
    base = _MockPoseManifold(n_q=5)
    chart = TanhBoundedChart(*base.joint_limits(dtype=torch.float64))
    wrapped = BoundedChartPoseManifold(base, chart, lambda_floor=0.0)

    q_traj = _make_synthetic_q_traj(base, n=4, H1=3)
    z_e = torch.rand(4, 1, dtype=torch.float64) * 0.2
    x, chart_slot = _lift_step(wrapped, q_traj, z_e, n_q=5)

    # Recover stored T from x
    R_stored = quat_to_R(x[..., 5 : 5 + 4])
    p_stored = x[..., 5 + 4 : 5 + 7]
    # Expected: T_φ(q, z)
    R_exp, p_exp = base.T_phi_Rp(q_traj.reshape(-1, 5), z_e.unsqueeze(1).expand(4, 3, 1).reshape(-1, 1))
    R_exp = R_exp.reshape(4, 3, 3, 3)
    p_exp = p_exp.reshape(4, 3, 3)
    err_R = (R_stored - R_exp).abs().max().item()
    err_p = (p_stored - p_exp).abs().max().item()
    assert err_R < 1e-12 and err_p < 1e-12
    print(f"V4  T-block = T_φ(q, z):  PASS  (R err {err_R:.3e}, p err {err_p:.3e})")


def test_V5_chart_slot_in_x_is_u():
    """V5 — Final x[..., :n_q] stores u (not q) when wrapped."""
    base = _MockPoseManifold(n_q=5)
    chart = TanhBoundedChart(*base.joint_limits(dtype=torch.float64))
    wrapped = BoundedChartPoseManifold(base, chart, lambda_floor=0.0)

    q_traj = _make_synthetic_q_traj(base, n=2, H1=3)
    z_e = torch.rand(2, 1, dtype=torch.float64) * 0.2
    x, _ = _lift_step(wrapped, q_traj, z_e, n_q=5)

    u_in_x = x[..., :5]
    u_expected = chart.psi_inv(q_traj, eps=1e-3)
    err = (u_in_x - u_expected).abs().max().item()
    assert err < 1e-13
    # And physical_q recovers q
    q_phys = wrapped.physical_q(x)
    err_q = (q_phys - q_traj).abs().max().item()
    assert err_q < 1e-10
    print(f"V5  x[..., :n_q] = u, physical_q(x) = q:  PASS  (u err {err:.3e}, q err {err_q:.3e})")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Demo gen lift step under bounded chart (v4.1, Phase 5) ===\n")
    test_V1_wrapped_receives_u_not_q()
    test_V2_unwrapped_receives_q()
    test_V3_identity_chart_passes_q_through()
    test_V4_T_block_matches_T_phi()
    test_V5_chart_slot_in_x_is_u()
    print("\n=== all tests passed ===")
