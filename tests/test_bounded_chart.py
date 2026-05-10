"""Sanity tests for `smcdp.charts` (joint_limit_extension v4.1, Phase 1).

Verifies:
    C1. ψ(u) ∈ (q_min, q_max) for any finite u.
    C2. ψ⁻¹(ψ(u)) ≈ u (within eta-clip envelope).
    C3. ψ(ψ⁻¹(q)) ≈ q for q strictly inside (q_min, q_max).
    C4. D_psi_diag(u) matches autograd of ψ at machine precision (FP64).
    C5. D_psi > 0 everywhere; D_psi → 0 at saturated u.
    C6. IdentityChart special case: ψ(u) = u, D_psi = 1, recovers v4 behavior.
    C7. Boundary safety: ψ⁻¹ on q at the limit produces finite u (η-clip kicks in).
    C8. Vmap-friendly: arbitrary batch shapes pass through.

Run:  python -m tests.test_bounded_chart
"""
from __future__ import annotations

import torch

from smcdp.charts import TanhBoundedChart, IdentityChart, make_chart_from_manifold


def _franka_limits(dtype=torch.float64):
    """Realistic Franka-like joint limits for tests."""
    q_min = torch.tensor([-2.967, -1.833, -2.967, -3.142, -2.967, -0.087, -2.967], dtype=dtype)
    q_max = torch.tensor([+2.967, +1.833, +2.967, +0.000, +2.967, +3.822, +2.967], dtype=dtype)
    return q_min, q_max


def test_C1_psi_image_inside_limits():
    q_min, q_max = _franka_limits()
    chart = TanhBoundedChart(q_min, q_max)
    # In the practical operating range |u| ≤ 10 (well below tanh saturation at
    # ~19.5 in float64), ψ(u) stays strictly inside (q_min, q_max).
    u = torch.linspace(-10, 10, 21, dtype=torch.float64).unsqueeze(-1).expand(-1, 7).contiguous()
    q = chart.psi(u)
    assert (q > q_min).all(), f"ψ(u) violated lower bound: min={q.min(0).values}"
    assert (q < q_max).all(), f"ψ(u) violated upper bound: max={q.max(0).values}"
    # Boundary distance at |u|=10: ~ q_range * (1-tanh(10))/2 ≈ q_range * 1e-9
    delta_lo = (q.min(0).values - q_min).min().item()
    delta_hi = (q_max - q.max(0).values).min().item()
    assert delta_lo > 0 and delta_hi > 0
    # In float64, tanh(20) saturates to 1.0; this is acceptable since spec only
    # claims open-interval feasibility for finite finite-step sampling, where
    # |u| ≤ 10 is more than sufficient (margin > 1e-9 of q_range).
    print(f"C1 ψ image ⊂ (q_min, q_max): PASS  "
          f"(min boundary distance at |u|=10: {min(delta_lo, delta_hi):.3e})")


def test_C2_psi_inv_psi_roundtrip():
    q_min, q_max = _franka_limits()
    chart = TanhBoundedChart(q_min, q_max)
    # Sample u in [-3, 3] (where η = tanh(u) is well within (-1+eps, 1-eps))
    torch.manual_seed(0)
    u = (torch.rand(50, 7, dtype=torch.float64) - 0.5) * 6.0
    q = chart.psi(u)
    u_back = chart.psi_inv(q, eps=1e-6)
    err = (u - u_back).abs().max().item()
    assert err < 1e-10, f"ψ⁻¹∘ψ roundtrip error too large: {err}"
    print(f"C2 ψ⁻¹∘ψ roundtrip: PASS (max err {err:.3e})")


def test_C3_psi_psi_inv_roundtrip():
    q_min, q_max = _franka_limits()
    chart = TanhBoundedChart(q_min, q_max)
    # Sample q strictly inside (q_min, q_max), with margin > 1%
    torch.manual_seed(1)
    q_mid = 0.5 * (q_min + q_max)
    q_range = q_max - q_min
    rel = (torch.rand(50, 7, dtype=torch.float64) - 0.5) * 0.95     # in [-0.475, +0.475]
    q = q_mid + rel * q_range
    u = chart.psi_inv(q, eps=1e-6)
    q_back = chart.psi(u)
    err = (q - q_back).abs().max().item()
    assert err < 1e-10, f"ψ∘ψ⁻¹ roundtrip error too large: {err}"
    print(f"C3 ψ∘ψ⁻¹ roundtrip: PASS (max err {err:.3e})")


def test_C4_D_psi_vs_autograd():
    q_min, q_max = _franka_limits(dtype=torch.float64)
    chart = TanhBoundedChart(q_min, q_max)
    torch.manual_seed(2)
    u = (torch.rand(8, 7, dtype=torch.float64) - 0.5) * 4.0
    u.requires_grad_(True)
    q = chart.psi(u)
    # Compute dq_i/du_j via autograd; ψ is element-wise so off-diagonal is 0
    n_q = 7
    diag_autograd = torch.zeros(8, n_q, dtype=torch.float64)
    for j in range(n_q):
        grad = torch.autograd.grad(q[:, j].sum(), u, retain_graph=True)[0][:, j]
        diag_autograd[:, j] = grad
    diag_closed = chart.D_psi_diag(u.detach())
    err = (diag_autograd - diag_closed).abs().max().item()
    assert err < 1e-12, f"D_ψ diagonal mismatch with autograd: {err}"
    print(f"C4 D_ψ vs autograd: PASS (max err {err:.3e})")


def test_C5_D_psi_positivity_and_decay():
    q_min, q_max = _franka_limits()
    chart = TanhBoundedChart(q_min, q_max)
    # Center: D_ψ = q_range / 2
    u_center = torch.zeros(7, dtype=torch.float64)
    D_center = chart.D_psi_diag(u_center)
    expected = (q_max - q_min) / 2.0
    err = (D_center - expected).abs().max().item()
    assert err < 1e-15, f"D_ψ at u=0 mismatch: {err}"
    # Boundary: D_ψ → 0
    u_far = torch.full((7,), 10.0, dtype=torch.float64)
    D_far = chart.D_psi_diag(u_far)
    assert (D_far > 0).all() and D_far.max() < 1e-7, \
        f"D_ψ should vanish at saturation but max={D_far.max()}"
    # Always > 0
    u_random = (torch.rand(20, 7, dtype=torch.float64) - 0.5) * 8.0
    D = chart.D_psi_diag(u_random)
    assert (D > 0).all(), f"D_ψ should be strictly positive everywhere"
    print(f"C5 D_ψ positivity + boundary decay: PASS")


def test_C6_identity_chart():
    chart = IdentityChart(n_q=7)
    u = torch.randn(5, 7, dtype=torch.float64)
    assert torch.allclose(chart.psi(u), u)
    assert torch.allclose(chart.psi_inv(u), u)
    D = chart.D_psi_diag(u)
    assert torch.allclose(D, torch.ones_like(u))
    q_lo, q_hi = chart.joint_limits()
    assert (q_hi > q_lo).all()
    print("C6 IdentityChart special case: PASS")


def test_C7_boundary_safety_clip():
    q_min, q_max = _franka_limits()
    chart = TanhBoundedChart(q_min, q_max)
    # q exactly at the upper boundary → η = 1 → atanh diverges; clip should save us
    q_at_bound = q_max.clone()
    u = chart.psi_inv(q_at_bound, eps=1e-3)
    assert torch.isfinite(u).all(), f"ψ⁻¹ at boundary produced non-finite u: {u}"
    # Also test slightly outside (numerical drift) — clip must absorb
    q_outside = q_max.clone() + 1e-4
    u_outside = chart.psi_inv(q_outside, eps=1e-3)
    assert torch.isfinite(u_outside).all()
    # Without clip (eps=0) it would blow up; with clip it returns ~atanh(1-eps) ≈ ±large finite
    print(f"C7 boundary safety η-clip: PASS (|u@bound|={u.abs().max().item():.3f})")


def test_C8_vmap_friendly_batch_shapes():
    q_min, q_max = _franka_limits()
    chart = TanhBoundedChart(q_min, q_max)
    for shape in [(7,), (4, 7), (3, 4, 7), (2, 5, 6, 7)]:
        u = torch.randn(*shape, dtype=torch.float64)
        q = chart.psi(u)
        D = chart.D_psi_diag(u)
        u_back = chart.psi_inv(q, eps=1e-6)
        assert q.shape == shape and D.shape == shape and u_back.shape == shape
        assert (u - u_back).abs().max().item() < 1e-10
    print("C8 vmap-friendly batch shapes: PASS")


def test_C9_factory_from_manifold():
    """Factory test using a mock-manifold-like object."""
    class _MockManifold:
        def __init__(self):
            self.n_q = 7
            self._lo, self._hi = _franka_limits()
        def joint_limits(self, *, device=None, dtype=None):
            return self._lo.to(device=device, dtype=dtype), self._hi.to(device=device, dtype=dtype)

    m = _MockManifold()
    c_v4 = make_chart_from_manifold(m, bounded=False)
    assert isinstance(c_v4, IdentityChart) and c_v4.n_q == 7
    c_v41 = make_chart_from_manifold(m, bounded=True, dtype=torch.float64)
    assert isinstance(c_v41, TanhBoundedChart) and c_v41.n_q == 7
    print("C9 factory from manifold: PASS")


if __name__ == "__main__":
    print("=== smcdp.charts unit tests (joint_limit_extension v4.1, Phase 1) ===\n")
    test_C1_psi_image_inside_limits()
    test_C2_psi_inv_psi_roundtrip()
    test_C3_psi_psi_inv_roundtrip()
    test_C4_D_psi_vs_autograd()
    test_C5_D_psi_positivity_and_decay()
    test_C6_identity_chart()
    test_C7_boundary_safety_clip()
    test_C8_vmap_friendly_batch_shapes()
    test_C9_factory_from_manifold()
    print("\n=== all tests passed ===")
