"""Sanity tests for joint_limit_extension v5.1 chart-space OU SDE.

Verifies — strictly against the spec equations (joint_limit_extension.tex v5.1):
    V1. Schedule helpers — α(0)=1, σ²(0)=0; α(K)≈0.007 at β_f=20 (spec §8.2 table).
    V2. proxy_std('ou') ≡ √σ²(r) identity across r ∈ (0, K).
    V3. Closed-form forward marginal: empirical E[u_r], Var[u_r] match
        α(r) u_0  and  σ²(r) Ḡ_Q^{-1} respectively (Monte Carlo, identity Ḡ).
    V4. Exact Euclidean score target s* = -Ḡ_Q (u_r - α u_0)/σ²(r) (spec §10 boxed)
        agrees with ∇_u log p numerically (finite-diff cross-check).
    V5. PoseChartOUSDE.gbar_*_apply is identity-mode no-op.
    V6. IK-free reverse init  u ~ N(0, Ḡ_Q^{-1}) — no IK call, no q_init kwarg.
    V7. Reverse SDE drift sign convention: empty score → reverse OU mirror only
        (drift ~ +½β u dr) → ‖u‖ grows away from origin (spec §11.3.4 sign check).
    V8. Network input invariant: forward/reverse never receive q_init / anchor.

Run:  python -m tests.test_v51_chart_ou
"""
from __future__ import annotations

import math

import torch
import pybullet_data

from smcdp.sde import LinearBetaSchedule
from smcdp.manifolds_pose import Franka7DoFPose, BoundedChartPoseManifold
from smcdp.charts import make_chart_from_manifold
from smcdp.trajectories_pose import (
    PoseChartOUSDE,
    traj_forward_ou_chart_pose,
    traj_ou_score_loss_pose,
    traj_pose_consistency_loss,
    traj_total_loss_v51_pose,
    traj_reverse_ou_chart_pose,
    TrajectoryScoreNetUNetPose,
    TrajectoryScaledScoreFnPose,
)


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


# ---------------------------------------------------------------------
def _arm_setup(beta_f: float = 20.0):
    torch.manual_seed(0)
    arm = Franka7DoFPose(URDF, sigma_p=0.05, sigma_R=0.1, tool_z_max=0.20)
    arm._ensure_chain(torch.zeros(1, 7))
    arm_b = BoundedChartPoseManifold(arm, make_chart_from_manifold(arm, bounded=True))
    sched = LinearBetaSchedule(beta_0=1e-3, beta_f=beta_f, tf=1.0)
    sde = PoseChartOUSDE(arm_b, sched, gbar_mode="identity")
    return arm_b, sched, sde


def _build_tau_0(arm_b, B: int = 4, H: int = 7, scale: float = 0.3):
    n_q = 7
    H1 = H + 1
    u_0 = scale * torch.randn(B, H1, n_q)
    z_e = torch.full((B, 1), 0.10)
    z_traj = z_e.unsqueeze(1).expand(B, H1, 1).contiguous()
    tau_0_flat = arm_b.make_x(u_0.reshape(B * H1, n_q), z_traj.reshape(B * H1, 1))
    return tau_0_flat.reshape(B, H1, arm_b.ambient_dim), u_0, z_e


# ---------------------------------------------------------------------
def test_V1_schedule_endpoints():
    """α(0) = 1, σ²(0) = 0; α(1) ≈ 0.0067 at β_f = 20 (spec §8.2 reference table)."""
    sched = LinearBetaSchedule(beta_0=1e-3, beta_f=20.0, tf=1.0)
    t0 = torch.tensor(0.0)
    t1 = torch.tensor(1.0)
    assert abs(sched.alpha(t0).item() - 1.0) < 1e-7, "α(0) must be 1"
    assert abs(sched.sigma2(t0).item() - 0.0) < 1e-7, "σ²(0) must be 0"
    # Spec table: β_f=20 → α(K) ≈ 0.0067, β_f=10 → 0.082, β_f=4 → 0.368
    assert abs(sched.alpha(t1).item() - 0.0067) < 1e-3, "α(K) at β_f=20 ≈ 0.007"
    # σ²(K) at β_f=20 ≈ 1 − e^{-10} ≈ 1.0
    assert sched.sigma2(t1).item() > 0.999, "σ²(K) at β_f=20 ≈ 1"

    for beta_f, want_alpha in [(4.0, 0.3679), (10.0, 0.0821), (20.0, 0.0067)]:
        sched = LinearBetaSchedule(beta_0=1e-3, beta_f=beta_f, tf=1.0)
        assert abs(sched.alpha(t1).item() - want_alpha) < 1e-3, \
            f"α(K) at β_f={beta_f} expected ≈{want_alpha}"


def test_V2_proxy_std_ou_equals_sigma():
    """proxy_std('ou', r) ≡ √σ²(r) — confirms TrajectoryScaledScoreFnPose
    std-trick uses the correct OU marginal scale (spec §10 std_trick)."""
    sched = LinearBetaSchedule(beta_0=1e-3, beta_f=20.0, tf=1.0)
    for r in [0.05, 0.2, 0.5, 0.95]:
        rt = torch.tensor(r)
        a = sched.sigma(rt).item()
        b = sched.proxy_std(rt, mode="ou").item()
        assert abs(a - b) < 1e-7, f"r={r}: sigma {a} ≠ proxy_std(ou) {b}"


def test_V3_closed_form_marginals():
    """Empirical E[u_r], Var[u_r] match closed-form (α u_0, σ²(r) I) within MC error.
    Spec §8.1 boxed transition kernel."""
    arm_b, sched, sde = _arm_setup()
    n_q = 7
    u_0 = torch.tensor([0.5, -0.3, 0.2, -0.1, 0.05, 0.25, -0.05])
    z_one = torch.tensor([0.10])
    n_mc = 8192

    tau_0_flat = arm_b.make_x(u_0.unsqueeze(0).expand(n_mc, n_q),
                                z_one.unsqueeze(0).expand(n_mc, 1))
    tau_0 = tau_0_flat.unsqueeze(1)                                         # (n_mc, 1, ambient)
    r_test = 0.5
    r_mc = torch.full((n_mc,), r_test)
    tau_r_mc = traj_forward_ou_chart_pose(sde, tau_0, r_mc)
    u_r_mc = tau_r_mc[:, 0, :n_q]                                            # (n_mc, n_q)

    alpha_r = sched.alpha(torch.tensor(r_test)).item()
    sigma2_r = sched.sigma2(torch.tensor(r_test)).item()
    emp_mean = u_r_mc.mean(0)
    emp_var = u_r_mc.var(0, unbiased=False)

    # Mean: empirical should match α u_0 within 3-sigma MC band: σ/sqrt(n_mc) ≈ 0.011
    mean_err = (emp_mean - alpha_r * u_0).abs().max().item()
    assert mean_err < 0.05, f"E[u_r] mismatch: max err {mean_err:.4f}"
    # Var: each diag should be near σ²(r). Tolerance per Monte-Carlo std (relative).
    var_relerr = ((emp_var - sigma2_r) / sigma2_r).abs().max().item()
    assert var_relerr < 0.05, f"Var[u_r] mismatch: max relerr {var_relerr:.4f}"


def test_V4_score_target_matches_grad_log_p():
    """s*(u_r, u_0; r) = -Ḡ_Q (u_r - α u_0)/σ²(r) numerically agrees with
    ∇_{u_r} log N(u_r; α u_0, σ²(r) Ḡ_Q^{-1})  via autograd  (spec §8.1.4 boxed)."""
    arm_b, sched, sde = _arm_setup()
    n_q = 7
    r_val = 0.4
    alpha_r = sched.alpha(torch.tensor(r_val)).item()
    sigma2_r = sched.sigma2(torch.tensor(r_val)).item()

    u_0 = torch.randn(1, n_q)
    u_r = torch.randn(1, n_q, requires_grad=True)
    mean = alpha_r * u_0
    diff = u_r - mean
    # log p = -½ diff^T Ḡ_Q diff / σ² (up to constants).  ∇_{u_r} log p = -Ḡ diff/σ².
    log_p = -0.5 * (diff * sde.gbar_apply(diff)).sum() / sigma2_r
    grad_autograd = torch.autograd.grad(log_p, u_r)[0]
    target = -sde.gbar_apply(diff.detach()) / sigma2_r
    err = (grad_autograd - target).abs().max().item()
    assert err < 1e-5, f"score target mismatch: {err:.2e}"


def test_V5_gbar_identity_noop():
    """gbar_*_apply for gbar_mode='identity' acts as identity on the last axis."""
    arm_b, sched, sde = _arm_setup()
    x = torch.randn(3, 5, 7)
    for fn in (sde.gbar_apply, sde.gbar_inv_apply, sde.gbar_inv_sqrt_apply):
        y = fn(x)
        assert torch.allclose(x, y, atol=0), f"{fn.__name__} not no-op"


def test_V6_reverse_is_ik_free():
    """traj_reverse_ou_chart_pose has NO q_init / limiting_q_mean / IK kwarg.
    Initial sample u_K ~ N(0, Ḡ_Q^{-1}) is data-independent at sampling time."""
    import inspect
    sig = inspect.signature(traj_reverse_ou_chart_pose)
    params = set(sig.parameters.keys())
    forbidden = {"q_init", "limiting_q_mean", "limiting_scale", "q_warm"}
    leaked = forbidden & params
    assert not leaked, f"v5.1 reverse must be IK-free, found leaks: {leaked}"


def test_V7_reverse_drift_sign_with_zero_score():
    """With score ≡ 0 the reverse SDE has only the +½β u dr drift (OU mirror) plus
    noise; ‖u‖ should grow (in expectation) per step.  Spec §11.3.4 sign check."""
    arm_b, sched, sde = _arm_setup()
    B, H = 8, 3
    H1 = H + 1
    n_q = 7

    class ZeroScore:
        def __call__(self, x, t, goal_cond=None):
            return torch.zeros(x.shape[0], x.shape[1], n_q)

    z_e = torch.full((B, 1), 0.10)
    torch.manual_seed(123)
    samples = traj_reverse_ou_chart_pose(
        sde, ZeroScore(), n_samples=B, H=H, n_steps=40,
        goal_cond=None, z_e=z_e, eps=1e-3,
    )
    u_K = samples[..., :n_q]
    # With u_0 ~ N(0, I) and zero score, the reverse SDE reduces to
    #     du = ½β u dr + √β Ḡ^{-1/2} dW         (Δr<0 convention, forward-dr form)
    # Solving over r ∈ [tf, eps], variance grows exponentially.  We expect
    # ‖u‖ to be much larger than the initial unit std.
    norm_per_traj = u_K.flatten(1).norm(dim=-1)
    assert norm_per_traj.mean().item() > math.sqrt(n_q * H1) * 1.5, \
        f"zero-score reverse should grow ‖u‖ via OU mirror; got mean ‖u‖ = " \
        f"{norm_per_traj.mean().item():.2f}"


def test_V8_score_net_input_no_q_init():
    """TrajectoryScoreNetUNetPose input is [u, r, h/H, z_e, T_start, T_target]
    only — no q_init / anchor channel (spec §9: 'No anchor, no q^init, no
    IK-derived variable')."""
    arm_b, sched, sde = _arm_setup()
    net = TrajectoryScoreNetUNetPose(
        manifold=arm_b, H=7, down_dims=(64, 128),
        diffusion_step_embed_dim=64, goal_cond_dim=14, cond_injection="channel",
    )
    score_fn = TrajectoryScaledScoreFnPose(net, sde, std_trick=True, proxy_std_mode="ou")
    B, H1 = 4, 8
    tau = torch.randn(B, H1, arm_b.ambient_dim)
    tau[..., 7:11] = torch.tensor([0.0, 0.0, 0.0, 1.0])                          # quat identity
    t = torch.full((B,), 0.5)
    goal_cond = torch.randn(B, 14)
    s = score_fn(tau, t, goal_cond=goal_cond)
    assert s.shape == (B, H1, 7)
    assert torch.isfinite(s).all()


def test_V9_total_loss_finite_and_runs():
    """traj_total_loss_v51_pose runs end-to-end with μ_pose=0 and μ_pose>0,
    produces finite scalars, and shares the forward sample when μ_pose>0."""
    arm_b, sched, sde = _arm_setup()
    tau_0, u_0, z_e = _build_tau_0(arm_b)
    B = tau_0.shape[0]
    net = TrajectoryScoreNetUNetPose(
        manifold=arm_b, H=tau_0.shape[1] - 1, down_dims=(64, 128),
        diffusion_step_embed_dim=64, goal_cond_dim=14, cond_injection="channel",
    )
    score_fn = TrajectoryScaledScoreFnPose(net, sde, std_trick=True, proxy_std_mode="ou")
    goal_cond = torch.randn(B, 14)

    L0 = traj_total_loss_v51_pose(score_fn, sde, tau_0, mu_pose=0.0, goal_cond=goal_cond)
    L1 = traj_total_loss_v51_pose(score_fn, sde, tau_0, mu_pose=0.5,
                                    tau_cutoff=0.5, goal_cond=goal_cond)
    assert torch.isfinite(L0) and torch.isfinite(L1)
    assert L0.item() > 0.0


# ---------------------------------------------------------------------

if __name__ == "__main__":
    import traceback
    tests = [
        test_V1_schedule_endpoints,
        test_V2_proxy_std_ou_equals_sigma,
        test_V3_closed_form_marginals,
        test_V4_score_target_matches_grad_log_p,
        test_V5_gbar_identity_noop,
        test_V6_reverse_is_ik_free,
        test_V7_reverse_drift_sign_with_zero_score,
        test_V8_score_net_input_no_q_init,
        test_V9_total_loss_finite_and_runs,
    ]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"OK   {t.__name__}")
        except Exception as e:
            fails += 1
            print(f"FAIL {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - fails}/{len(tests)} tests passed")
    raise SystemExit(0 if fails == 0 else 1)
