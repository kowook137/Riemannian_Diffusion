"""Tier 2 (boundary-active) demo generation pre-flight check.

Reference: Experiment_plan.md §2.2 Tier 2 + §2.3 검증 protocol.

This script is the versioned form of the previously ad-hoc /tmp/tier2_demo_check
scripts that produced the validated v5 Tier 2 config:

    q_rest_A = [+0.0, -0.3, 0.0, -0.40, 0.0, +3.40, 0.0]   # q[3], q[5] near limit
    q_rest_B = [+0.0, -0.3, 0.0, -1.50, 0.0, +1.50, 0.0]   # safe, center-ish
    p_box    = ([0.40,-0.05,0.40], [0.50,+0.05,0.50])      # Tier 0 reachable box
    n_ik_steps = 25, ik_alpha_null = 0.25, ik_clamp_to_limits = True

On 2026-05-10 this configuration produced:
    1%-tile rel-margin = 0.0010   (< 0.05 ✓)
    active_joint_ratio = 0.1158   (> 0.10 ✓)
    b_max p90          = 0.999    (> 0.95 ✓)
    feasible fraction  = 1.0000   (> 0.99 ✓)
    Tier 2 verdict     = PASS

It also reports the Tier 0 control (no clamp, current paper config) for
backward-compat reference.

Usage:
    python -m smcdp.experiments.tier2_demo_preflight \\
        --stage1-ckpt outputs/franka_stage1_pose/xi_phi.pt \\
        [--n 2048] [--tier {0,2}]

Notes:
    * `T_target_used` is set to the *realized* endpoint T_phi(q_H, z_e)
      (not the IK target) — see demo_gen_pose.py §7.  This is critical for
      Tier 2 because the IK clamp prevents convergence when seeds are
      near the boundary.
    * Pass criteria mirror Experiment_plan.md §2.3.
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
import pybullet_data

from smcdp.manifolds_pose import Franka7DoFPose
from smcdp.franka.self_model_pose import (
    PoseResidualMLP, LearnedSelfModelFranka7DoFPose,
)
from smcdp.franka.demo_gen_pose import FrankaBimodalReachingDemoPose
from smcdp.lie_se3 import log_relative_Rp, quat_to_R


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


# -----------------------------------------------------------------------------
# Tier 0 (control, no clamp) and Tier 2 v5 (validated 2026-05-10) presets.
# Keep these in sync with Experiment_plan.md §2.2.
# -----------------------------------------------------------------------------
PRESETS = {
    0: dict(
        label="Tier 0 (control, no clamp)",
        q_rest_A=[+0.6, -0.3, 0.0, -1.7, 0.0, 1.4, 0.0],
        q_rest_B=[-0.6, -0.3, 0.0, -1.7, 0.0, 1.4, 0.0],
        p_box_lo=[0.40, -0.05, 0.40],
        p_box_hi=[0.50, +0.05, 0.50],
        jitter_q=0.05,
        target_perturb_deg=30.0,
        n_ik_steps=10,
        ik_alpha=0.5,
        ik_alpha_null=0.3,
        ik_lam=0.05,
        ik_clamp_to_limits=False,
        ik_clamp_margin_frac=0.001,
    ),
    2: dict(
        label="Tier 2 v5 (2-joint boundary q[3]+q[5], clamp on)",
        q_rest_A=[+0.0, -0.3, 0.0, -0.40, 0.0, +3.40, 0.0],
        q_rest_B=[+0.0, -0.3, 0.0, -1.50, 0.0, +1.50, 0.0],
        p_box_lo=[0.40, -0.05, 0.40],
        p_box_hi=[0.50, +0.05, 0.50],
        jitter_q=0.05,
        target_perturb_deg=20.0,
        n_ik_steps=25,
        ik_alpha=0.5,
        ik_alpha_null=0.25,
        ik_lam=0.05,
        ik_clamp_to_limits=True,
        ik_clamp_margin_frac=0.001,
    ),
}


def _load_arm(stage1_ckpt: str | None, device, dtype):
    """Build the analytic IK arm and (optionally) the learned self-model arm.

    If `stage1_ckpt` is None or doesn't exist, fall back to the analytic arm
    for both roles.  Joint-margin / feasibility / b_max metrics depend only
    on q, so analytic-only mode reproduces the boundary statistics exactly
    (only the conditioning consistency check uses T_phi, and even there the
    analytic arm gives a meaningful "T_phi(q_H) vs realized" sanity check).
    """
    arm_a = Franka7DoFPose(URDF, sigma_p=0.05, sigma_R=0.1, tool_z_max=0.20)
    arm_a._ensure_chain(torch.zeros(1, 7, device=device))
    if stage1_ckpt is None:
        print("[preflight] analytic-only mode (no learned residual) — "
              "joint-margin metrics are still exact.")
        return arm_a, arm_a
    s1 = torch.load(stage1_ckpt, map_location=device, weights_only=False)
    res = PoseResidualMLP(
        n_q=7, n_z=1,
        hidden=s1["args"]["hidden"], n_layers=s1["args"]["n_layers"],
        activation=torch.nn.Softplus, final_init_scale=1e-3, output_omega=True,
    ).to(device=device, dtype=dtype)
    res.load_state_dict(s1["residual_net_state"])
    res.eval()
    arm = LearnedSelfModelFranka7DoFPose(
        res, URDF, sigma_p=0.05, sigma_R=0.1, tool_z_max=0.20,
    )
    arm._ensure_chain(torch.zeros(1, 7, device=device))
    return arm_a, arm


def measure(label, arm, arm_a, cfg, n, device, dtype):
    print(f"\n=== {label} (n={n}) ===")
    demo = FrankaBimodalReachingDemoPose(
        manifold=arm, ik_arm=arm_a, H=15,
        q_rest_A=cfg["q_rest_A"], q_rest_B=cfg["q_rest_B"],
        p_box_lo=tuple(cfg["p_box_lo"]), p_box_hi=tuple(cfg["p_box_hi"]),
        z_e_range=(0.05, 0.15), branch_p_A=0.5,
        jitter_q=cfg["jitter_q"], n_ik_steps=cfg["n_ik_steps"],
        ik_alpha=cfg["ik_alpha"], ik_alpha_null=cfg["ik_alpha_null"],
        ik_lam=cfg["ik_lam"],
        R_anchor_axis_angle=(3.14159265, 0.0, 0.0),
        target_perturb_rad=cfg["target_perturb_deg"] * 3.14159265 / 180.0,
        ik_clamp_to_limits=cfg["ik_clamp_to_limits"],
        ik_clamp_margin_frac=cfg["ik_clamp_margin_frac"],
    )
    x_demo, branch_A, z_e, T_target, T_start = demo.sample(n, device=device, dtype=dtype)

    # 1) Conditioning consistency: T_target_used == T_phi(q_H, z_e)?
    n_q = 7
    q = x_demo[..., :n_q]
    R_q_H, p_q_H = arm.T_phi_Rp(q[:, -1, :], z_e)
    R_target = quat_to_R(T_target[..., :4])
    p_target = T_target[..., 4:]
    e_endpt = log_relative_Rp(R_q_H, p_q_H, R_target, p_target)
    e_p = e_endpt[..., :3].norm(dim=-1).cpu().numpy()
    e_R = e_endpt[..., 3:].norm(dim=-1).cpu().numpy()
    print(f"  conditioning consistency  pos {100*e_p.max():.2e}cm  "
          f"rot {np.degrees(e_R.max()):.2e}°  (should be ~0)")

    # 2) Per-step relative joint margin
    q_lo, q_hi = arm.joint_limits(device=device, dtype=dtype)
    q_range = q_hi - q_lo
    rel_lo = (q - q_lo) / q_range
    rel_hi = (q_hi - q) / q_range
    pj_margin = torch.minimum(rel_lo, rel_hi)                                 # (n, H+1, n_q)
    flat_min = pj_margin.min(-1).values.reshape(-1).cpu().numpy()
    p1 = float(np.percentile(flat_min, 1))
    p5 = float(np.percentile(flat_min, 5))
    p50 = float(np.percentile(flat_min, 50))
    feasible_frac = float((flat_min >= 0).mean())
    print(f"  per-step min margin: 1%={p1:.4f} 5%={p5:.4f} 50%={p50:.4f}  "
          f"min={flat_min.min():.4f}")
    print(f"  feasible (margin>=0) fraction: {feasible_frac:.4f}")

    print("  per-joint 1%-tile margin:")
    for i in range(n_q):
        pj = pj_margin[..., i].reshape(-1).cpu().numpy()
        print(f"    q[{i}]: 1%={np.percentile(pj, 1):+.4f}  "
              f"min={pj.min():+.4f}  med={np.percentile(pj, 50):.4f}")

    # 3) active_joint_ratio (Experiment_plan.md §2.3 with delta=0.05)
    delta_active = 0.05
    active = ((pj_margin > 0) & (pj_margin < delta_active)).float()
    air = float(active.mean().item())
    print(f"  active_joint_ratio (0 < margin < 0.05): {air:.4f}")

    # 4) b_max(τ) = 1 - min margin per trajectory
    b_max = 1.0 - pj_margin.amin(dim=(1, 2))
    bm = b_max.cpu().numpy()
    p90 = float(np.percentile(bm, 90))
    print(f"  b_max(τ): mean={bm.mean():.4f}  p50={np.percentile(bm,50):.4f}  "
          f"p90={p90:.4f}  p99={np.percentile(bm,99):.4f}")

    # 5) Per-mode breakdown
    A = branch_A.cpu().numpy().astype(bool)
    margin_traj = pj_margin.amin(dim=(1, 2)).cpu().numpy()
    print(f"  mode A (n={A.sum()}): margin mean={margin_traj[A].mean():.4f}  "
          f"1%={np.percentile(margin_traj[A],1):+.4f}  "
          f"feas={(margin_traj[A]>=0).mean():.3f}")
    print(f"  mode B (n={(~A).sum()}): margin mean={margin_traj[~A].mean():.4f}  "
          f"1%={np.percentile(margin_traj[~A],1):+.4f}  "
          f"feas={(margin_traj[~A]>=0).mean():.3f}")

    # 6) Tier 2 verdict
    pass_ = (p1 < 0.05) and (air > 0.10) and (p90 > 0.95) and (feasible_frac > 0.99)
    print(f"\n  --> Tier 2 verdict: {'PASS' if pass_ else 'FAIL'}  "
          f"(p1<0.05? {p1<0.05}, AIR>0.10? {air>0.10}, "
          f"b_p90>0.95? {p90>0.95}, feas>0.99? {feasible_frac>0.99})")
    return {
        "p1": p1, "p5": p5, "p50": p50, "feasible_frac": feasible_frac,
        "active_joint_ratio": air, "b_max_p90": p90, "pass": pass_,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-ckpt", type=str,
                   default="outputs/franka_stage1_pose/xi_phi.pt")
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--tier", type=int, choices=[0, 2], nargs="+", default=[0, 2])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    torch.manual_seed(args.seed)
    arm_a, arm = _load_arm(args.stage1_ckpt, device=device, dtype=dtype)

    for tier in args.tier:
        cfg = PRESETS[tier]
        measure(cfg["label"], arm, arm_a, cfg, n=args.n,
                device=device, dtype=dtype)


if __name__ == "__main__":
    main()
