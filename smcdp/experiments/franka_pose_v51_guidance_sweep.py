"""v5.1 guidance + chart-norm sampling sweep on a fixed checkpoint.

Implements `diagnostic_plan.md` §3-§5 — sweeps a grid of
(alpha_start, alpha_goal, alpha_u) at sampling time WITHOUT retraining the
score net. Output: per-config metric.md metrics + chart-saturation stats
+ markdown summary table.

Convention: alpha-scalars are applied uniformly to (alpha_p, alpha_R)
components for both start and goal anchors.  i.e. for "G2 spec default"
(alpha_g=1.0, alpha_s=2.0):  start_alpha_p=start_alpha_R=2.0,
goal_alpha_p=goal_alpha_R=1.0.

Usage:
    python -m smcdp.experiments.franka_pose_v51_guidance_sweep \
        --ckpt outputs/v51_tier2_50k_baseline/ours_v2_pose.pt \
        --n-eval-per-z 64 --n-sample-steps 200 \
        --z-list 0.05 0.10 0.15 0.20 \
        --out-name guidance_sweep_v51.json
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import torch
import pybullet_data

from smcdp.sde import LinearBetaSchedule
from smcdp.manifolds_pose import Franka7DoFPose, BoundedChartPoseManifold
from smcdp.charts import make_chart_from_manifold
from smcdp.franka.self_model_pose import (PoseResidualMLP, LearnedSelfModelFranka7DoFPose)
from smcdp.franka.demo_gen_pose import FrankaBimodalReachingDemoPose
from smcdp.trajectories_pose import (
    TrajectoryScoreNetUNetPose, TrajectoryScaledScoreFnPose,
    PoseChartOUSDE, traj_reverse_ou_chart_pose,
)
from smcdp.franka.eval_metrics_pose import (
    compute_pose_metrics, compute_chart_saturation,
)
from smcdp.lie_se3 import pose7_to_Rp


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


# Sweep grid (diagnostic_plan.md §4 first sweep)
# Convention: alpha-scalars are multipliers over the spec's metric weights
# W_p = 1/σ_p² and W_R = 1/σ_R² (extension.tex §1.2).  i.e. for a config
# with α_g = 1.0:  goal_alpha_p = 1.0·W_p,  goal_alpha_R = 1.0·W_R.
# This matches the natural-gradient `G^{-1} ∇R` magnitude with the score
# scale (otherwise guidance vanishes near the chart boundary where D_ψ→0).
GRID = [
    # (name, alpha_g, alpha_s, alpha_u, note)
    ("G0", 0.0, 0.0, 0.0,   "current baseline"),
    ("G1", 0.5, 1.0, 0.0,   "mild guidance"),
    ("G2", 1.0, 2.0, 0.0,   "spec default α_s=2 α_g"),
    ("G3", 2.0, 4.0, 0.0,   "strong guidance"),
    ("G4", 0.5, 1.0, 0.1,   "mild + anti-saturation"),
    ("G5", 1.0, 2.0, 0.1,   "default + anti-saturation"),
    ("G6", 2.0, 4.0, 0.1,   "strong + anti-saturation"),
    ("G7", 1.0, 2.0, 0.5,   "stronger anti-saturation"),
    ("G8", 2.0, 4.0, 0.5,   "aggressive"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--n-eval-per-z", type=int, default=64)
    p.add_argument("--n-sample-steps", type=int, default=200)
    p.add_argument("--z-list", type=float, nargs="+",
                   default=[0.05, 0.10, 0.15, 0.20])
    p.add_argument("--out-name", type=str, default="guidance_sweep_v51.json")
    p.add_argument("--md-name", type=str, default="guidance_sweep_v51.md")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--configs", type=str, nargs="*", default=None,
                   help="Subset of config names to run (e.g. G0 G1 G2). Default = all.")
    return p.parse_args()


def _reconstruct(ckpt_path: str, device, dtype):
    """Load v5.1 ckpt + reconstruct arm, schedule, SDE, score net."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    if not a.get("use_v51"):
        raise RuntimeError(f"ckpt {ckpt_path} is not v5.1 (use_v51=False)")
    # Stage 1
    s1ck = torch.load(a["stage1_pose_ckpt"], map_location=device, weights_only=False)
    arm_a = Franka7DoFPose(URDF, sigma_p=a["sigma_p"], sigma_R=a["sigma_R"],
                            tool_z_max=a["z_max"], tikhonov_frac=a.get("tikhonov_frac", 0.0))
    arm_a._ensure_chain(torch.zeros(1, 7, device=device))
    res = PoseResidualMLP(n_q=7, n_z=1, hidden=s1ck["args"]["hidden"],
                           n_layers=s1ck["args"]["n_layers"], activation=torch.nn.Softplus,
                           final_init_scale=1e-3, output_omega=True).to(device=device, dtype=dtype)
    res.load_state_dict(s1ck["residual_net_state"]); res.eval()
    arm = LearnedSelfModelFranka7DoFPose(res, URDF, sigma_p=a["sigma_p"],
                                           sigma_R=a["sigma_R"], tool_z_max=a["z_max"])
    arm.tikhonov_frac = float(a.get("tikhonov_frac", 0.0))
    arm._ensure_chain(torch.zeros(1, 7, device=device))
    if a.get("bounded_chart", False):
        arm = BoundedChartPoseManifold(
            arm, make_chart_from_manifold(arm, bounded=True),
            lambda_floor=float(a.get("lambda_floor", 1e-4)),
        )
    schedule = LinearBetaSchedule(beta_0=a["beta_0"], beta_f=a["beta_f"], tf=1.0)
    sde = PoseChartOUSDE(arm, schedule, gbar_mode=a.get("gbar_mode", "identity"))
    net = TrajectoryScoreNetUNetPose(manifold=arm, H=a["H"],
        down_dims=tuple(a["down_dims"]),
        diffusion_step_embed_dim=a["diff_step_embed"],
        n_groups=a["unet_groups"], kernel_size=a["unet_kernel"],
        cond_predict_scale=False, t_scale=a["t_scale"],
        goal_cond_dim=14, cond_injection=a["cond_injection"]).to(device=device, dtype=dtype)
    net.load_state_dict(ckpt["ema_net"]); net.eval()
    score_fn = TrajectoryScaledScoreFnPose(net, sde, std_trick=True, proxy_std_mode="ou")
    return ckpt, a, arm, arm_a, sde, score_fn


def _build_demo(a, arm, arm_a, z_val):
    target_perturb_rad = a["target_perturb_deg"] * 3.14159265 / 180.0
    return FrankaBimodalReachingDemoPose(
        manifold=arm, ik_arm=arm_a, H=a["H"],
        q_rest_A=list(a["q_rest_A"]), q_rest_B=list(a["q_rest_B"]),
        p_box_lo=tuple(a["p_box_lo"]), p_box_hi=tuple(a["p_box_hi"]),
        z_e_range=(z_val, z_val), branch_p_A=a["branch_p_A"],
        jitter_q=a["jitter_q"], n_ik_steps=a["n_ik_steps"],
        ik_alpha=a.get("ik_alpha", 0.5),
        ik_alpha_null=a.get("ik_alpha_null", 0.3),
        ik_lam=a.get("ik_lam", 0.05),
        R_anchor_axis_angle=tuple(a["R_anchor_aa"]),
        target_perturb_rad=target_perturb_rad,
        ik_clamp_to_limits=a.get("ik_clamp_to_limits", False),
        ik_clamp_margin_frac=a.get("ik_clamp_margin_frac", 0.001),
    )


def _run_config(sde, score_fn, args, a, arm, arm_a, alpha_g, alpha_s, alpha_u, z_val,
                 device, dtype, seed_base: int):
    """Sample N trajectories at z_val with the given (α_s, α_g, α_u)."""
    torch.manual_seed(seed_base)
    demo = _build_demo(a, arm, arm_a, z_val)
    x_demo, _, _, T_target, T_start = demo.sample(args.n_eval_per_z, device=device, dtype=dtype)
    goal_cond = torch.cat([T_start, T_target], dim=-1)
    z_e = torch.full((args.n_eval_per_z, 1), z_val, device=device, dtype=dtype)
    T_start_Rp = pose7_to_Rp(T_start) if (alpha_s > 0) else None
    T_target_Rp = pose7_to_Rp(T_target) if (alpha_g > 0) else None
    H1 = a["H"] + 1
    if a.get("goal_h_mask") == "all":      goal_h = list(range(H1))
    elif a.get("goal_h_mask") == "last_half":  goal_h = list(range(H1 // 2, H1))
    elif a.get("goal_h_mask") == "last_quarter": goal_h = list(range(3 * H1 // 4, H1))
    else: goal_h = [H1 - 1]

    # spec §1.2 metric weights — α_p, α_R use the spec's W as base scale
    W_p = 1.0 / (float(a["sigma_p"]) ** 2)                                      # default σ_p=0.05 → 400
    W_R = 1.0 / (float(a["sigma_R"]) ** 2)                                      # default σ_R=0.1  → 100

    samples = traj_reverse_ou_chart_pose(
        sde, score_fn, n_samples=args.n_eval_per_z, H=a["H"],
        n_steps=args.n_sample_steps, goal_cond=goal_cond, z_e=z_e,
        eps=a["eps"], device=device, dtype=dtype,
        T_start_Rp=T_start_Rp,
        start_alpha_p=float(alpha_s) * W_p, start_alpha_R=float(alpha_s) * W_R,
        start_h_indices=[0],
        T_target_Rp=T_target_Rp,
        goal_alpha_p=float(alpha_g) * W_p, goal_alpha_R=float(alpha_g) * W_R,
        goal_h_indices=goal_h,
        smoothness_alpha_vel=0.0, smoothness_alpha_acc=0.0,
        chart_norm_alpha=float(alpha_u),
    )
    m = compute_pose_metrics(arm, samples, T_target,
                              x_demo=x_demo, sigma_R=a["sigma_R"],
                              q_rest_A=list(a["q_rest_A"]),
                              q_rest_B=list(a["q_rest_B"]))
    sat = compute_chart_saturation(samples, n_q=arm.n_q)
    m.update(sat)
    m["z_e"] = z_val
    return m


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    torch.manual_seed(args.seed)
    ckpt_path = Path(args.ckpt)
    out_dir = ckpt_path.parent

    print(f"loading ckpt: {ckpt_path}")
    _, a, arm, arm_a, sde, score_fn = _reconstruct(str(ckpt_path), device, dtype)
    print(f"ckpt: use_v51={a.get('use_v51')}  bounded_chart={a.get('bounded_chart')}  "
          f"β_f={a['beta_f']}  μ_pose={a.get('mu_pose', 0)}")

    grid = [g for g in GRID if (args.configs is None or g[0] in args.configs)]
    print(f"sweep: {len(grid)} configs × {len(args.z_list)} z_e × {args.n_eval_per_z} samples")

    results = {"args": vars(args), "ckpt_args": a, "configs": []}
    print()
    header = (f"{'name':<3} | {'α_g':>5} | {'α_s':>5} | {'α_u':>6} | "
              f"{'z':>4} | {'pos cm':>6} | {'rot°':>5} | "
              f"{'5/5':>5} | {'5/10':>5} | {'u99':>5} | {'mfe':>4} | {'jvio':>5}")
    print(header)
    print("-" * len(header))

    for (name, alpha_g, alpha_s, alpha_u, note) in grid:
        per_z = []
        for z_val in args.z_list:
            m = _run_config(sde, score_fn, args, a, arm, arm_a,
                             alpha_g, alpha_s, alpha_u, z_val,
                             device, dtype, seed_base=args.seed + 1000)
            per_z.append(m)
            print(f"{name:<3} | {alpha_g:>5.2f} | {alpha_s:>5.2f} | {alpha_u:>6.0e} | "
                  f"{z_val:>4.2f} | {m['pos_err_mean_cm']:>6.2f} | {m['rot_err_mean_deg']:>5.2f} | "
                  f"{100*m['pose_succ_5cm_5deg']:>5.1f} | {100*m['pose_succ_5cm_10deg']:>5.1f} | "
                  f"{m['u_inf_p99']:>5.2f} | {m['mode_frac_err']:>4.2f} | "
                  f"{100*m['joint_viol_rate']:>4.1f}%")
        # per-config aggregate
        keys = ["pos_err_mean_cm", "rot_err_mean_deg", "pose_succ_5cm_5deg",
                 "pose_succ_5cm_10deg", "mode_frac_err", "joint_viol_rate",
                 "u_inf_p99", "u_inf_p90", "u_inf_p50", "saturation_rate"]
        avg = {k: float(sum(m[k] for m in per_z) / len(per_z)) for k in keys}
        results["configs"].append({
            "name": name, "alpha_g": alpha_g, "alpha_s": alpha_s,
            "alpha_u": alpha_u, "note": note,
            "per_z": per_z, "avg": avg,
        })

    # Markdown summary
    md = [
        f"# v5.1 guidance + R_u sweep — {ckpt_path.name}",
        "",
        "| name | α_g | α_s | α_u | pos cm avg | rot° avg | succ55 avg | succ510 avg | ‖u‖∞ p99 avg | sat>3 | mfe avg | jvio avg | note |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for c in results["configs"]:
        a_ = c["avg"]
        md.append(
            f"| {c['name']} | {c['alpha_g']:.2f} | {c['alpha_s']:.2f} | "
            f"{c['alpha_u']:.0e} | {a_['pos_err_mean_cm']:.2f} | {a_['rot_err_mean_deg']:.2f} | "
            f"{100*a_['pose_succ_5cm_5deg']:.1f}% | {100*a_['pose_succ_5cm_10deg']:.1f}% | "
            f"{a_['u_inf_p99']:.2f} | {100*a_['saturation_rate']:.1f}% | "
            f"{a_['mode_frac_err']:.3f} | {100*a_['joint_viol_rate']:.1f}% | {c['note']} |"
        )

    # Selection per diagnostic_plan §5
    eligible = [c for c in results["configs"] if c["avg"]["u_inf_p99"] < 3.0]
    md.append("")
    if eligible:
        best = max(eligible, key=lambda c: c["avg"]["pose_succ_5cm_5deg"])
        md.append(f"**Selected (‖u‖∞ p99 < 3.0 + max succ55)**: **{best['name']}** "
                  f"(α_g={best['alpha_g']}, α_s={best['alpha_s']}, α_u={best['alpha_u']}) — "
                  f"succ55 = {100*best['avg']['pose_succ_5cm_5deg']:.1f}%, "
                  f"‖u‖∞ p99 = {best['avg']['u_inf_p99']:.2f}")
    else:
        md.append("**No config met ‖u‖∞ p99 < 3.0 constraint** — recipe needs stronger R_u or μ_pose retrain.")

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / args.out_name
    md_path = out_dir / args.md_name
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    md_path.write_text("\n".join(md) + "\n")
    print()
    print(f"saved {json_path}")
    print(f"saved {md_path}")


if __name__ == "__main__":
    main()
