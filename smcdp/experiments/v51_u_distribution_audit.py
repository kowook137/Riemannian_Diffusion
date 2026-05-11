"""Audit demo vs generated u-distribution for v5.1 (diagnostic_plan §0c).

For a given v5.1 ckpt, compute the chart-norm distribution of:
  (a) demo trajectories  — u_demo = psi^{-1}(q_demo)
  (b) generated trajectories — u_gen from the reverse SDE sampler (G0, no guidance)

Report per-z:
  ‖u‖∞ percentiles {p50, p90, p99, max}, sat>2, sat>3 rates, mean ‖u‖_2.

If demo sat>3 ≈ gen sat>3 → chart_temp widening is the direct lever (problem is
the data covers the saturation regime).  If demo sat>3 ≪ gen sat>3 → score net
pushes mass OOD; the fix is architectural (endpoint conditioning, prediction
target ablation), not chart parameterization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import pybullet_data

from smcdp.manifolds_pose import Franka7DoFPose, BoundedChartPoseManifold
from smcdp.charts import make_chart_from_manifold
from smcdp.franka.self_model_pose import (
    PoseResidualMLP, LearnedSelfModelFranka7DoFPose,
)
from smcdp.franka.demo_gen_pose import FrankaBimodalReachingDemoPose
from smcdp.trajectories_pose import (
    PoseChartOUSDE, TrajectoryScaledScoreFnPose, traj_reverse_ou_chart_pose,
    TrajectoryScoreNetUNetPose,
)
from smcdp.sde import LinearBetaSchedule


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def _stats(u_inf):
    """u_inf : (N,) — per-waypoint ‖u‖∞."""
    n = u_inf.numel()
    s = u_inf.sort().values
    def pct(p):
        idx = max(0, min(n - 1, int(round(p * (n - 1)))))
        return s[idx].item()
    return {
        "n_waypoints": n,
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "max": u_inf.max().item(),
        "sat_gt2_rate": (u_inf > 2.0).float().mean().item(),
        "sat_gt3_rate": (u_inf > 3.0).float().mean().item(),
        "mean_l2": u_inf.mean().item(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--z-list", type=float, nargs="+",
                   default=[0.05, 0.10, 0.15, 0.20])
    p.add_argument("--n-samples", type=int, default=256,
                   help="Per z_e for both demo and gen.")
    p.add_argument("--n-sample-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-name", type=str, default="u_distribution_audit.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    a = ckpt["args"]
    if not a.get("bounded_chart"):
        raise RuntimeError("ckpt is not bounded-chart; u-distribution audit N/A")

    # ---- reconstruct manifold + sde + score net ----
    s1ck = torch.load(a["stage1_pose_ckpt"], map_location=device, weights_only=False)
    arm_a = Franka7DoFPose(URDF, sigma_p=a["sigma_p"], sigma_R=a["sigma_R"],
                            tool_z_max=a["z_max"],
                            tikhonov_frac=a.get("tikhonov_frac", 0.0))
    arm_a._ensure_chain(torch.zeros(1, 7, device=device))
    res = PoseResidualMLP(n_q=7, n_z=1, hidden=s1ck["args"]["hidden"],
                           n_layers=s1ck["args"]["n_layers"], activation=torch.nn.Softplus,
                           final_init_scale=1e-3, output_omega=True).to(device=device, dtype=dtype)
    res.load_state_dict(s1ck["residual_net_state"]); res.eval()
    arm = LearnedSelfModelFranka7DoFPose(res, URDF, sigma_p=a["sigma_p"],
                                          sigma_R=a["sigma_R"], tool_z_max=a["z_max"])
    arm.tikhonov_frac = float(a.get("tikhonov_frac", 0.0))
    arm._ensure_chain(torch.zeros(1, 7, device=device))
    arm = BoundedChartPoseManifold(
        arm,
        make_chart_from_manifold(
            arm, bounded=True,
            chart_temp=float(a.get("chart_temp", 1.0)),
        ),
        lambda_floor=float(a.get("lambda_floor", 1e-4)),
    )
    schedule = LinearBetaSchedule(beta_0=a["beta_0"], beta_f=a["beta_f"], tf=1.0)
    sde = PoseChartOUSDE(arm, schedule, gbar_mode=a.get("gbar_mode", "identity"))
    net = TrajectoryScoreNetUNetPose(manifold=arm, H=a["H"],
        down_dims=tuple(a["down_dims"]),
        diffusion_step_embed_dim=a["diff_step_embed"],
        n_groups=a["unet_groups"], kernel_size=a["unet_kernel"],
        cond_predict_scale=False, t_scale=a["t_scale"],
        goal_cond_dim=14, cond_injection=a["cond_injection"],
        endpoint_rel_cond=bool(a.get("endpoint_rel_cond", False)),
    ).to(device=device, dtype=dtype)
    net.load_state_dict(ckpt["ema_net"]); net.eval()
    score_fn = TrajectoryScaledScoreFnPose(net, sde, std_trick=True, proxy_std_mode="ou")
    target_perturb_rad = a["target_perturb_deg"] * 3.14159265 / 180.0

    out = {"ckpt": str(args.ckpt), "per_z": [], "args": vars(args)}
    print(f"\n{'z_e':>5} | {'src':<6} | {'p50':>5} | {'p90':>5} | {'p99':>5} | {'max':>6} | "
          f"{'sat>2':>6} | {'sat>3':>6} | n_wp")
    print("-" * 80)
    for z_val in args.z_list:
        torch.manual_seed(args.seed + 1000)
        demo = FrankaBimodalReachingDemoPose(
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
        x_demo, _, _, T_target, T_start = demo.sample(args.n_samples, device=device, dtype=dtype)
        u_demo = x_demo[..., :7]                              # (n, H+1, 7) bounded → already u
        u_demo_inf = u_demo.abs().reshape(-1, 7).max(-1).values
        d_demo = _stats(u_demo_inf)

        # generated
        with torch.no_grad():
            z_e = torch.full((args.n_samples, 1), z_val, device=device, dtype=dtype)
            goal_cond = torch.cat([T_start, T_target], dim=-1)
            x_gen = traj_reverse_ou_chart_pose(
                sde, score_fn, n_samples=args.n_samples, H=a["H"],
                n_steps=args.n_sample_steps, goal_cond=goal_cond, z_e=z_e,
                device=device, dtype=dtype, integrator="euler",
            )
        u_gen = x_gen[..., :7]
        u_gen_inf = u_gen.abs().reshape(-1, 7).max(-1).values
        d_gen = _stats(u_gen_inf)

        row = {"z_e": z_val, "demo": d_demo, "gen": d_gen}
        out["per_z"].append(row)
        for name, d in (("demo", d_demo), ("gen", d_gen)):
            print(f"{z_val:>5.2f} | {name:<6} | {d['p50']:>5.2f} | {d['p90']:>5.2f} | "
                  f"{d['p99']:>5.2f} | {d['max']:>6.2f} | "
                  f"{100*d['sat_gt2_rate']:>5.1f}% | {100*d['sat_gt3_rate']:>5.1f}% | "
                  f"{d['n_waypoints']}")

    out_path = Path(args.ckpt).parent / args.out_name
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {out_path}")

    # ---- aggregate verdict ----
    demo_avg_sat3 = sum(r["demo"]["sat_gt3_rate"] for r in out["per_z"]) / len(out["per_z"])
    gen_avg_sat3  = sum(r["gen"]["sat_gt3_rate"]  for r in out["per_z"]) / len(out["per_z"])
    print(f"\nverdict — demo sat>3 avg = {100*demo_avg_sat3:.1f}%, "
          f"gen sat>3 avg = {100*gen_avg_sat3:.1f}%")
    if demo_avg_sat3 > 0.20:
        print("→ demo itself sits in saturation regime; chart_temp widening is the direct fix.")
    elif gen_avg_sat3 > 2 * max(demo_avg_sat3, 0.01):
        print("→ score net pushes mass OOD relative to demo; endpoint/architectural fix needed.")
    else:
        print("→ demo and gen are comparable; saturation is mild — guidance/architectural neutral.")


if __name__ == "__main__":
    main()
