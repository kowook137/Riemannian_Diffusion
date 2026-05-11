"""Re-eval an Ours-V2 (Method A or ablation) ckpt with the comprehensive
metric.md metric set.  Loads ckpt from existing training run, no retraining.

Usage:
    python -m smcdp.experiments.franka_pose_reeval \
        --ckpt outputs/franka_traj_unet_pose_method_a/ours_v2_pose.pt
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
    PoseLangevinSDE, traj_reverse_grw_pose,
    PoseChartOUSDE, traj_reverse_ou_chart_pose,
)
from smcdp.franka.eval_metrics_pose import (
    compute_pose_metrics, format_header, format_row,
)
from smcdp.lie_se3 import pose7_to_Rp


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--n-eval-per-z", type=int, default=64)
    p.add_argument("--z-list", type=float, nargs="+",
                   default=[0.05, 0.10, 0.15, 0.20])
    p.add_argument("--out-name", type=str, default="eval_metrics_full.json")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    torch.manual_seed(args.seed)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    a = ckpt["args"]
    print(f"loaded {args.ckpt}  method_a={a.get('method_a', False)}  "
          f"sigma_p={a['sigma_p']}  kappa={a.get('confining_kappa', 0)}")

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

    # v4.1: reconstruct bounded-chart wrapper if ckpt was trained with one.
    # Args saved via vars(args), so flags propagate transparently.
    if a.get("bounded_chart", False):
        arm = BoundedChartPoseManifold(
            arm, make_chart_from_manifold(arm, bounded=True),
            lambda_floor=float(a.get("lambda_floor", 1e-4)),
        )
        print(f"[v4.1] bounded chart wrapper reconstructed "
              f"(TanhBoundedChart, lambda_floor={a.get('lambda_floor', 1e-4):.1e})")

    # Score net + SDE — v5.1 (chart-space OU) vs v4.1 (Langevin Brownian) dispatch.
    schedule = LinearBetaSchedule(beta_0=a["beta_0"], beta_f=a["beta_f"], tf=1.0)
    use_v51 = bool(a.get("use_v51", False))
    if use_v51:
        sde = PoseChartOUSDE(arm, schedule, gbar_mode=a.get("gbar_mode", "identity"))
        proxy_std_mode = "ou"
        print(f"[v5.1] PoseChartOUSDE reconstructed (gbar_mode={a.get('gbar_mode','identity')})")
    else:
        sde = PoseLangevinSDE(arm, schedule,
                               limiting_q_mean=torch.tensor(a["limiting_mean_q"], dtype=dtype),
                               limiting_scale=a.get("limiting_scale", None),
                               forward_langevin_drift=a["forward_langevin_drift"],
                               confining_kappa=a.get("confining_kappa", 0.0),
                               confining_epsilon_frac=a.get("confining_epsilon_frac", 0.05))
        proxy_std_mode = a.get("proxy_std_mode") or ("brownian" if a.get("method_a") else "ou")
    net = TrajectoryScoreNetUNetPose(manifold=arm, H=a["H"],
        down_dims=tuple(a["down_dims"]),
        diffusion_step_embed_dim=a["diff_step_embed"],
        n_groups=a["unet_groups"], kernel_size=a["unet_kernel"],
        cond_predict_scale=False, t_scale=a["t_scale"],
        goal_cond_dim=14, cond_injection=a["cond_injection"]).to(device=device, dtype=dtype)
    net.load_state_dict(ckpt["ema_net"]); net.eval()
    score_fn = TrajectoryScaledScoreFnPose(net, sde, std_trick=True,
                                             proxy_std_mode=proxy_std_mode)

    # Demo + eval
    target_perturb_rad = a["target_perturb_deg"] * 3.14159265 / 180.0
    out_dir = Path(args.ckpt).parent
    metrics = {"per_z": [], "args": vars(args), "source_ckpt": str(args.ckpt),
                "method_a": a.get("method_a", False),
                "use_v51": use_v51,
                "config": {k: a.get(k) for k in ["sigma_p", "tikhonov_frac",
                                                   "confining_kappa", "forward_langevin_drift",
                                                   "limiting_scale", "proxy_std_mode",
                                                   "bounded_chart", "use_v51", "gbar_mode",
                                                   "mu_pose", "tau_cutoff", "loss_metric",
                                                   "alpha_s", "alpha_g", "beta_f"]}}
    print("\n" + format_header())
    H1 = a["H"] + 1
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
        x_demo, _, _, T_target, T_start = demo.sample(args.n_eval_per_z, device=device, dtype=dtype)
        goal_cond = torch.cat([T_start, T_target], dim=-1)
        z_e = torch.full((args.n_eval_per_z, 1), z_val, device=device, dtype=dtype)
        # decode anchor poses for guidance (if used at training)
        T_start_Rp = pose7_to_Rp(T_start) if (a.get("start_alpha_p", 0) > 0 or a.get("start_alpha_R", 0) > 0) else None
        T_target_Rp = pose7_to_Rp(T_target) if (a.get("goal_alpha_p", 0) > 0 or a.get("goal_alpha_R", 0) > 0) else None
        H_idx = H1 - 1
        if a.get("goal_h_mask") == "all":      goal_h = list(range(H1))
        elif a.get("goal_h_mask") == "last_half":  goal_h = list(range(H1 // 2, H1))
        elif a.get("goal_h_mask") == "last_quarter": goal_h = list(range(3 * H1 // 4, H1))
        else: goal_h = [H_idx]

        if use_v51:
            # v5.1: IK-free reverse — NO q_init, NO limiting_q_mean.
            # Optional spec §12.5 alpha_s / alpha_g scaling honored if stored.
            alpha_s = a.get("alpha_s")
            alpha_g = a.get("alpha_g")
            start_ap = (alpha_s * a.get("start_alpha_p", 0.0)
                        if alpha_s is not None else a.get("start_alpha_p", 0.0))
            start_aR = (alpha_s * a.get("start_alpha_R", 0.0)
                        if alpha_s is not None else a.get("start_alpha_R", 0.0))
            goal_ap = (alpha_g * a.get("goal_alpha_p", 0.0)
                       if alpha_g is not None else a.get("goal_alpha_p", 0.0))
            goal_aR = (alpha_g * a.get("goal_alpha_R", 0.0)
                       if alpha_g is not None else a.get("goal_alpha_R", 0.0))
            samples = traj_reverse_ou_chart_pose(
                sde, score_fn, n_samples=args.n_eval_per_z, H=a["H"],
                n_steps=a["n_sample_steps"], goal_cond=goal_cond, z_e=z_e,
                eps=a["eps"], device=device, dtype=dtype,
                T_start_Rp=T_start_Rp,
                start_alpha_p=start_ap, start_alpha_R=start_aR,
                start_h_indices=[0],
                T_target_Rp=T_target_Rp,
                goal_alpha_p=goal_ap, goal_alpha_R=goal_aR,
                goal_h_indices=goal_h,
                smoothness_alpha_vel=a.get("smoothness_alpha_vel", 0.0),
                smoothness_alpha_acc=a.get("smoothness_alpha_acc", 0.0),
            )
        else:
            # Method A: per-traj q_init from x_demo[0]; legacy: None
            q_init_eval = (x_demo[:, 0, :arm.n_q].detach().to(device=device, dtype=dtype)
                           if a.get("method_a") else None)
            samples = traj_reverse_grw_pose(
                sde, score_fn, n_samples=args.n_eval_per_z, H=a["H"],
                n_steps=a["n_sample_steps"], goal_cond=goal_cond, z_e=z_e,
                limiting_q_mean=torch.tensor(a["limiting_mean_q"]),
                q_init=q_init_eval, limiting_scale=a.get("limiting_scale"),
                eps=a["eps"], device=device, dtype=dtype,
                T_start_Rp=T_start_Rp,
                start_alpha_p=a.get("start_alpha_p", 0.0), start_alpha_R=a.get("start_alpha_R", 0.0),
                start_h_indices=[0],
                T_target_Rp=T_target_Rp,
                goal_alpha_p=a.get("goal_alpha_p", 0.0), goal_alpha_R=a.get("goal_alpha_R", 0.0),
                goal_h_indices=goal_h,
                smoothness_alpha_vel=a.get("smoothness_alpha_vel", 0.0),
                smoothness_alpha_acc=a.get("smoothness_alpha_acc", 0.0),
            )
        m = compute_pose_metrics(arm, samples, T_target,
                                  x_demo=x_demo, sigma_R=a["sigma_R"],
                                  q_rest_A=list(a["q_rest_A"]),
                                  q_rest_B=list(a["q_rest_B"]))
        m["z_e"] = z_val
        metrics["per_z"].append(m)
        print(format_row(z_val, m))

    out_path = out_dir / args.out_name
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
