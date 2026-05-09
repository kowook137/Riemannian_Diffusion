"""Pose-extended baseline eval — matches Ours-V2 pose Method A eval protocol.

For each baseline ckpt (BC / DP-official / Projected, pose-extended):
  - Resamples N (T_start, T_target, z_e) targets from the pose demo distribution
  - Runs baseline-specific sampling
  - Computes endpoint pose error (e_p, e_R) via Log_SE(3)
  - Reports succ@(pos < 0.05 m AND rot < 0.262 rad ≈ 15°) — matches Method A
  - Manifold adherence ‖g_φ‖_max for sanity

Output: outputs/<run-dir>/eval_metrics.json with same per-z structure as Ours-V2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import pybullet_data

from smcdp.manifolds_pose import Franka7DoFPose
from smcdp.franka.self_model_pose import (
    PoseResidualMLP, LearnedSelfModelFranka7DoFPose,
)
from smcdp.franka.demo_gen_pose import FrankaBimodalReachingDemoPose
from smcdp.lie_se3 import log_relative_Rp, quat_to_R, pose7_to_Rp
from smcdp.baselines import (
    BCTrajectoryPredictor,
    make_official_diffusion_policy, official_dp_sample,
    channel_concat_dp_sample,
)
from smcdp.franka.eval_metrics_pose import (
    compute_pose_metrics, format_header, format_row,
)


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to baseline ckpt.pt (pose-extended).")
    p.add_argument("--n-eval-per-z", type=int, default=64)
    p.add_argument("--z-list", type=float, nargs="+",
                   default=[0.05, 0.10, 0.15, 0.20])
    p.add_argument("--success-pos", type=float, default=0.05,
                   help="Success threshold on position (m).")
    p.add_argument("--success-rot", type=float, default=0.262,
                   help="Success threshold on rotation (rad), default 15°.")
    p.add_argument("--n-inference-steps", type=int, default=100)
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    torch.manual_seed(args.seed)

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    a = ck["args"]
    s1 = ck["stage1_pose_args"]
    print(f"loaded {args.ckpt}")
    print(f"  baseline={a['baseline']}  cond_injection={a.get('cond_injection', '-')}")

    # ---- Manifold (matches training) ----
    arm_a = Franka7DoFPose(
        urdf_path=URDF, end_link="panda_hand", tool_z_max=a["z_max_tool"],
        sigma_p=a["sigma_p"], sigma_R=a["sigma_R"],
    )
    arm_a._ensure_chain(torch.zeros(1, 7, device=device))
    residual_net = PoseResidualMLP(
        n_q=7, n_z=1, hidden=s1["hidden"], n_layers=s1["n_layers"],
        activation=torch.nn.Softplus, final_init_scale=1e-3, output_omega=True,
    ).to(device=device, dtype=dtype)
    # We saved stage1 args but not the state — must reload from xi_phi.pt
    stage1_ckpt = torch.load(a["stage1_pose_ckpt"], map_location=device, weights_only=False)
    residual_net.load_state_dict(stage1_ckpt["residual_net_state"])
    residual_net.eval()
    arm = LearnedSelfModelFranka7DoFPose(
        residual_net=residual_net, urdf_path=URDF, end_link="panda_hand",
        tool_z_max=a["z_max_tool"], sigma_p=a["sigma_p"], sigma_R=a["sigma_R"],
    )
    arm._ensure_chain(torch.zeros(1, 7, device=device))

    H1 = a["H"] + 1
    n_q, n_T = 7, 7
    CTX_DIM = n_T + n_T + 1                                 # 15

    # ---- Build & load model ----
    if a["baseline"] == "bc":
        model = BCTrajectoryPredictor(
            n_q=n_q, H=a["H"], ctx_dim=CTX_DIM,
            hidden=a["bc_hidden"], n_layers=a["bc_layers"],
            activation="sin",
        ).to(device)
        scheduler = None
    elif a["baseline"] == "dp_official":
        if a.get("cond_injection", "global") == "channel":
            model, scheduler = make_official_diffusion_policy(
                n_q=n_q + CTX_DIM, global_cond_dim=None,
                down_dims=list(a["down_dims"]),
                diffusion_step_embed_dim=a["diff_step_embed"],
                n_train_timesteps=a["dp_train_timesteps"],
            )
        else:
            model, scheduler = make_official_diffusion_policy(
                n_q=n_q, global_cond_dim=CTX_DIM,
                down_dims=list(a["down_dims"]),
                diffusion_step_embed_dim=a["diff_step_embed"],
                n_train_timesteps=a["dp_train_timesteps"],
            )
        model = model.to(device)
    elif a["baseline"] == "projected":
        model, scheduler = make_official_diffusion_policy(
            n_q=n_q + n_T, global_cond_dim=CTX_DIM,
            down_dims=list(a["down_dims"]),
            diffusion_step_embed_dim=a["diff_step_embed"],
            n_train_timesteps=a["dp_train_timesteps"],
        )
        model = model.to(device)
    else:
        raise ValueError(a["baseline"])

    state = ck["ema_state"] if (args.use_ema and "ema_state" in ck) else ck["model_state"]
    model.load_state_dict(state)
    model.eval()

    # ---- Demo distribution for sampling targets ----
    target_perturb_rad = a["target_perturb_deg"] * 3.14159265 / 180.0
    out_dir = Path(args.ckpt).parent
    metrics = {"per_z": [], "args": vars(args), "ckpt": str(args.ckpt),
                "baseline": a["baseline"], "cond_injection": a.get("cond_injection", "-")}

    print("\n" + format_header())
    for z_val in args.z_list:
        torch.manual_seed(args.seed + 1000)
        # Sample fresh targets from the demo distribution at this z_e
        data_z = FrankaBimodalReachingDemoPose(
            manifold=arm, ik_arm=arm_a, H=a["H"],
            q_rest_A=list(a["q_rest_A"]), q_rest_B=list(a["q_rest_B"]),
            p_box_lo=tuple(a["p_box_lo"]), p_box_hi=tuple(a["p_box_hi"]),
            z_e_range=(z_val, z_val), branch_p_A=a["branch_p_A"],
            jitter_q=a["jitter_q"], n_ik_steps=a["n_ik_steps"],
            R_anchor_axis_angle=tuple(a["R_anchor_aa"]),
            target_perturb_rad=target_perturb_rad,
        )
        x_demo, _, _, T_target, T_start = data_z.sample(
            args.n_eval_per_z, device=device, dtype=dtype,
        )
        ctx = torch.cat([T_start, T_target, torch.full((args.n_eval_per_z, 1), z_val,
                                                        device=device, dtype=dtype)],
                         dim=-1)
        z_tensor = torch.full((args.n_eval_per_z, 1), z_val, device=device, dtype=dtype)

        # ---- Sample per baseline ----
        if a["baseline"] == "bc":
            with torch.no_grad():
                q_gen = model(ctx)                                  # (n, H+1, 7)
            q_flat = q_gen.reshape(-1, n_q)
            z_flat = z_tensor.unsqueeze(1).expand(-1, H1, -1).reshape(-1, 1)
            x_gen = arm.make_x(q_flat, z_flat).reshape(args.n_eval_per_z, H1, arm.ambient_dim)
        elif a["baseline"] == "dp_official":
            cond_inj = a.get("cond_injection", "global")
            with torch.no_grad():
                if cond_inj == "channel":
                    ctx_per_h = ctx.unsqueeze(1).expand(-1, H1, -1)
                    q_gen = channel_concat_dp_sample(
                        model, scheduler, batch_size=args.n_eval_per_z,
                        horizon=H1, n_q=n_q, ctx_per_h=ctx_per_h, device=device,
                        n_inference_steps=args.n_inference_steps,
                    )
                else:
                    q_gen = official_dp_sample(
                        model, scheduler, batch_size=args.n_eval_per_z,
                        horizon=H1, n_q=n_q, ctx=ctx, device=device,
                        n_inference_steps=args.n_inference_steps,
                    )
            q_flat = q_gen.reshape(-1, n_q)
            z_flat = z_tensor.unsqueeze(1).expand(-1, H1, -1).reshape(-1, 1)
            x_gen = arm.make_x(q_flat, z_flat).reshape(args.n_eval_per_z, H1, arm.ambient_dim)
        elif a["baseline"] == "projected":
            with torch.no_grad():
                x_amb = official_dp_sample(
                    model, scheduler, batch_size=args.n_eval_per_z,
                    horizon=H1, n_q=n_q + n_T, ctx=ctx, device=device,
                    n_inference_steps=args.n_inference_steps,
                )
                # Project: replace T-block with T_φ(q-block, z_e).
                q_part = x_amb[..., :n_q]
                z_flat = z_tensor.unsqueeze(1).expand(-1, H1, -1).reshape(-1, 1)
                x_gen = arm.make_x(q_part.reshape(-1, n_q), z_flat).reshape(
                    args.n_eval_per_z, H1, arm.ambient_dim
                )
                q_gen = q_part

        # ---- Comprehensive metrics per metric.md ----
        m = compute_pose_metrics(
            arm, x_gen, T_target,
            x_demo=x_demo, sigma_R=a["sigma_R"],
        )
        m["z_e"] = z_val
        metrics["per_z"].append(m)
        print(format_row(z_val, m))

    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nsaved {out_dir / 'eval_metrics.json'}")


if __name__ == "__main__":
    main()
