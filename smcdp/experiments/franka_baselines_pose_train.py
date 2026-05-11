"""Franka 7-DoF pose-extended baselines (BC / DP-official / Projected) — unified trainer.

Mirrors `franka_baselines_train.py` but for the SE(3) pose-extended task:
  - Manifold:  LearnedSelfModelFranka7DoFPose (Stage-1 ξ_φ frozen, full pose self-model)
  - Demo data: FrankaBimodalReachingDemoPose (T_target with rotation perturbation)
  - Context:   c = (T_start ⊕ T_target ⊕ z_e) ∈ R^{7+7+1=15}   (storage form)
  - Same H+1=16, batch=64, steps=15000 as Ours-V2 pose
  - Eval metric: e_p (position) + e_R (rotation) + combined succ@5cm

Baselines:
  - bc           : deterministic regressor c → q-trajectory
  - dp_official  : Diffusion Policy (canonical=global cond, or channel=DP-A variant)
  - projected    : DP on ambient (q, T_storage) + post-step projection via H_φ^pose

Storage form: T = (q_R ∈ S^3 quat, p ∈ R^3) ∈ R^7.
  Projected baseline diffuses on (q, T) ∈ R^{n_q+7=14}, post-step projects q-block
  through T_φ to get manifold-adherent T.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
import pybullet_data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from smcdp.manifolds_pose import Franka7DoFPose, BoundedChartPoseManifold
from smcdp.charts import make_chart_from_manifold
from smcdp.franka.self_model_pose import (
    PoseResidualMLP, LearnedSelfModelFranka7DoFPose,
)
from smcdp.franka.demo_gen_pose import FrankaBimodalReachingDemoPose
from smcdp.baselines import (
    BCTrajectoryPredictor, bc_loss,
    make_official_diffusion_policy, official_dp_loss, official_dp_sample,
    channel_concat_dp_loss, channel_concat_dp_sample,
)


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=str, required=True,
                   choices=["bc", "dp_official", "projected"])
    p.add_argument("--stage1-pose-ckpt", type=str,
                   default="outputs/franka_stage1_pose/xi_phi.pt")
    p.add_argument("--H", type=int, default=15)
    # demo (match Ours-V2 pose)
    p.add_argument("--q-rest-A", type=float, nargs=7,
                   default=[+0.6, -0.3, 0.0, -1.7, 0.0, 1.4, 0.0])
    p.add_argument("--q-rest-B", type=float, nargs=7,
                   default=[-0.6, -0.3, 0.0, -1.7, 0.0, 1.4, 0.0])
    p.add_argument("--p-box-lo", type=float, nargs=3, default=[0.40, -0.05, 0.40])
    p.add_argument("--p-box-hi", type=float, nargs=3, default=[0.50, +0.05, 0.50])
    p.add_argument("--branch-p-A", type=float, default=0.5)
    p.add_argument("--jitter-q", type=float, default=0.05)
    p.add_argument("--n-ik-steps", type=int, default=10)
    p.add_argument("--ik-alpha", type=float, default=0.5)
    p.add_argument("--ik-alpha-null", type=float, default=0.3)
    p.add_argument("--ik-lam", type=float, default=0.05)
    p.add_argument("--ik-clamp-to-limits", action="store_true",
                   help="Clamp post-step q to (q_min+δ, q_max-δ) at each IK "
                        "iteration. Required for Tier 1/2 boundary-active demos "
                        "(Experiment_plan.md §2.2).")
    p.add_argument("--ik-clamp-margin-frac", type=float, default=0.001)
    p.add_argument("--bounded-chart", action="store_true",
                   help="v4.1: wrap manifold with TanhBoundedChart so chart slot "
                        "stores u = psi^-1(q); psi(u) auto-enforces q in (q_min, q_max). "
                        "For DP this turns dp_official into DP-bounded (joint feasibility "
                        "by construction at sampling time).  See "
                        "joint_limit_extension.tex Sec 3-5.")
    p.add_argument("--lambda-floor", type=float, default=1e-4,
                   help="Tikhonov floor on G_Q^A; only active when --bounded-chart.")
    p.add_argument("--demo-pool-size", type=int, default=0,
                   help="If > 0, pre-generate this many demos once at startup and "
                        "sample minibatches from the cached pool during training. "
                        "Default 0 = legacy online-resampling each step (slow with "
                        "n_ik_steps=25).  Tier 2 recommended: 8192.")
    p.add_argument("--target-perturb-deg", type=float, default=30.0)
    p.add_argument("--R-anchor-aa", type=float, nargs=3,
                   default=[3.14159265, 0.0, 0.0])
    p.add_argument("--z-min", type=float, default=0.05)
    p.add_argument("--z-max", type=float, default=0.15)
    p.add_argument("--z-max-tool", type=float, default=0.20,
                   help="tool_z_max for the manifold (eval supports z_e up to this).")
    p.add_argument("--sigma-p", type=float, default=0.05)
    p.add_argument("--sigma-R", type=float, default=0.1)
    # training
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema", type=float, default=0.999)
    # arch
    p.add_argument("--bc-hidden", type=int, default=512)
    p.add_argument("--bc-layers", type=int, default=5)
    p.add_argument("--down-dims", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--diff-step-embed", type=int, default=128)
    p.add_argument("--dp-train-timesteps", type=int, default=100)
    p.add_argument("--cond-injection", type=str, default="global",
                   choices=["global", "channel"],
                   help="DP variant: 'global' (canonical Chi23) or 'channel' (DP-A).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default=None,
                   help="Default: outputs/franka_baseline_pose_{baseline}{_cond_injection}")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    torch.manual_seed(args.seed)
    if args.out_dir is None:
        suffix = f"_{args.cond_injection}" if args.cond_injection != "global" else ""
        args.out_dir = f"outputs/franka_baseline_pose_{args.baseline}{suffix}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"baseline={args.baseline} pose-extended  device={device}  out_dir={out_dir}")

    # ---- Manifold setup (pose-extended) ----
    arm_a = Franka7DoFPose(
        urdf_path=URDF, end_link="panda_hand", tool_z_max=args.z_max_tool,
        sigma_p=args.sigma_p, sigma_R=args.sigma_R,
    )
    arm_a._ensure_chain(torch.zeros(1, 7, device=device))

    ck = torch.load(args.stage1_pose_ckpt, map_location=device, weights_only=False)
    s1 = ck["args"]
    residual_net = PoseResidualMLP(
        n_q=7, n_z=1, hidden=s1["hidden"], n_layers=s1["n_layers"],
        activation=torch.nn.Softplus, final_init_scale=1e-3,
        output_omega=True,
    ).to(device=device, dtype=dtype)
    residual_net.load_state_dict(ck["residual_net_state"])
    residual_net.eval()
    print(f"  loaded ξ_φ Stage-1: {ck['metrics']}")

    arm = LearnedSelfModelFranka7DoFPose(
        residual_net=residual_net, urdf_path=URDF, end_link="panda_hand",
        tool_z_max=args.z_max_tool, sigma_p=args.sigma_p, sigma_R=args.sigma_R,
    )
    arm._ensure_chain(torch.zeros(1, 7, device=device))

    if args.bounded_chart:
        arm = BoundedChartPoseManifold(
            arm, make_chart_from_manifold(arm, bounded=True),
            lambda_floor=float(args.lambda_floor),
        )
        print(f"[bounded-chart] enabled (TanhBoundedChart, λ_floor={args.lambda_floor:.1e}) "
              f"— DP-bounded mode; chart slot = u, psi(u) auto-feasible.")
    else:
        print(f"[bounded-chart] disabled — DP-raw mode (chart slot = q in R^7).")

    # ---- Demo distribution (pose-extended) ----
    target_perturb_rad = args.target_perturb_deg * 3.14159265 / 180.0
    data = FrankaBimodalReachingDemoPose(
        manifold=arm, ik_arm=arm_a, H=args.H,
        q_rest_A=list(args.q_rest_A), q_rest_B=list(args.q_rest_B),
        p_box_lo=tuple(args.p_box_lo), p_box_hi=tuple(args.p_box_hi),
        z_e_range=(args.z_min, args.z_max),
        branch_p_A=args.branch_p_A, jitter_q=args.jitter_q,
        n_ik_steps=args.n_ik_steps,
        ik_alpha=args.ik_alpha, ik_alpha_null=args.ik_alpha_null, ik_lam=args.ik_lam,
        R_anchor_axis_angle=tuple(args.R_anchor_aa),
        target_perturb_rad=target_perturb_rad,
        ik_clamp_to_limits=args.ik_clamp_to_limits,
        ik_clamp_margin_frac=args.ik_clamp_margin_frac,
    )

    H1 = args.H + 1
    n_q, n_p = 7, 3
    n_T = 7                                     # (quat 4 + p 3)
    n_z = 1
    CTX_DIM = n_T + n_T + n_z                   # T_start + T_target + z_e = 15

    # ---- Build model per baseline ----
    if args.baseline == "bc":
        model = BCTrajectoryPredictor(
            n_q=n_q, H=args.H, ctx_dim=CTX_DIM,
            hidden=args.bc_hidden, n_layers=args.bc_layers,
            activation="sin",
        ).to(device)
        scheduler = None
    elif args.baseline == "dp_official":
        # clip_sample=False: q range exceeds [-1,1] (Franka q[5] ∈ (-0.087, 3.822)).
        # The diffusers DDPM default clip_sample=True clips x_0 prediction to
        # [-1, 1] each reverse step → unreachable for Tier 2 boundary modes.
        if args.cond_injection == "channel":
            model, scheduler = make_official_diffusion_policy(
                n_q=n_q + CTX_DIM, global_cond_dim=None,
                down_dims=list(args.down_dims),
                diffusion_step_embed_dim=args.diff_step_embed,
                n_train_timesteps=args.dp_train_timesteps,
                clip_sample=False,
            )
        else:
            model, scheduler = make_official_diffusion_policy(
                n_q=n_q, global_cond_dim=CTX_DIM,
                down_dims=list(args.down_dims),
                diffusion_step_embed_dim=args.diff_step_embed,
                n_train_timesteps=args.dp_train_timesteps,
                clip_sample=False,
            )
        model = model.to(device)
    elif args.baseline == "projected":
        # Diffuse on ambient (q, T_storage) ∈ R^{n_q + 7 = 14}
        model, scheduler = make_official_diffusion_policy(
            n_q=n_q + n_T, global_cond_dim=CTX_DIM,
            down_dims=list(args.down_dims),
            diffusion_step_embed_dim=args.diff_step_embed,
            n_train_timesteps=args.dp_train_timesteps,
            clip_sample=False,
        )
        model = model.to(device)
    else:
        raise ValueError(args.baseline)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params/1e6:.2f}M")

    # ---- EMA ----
    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # ---- Optimizer ----
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    def lr_lambda(step):
        if args.warmup_steps <= 0: return 1.0
        return min(1.0, (step + 1) / args.warmup_steps)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    # ---- Optional: pre-generate demo pool (avoids online IK in training loop) ----
    pool = None
    if args.demo_pool_size > 0:
        n_pool = int(args.demo_pool_size)
        print(f"[demo-pool] pre-generating {n_pool} trajectories once "
              f"(avoids per-step online IK)...")
        with torch.no_grad():
            x_p, _, z_p, T_t_p, T_s_p = data.sample(n_pool, device=device, dtype=dtype)
        pool = (x_p, z_p, T_t_p, T_s_p)
        print(f"[demo-pool] cached pool: x={tuple(x_p.shape)}, "
              f"z_e={tuple(z_p.shape)}, T_target={tuple(T_t_p.shape)}, "
              f"T_start={tuple(T_s_p.shape)}; {(x_p.numel()*x_p.element_size())/1e6:.1f} MB")

    # ---- Training loop ----
    losses = []
    pbar = tqdm(range(args.steps), desc=f"train {args.baseline} pose")
    for step in pbar:
        if pool is not None:
            idx = torch.randint(0, pool[0].shape[0], (args.batch,), device=device)
            x_demo  = pool[0][idx]
            z_e     = pool[1][idx]
            T_target = pool[2][idx]
            T_start  = pool[3][idx]
        else:
            x_demo, _, z_e, T_target, T_start = data.sample(args.batch, device=device, dtype=dtype)
        # ctx = (T_start, T_target, z_e) ∈ R^{15}
        ctx = torch.cat([T_start, T_target, z_e], dim=-1)
        q_demo = x_demo[..., :n_q]                                  # (B, H+1, 7)

        if args.baseline == "bc":
            loss = bc_loss(model, ctx, q_demo)
        elif args.baseline == "dp_official":
            if args.cond_injection == "channel":
                ctx_per_h = ctx.unsqueeze(1).expand(-1, H1, -1)     # (B, H+1, 15)
                loss = channel_concat_dp_loss(model, scheduler, q_demo, ctx_per_h)
            else:
                loss = official_dp_loss(model, scheduler, q_demo, ctx)
        elif args.baseline == "projected":
            # Ambient state per-h: (q, T_storage) ∈ R^{14}  (slice from x_demo)
            x_amb = x_demo[..., : n_q + n_T]                        # (B, H+1, 14)
            loss = official_dp_loss(model, scheduler, x_amb, ctx)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        lr_sched.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema_state[k].mul_(args.ema).add_(v, alpha=1.0 - args.ema)
                else:
                    ema_state[k].copy_(v)
        losses.append(loss.item())
        if step % 200 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    # ---- Save ----
    torch.save({
        "args": vars(args),
        "stage1_pose_args": s1,
        "model_state": model.state_dict(),
        "ema_state": ema_state,
        "scheduler_config": (
            dict(scheduler.config) if scheduler is not None else None
        ),
    }, out_dir / "ckpt.pt")
    print(f"saved {out_dir / 'ckpt.pt'}")

    # Loss plot
    fig, ax = plt.subplots(figsize=(7, 4))
    losses_t = torch.tensor(losses)
    win = max(1, len(losses_t) // 200)
    smooth = torch.nn.functional.avg_pool1d(losses_t.view(1, 1, -1),
                                              kernel_size=win, stride=win).flatten()
    ax.plot(torch.arange(len(smooth)) * win, smooth)
    ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel(f"{args.baseline} pose loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
