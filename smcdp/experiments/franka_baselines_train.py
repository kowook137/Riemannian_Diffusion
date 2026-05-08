"""Franka 7-DoF baselines (BC / DP-official / Projected) — unified trainer.

Same setup as Ours-V2:
  - Manifold: LearnedSelfModelFranka7DoF (Stage-1 Δ_φ frozen)
  - Demo data: FrankaBimodalReachingDemo (bimodal IK, p_box, z_e ∈ [0.05, 0.15])
  - H+1 = 16, batch = 64, steps = 15000
  - Context: c = (p_target, p_start, z_e) ∈ R^7
  - Eval: same z_e values incl. OOD, same multi-radius success metrics

Each baseline outputs a checkpoint at outputs/franka_baseline_{name}/ckpt.pt.
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

from smcdp.manifolds import Franka7DoF
from smcdp.toy3.self_model import DeltaResidualMLP
from smcdp.franka.self_model import LearnedSelfModelFranka7DoF
from smcdp.franka.demo_gen import FrankaBimodalReachingDemo
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
    p.add_argument("--stage1-ckpt", type=str, default="outputs/franka_stage1/delta_phi.pt")
    p.add_argument("--H", type=int, default=15)
    # demo (match V2)
    p.add_argument("--q-rest-A", type=float, nargs=7,
                   default=[+0.6, -0.3, 0.0, -1.7, 0.0, 1.4, 0.0])
    p.add_argument("--q-rest-B", type=float, nargs=7,
                   default=[-0.6, -0.3, 0.0, -1.7, 0.0, 1.4, 0.0])
    p.add_argument("--p-box-lo", type=float, nargs=3, default=[0.40, -0.05, 0.40])
    p.add_argument("--p-box-hi", type=float, nargs=3, default=[0.50, +0.05, 0.50])
    p.add_argument("--branch-p-A", type=float, default=0.5)
    p.add_argument("--jitter-q", type=float, default=0.05)
    p.add_argument("--n-ik-steps", type=int, default=12)
    p.add_argument("--z-min", type=float, default=0.05)
    p.add_argument("--z-max", type=float, default=0.15)
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
                   help="DP variant: 'global' (canonical Chi23) or 'channel' (DP-A, architecture parity with Ours-V2)")
    p.add_argument("--perfect-fk", action="store_true",
                   help="Experiment_plan §2.2 Regime A: use analytic-only Franka7DoF as ground truth manifold (no Δ_φ).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default=None,
                   help="Default: outputs/franka_baseline_{baseline}{_cond_injection}")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    if args.out_dir is None:
        suffix = f"_{args.cond_injection}" if args.cond_injection != "global" else ""
        if args.perfect_fk:
            suffix = suffix + "_perfect"
        args.out_dir = f"outputs/franka_baseline_{args.baseline}{suffix}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"baseline={args.baseline}  device={device}  out_dir={out_dir}")

    # ---- Manifold setup ----
    arm_a = Franka7DoF(urdf_path=URDF, end_link="panda_hand", tool_z_max=0.25)
    arm_a._ensure_chain(torch.zeros(1, 7, device=device))
    if args.perfect_fk:
        # Perfect FK regime: ground truth = analytic FK, no learned residual
        ck = None
        s1 = {"hidden": 128, "n_layers": 3}   # placeholder, not used
        arm = arm_a
        print(f"  PERFECT FK regime: arm = Franka7DoF (analytic only, no Δ_φ)")
    else:
        # Standard regime: ground truth = analytic + learned Δ_φ (Stage 1)
        ck = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
        s1 = ck["args"]
        delta_net = DeltaResidualMLP(n_q=7, n_p=3, n_z=1,
                                      hidden=s1["hidden"], n_layers=s1["n_layers"],
                                      activation=torch.nn.Softplus,
                                      final_init_scale=1e-3).to(device)
        delta_net.load_state_dict(ck["delta_net_state"]); delta_net.eval()
        arm = LearnedSelfModelFranka7DoF(
            delta_net=delta_net, urdf_path=URDF, end_link="panda_hand", tool_z_max=0.25,
        )
        arm._ensure_chain(torch.zeros(1, 7, device=device))

    # ---- Demo distribution ----
    data = FrankaBimodalReachingDemo(
        manifold=arm, ik_arm=arm_a, H=args.H,
        q_rest_A=list(args.q_rest_A), q_rest_B=list(args.q_rest_B),
        p_box_lo=tuple(args.p_box_lo), p_box_hi=tuple(args.p_box_hi),
        z_e_range=(args.z_min, args.z_max),
        branch_p_A=args.branch_p_A, jitter_q=args.jitter_q,
        n_ik_steps=args.n_ik_steps,
    )

    H1 = args.H + 1
    n_q, n_p, n_z = 7, 3, 1
    CTX_DIM = 3 + 3 + 1                 # p_target + p_start + z_e

    # ---- Build model per baseline ----
    if args.baseline == "bc":
        model = BCTrajectoryPredictor(
            n_q=n_q, H=args.H, ctx_dim=CTX_DIM,
            hidden=args.bc_hidden, n_layers=args.bc_layers,
            activation="sin",
        ).to(device)
        scheduler = None
    elif args.baseline == "dp_official":
        if args.cond_injection == "channel":
            # DP-A: channel-concat cond (architecture parity with Ours-V2)
            # input_dim = n_q + ctx_dim, output_dim = same (we extract n_q for ε)
            model, scheduler = make_official_diffusion_policy(
                n_q=n_q + CTX_DIM, global_cond_dim=None,
                down_dims=list(args.down_dims),
                diffusion_step_embed_dim=args.diff_step_embed,
                n_train_timesteps=args.dp_train_timesteps,
            )
        else:
            # canonical Chi23: global_cond
            model, scheduler = make_official_diffusion_policy(
                n_q=n_q, global_cond_dim=CTX_DIM,
                down_dims=list(args.down_dims),
                diffusion_step_embed_dim=args.diff_step_embed,
                n_train_timesteps=args.dp_train_timesteps,
            )
        model = model.to(device)
    elif args.baseline == "projected":
        # ambient (q, p) state; we diffuse on (q, p) jointly.  Wrap official
        # ConditionalUnet1D with input_dim = n_q + n_p = 10.
        model, scheduler = make_official_diffusion_policy(
            n_q=n_q + n_p, global_cond_dim=CTX_DIM,
            down_dims=list(args.down_dims),
            diffusion_step_embed_dim=args.diff_step_embed,
            n_train_timesteps=args.dp_train_timesteps,
        )
        model = model.to(device)
    else:
        raise ValueError(args.baseline)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params/1e6:.2f}M")

    # ---- EMA ----
    ema_model = type(model)(*[]) if False else None
    # simple: keep a state_dict-based EMA
    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # ---- Optimizer ----
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    def lr_lambda(step):
        if args.warmup_steps <= 0: return 1.0
        return min(1.0, (step + 1) / args.warmup_steps)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    # ---- Training loop ----
    losses = []
    pbar = tqdm(range(args.steps), desc=f"train {args.baseline}")
    for step in pbar:
        x_demo, _, z_e, p_target, p_start = data.sample(args.batch, device=device)
        ctx = torch.cat([p_target, p_start, z_e], dim=-1)           # (B, 7)
        q_demo = x_demo[..., :n_q]                                   # (B, H+1, 7)

        if args.baseline == "bc":
            loss = bc_loss(model, ctx, q_demo)
        elif args.baseline == "dp_official":
            if args.cond_injection == "channel":
                ctx_per_h = ctx.unsqueeze(1).expand(-1, H1, -1)        # (B, H+1, 7)
                loss = channel_concat_dp_loss(model, scheduler, q_demo, ctx_per_h)
            else:
                loss = official_dp_loss(model, scheduler, q_demo, ctx)
        elif args.baseline == "projected":
            x_amb = x_demo[..., :n_q + n_p]                          # (B, H+1, 10)
            loss = official_dp_loss(model, scheduler, x_amb, ctx)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        lr_sched.step()
        # EMA update
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
        "stage1_args": s1,
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
    smooth = torch.nn.functional.avg_pool1d(losses_t.view(1, 1, -1), kernel_size=win, stride=win).flatten()
    ax.plot(torch.arange(len(smooth)) * win, smooth)
    ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel(f"{args.baseline} training loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
