"""Ours-Analytic — V2 framework with arm = Franka7DoF (no learned Δ_φ).

Per Experiment_plan §2.2 Regime A (Perfect FK):
  F_φ = F_analytic                            (no residual)
  M_ana(z_e) = {(q, p) : p = F_analytic(q, z_e)}
  G_ana = I + J_F_analytic^T J_F_analytic     (Riemannian metric still defined)

All other components identical to V2 (channel-cond + p_start cond + multi-component
analytic guidance + lift_chart_to_tangent + DSM-Varadhan).

This isolates Claim A (Riemannian framework) from Claim B (learned residual).

Run:
    python -m smcdp.experiments.franka_v2_analytic_train
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm
import pybullet_data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from smcdp.manifolds import Franka7DoF
from smcdp.sde import LangevinSDE, LinearBetaSchedule
from smcdp.franka.distributions import WrappedNormalFranka7DoF
from smcdp.franka.demo_gen import FrankaBimodalReachingDemo
from smcdp.trajectories import (
    TrajectoryScoreNetUNet,
    TrajectoryScaledScoreFn,
    traj_dsm_varadhan_loss,
    traj_reverse_grw,
)


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--H", type=int, default=15)
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
    # net + training
    p.add_argument("--steps", type=int, default=15_000)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--down-dims", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--diff-step-embed", type=int, default=128)
    p.add_argument("--unet-groups", type=int, default=8)
    p.add_argument("--unet-kernel", type=int, default=3)
    p.add_argument("--t-scale", type=float, default=1000.0)
    p.add_argument("--beta-0", type=float, default=0.001)
    p.add_argument("--beta-f", type=float, default=4.0)
    p.add_argument("--eps", type=float, default=2e-4)
    p.add_argument("--n-grw-steps", type=int, default=10)
    p.add_argument("--n-sample-steps", type=int, default=200)
    p.add_argument("--metric", type=str, default="riemannian",
                   choices=["riemannian", "chart_euclidean"])
    p.add_argument("--weight", type=str, default="sigma2",
                   choices=["sigma2", "beta", "none"])
    p.add_argument("--limiting-mean-q", type=float, nargs=7,
                   default=[0.0, -0.3, 0.0, -1.7, 0.0, 1.4, 0.0])
    p.add_argument("--limiting-scale", type=float, default=0.6)
    # CFG / guidance
    p.add_argument("--cond-drop-prob", type=float, default=0.10)
    p.add_argument("--guidance-scale", type=float, default=0.0)
    p.add_argument("--cond-injection", type=str, default="channel",
                   choices=["global", "channel"])
    p.add_argument("--endpoint-weight", type=float, default=1.0)
    p.add_argument("--use-p-start-cond", action="store_true", default=True)
    # eval-time guidance
    p.add_argument("--goal-residual-alpha", type=float, default=100.0)
    p.add_argument("--goal-residual-h-mode", type=str, default="last_half")
    p.add_argument("--start-anchor-alpha", type=float, default=100.0)
    p.add_argument("--smoothness-alpha-vel", type=float, default=5.0)
    p.add_argument("--smoothness-alpha-acc", type=float, default=0.0)
    # eval
    p.add_argument("--n-eval-per-z", type=int, default=256)
    p.add_argument("--z-eval", type=float, nargs="+",
                   default=[0.05, 0.10, 0.15, 0.20])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="outputs/franka_traj_unet_v2_analytic")
    p.add_argument("--resume-from", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ours-Analytic (Perfect FK regime, no learned residual)")
    print(f"  device={device}  out_dir={out_dir}  metric={args.metric}  H+1={args.H+1}")
    print(f"  cond_injection={args.cond_injection}  use_p_start={args.use_p_start_cond}")

    # ---- Manifold: analytic ONLY (no Δ_φ) ----
    arm = Franka7DoF(urdf_path=URDF, end_link="panda_hand",
                     tool_z_max=max(args.z_max, args.z_eval[-1]) + 0.05,
                     metric=args.metric)
    arm._ensure_chain(torch.zeros(1, 7, device=device))
    # Demo generation arm = same analytic arm (Perfect FK regime: ground truth = analytic)
    arm_analytic = arm

    # ---- SDE ----
    schedule = LinearBetaSchedule(beta_0=args.beta_0, beta_f=args.beta_f, t0=0.0, tf=1.0)
    limiting = WrappedNormalFranka7DoF(
        arm, mean_q=list(args.limiting_mean_q), scale=args.limiting_scale,
        z_e_range=(args.z_min, args.z_max),
    )
    sde = LangevinSDE(arm, schedule, limiting)

    # ---- Demo distribution: Perfect FK (manifold = analytic, no compliance) ----
    data = FrankaBimodalReachingDemo(
        manifold=arm, ik_arm=arm_analytic, H=args.H,
        q_rest_A=list(args.q_rest_A), q_rest_B=list(args.q_rest_B),
        p_box_lo=tuple(args.p_box_lo), p_box_hi=tuple(args.p_box_hi),
        z_e_range=(args.z_min, args.z_max),
        branch_p_A=args.branch_p_A, jitter_q=args.jitter_q,
        n_ik_steps=args.n_ik_steps,
    )

    # ---- Score net (V2 architecture, channel cond + p_start cond) ----
    GOAL_DIM = 6 if args.use_p_start_cond else 3
    def make_net():
        return TrajectoryScoreNetUNet(
            arm, H=args.H,
            down_dims=tuple(args.down_dims),
            diffusion_step_embed_dim=args.diff_step_embed,
            n_groups=args.unet_groups,
            kernel_size=args.unet_kernel,
            t_scale=args.t_scale,
            goal_cond_dim=GOAL_DIM,
            cond_injection=args.cond_injection,
        ).to(device)

    net = make_net()
    ema_net = make_net()
    ema_net.load_state_dict(net.state_dict())
    for p in ema_net.parameters():
        p.requires_grad_(False)
    print(f"UNet params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")

    score_fn_train = TrajectoryScaledScoreFn(net, sde)
    score_fn_eval = TrajectoryScaledScoreFn(ema_net, sde)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr,
                             betas=(0.9, 0.999), eps=1e-8)
    def lr_lambda(step):
        if args.warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / args.warmup_steps)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    if args.resume_from is not None:
        rck = torch.load(args.resume_from, map_location=device, weights_only=False)
        net.load_state_dict(rck["net_state"])
        ema_net.load_state_dict(rck["ema_state"])
        if "optim_state" in rck:
            optim.load_state_dict(rck["optim_state"])
        lr_sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda s: 1.0)
        print(f"resumed from {args.resume_from}")

    # ---- Training ----
    losses = []
    pbar = tqdm(range(args.steps), desc=f"train Ours-Analytic/{args.metric}")
    for step in pbar:
        x_data, _, _, p_target, p_start = data.sample(args.batch, device=device)
        if args.use_p_start_cond:
            cond_for_net = torch.cat([p_target, p_start], dim=-1)
        else:
            cond_for_net = p_target
        loss = traj_dsm_varadhan_loss(
            score_fn_train, sde, x_data,
            eps=args.eps, weight=args.weight,
            n_grw_steps=args.n_grw_steps,
            goal_cond=cond_for_net,
            cond_drop_prob=args.cond_drop_prob,
            endpoint_weight=args.endpoint_weight,
        )
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        lr_sched.step()
        with torch.no_grad():
            for pe, pn in zip(ema_net.parameters(), net.parameters()):
                pe.mul_(args.ema).add_(pn, alpha=1.0 - args.ema)
        losses.append(loss.item())
        if step % 200 == 0:
            pbar.set_postfix(loss=f"{loss.item():.3f}")

    # ---- Eval ----
    print("\nEvaluating Ours-Analytic at z_e =", args.z_eval)
    H1 = args.H + 1
    n = args.n_eval_per_z
    d = arm.ambient_dim
    h_mid = H1 // 2
    metrics_per_z = {}

    if args.goal_residual_h_mode == "last_only":
        grh = [H1 - 1]
    elif args.goal_residual_h_mode == "last_quarter":
        grh = list(range(H1 - H1 // 4, H1))
    elif args.goal_residual_h_mode == "last_half":
        grh = list(range(H1 - H1 // 2, H1))
    else:
        grh = list(range(H1))
    sah = [0]

    fig, axes = plt.subplots(2, len(args.z_eval), figsize=(4.0 * len(args.z_eval), 8))
    if len(args.z_eval) == 1:
        axes = axes.reshape(2, 1)

    for i, z_val in enumerate(args.z_eval):
        torch.manual_seed(args.seed + 1000 + i)
        box_lo = torch.tensor(args.p_box_lo, device=device)
        box_hi = torch.tensor(args.p_box_hi, device=device)
        p_targets = box_lo + (box_hi - box_lo) * torch.rand(n, 3, device=device)
        torch.manual_seed(args.seed + 2000 + i)
        p_starts = box_lo + (box_hi - box_lo) * torch.rand(n, 3, device=device)

        data_z = FrankaBimodalReachingDemo(
            manifold=arm, ik_arm=arm_analytic, H=args.H,
            q_rest_A=list(args.q_rest_A), q_rest_B=list(args.q_rest_B),
            p_box_lo=tuple(args.p_box_lo), p_box_hi=tuple(args.p_box_hi),
            z_e_range=(z_val, z_val),
            branch_p_A=args.branch_p_A, jitter_q=args.jitter_q,
            n_ik_steps=args.n_ik_steps,
        )
        x_data, branch_A_data, _, _, p_starts_data = data_z.sample(
            n, device=device, p_target=p_targets, p_start=p_starts,
        )
        if args.use_p_start_cond:
            cond_for_eval = torch.cat([p_targets, p_starts], dim=-1)
        else:
            cond_for_eval = p_targets

        z_tensor = torch.full((n, 1), z_val, device=device)
        z_lim = z_tensor.unsqueeze(1).expand(-1, H1, -1).reshape(n * H1, -1)
        torch.manual_seed(args.seed + 3000 + i)
        tau_T = limiting.sample(n * H1, device=device, z_e=z_lim).reshape(n, H1, d)

        with torch.no_grad():
            tau_gen = traj_reverse_grw(
                sde, score_fn_eval, tau_T,
                n_steps=args.n_sample_steps, eps=args.eps,
                goal_cond=cond_for_eval, guidance_scale=args.guidance_scale,
                goal_residual_alpha=args.goal_residual_alpha, goal_residual_h=grh,
                p_start=p_starts, start_anchor_alpha=args.start_anchor_alpha, start_anchor_h=sah,
                smoothness_alpha_vel=args.smoothness_alpha_vel,
                smoothness_alpha_acc=args.smoothness_alpha_acc,
            )

        g = arm.constraint(tau_gen.reshape(-1, d)).norm(dim=-1)
        max_g = g.max().item()

        p_end = tau_gen[:, -1, 7:10]
        pos_err = (p_end - p_targets).norm(dim=-1)
        succ_2 = (pos_err < 0.02).float().mean().item()
        succ_5 = (pos_err < 0.05).float().mean().item()
        succ_10 = (pos_err < 0.10).float().mean().item()

        q1_mid_data = x_data[:, h_mid, 0]
        q1_mid_gen = tau_gen[:, h_mid, 0]
        mode_A_data = q1_mid_data > 0
        mode_A_gen = q1_mid_gen > 0
        frac_A_data = mode_A_data.float().mean().item()
        frac_A_gen = mode_A_gen.float().mean().item()

        n_dir = 64
        dirs = torch.randn(n_dir, 7, device=device)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        def per_mode_w1(tau_a, tau_b):
            m = min(tau_a.shape[0], tau_b.shape[0])
            if m < 8:
                return float("nan")
            ia = torch.randperm(tau_a.shape[0], device=device)[:m]
            ib = torch.randperm(tau_b.shape[0], device=device)[:m]
            a = tau_a[ia, :, :7]; b = tau_b[ib, :, :7]
            ws = []
            for h in range(H1):
                pa = (a[:, h] @ dirs.T).sort(dim=0).values
                pb = (b[:, h] @ dirs.T).sort(dim=0).values
                ws.append((pa - pb).abs().mean().item())
            return sum(ws) / len(ws)
        w1_A = per_mode_w1(tau_gen[mode_A_gen], x_data[mode_A_data])
        w1_B = per_mode_w1(tau_gen[~mode_A_gen], x_data[~mode_A_data])

        viol = arm.violates_limits(tau_gen.reshape(-1, d)[..., :7]).float().mean().item()
        vel = (tau_gen[:, 1:, :7] - tau_gen[:, :-1, :7]).norm(dim=-1).mean().item()

        metrics_per_z[z_val] = {
            "max_g": max_g,
            "pos_err_mean": pos_err.mean().item(),
            "pos_err_med": pos_err.median().item(),
            "pos_err_std": pos_err.std().item(),
            "succ_2cm": succ_2, "succ_5cm": succ_5, "succ_10cm": succ_10,
            "frac_A_data": frac_A_data, "frac_A_gen": frac_A_gen,
            "W1_A": w1_A, "W1_B": w1_B,
            "vel_mean": vel, "viol": viol,
        }
        ood = " (OOD)" if (z_val < args.z_min or z_val > args.z_max) else ""
        print(f"  z_e={z_val:.3f}{ood}: max|g|={max_g:.1e}  pos_err={pos_err.mean():.4f}m  "
              f"succ@5cm={succ_5*100:.1f}%  succ@10cm={succ_10*100:.1f}%  "
              f"frac_A={frac_A_gen:.3f}  W₁={w1_A:.3f}/{w1_B:.3f}  vel={vel:.3f}  viol={viol*100:.1f}%")

        ax = axes[0, i]
        ax.hist(q1_mid_data.cpu().numpy(), bins=60, density=True, alpha=0.5,
                label="data", color="C0")
        ax.hist(q1_mid_gen.cpu().numpy(), bins=60, density=True, alpha=0.5,
                label="model", color="C1")
        ax.axvline(0.0, color="k", lw=0.5, alpha=0.4)
        ax.set_xlabel("q_1 mid"); ax.set_title(f"z_e={z_val:.3f}{ood}")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[1, i]
        ax.scatter(p_targets[:, 0].cpu(), p_targets[:, 2].cpu(), alpha=0.5, s=12,
                   label="target", color="k", marker="x")
        ax.scatter(p_end[:, 0].cpu(), p_end[:, 2].cpu(), alpha=0.5, s=8,
                   label="gen", color="C1")
        ax.set_xlabel("p_x"); ax.set_ylabel("p_z")
        ax.set_title(f"end-EE  pos_err={pos_err.mean():.4f}\nsucc@5cm={succ_5*100:.1f}%")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        ax.set_aspect("equal")

    fig.suptitle(f"Ours-Analytic (Perfect FK, V2 architecture)  ({args.metric})")
    fig.tight_layout()
    fig.savefig(out_dir / f"analytic_{args.metric}.png", dpi=120)
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(7, 4))
    losses_t = torch.tensor(losses)
    win = max(1, len(losses_t) // 200)
    smooth = torch.nn.functional.avg_pool1d(losses_t.view(1, 1, -1), kernel_size=win, stride=win).flatten()
    ax.plot(torch.arange(len(smooth)) * win, smooth)
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("traj DSM loss")
    ax.set_title(f"Ours-Analytic training loss ({args.metric})")
    ax.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out_dir / f"analytic_loss_{args.metric}.png", dpi=120)
    plt.close(fig2)

    torch.save({
        "args": vars(args),
        "net_state": net.state_dict(),
        "ema_state": ema_net.state_dict(),
        "optim_state": optim.state_dict(),
        "metrics_per_z": metrics_per_z,
    }, out_dir / f"ckpt_{args.metric}.pt")
    print(f"\nsaved {out_dir / f'ckpt_{args.metric}.pt'}")


if __name__ == "__main__":
    main()
