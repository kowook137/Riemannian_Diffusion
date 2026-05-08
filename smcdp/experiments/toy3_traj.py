"""Toy 3 — trajectory diffusion on the LEARNED manifold M_φ(z_e).

Combines Stage-1 learned self-model (frozen Δ_φ) with trajectory-level
Riemannian SGM (Idea_formulation §4.6, §6.3).  The full Idea pipeline:

  Stage 1  (toy3_stage1_selfmodel.py): learn Δ_φ from self-exploration data.
  Stage 5  (toy3_stage5_score.py):     score-net on learned M_φ, single point.
  Stage 5' (this script):              score-net on learned M_φ, TRAJECTORY level.

Each demo τ = (x_0, …, x_H) carries one z_e (frozen across the trajectory):
that matches the physical reality of a robot whose hardware is constant for
the duration of one demonstration.

What we check
  (i)   per-timestep adherence to the LEARNED manifold:  max_h |g_φ(x_h)| ≡ 0
  (ii)  per-timestep adherence to the TRUE manifold (oracle compliance):
            mean_h |p_h − p_true(q_h, z_e)|  — Stage-1 fit error in-distribution
  (iii) embodiment generalization across z_e (in / OOD)
  (iv)  trajectory-shape recovery (chart-line structure preserved by sampling)

Run:
    python -m smcdp.experiments.toy3_traj \
        --stage1-ckpt outputs/toy3_stage1/delta_phi.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from smcdp.sde import LangevinSDE, LinearBetaSchedule
from smcdp.toy3.ground_truth import TrueArmCompliance
from smcdp.toy3.self_model import DeltaResidualMLP, LearnedSelfModelArm
from smcdp.toy3.distributions import WrappedNormalEmbodiment
from smcdp.trajectories import (
    LinearChartTrajectoryDistEmb,
    TrajectoryScoreNet,
    TrajectoryScaledScoreFn,
    traj_reverse_grw,
    traj_dsm_varadhan_loss,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-ckpt", type=str,
                   default="outputs/toy3_stage1/delta_phi.pt")
    # trajectory horizon
    p.add_argument("--H", type=int, default=8)
    # net / training
    p.add_argument("--steps", type=int, default=15_000)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=5)
    p.add_argument("--activation", type=str, default="sin")
    p.add_argument("--time-embedding", type=str, default="raw")
    p.add_argument("--beta-0", type=float, default=0.001)
    p.add_argument("--beta-f", type=float, default=6.0)
    p.add_argument("--eps", type=float, default=2e-4)
    p.add_argument("--n-grw-steps", type=int, default=10)
    p.add_argument("--metric", type=str, default="riemannian",
                   choices=["riemannian", "chart_euclidean"])
    p.add_argument("--weight", type=str, default="sigma2",
                   choices=["sigma2", "beta", "none"])
    # data / limiting
    p.add_argument("--mean-q", type=float, nargs=2, default=[0.5, 0.6])
    p.add_argument("--scale-endpoint", type=float, default=0.3)
    p.add_argument("--limiting-scale", type=float, default=1.0)
    p.add_argument("--z-min", type=float, default=0.0)
    p.add_argument("--z-max", type=float, default=0.3)
    # eval
    p.add_argument("--n-sample-steps", type=int, default=200)
    p.add_argument("--n-eval", type=int, default=2048)
    p.add_argument("--z-eval", type=float, nargs="+",
                   default=[0.0, 0.15, 0.30, 0.45],
                   help="z_e values to sample at (last one is OOD if > z-max)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="outputs/toy3_traj")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out_dir={out_dir}  H+1={args.H + 1}  metric={args.metric}")

    # --- load Stage-1 self-model ---
    ck = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
    s1 = ck["args"]
    print(f"loaded Stage-1: improvement={ck['metrics']['improvement_factor']:.1f}x")
    delta_net = DeltaResidualMLP(
        n_q=2, n_p=2, n_z=1, hidden=s1["hidden"], n_layers=s1["n_layers"],
    ).to(device)
    delta_net.load_state_dict(ck["delta_net_state"])
    delta_net.eval()

    arm = LearnedSelfModelArm(
        delta_net=delta_net,
        l1=s1["l1"], l2_base=s1["l2_base"],
        metric=args.metric,
    )
    truth = TrueArmCompliance(
        l1=s1["l1"], l2_base=s1["l2_base"],
        K_grav=s1["K_grav"], K_offset=s1["K_offset"],
    )

    # --- SDE and distributions ---
    schedule = LinearBetaSchedule(beta_0=args.beta_0, beta_f=args.beta_f, t0=0.0, tf=1.0)
    limiting = WrappedNormalEmbodiment(
        arm, mean_q=args.mean_q, scale=args.limiting_scale,
        z_e_range=(args.z_min, args.z_max),
    )
    sde = LangevinSDE(arm, schedule, limiting)
    data = LinearChartTrajectoryDistEmb(
        arm, H=args.H,
        mean_q=args.mean_q, scale_endpoint=args.scale_endpoint,
        z_e_range=(args.z_min, args.z_max),
    )

    # --- score net + EMA ---
    def make_net():
        return TrajectoryScoreNet(
            arm, H=args.H,
            hidden=args.hidden, n_layers=args.n_layers,
            t_embed_dim=64, activation=args.activation,
            time_embedding=args.time_embedding,
            final_init_scale=1.0,
        ).to(device)

    net = make_net()
    ema_net = make_net()
    ema_net.load_state_dict(net.state_dict())
    for p in ema_net.parameters():
        p.requires_grad_(False)

    score_fn_train = TrajectoryScaledScoreFn(net, sde)
    score_fn_eval = TrajectoryScaledScoreFn(ema_net, sde)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr,
                             betas=(0.9, 0.999), eps=1e-8)
    def lr_lambda(step):
        if args.warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / args.warmup_steps)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    # --- training ---
    losses_log: list[float] = []
    pbar = tqdm(range(args.steps), desc=f"train traj/{args.metric}")
    for step in pbar:
        tau_0 = data.sample(args.batch, device=device)                      # (B, H+1, 5)
        loss = traj_dsm_varadhan_loss(score_fn_train, sde, tau_0,
                                      eps=args.eps, weight=args.weight,
                                      n_grw_steps=args.n_grw_steps)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        lr_sched.step()
        with torch.no_grad():
            for pe, pn in zip(ema_net.parameters(), net.parameters()):
                pe.mul_(args.ema).add_(pn, alpha=(1.0 - args.ema))
        losses_log.append(loss.item())
        if step % 200 == 0:
            pbar.set_postfix(loss=f"{loss.item():.3f}")

    # --- evaluation: sample at multiple z_e ---
    print("\nevaluating at z_e =", args.z_eval)
    H1 = args.H + 1
    n = args.n_eval
    metrics_per_z = {}

    fig, axes = plt.subplots(3, len(args.z_eval), figsize=(4.2 * len(args.z_eval), 11))
    if len(args.z_eval) == 1:
        axes = axes.reshape(3, 1)

    for i, z_val in enumerate(args.z_eval):
        z_tensor = torch.full((n, 1), z_val, device=device)
        # data trajectories at this z_e
        tau_data = data.sample(n, device=device, z_e=z_tensor)               # (n, H+1, 5)
        # limiting prior trajectories — each timestep iid from limiting at this z_e
        tau_T_flat = limiting.sample(n * H1, device=device,
                                     z_e=z_tensor.unsqueeze(1).expand(-1, H1, -1).reshape(n*H1, -1))
        tau_T = tau_T_flat.reshape(n, H1, 5)

        with torch.no_grad():
            tau_gen = traj_reverse_grw(sde, score_fn_eval, tau_T,
                                       n_steps=args.n_sample_steps, eps=args.eps)

        # --- adherence on learned manifold (by construction) ---
        g_learned = arm.constraint(tau_gen.reshape(-1, 5)).norm(dim=-1)
        # --- adherence on TRUE manifold (compares to physical compliance) ---
        with torch.no_grad():
            x_pts = tau_gen.reshape(-1, 5)
            q_pts = x_pts[..., :2]
            p_pts = x_pts[..., 2:4]
            z_pts = x_pts[..., 4:]
            p_true_at_q = truth.p_true(q_pts, z_pts)
        g_truth = (p_pts - p_true_at_q).norm(dim=-1)

        # --- per-h chart sliced-W₁ ---
        n_dir = 32
        dirs = torch.randn(n_dir, 2, device=device)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        w1_per_h = []
        for h in range(H1):
            q_g = tau_gen[:, h, :2]
            q_d = tau_data[:, h, :2]
            proj_g = (q_g @ dirs.T).sort(dim=0).values
            proj_d = (q_d @ dirs.T).sort(dim=0).values
            w1_per_h.append((proj_g - proj_d).abs().mean().item())
        w1_mean = sum(w1_per_h) / len(w1_per_h)

        metrics_per_z[z_val] = {
            "max_g_learned": g_learned.max().item(),
            "mean_g_learned": g_learned.mean().item(),
            "mean_g_truth": g_truth.mean().item(),
            "w1_mean": w1_mean,
            "w1_per_h": w1_per_h,
        }
        print(f"  z_e = {z_val:.2f} :  max|g_learned| = {g_learned.max().item():.2e}   "
              f"mean|g_truth| = {g_truth.mean().item():.4e}   sliced-W₁ = {w1_mean:.4f}")

        # --- plots: chart trajs, end-effector trajs, per-h W₁ ---
        n_show = 30
        ax = axes[0, i]
        for j in range(n_show):
            ax.plot(tau_data[j, :, 0].cpu(), tau_data[j, :, 1].cpu(),
                    color="C0", alpha=0.25, lw=1)
            ax.plot(tau_gen[j, :, 0].cpu(), tau_gen[j, :, 1].cpu(),
                    color="C1", alpha=0.25, lw=1)
        ax.set_aspect("equal")
        ax.set_title(f"z_e={z_val:.2f}  W₁={w1_mean:.3f}")
        ax.set_xlabel("q_1"); ax.set_ylabel("q_2")
        ax.grid(alpha=0.3)

        ax = axes[1, i]
        for j in range(n_show):
            ax.plot(tau_data[j, :, 2].cpu(), tau_data[j, :, 3].cpu(),
                    color="C0", alpha=0.25, lw=1)
            ax.plot(tau_gen[j, :, 2].cpu(), tau_gen[j, :, 3].cpu(),
                    color="C1", alpha=0.25, lw=1)
        ax.set_aspect("equal")
        ax.set_title(f"end-eff  |g_truth|={g_truth.mean().item():.1e}")
        ax.set_xlabel("p_x"); ax.set_ylabel("p_y")
        ax.grid(alpha=0.3)

        ax = axes[2, i]
        ax.plot(range(H1), w1_per_h, "o-")
        ax.set_xlabel("h"); ax.set_ylabel("sliced-W₁")
        ax.set_title("per-timestep W₁")
        ax.grid(alpha=0.3)

    fig.suptitle(f"Toy 3 trajectory — Riemannian SGM on learned M_φ(z_e), {args.metric}")
    fig.tight_layout()
    out = out_dir / f"toy3_traj_{args.metric}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"plots saved to {out}")

    torch.save({
        "args": vars(args),
        "stage1_args": s1,
        "net_state": net.state_dict(),
        "ema_state": ema_net.state_dict(),
        "metrics_per_z": metrics_per_z,
    }, out_dir / f"ckpt_{args.metric}.pt")


if __name__ == "__main__":
    main()
