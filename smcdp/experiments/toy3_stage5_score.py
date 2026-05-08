"""Toy 3 — Stage 5: train the score net on the learned manifold M_φ(z_e).

Uses the FROZEN Δ_φ from Stage 1 to define M_φ(z_e), then trains a
z_e-conditioned ChartScoreNet via DSM-Varadhan on the Langevin SDE driven by
the wrapped-Gaussian limiting (with proper log det G correction).

Sampling at user-specified z_e values lets us check:
  - in-distribution z_e (∈ training range): does the model recover the data?
  - out-of-distribution z_e: does the framework adapt the manifold (Δ_φ
    extrapolates) and still produce manifold-adherent samples?

Run:
  python -m smcdp.experiments.toy3_stage5_score \
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
from smcdp.sampling import reverse_grw
from smcdp.score_net import ChartScoreNet, ScaledScoreFn
from smcdp.losses import dsm_varadhan_loss
from smcdp.toy3.ground_truth import TrueArmCompliance
from smcdp.toy3.self_model import DeltaResidualMLP, LearnedSelfModelArm
from smcdp.toy3.distributions import (
    ChartGaussianOnEmbodiment,
    WrappedNormalEmbodiment,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-ckpt", type=str,
                   default="outputs/toy3_stage1/delta_phi.pt",
                   help="Stage-1 self-model checkpoint")
    # SDE & training
    p.add_argument("--steps", type=int, default=15_000)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--beta-0", type=float, default=0.001)
    p.add_argument("--beta-f", type=float, default=6.0)
    p.add_argument("--eps", type=float, default=2e-4)
    p.add_argument("--n-grw-steps", type=int, default=10)
    p.add_argument("--metric", type=str, default="riemannian",
                   choices=["riemannian", "chart_euclidean"])
    p.add_argument("--weight", type=str, default="sigma2",
                   choices=["sigma2", "beta", "none"])
    # network
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--activation", type=str, default="sin")
    p.add_argument("--time-embedding", type=str, default="raw")
    # data / limiting
    p.add_argument("--mean-q", type=float, nargs=2, default=[0.5, 0.6])
    p.add_argument("--scale-q", type=float, default=0.2)
    p.add_argument("--limiting-scale", type=float, default=1.0)
    p.add_argument("--z-min", type=float, default=0.0)
    p.add_argument("--z-max", type=float, default=0.3)
    # sampling
    p.add_argument("--n-sample-steps", type=int, default=200)
    p.add_argument("--n-eval", type=int, default=4096)
    p.add_argument("--z-eval", type=float, nargs="+",
                   default=[0.0, 0.15, 0.3, 0.45],
                   help="z_e values to sample at (last is OOD if > z-max)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="outputs/toy3_stage5")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out_dir={out_dir}  metric={args.metric}")

    # --- load Stage-1 checkpoint and build learned manifold ---
    ck = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
    s1 = ck["args"]
    print(f"loaded Stage-1: fit_err={ck['metrics']['p_err_val']:.2e}, "
          f"oracle_err={ck['metrics']['p_analytic_err_val']:.2e}, "
          f"improvement={ck['metrics']['improvement_factor']:.1f}x")

    delta_net = DeltaResidualMLP(
        n_q=2, n_p=2, n_z=1,
        hidden=s1["hidden"], n_layers=s1["n_layers"],
    ).to(device)
    delta_net.load_state_dict(ck["delta_net_state"])
    delta_net.eval()

    # LearnedSelfModelArm is a Manifold, not an nn.Module — only delta_net
    # carries device-bound parameters (already moved to `device` above).
    arm = LearnedSelfModelArm(
        delta_net=delta_net,
        l1=s1["l1"],
        l2_base=s1["l2_base"],
        metric=args.metric,
    )

    # ground-truth oracle (only used for evaluation, never seen during training)
    truth = TrueArmCompliance(
        l1=s1["l1"], l2_base=s1["l2_base"],
        K_grav=s1["K_grav"], K_offset=s1["K_offset"],
    )

    # --- SDE: Langevin with wrapped-Gauss limiting on the learned manifold ---
    schedule = LinearBetaSchedule(beta_0=args.beta_0, beta_f=args.beta_f, t0=0.0, tf=1.0)
    limiting = WrappedNormalEmbodiment(
        arm,
        mean_q=args.mean_q,
        scale=args.limiting_scale,
        z_e_range=(args.z_min, args.z_max),
    )
    sde = LangevinSDE(arm, schedule, limiting)

    data_dist = ChartGaussianOnEmbodiment(
        arm,
        mean_q=args.mean_q,
        scale_q=args.scale_q,
        z_e_range=(args.z_min, args.z_max),
    )

    # --- score net + EMA ---
    def make_net():
        return ChartScoreNet(
            arm,
            hidden=args.hidden,
            n_layers=args.n_layers,
            t_embed_dim=64,
            activation=args.activation,
            time_embedding=args.time_embedding,
            final_init_scale=1.0,
        ).to(device)

    net = make_net()
    ema_net = make_net()
    ema_net.load_state_dict(net.state_dict())
    for p in ema_net.parameters():
        p.requires_grad_(False)

    score_fn_train = ScaledScoreFn(net, sde)
    score_fn_eval = ScaledScoreFn(ema_net, sde)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr,
                             betas=(0.9, 0.999), eps=1e-8)
    def lr_lambda(step):
        if args.warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / args.warmup_steps)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    # --- training ---
    losses_log: list[float] = []
    pbar = tqdm(range(args.steps), desc=f"train score on M_φ ({args.metric})")
    for step in pbar:
        x_0 = data_dist.sample(args.batch, device=device)
        loss = dsm_varadhan_loss(score_fn_train, sde, x_0,
                                 eps=args.eps, weight=args.weight,
                                 target="varadhan",
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
    n_eval = args.n_eval
    fig, axes = plt.subplots(2, len(args.z_eval), figsize=(4.2 * len(args.z_eval), 8.5))
    if len(args.z_eval) == 1:
        axes = axes.reshape(2, 1)
    metrics_per_z = {}
    for i, z_val in enumerate(args.z_eval):
        z_tensor = torch.full((n_eval, 1), z_val, device=device)
        # sample x_T from limiting at this z_e (joint-fix z_e, var q)
        x_T = limiting.sample(n_eval, device=device, z_e=z_tensor)
        # reverse SDE
        x_gen = reverse_grw(sde, score_fn_eval, x_T,
                            n_steps=args.n_sample_steps, eps=args.eps)
        # data at this z_e (using LEARNED manifold to define x_data — i.e. data
        # at z_val using the learned F_φ; this is what the model is trained to recover)
        x_data = data_dist.sample(n_eval, device=device, z_e=z_tensor)

        # adherence on LEARNED manifold (should be machine-zero since exp is closed)
        g_learned = arm.constraint(x_gen).norm(dim=-1).max().item()
        # adherence on TRUE manifold (the actual physical surface)
        with torch.no_grad():
            q_gen = x_gen[..., :2]
            p_gen = x_gen[..., 2:4]
            z_gen = x_gen[..., 4:]
            p_true_at_q = truth.p_true(q_gen, z_gen)
            g_truth = (p_gen - p_true_at_q).norm(dim=-1).mean().item()

        # chart sliced-W₁ vs data
        q_g = q_gen.cpu()
        q_d = x_data[..., :2].cpu()
        dirs = torch.randn(64, 2)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        proj_g = q_g @ dirs.T
        proj_d = q_d @ dirs.T
        proj_g, _ = torch.sort(proj_g, dim=0)
        proj_d, _ = torch.sort(proj_d, dim=0)
        sliced_w1 = (proj_g - proj_d).abs().mean().item()

        metrics_per_z[z_val] = {
            "max_g_learned": g_learned,
            "mean_g_truth": g_truth,
            "sliced_w1": sliced_w1,
        }
        print(f"  z_e = {z_val:.2f} :  max|g_learned| = {g_learned:.2e}   "
              f"mean|g_truth| = {g_truth:.4e}   sliced-W₁(chart) = {sliced_w1:.4f}")

        # plots
        ax = axes[0, i]
        ax.scatter(*q_d.numpy().T, s=4, alpha=0.4, label="data", color="C0")
        ax.scatter(*q_g.numpy().T, s=4, alpha=0.4, label="model", color="C1")
        ax.set_aspect("equal")
        ax.set_title(f"chart  (z_e={z_val:.2f},  W₁={sliced_w1:.3f})")
        ax.set_xlabel("q_1"); ax.set_ylabel("q_2")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1, i]
        p_g = p_gen.cpu().numpy()
        p_d = x_data[..., 2:4].cpu().numpy()
        ax.scatter(*p_d.T, s=4, alpha=0.4, label="data", color="C0")
        ax.scatter(*p_g.T, s=4, alpha=0.4, label="model", color="C1")
        ax.set_aspect("equal")
        ax.set_title(f"end-eff  (mean|g_truth|={g_truth:.1e})")
        ax.set_xlabel("p_x"); ax.set_ylabel("p_y")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Toy 3 Stage 5 — score on learned M_φ(z_e)  ({args.metric})")
    fig.tight_layout()
    fig.savefig(out_dir / f"stage5_score_{args.metric}.png", dpi=120)
    plt.close(fig)

    # loss curve
    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 4))
    losses_t = torch.tensor(losses_log)
    win = max(1, len(losses_t) // 200)
    smooth = torch.nn.functional.avg_pool1d(
        losses_t.view(1, 1, -1), kernel_size=win, stride=win
    ).flatten()
    xs = torch.arange(len(smooth)) * win
    ax2.plot(xs, smooth)
    ax2.set_yscale("log")
    ax2.set_xlabel("step")
    ax2.set_ylabel("DSM loss")
    ax2.set_title(f"Stage 5 training loss ({args.metric})")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out_dir / f"stage5_loss_{args.metric}.png", dpi=120)
    plt.close(fig2)

    torch.save({
        "args": vars(args),
        "stage1_args": s1,
        "net_state": net.state_dict(),
        "ema_state": ema_net.state_dict(),
        "metrics_per_z": metrics_per_z,
    }, out_dir / f"stage5_ckpt_{args.metric}.pt")
    print(f"\ncheckpoint saved to  {out_dir / f'stage5_ckpt_{args.metric}.pt'}")


if __name__ == "__main__":
    main()
