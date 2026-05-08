"""Toy 3.5 (single-mode) — trajectory diffusion on a 3-link redundant arm.

Step B in the framework-validation roadmap: scale the substrate from
n_q = 2 (Toy 2/3) to n_q = 3 (1-DoF kinematic redundancy), with a still-
unimodal demonstration distribution.  The redundancy IS present
geometrically (J_F has null space dim 1, so multiple q's reach the same p),
but the demo we feed in stays single-mode by sampling each q-endpoint
from one chart Gaussian — i.e. only one IK branch is shown.  Step C
will introduce multi-modal demos that exploit the null space.

What we check
  (i)   substrate scales:  same code paths, n_q = 3, max_h |g_φ(x_h)| ≡ 0
  (ii)  per-timestep marginal recovery in 3D chart (sliced-W₁)
  (iii) trajectory shape recovery (chart linear-interp preserved)

Run:
    python -m smcdp.experiments.toy3p5_traj_unimodal
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D    # noqa: F401  (registers 3d projection)

from smcdp.manifolds import NLinkPlanarArm
from smcdp.sde import LangevinSDE, LinearBetaSchedule
from smcdp.distributions import WrappedNormalGraph
from smcdp.trajectories import (
    LinearChartTrajectoryDist,
    TrajectoryScoreNet,
    TrajectoryScaledScoreFn,
    traj_reverse_grw,
    traj_dsm_varadhan_loss,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--link-lengths", type=float, nargs="+",
                   default=[1.0, 1.0, 0.5],
                   help="link lengths l_1, ..., l_N (N ≥ 2). default: 3-link [1, 1, 0.5]")
    p.add_argument("--H", type=int, default=8)
    # network/schedule (RSGM-style)
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
    p.add_argument("--n-sample-steps", type=int, default=200)
    p.add_argument("--metric", type=str, default="riemannian",
                   choices=["riemannian", "chart_euclidean"])
    p.add_argument("--weight", type=str, default="sigma2",
                   choices=["sigma2", "beta", "none"])
    # data / limiting
    p.add_argument("--mean-q", type=float, nargs="+", default=None,
                   help="data chart-Gaussian mean (length n_q); default zeros")
    p.add_argument("--scale-endpoint", type=float, default=0.3)
    p.add_argument("--limiting-mean-q", type=float, nargs="+", default=None,
                   help="limiting wrapped-Gauss mean (default: same as mean-q)")
    p.add_argument("--limiting-scale", type=float, default=1.0)
    p.add_argument("--n-eval", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="outputs/toy3p5_traj_uni")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_q = len(args.link_lengths)
    print(f"device={device}  out_dir={out_dir}  n_q={n_q}  H+1={args.H+1}  metric={args.metric}")

    # ---- problem setup ----
    arm = NLinkPlanarArm(args.link_lengths, metric=args.metric)
    schedule = LinearBetaSchedule(beta_0=args.beta_0, beta_f=args.beta_f, t0=0.0, tf=1.0)
    mean_q = args.mean_q if args.mean_q is not None else [0.0] * n_q
    assert len(mean_q) == n_q, f"--mean-q must have length n_q = {n_q}"
    lim_mean_q = args.limiting_mean_q if args.limiting_mean_q is not None else mean_q
    assert len(lim_mean_q) == n_q

    limiting = WrappedNormalGraph(arm, mean_q=lim_mean_q, scale=args.limiting_scale)
    sde = LangevinSDE(arm, schedule, limiting)
    data = LinearChartTrajectoryDist(arm, H=args.H, mean_q=mean_q,
                                     scale_endpoint=args.scale_endpoint)

    # ---- model + EMA ----
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

    # ---- training ----
    losses_log: list[float] = []
    pbar = tqdm(range(args.steps), desc=f"train traj/{args.metric} (N={n_q})")
    for step in pbar:
        tau_0 = data.sample(args.batch, device=device)
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

    # ---- sampling ----
    print("\nsampling reverse trajectories ...")
    H1 = args.H + 1
    n = args.n_eval
    d = arm.ambient_dim
    tau_T = limiting.sample(n * H1, device=device).reshape(n, H1, d)
    with torch.no_grad():
        tau_gen = traj_reverse_grw(sde, score_fn_eval, tau_T,
                                   n_steps=args.n_sample_steps, eps=args.eps)
    tau_data = data.sample(n, device=device)

    # ---- adherence ----
    g_per_pt = arm.constraint(tau_gen.reshape(-1, d)).norm(dim=-1)
    print(f"per-timestep adherence:  mean|g| = {g_per_pt.mean():.2e}   "
          f"max|g| = {g_per_pt.max():.2e}")

    # ---- per-timestep sliced-W₁ on chart (R^{n_q}) ----
    n_dir = 64
    dirs = torch.randn(n_dir, n_q, device=device)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    w1_per_h = []
    for h in range(H1):
        q_g = tau_gen[:, h, :n_q]
        q_d = tau_data[:, h, :n_q]
        proj_g = (q_g @ dirs.T).sort(dim=0).values
        proj_d = (q_d @ dirs.T).sort(dim=0).values
        w1_per_h.append((proj_g - proj_d).abs().mean().item())
    w1_mean = sum(w1_per_h) / len(w1_per_h)
    print(f"per-timestep sliced-W₁ : mean = {w1_mean:.4f}   "
          f"per h = {[f'{x:.3f}' for x in w1_per_h]}")

    # ---- plots ----
    fig = plt.figure(figsize=(16, 10))

    # (a) loss curve
    ax = fig.add_subplot(2, 3, 1)
    losses_t = torch.tensor(losses_log)
    win = max(1, len(losses_t) // 200)
    smooth = torch.nn.functional.avg_pool1d(
        losses_t.view(1, 1, -1), kernel_size=win, stride=win
    ).flatten()
    ax.plot(torch.arange(len(smooth)) * win, smooth)
    ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel("traj DSM loss")
    ax.set_title(f"loss  (N={n_q}, H+1={H1})")
    ax.grid(alpha=0.3)

    # (b) per-timestep W₁
    ax = fig.add_subplot(2, 3, 2)
    ax.plot(range(H1), w1_per_h, "o-")
    ax.set_xlabel("h"); ax.set_ylabel("sliced-W₁ (chart)")
    ax.set_title("per-timestep distribution match")
    ax.grid(alpha=0.3)

    # (c) end-effector trajectories (data + model)
    ax = fig.add_subplot(2, 3, 3)
    n_show = 50
    p_idx = n_q                                        # p starts at index n_q
    for i in range(n_show):
        ax.plot(tau_data[i, :, p_idx].cpu(),     tau_data[i, :, p_idx + 1].cpu(),
                color="C0", alpha=0.3, lw=1)
        ax.plot(tau_gen[i, :, p_idx].cpu(),      tau_gen[i, :, p_idx + 1].cpu(),
                color="C1", alpha=0.3, lw=1)
    ax.set_aspect("equal")
    ax.set_title(f"end-eff trajs  (max|g|={g_per_pt.max():.0e})")
    ax.set_xlabel("p_x"); ax.set_ylabel("p_y")
    ax.grid(alpha=0.3)

    # (d, e) chart trajectories — for n_q ≥ 3 use the first 3 dims as 3D plot
    n_show_chart = 40
    if n_q >= 3:
        ax = fig.add_subplot(2, 3, 4, projection="3d")
        for i in range(n_show_chart):
            ax.plot(tau_data[i, :, 0].cpu(), tau_data[i, :, 1].cpu(), tau_data[i, :, 2].cpu(),
                    color="C0", alpha=0.3, lw=1)
        ax.set_title(f"data trajs (chart 3D)")
        ax.set_xlabel("q_1"); ax.set_ylabel("q_2"); ax.set_zlabel("q_3")

        ax = fig.add_subplot(2, 3, 5, projection="3d")
        for i in range(n_show_chart):
            ax.plot(tau_gen[i, :, 0].cpu(), tau_gen[i, :, 1].cpu(), tau_gen[i, :, 2].cpu(),
                    color="C1", alpha=0.3, lw=1)
        ax.set_title(f"model trajs (chart 3D)")
        ax.set_xlabel("q_1"); ax.set_ylabel("q_2"); ax.set_zlabel("q_3")
    else:
        ax = fig.add_subplot(2, 3, 4)
        for i in range(n_show_chart):
            ax.plot(tau_data[i, :, 0].cpu(), tau_data[i, :, 1].cpu(),
                    color="C0", alpha=0.3, lw=1)
        ax.set_aspect("equal"); ax.set_title("data trajs (chart 2D)")
        ax = fig.add_subplot(2, 3, 5)
        for i in range(n_show_chart):
            ax.plot(tau_gen[i, :, 0].cpu(), tau_gen[i, :, 1].cpu(),
                    color="C1", alpha=0.3, lw=1)
        ax.set_aspect("equal"); ax.set_title("model trajs (chart 2D)")

    # (f) endpoint comparison (q_start q_end as scatter on first 2 dims)
    ax = fig.add_subplot(2, 3, 6)
    ax.scatter(tau_data[:, 0, 0].cpu(), tau_data[:, 0, 1].cpu(),
               s=4, alpha=0.3, label="data start", color="C0")
    ax.scatter(tau_data[:, -1, 0].cpu(), tau_data[:, -1, 1].cpu(),
               s=4, alpha=0.3, label="data end", color="C2")
    ax.scatter(tau_gen[:, 0, 0].cpu(), tau_gen[:, 0, 1].cpu(),
               s=4, alpha=0.3, label="model start", color="C1", marker="x")
    ax.scatter(tau_gen[:, -1, 0].cpu(), tau_gen[:, -1, 1].cpu(),
               s=4, alpha=0.3, label="model end", color="C3", marker="x")
    ax.set_aspect("equal")
    ax.set_title("endpoints (chart q_1 / q_2)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Toy 3.5 (single-mode) — Riemannian SGM on M^{{H+1}}, N={n_q}, {args.metric}")
    fig.tight_layout()
    out = out_dir / f"toy3p5_traj_uni_{args.metric}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"plots saved to {out}")

    torch.save({
        "args": vars(args),
        "net_state": net.state_dict(),
        "ema_state": ema_net.state_dict(),
        "metrics": {
            "max_g": g_per_pt.max().item(),
            "mean_g": g_per_pt.mean().item(),
            "w1_per_h": w1_per_h,
            "w1_mean": w1_mean,
        },
    }, out_dir / f"ckpt_{args.metric}.pt")


if __name__ == "__main__":
    main()
