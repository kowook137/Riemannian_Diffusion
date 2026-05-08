"""Toy 3.5 (multi-modal) — bimodal trajectory diffusion on a 3-link redundant arm.

Step C in the framework-validation roadmap.  This is the FIRST true test of the
multi-modal claim in Idea_formulation §10/§11/§15: the demo distribution is
bimodal in q-space (two IK branches reach the same end-effector trajectory),
and we check whether the framework captures BOTH modes vs collapses or averages.

Setup
  - 3-link planar arm  ( ℓ_1 = 1, ℓ_2 = 1, ℓ_3 = 0.5 )
  - End-effector trajectory:  smooth p_h interp from p_start ≈ (1.6, +0.6) to (1.6, −0.6)
  - Demo:  per trajectory, one branch ∈ {elbow-up, elbow-down}, 50/50 random
  - Both branches reach the same p-trajectory; the q-trajectories cluster into
    two modes ≈ 4.7 rad apart in chart space.

Evaluation
  (i)   per-timestep adherence  max_h |g_φ(x_h)|  ≡ 0  by construction
  (ii)  mode classification:    sign of q_2 at midpoint cleanly separates
                                 branches in the demo (verified: 100% accuracy);
                                 apply same rule to generated samples
  (iii) coverage:  frac(branch=up) for model vs demo (target = 0.5)
  (iv)  per-mode shape:  per-branch chart sliced-W₁ vs the corresponding demo
                          half-distribution
  (v)   mode-collapse / averaging detection:  if a sample's q_2 stays near 0
        across the trajectory, it's between modes (averaging) — count fraction.

Run:
    python -m smcdp.experiments.toy3p5_traj_bimodal
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D    # noqa: F401

from smcdp.manifolds import NLinkPlanarArm
from smcdp.sde import LangevinSDE, LinearBetaSchedule
from smcdp.distributions import WrappedNormalGraph
from smcdp.trajectories import (
    BimodalRedundantTrajectoryDist,
    TrajectoryScoreNet,
    TrajectoryScaledScoreFn,
    traj_reverse_grw,
    traj_dsm_varadhan_loss,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--link-lengths", type=float, nargs=3, default=[1.0, 1.0, 0.5])
    p.add_argument("--H", type=int, default=8)
    # demo
    p.add_argument("--mu-p-start", type=float, nargs=2, default=[1.6, 0.6])
    p.add_argument("--mu-p-end", type=float, nargs=2, default=[1.6, -0.6])
    p.add_argument("--jitter-p", type=float, default=0.05)
    p.add_argument("--branch-p-up", type=float, default=0.5)
    # net + training (RSGM-style; bigger than unimodal because bimodal is harder)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--warmup-steps", type=int, default=200)
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
    # limiting (wider than unimodal so it spans both modes)
    p.add_argument("--limiting-mean-q", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--limiting-scale", type=float, default=1.5)
    # eval
    p.add_argument("--n-eval", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="outputs/toy3p5_bimodal")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out_dir={out_dir}  metric={args.metric}")

    # ---- problem setup ----
    arm = NLinkPlanarArm(args.link_lengths, metric=args.metric)
    schedule = LinearBetaSchedule(beta_0=args.beta_0, beta_f=args.beta_f, t0=0.0, tf=1.0)
    limiting = WrappedNormalGraph(arm, mean_q=args.limiting_mean_q, scale=args.limiting_scale)
    sde = LangevinSDE(arm, schedule, limiting)
    data = BimodalRedundantTrajectoryDist(
        arm, H=args.H,
        mu_p_start=tuple(args.mu_p_start),
        mu_p_end=tuple(args.mu_p_end),
        jitter_p=args.jitter_p,
        branch_p_up=args.branch_p_up,
    )

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
    pbar = tqdm(range(args.steps), desc=f"train bimodal traj/{args.metric}")
    for step in pbar:
        tau_0 = data.sample_x(args.batch, device=device)                     # (B, H+1, 5)
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
    tau_data, branch_data = data.sample(n, device=device)

    # ---- adherence ----
    g_per_pt = arm.constraint(tau_gen.reshape(-1, d)).norm(dim=-1)
    print(f"per-timestep adherence:  mean|g| = {g_per_pt.mean():.2e}   "
          f"max|g| = {g_per_pt.max():.2e}")

    # ---- mode classification (use q_2 at midpoint) ----
    h_mid = H1 // 2
    q2_mid_data = tau_data[:, h_mid, 1]
    q2_mid_gen = tau_gen[:, h_mid, 1]

    # demo branch label (ground truth we have): branch_data
    # generated branch: sign of q_2 mid (gating threshold around 0)
    mode_up_data = (q2_mid_data > 0)
    mode_up_gen = (q2_mid_gen > 0)
    frac_up_data = mode_up_data.float().mean().item()
    frac_up_gen = mode_up_gen.float().mean().item()
    # consistency check: data branch label vs sign(q_2)
    label_consistency = (mode_up_data == branch_data).float().mean().item()
    print(f"\nmode-classification rule  (sign of q_2 at h={h_mid}):")
    print(f"  data label-vs-sign consistency      = {label_consistency:.3f}   "
          f"(expect ≈ 1.0)")
    print(f"  frac(branch=up)   data  = {frac_up_data:.3f}    "
          f"model = {frac_up_gen:.3f}    "
          f"(target = {args.branch_p_up:.2f})")

    # ---- mode-collapse / averaging detection ----
    # samples whose |q_2_mid| < 0.5 are "between modes" (within ~25% of separation)
    between = (q2_mid_gen.abs() < 0.5).float().mean().item()
    print(f"  frac of model samples in mode-AVERAGING region |q_2_mid|<0.5  = {between:.3f}   "
          f"(expect ≈ 0)")

    # ---- per-mode chart sliced-W₁ ----
    n_dir = 64
    dirs = torch.randn(n_dir, 3, device=device)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)

    def per_mode_w1(tau_a: torch.Tensor, tau_b: torch.Tensor):
        """Compare two q-distributions at each h via sliced-W₁ (averaged over h)."""
        H1_local = tau_a.shape[1]
        # equalise sample counts via random subsample
        m = min(tau_a.shape[0], tau_b.shape[0])
        if m < 8:
            return float("nan")
        ia = torch.randperm(tau_a.shape[0], device=device)[:m]
        ib = torch.randperm(tau_b.shape[0], device=device)[:m]
        a = tau_a[ia, :, :3]; b = tau_b[ib, :, :3]
        ws = []
        for h in range(H1_local):
            pa = (a[:, h] @ dirs.T).sort(dim=0).values
            pb = (b[:, h] @ dirs.T).sort(dim=0).values
            ws.append((pa - pb).abs().mean().item())
        return sum(ws) / len(ws)

    data_up = tau_data[mode_up_data]
    data_dn = tau_data[~mode_up_data]
    gen_up = tau_gen[mode_up_gen]
    gen_dn = tau_gen[~mode_up_gen]

    w1_up = per_mode_w1(gen_up, data_up)
    w1_dn = per_mode_w1(gen_dn, data_dn)
    print(f"\nper-mode chart sliced-W₁:")
    print(f"  up    : {w1_up:.4f}   (n_data_up = {data_up.shape[0]}, n_gen_up = {gen_up.shape[0]})")
    print(f"  down  : {w1_dn:.4f}   (n_data_down = {data_dn.shape[0]}, n_gen_down = {gen_dn.shape[0]})")

    # ---- plots ----
    fig = plt.figure(figsize=(18, 10))

    # (a) loss
    ax = fig.add_subplot(2, 4, 1)
    losses_t = torch.tensor(losses_log)
    win = max(1, len(losses_t) // 200)
    smooth = torch.nn.functional.avg_pool1d(
        losses_t.view(1, 1, -1), kernel_size=win, stride=win).flatten()
    ax.plot(torch.arange(len(smooth)) * win, smooth)
    ax.set_yscale("log"); ax.grid(alpha=0.3)
    ax.set_xlabel("step"); ax.set_ylabel("traj DSM loss"); ax.set_title("loss")

    # (b) end-effector trajectories (data vs model — should overlap because both reach same p)
    ax = fig.add_subplot(2, 4, 2)
    n_show = 80
    for i in range(n_show):
        ax.plot(tau_data[i, :, 3].cpu(), tau_data[i, :, 4].cpu(),
                color="C0", alpha=0.3, lw=1)
        ax.plot(tau_gen[i, :, 3].cpu(), tau_gen[i, :, 4].cpu(),
                color="C1", alpha=0.3, lw=1)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("p_x"); ax.set_ylabel("p_y")
    ax.set_title(f"end-effector trajs  (max|g|={g_per_pt.max():.0e})")

    # (c) q_2 midpoint histogram — bimodality test
    ax = fig.add_subplot(2, 4, 3)
    bins = 60
    ax.hist(q2_mid_data.cpu().numpy(), bins=bins, density=True, alpha=0.5,
            label="data", color="C0")
    ax.hist(q2_mid_gen.cpu().numpy(), bins=bins, density=True, alpha=0.5,
            label="model", color="C1")
    ax.axvline(0.0, color="k", lw=0.7, alpha=0.4)
    ax.set_xlabel("q_2 at h = H/2"); ax.set_ylabel("density")
    ax.set_title(f"q_2 midpoint  (frac_up data={frac_up_data:.2f}, model={frac_up_gen:.2f})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (d) chart trajectories — DATA, coloured by branch
    ax = fig.add_subplot(2, 4, 4, projection="3d")
    n_show_chart = 30
    for i in torch.where(branch_data)[0][:n_show_chart].cpu().tolist():
        ax.plot(tau_data[i, :, 0].cpu(), tau_data[i, :, 1].cpu(), tau_data[i, :, 2].cpu(),
                color="C0", alpha=0.5, lw=1)
    for i in torch.where(~branch_data)[0][:n_show_chart].cpu().tolist():
        ax.plot(tau_data[i, :, 0].cpu(), tau_data[i, :, 1].cpu(), tau_data[i, :, 2].cpu(),
                color="C2", alpha=0.5, lw=1)
    ax.set_xlabel("q_1"); ax.set_ylabel("q_2"); ax.set_zlabel("q_3")
    ax.set_title("data trajs in chart  (up=blue, down=green)")

    # (e) chart trajectories — MODEL, coloured by classified branch
    ax = fig.add_subplot(2, 4, 5, projection="3d")
    for i in torch.where(mode_up_gen)[0][:n_show_chart].cpu().tolist():
        ax.plot(tau_gen[i, :, 0].cpu(), tau_gen[i, :, 1].cpu(), tau_gen[i, :, 2].cpu(),
                color="C1", alpha=0.5, lw=1)
    for i in torch.where(~mode_up_gen)[0][:n_show_chart].cpu().tolist():
        ax.plot(tau_gen[i, :, 0].cpu(), tau_gen[i, :, 1].cpu(), tau_gen[i, :, 2].cpu(),
                color="C3", alpha=0.5, lw=1)
    ax.set_xlabel("q_1"); ax.set_ylabel("q_2"); ax.set_zlabel("q_3")
    ax.set_title("model trajs in chart  (up=orange, down=red)")

    # (f) q_1 vs q_2 scatter at midpoint (mode visualisation)
    ax = fig.add_subplot(2, 4, 6)
    ax.scatter(tau_data[branch_data, h_mid, 0].cpu(),
               tau_data[branch_data, h_mid, 1].cpu(),
               s=4, alpha=0.4, color="C0", label="data up")
    ax.scatter(tau_data[~branch_data, h_mid, 0].cpu(),
               tau_data[~branch_data, h_mid, 1].cpu(),
               s=4, alpha=0.4, color="C2", label="data down")
    ax.scatter(tau_gen[mode_up_gen, h_mid, 0].cpu(),
               tau_gen[mode_up_gen, h_mid, 1].cpu(),
               s=4, alpha=0.4, color="C1", marker="x", label="model up")
    ax.scatter(tau_gen[~mode_up_gen, h_mid, 0].cpu(),
               tau_gen[~mode_up_gen, h_mid, 1].cpu(),
               s=4, alpha=0.4, color="C3", marker="x", label="model down")
    ax.set_xlabel("q_1"); ax.set_ylabel("q_2")
    ax.set_title(f"q_1×q_2 at h={h_mid}  (mid-trajectory)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (g) per-mode W₁ bar
    ax = fig.add_subplot(2, 4, 7)
    ax.bar(["up", "down"], [w1_up, w1_dn], color=["C1", "C3"])
    ax.set_ylabel("chart sliced-W₁")
    ax.set_title("per-mode distribution match")
    ax.grid(alpha=0.3, axis="y")

    # (h) mode coverage bar
    ax = fig.add_subplot(2, 4, 8)
    width = 0.35
    x = torch.arange(2)
    ax.bar(x - width/2, [frac_up_data, 1 - frac_up_data], width=width, label="data", color="C0")
    ax.bar(x + width/2, [frac_up_gen, 1 - frac_up_gen], width=width, label="model", color="C1")
    ax.set_xticks(x); ax.set_xticklabels(["up", "down"])
    ax.axhline(args.branch_p_up, color="k", lw=0.5, ls="--", alpha=0.5, label="target P(up)")
    ax.set_ylabel("fraction"); ax.set_ylim(0, 1)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("mode coverage")

    fig.suptitle(f"Toy 3.5 (bimodal) — kinematic-redundancy multi-modal capture, {args.metric}")
    fig.tight_layout()
    out = out_dir / f"toy3p5_bimodal_{args.metric}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nplots saved to {out}")

    torch.save({
        "args": vars(args),
        "net_state": net.state_dict(),
        "ema_state": ema_net.state_dict(),
        "metrics": {
            "max_g": g_per_pt.max().item(),
            "mean_g": g_per_pt.mean().item(),
            "frac_up_data": frac_up_data,
            "frac_up_gen": frac_up_gen,
            "frac_between_modes": between,
            "w1_up": w1_up,
            "w1_dn": w1_dn,
            "label_consistency": label_consistency,
        },
    }, out_dir / f"ckpt_{args.metric}.pt")


if __name__ == "__main__":
    main()
