"""Step D-2 — full pipeline: learned residual + z_e + multi-modal trajectory.

Combines every component of the Idea_formulation framework:
  - 3-link redundant arm  (n_q = 3 > n_p = 2)
  - Embodiment context z_e (last-link tool extension)
  - LEARNED residual self-model Δ_φ trained in Stage 1
        F_φ(q, z_e) = analytic_FK(q, z_e) + Δ_φ(q, z_e)
        M_φ(z_e) = {(q, p) : p = F_φ(q, z_e)}
  - Bimodal IK demos (elbow-up / elbow-down) reaching a fixed end-effector trajectory
  - Trajectory-level Riemannian SGM on M_φ(z_e)^{H+1}
  - Same 5×512 sin flat-MLP score architecture (no Transformer)

Compared to Step D-1, the only change is that the manifold is now the LEARNED
manifold M_φ(z_e), not the analytic one.  This validates Idea §10 main
contribution C1 ("Riemannian SGM on _learned_ self-model manifold") and C3
(z_e as deformation parameter of the LEARNED manifold) in the integrated
multi-modal + trajectory setting.

Evaluation per z_e
  (i)   adherence to LEARNED manifold:  max_h |g_φ_learned(x_h)|  ≡ 0
  (ii)  adherence to TRUE manifold:    mean_h |g_truth(x_h)|  ~ Stage-1 fit error
  (iii) mode coverage  +  per-mode chart W₁
  (iv)  end-effector reach error to target trajectory
  (v)   in-distribution z_e  +  OOD z_e

Run:
    python -m smcdp.experiments.toy3p5_stepD2 \
        --stage1-ckpt outputs/toy3p5_stage1/delta_phi.pt
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
from smcdp.toy3.ground_truth import TrueNLinkArmCompliance
from smcdp.toy3.self_model import DeltaResidualMLP, LearnedSelfModelNLinkArm
from smcdp.toy3.distributions import WrappedNormalEmbodiment
from smcdp.trajectories import (
    BimodalRedundantTrajectoryDistEmb,
    TrajectoryScoreNet,
    TrajectoryScaledScoreFn,
    traj_reverse_grw,
    traj_dsm_varadhan_loss,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-ckpt", type=str,
                   default="outputs/toy3p5_stage1/delta_phi.pt")
    p.add_argument("--H", type=int, default=8)
    # demo
    p.add_argument("--mu-p-start", type=float, nargs=2, default=[1.6, 0.6])
    p.add_argument("--mu-p-end", type=float, nargs=2, default=[1.6, -0.6])
    p.add_argument("--jitter-p", type=float, default=0.05)
    p.add_argument("--branch-p-up", type=float, default=0.5)
    p.add_argument("--z-min", type=float, default=0.0)
    p.add_argument("--z-max", type=float, default=0.3)
    # net + training (same arch as Step D-1)
    p.add_argument("--steps", type=int, default=25_000)
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
    p.add_argument("--limiting-mean-q", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--limiting-scale", type=float, default=1.5)
    p.add_argument("--n-eval-per-z", type=int, default=2048)
    p.add_argument("--z-eval", type=float, nargs="+",
                   default=[0.00, 0.15, 0.30, 0.45])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="outputs/toy3p5_stepD2")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out_dir={out_dir}  metric={args.metric}")

    # ---- load Stage-1 self-model ----
    ck = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
    s1 = ck["args"]
    print(f"loaded Stage-1: improvement={ck['metrics']['improvement_factor']:.1f}x   "
          f"link_lengths_base={s1['link_lengths_base']}")

    n_q = len(s1["link_lengths_base"])
    delta_net = DeltaResidualMLP(
        n_q=n_q, n_p=2, n_z=1,
        hidden=s1["hidden"], n_layers=s1["n_layers"],
    ).to(device)
    delta_net.load_state_dict(ck["delta_net_state"])
    delta_net.eval()

    arm = LearnedSelfModelNLinkArm(
        delta_net=delta_net,
        link_lengths_base=s1["link_lengths_base"],
        metric=args.metric,
    )
    truth = TrueNLinkArmCompliance(
        link_lengths_base=s1["link_lengths_base"],
        K_grav=s1["K_grav"], K_offset=s1["K_offset"],
    )

    # ---- SDE + distributions ----
    schedule = LinearBetaSchedule(beta_0=args.beta_0, beta_f=args.beta_f, t0=0.0, tf=1.0)
    limiting = WrappedNormalEmbodiment(
        arm, mean_q=args.limiting_mean_q, scale=args.limiting_scale,
        z_e_range=(args.z_min, args.z_max),
    )
    sde = LangevinSDE(arm, schedule, limiting)
    data = BimodalRedundantTrajectoryDistEmb(
        arm, H=args.H,
        mu_p_start=tuple(args.mu_p_start), mu_p_end=tuple(args.mu_p_end),
        jitter_p=args.jitter_p, z_e_range=(args.z_min, args.z_max),
        branch_p_up=args.branch_p_up,
    )

    # ---- score net + EMA ----
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
    pbar = tqdm(range(args.steps), desc=f"train Step D-2/{args.metric}")
    for step in pbar:
        tau_0 = data.sample_x(args.batch, device=device)
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

    # ---- evaluation per z_e ----
    print("\nevaluating at z_e =", args.z_eval)
    H1 = args.H + 1
    n = args.n_eval_per_z
    d = arm.ambient_dim
    h_mid = H1 // 2
    metrics_per_z = {}

    n_dir = 64
    dirs = torch.randn(n_dir, n_q, device=device)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)

    fig, axes = plt.subplots(3, len(args.z_eval), figsize=(4.0 * len(args.z_eval), 11))
    if len(args.z_eval) == 1:
        axes = axes.reshape(3, 1)

    for i, z_val in enumerate(args.z_eval):
        z_tensor = torch.full((n, 1), z_val, device=device)
        # data at this z_e (bimodal IK on LEARNED manifold via make_x)
        z_traj_data = z_tensor.unsqueeze(1).expand(-1, H1, -1).contiguous()
        mu_s = data.mu_p_start.to(device=device)
        mu_e = data.mu_p_end.to(device=device)
        p_start = mu_s + data.jitter_p * torch.randn(n, 2, device=device)
        p_end = mu_e + data.jitter_p * torch.randn(n, 2, device=device)
        s = torch.linspace(0, 1, H1, device=device).view(1, H1, 1)
        p_traj = p_start.unsqueeze(1) + s * (p_end - p_start).unsqueeze(1)
        branch_up_data = torch.rand(n, device=device) < data.branch_p_up
        branch_up_traj = branch_up_data.unsqueeze(1).expand(n, H1)
        q_traj_data = data._ik_3link_emb(p_traj, branch_up_traj, z_tensor)
        x_data_flat = arm.make_x(q_traj_data.reshape(-1, n_q),
                                 z_traj_data.reshape(-1, 1))
        tau_data = x_data_flat.reshape(n, H1, d)

        # generate from limiting at this z_e
        z_lim = z_tensor.unsqueeze(1).expand(-1, H1, -1).reshape(n * H1, -1)
        tau_T = limiting.sample(n * H1, device=device, z_e=z_lim).reshape(n, H1, d)
        with torch.no_grad():
            tau_gen = traj_reverse_grw(sde, score_fn_eval, tau_T,
                                       n_steps=args.n_sample_steps, eps=args.eps)

        # adherence (LEARNED manifold)
        g_learned = arm.constraint(tau_gen.reshape(-1, d)).norm(dim=-1)
        # adherence (TRUE manifold) — uses the oracle compliance
        with torch.no_grad():
            x_pts = tau_gen.reshape(-1, d)
            q_pts = x_pts[..., :n_q]
            p_pts = x_pts[..., n_q : n_q + 2]
            z_pts = x_pts[..., n_q + 2 :]
            p_truth = truth.p_true(q_pts, z_pts)
        g_truth = (p_pts - p_truth).norm(dim=-1)

        # mode classification (sign of q_2 at midpoint — bimodal IK has q_2 ≫ 0 for up)
        q2_mid_data = tau_data[:, h_mid, 1]
        q2_mid_gen = tau_gen[:, h_mid, 1]
        mode_up_data = q2_mid_data > 0
        mode_up_gen = q2_mid_gen > 0
        frac_up_data = mode_up_data.float().mean().item()
        frac_up_gen = mode_up_gen.float().mean().item()
        between = (q2_mid_gen.abs() < 0.5).float().mean().item()

        # per-mode chart W₁
        def per_mode_w1(tau_a, tau_b):
            m = min(tau_a.shape[0], tau_b.shape[0])
            if m < 8:
                return float("nan")
            ia = torch.randperm(tau_a.shape[0], device=device)[:m]
            ib = torch.randperm(tau_b.shape[0], device=device)[:m]
            a = tau_a[ia, :, :n_q]; b = tau_b[ib, :, :n_q]
            ws = []
            for h in range(H1):
                pa = (a[:, h] @ dirs.T).sort(dim=0).values
                pb = (b[:, h] @ dirs.T).sort(dim=0).values
                ws.append((pa - pb).abs().mean().item())
            return sum(ws) / len(ws)

        w1_up = per_mode_w1(tau_gen[mode_up_gen], tau_data[mode_up_data])
        w1_dn = per_mode_w1(tau_gen[~mode_up_gen], tau_data[~mode_up_data])

        # end-effector reach error
        p_gen_end = tau_gen[:, -1, n_q : n_q + 2]
        reach_err = (p_gen_end - mu_e).norm(dim=-1).mean().item()

        metrics_per_z[z_val] = {
            "max_g_learned": g_learned.max().item(),
            "mean_g_learned": g_learned.mean().item(),
            "mean_g_truth": g_truth.mean().item(),
            "frac_up_data": frac_up_data,
            "frac_up_gen": frac_up_gen,
            "frac_between_modes": between,
            "w1_up": w1_up,
            "w1_dn": w1_dn,
            "reach_err": reach_err,
        }
        print(f"  z_e = {z_val:.2f}:  max|g_learned|={g_learned.max():.1e}   "
              f"mean|g_truth|={g_truth.mean():.1e}   "
              f"frac_up data/gen = {frac_up_data:.3f}/{frac_up_gen:.3f}   "
              f"between={between:.3f}   "
              f"W₁ up/dn = {w1_up:.3f}/{w1_dn:.3f}   "
              f"reach={reach_err:.3f}")

        # plots
        ax = axes[0, i]
        ax.hist(q2_mid_data.cpu().numpy(), bins=60, density=True, alpha=0.5,
                label="data", color="C0")
        ax.hist(q2_mid_gen.cpu().numpy(), bins=60, density=True, alpha=0.5,
                label="model", color="C1")
        ax.axvline(0.0, color="k", lw=0.5, alpha=0.4)
        ax.set_xlabel("q_2 mid"); ax.set_ylabel("density")
        ax.set_title(f"z_e={z_val:.2f}  frac_up={frac_up_gen:.2f}")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[1, i]
        n_show = 30
        for j in torch.where(mode_up_data)[0][:n_show].cpu().tolist():
            ax.plot(tau_data[j, :, n_q].cpu(), tau_data[j, :, n_q + 1].cpu(),
                    color="C0", alpha=0.4, lw=1)
        for j in torch.where(~mode_up_data)[0][:n_show].cpu().tolist():
            ax.plot(tau_data[j, :, n_q].cpu(), tau_data[j, :, n_q + 1].cpu(),
                    color="C2", alpha=0.4, lw=1)
        for j in torch.where(mode_up_gen)[0][:n_show].cpu().tolist():
            ax.plot(tau_gen[j, :, n_q].cpu(), tau_gen[j, :, n_q + 1].cpu(),
                    color="C1", alpha=0.4, lw=1)
        for j in torch.where(~mode_up_gen)[0][:n_show].cpu().tolist():
            ax.plot(tau_gen[j, :, n_q].cpu(), tau_gen[j, :, n_q + 1].cpu(),
                    color="C3", alpha=0.4, lw=1)
        ax.set_aspect("equal")
        ax.set_xlabel("p_x"); ax.set_ylabel("p_y")
        ax.set_title(f"end-eff trajs  reach={reach_err:.3f}\n|g_truth|≈{g_truth.mean():.1e}")
        ax.grid(alpha=0.3)

        ax = axes[2, i]
        for j in torch.where(branch_up_data)[0][:n_show].cpu().tolist():
            ax.plot(tau_data[j, :, 0].cpu(), tau_data[j, :, 1].cpu(),
                    color="C0", alpha=0.4, lw=1)
        for j in torch.where(~branch_up_data)[0][:n_show].cpu().tolist():
            ax.plot(tau_data[j, :, 0].cpu(), tau_data[j, :, 1].cpu(),
                    color="C2", alpha=0.4, lw=1)
        for j in torch.where(mode_up_gen)[0][:n_show].cpu().tolist():
            ax.plot(tau_gen[j, :, 0].cpu(), tau_gen[j, :, 1].cpu(),
                    color="C1", alpha=0.4, lw=1)
        for j in torch.where(~mode_up_gen)[0][:n_show].cpu().tolist():
            ax.plot(tau_gen[j, :, 0].cpu(), tau_gen[j, :, 1].cpu(),
                    color="C3", alpha=0.4, lw=1)
        ax.set_xlabel("q_1"); ax.set_ylabel("q_2")
        ax.set_title(f"chart q_1×q_2  W₁={(w1_up + w1_dn)/2:.3f}")
        ax.grid(alpha=0.3)

    fig.suptitle(f"Step D-2 — full pipeline (LEARNED M_φ + z_e + bimodal)  ({args.metric})")
    fig.tight_layout()
    out = out_dir / f"stepD2_{args.metric}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nplots saved to {out}")

    # loss curve
    fig2, ax = plt.subplots(figsize=(7, 4))
    losses_t = torch.tensor(losses_log)
    win = max(1, len(losses_t) // 200)
    smooth = torch.nn.functional.avg_pool1d(
        losses_t.view(1, 1, -1), kernel_size=win, stride=win
    ).flatten()
    ax.plot(torch.arange(len(smooth)) * win, smooth)
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("traj DSM loss")
    ax.set_title(f"Step D-2 training loss ({args.metric})")
    ax.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out_dir / f"stepD2_loss_{args.metric}.png", dpi=120)
    plt.close(fig2)

    torch.save({
        "args": vars(args),
        "stage1_args": s1,
        "net_state": net.state_dict(),
        "ema_state": ema_net.state_dict(),
        "metrics_per_z": metrics_per_z,
    }, out_dir / f"ckpt_{args.metric}.pt")


if __name__ == "__main__":
    main()
