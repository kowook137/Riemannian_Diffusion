"""P2 smoke test — TrajectoryScoreNetUNet on Franka7DoF (unimodal).

Goal: confirm that
  (a) the ConditionalUnet1D wrapper produces a tangent-correct score
      (J_g · score = 0 by construction via lift_chart_to_tangent),
  (b) forward / reverse trajectory GRW on Franka7DoF runs end-to-end without
      autograd / vmap / dtype issues stemming from pytorch_kinematics,
  (c) DSM-Varadhan loss decreases monotonically and the EMA-eval reverse
      samples converge toward the data chart-Gaussian (mean q match).

Not a paper-quality experiment — a substrate plumbing test.

Run: python -m smcdp.experiments.franka_smoke_unet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import pybullet_data
from tqdm import tqdm

from smcdp.manifolds import Franka7DoF
from smcdp.sde import LangevinSDE, LinearBetaSchedule
from smcdp.franka.distributions import WrappedNormalFranka7DoF
from smcdp.trajectories import (
    LinearChartTrajectoryDistEmb,
    TrajectoryScoreNetUNet,
    TrajectoryScaledScoreFn,
    traj_reverse_grw,
    traj_dsm_varadhan_loss,
)


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--H", type=int, default=15)            # H+1=16 (UNet down=2 → div by 4)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--beta-0", type=float, default=0.001)
    p.add_argument("--beta-f", type=float, default=4.0)
    p.add_argument("--eps", type=float, default=2e-4)
    p.add_argument("--n-grw-steps", type=int, default=10)
    p.add_argument("--n-sample-steps", type=int, default=100)
    p.add_argument("--metric", type=str, default="riemannian",
                   choices=["riemannian", "chart_euclidean"])
    p.add_argument("--mean-q", type=float, nargs=7,
                   default=[0.0, -0.7, 0.0, -2.0, 0.0, 1.5, 0.0])
    p.add_argument("--scale-endpoint", type=float, default=0.10)
    p.add_argument("--limiting-scale", type=float, default=0.30)
    p.add_argument("--z-min", type=float, default=0.05)
    p.add_argument("--z-max", type=float, default=0.15)
    p.add_argument("--n-eval", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="outputs/franka_smoke_unet")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  metric={args.metric}  H+1={args.H+1}")

    arm = Franka7DoF(urdf_path=URDF, end_link="panda_hand",
                     tool_z_max=max(args.z_max, 0.20), metric=args.metric)
    arm._ensure_chain(torch.zeros(1, 7, device=device))    # warm chain on device

    schedule = LinearBetaSchedule(beta_0=args.beta_0, beta_f=args.beta_f, t0=0.0, tf=1.0)
    limiting = WrappedNormalFranka7DoF(
        arm, mean_q=list(args.mean_q), scale=args.limiting_scale,
        z_e_range=(args.z_min, args.z_max),
    )
    sde = LangevinSDE(arm, schedule, limiting)
    data = LinearChartTrajectoryDistEmb(
        arm, H=args.H, mean_q=list(args.mean_q),
        scale_endpoint=args.scale_endpoint,
        z_e_range=(args.z_min, args.z_max),
    )

    def make_net():
        return TrajectoryScoreNetUNet(
            arm, H=args.H,
            down_dims=(64, 128, 256),
            diffusion_step_embed_dim=128,
            n_groups=8,
            kernel_size=3,
            t_scale=1000.0,
        ).to(device)

    net = make_net()
    ema_net = make_net()
    ema_net.load_state_dict(net.state_dict())
    for p in ema_net.parameters():
        p.requires_grad_(False)

    print(f"UNet params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")

    score_fn_train = TrajectoryScaledScoreFn(net, sde)
    score_fn_eval = TrajectoryScaledScoreFn(ema_net, sde)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))

    losses = []
    pbar = tqdm(range(args.steps), desc="smoke train")
    for step in pbar:
        tau_0 = data.sample(args.batch, device=device)
        loss = traj_dsm_varadhan_loss(score_fn_train, sde, tau_0,
                                      eps=args.eps, weight="sigma2",
                                      n_grw_steps=args.n_grw_steps)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        with torch.no_grad():
            for pe, pn in zip(ema_net.parameters(), net.parameters()):
                pe.mul_(args.ema).add_(pn, alpha=1.0 - args.ema)
        losses.append(loss.item())
        if step % 50 == 0:
            pbar.set_postfix(loss=f"{loss.item():.3f}")

    # --- evaluate ---
    print("\nEvaluating reverse samples...")
    n = args.n_eval
    H1 = args.H + 1
    d = arm.ambient_dim
    z_eval = 0.5 * (args.z_min + args.z_max)
    z_tensor = torch.full((n * H1, 1), z_eval, device=device)
    tau_T = limiting.sample(n * H1, device=device, z_e=z_tensor).reshape(n, H1, d)
    with torch.no_grad():
        tau_gen = traj_reverse_grw(sde, score_fn_eval, tau_T,
                                   n_steps=args.n_sample_steps, eps=args.eps)

    # data baseline
    tau_data = data.sample(n, device=device, z_e=torch.full((n, 1), z_eval, device=device))

    # diagnostics
    g_gen = arm.constraint(tau_gen.reshape(-1, d)).norm(dim=-1).max().item()
    g_data = arm.constraint(tau_data.reshape(-1, d)).norm(dim=-1).max().item()

    q_gen = tau_gen[..., :7]
    q_data = tau_data[..., :7]
    mean_q_gen = q_gen.mean(dim=(0, 1))
    mean_q_data = q_data.mean(dim=(0, 1))
    mean_err = (mean_q_gen - mean_q_data).norm().item()
    std_q_gen = q_gen.std(dim=(0, 1))
    std_q_data = q_data.std(dim=(0, 1))

    print(f"  loss(start) = {losses[0]:.3f}  loss(end) = {losses[-1]:.3f}  "
          f"loss_min = {min(losses):.3f}")
    print(f"  max ‖constraint‖  data={g_data:.1e}   gen={g_gen:.1e}")
    print(f"  ‖mean_q_gen − mean_q_data‖ = {mean_err:.4f}")
    print(f"  std_q_gen  = {std_q_gen.tolist()}")
    print(f"  std_q_data = {std_q_data.tolist()}")

    # success thresholds: substrate plumbing only
    ok_constraint = g_gen < 1e-5
    ok_loss = losses[-1] < losses[0]
    print()
    print(f"  constraint adherence (gen on M):  {'PASS' if ok_constraint else 'FAIL'}")
    print(f"  loss decreased over training:     {'PASS' if ok_loss else 'FAIL'}")

    torch.save({
        "args": vars(args),
        "losses": losses,
        "mean_err": mean_err,
        "g_gen_max": g_gen,
    }, out_dir / "smoke_summary.pt")

    if not (ok_constraint and ok_loss):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
