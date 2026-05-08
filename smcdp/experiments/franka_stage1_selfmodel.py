"""Stage 1 — train the residual self-model Δ_φ on Franka 7-DoF.

Idea_formulation §7.1, §7.2:
    g_φ(q, p, z_e) = p − FK_analytic(q, z_e) − Δ_φ(q, z_e)
    minimise        ‖g_φ‖² + smoothness(Δ_φ)         on (q, z_e, p_true) data

After training Δ_φ ≈ Δ_true within the calibration regime, the learned
manifold M_φ(z_e) = {(q, p) : p = FK_analytic(q, z_e) + Δ_φ(q, z_e)} is
used as the diffusion substrate in Stage 5+.

Run:
    python -m smcdp.experiments.franka_stage1_selfmodel
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
from smcdp.toy3.self_model import DeltaResidualMLP, self_model_loss
from smcdp.franka.ground_truth import TrueFrankaCompliance
from smcdp.franka.demo_gen import FrankaUniformJointDemo


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=8_000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--smoothness-weight", type=float, default=1e-3)
    p.add_argument("--K-grav", type=float, default=0.025)
    p.add_argument("--K-offset", type=float, default=0.005)
    p.add_argument("--K-tool", type=float, default=0.20)
    p.add_argument("--z-min", type=float, default=0.05)
    p.add_argument("--z-max", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="outputs/franka_stage1")
    p.add_argument("--n-eval", type=int, default=4096)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out_dir={out_dir}")

    arm = Franka7DoF(urdf_path=URDF, end_link="panda_hand", tool_z_max=args.z_max)
    arm._ensure_chain(torch.zeros(1, 7, device=device))
    truth = TrueFrankaCompliance(
        urdf_path=URDF, end_link="panda_hand",
        K_grav=args.K_grav, K_offset=args.K_offset, K_tool=args.K_tool,
    )
    demo = FrankaUniformJointDemo(
        arm, truth, z_e_range=(args.z_min, args.z_max),
    )

    delta_net = DeltaResidualMLP(
        n_q=7, n_p=3, n_z=1, hidden=args.hidden, n_layers=args.n_layers,
        activation=torch.nn.Softplus, final_init_scale=1e-3,
    ).to(device)
    optim = torch.optim.Adam(delta_net.parameters(), lr=args.lr,
                             betas=(0.9, 0.999), eps=1e-8)

    losses_log = []
    fit_log = []
    pbar = tqdm(range(args.steps), desc="stage 1 fit")
    for step in pbar:
        q, z_e, p_true = demo.sample(args.batch, device=device)
        loss, info = self_model_loss(
            delta_net, q, z_e, p_true,
            fk_analytic_fn=lambda q_, z_: arm.F(q_, z_),
            smoothness_weight=args.smoothness_weight,
        )
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        losses_log.append(loss.item())
        fit_log.append(info["fit"])
        if step % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.3e}", fit=f"{info['fit']:.3e}")

    # ---- evaluation ----
    delta_net.eval()
    n = args.n_eval
    q, z_e, p_true = demo.sample(n, device=device)
    with torch.no_grad():
        p_analytic = arm.F(q, z_e)
        delta_pred = delta_net(q, z_e)
        p_learned = p_analytic + delta_pred
        delta_true = truth.delta_true(q, z_e)
    err_analytic = (p_true - p_analytic).norm(dim=-1)
    err_learned = (p_true - p_learned).norm(dim=-1)
    err_residual = (delta_true - delta_pred).norm(dim=-1)

    improvement_factor = (err_analytic.mean() / err_learned.mean().clamp(min=1e-12)).item()
    print()
    print(f"Eval over {n} fresh samples:")
    print(f"  ‖p_true − FK_analytic‖   :  mean = {err_analytic.mean():.4e}   "
          f"max = {err_analytic.max():.4e}")
    print(f"  ‖p_true − F_φ_learned‖   :  mean = {err_learned.mean():.4e}   "
          f"max = {err_learned.max():.4e}")
    print(f"  ‖Δ_true − Δ_φ‖           :  mean = {err_residual.mean():.4e}   "
          f"max = {err_residual.max():.4e}")
    print(f"  improvement factor       :  {improvement_factor:.1f}x")

    # save ckpt
    torch.save({
        "args": vars(args),
        "delta_net_state": delta_net.state_dict(),
        "metrics": {
            "err_analytic_mean": err_analytic.mean().item(),
            "err_learned_mean": err_learned.mean().item(),
            "err_residual_mean": err_residual.mean().item(),
            "improvement_factor": improvement_factor,
        },
    }, out_dir / "delta_phi.pt")

    # plots
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    losses_t = torch.tensor(losses_log)
    fit_t = torch.tensor(fit_log)
    win = max(1, len(losses_t) // 200)
    sm_loss = torch.nn.functional.avg_pool1d(losses_t.view(1, 1, -1), kernel_size=win, stride=win).flatten()
    sm_fit = torch.nn.functional.avg_pool1d(fit_t.view(1, 1, -1), kernel_size=win, stride=win).flatten()
    xs = torch.arange(len(sm_loss)) * win
    ax[0].plot(xs, sm_loss, label="total"); ax[0].plot(xs, sm_fit, label="fit", alpha=0.7)
    ax[0].set_yscale("log"); ax[0].set_xlabel("step"); ax[0].set_ylabel("loss")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title(f"Stage-1 Δ_φ training")

    ax[1].hist(err_analytic.cpu().numpy(), bins=60, alpha=0.5, label="‖p−FK_analytic‖")
    ax[1].hist(err_learned.cpu().numpy(), bins=60, alpha=0.5, label="‖p−F_φ‖")
    ax[1].set_xlabel("EE position error (m)"); ax[1].set_ylabel("count")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[1].set_title(f"improvement {improvement_factor:.1f}x")
    fig.tight_layout()
    fig.savefig(out_dir / "stage1.png", dpi=120)
    plt.close(fig)
    print(f"saved {out_dir / 'stage1.png'} and {out_dir / 'delta_phi.pt'}")


if __name__ == "__main__":
    main()
