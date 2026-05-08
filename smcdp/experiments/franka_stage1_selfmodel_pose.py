"""Stage 1 — train the SE(3) residual self-model ξ_φ on Franka 7-DoF
(extension.tex Sec. 1).

Mirrors `franka_stage1_selfmodel.py` but learns a se(3) residual twist
ξ_φ(q, z_e) = (ρ_φ, ω_φ) instead of a 3D Δ_φ.  The Stage-1 loss is the
weighted body-frame error twist (extension.tex Eq. (5)):

    L_self = w_p ‖e_ρ‖² + w_R ‖e_ω‖²
             + β_p ‖∂_q ρ_φ‖_F² + β_R ‖∂_q ω_φ‖_F²
    where (e_ρ, e_ω) = Log_SE(3)(T_φ(q, z_e)^{-1} · T_true(q, z_e))

Run:
    python -m smcdp.experiments.franka_stage1_selfmodel_pose
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

from smcdp.manifolds_pose import Franka7DoFPose
from smcdp.franka.self_model_pose import (
    PoseResidualMLP, pose_self_model_loss,
)
from smcdp.franka.ground_truth_pose import TrueFrankaCompliancePose
from smcdp.lie_se3 import log_relative_Rp


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--w-p", type=float, default=1.0,
                   help="position fidelity weight (eq. 5).")
    p.add_argument("--w-R", type=float, default=1.0,
                   help="rotation fidelity weight (eq. 5).")
    p.add_argument("--beta-p", type=float, default=1e-3)
    p.add_argument("--beta-R", type=float, default=1e-3)
    p.add_argument("--K-grav", type=float, default=0.025)
    p.add_argument("--K-offset", type=float, default=0.005)
    p.add_argument("--K-tool", type=float, default=0.20)
    p.add_argument("--K-R", type=float, default=0.025,
                   help="rotation compliance scale (rad ~ 1.4° at base)")
    p.add_argument("--A-x", type=float, default=1.0)
    p.add_argument("--A-y", type=float, default=1.0)
    p.add_argument("--A-z", type=float, default=0.5)
    p.add_argument("--K-tool-R", type=float, default=0.5)
    p.add_argument("--z-min", type=float, default=0.05)
    p.add_argument("--z-max", type=float, default=0.15)
    p.add_argument("--joint-margin", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="outputs/franka_stage1_pose")
    p.add_argument("--n-eval", type=int, default=4096)
    return p.parse_args()


def _sample_chart(arm: Franka7DoFPose, n: int, z_min: float, z_max: float,
                   margin: float, device, dtype=torch.float32):
    lower, upper = arm.joint_limits(device=device, dtype=dtype)
    m = margin * (upper - lower)
    lo = lower + m
    hi = upper - m
    q = lo + (hi - lo) * torch.rand(n, 7, device=device, dtype=dtype)
    z = z_min + (z_max - z_min) * torch.rand(n, 1, device=device, dtype=dtype)
    return q, z


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out_dir={out_dir}")

    arm = Franka7DoFPose(
        urdf_path=URDF, end_link="panda_hand", tool_z_max=args.z_max,
        joint_limit_margin_frac=args.joint_margin,
    )
    arm._ensure_chain(torch.zeros(1, 7, device=device))
    truth = TrueFrankaCompliancePose(
        urdf_path=URDF, end_link="panda_hand",
        K_grav=args.K_grav, K_offset=args.K_offset, K_tool=args.K_tool,
        K_R=args.K_R, A_x=args.A_x, A_y=args.A_y, A_z=args.A_z,
        K_tool_R=args.K_tool_R,
    )

    residual_net = PoseResidualMLP(
        n_q=7, n_z=1, hidden=args.hidden, n_layers=args.n_layers,
        activation=torch.nn.Softplus, final_init_scale=1e-3,
        output_omega=True,
    ).to(device)
    optim = torch.optim.Adam(residual_net.parameters(), lr=args.lr,
                              betas=(0.9, 0.999), eps=1e-8)

    losses_log: list[float] = []
    fit_p_log: list[float] = []
    fit_R_log: list[float] = []
    pbar = tqdm(range(args.steps), desc="stage 1 pose fit")
    for step in pbar:
        q, z_e = _sample_chart(arm, args.batch, args.z_min, args.z_max,
                                args.joint_margin, device=device)
        R_t, p_t = truth.T_true_Rp(q, z_e)
        loss, info = pose_self_model_loss(
            residual_net, q, z_e, (R_t, p_t),
            fk_analytic_Rp=lambda q_, z_: arm.T_phi_Rp(q_, z_),
            w_p=args.w_p, w_R=args.w_R,
            beta_p=args.beta_p, beta_R=args.beta_R,
        )
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        losses_log.append(loss.item())
        fit_p_log.append(info["fit_p"])
        fit_R_log.append(info["fit_R"])
        if step % 100 == 0:
            pbar.set_postfix(
                loss=f"{loss.item():.3e}",
                fit_p=f"{info['fit_p']:.3e}",
                fit_R=f"{info['fit_R']:.3e}",
            )

    # ---- evaluation ----
    residual_net.eval()
    n = args.n_eval
    q, z_e = _sample_chart(arm, n, args.z_min, args.z_max,
                             args.joint_margin, device=device)
    with torch.no_grad():
        # Errors of analytic FK alone
        R_a, p_a = arm.T_phi_Rp(q, z_e)
        R_t, p_t = truth.T_true_Rp(q, z_e)
        e_analytic = log_relative_Rp(R_a, p_a, R_t, p_t)
        err_a_p = e_analytic[..., :3].norm(dim=-1)
        err_a_R = e_analytic[..., 3:].norm(dim=-1)

        # Errors of learned T_φ
        from smcdp.lie_se3 import exp_SE3, compose_Rp
        xi = residual_net(q, z_e)
        R_d, p_d = exp_SE3(xi)
        R_phi, p_phi = compose_Rp(R_a, p_a, R_d, p_d)
        e_learned = log_relative_Rp(R_phi, p_phi, R_t, p_t)
        err_l_p = e_learned[..., :3].norm(dim=-1)
        err_l_R = e_learned[..., 3:].norm(dim=-1)

    impr_p = (err_a_p.mean() / err_l_p.mean().clamp(min=1e-12)).item()
    impr_R = (err_a_R.mean() / err_l_R.mean().clamp(min=1e-12)).item()
    print()
    print(f"Eval over {n} fresh samples:")
    print(f"  position error  ‖e_ρ‖")
    print(f"     analytic FK     mean = {err_a_p.mean():.4e}   max = {err_a_p.max():.4e}")
    print(f"     learned  T_φ    mean = {err_l_p.mean():.4e}   max = {err_l_p.max():.4e}")
    print(f"     improvement     {impr_p:.1f}x")
    print(f"  rotation error  ‖e_ω‖  (rad)")
    print(f"     analytic FK     mean = {err_a_R.mean():.4e}   max = {err_a_R.max():.4e}")
    print(f"     learned  T_φ    mean = {err_l_R.mean():.4e}   max = {err_l_R.max():.4e}")
    print(f"     improvement     {impr_R:.1f}x")

    torch.save({
        "args": vars(args),
        "residual_net_state": residual_net.state_dict(),
        "metrics": {
            "err_analytic_pos_mean": err_a_p.mean().item(),
            "err_learned_pos_mean": err_l_p.mean().item(),
            "err_analytic_rot_mean_rad": err_a_R.mean().item(),
            "err_learned_rot_mean_rad": err_l_R.mean().item(),
            "improvement_pos": impr_p,
            "improvement_rot": impr_R,
        },
    }, out_dir / "xi_phi.pt")

    # plots
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    losses_t = torch.tensor(losses_log)
    fit_p_t = torch.tensor(fit_p_log)
    fit_R_t = torch.tensor(fit_R_log)
    win = max(1, len(losses_t) // 200)
    sm_loss = torch.nn.functional.avg_pool1d(losses_t.view(1, 1, -1),
                                               kernel_size=win, stride=win).flatten()
    sm_p = torch.nn.functional.avg_pool1d(fit_p_t.view(1, 1, -1),
                                            kernel_size=win, stride=win).flatten()
    sm_R = torch.nn.functional.avg_pool1d(fit_R_t.view(1, 1, -1),
                                            kernel_size=win, stride=win).flatten()
    xs = torch.arange(len(sm_loss)) * win
    ax[0].plot(xs, sm_loss, label="total")
    ax[0].plot(xs, sm_p, label="fit_p", alpha=0.7)
    ax[0].plot(xs, sm_R, label="fit_R", alpha=0.7)
    ax[0].set_yscale("log"); ax[0].set_xlabel("step"); ax[0].set_ylabel("loss")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title("Stage-1 ξ_φ training")

    ax[1].hist(err_a_p.cpu().numpy(), bins=60, alpha=0.5, label="‖e_ρ analytic‖")
    ax[1].hist(err_l_p.cpu().numpy(), bins=60, alpha=0.5, label="‖e_ρ learned‖")
    ax[1].set_xlabel("position error (m)"); ax[1].set_ylabel("count")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[1].set_title(f"position improvement {impr_p:.1f}x")

    ax[2].hist(err_a_R.cpu().numpy(), bins=60, alpha=0.5, label="‖e_ω analytic‖")
    ax[2].hist(err_l_R.cpu().numpy(), bins=60, alpha=0.5, label="‖e_ω learned‖")
    ax[2].set_xlabel("rotation error (rad)"); ax[2].set_ylabel("count")
    ax[2].legend(); ax[2].grid(alpha=0.3)
    ax[2].set_title(f"rotation improvement {impr_R:.1f}x")

    fig.tight_layout()
    fig.savefig(out_dir / "stage1_pose.png", dpi=120)
    plt.close(fig)
    print(f"saved {out_dir / 'stage1_pose.png'} and {out_dir / 'xi_phi.pt'}")


if __name__ == "__main__":
    main()
