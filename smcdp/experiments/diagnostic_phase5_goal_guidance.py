"""diagnostic_plan.md Phase 5.3 — Goal residual guidance sweep (no retrain).

Sweeps `goal_residual_alpha` and (last-only vs ramped weight across H+1) to find
the configuration that minimises pos_err while preserving:
  - manifold adherence (max ‖g_φ‖ = 0)
  - mode capture (frac_A near 0.5)
  - joint limit violation (low)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import pybullet_data

from smcdp.manifolds import Franka7DoF
from smcdp.toy3.self_model import DeltaResidualMLP
from smcdp.franka.self_model import LearnedSelfModelFranka7DoF
from smcdp.franka.distributions import WrappedNormalFranka7DoF
from smcdp.franka.demo_gen import FrankaBimodalReachingDemo
from smcdp.sde import LangevinSDE, LinearBetaSchedule
from smcdp.trajectories import (
    TrajectoryScoreNetUNet, TrajectoryScaledScoreFn, traj_reverse_grw,
)


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="outputs/franka_traj_unet/ckpt_riemannian.pt")
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--n-sample-steps", type=int, default=200)
    p.add_argument("--success-radius", type=float, default=0.02)
    p.add_argument("--alpha-list", type=float, nargs="+",
                   default=[0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    p.add_argument("--apply-h-mode", type=str, default="last_only",
                   choices=["last_only", "all", "last_quarter", "last_half"])
    p.add_argument("--z-list", type=float, nargs="+", default=[0.10])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="outputs/diagnostic/phase5_goal_guidance.json")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    a = ck["args"]
    s1 = ck["stage1_args"]

    delta_net = DeltaResidualMLP(n_q=7, n_p=3, n_z=1,
                                  hidden=s1["hidden"], n_layers=s1["n_layers"],
                                  activation=torch.nn.Softplus,
                                  final_init_scale=1e-3).to(device)
    stage1_ck = torch.load("outputs/franka_stage1/delta_phi.pt",
                           map_location=device, weights_only=False)
    delta_net.load_state_dict(stage1_ck["delta_net_state"])
    delta_net.eval()

    arm_analytic = Franka7DoF(urdf_path=URDF, end_link="panda_hand",
                              tool_z_max=max(a["z_max"], a["z_eval"][-1]) + 0.05)
    arm_analytic._ensure_chain(torch.zeros(1, 7, device=device))
    arm = LearnedSelfModelFranka7DoF(
        delta_net=delta_net, urdf_path=URDF, end_link="panda_hand",
        tool_z_max=max(a["z_max"], a["z_eval"][-1]) + 0.05, metric=a["metric"],
    )
    arm._ensure_chain(torch.zeros(1, 7, device=device))

    schedule = LinearBetaSchedule(beta_0=a["beta_0"], beta_f=a["beta_f"], t0=0.0, tf=1.0)
    limiting = WrappedNormalFranka7DoF(arm, mean_q=list(a["limiting_mean_q"]),
                                       scale=a["limiting_scale"],
                                       z_e_range=(a["z_min"], a["z_max"]))
    sde = LangevinSDE(arm, schedule, limiting)

    net_ema = TrajectoryScoreNetUNet(
        arm, H=a["H"], down_dims=tuple(a["down_dims"]),
        diffusion_step_embed_dim=a["diff_step_embed"],
        n_groups=a["unet_groups"], kernel_size=a["unet_kernel"],
        t_scale=a["t_scale"], goal_cond_dim=3,
    ).to(device)
    net_ema.load_state_dict(ck["ema_state"]); net_ema.eval()
    score_fn = TrajectoryScaledScoreFn(net_ema, sde)

    H1 = a["H"] + 1
    d = arm.ambient_dim

    box_lo = torch.tensor(a["p_box_lo"], device=device)
    box_hi = torch.tensor(a["p_box_hi"], device=device)

    if args.apply_h_mode == "last_only":
        h_indices = [H1 - 1]
    elif args.apply_h_mode == "all":
        h_indices = list(range(H1))
    elif args.apply_h_mode == "last_quarter":
        h_indices = list(range(H1 - H1 // 4, H1))
    elif args.apply_h_mode == "last_half":
        h_indices = list(range(H1 - H1 // 2, H1))
    else:
        h_indices = [H1 - 1]

    print(f"Phase 5.3 — Goal residual guidance sweep")
    print(f"  ckpt = {args.ckpt}")
    print(f"  n = {args.n}, n_sample_steps = {args.n_sample_steps}")
    print(f"  apply_h_mode = {args.apply_h_mode} → h_indices = {h_indices}")
    print()
    print(f"{'z_e':>5} {'α':>6}  {'pos_err':>9} {'med':>9} {'std':>9} {'succ%':>6} "
          f"{'frac_A':>7} {'viol%':>6} {'max|g|':>9}")

    results = {}
    for z_val in args.z_list:
        results[f"z={z_val}"] = {}
        # Pre-sample targets and noise for this z_e (same across α for fair comparison)
        z_tensor = torch.full((args.n, 1), z_val, device=device)
        z_lim = z_tensor.unsqueeze(1).expand(-1, H1, -1).reshape(args.n * H1, -1)
        torch.manual_seed(args.seed + 1000)
        p_targets = box_lo + (box_hi - box_lo) * torch.rand(args.n, 3, device=device)
        torch.manual_seed(args.seed + 2000)
        tau_T = limiting.sample(args.n * H1, device=device, z_e=z_lim).reshape(args.n, H1, d)

        for alpha in args.alpha_list:
            with torch.no_grad():
                tau_gen = traj_reverse_grw(sde, score_fn, tau_T,
                                           n_steps=args.n_sample_steps, eps=a["eps"],
                                           goal_cond=p_targets, guidance_scale=0.0,
                                           goal_residual_alpha=alpha,
                                           goal_residual_h=h_indices)
            p_end = tau_gen[:, -1, 7:10]
            pos_err = (p_end - p_targets).norm(dim=-1)
            succ = (pos_err < args.success_radius).float().mean().item()
            q1_mid = tau_gen[:, H1 // 2, 0]
            mode_A = q1_mid > 0
            frac_A = mode_A.float().mean().item()
            g = arm.constraint(tau_gen.reshape(-1, d)).norm(dim=-1).max().item()
            viol = arm.violates_limits(tau_gen.reshape(-1, d)[..., :7]).float().mean().item()
            results[f"z={z_val}"][f"α={alpha}"] = {
                "pos_err_mean": pos_err.mean().item(),
                "pos_err_med":  pos_err.median().item(),
                "pos_err_std":  pos_err.std().item(),
                "succ":         succ,
                "frac_A":       frac_A,
                "viol":         viol,
                "max_g":        g,
            }
            print(f"{z_val:>5.3f} {alpha:>6.2f}  "
                  f"{pos_err.mean().item():>8.4f}m {pos_err.median().item():>8.4f}m "
                  f"{pos_err.std().item():>8.4f}m {succ*100:>5.1f}% "
                  f"{frac_A:>7.3f} {viol*100:>5.1f}% {g:>9.1e}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
