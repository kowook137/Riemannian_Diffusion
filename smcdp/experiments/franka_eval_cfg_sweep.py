"""Sweep CFG guidance_scale on the trained Ours-UNet ckpt and report metrics.

Cost: each w requires one full reverse SDE sweep over n_eval_per_z×len(z_eval) samples.
"""
from __future__ import annotations

import argparse
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="outputs/franka_traj_unet/ckpt_riemannian.pt")
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--n-sample-steps", type=int, default=200)
    p.add_argument("--success-radius", type=float, default=0.02)
    p.add_argument("--w-list", type=float, nargs="+", default=[0.0, 1.0, 2.0, 4.0, 6.0, 10.0])
    p.add_argument("--z-list", type=float, nargs="+", default=[0.10])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    device = "cuda"
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
                              tool_z_max=max(a["z_max"], a["z_eval"][-1]) + 0.05,
                              metric=a["metric"])
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

    net = TrajectoryScoreNetUNet(
        arm, H=a["H"], down_dims=tuple(a["down_dims"]),
        diffusion_step_embed_dim=a["diff_step_embed"],
        n_groups=a["unet_groups"], kernel_size=a["unet_kernel"],
        t_scale=a["t_scale"], goal_cond_dim=3,
    ).to(device)
    net.load_state_dict(ck["ema_state"])
    net.eval()
    score_fn = TrajectoryScaledScoreFn(net, sde)

    H1 = a["H"] + 1
    d = arm.ambient_dim
    h_mid = H1 // 2
    n = args.n

    box_lo = torch.tensor(a["p_box_lo"], device=device)
    box_hi = torch.tensor(a["p_box_hi"], device=device)

    print(f"{'z_e':>5} {'w':>5}  {'pos_err':>8} {'success':>8} {'frac_A':>7} {'W1_A':>6} {'W1_B':>6} {'viol%':>6}")
    for z_val in args.z_list:
        # Single seed for fair comparison across w
        torch.manual_seed(args.seed + 1000)
        p_targets = box_lo + (box_hi - box_lo) * torch.rand(n, 3, device=device)
        z_tensor = torch.full((n, 1), z_val, device=device)
        z_lim = z_tensor.unsqueeze(1).expand(-1, H1, -1).reshape(n * H1, -1)
        torch.manual_seed(args.seed)
        tau_T = limiting.sample(n * H1, device=device, z_e=z_lim).reshape(n, H1, d)

        # Data reference
        data_z = FrankaBimodalReachingDemo(
            manifold=arm, ik_arm=arm_analytic, H=a["H"],
            q_rest_A=list(a["q_rest_A"]), q_rest_B=list(a["q_rest_B"]),
            p_box_lo=tuple(a["p_box_lo"]), p_box_hi=tuple(a["p_box_hi"]),
            z_e_range=(z_val, z_val), branch_p_A=a["branch_p_A"],
            jitter_q=a["jitter_q"], n_ik_steps=a["n_ik_steps"],
        )
        x_data, _, _, _ = data_z.sample(n, device=device, p_target=p_targets)

        for w in args.w_list:
            with torch.no_grad():
                tau_gen = traj_reverse_grw(sde, score_fn, tau_T,
                                           n_steps=args.n_sample_steps, eps=a["eps"],
                                           goal_cond=p_targets, guidance_scale=w)
            p_end = tau_gen[:, -1, 7:10]
            pos_err = (p_end - p_targets).norm(dim=-1)
            succ = (pos_err < args.success_radius).float().mean().item()

            q1_mid = tau_gen[:, h_mid, 0]
            q1_mid_data = x_data[:, h_mid, 0]
            mode_A_data = q1_mid_data > 0
            mode_A = q1_mid > 0
            frac_A = mode_A.float().mean().item()

            n_dir = 64
            dirs = torch.randn(n_dir, 7, device=device)
            dirs = dirs / dirs.norm(dim=-1, keepdim=True)
            def per_mode_w1(ta, tb):
                m = min(ta.shape[0], tb.shape[0])
                if m < 8: return float("nan")
                ia = torch.randperm(ta.shape[0], device=device)[:m]
                ib = torch.randperm(tb.shape[0], device=device)[:m]
                a_, b_ = ta[ia, :, :7], tb[ib, :, :7]
                ws = []
                for h in range(H1):
                    pa = (a_[:, h] @ dirs.T).sort(dim=0).values
                    pb = (b_[:, h] @ dirs.T).sort(dim=0).values
                    ws.append((pa - pb).abs().mean().item())
                return sum(ws) / len(ws)
            w1A = per_mode_w1(tau_gen[mode_A], x_data[mode_A_data])
            w1B = per_mode_w1(tau_gen[~mode_A], x_data[~mode_A_data])
            viol = arm.violates_limits(tau_gen.reshape(-1, d)[..., :7]).float().mean().item()

            print(f"{z_val:>5.3f} {w:>5.1f}  "
                  f"{pos_err.mean().item():>8.4f} "
                  f"{succ:>8.3f} "
                  f"{frac_A:>7.3f} "
                  f"{w1A:>6.3f} "
                  f"{w1B:>6.3f} "
                  f"{viol*100:>5.1f}")


if __name__ == "__main__":
    main()
