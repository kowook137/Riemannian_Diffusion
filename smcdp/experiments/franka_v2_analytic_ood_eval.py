"""Quick eval-only script for Ours-Analytic at custom z_e values."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import pybullet_data

from smcdp.manifolds import Franka7DoF
from smcdp.franka.distributions import WrappedNormalFranka7DoF
from smcdp.franka.demo_gen import FrankaBimodalReachingDemo
from smcdp.sde import LangevinSDE, LinearBetaSchedule
from smcdp.trajectories import (
    TrajectoryScoreNetUNet, TrajectoryScaledScoreFn, traj_reverse_grw,
)


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="outputs/franka_traj_unet/ckpt_riemannian.pt".replace("ckpt_riemannian", "outputs/franka_traj_unet_v2_analytic/ckpt_riemannian"))
    p.add_argument("--ckpt-path", type=str, default="outputs/franka_traj_unet_v2_analytic/ckpt_riemannian.pt")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--z-list", type=float, nargs="+", default=[0.25])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="outputs/diagnostic/ours_analytic_ood25.json")
    args = p.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed)

    ck = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    a = ck["args"]
    print(f"Ours-Analytic OOD eval — ckpt {args.ckpt_path}")

    # Analytic-only manifold (matches training)
    arm = Franka7DoF(urdf_path=URDF, end_link="panda_hand", tool_z_max=0.30)
    arm._ensure_chain(torch.zeros(1, 7, device=device))
    arm_a = arm

    schedule = LinearBetaSchedule(beta_0=a["beta_0"], beta_f=a["beta_f"], t0=0.0, tf=1.0)
    limiting = WrappedNormalFranka7DoF(arm, mean_q=list(a["limiting_mean_q"]),
                                       scale=a["limiting_scale"],
                                       z_e_range=(a["z_min"], a["z_max"]))
    sde = LangevinSDE(arm, schedule, limiting)

    use_p_start = bool(a.get("use_p_start_cond", False))
    GOAL_DIM = 6 if use_p_start else 3
    cond_inj = a.get("cond_injection", "global")
    net = TrajectoryScoreNetUNet(
        arm, H=a["H"], down_dims=tuple(a["down_dims"]),
        diffusion_step_embed_dim=a["diff_step_embed"],
        n_groups=a["unet_groups"], kernel_size=a["unet_kernel"],
        t_scale=a["t_scale"], goal_cond_dim=GOAL_DIM, cond_injection=cond_inj,
    ).to(device)
    net.load_state_dict(ck["ema_state"]); net.eval()
    score_fn = TrajectoryScaledScoreFn(net, sde)

    H1 = a["H"] + 1
    h_mid = H1 // 2
    d = arm.ambient_dim
    box_lo = torch.tensor(a["p_box_lo"], device=device)
    box_hi = torch.tensor(a["p_box_hi"], device=device)
    grh = list(range(H1 - H1 // 2, H1)); sah = [0]

    print(f"\n{'z_e':>6}  {'pos_err':>9}  {'med':>9}  {'std':>9}  {'s@2':>5}  {'s@5':>5}  {'s@10':>5}  "
          f"{'frac_A':>7}  {'W1_A':>6}  {'W1_B':>6}  {'vel':>6}  {'viol%':>5}  {'max|g|':>9}")

    results = {}
    for z_val in args.z_list:
        n = args.n
        z_t = torch.full((n, 1), z_val, device=device)
        z_lim = z_t.unsqueeze(1).expand(-1, H1, -1).reshape(n * H1, -1)
        torch.manual_seed(args.seed + 1000)
        p_targets = box_lo + (box_hi - box_lo) * torch.rand(n, 3, device=device)
        torch.manual_seed(args.seed + 2000)
        p_starts = box_lo + (box_hi - box_lo) * torch.rand(n, 3, device=device)
        torch.manual_seed(args.seed + 3000)
        tau_T = limiting.sample(n * H1, device=device, z_e=z_lim).reshape(n, H1, d)

        cond_for_eval = torch.cat([p_targets, p_starts], dim=-1) if use_p_start else p_targets

        # data ref
        data_z = FrankaBimodalReachingDemo(
            manifold=arm, ik_arm=arm_a, H=a["H"],
            q_rest_A=list(a["q_rest_A"]), q_rest_B=list(a["q_rest_B"]),
            p_box_lo=tuple(a["p_box_lo"]), p_box_hi=tuple(a["p_box_hi"]),
            z_e_range=(z_val, z_val), branch_p_A=a["branch_p_A"],
            jitter_q=a["jitter_q"], n_ik_steps=a["n_ik_steps"],
        )
        x_data, _, _, _, _ = data_z.sample(n, device=device, p_target=p_targets, p_start=p_starts)

        with torch.no_grad():
            tau_gen = traj_reverse_grw(
                sde, score_fn, tau_T,
                n_steps=a["n_sample_steps"], eps=a["eps"],
                goal_cond=cond_for_eval, guidance_scale=a.get("guidance_scale", 0.0),
                goal_residual_alpha=a.get("goal_residual_alpha", 100.0), goal_residual_h=grh,
                p_start=p_starts,
                start_anchor_alpha=a.get("start_anchor_alpha", 100.0), start_anchor_h=sah,
                smoothness_alpha_vel=a.get("smoothness_alpha_vel", 5.0),
                smoothness_alpha_acc=a.get("smoothness_alpha_acc", 0.0),
            )

        max_g = arm.constraint(tau_gen.reshape(-1, d)).norm(dim=-1).max().item()
        p_end = tau_gen[:, -1, 7:10]
        pos_err = (p_end - p_targets).norm(dim=-1)
        s2 = (pos_err < 0.02).float().mean().item()
        s5 = (pos_err < 0.05).float().mean().item()
        s10 = (pos_err < 0.10).float().mean().item()

        q1_mid = tau_gen[:, h_mid, 0]
        q1_mid_data = x_data[:, h_mid, 0]
        mode_A = q1_mid > 0
        mode_A_data = q1_mid_data > 0
        frac_A = mode_A.float().mean().item()
        n_dir = 64
        dirs = torch.randn(n_dir, 7, device=device); dirs = dirs / dirs.norm(dim=-1, keepdim=True)
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
        w1A = per_mode_w1(tau_gen[mode_A], x_data[mode_A_data]) if mode_A.any() and mode_A_data.any() else float("nan")
        w1B = per_mode_w1(tau_gen[~mode_A], x_data[~mode_A_data]) if (~mode_A).any() and (~mode_A_data).any() else float("nan")
        viol = arm.violates_limits(tau_gen.reshape(-1, d)[..., :7]).float().mean().item()
        vel = (tau_gen[:, 1:, :7] - tau_gen[:, :-1, :7]).norm(dim=-1).mean().item()

        ood = " (OOD)" if (z_val < a["z_min"] or z_val > a["z_max"]) else ""
        print(f"{z_val:>6.3f}  {pos_err.mean().item():>8.4f}m  {pos_err.median().item():>8.4f}m  "
              f"{pos_err.std().item():>8.4f}m  {s2*100:>4.1f}%  {s5*100:>4.1f}%  {s10*100:>4.1f}%  "
              f"{frac_A:>7.3f}  {w1A:>6.3f}  {w1B:>6.3f}  {vel:>6.3f}  {viol*100:>4.1f}%  {max_g:>9.1e}{ood}")

        results[f"z={z_val}"] = {
            "pos_err_mean": pos_err.mean().item(), "pos_err_med": pos_err.median().item(),
            "pos_err_std": pos_err.std().item(),
            "succ_2cm": s2, "succ_5cm": s5, "succ_10cm": s10,
            "frac_A": frac_A, "W1_A": w1A, "W1_B": w1B,
            "vel_mean": vel, "viol": viol, "max_g": max_g,
        }

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
