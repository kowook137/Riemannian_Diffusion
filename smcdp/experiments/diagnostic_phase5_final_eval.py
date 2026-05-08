"""Final evaluation of best Phase 5.3 config across multiple success radii.

Best config from sweep:
  - apply_h_mode = last_half (h_indices = [8..15])
  - alpha = 100
  - guidance_scale (CFG) = 0
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="outputs/franka_traj_unet/ckpt_riemannian.pt")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--alpha", type=float, default=100.0)
    p.add_argument("--apply-h-mode", type=str, default="last_half")
    p.add_argument("--z-list", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20])
    p.add_argument("--success-radii", type=float, nargs="+",
                   default=[0.02, 0.05, 0.08, 0.10, 0.15])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="outputs/diagnostic/phase5_final.json")
    args = p.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
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

    net = TrajectoryScoreNetUNet(
        arm, H=a["H"], down_dims=tuple(a["down_dims"]),
        diffusion_step_embed_dim=a["diff_step_embed"],
        n_groups=a["unet_groups"], kernel_size=a["unet_kernel"],
        t_scale=a["t_scale"], goal_cond_dim=3,
    ).to(device)
    net.load_state_dict(ck["ema_state"]); net.eval()
    score_fn = TrajectoryScaledScoreFn(net, sde)

    H1 = a["H"] + 1
    d = arm.ambient_dim
    if args.apply_h_mode == "last_only":
        h_indices = [H1 - 1]
    elif args.apply_h_mode == "last_quarter":
        h_indices = list(range(H1 - H1 // 4, H1))
    elif args.apply_h_mode == "last_half":
        h_indices = list(range(H1 - H1 // 2, H1))
    elif args.apply_h_mode == "all":
        h_indices = list(range(H1))
    else:
        h_indices = [H1 - 1]

    box_lo = torch.tensor(a["p_box_lo"], device=device)
    box_hi = torch.tensor(a["p_box_hi"], device=device)

    print(f"Phase 5 final eval — Ours-UNet + goal residual guidance")
    print(f"  ckpt = {args.ckpt}")
    print(f"  apply_h_mode = {args.apply_h_mode} (h_indices = {h_indices})")
    print(f"  alpha = {args.alpha}")
    print(f"  n = {args.n} per z_e\n")

    # Header
    rad_hdrs = "  ".join([f"r={r*100:.0f}cm" for r in args.success_radii])
    print(f"{'z_e':>5}  {'pos_err':>8}  {'med':>8}  {'std':>8}  {rad_hdrs}  {'frac_A':>7}  {'viol%':>6}  {'W1_A':>6}  {'W1_B':>6}  {'max|g|':>9}")

    results = {}
    for z_val in args.z_list:
        z_tensor = torch.full((args.n, 1), z_val, device=device)
        z_lim = z_tensor.unsqueeze(1).expand(-1, H1, -1).reshape(args.n * H1, -1)
        torch.manual_seed(args.seed + 1000)
        p_targets = box_lo + (box_hi - box_lo) * torch.rand(args.n, 3, device=device)
        torch.manual_seed(args.seed + 2000)
        tau_T = limiting.sample(args.n * H1, device=device, z_e=z_lim).reshape(args.n, H1, d)

        # Data reference for W1
        data_z = FrankaBimodalReachingDemo(
            manifold=arm, ik_arm=arm_analytic, H=a["H"],
            q_rest_A=list(a["q_rest_A"]), q_rest_B=list(a["q_rest_B"]),
            p_box_lo=tuple(a["p_box_lo"]), p_box_hi=tuple(a["p_box_hi"]),
            z_e_range=(z_val, z_val), branch_p_A=a["branch_p_A"],
            jitter_q=a["jitter_q"], n_ik_steps=a["n_ik_steps"],
        )
        x_data, _, _, _ = data_z.sample(args.n, device=device, p_target=p_targets)

        with torch.no_grad():
            tau_gen = traj_reverse_grw(sde, score_fn, tau_T,
                                       n_steps=200, eps=a["eps"],
                                       goal_cond=p_targets, guidance_scale=0.0,
                                       goal_residual_alpha=args.alpha,
                                       goal_residual_h=h_indices)

        p_end = tau_gen[:, -1, 7:10]
        pos_err = (p_end - p_targets).norm(dim=-1)
        succ_at = {f"r={r*100:.0f}cm": (pos_err < r).float().mean().item()
                   for r in args.success_radii}

        # mode capture
        h_mid = H1 // 2
        q1_mid = tau_gen[:, h_mid, 0]
        q1_mid_data = x_data[:, h_mid, 0]
        mode_A_data = q1_mid_data > 0
        mode_A = q1_mid > 0
        frac_A = mode_A.float().mean().item()

        # per-mode W1
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
        max_g = arm.constraint(tau_gen.reshape(-1, d)).norm(dim=-1).max().item()

        rad_vals = "  ".join([f"{succ_at[k]*100:>5.1f}%" for k in
                              [f"r={r*100:.0f}cm" for r in args.success_radii]])
        ood = " (OOD)" if (z_val < a["z_min"] or z_val > a["z_max"]) else ""
        print(f"{z_val:>5.3f}  {pos_err.mean().item():>7.4f}m {pos_err.median().item():>7.4f}m "
              f"{pos_err.std().item():>7.4f}m  {rad_vals}  {frac_A:>7.3f}  {viol*100:>5.1f}%  "
              f"{w1A:>6.3f}  {w1B:>6.3f}  {max_g:>9.1e}{ood}")

        results[f"z={z_val}"] = {
            "pos_err_mean": pos_err.mean().item(),
            "pos_err_med":  pos_err.median().item(),
            "pos_err_std":  pos_err.std().item(),
            **succ_at,
            "frac_A": frac_A,
            "W1_A": w1A,
            "W1_B": w1B,
            "viol": viol,
            "max_g": max_g,
        }

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
