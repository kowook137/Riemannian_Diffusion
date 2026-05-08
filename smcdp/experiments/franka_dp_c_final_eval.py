"""DP-C final eval — best config across all z_e (in-dist + OOD) for paper table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import pybullet_data

from smcdp.manifolds import Franka7DoF
from smcdp.toy3.self_model import DeltaResidualMLP
from smcdp.franka.self_model import LearnedSelfModelFranka7DoF
from smcdp.franka.demo_gen import FrankaBimodalReachingDemo
from smcdp.baselines import (
    make_official_diffusion_policy, channel_concat_dp_sample_guided,
)


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="outputs/franka_baseline_dp_official_channel/ckpt.pt")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--n-inference-steps", type=int, default=100)
    p.add_argument("--alpha-g", type=float, default=30.0)
    p.add_argument("--alpha-s", type=float, default=30.0)
    p.add_argument("--alpha-v", type=float, default=3.0)
    p.add_argument("--z-list", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20])
    p.add_argument("--success-radii", type=float, nargs="+",
                   default=[0.02, 0.05, 0.08, 0.10, 0.15])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="outputs/diagnostic/dp_c_final.json")
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
    arm = LearnedSelfModelFranka7DoF(
        delta_net=delta_net, urdf_path=URDF, end_link="panda_hand", tool_z_max=0.25,
    )
    arm._ensure_chain(torch.zeros(1, 7, device=device))
    arm_a = Franka7DoF(urdf_path=URDF, end_link="panda_hand", tool_z_max=0.25)
    arm_a._ensure_chain(torch.zeros(1, 7, device=device))

    H1 = a["H"] + 1
    n_q, n_p = 7, 3
    h_mid = H1 // 2

    model, scheduler = make_official_diffusion_policy(
        n_q=n_q + 7, global_cond_dim=None,
        down_dims=list(a["down_dims"]),
        diffusion_step_embed_dim=a["diff_step_embed"],
        n_train_timesteps=a["dp_train_timesteps"],
    )
    model = model.to(device)
    model.load_state_dict(ck["ema_state"]); model.eval()

    box_lo = torch.tensor(a["p_box_lo"], device=device)
    box_hi = torch.tensor(a["p_box_hi"], device=device)
    grh = list(range(H1 - H1 // 2, H1)); sah = [0]

    rad_hdrs = "  ".join([f"r={r*100:.0f}cm" for r in args.success_radii])
    print(f"DP-C final  α_g={args.alpha_g} α_s={args.alpha_s} α_v={args.alpha_v}  n={args.n}\n")
    print(f"{'z_e':>5}  {'pos_err':>8}  {'med':>8}  {'std':>8}  {rad_hdrs}  "
          f"{'frac_A':>7}  {'vel':>6}  {'viol%':>5}  {'W1_A':>6}  {'W1_B':>6}  {'max|g|':>9}")

    results = {}
    for z_val in args.z_list:
        n = args.n
        z_e_per_traj = torch.full((n, 1), z_val, device=device)
        torch.manual_seed(args.seed + 1000)
        p_targets = box_lo + (box_hi - box_lo) * torch.rand(n, 3, device=device)
        torch.manual_seed(args.seed + 2000)
        p_starts = box_lo + (box_hi - box_lo) * torch.rand(n, 3, device=device)
        ctx = torch.cat([p_targets, p_starts, z_e_per_traj], dim=-1)
        ctx_per_h = ctx.unsqueeze(1).expand(-1, H1, -1)

        # data ref for W1
        data_z = FrankaBimodalReachingDemo(
            manifold=arm, ik_arm=arm_a, H=a["H"],
            q_rest_A=list(a["q_rest_A"]), q_rest_B=list(a["q_rest_B"]),
            p_box_lo=tuple(a["p_box_lo"]), p_box_hi=tuple(a["p_box_hi"]),
            z_e_range=(z_val, z_val), branch_p_A=a["branch_p_A"],
            jitter_q=a["jitter_q"], n_ik_steps=a["n_ik_steps"],
        )
        x_data, _, _, _, _ = data_z.sample(n, device=device, p_target=p_targets, p_start=p_starts)

        q_gen = channel_concat_dp_sample_guided(
            model, scheduler, batch_size=n, horizon=H1, n_q=n_q,
            ctx_per_h=ctx_per_h, device=device, n_inference_steps=args.n_inference_steps,
            fk_fn=arm.F, z_e_per_traj=z_e_per_traj,
            p_target=p_targets, alpha_g=args.alpha_g, h_indices_goal=grh,
            p_start_anchor=p_starts, alpha_s=args.alpha_s, h_indices_start=sah,
            alpha_v=args.alpha_v,
        )

        z_flat = z_e_per_traj.unsqueeze(1).expand(-1, H1, -1).reshape(-1, 1)
        p_gen = arm.F(q_gen.reshape(-1, n_q), z_flat).reshape(n, H1, n_p)
        p_end = p_gen[:, -1, :]
        pos_err = (p_end - p_targets).norm(dim=-1)
        succ_at = {f"r={r*100:.0f}cm": (pos_err < r).float().mean().item()
                   for r in args.success_radii}

        q1_mid = q_gen[:, h_mid, 0]
        q1_mid_data = x_data[:, h_mid, 0]
        mode_A_data = q1_mid_data > 0
        mode_A = q1_mid > 0
        frac_A = mode_A.float().mean().item()

        # per-mode W1 (q-block)
        n_dir = 64
        dirs = torch.randn(n_dir, 7, device=device)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        def per_mode_w1(ta_q, tb_x):
            m = min(ta_q.shape[0], tb_x.shape[0])
            if m < 8: return float("nan")
            ia = torch.randperm(ta_q.shape[0], device=device)[:m]
            ib = torch.randperm(tb_x.shape[0], device=device)[:m]
            a_, b_ = ta_q[ia], tb_x[ib, :, :7]
            ws = []
            for h in range(H1):
                pa = (a_[:, h] @ dirs.T).sort(dim=0).values
                pb = (b_[:, h] @ dirs.T).sort(dim=0).values
                ws.append((pa - pb).abs().mean().item())
            return sum(ws) / len(ws)
        w1A = per_mode_w1(q_gen[mode_A], x_data[mode_A_data]) if mode_A.any() and mode_A_data.any() else float("nan")
        w1B = per_mode_w1(q_gen[~mode_A], x_data[~mode_A_data]) if (~mode_A).any() and (~mode_A_data).any() else float("nan")

        vel = (q_gen[:, 1:] - q_gen[:, :-1]).norm(dim=-1).mean().item()
        viol = arm.violates_limits(q_gen.reshape(-1, n_q)).float().mean().item()
        p_analytic = arm_a.F(q_gen.reshape(-1, n_q), z_flat)
        p_learned = arm.F(q_gen.reshape(-1, n_q), z_flat)
        max_g = (p_analytic - p_learned).norm(dim=-1).max().item()

        rad_vals = "  ".join([f"{succ_at[k]*100:>5.1f}%" for k in
                              [f"r={r*100:.0f}cm" for r in args.success_radii]])
        ood = " (OOD)" if (z_val < a["z_min"] or z_val > a["z_max"]) else ""
        print(f"{z_val:>5.3f}  {pos_err.mean().item():>7.4f}m {pos_err.median().item():>7.4f}m "
              f"{pos_err.std().item():>7.4f}m  {rad_vals}  "
              f"{frac_A:>7.3f}  {vel:>6.3f}  {viol*100:>4.1f}%  "
              f"{w1A:>6.3f}  {w1B:>6.3f}  {max_g:>9.1e}{ood}")

        results[f"z={z_val}"] = {
            "pos_err_mean": pos_err.mean().item(), "pos_err_med": pos_err.median().item(),
            "pos_err_std": pos_err.std().item(),
            **succ_at, "frac_A": frac_A, "vel_mean": vel, "viol": viol,
            "W1_A": w1A, "W1_B": w1B, "max_g_raw": max_g,
        }

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
