"""DP-B eval — DP-canonical ckpt + classifier guidance (sampling-time, no retrain).

Tests whether classifier guidance with multi-component reward (R_goal+R_start+R_vel)
applied to global-cond DP (canonical Chi23 architecture) is enough — isolating
the contribution of architecture (global vs channel-concat) given identical
sampling-time guidance.

Comparison structure:
  DP-canonical (global, no guide) → DP-B (global, +guide)         tests guide effect on global arch
  DP-A (channel, no guide)        → DP-C (channel, +guide)         tests guide effect on channel arch
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
from smcdp.baselines import (
    make_official_diffusion_policy, official_dp_sample_guided,
)


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="outputs/franka_baseline_dp_official/ckpt.pt",
                   help="DP-canonical ckpt (global cond)")
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--n-inference-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="outputs/diagnostic/dp_b_eval.json")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed)

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    a = ck["args"]
    s1 = ck["stage1_args"]
    print(f"DP-B eval — using ckpt {args.ckpt}")
    print(f"  cond_injection={a.get('cond_injection', 'global')}, baseline={a['baseline']}")
    assert a.get("cond_injection", "global") == "global", \
        f"DP-B requires global-cond ckpt, got cond_injection={a.get('cond_injection')}"

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
        n_q=n_q, global_cond_dim=7,
        down_dims=list(a["down_dims"]),
        diffusion_step_embed_dim=a["diff_step_embed"],
        n_train_timesteps=a["dp_train_timesteps"],
    )
    model = model.to(device)
    model.load_state_dict(ck["ema_state"]); model.eval()

    box_lo = torch.tensor(a["p_box_lo"], device=device)
    box_hi = torch.tensor(a["p_box_hi"], device=device)
    z_val = 0.10
    n = args.n
    z_e_per_traj = torch.full((n, 1), z_val, device=device)
    torch.manual_seed(args.seed + 1000)
    p_targets = box_lo + (box_hi - box_lo) * torch.rand(n, 3, device=device)
    torch.manual_seed(args.seed + 2000)
    p_starts = box_lo + (box_hi - box_lo) * torch.rand(n, 3, device=device)
    ctx = torch.cat([p_targets, p_starts, z_e_per_traj], dim=-1)

    print(f"\n{'cell':>40s}  {'pos_err':>9} {'med':>9} "
          f"{'s@2':>5} {'s@5':>5} {'s@10':>5} {'frac_A':>7} {'vel':>6} {'viol%':>5} {'max|g|':>9}")

    grh = list(range(H1 - H1 // 2, H1)); sah = [0]
    grid = [
        ("(a0) no guidance",           0,    0,    0   ),
        ("(a1) goal only α=10",        10,   0,    0   ),
        ("(a2) goal+start α=10/10",    10,   10,   0   ),
        ("(a3) all α=10/10/1",         10,   10,   1   ),
        ("(a4) all α=30/30/3",         30,   30,   3   ),
        ("(a5) all α=50/50/5",         50,   50,   5   ),
        ("(a6) all α=100/100/5",       100,  100,  5   ),
        ("(a7) all α=200/200/10",      200,  200,  10  ),
    ]

    rows = []
    for label, ag, as_, av in grid:
        q_gen = official_dp_sample_guided(
            model, scheduler, batch_size=n, horizon=H1, n_q=n_q,
            ctx=ctx, device=device, n_inference_steps=args.n_inference_steps,
            fk_fn=arm.F, z_e_per_traj=z_e_per_traj,
            p_target=p_targets, alpha_g=ag, h_indices_goal=grh,
            p_start_anchor=p_starts, alpha_s=as_, h_indices_start=sah,
            alpha_v=av,
        )
        z_flat = z_e_per_traj.unsqueeze(1).expand(-1, H1, -1).reshape(-1, 1)
        p_gen = arm.F(q_gen.reshape(-1, n_q), z_flat).reshape(n, H1, n_p)
        p_end = p_gen[:, -1, :]
        pos_err = (p_end - p_targets).norm(dim=-1)
        s2 = (pos_err < 0.02).float().mean().item()
        s5 = (pos_err < 0.05).float().mean().item()
        s10 = (pos_err < 0.10).float().mean().item()
        q1_mid = q_gen[:, h_mid, 0]
        frac_A = (q1_mid > 0).float().mean().item()
        vel = (q_gen[:, 1:] - q_gen[:, :-1]).norm(dim=-1).mean().item()
        viol = arm.violates_limits(q_gen.reshape(-1, n_q)).float().mean().item()
        p_analytic = arm_a.F(q_gen.reshape(-1, n_q), z_flat)
        p_learned = arm.F(q_gen.reshape(-1, n_q), z_flat)
        max_g = (p_analytic - p_learned).norm(dim=-1).max().item()
        rows.append({
            "label": label, "alpha_g": ag, "alpha_s": as_, "alpha_v": av,
            "pos_err_mean": pos_err.mean().item(),
            "pos_err_med": pos_err.median().item(),
            "succ_2cm": s2, "succ_5cm": s5, "succ_10cm": s10,
            "frac_A": frac_A, "vel_mean": vel, "viol": viol, "max_g_raw": max_g,
        })
        print(f"{label:>40s}  {pos_err.mean().item():>8.4f}m {pos_err.median().item():>8.4f}m "
              f"{s2*100:>4.1f}% {s5*100:>4.1f}% {s10*100:>4.1f}% "
              f"{frac_A:>7.3f} {vel:>6.3f} {viol*100:>4.1f}% {max_g:>9.1e}")

    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
