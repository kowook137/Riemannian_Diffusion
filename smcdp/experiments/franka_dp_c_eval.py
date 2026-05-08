"""DP-C eval — DP-A ckpt + classifier guidance (sampling-time, no retrain).

Same multi-component reward as Ours-V2:
  R_total = α_g R_goal + α_s R_start + α_v R_vel
  R_goal  = -‖F(q_H, z_e) - p_target‖²           (Euclidean, no G^{-1})
  R_start = -‖F(q_0, z_e) - p_start‖²
  R_vel   = -∑_h ‖q_{h+1} - q_h‖²

Sweep α values to find best DP-C config; compare to Ours-V2.
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
from smcdp.franka.demo_gen import FrankaBimodalReachingDemo
from smcdp.baselines import (
    make_official_diffusion_policy, channel_concat_dp_sample_guided,
)


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="outputs/franka_baseline_dp_official_channel/ckpt.pt",
                   help="DP-A ckpt (channel-cond DP)")
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--n-inference-steps", type=int, default=100)
    p.add_argument("--alpha-grid", type=str, default="sweep",
                   choices=["sweep", "single"])
    p.add_argument("--alpha-g", type=float, default=10.0)
    p.add_argument("--alpha-s", type=float, default=10.0)
    p.add_argument("--alpha-v", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="outputs/diagnostic/dp_c_eval.json")
    return p.parse_args()


def evaluate(model, scheduler, arm, arm_a, demo_gen, ctx_per_h, ctx, p_targets, p_starts,
             z_e_per_traj, args, alpha_g, alpha_s, alpha_v, n, H1, n_q, n_p, device, h_mid):
    n_eval_inference = args.n_inference_steps
    grh = list(range(H1 - H1 // 2, H1))
    sah = [0]

    z_flat = z_e_per_traj.unsqueeze(1).expand(-1, H1, -1).reshape(-1, 1)

    q_gen = channel_concat_dp_sample_guided(
        model, scheduler, batch_size=n, horizon=H1, n_q=n_q,
        ctx_per_h=ctx_per_h, device=device, n_inference_steps=n_eval_inference,
        fk_fn=arm.F,
        z_e_per_traj=z_e_per_traj,
        p_target=p_targets,
        alpha_g=alpha_g, h_indices_goal=grh,
        p_start_anchor=p_starts,
        alpha_s=alpha_s, h_indices_start=sah,
        alpha_v=alpha_v,
    )
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

    # max|g_φ| (q vs F_φ)
    p_analytic = arm_a.F(q_gen.reshape(-1, n_q), z_flat)
    p_learned = arm.F(q_gen.reshape(-1, n_q), z_flat)
    max_g_raw = (p_analytic - p_learned).norm(dim=-1).max().item()

    return {
        "alpha_g": alpha_g, "alpha_s": alpha_s, "alpha_v": alpha_v,
        "pos_err_mean": pos_err.mean().item(), "pos_err_med": pos_err.median().item(),
        "pos_err_std": pos_err.std().item(),
        "succ_2cm": s2, "succ_5cm": s5, "succ_10cm": s10,
        "frac_A": frac_A, "vel_mean": vel, "viol": viol, "max_g_raw": max_g_raw,
    }


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed)

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    a = ck["args"]
    s1 = ck["stage1_args"]
    print(f"DP-C eval — using ckpt {args.ckpt}")
    print(f"  cond_injection={a.get('cond_injection', 'global')}, baseline={a['baseline']}")

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
    CTX_DIM = 7

    model, scheduler = make_official_diffusion_policy(
        n_q=n_q + CTX_DIM, global_cond_dim=None,
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
    ctx_per_h = ctx.unsqueeze(1).expand(-1, H1, -1)

    print(f"\n{'cell':>40s}  {'pos_err':>9} {'med':>9} "
          f"{'s@2':>5} {'s@5':>5} {'s@10':>5} {'frac_A':>7} {'vel':>6} {'viol%':>5} {'max|g|':>9}")

    rows = []
    if args.alpha_grid == "sweep":
        grid = [
            ("(b0) no guidance",            0,    0,    0   ),
            ("(b1) goal only α=10",         10,   0,    0   ),
            ("(b2) goal+start α=10/10",     10,   10,   0   ),
            ("(b3) all α=10/10/1",          10,   10,   1   ),
            ("(b4) goal+start+vel α=30/30/3", 30, 30,   3   ),
            ("(b5) goal+start+vel α=50/50/5", 50, 50,   5   ),
            ("(b6) goal+start+vel α=100/100/5", 100, 100, 5 ),
            ("(b7) goal+start+vel α=200/200/10", 200, 200, 10 ),
        ]
    else:
        grid = [(f"(custom) α={args.alpha_g}/{args.alpha_s}/{args.alpha_v}",
                 args.alpha_g, args.alpha_s, args.alpha_v)]

    for label, ag, as_, av in grid:
        r = evaluate(model, scheduler, arm, arm_a, None, ctx_per_h, ctx,
                     p_targets, p_starts, z_e_per_traj, args, ag, as_, av,
                     n, H1, n_q, n_p, device, h_mid)
        r["label"] = label
        rows.append(r)
        print(f"{label:>40s}  {r['pos_err_mean']:>8.4f}m {r['pos_err_med']:>8.4f}m "
              f"{r['succ_2cm']*100:>4.1f}% {r['succ_5cm']*100:>4.1f}% {r['succ_10cm']*100:>4.1f}% "
              f"{r['frac_A']:>7.3f} {r['vel_mean']:>6.3f} {r['viol']*100:>4.1f}% {r['max_g_raw']:>9.1e}")

    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
