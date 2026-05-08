"""diagnostic_plan.md Phase 1.5 — Sampling discretization + EMA comparison.

Diagnoses (no retrain):
  3.1 Reverse step count K ∈ {100, 200, 400, 800, 1600} sweep
  3.2 EMA vs raw model comparison

판단 기준 (diagnostic_plan §3.3):
    | K=200 → K=800 pos_err diff | 결론 |
    | > 0.05m 감소               | Sampling main bottleneck |
    | < 0.02m 감소               | Sampling sufficient → Phase 2 |
    | 0.02-0.05m                 | Partial contribution |
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
    p.add_argument("--K-list", type=int, nargs="+", default=[100, 200, 400, 800, 1600])
    p.add_argument("--success-radius", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="outputs/diagnostic/phase1_5.json")
    return p.parse_args()


def evaluate_with_state(
    arm, sde, score_fn, limiting, n, p_targets, z_val, K_steps, eps,
    success_radius, device,
):
    H1 = score_fn.net.H + 1
    d = arm.ambient_dim
    z_tensor = torch.full((n, 1), z_val, device=device)
    z_lim = z_tensor.unsqueeze(1).expand(-1, H1, -1).reshape(n * H1, -1)
    torch.manual_seed(0)
    tau_T = limiting.sample(n * H1, device=device, z_e=z_lim).reshape(n, H1, d)
    with torch.no_grad():
        tau_gen = traj_reverse_grw(sde, score_fn, tau_T,
                                   n_steps=K_steps, eps=eps,
                                   goal_cond=p_targets, guidance_scale=0.0)
    p_end = tau_gen[:, -1, 7:10]
    pos_err = (p_end - p_targets).norm(dim=-1)
    succ = (pos_err < success_radius).float().mean().item()
    g = arm.constraint(tau_gen.reshape(-1, d)).norm(dim=-1).max().item()
    return {
        "pos_err_mean": pos_err.mean().item(),
        "pos_err_med":  pos_err.median().item(),
        "pos_err_std":  pos_err.std().item(),
        "succ":         succ,
        "max_g":        g,
    }


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

    # nets
    def make_net():
        return TrajectoryScoreNetUNet(
            arm, H=a["H"], down_dims=tuple(a["down_dims"]),
            diffusion_step_embed_dim=a["diff_step_embed"],
            n_groups=a["unet_groups"], kernel_size=a["unet_kernel"],
            t_scale=a["t_scale"], goal_cond_dim=3,
        ).to(device)
    net_ema = make_net(); net_ema.load_state_dict(ck["ema_state"]); net_ema.eval()
    net_raw = make_net(); net_raw.load_state_dict(ck["net_state"]); net_raw.eval()
    score_ema = TrajectoryScaledScoreFn(net_ema, sde)
    score_raw = TrajectoryScaledScoreFn(net_raw, sde)

    # Targets (in-distribution)
    box_lo = torch.tensor(a["p_box_lo"], device=device)
    box_hi = torch.tensor(a["p_box_hi"], device=device)
    z_val = 0.10
    torch.manual_seed(args.seed + 1)
    p_targets = box_lo + (box_hi - box_lo) * torch.rand(args.n, 3, device=device)

    print(f"Phase 1.5  K sweep at z_e={z_val}, n={args.n}\n")
    print(f"{'model':>6} {'K':>5}  {'pos_err':>9} {'med':>9} {'std':>9} {'succ%':>7} {'max|g|':>9}")

    results = {}
    for label, score_fn in [("EMA", score_ema), ("RAW", score_raw)]:
        results[label] = {}
        for K in args.K_list:
            r = evaluate_with_state(arm, sde, score_fn, limiting, args.n, p_targets, z_val,
                                    K, a["eps"], args.success_radius, device)
            results[label][K] = r
            print(f"{label:>6} {K:>5}  {r['pos_err_mean']:>8.4f}m {r['pos_err_med']:>8.4f}m "
                  f"{r['pos_err_std']:>8.4f}m {r['succ']*100:>6.1f}% {r['max_g']:>9.1e}")

    # Decision
    diff_K200_K800_ema = results["EMA"][200]["pos_err_mean"] - results["EMA"][800]["pos_err_mean"]
    print("\n--- Decision (diagnostic_plan §3.3) ---")
    if diff_K200_K800_ema > 0.05:
        verdict = (f"K=200→K=800 pos_err 감소 {diff_K200_K800_ema:.3f}m (>0.05m): "
                   "Sampling main bottleneck → use K=800+ for evaluation")
    elif diff_K200_K800_ema > 0.02:
        verdict = (f"K=200→K=800 pos_err 감소 {diff_K200_K800_ema:.3f}m (0.02-0.05m): "
                   "Sampling partial contribution; proceed to Phase 2")
    else:
        verdict = (f"K=200→K=800 pos_err 감소 {diff_K200_K800_ema:.3f}m (<0.02m): "
                   "Sampling sufficient at K=200; main bottleneck elsewhere → Phase 2")
    print(f"  {verdict}")

    diff_ema_raw_at_K200 = results["RAW"][200]["pos_err_mean"] - results["EMA"][200]["pos_err_mean"]
    print(f"  EMA-vs-RAW pos_err diff @K=200: {diff_ema_raw_at_K200:+.4f}m  "
          f"({'EMA helps' if diff_ema_raw_at_K200 > 0.005 else 'EMA neutral'})")

    summary = {
        "config": {
            "ckpt": args.ckpt, "n": args.n, "K_list": args.K_list,
            "z_val": z_val, "success_radius": args.success_radius,
        },
        "results": results,
        "K_diff_200_800_EMA": diff_K200_K800_ema,
        "EMA_minus_RAW_at_K200": diff_ema_raw_at_K200,
        "verdict": verdict,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
