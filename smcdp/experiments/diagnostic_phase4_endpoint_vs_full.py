"""diagnostic_plan.md Phase 4.3 D — Endpoint vs Full-trajectory error.

Measures whether the model is sacrificing endpoint sharpness for trajectory-wide
fit, by comparing two error metrics on generated trajectories:

  - Full-trajectory pos_err: (1/(H+1)) Σ_h ‖p_h − p_target‖
  - Endpoint pos_err:        ‖p_H − p_target‖
  - Demo full-trajectory pos_err: same on demo data (as reference; demos are
                                  sharp at endpoint by IK design)

Decision (diagnostic_plan §6.4):
  endpoint err >> full traj err  → endpoint loss reweighting candidate (Phase 5.2)

Also measures:
  - Trajectory smoothness:  vel = ‖q_{h+1} − q_h‖²,  accel = ‖q_{h+1} − 2q_h + q_{h-1}‖²
  - G condition number κ(G) at random q's
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
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--n-sample-steps", type=int, default=200)
    p.add_argument("--alpha", type=float, default=100.0)
    p.add_argument("--apply-h-mode", type=str, default="last_half")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="outputs/diagnostic/phase4_endpoint.json")
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
    h_indices = list(range(H1 - H1 // 2, H1)) if args.apply_h_mode == "last_half" else [H1 - 1]

    box_lo = torch.tensor(a["p_box_lo"], device=device)
    box_hi = torch.tensor(a["p_box_hi"], device=device)
    z_val = 0.10
    z_tensor = torch.full((args.n, 1), z_val, device=device)
    z_lim = z_tensor.unsqueeze(1).expand(-1, H1, -1).reshape(args.n * H1, -1)
    torch.manual_seed(args.seed + 1000)
    p_targets = box_lo + (box_hi - box_lo) * torch.rand(args.n, 3, device=device)
    torch.manual_seed(args.seed + 2000)
    tau_T = limiting.sample(args.n * H1, device=device, z_e=z_lim).reshape(args.n, H1, d)

    # Generate samples (with Phase 5.3 guidance to match best config)
    print(f"Phase 4.3 D — Endpoint vs full-traj error (z_e={z_val}, n={args.n})\n")
    print(f"  config: alpha={args.alpha}, h_mode={args.apply_h_mode}\n")

    with torch.no_grad():
        tau_gen = traj_reverse_grw(sde, score_fn, tau_T,
                                   n_steps=args.n_sample_steps, eps=a["eps"],
                                   goal_cond=p_targets, guidance_scale=0.0,
                                   goal_residual_alpha=args.alpha,
                                   goal_residual_h=h_indices)

    # Demo reference (sharp endpoint by IK design)
    data_z = FrankaBimodalReachingDemo(
        manifold=arm, ik_arm=arm_analytic, H=a["H"],
        q_rest_A=list(a["q_rest_A"]), q_rest_B=list(a["q_rest_B"]),
        p_box_lo=tuple(a["p_box_lo"]), p_box_hi=tuple(a["p_box_hi"]),
        z_e_range=(z_val, z_val), branch_p_A=a["branch_p_A"],
        jitter_q=a["jitter_q"], n_ik_steps=a["n_ik_steps"],
    )
    x_data, _, _, _ = data_z.sample(args.n, device=device, p_target=p_targets)

    # Compute per-timestep p_ee on gen and data
    q_gen = tau_gen[..., :7]                                                       # (n, H+1, 7)
    z_e_gen = tau_gen[..., 7+3:7+3+1]                                              # (n, H+1, 1)
    p_gen = arm.F(q_gen.reshape(-1, 7), z_e_gen.reshape(-1, 1)).reshape(args.n, H1, 3)

    q_data = x_data[..., :7]
    z_e_data = x_data[..., 7+3:7+3+1]
    p_data = arm.F(q_data.reshape(-1, 7), z_e_data.reshape(-1, 1)).reshape(args.n, H1, 3)

    # Per-timestep target for full-traj error: linear interp from p_start (assumed = p_data[:,0])
    p_starts_data = p_data[:, 0, :]                                                # (n, 3)
    s = torch.linspace(0, 1, H1, device=device).view(1, H1, 1)
    p_target_per_h = p_starts_data.unsqueeze(1) + s * (p_targets - p_starts_data).unsqueeze(1)

    # Endpoint vs full-traj
    err_gen_per_h = (p_gen - p_target_per_h).norm(dim=-1)                          # (n, H+1)
    err_data_per_h = (p_data - p_target_per_h).norm(dim=-1)
    err_gen_endpoint = err_gen_per_h[:, -1]                                         # (n,)
    err_data_endpoint = err_data_per_h[:, -1]
    err_gen_full = err_gen_per_h.mean(dim=-1)                                       # (n,)
    err_data_full = err_data_per_h.mean(dim=-1)

    # Per-timestep mean
    print(f"Per-timestep pos_err to interpolated target (mean):")
    print(f"  {'h':>3}  {'gen':>10}  {'data':>10}")
    for h in [0, 4, 8, 12, 14, 15]:
        print(f"  {h:>3}  {err_gen_per_h[:, h].mean().item():>9.4f}m  "
              f"{err_data_per_h[:, h].mean().item():>9.4f}m")

    print(f"\nEndpoint pos_err (h=15):")
    print(f"  gen:  mean={err_gen_endpoint.mean():.4f}m  med={err_gen_endpoint.median():.4f}m")
    print(f"  data: mean={err_data_endpoint.mean():.4f}m  med={err_data_endpoint.median():.4f}m")
    print(f"\nFull-trajectory pos_err (mean over h):")
    print(f"  gen:  mean={err_gen_full.mean():.4f}m  med={err_gen_full.median():.4f}m")
    print(f"  data: mean={err_data_full.mean():.4f}m  med={err_data_full.median():.4f}m")

    ratio_endpoint_to_full = (err_gen_endpoint.mean() / err_gen_full.mean()).item()
    print(f"\n  endpoint/full ratio (gen): {ratio_endpoint_to_full:.3f}")

    # Smoothness
    vel = (q_gen[:, 1:] - q_gen[:, :-1]).norm(dim=-1)                               # (n, H)
    accel = (q_gen[:, 2:] - 2 * q_gen[:, 1:-1] + q_gen[:, :-2]).norm(dim=-1)        # (n, H-1)
    vel_data = (q_data[:, 1:] - q_data[:, :-1]).norm(dim=-1)
    accel_data = (q_data[:, 2:] - 2 * q_data[:, 1:-1] + q_data[:, :-2]).norm(dim=-1)

    print(f"\nTrajectory smoothness (joint-space):")
    print(f"  vel   gen:  mean={vel.mean():.4f}  max={vel.max():.4f}")
    print(f"  vel   data: mean={vel_data.mean():.4f}  max={vel_data.max():.4f}")
    print(f"  accel gen:  mean={accel.mean():.4f}  max={accel.max():.4f}")
    print(f"  accel data: mean={accel_data.mean():.4f}  max={accel_data.max():.4f}")

    # G condition number at random q in trajectory
    q_sample = q_gen.reshape(-1, 7)[::64]
    z_sample = z_e_gen.reshape(-1, 1)[::64]
    G = arm.G(q_sample, z_sample)
    eigs = torch.linalg.eigvalsh(G)
    kappa = (eigs.max(dim=-1).values / eigs.min(dim=-1).values.clamp(min=1e-12))
    print(f"\nG condition number κ(G):")
    print(f"  mean={kappa.mean():.2f}  max={kappa.max():.2f}  min={kappa.min():.2f}")

    # Decision
    print("\n--- Decision (diagnostic_plan §6.4) ---")
    if ratio_endpoint_to_full > 1.5:
        verdict = (f"Endpoint err {ratio_endpoint_to_full:.2f}x full-traj err "
                   f"→ endpoint sharpness 약함, **Phase 5.2 endpoint loss reweighting** 우선")
    elif ratio_endpoint_to_full > 1.1:
        verdict = (f"Endpoint err mildly higher ({ratio_endpoint_to_full:.2f}x) "
                   f"→ endpoint and full-traj both contribute, both Phase 5.2 and 6.1 candidates")
    else:
        verdict = (f"Endpoint ≈ full-traj error ({ratio_endpoint_to_full:.2f}x) "
                   f"→ trajectory-wide weakness, **Phase 6.1 channel concat** 우선")
    print(f"  {verdict}")

    summary = {
        "config": {"alpha": args.alpha, "h_mode": args.apply_h_mode, "n": args.n},
        "endpoint_pos_err": {
            "gen_mean":  err_gen_endpoint.mean().item(),
            "data_mean": err_data_endpoint.mean().item(),
        },
        "full_traj_pos_err": {
            "gen_mean":  err_gen_full.mean().item(),
            "data_mean": err_data_full.mean().item(),
        },
        "ratio_endpoint_to_full": ratio_endpoint_to_full,
        "smoothness": {
            "vel_gen_mean":   vel.mean().item(),
            "vel_data_mean":  vel_data.mean().item(),
            "accel_gen_mean": accel.mean().item(),
            "accel_data_mean": accel_data.mean().item(),
        },
        "G_condition": {
            "mean": kappa.mean().item(),
            "max":  kappa.max().item(),
            "min":  kappa.min().item(),
        },
        "verdict": verdict,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
