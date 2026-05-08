"""Noise robustness sweep — sampling-time observation noise on p_target cond.

Per Experiment_plan §2.2 Regime B: simulate noisy goal observation by perturbing
p_target at sampling time. Measure pos_err to TRUE p_target across σ_p levels.

Methods compared (in their respective regimes):
  - BC (compliance), Ours-V2 (compliance), DP-C (compliance) [no retrain]
  - BC-perfect, Ours-Analytic, DP-C-perfect [no retrain]
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
from smcdp.sde import LangevinSDE, LinearBetaSchedule
from smcdp.trajectories import (
    TrajectoryScoreNetUNet, TrajectoryScaledScoreFn, traj_reverse_grw,
)
from smcdp.baselines import (
    BCTrajectoryPredictor, make_official_diffusion_policy, official_dp_sample,
    channel_concat_dp_sample_guided,
)


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--sigma-list", type=float, nargs="+", default=[0.0, 0.001, 0.003, 0.005, 0.010])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="outputs/diagnostic/noise_robustness.json")
    args = p.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed)

    arm_a = Franka7DoF(urdf_path=URDF, end_link="panda_hand", tool_z_max=0.30)
    arm_a._ensure_chain(torch.zeros(1, 7, device=device))

    # Stage-1 self-model for compliance regime
    stage1_ck = torch.load("outputs/franka_stage1/delta_phi.pt",
                           map_location=device, weights_only=False)
    s1 = stage1_ck["args"]
    delta_net = DeltaResidualMLP(n_q=7, n_p=3, n_z=1,
                                  hidden=s1["hidden"], n_layers=s1["n_layers"],
                                  activation=torch.nn.Softplus,
                                  final_init_scale=1e-3).to(device)
    delta_net.load_state_dict(stage1_ck["delta_net_state"]); delta_net.eval()
    arm_learned = LearnedSelfModelFranka7DoF(
        delta_net=delta_net, urdf_path=URDF, end_link="panda_hand", tool_z_max=0.30,
    )
    arm_learned._ensure_chain(torch.zeros(1, 7, device=device))

    box_lo = torch.tensor([0.40, -0.05, 0.40], device=device)
    box_hi = torch.tensor([0.50, 0.05, 0.50], device=device)
    z_val = 0.10
    n = args.n
    z_t = torch.full((n, 1), z_val, device=device)

    torch.manual_seed(args.seed + 1000)
    p_targets_true = box_lo + (box_hi - box_lo) * torch.rand(n, 3, device=device)
    torch.manual_seed(args.seed + 2000)
    p_starts = box_lo + (box_hi - box_lo) * torch.rand(n, 3, device=device)

    print(f"Noise robustness sweep  z_e={z_val}  n={n}")
    print(f"  σ_p ∈ {args.sigma_list} (m)\n")

    results = {}

    # ---- Method: BC-perfect ----
    bc_ckpt = torch.load("outputs/franka_baseline_bc_perfect/ckpt.pt",
                          map_location=device, weights_only=False)
    bc_a = bc_ckpt["args"]
    bc = BCTrajectoryPredictor(n_q=7, H=bc_a["H"], ctx_dim=7,
                                hidden=bc_a["bc_hidden"], n_layers=bc_a["bc_layers"],
                                activation="sin").to(device)
    bc.load_state_dict(bc_ckpt["ema_state"]); bc.eval()
    H1_bc = bc_a["H"] + 1
    print(f"\n{'method':>22s}  {'σ_p':>7s}  {'pos_err':>9s}  {'med':>9s}  {'s@2':>5s}  {'s@5':>5s}  {'s@10':>5s}")
    bc_rows = []
    for sigma in args.sigma_list:
        torch.manual_seed(args.seed + 3000)
        noise = sigma * torch.randn(n, 3, device=device)
        p_targets_noisy = p_targets_true + noise
        ctx = torch.cat([p_targets_noisy, p_starts, z_t], dim=-1)
        with torch.no_grad():
            q_gen = bc(ctx)
        z_flat = z_t.unsqueeze(1).expand(-1, H1_bc, -1).reshape(-1, 1)
        p_end = arm_a.F(q_gen[:, -1, :], z_t)
        pos_err = (p_end - p_targets_true).norm(dim=-1)
        s2 = (pos_err < 0.02).float().mean().item()
        s5 = (pos_err < 0.05).float().mean().item()
        s10 = (pos_err < 0.10).float().mean().item()
        bc_rows.append({"sigma": sigma, "pos_err_mean": pos_err.mean().item(),
                        "succ_2cm": s2, "succ_5cm": s5, "succ_10cm": s10})
        print(f"{'BC-perfect':>22s}  {sigma*1000:>6.1f}mm  {pos_err.mean().item():>8.4f}m  "
              f"{pos_err.median().item():>8.4f}m  {s2*100:>4.1f}%  {s5*100:>4.1f}%  {s10*100:>4.1f}%")
    results["BC-perfect"] = bc_rows

    # ---- Method: DP-C-perfect ----
    print()
    dpa_ckpt = torch.load("outputs/franka_baseline_dp_official_channel_perfect/ckpt.pt",
                          map_location=device, weights_only=False)
    dpa_a = dpa_ckpt["args"]
    dp_model, dp_sched = make_official_diffusion_policy(
        n_q=7+7, global_cond_dim=None,
        down_dims=list(dpa_a["down_dims"]),
        diffusion_step_embed_dim=dpa_a["diff_step_embed"],
        n_train_timesteps=dpa_a["dp_train_timesteps"],
    )
    dp_model = dp_model.to(device); dp_model.load_state_dict(dpa_ckpt["ema_state"]); dp_model.eval()
    H1_dp = dpa_a["H"] + 1
    grh = list(range(H1_dp - H1_dp // 2, H1_dp)); sah = [0]
    dpc_rows = []
    for sigma in args.sigma_list:
        torch.manual_seed(args.seed + 3000)
        noise = sigma * torch.randn(n, 3, device=device)
        p_targets_noisy = p_targets_true + noise
        ctx = torch.cat([p_targets_noisy, p_starts, z_t], dim=-1)
        ctx_per_h = ctx.unsqueeze(1).expand(-1, H1_dp, -1)
        with torch.no_grad():
            q_gen = channel_concat_dp_sample_guided(
                dp_model, dp_sched, batch_size=n, horizon=H1_dp, n_q=7,
                ctx_per_h=ctx_per_h, device=device, n_inference_steps=100,
                fk_fn=arm_a.F, z_e_per_traj=z_t,
                p_target=p_targets_noisy, alpha_g=30.0, h_indices_goal=grh,
                p_start_anchor=p_starts, alpha_s=30.0, h_indices_start=sah,
                alpha_v=3.0,
            )
        p_end = arm_a.F(q_gen[:, -1, :], z_t)
        pos_err = (p_end - p_targets_true).norm(dim=-1)
        s2 = (pos_err < 0.02).float().mean().item()
        s5 = (pos_err < 0.05).float().mean().item()
        s10 = (pos_err < 0.10).float().mean().item()
        dpc_rows.append({"sigma": sigma, "pos_err_mean": pos_err.mean().item(),
                         "succ_2cm": s2, "succ_5cm": s5, "succ_10cm": s10})
        print(f"{'DP-C-perfect':>22s}  {sigma*1000:>6.1f}mm  {pos_err.mean().item():>8.4f}m  "
              f"{pos_err.median().item():>8.4f}m  {s2*100:>4.1f}%  {s5*100:>4.1f}%  {s10*100:>4.1f}%")
    results["DP-C-perfect"] = dpc_rows

    # ---- Method: Ours-Analytic ----
    print()
    ours_ckpt = torch.load("outputs/franka_traj_unet_v2_analytic/ckpt_riemannian.pt",
                            map_location=device, weights_only=False)
    a = ours_ckpt["args"]
    schedule = LinearBetaSchedule(beta_0=a["beta_0"], beta_f=a["beta_f"], t0=0.0, tf=1.0)
    limiting = WrappedNormalFranka7DoF(arm_a, mean_q=list(a["limiting_mean_q"]),
                                       scale=a["limiting_scale"],
                                       z_e_range=(a["z_min"], a["z_max"]))
    sde = LangevinSDE(arm_a, schedule, limiting)
    use_p_start = bool(a.get("use_p_start_cond", False))
    GOAL_DIM = 6 if use_p_start else 3
    cond_inj = a.get("cond_injection", "global")
    net = TrajectoryScoreNetUNet(
        arm_a, H=a["H"], down_dims=tuple(a["down_dims"]),
        diffusion_step_embed_dim=a["diff_step_embed"],
        n_groups=a["unet_groups"], kernel_size=a["unet_kernel"],
        t_scale=a["t_scale"], goal_cond_dim=GOAL_DIM, cond_injection=cond_inj,
    ).to(device)
    net.load_state_dict(ours_ckpt["ema_state"]); net.eval()
    score_fn = TrajectoryScaledScoreFn(net, sde)
    H1_o = a["H"] + 1
    d = arm_a.ambient_dim
    grh_o = list(range(H1_o - H1_o // 2, H1_o)); sah_o = [0]
    ours_rows = []
    for sigma in args.sigma_list:
        torch.manual_seed(args.seed + 3000)
        noise = sigma * torch.randn(n, 3, device=device)
        p_targets_noisy = p_targets_true + noise
        z_lim = z_t.unsqueeze(1).expand(-1, H1_o, -1).reshape(n * H1_o, -1)
        torch.manual_seed(args.seed + 4000)
        tau_T = limiting.sample(n * H1_o, device=device, z_e=z_lim).reshape(n, H1_o, d)
        cond_for_eval = torch.cat([p_targets_noisy, p_starts], dim=-1) if use_p_start else p_targets_noisy
        with torch.no_grad():
            tau_gen = traj_reverse_grw(
                sde, score_fn, tau_T, n_steps=200, eps=a["eps"],
                goal_cond=cond_for_eval, guidance_scale=0.0,
                goal_residual_alpha=100.0, goal_residual_h=grh_o,
                p_start=p_starts, start_anchor_alpha=100.0, start_anchor_h=sah_o,
                smoothness_alpha_vel=5.0, smoothness_alpha_acc=0.0,
            )
        p_end = tau_gen[:, -1, 7:10]
        pos_err = (p_end - p_targets_true).norm(dim=-1)
        s2 = (pos_err < 0.02).float().mean().item()
        s5 = (pos_err < 0.05).float().mean().item()
        s10 = (pos_err < 0.10).float().mean().item()
        ours_rows.append({"sigma": sigma, "pos_err_mean": pos_err.mean().item(),
                          "succ_2cm": s2, "succ_5cm": s5, "succ_10cm": s10})
        print(f"{'Ours-Analytic':>22s}  {sigma*1000:>6.1f}mm  {pos_err.mean().item():>8.4f}m  "
              f"{pos_err.median().item():>8.4f}m  {s2*100:>4.1f}%  {s5*100:>4.1f}%  {s10*100:>4.1f}%")
    results["Ours-Analytic"] = ours_rows

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
