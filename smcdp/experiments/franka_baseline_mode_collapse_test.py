"""Per-ctx mode collapse test — fix one (p_target, p_start, z_e) and sample many times.

If the model has TRUE mode capture, repeated samples for the same ctx should produce
both modes ~50/50.  If mode-collapsed, all samples will be in the same mode.
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
from smcdp.baselines import (
    BCTrajectoryPredictor, make_official_diffusion_policy, official_dp_sample,
)


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=64,
                   help="Number of repeated samples for the SAME ctx")
    p.add_argument("--n-targets", type=int, default=8,
                   help="Number of distinct ctxs to test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="outputs/diagnostic/per_ctx_mode_collapse.json")
    args = p.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed)

    # Common manifold setup
    stage1_ck = torch.load("outputs/franka_stage1/delta_phi.pt",
                           map_location=device, weights_only=False)
    s1 = stage1_ck["args"]
    delta_net = DeltaResidualMLP(n_q=7, n_p=3, n_z=1,
                                  hidden=s1["hidden"], n_layers=s1["n_layers"],
                                  activation=torch.nn.Softplus,
                                  final_init_scale=1e-3).to(device)
    delta_net.load_state_dict(stage1_ck["delta_net_state"])
    delta_net.eval()
    arm = LearnedSelfModelFranka7DoF(
        delta_net=delta_net, urdf_path=URDF, end_link="panda_hand", tool_z_max=0.25,
    )
    arm._ensure_chain(torch.zeros(1, 7, device=device))

    # Fixed ctxs
    box_lo = torch.tensor([0.40, -0.05, 0.40], device=device)
    box_hi = torch.tensor([0.50, 0.05, 0.50], device=device)
    z_val = 0.10
    z_tensor = torch.full((1, 1), z_val, device=device)
    torch.manual_seed(args.seed + 1000)
    p_targets = box_lo + (box_hi - box_lo) * torch.rand(args.n_targets, 3, device=device)
    torch.manual_seed(args.seed + 2000)
    p_starts = box_lo + (box_hi - box_lo) * torch.rand(args.n_targets, 3, device=device)

    print(f"Per-ctx mode collapse test  (n={args.n} repeats per ctx, {args.n_targets} ctxs)\n")
    print(f"{'method':>20s}  {'ctx idx':>7s}  {'frac_A':>7s}  {'collapse?':>10s}")

    results = {}

    # ---- Method 1: BC ----
    bc_ckpt = torch.load("outputs/franka_baseline_bc/ckpt.pt",
                         map_location=device, weights_only=False)
    bc_a = bc_ckpt["args"]
    bc = BCTrajectoryPredictor(n_q=7, H=bc_a["H"], ctx_dim=7,
                                hidden=bc_a["bc_hidden"], n_layers=bc_a["bc_layers"],
                                activation="sin").to(device)
    bc.load_state_dict(bc_ckpt["ema_state"]); bc.eval()

    bc_results = []
    for ti in range(args.n_targets):
        ctx = torch.cat([p_targets[ti:ti+1], p_starts[ti:ti+1], z_tensor], dim=-1)
        ctx_rep = ctx.expand(args.n, -1)
        with torch.no_grad():
            q_gen = bc(ctx_rep)                                                      # (n, H+1, 7)
        h_mid = (bc_a["H"] + 1) // 2
        q1_mid = q_gen[:, h_mid, 0]
        # check if all outputs are identical (deterministic = collapse)
        q_std = q1_mid.std().item()
        frac_A = (q1_mid > 0).float().mean().item()
        collapse = "YES" if q_std < 1e-4 else f"q_std={q_std:.4f}"
        bc_results.append({"ctx_idx": ti, "frac_A": frac_A, "q_std": q_std, "collapse": collapse})
        print(f"{'BC':>20s}  {ti:>7d}  {frac_A:>7.3f}  {collapse:>10s}")
    results["BC"] = bc_results

    # ---- Method 2: Ours-V2 ----
    print()
    ours_ckpt = torch.load("outputs/franka_traj_unet_v2/ckpt_riemannian.pt",
                            map_location=device, weights_only=False)
    a = ours_ckpt["args"]
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
    net.load_state_dict(ours_ckpt["ema_state"]); net.eval()
    score_fn = TrajectoryScaledScoreFn(net, sde)

    ours_results = []
    H1 = a["H"] + 1
    d = arm.ambient_dim
    h_mid = H1 // 2
    grh = list(range(H1 - H1 // 2, H1)); sah = [0]
    for ti in range(args.n_targets):
        z_t = torch.full((args.n, 1), z_val, device=device)
        z_lim = z_t.unsqueeze(1).expand(-1, H1, -1).reshape(args.n * H1, -1)
        torch.manual_seed(args.seed + 3000 + ti)
        tau_T = limiting.sample(args.n * H1, device=device, z_e=z_lim).reshape(args.n, H1, d)
        p_t_rep = p_targets[ti:ti+1].expand(args.n, -1)
        p_s_rep = p_starts[ti:ti+1].expand(args.n, -1)
        if use_p_start:
            cond_for_eval = torch.cat([p_t_rep, p_s_rep], dim=-1)
        else:
            cond_for_eval = p_t_rep
        with torch.no_grad():
            tau_gen = traj_reverse_grw(sde, score_fn, tau_T, n_steps=200, eps=a["eps"],
                                       goal_cond=cond_for_eval, guidance_scale=0.0,
                                       goal_residual_alpha=100.0, goal_residual_h=grh,
                                       p_start=p_s_rep, start_anchor_alpha=100.0, start_anchor_h=sah,
                                       smoothness_alpha_vel=5.0, smoothness_alpha_acc=0.0)
        q1_mid = tau_gen[:, h_mid, 0]
        q_std = q1_mid.std().item()
        frac_A = (q1_mid > 0).float().mean().item()
        # for stochastic mode capture, frac_A should be ~0.5 AND q_std should be large
        # (samples land near +0.34 or -0.34, std ~0.34)
        if 0.30 < frac_A < 0.70 and q_std > 0.20:
            collapse = "BIMODAL"
        elif q_std < 0.05:
            collapse = "COLLAPSED"
        else:
            collapse = "PARTIAL"
        ours_results.append({"ctx_idx": ti, "frac_A": frac_A, "q_std": q_std, "collapse": collapse})
        print(f"{'Ours-V2':>20s}  {ti:>7d}  {frac_A:>7.3f}  {collapse:>10s}")
    results["Ours-V2"] = ours_results

    # ---- Method 3: DP-official (DDPM stochastic) ----
    print()
    dp_ckpt = torch.load("outputs/franka_baseline_dp_official/ckpt.pt",
                          map_location=device, weights_only=False)
    dp_a = dp_ckpt["args"]
    from smcdp.baselines import make_official_diffusion_policy, official_dp_sample
    dp_model, dp_scheduler = make_official_diffusion_policy(
        n_q=7, global_cond_dim=7,
        down_dims=list(dp_a["down_dims"]),
        diffusion_step_embed_dim=dp_a["diff_step_embed"],
        n_train_timesteps=dp_a["dp_train_timesteps"],
    )
    dp_model = dp_model.to(device)
    dp_model.load_state_dict(dp_ckpt["ema_state"]); dp_model.eval()

    dp_results = []
    H1_dp = dp_a["H"] + 1
    h_mid_dp = H1_dp // 2
    for ti in range(args.n_targets):
        ctx = torch.cat([p_targets[ti:ti+1], p_starts[ti:ti+1], z_tensor], dim=-1)
        ctx_rep = ctx.expand(args.n, -1)
        with torch.no_grad():
            q_gen = official_dp_sample(dp_model, dp_scheduler,
                                        batch_size=args.n, horizon=H1_dp, n_q=7,
                                        ctx=ctx_rep, device=device,
                                        n_inference_steps=100)
        q1_mid = q_gen[:, h_mid_dp, 0]
        q_std = q1_mid.std().item()
        frac_A = (q1_mid > 0).float().mean().item()
        if 0.30 < frac_A < 0.70 and q_std > 0.20:
            collapse = "BIMODAL"
        elif q_std < 0.05:
            collapse = "COLLAPSED"
        else:
            collapse = "PARTIAL"
        dp_results.append({"ctx_idx": ti, "frac_A": frac_A, "q_std": q_std, "collapse": collapse})
        print(f"{'DP-official':>20s}  {ti:>7d}  {frac_A:>7.3f}  {collapse:>10s}")
    results["DP-official"] = dp_results

    # ---- Method 4: Projected (DDPM ambient + projection) ----
    print()
    pj_ckpt = torch.load("outputs/franka_baseline_projected/ckpt.pt",
                          map_location=device, weights_only=False)
    pj_a = pj_ckpt["args"]
    pj_model, pj_scheduler = make_official_diffusion_policy(
        n_q=10, global_cond_dim=7,                 # ambient (q, p) = 10 dim
        down_dims=list(pj_a["down_dims"]),
        diffusion_step_embed_dim=pj_a["diff_step_embed"],
        n_train_timesteps=pj_a["dp_train_timesteps"],
    )
    pj_model = pj_model.to(device)
    pj_model.load_state_dict(pj_ckpt["ema_state"]); pj_model.eval()

    pj_results = []
    H1_pj = pj_a["H"] + 1
    h_mid_pj = H1_pj // 2
    for ti in range(args.n_targets):
        ctx = torch.cat([p_targets[ti:ti+1], p_starts[ti:ti+1], z_tensor], dim=-1)
        ctx_rep = ctx.expand(args.n, -1)
        with torch.no_grad():
            x_amb = official_dp_sample(pj_model, pj_scheduler,
                                        batch_size=args.n, horizon=H1_pj, n_q=10,
                                        ctx=ctx_rep, device=device,
                                        n_inference_steps=100)
        # extract q from ambient (q, p)
        q_gen = x_amb[..., :7]
        q1_mid = q_gen[:, h_mid_pj, 0]
        q_std = q1_mid.std().item()
        frac_A = (q1_mid > 0).float().mean().item()
        if 0.30 < frac_A < 0.70 and q_std > 0.20:
            collapse = "BIMODAL"
        elif q_std < 0.05:
            collapse = "COLLAPSED"
        else:
            collapse = "PARTIAL"
        pj_results.append({"ctx_idx": ti, "frac_A": frac_A, "q_std": q_std, "collapse": collapse})
        print(f"{'Projected':>20s}  {ti:>7d}  {frac_A:>7.3f}  {collapse:>10s}")
    results["Projected"] = pj_results

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
