"""Diagnose whether the trained Ours-UNet score net actually USES p_target cond.

Test: same noise τ_T with two distinct p_target values at fixed z_e.
If cond works, gen end-EE should track p_target.  If not, gen distribution
is independent of p_target (confirming the score net is collapsing to the
SDE limiting / b_fwd signal and ignoring cond).
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
from smcdp.sde import LangevinSDE, LinearBetaSchedule
from smcdp.trajectories import (
    TrajectoryScoreNetUNet, TrajectoryScaledScoreFn, traj_reverse_grw,
)


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="outputs/franka_traj_unet/ckpt_riemannian.pt")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--n-sample-steps", type=int, default=200)
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
    n = args.n
    z_val = 0.10
    z_tensor = torch.full((n, 1), z_val, device=device)

    # Same noise for both targets
    z_lim = z_tensor.unsqueeze(1).expand(-1, H1, -1).reshape(n * H1, -1)
    torch.manual_seed(args.seed)
    tau_T = limiting.sample(n * H1, device=device, z_e=z_lim).reshape(n, H1, d)

    # Two distinct targets
    p_target_1 = torch.tensor([[0.42, -0.04, 0.42]], device=device).expand(n, 3).contiguous()
    p_target_2 = torch.tensor([[0.48, +0.04, 0.48]], device=device).expand(n, 3).contiguous()

    print(f"Diagnostic: same τ_T, two p_targets at z_e={z_val}")
    print(f"  p_target_1 = {p_target_1[0].tolist()}")
    print(f"  p_target_2 = {p_target_2[0].tolist()}")
    print(f"  ‖p_t1 − p_t2‖ = {(p_target_1[0] - p_target_2[0]).norm():.4f}")
    print()

    with torch.no_grad():
        tau_gen_1 = traj_reverse_grw(sde, score_fn, tau_T,
                                     n_steps=args.n_sample_steps, eps=a["eps"],
                                     goal_cond=p_target_1)
        tau_gen_2 = traj_reverse_grw(sde, score_fn, tau_T,
                                     n_steps=args.n_sample_steps, eps=a["eps"],
                                     goal_cond=p_target_2)

    # Compare end-EE
    p_end_1 = tau_gen_1[:, -1, 7:10]                     # (n, 3)
    p_end_2 = tau_gen_2[:, -1, 7:10]
    err_1 = (p_end_1 - p_target_1).norm(dim=-1).mean()
    err_2 = (p_end_2 - p_target_2).norm(dim=-1).mean()
    err_swap = (p_end_1 - p_target_2).norm(dim=-1).mean()

    print(f"Result:")
    print(f"  ‖gen_1 − target_1‖_mean = {err_1:.4f}    (gen with t1, dist to t1)")
    print(f"  ‖gen_2 − target_2‖_mean = {err_2:.4f}    (gen with t2, dist to t2)")
    print(f"  ‖gen_1 − target_2‖_mean = {err_swap:.4f}  (gen with t1, dist to t2)")
    print(f"  ‖gen_1 − gen_2‖_mean    = {(p_end_1 - p_end_2).norm(dim=-1).mean():.4f}")
    print()
    print(f"  mean(p_end_1) = {p_end_1.mean(0).tolist()}")
    print(f"  mean(p_end_2) = {p_end_2.mean(0).tolist()}")
    print(f"  std(p_end_1)  = {p_end_1.std(0).tolist()}")
    print(f"  std(p_end_2)  = {p_end_2.std(0).tolist()}")

    # Verdict
    if (p_end_1.mean(0) - p_end_2.mean(0)).norm() < 0.01:
        print("\n  => cond is being IGNORED by the score net (gen distribution doesn't move with p_target)")
    elif err_1 < 0.05 and err_2 < 0.05:
        print("\n  => cond is being USED and gen reaches target")
    else:
        print("\n  => cond is partially used (gen shifts toward target but doesn't reach)")


if __name__ == "__main__":
    main()
