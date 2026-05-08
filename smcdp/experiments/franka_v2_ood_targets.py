"""V2 OOD target test — generalization beyond training p_box.

Training p_box: [0.40, 0.50] × [-0.05, 0.05] × [0.40, 0.50]
Tests:
  - in-box (reference)
  - 5cm outside (each axis)
  - 10cm outside
  - 2x bigger box
  - corner extrapolations

Reports pos_err and success at each OOD condition.  If the model is just fitting
the training distribution, OOD targets should fail.  If it generalizes (e.g., via
the explicit goal residual guidance + the learned trajectory smoothness prior),
moderate extrapolation should still work.
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


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="outputs/franka_traj_unet_v2/ckpt_riemannian.pt")
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--n-sample-steps", type=int, default=200)
    p.add_argument("--alpha-goal", type=float, default=100.0)
    p.add_argument("--alpha-start", type=float, default=100.0)
    p.add_argument("--alpha-vel", type=float, default=5.0)
    p.add_argument("--alpha-acc", type=float, default=0.0)
    p.add_argument("--z-val", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="outputs/diagnostic/v2_ood_targets.json")
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

    use_p_start = bool(a.get("use_p_start_cond", False))
    GOAL_DIM = 6 if use_p_start else 3
    cond_inj = a.get("cond_injection", "global")
    net = TrajectoryScoreNetUNet(
        arm, H=a["H"], down_dims=tuple(a["down_dims"]),
        diffusion_step_embed_dim=a["diff_step_embed"],
        n_groups=a["unet_groups"], kernel_size=a["unet_kernel"],
        t_scale=a["t_scale"], goal_cond_dim=GOAL_DIM,
        cond_injection=cond_inj,
    ).to(device)
    net.load_state_dict(ck["ema_state"]); net.eval()
    score_fn = TrajectoryScaledScoreFn(net, sde)

    H1 = a["H"] + 1
    d = arm.ambient_dim
    box_lo = torch.tensor(a["p_box_lo"], device=device)
    box_hi = torch.tensor(a["p_box_hi"], device=device)
    grh = list(range(H1 - H1 // 2, H1))
    sah = [0]

    # Define test conditions:  (label, target_box_lo, target_box_hi)
    conditions = [
        ("(I) in-box (reference)",        a["p_box_lo"],                 a["p_box_hi"]),
        ("(II) +5cm OOD",                 [v + 0.05 for v in a["p_box_lo"]],  [v + 0.05 for v in a["p_box_hi"]]),
        ("(III) +10cm OOD",               [v + 0.10 for v in a["p_box_lo"]],  [v + 0.10 for v in a["p_box_hi"]]),
        ("(IV) -5cm OOD",                 [v - 0.05 for v in a["p_box_lo"]],  [v - 0.05 for v in a["p_box_hi"]]),
        ("(V) y-shifted +5cm",            [a["p_box_lo"][0], a["p_box_lo"][1]+0.05, a["p_box_lo"][2]],
                                           [a["p_box_hi"][0], a["p_box_hi"][1]+0.05, a["p_box_hi"][2]]),
        ("(VI) z-up +10cm",               [a["p_box_lo"][0], a["p_box_lo"][1], a["p_box_lo"][2]+0.10],
                                           [a["p_box_hi"][0], a["p_box_hi"][1], a["p_box_hi"][2]+0.10]),
        ("(VII) 2x box (centered)",
            [(a["p_box_lo"][i] + a["p_box_hi"][i]) / 2 - (a["p_box_hi"][i] - a["p_box_lo"][i]) for i in range(3)],
            [(a["p_box_lo"][i] + a["p_box_hi"][i]) / 2 + (a["p_box_hi"][i] - a["p_box_lo"][i]) for i in range(3)]),
    ]

    print(f"V2 OOD target test  z_e={args.z_val}  n={args.n}")
    print(f"  (start sampled from training p_box for all conditions; only target is OOD)\n")
    print(f"{'condition':>30s}  {'target box':>40s}  {'pos_err':>9} {'med':>9} "
          f"{'s@2':>5} {'s@5':>5} {'s@10':>5} {'s@15':>5} {'frac_A':>7} {'viol%':>5} {'max|g|':>9}")

    rows = []
    for label, t_lo, t_hi in conditions:
        t_lo_t = torch.tensor(t_lo, device=device, dtype=torch.float32)
        t_hi_t = torch.tensor(t_hi, device=device, dtype=torch.float32)

        torch.manual_seed(args.seed + 1000)
        p_targets = t_lo_t + (t_hi_t - t_lo_t) * torch.rand(args.n, 3, device=device)
        # start always from training box (only target is OOD)
        torch.manual_seed(args.seed + 2000)
        p_starts = box_lo + (box_hi - box_lo) * torch.rand(args.n, 3, device=device)

        z_tensor = torch.full((args.n, 1), args.z_val, device=device)
        z_lim = z_tensor.unsqueeze(1).expand(-1, H1, -1).reshape(args.n * H1, -1)
        torch.manual_seed(args.seed + 3000)
        tau_T = limiting.sample(args.n * H1, device=device, z_e=z_lim).reshape(args.n, H1, d)

        if use_p_start:
            cond_for_eval = torch.cat([p_targets, p_starts], dim=-1)
        else:
            cond_for_eval = p_targets

        with torch.no_grad():
            tau_gen = traj_reverse_grw(sde, score_fn, tau_T,
                                       n_steps=args.n_sample_steps, eps=a["eps"],
                                       goal_cond=cond_for_eval, guidance_scale=0.0,
                                       goal_residual_alpha=args.alpha_goal,
                                       goal_residual_h=grh,
                                       p_start=p_starts,
                                       start_anchor_alpha=args.alpha_start,
                                       start_anchor_h=sah,
                                       smoothness_alpha_vel=args.alpha_vel,
                                       smoothness_alpha_acc=args.alpha_acc)

        p_end = tau_gen[:, -1, 7:10]
        pos_err = (p_end - p_targets).norm(dim=-1)
        s2 = (pos_err < 0.02).float().mean().item()
        s5 = (pos_err < 0.05).float().mean().item()
        s10 = (pos_err < 0.10).float().mean().item()
        s15 = (pos_err < 0.15).float().mean().item()

        q1_mid = tau_gen[:, H1 // 2, 0]
        frac_A = (q1_mid > 0).float().mean().item()
        viol = arm.violates_limits(tau_gen.reshape(-1, d)[..., :7]).float().mean().item()
        max_g = arm.constraint(tau_gen.reshape(-1, d)).norm(dim=-1).max().item()

        box_str = f"[{t_lo[0]:.2f}-{t_hi[0]:.2f}]×[{t_lo[1]:+.2f}-{t_hi[1]:+.2f}]×[{t_lo[2]:.2f}-{t_hi[2]:.2f}]"
        print(f"{label:>30s}  {box_str:>40s}  "
              f"{pos_err.mean().item():>8.4f}m {pos_err.median().item():>8.4f}m "
              f"{s2*100:>4.1f}% {s5*100:>4.1f}% {s10*100:>4.1f}% {s15*100:>4.1f}% "
              f"{frac_A:>7.3f} {viol*100:>4.1f}% {max_g:>9.1e}")

        rows.append({
            "label": label, "target_box": [t_lo, t_hi],
            "pos_err_mean": pos_err.mean().item(),
            "pos_err_med":  pos_err.median().item(),
            "succ_2cm": s2, "succ_5cm": s5, "succ_10cm": s10, "succ_15cm": s15,
            "frac_A": frac_A, "viol": viol, "max_g": max_g,
        })

    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
