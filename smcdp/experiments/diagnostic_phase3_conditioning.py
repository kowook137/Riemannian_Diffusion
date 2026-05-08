"""diagnostic_plan.md Phase 3 — Conditioning signal magnitude + direction (no retrain).

Diagnoses:
  5.1 Cond/uncond score difference magnitude
  5.2 Direction alignment: does (s_cond − s_uncond) push toward goal?
       (one chart-step in this direction; does p_ee error decrease?)
  5.3 Random-c vs correct-c sample ablation (with reverse SDE)
  5.4 Cond ablation summary

기준 (diagnostic_plan §5.6):
    | (s_c−s_u)/s_c magnitude | direction Δe < 0 | 결론 |
    | < 0.10                  | any              | Cond broken / nullified |
    | 0.10-0.30               | yes              | Cond weak but correct dir |
    | 0.10-0.30               | no               | Cond wrong direction |
    | > 0.30                  | yes              | Cond OK, 다른 issue |
    | > 0.30                  | no               | Cond strong but wrong (architecture) |
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
    TrajectoryScoreNetUNet, TrajectoryScaledScoreFn,
    traj_forward_grw, traj_reverse_grw,
)


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="outputs/franka_traj_unet/ckpt_riemannian.pt")
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="outputs/diagnostic/phase3.json")
    return p.parse_args()


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

    net_ema = TrajectoryScoreNetUNet(
        arm, H=a["H"], down_dims=tuple(a["down_dims"]),
        diffusion_step_embed_dim=a["diff_step_embed"],
        n_groups=a["unet_groups"], kernel_size=a["unet_kernel"],
        t_scale=a["t_scale"], goal_cond_dim=3,
    ).to(device)
    net_ema.load_state_dict(ck["ema_state"]); net_ema.eval()
    score_fn = TrajectoryScaledScoreFn(net_ema, sde)

    H1 = a["H"] + 1
    d = arm.ambient_dim

    # ---- demo data + targets ----
    data = FrankaBimodalReachingDemo(
        manifold=arm, ik_arm=arm_analytic, H=a["H"],
        q_rest_A=list(a["q_rest_A"]), q_rest_B=list(a["q_rest_B"]),
        p_box_lo=tuple(a["p_box_lo"]), p_box_hi=tuple(a["p_box_hi"]),
        z_e_range=(a["z_min"], a["z_max"]),
        branch_p_A=a["branch_p_A"], jitter_q=a["jitter_q"],
        n_ik_steps=a["n_ik_steps"],
    )
    torch.manual_seed(args.seed)
    x_data, _, _, p_target = data.sample(args.n, device=device)

    print(f"Phase 3 — Conditioning signal analysis (n={args.n})\n")

    # ---- 5.1 Cond/Uncond magnitude across r ----
    print("5.1 Cond vs Uncond score magnitude (across r ∈ [0.1, 0.5, 0.9])")
    summary_5_1 = {}
    for r_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
        r = torch.full((args.n,), r_val, device=device)
        # forward to time r
        with torch.no_grad():
            tau_r = traj_forward_grw(sde, x_data, r, a["n_grw_steps"])
        # cond and uncond scores
        with torch.no_grad():
            s_cond = score_fn(tau_r, r, goal_cond=p_target)
            s_uncond = score_fn(tau_r, r, goal_cond=torch.zeros_like(p_target))
        diff = (s_cond - s_uncond).reshape(args.n, -1).norm(dim=-1)                # (n,)
        n_cond = s_cond.reshape(args.n, -1).norm(dim=-1)                            # (n,)
        n_uncond = s_uncond.reshape(args.n, -1).norm(dim=-1)
        rel_mag = (diff / n_cond.clamp(min=1e-12)).mean().item()
        summary_5_1[f"r={r_val:.1f}"] = {
            "cond_norm_mean":   n_cond.mean().item(),
            "uncond_norm_mean": n_uncond.mean().item(),
            "diff_norm_mean":   diff.mean().item(),
            "diff_over_cond":   rel_mag,
        }
        print(f"  r={r_val:.1f}:  ‖s_cond‖={n_cond.mean():.3f}  ‖s_uncond‖={n_uncond.mean():.3f}  "
              f"‖Δ‖={diff.mean():.3f}  ratio Δ/cond={rel_mag:.3f}")

    # ---- 5.2 Direction alignment ----
    # For each sample: take last-timestep chart delta (s_cond − s_uncond)[-1, :n_q],
    # take a small step ε in this direction in q, recompute p_ee, see if error decreases.
    print("\n5.2 Direction alignment: does (s_cond − s_uncond) push p_ee toward target?")
    n_q = 7
    eps_step = 0.02
    summary_5_2 = {}
    for r_val in [0.3, 0.7]:
        r = torch.full((args.n,), r_val, device=device)
        with torch.no_grad():
            tau_r = traj_forward_grw(sde, x_data, r, a["n_grw_steps"])
            s_cond = score_fn(tau_r, r, goal_cond=p_target)
            s_uncond = score_fn(tau_r, r, goal_cond=torch.zeros_like(p_target))
            delta_score = s_cond - s_uncond                                        # (n, H+1, d)
            # Use last-step's chart component (q-block) as direction
            delta_q_last = delta_score[:, -1, :n_q]                                 # (n, 7)
            # Normalize so we make a small step
            dq_norm = delta_q_last.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            dq = eps_step * delta_q_last / dq_norm                                   # (n, 7)
            # current q_last and p_ee
            q_last = tau_r[:, -1, :n_q]
            z_e = tau_r[:, -1, n_q + 3:n_q + 4]                                      # (n, 1)
            p_before = arm.F(q_last, z_e)                                            # (n, 3)
            err_before = (p_before - p_target).norm(dim=-1)                          # (n,)
            # step
            q_after = q_last + dq
            p_after = arm.F(q_after, z_e)
            err_after = (p_after - p_target).norm(dim=-1)
            delta_err = (err_after - err_before)                                     # (n,)
        frac_decreased = (delta_err < 0).float().mean().item()
        summary_5_2[f"r={r_val:.1f}"] = {
            "frac_decreased": frac_decreased,
            "mean_delta_err": delta_err.mean().item(),
            "median_delta_err": delta_err.median().item(),
        }
        print(f"  r={r_val:.1f}:  step ε={eps_step}  frac with Δe<0 = {frac_decreased*100:.1f}%  "
              f"mean Δe = {delta_err.mean():+.4f}  median Δe = {delta_err.median():+.4f}")

    # ---- 5.3 Random-c vs Correct-c sample ablation ----
    print("\n5.3 Sample ablation: correct vs random p_target with full reverse SDE")
    n_smaller = min(64, args.n)
    z_val = 0.10
    z_tensor = torch.full((n_smaller, 1), z_val, device=device)
    z_lim = z_tensor.unsqueeze(1).expand(-1, H1, -1).reshape(n_smaller * H1, -1)

    box_lo = torch.tensor(a["p_box_lo"], device=device)
    box_hi = torch.tensor(a["p_box_hi"], device=device)

    torch.manual_seed(args.seed)
    p_correct = box_lo + (box_hi - box_lo) * torch.rand(n_smaller, 3, device=device)
    p_random = box_lo + (box_hi - box_lo) * torch.rand(n_smaller, 3, device=device)
    p_zero = torch.zeros(n_smaller, 3, device=device)

    summary_5_3 = {}
    for label, p_t_used in [("correct_c", p_correct), ("random_c", p_random), ("null_c", p_zero)]:
        torch.manual_seed(args.seed + 11)
        tau_T = limiting.sample(n_smaller * H1, device=device,
                                z_e=z_lim).reshape(n_smaller, H1, d)
        with torch.no_grad():
            tau_gen = traj_reverse_grw(sde, score_fn, tau_T,
                                       n_steps=200, eps=a["eps"],
                                       goal_cond=p_t_used, guidance_scale=0.0)
        p_end = tau_gen[:, -1, 7:10]
        # measure dist to CORRECT target (always)
        err = (p_end - p_correct).norm(dim=-1)
        summary_5_3[label] = {
            "pos_err_to_correct_target_mean": err.mean().item(),
            "pos_err_to_correct_target_med":  err.median().item(),
        }
        print(f"  cond={label:10s}: ‖p_end - p_correct‖ mean={err.mean():.4f}m  "
              f"med={err.median():.4f}m")

    # the cond effect
    correct_minus_random = (
        summary_5_3["correct_c"]["pos_err_to_correct_target_mean"]
        - summary_5_3["random_c"]["pos_err_to_correct_target_mean"]
    )
    correct_minus_null = (
        summary_5_3["correct_c"]["pos_err_to_correct_target_mean"]
        - summary_5_3["null_c"]["pos_err_to_correct_target_mean"]
    )
    print(f"\n  Effect (smaller is better when cond=correct):")
    print(f"    correct − random = {correct_minus_random:+.4f}m  "
          f"({'cond used' if correct_minus_random < -0.02 else 'cond weak'})")
    print(f"    correct − null   = {correct_minus_null:+.4f}m  "
          f"({'cond used' if correct_minus_null < -0.02 else 'cond weak'})")

    # ---- Decision ----
    rel_mag_at_r05 = summary_5_1["r=0.5"]["diff_over_cond"]
    direction_at_r03 = summary_5_2["r=0.3"]["frac_decreased"]
    direction_at_r07 = summary_5_2["r=0.7"]["frac_decreased"]

    notes = []
    if rel_mag_at_r05 < 0.10:
        notes.append(f"Cond magnitude WEAK at r=0.5 (Δ/cond={rel_mag_at_r05:.3f} < 0.10)")
    elif rel_mag_at_r05 > 0.30:
        notes.append(f"Cond magnitude STRONG at r=0.5 (Δ/cond={rel_mag_at_r05:.3f} > 0.30)")
    else:
        notes.append(f"Cond magnitude MODERATE at r=0.5 (Δ/cond={rel_mag_at_r05:.3f})")

    if direction_at_r03 < 0.55 and direction_at_r07 < 0.55:
        notes.append(f"Cond direction NOT consistently toward goal (r=0.3 frac={direction_at_r03:.2%}, "
                     f"r=0.7 frac={direction_at_r07:.2%})")
    elif direction_at_r03 > 0.65 or direction_at_r07 > 0.65:
        notes.append(f"Cond direction TOWARD goal (r=0.3 frac={direction_at_r03:.2%}, "
                     f"r=0.7 frac={direction_at_r07:.2%})")
    else:
        notes.append(f"Cond direction WEAK signal (frac ~50%)")

    if correct_minus_random < -0.05 and correct_minus_null < -0.05:
        notes.append(f"Sample ablation: cond IS USED (correct beats random by "
                     f"{-correct_minus_random:.3f}m)")
    elif abs(correct_minus_random) < 0.02:
        notes.append(f"Sample ablation: cond NOT USED (correct ≈ random)")
    else:
        notes.append(f"Sample ablation: cond partially used (correct − random = "
                     f"{correct_minus_random:+.3f}m)")

    print("\n--- Decision (diagnostic_plan §5.6) ---")
    for note in notes:
        print(f"  • {note}")

    summary = {
        "config": {"ckpt": args.ckpt, "n": args.n},
        "section_5_1_magnitude": summary_5_1,
        "section_5_2_direction": summary_5_2,
        "section_5_3_ablation": summary_5_3,
        "decision_notes": notes,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
