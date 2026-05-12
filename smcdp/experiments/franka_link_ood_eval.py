"""Zero-shot link-length OOD evaluation of a base-trained v5.1 policy.

The policy was trained on the canonical Franka panda URDF (scalar z_e = tool
length only).  This driver loads the trained ckpt and evaluates it under a
*silent* link-length perturbation of the real robot:

   conditioning (T_start, T_target, z_e)  — sampled from base FK distribution
   policy generates q-trajectory          — uses base self-model T_phi
   execution                               — actual robot has perturbed FK
   error                                   — || perturbed_FK(q_H, z_e) - T_target ||

This measures policy robustness when the link 3 / link 5 lengths differ from
training by Δl_3, Δl_5.  No retraining; only eval-side FK substitution.

Outputs per (Δl_3, Δl_5):
   pos_err_mean / rot_err_mean
   pose_succ_(5cm, 5°)  and  (5cm, 10°)
   jvio (q-space, must remain 0% under bounded chart by construction)
   manifold gap: base T_phi(q) vs perturbed FK(q) — by design > 0 here

Reference:  extension_report.md §14 best ours, Experiment_plan.md §B (zero-shot
link-length variant).

Usage:
    python -m smcdp.experiments.franka_link_ood_eval \\
        --ckpt outputs/v51_tier2_300k_c2_endpt_resume/ours_v2_pose.pt \\
        --dl3-list 0.0 0.015 -0.015 0.03 -0.03 0.045 -0.045 \\
        --dl5-list 0.0 0.015 -0.015 0.03 -0.03 0.045 -0.045 \\
        --n-eval-per-cell 64 --n-sample-steps 1000
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import pybullet_data

from smcdp.manifolds_pose import (
    Franka7DoFPose, BoundedChartPoseManifold,
)
from smcdp.charts import make_chart_from_manifold
from smcdp.franka.self_model_pose import PoseResidualMLP, LearnedSelfModelFranka7DoFPose
from smcdp.franka.demo_gen_pose import FrankaBimodalReachingDemoPose
from smcdp.lie_se3 import log_relative_Rp, pose7_to_Rp
from smcdp.trajectories_pose import (
    PoseChartOUSDE, TrajectoryScaledScoreFnPose, traj_reverse_ou_chart_pose,
    TrajectoryScoreNetUNetPose,
)
from smcdp.sde import LinearBetaSchedule


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def _reconstruct_policy(ckpt_path: str, device, dtype):
    """Reconstruct manifold (base FK + learned residual), SDE, score net from ckpt."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    s1ck = torch.load(a["stage1_pose_ckpt"], map_location=device, weights_only=False)

    arm_a_base = Franka7DoFPose(
        URDF, sigma_p=a["sigma_p"], sigma_R=a["sigma_R"],
        tool_z_max=a["z_max"], tikhonov_frac=a.get("tikhonov_frac", 0.0),
    )
    arm_a_base._ensure_chain(torch.zeros(1, 7, device=device))

    res = PoseResidualMLP(
        n_q=7, n_z=1, hidden=s1ck["args"]["hidden"],
        n_layers=s1ck["args"]["n_layers"], activation=torch.nn.Softplus,
        final_init_scale=1e-3, output_omega=True,
    ).to(device=device, dtype=dtype)
    res.load_state_dict(s1ck["residual_net_state"]); res.eval()

    arm = LearnedSelfModelFranka7DoFPose(
        res, URDF, sigma_p=a["sigma_p"], sigma_R=a["sigma_R"], tool_z_max=a["z_max"],
    )
    arm.tikhonov_frac = float(a.get("tikhonov_frac", 0.0))
    arm._ensure_chain(torch.zeros(1, 7, device=device))

    if a.get("bounded_chart", False):
        arm = BoundedChartPoseManifold(
            arm,
            make_chart_from_manifold(
                arm, bounded=True, chart_temp=float(a.get("chart_temp", 1.0)),
            ),
            lambda_floor=float(a.get("lambda_floor", 1e-4)),
        )

    schedule = LinearBetaSchedule(beta_0=a["beta_0"], beta_f=a["beta_f"], tf=1.0)
    sde = PoseChartOUSDE(arm, schedule, gbar_mode=a.get("gbar_mode", "identity"))
    net = TrajectoryScoreNetUNetPose(
        manifold=arm, H=a["H"],
        down_dims=tuple(a["down_dims"]),
        diffusion_step_embed_dim=a["diff_step_embed"],
        n_groups=a["unet_groups"], kernel_size=a["unet_kernel"],
        cond_predict_scale=False, t_scale=a["t_scale"],
        goal_cond_dim=14, cond_injection=a["cond_injection"],
        endpoint_rel_cond=bool(a.get("endpoint_rel_cond", False)),
    ).to(device=device, dtype=dtype)
    net.load_state_dict(ckpt["ema_net"]); net.eval()
    score_fn = TrajectoryScaledScoreFnPose(net, sde, std_trick=True, proxy_std_mode="ou")
    return ckpt, a, arm, arm_a_base, sde, score_fn, res


def _build_demo(a, arm, arm_a, z_val):
    target_perturb_rad = a["target_perturb_deg"] * 3.14159265 / 180.0
    return FrankaBimodalReachingDemoPose(
        manifold=arm, ik_arm=arm_a, H=a["H"],
        q_rest_A=list(a["q_rest_A"]), q_rest_B=list(a["q_rest_B"]),
        p_box_lo=tuple(a["p_box_lo"]), p_box_hi=tuple(a["p_box_hi"]),
        z_e_range=(z_val, z_val), branch_p_A=a["branch_p_A"],
        jitter_q=a["jitter_q"], n_ik_steps=a["n_ik_steps"],
        ik_alpha=a.get("ik_alpha", 0.5),
        ik_alpha_null=a.get("ik_alpha_null", 0.3),
        ik_lam=a.get("ik_lam", 0.05),
        R_anchor_axis_angle=tuple(a["R_anchor_aa"]),
        target_perturb_rad=target_perturb_rad,
        ik_clamp_to_limits=a.get("ik_clamp_to_limits", False),
        ik_clamp_margin_frac=a.get("ik_clamp_margin_frac", 0.001),
    )


def _build_perturbed_fk(residual_net, a, device, dtype, dl3: float, dl5: float):
    """LearnedSelfModelFranka7DoFPose with perturbed link lengths.

    Uses the same learned residual ξ_φ that the policy trained against (so the
    (0,0) baseline reproduces training-distribution numerics), but with the
    analytic FK chain rebuilt from a URDF whose link-3 / link-5 origins are
    shifted by Δl_3 / Δl_5.  This isolates the *link length* effect from any
    confound with the residual term.
    """
    arm_pert = LearnedSelfModelFranka7DoFPose(
        residual_net=residual_net,
        urdf_path=URDF,
        sigma_p=a["sigma_p"], sigma_R=a["sigma_R"],
        tool_z_max=a["z_max"],
        link_perturb_dl3=dl3, link_perturb_dl5=dl5,
    )
    arm_pert.tikhonov_frac = float(a.get("tikhonov_frac", 0.0))
    arm_pert._ensure_chain(torch.zeros(1, 7, device=device))
    return arm_pert


def _eval_cell(args, a, arm, arm_a_base, sde, score_fn, residual_net,
                z_val: float, dl3: float, dl5: float,
                device, dtype):
    """For one (z_val, dl3, dl5) cell, sample policy then measure error on
    perturbed FK.  Demos / conditioning are generated with the *base* FK
    (training distribution); only execution uses perturbed FK.
    """
    n = args.n_eval_per_cell
    H1 = a["H"] + 1

    torch.manual_seed(args.seed + int(1e6 * z_val) + int(1e4 * dl3) + int(1e2 * dl5))
    demo = _build_demo(a, arm, arm_a_base, z_val)
    x_demo, _, _, T_target, T_start = demo.sample(n, device=device, dtype=dtype)
    z_e = torch.full((n, 1), z_val, device=device, dtype=dtype)
    goal_cond = torch.cat([T_start, T_target], dim=-1)

    # Sample policy (uses base self-model internally)
    x_gen = traj_reverse_ou_chart_pose(
        sde, score_fn, n_samples=n, H=a["H"], n_steps=args.n_sample_steps,
        goal_cond=goal_cond, z_e=z_e, device=device, dtype=dtype, integrator="euler",
    )                                                                       # (n, H+1, ambient)

    # Recover physical q at endpoint (chart slot stores u; physical_q applies ψ)
    if hasattr(arm, "physical_q"):
        q_traj = arm.physical_q(x_gen)
    else:
        q_traj = x_gen[..., :7]
    q_H = q_traj[:, -1, :]                                                  # (n, 7)

    # Build perturbed FK (real-robot kinematics, same residual + perturbed links)
    arm_pert = _build_perturbed_fk(residual_net, a, device, dtype, dl3, dl5)
    R_actual, p_actual = arm_pert.T_phi_Rp(q_H, z_e)                        # (n, 3, 3), (n, 3)
    R_target, p_target = pose7_to_Rp(T_target)
    e_perturbed = log_relative_Rp(R_actual, p_actual, R_target, p_target)
    e_p = e_perturbed[..., :3].norm(dim=-1)                                 # m
    e_R = e_perturbed[..., 3:].norm(dim=-1)                                 # rad

    # Also compute what the policy *thinks* it reached (base self-model)
    R_base, p_base = arm.T_phi_Rp(q_H if not hasattr(arm, "physical_q") else x_gen[:, -1, :7], z_e)
    e_base = log_relative_Rp(R_base, p_base, R_target, p_target)
    e_p_base = e_base[..., :3].norm(dim=-1)
    e_R_base = e_base[..., 3:].norm(dim=-1)

    # Manifold gap induced by FK mismatch: base prediction vs perturbed actual
    gap = log_relative_Rp(R_base, p_base, R_actual, p_actual)
    gap_p_mm = 1000.0 * gap[..., :3].norm(dim=-1)
    gap_R_deg = (180.0 / math.pi) * gap[..., 3:].norm(dim=-1)

    # Joint feasibility (q-space, unaffected by FK change)
    q_lo, q_hi = arm.joint_limits(device=device, dtype=dtype)
    viol = ((q_traj < q_lo) | (q_traj > q_hi)).any(-1).any(-1)
    jvio = viol.float().mean().item()

    th_5cm = 0.05
    th_5deg = math.radians(5.0)
    th_10deg = math.radians(10.0)
    pose_ok_55 = ((e_p < th_5cm) & (e_R < th_5deg))
    pose_ok_510 = ((e_p < th_5cm) & (e_R < th_10deg))
    joint_safe = ~viol
    return {
        "z_e": z_val,
        "dl_3": dl3,
        "dl_5": dl5,
        # Perturbed-FK execution error (the OOD test, primary metric)
        "pos_err_mean_cm": 100.0 * e_p.mean().item(),
        "pos_err_median_cm": 100.0 * float(e_p.median().item()),
        "rot_err_mean_deg": math.degrees(e_R.mean().item()),
        "rot_err_median_deg": math.degrees(float(e_R.median().item())),
        "pose_succ_5cm_5deg": pose_ok_55.float().mean().item(),
        "pose_succ_5cm_10deg": pose_ok_510.float().mean().item(),
        "eff_pose_succ_5cm_5deg":  (pose_ok_55 & joint_safe).float().mean().item(),
        "eff_pose_succ_5cm_10deg": (pose_ok_510 & joint_safe).float().mean().item(),
        # Base-FK self-belief error (training-distribution diagnostic; expected small)
        "pos_err_base_mean_cm": 100.0 * e_p_base.mean().item(),
        "rot_err_base_mean_deg": math.degrees(e_R_base.mean().item()),
        # Manifold gap: how much does the policy's internal pose belief disagree
        # with the actual perturbed-robot endpoint?  Expected to grow with |Δl|.
        "gap_p_mm_mean": gap_p_mm.mean().item(),
        "gap_p_mm_max": gap_p_mm.max().item(),
        "gap_R_deg_mean": gap_R_deg.mean().item(),
        "gap_R_deg_max": gap_R_deg.max().item(),
        "joint_viol_rate": jvio,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dl3-list", type=float, nargs="+",
                   default=[0.0, 0.015, -0.015, 0.03, -0.03, 0.045, -0.045])
    p.add_argument("--dl5-list", type=float, nargs="+",
                   default=[0.0, 0.015, -0.015, 0.03, -0.03, 0.045, -0.045])
    p.add_argument("--z-list", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20])
    p.add_argument("--n-eval-per-cell", type=int, default=64)
    p.add_argument("--n-sample-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-name", type=str, default="link_ood_eval.json")
    p.add_argument("--sweep-mode", type=str, default="grid",
                   choices=["grid", "diag", "axis"],
                   help="grid = full Cartesian; diag = (Δl_3, Δl_5) paired "
                        "elementwise; axis = sweep one axis at a time (other=0).")
    return p.parse_args()


def _cell_iter(args):
    dl3s = args.dl3_list
    dl5s = args.dl5_list
    if args.sweep_mode == "grid":
        for a3 in dl3s:
            for a5 in dl5s:
                yield (a3, a5)
    elif args.sweep_mode == "diag":
        for a3, a5 in zip(dl3s, dl5s):
            yield (a3, a5)
    elif args.sweep_mode == "axis":
        seen = set()
        for a3 in dl3s:
            if (a3, 0.0) not in seen:
                seen.add((a3, 0.0)); yield (a3, 0.0)
        for a5 in dl5s:
            if (0.0, a5) not in seen:
                seen.add((0.0, a5)); yield (0.0, a5)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    torch.manual_seed(args.seed)

    print(f"loading {args.ckpt}...")
    ckpt, a, arm, arm_a_base, sde, score_fn, residual_net = _reconstruct_policy(args.ckpt, device, dtype)
    print(f"  chart_temp={a.get('chart_temp', 1.0)}  endpoint_rel_cond="
          f"{a.get('endpoint_rel_cond', False)}  use_v51={a.get('use_v51')}  "
          f"steps_trained={a['steps']}")

    cells = list(_cell_iter(args))
    print(f"sweep: {len(cells)} (Δl_3, Δl_5) cells × {len(args.z_list)} z_e × "
          f"{args.n_eval_per_cell} samples; n_sample_steps={args.n_sample_steps}")

    rows = []
    header = f"{'dl_3':>6} | {'dl_5':>6} | {'z_e':>5} | {'pos cm':>6} | {'rot°':>5} | "\
             f"{'succ55':>7} | {'succ510':>8} | {'gap p mm':>8} | {'gap R°':>6} | {'jvio':>5}"
    print(header)
    print("-" * len(header))
    for dl3, dl5 in cells:
        for z_val in args.z_list:
            m = _eval_cell(args, a, arm, arm_a_base, sde, score_fn, residual_net,
                            z_val, dl3, dl5, device, dtype)
            rows.append(m)
            print(f"{dl3:>+6.3f} | {dl5:>+6.3f} | {z_val:>5.2f} | "
                  f"{m['pos_err_mean_cm']:>6.2f} | {m['rot_err_mean_deg']:>5.2f} | "
                  f"{100*m['pose_succ_5cm_5deg']:>6.1f}% | "
                  f"{100*m['pose_succ_5cm_10deg']:>7.1f}% | "
                  f"{m['gap_p_mm_mean']:>8.2f} | {m['gap_R_deg_mean']:>6.2f} | "
                  f"{100*m['joint_viol_rate']:>4.1f}%")

    out_dir = Path(args.ckpt).parent
    out_path = out_dir / args.out_name
    with open(out_path, "w") as f:
        json.dump({"ckpt": str(args.ckpt), "args": vars(args), "rows": rows}, f, indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
