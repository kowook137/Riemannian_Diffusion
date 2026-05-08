"""Morphology-transfer evaluation: train on original Franka, eval on a Franka
with one link length modified (fixed perturbation per eval).

Theoretical motivation: the framework's graph manifold M_φ(z_e) is the image of
F_φ(·, z_e); modifying a kinematic link length yields a *homeomorphic* deformed
manifold M_φ'(z_e).  How far can we push the deformation before the framework's
generative prior (trained on M_φ) breaks?

Setup:
  - All models loaded from existing checkpoints (trained on original Franka).
  - Sampling proceeds with original-Franka manifold (no adaptation).
  - Ground-truth EE position uses the *modified* Franka FK (separate
    pytorch_kinematics chain built from a modified URDF).

We sweep one joint origin's xyz component by ± Δ across {0, 10, 30, 50, 100} mm
and report pos_err / succ@{2,5,10}cm / W1 / mode capture / max‖g‖ for each
method on each modified Franka.
"""
from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from copy import deepcopy
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
    BCTrajectoryPredictor, make_official_diffusion_policy,
    official_dp_sample, channel_concat_dp_sample,
    channel_concat_dp_sample_guided,
)


URDF_ORIG = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


# ---------------------------------------------------------------------------
# URDF modification utility
# ---------------------------------------------------------------------------


def make_modified_urdf(orig_path: str, joint_name: str, axis: int, delta_m: float,
                       out_path: str) -> str:
    """Write a modified copy of `orig_path` where joint <name=joint_name>'s
    origin xyz[axis] is shifted by `delta_m` metres.  Other joints unchanged.

    axis: 0 = x, 1 = y, 2 = z.

    Returns the full path written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(orig_path)
    root = tree.getroot()
    found = False
    for joint in root.findall("joint"):
        if joint.get("name") == joint_name:
            origin = joint.find("origin")
            xyz = [float(v) for v in origin.get("xyz").split()]
            xyz[axis] += float(delta_m)
            origin.set("xyz", " ".join(f"{v:.9g}" for v in xyz))
            found = True
            break
    if not found:
        raise RuntimeError(f"joint '{joint_name}' not in {orig_path}")
    # Need to also copy referenced mesh/material files? — pytorch_kinematics
    # only parses kinematics, doesn't load meshes.  Plain URDF copy is fine.
    tree.write(str(out_path))
    return str(out_path)


# Joint-origin parameter table for Franka (which xyz component is the
# "link length" along the chain direction).
#
# panda_joint3: origin xyz = (0, -0.316, 0)  → axis 1 (y), magnitude 0.316m
#               increasing |y| extends panda_link2's effective length (d3 in DH)
# panda_joint5: origin xyz = (-0.0825, 0.384, 0) → axis 1 (y), magnitude 0.384m
#               increasing |y| extends forearm length (d5 in DH)
# panda_joint8 (fixed): origin xyz = (0, 0, 0.107) → axis 2 (z), magnitude 0.107m
#               this is wrist-to-flange distance.
LINK_PARAMS = {
    "d3": {"joint": "panda_joint3", "axis": 1, "base": -0.316, "sign": -1,  # negative-y
           "name": "upper_arm (panda_link2 → link3 along y)"},
    "d5": {"joint": "panda_joint5", "axis": 1, "base":  0.384, "sign": +1,  # positive-y
           "name": "forearm (panda_link4 → link5 along y)"},
    "d8": {"joint": "panda_joint8", "axis": 2, "base":  0.107, "sign": +1,  # positive-z
           "name": "wrist-to-flange (panda_link7 → link8 along z)"},
}


def applied_delta(spec_key: str, delta_mm: float) -> float:
    """Convert signed mm to actual xyz shift such that |new| = |base| + delta_mm/1000."""
    p = LINK_PARAMS[spec_key]
    # We want the *length* (|new|) = |base| + delta_m.
    # axis component shifts in the direction of `sign` so the magnitude grows.
    return float(p["sign"]) * (delta_mm * 1e-3)


# ---------------------------------------------------------------------------
# Per-method samplers — produce q-trajectory only (original-Franka sampling)
# ---------------------------------------------------------------------------


def _build_orig_arms(device: str, has_stage1: bool, s1_args: dict):
    arm_a = Franka7DoF(urdf_path=URDF_ORIG, end_link="panda_hand", tool_z_max=0.30)
    arm_a._ensure_chain(torch.zeros(1, 7, device=device))
    if not has_stage1:
        return arm_a, arm_a
    delta_net = DeltaResidualMLP(n_q=7, n_p=3, n_z=1,
                                  hidden=s1_args["hidden"], n_layers=s1_args["n_layers"],
                                  activation=torch.nn.Softplus,
                                  final_init_scale=1e-3).to(device)
    s1_ck = torch.load("outputs/franka_stage1/delta_phi.pt",
                       map_location=device, weights_only=False)
    delta_net.load_state_dict(s1_ck["delta_net_state"]); delta_net.eval()
    arm = LearnedSelfModelFranka7DoF(
        delta_net=delta_net, urdf_path=URDF_ORIG, end_link="panda_hand", tool_z_max=0.30)
    arm._ensure_chain(torch.zeros(1, 7, device=device))
    return arm_a, arm


def _sample_bc(ck, ctx, device):
    a = ck["args"]
    model = BCTrajectoryPredictor(
        n_q=7, H=a["H"], ctx_dim=ctx.shape[-1],
        hidden=a["bc_hidden"], n_layers=a["bc_layers"], activation="sin").to(device)
    state = ck["ema_state"] if "ema_state" in ck else ck["model_state"]
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        return model(ctx)                                            # (n, H+1, 7)


def _sample_dp(ck, ctx, device, channel: bool, n_inference_steps: int):
    a = ck["args"]; n = ctx.shape[0]; H1 = a["H"] + 1
    if channel:
        model, sched = make_official_diffusion_policy(
            n_q=7 + ctx.shape[-1], global_cond_dim=None,
            down_dims=list(a["down_dims"]),
            diffusion_step_embed_dim=a["diff_step_embed"],
            n_train_timesteps=a["dp_train_timesteps"])
    else:
        model, sched = make_official_diffusion_policy(
            n_q=7, global_cond_dim=ctx.shape[-1],
            down_dims=list(a["down_dims"]),
            diffusion_step_embed_dim=a["diff_step_embed"],
            n_train_timesteps=a["dp_train_timesteps"])
    model = model.to(device); model.load_state_dict(ck["ema_state"]); model.eval()
    if channel:
        ctx_per_h = ctx.unsqueeze(1).expand(-1, H1, -1)
        with torch.no_grad():
            return channel_concat_dp_sample(model, sched, batch_size=n, horizon=H1, n_q=7,
                                             ctx_per_h=ctx_per_h, device=device,
                                             n_inference_steps=n_inference_steps)
    else:
        with torch.no_grad():
            return official_dp_sample(model, sched, batch_size=n, horizon=H1, n_q=7,
                                       ctx=ctx, device=device,
                                       n_inference_steps=n_inference_steps)


def _sample_dp_c(ck, ctx, p_targets, p_starts, z_e_per_traj, device,
                 n_inference_steps: int, alpha_g: float, alpha_s: float, alpha_v: float,
                 fk_for_guidance):
    a = ck["args"]; n = ctx.shape[0]; H1 = a["H"] + 1
    model, sched = make_official_diffusion_policy(
        n_q=7 + ctx.shape[-1], global_cond_dim=None,
        down_dims=list(a["down_dims"]),
        diffusion_step_embed_dim=a["diff_step_embed"],
        n_train_timesteps=a["dp_train_timesteps"])
    model = model.to(device); model.load_state_dict(ck["ema_state"]); model.eval()
    ctx_per_h = ctx.unsqueeze(1).expand(-1, H1, -1)
    grh = list(range(H1 - H1 // 2, H1)); sah = [0]
    return channel_concat_dp_sample_guided(
        model, sched, batch_size=n, horizon=H1, n_q=7,
        ctx_per_h=ctx_per_h, device=device, n_inference_steps=n_inference_steps,
        fk_fn=fk_for_guidance, z_e_per_traj=z_e_per_traj,
        p_target=p_targets, alpha_g=alpha_g, h_indices_goal=grh,
        p_start_anchor=p_starts, alpha_s=alpha_s, h_indices_start=sah,
        alpha_v=alpha_v,
    )


def _sample_projected(ck, ctx, device, n_inference_steps: int):
    a = ck["args"]; n = ctx.shape[0]; H1 = a["H"] + 1
    model, sched = make_official_diffusion_policy(
        n_q=7 + 3, global_cond_dim=ctx.shape[-1],
        down_dims=list(a["down_dims"]),
        diffusion_step_embed_dim=a["diff_step_embed"],
        n_train_timesteps=a["dp_train_timesteps"])
    model = model.to(device); model.load_state_dict(ck["ema_state"]); model.eval()
    with torch.no_grad():
        x_amb = official_dp_sample(model, sched, batch_size=n, horizon=H1, n_q=7+3,
                                    ctx=ctx, device=device,
                                    n_inference_steps=n_inference_steps)
    return x_amb[..., :7]                                             # q only


def _sample_ours(ck, p_targets, p_starts, z_e, device,
                 n_steps: int, alpha_g: float, alpha_s: float, alpha_v: float):
    a = ck["args"]
    has_stage1 = "stage1_args" in ck
    s1 = ck["stage1_args"] if has_stage1 else {}
    arm_a, arm = _build_orig_arms(device, has_stage1, s1)
    if "metric" in a:
        arm.metric = a["metric"]
    schedule = LinearBetaSchedule(beta_0=a["beta_0"], beta_f=a["beta_f"], t0=0.0, tf=1.0)
    limiting = WrappedNormalFranka7DoF(arm, mean_q=list(a["limiting_mean_q"]),
                                       scale=a["limiting_scale"],
                                       z_e_range=(a["z_min"], a["z_max"]))
    sde = LangevinSDE(arm, schedule, limiting)
    H1 = a["H"] + 1; d = arm.ambient_dim; n = p_targets.shape[0]
    use_p_start = bool(a.get("use_p_start_cond", False))
    GOAL_DIM = 6 if use_p_start else 3
    cond_inj = a.get("cond_injection", "global")
    net = TrajectoryScoreNetUNet(
        arm, H=a["H"], down_dims=tuple(a["down_dims"]),
        diffusion_step_embed_dim=a["diff_step_embed"],
        n_groups=a["unet_groups"], kernel_size=a["unet_kernel"],
        t_scale=a["t_scale"], goal_cond_dim=GOAL_DIM, cond_injection=cond_inj,
    ).to(device)
    net.load_state_dict(ck["ema_state"]); net.eval()
    score_fn = TrajectoryScaledScoreFn(net, sde)
    cond_for_eval = torch.cat([p_targets, p_starts], dim=-1) if use_p_start else p_targets
    grh = list(range(H1 - H1 // 2, H1)); sah = [0]
    z_lim = z_e.unsqueeze(1).expand(-1, H1, -1).reshape(n * H1, -1)
    tau_T = limiting.sample(n * H1, device=device, z_e=z_lim).reshape(n, H1, d)
    with torch.no_grad():
        tau_gen = traj_reverse_grw(
            sde, score_fn, tau_T, n_steps=n_steps, eps=a["eps"],
            goal_cond=cond_for_eval, guidance_scale=0.0,
            goal_residual_alpha=alpha_g, goal_residual_h=grh,
            p_start=p_starts, start_anchor_alpha=alpha_s, start_anchor_h=sah,
            smoothness_alpha_vel=alpha_v, smoothness_alpha_acc=0.0,
        )
    return tau_gen[..., :7], arm, arm_a


# ---------------------------------------------------------------------------
# Main: sweep distortions, all methods
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--z-eval", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dp-inference-steps", type=int, default=100)
    p.add_argument("--ours-sample-steps", type=int, default=200)
    p.add_argument("--alpha-g", type=float, default=100.0)
    p.add_argument("--alpha-s", type=float, default=100.0)
    p.add_argument("--alpha-v", type=float, default=5.0)
    p.add_argument("--dp-c-alpha-g", type=float, default=30.0)
    p.add_argument("--dp-c-alpha-s", type=float, default=30.0)
    p.add_argument("--dp-c-alpha-v", type=float, default=3.0)
    p.add_argument("--links", type=str, default="d3,d5",
                   help="comma-separated list of links to sweep (subset of "
                        f"{list(LINK_PARAMS.keys())}); pass 'd5' for single-link.")
    p.add_argument("--deltas-mm", type=float, nargs="+",
                   default=[-50, -30, -10, 0, 10, 30, 50, 100])
    p.add_argument("--out", type=str,
                   default="outputs/diagnostic/link_morph_all.json",
                   help="single consolidated JSON with all (link × δ × method) results")
    p.add_argument("--md-out", type=str,
                   default="outputs/diagnostic/link_morph_all.md",
                   help="markdown summary file (paper-readable tables)")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "cpu"],
                   help="auto = cuda if free memory > 2GB else cpu")
    args = p.parse_args()
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    md_path = Path(args.md_out); md_path.parent.mkdir(parents=True, exist_ok=True)
    link_keys = [k.strip() for k in args.links.split(",") if k.strip()]
    for k in link_keys:
        if k not in LINK_PARAMS:
            raise ValueError(f"unknown link '{k}', choose from {list(LINK_PARAMS.keys())}")

    if args.device == "auto":
        if torch.cuda.is_available():
            free_b, _ = torch.cuda.mem_get_info()
            free_gb = free_b / (1 << 30)
            device = "cuda" if free_gb > 2.0 else "cpu"
            print(f"[device-auto] cuda free = {free_gb:.2f} GB → using {device}")
        else:
            device = "cpu"
    else:
        device = args.device
    torch.manual_seed(args.seed)
    n = args.n; H1 = 16

    box_lo = torch.tensor([0.40, -0.05, 0.40], device=device)
    box_hi = torch.tensor([0.50,  0.05, 0.50], device=device)
    torch.manual_seed(args.seed + 1000)
    p_targets = box_lo + (box_hi - box_lo) * torch.rand(n, 3, device=device)
    torch.manual_seed(args.seed + 2000)
    p_starts = box_lo + (box_hi - box_lo) * torch.rand(n, 3, device=device)
    z_e = torch.full((n, 1), args.z_eval, device=device)
    ctx = torch.cat([p_targets, p_starts, z_e], dim=-1)

    print(f"\nLink-morphology eval — perturbing {link_keys}")
    for k in link_keys:
        s = LINK_PARAMS[k]
        print(f"  {k}: {s['name']}, base |xyz[{s['axis']}]| = {abs(s['base']):.4f} m")
    print(f"  z_e (tool tip) = {args.z_eval} m, n = {n} per condition\n")

    # ---- Pre-load all checkpoints once ----
    ckpts = {
        "BC":            torch.load("outputs/franka_baseline_bc/ckpt.pt",
                                    map_location=device, weights_only=False),
        "DP-canonical":  torch.load("outputs/franka_baseline_dp_official/ckpt.pt",
                                    map_location=device, weights_only=False),
        "DP-A":          torch.load("outputs/franka_baseline_dp_official_channel/ckpt.pt",
                                    map_location=device, weights_only=False),
        "Projected":     torch.load("outputs/franka_baseline_projected/ckpt.pt",
                                    map_location=device, weights_only=False),
        "Ours-Analytic": torch.load("outputs/franka_traj_unet_v2_analytic/ckpt_riemannian.pt",
                                    map_location=device, weights_only=False),
        "Ours-V2":       torch.load("outputs/franka_traj_unet_v2/ckpt_riemannian.pt",
                                    map_location=device, weights_only=False),
    }
    # DP-C uses the DP-A ckpt + sampling-time guidance.
    ckpts["DP-C"] = ckpts["DP-A"]

    # ---- Sample once per method (sampling is morphology-independent) ----
    # Free models after sampling to reduce GPU memory pressure.
    print("Sampling q-trajectories (original-Franka manifold)...")
    method_q = {}
    def _flush():
        if device == "cuda":
            torch.cuda.empty_cache()

    method_q["BC"] = _sample_bc(ckpts["BC"], ctx, device).detach(); _flush()
    print(f"  BC done.")
    method_q["DP-canonical"] = _sample_dp(
        ckpts["DP-canonical"], ctx, device,
        channel=False, n_inference_steps=args.dp_inference_steps).detach(); _flush()
    print(f"  DP-canonical done.")
    method_q["DP-A"] = _sample_dp(
        ckpts["DP-A"], ctx, device,
        channel=True, n_inference_steps=args.dp_inference_steps).detach(); _flush()
    print(f"  DP-A done.")
    method_q["Projected"] = _sample_projected(
        ckpts["Projected"], ctx, device,
        n_inference_steps=args.dp_inference_steps).detach(); _flush()
    print(f"  Projected done.")
    # DP-C guidance uses the original-Franka FK (LearnedSelfModelFranka7DoF).
    has_stage1_dpc = "stage1_args" in ckpts["DP-C"]
    s1_dpc = ckpts["DP-C"]["stage1_args"] if has_stage1_dpc else {}
    arm_a_for_dpc, arm_for_dpc = _build_orig_arms(device, has_stage1_dpc, s1_dpc)
    method_q["DP-C"] = _sample_dp_c(
        ckpts["DP-C"], ctx, p_targets, p_starts, z_e, device,
        n_inference_steps=args.dp_inference_steps,
        alpha_g=args.dp_c_alpha_g, alpha_s=args.dp_c_alpha_s, alpha_v=args.dp_c_alpha_v,
        fk_for_guidance=arm_for_dpc.F).detach()
    del arm_for_dpc, arm_a_for_dpc; _flush()
    print(f"  DP-C done.")
    q_a, _, _ = _sample_ours(
        ckpts["Ours-Analytic"], p_targets, p_starts, z_e, device,
        n_steps=args.ours_sample_steps,
        alpha_g=args.alpha_g, alpha_s=args.alpha_s, alpha_v=args.alpha_v)
    method_q["Ours-Analytic"] = q_a.detach(); _flush()
    print(f"  Ours-Analytic done.")
    q_v2, _, _ = _sample_ours(
        ckpts["Ours-V2"], p_targets, p_starts, z_e, device,
        n_steps=args.ours_sample_steps,
        alpha_g=args.alpha_g, alpha_s=args.alpha_s, alpha_v=args.alpha_v)
    method_q["Ours-V2"] = q_v2.detach(); _flush()
    print(f"  Ours-V2 done.")
    print("  All sampling done.\n")

    # ---- Reference arm (learned manifold under original Franka) for drift metric ----
    arm_demo_a, arm_demo = _build_orig_arms(device, True, ckpts["Ours-V2"]["stage1_args"])
    h_mid = H1 // 2
    # Free ckpts (we only need state already loaded into method_q).
    del ckpts; _flush()

    # ---- Sweep over (link, distortion) ----
    results = {
        "config": {
            "z_eval": args.z_eval, "n": n,
            "deltas_mm": args.deltas_mm, "links": link_keys,
            "alpha_g": args.alpha_g, "alpha_s": args.alpha_s, "alpha_v": args.alpha_v,
            "dp_c_alpha_g": args.dp_c_alpha_g, "dp_c_alpha_s": args.dp_c_alpha_s,
            "dp_c_alpha_v": args.dp_c_alpha_v,
            "ours_sample_steps": args.ours_sample_steps,
            "dp_inference_steps": args.dp_inference_steps,
            "seed": args.seed,
        },
        "link_specs": {k: LINK_PARAMS[k] for k in link_keys},
        "by_link": {k: {"by_method": {m: {} for m in method_q.keys()}}
                     for k in link_keys},
    }

    print(f"{'link':>5}  {'δ (mm)':>8}  {'method':>15}  {'pos_err':>9}  {'std':>8}  "
          f"{'s@2':>5}  {'s@5':>5}  {'s@10':>5}  {'frac_A':>7}  "
          f"{'EE_excess':>9}  {'max|g|':>9}")

    for link_key in link_keys:
        spec = LINK_PARAMS[link_key]
        for delta_mm in args.deltas_mm:
            # Build modified URDF + chain
            urdf_mod_path = (f"outputs/franka_modified/"
                              f"panda_{link_key}_{int(delta_mm):+d}mm.urdf")
            delta_xyz = applied_delta(link_key, delta_mm)
            make_modified_urdf(URDF_ORIG, spec["joint"], spec["axis"],
                                delta_xyz, urdf_mod_path)
            arm_mod = Franka7DoF(urdf_path=urdf_mod_path, end_link="panda_hand",
                                  tool_z_max=0.30)
            arm_mod._ensure_chain(torch.zeros(1, 7, device=device))

            z_flat = z_e.unsqueeze(1).expand(-1, H1, -1).reshape(-1, 1)

            for mname, q_gen in method_q.items():
                q_flat = q_gen.reshape(-1, 7)
                # End-effector under modified Franka FK
                p_mod = arm_mod.F(q_flat, z_flat).reshape(n, H1, 3)
                p_end = p_mod[:, -1, :]
                pos_err = (p_end - p_targets).norm(dim=-1)
                s2 = (pos_err < 0.02).float().mean().item()
                s5 = (pos_err < 0.05).float().mean().item()
                s10 = (pos_err < 0.10).float().mean().item()

                q1_mid = q_gen[:, h_mid, 0]
                mode_A = q1_mid > 0
                frac_A = mode_A.float().mean().item()

                # EE path metrics under modified robot
                seg = (p_mod[:, 1:] - p_mod[:, :-1]).norm(dim=-1).sum(dim=1)
                straight = (p_mod[:, -1] - p_mod[:, 0]).norm(dim=-1)
                ee_excess = (seg / (straight + 1e-9)).mean().item()
                ee_path_len = seg.mean().item()

                # Manifold-adherence gap: |F_orig(q) - F_modified(q)| on generated q
                with torch.no_grad():
                    p_orig = arm_demo.F(q_flat, z_flat)
                    p_mod_flat = p_mod.reshape(-1, 3)
                    drift = (p_orig - p_mod_flat).norm(dim=-1)
                    max_drift = drift.max().item()
                    mean_drift = drift.mean().item()

                viol = arm_mod.violates_limits(q_flat).float().mean().item()
                vel = (q_gen[:, 1:] - q_gen[:, :-1]).norm(dim=-1).mean().item()

                print(f"{link_key:>5}  {delta_mm:+8.0f}  {mname:>15}  "
                      f"{pos_err.mean().item():>8.4f}m "
                      f"{pos_err.std().item():>7.4f}m  "
                      f"{s2*100:>4.1f}%  {s5*100:>4.1f}%  {s10*100:>4.1f}%  "
                      f"{frac_A:>7.3f}  {ee_excess:>9.2f}  "
                      f"{max_drift*1000:>7.1f}mm")

                results["by_link"][link_key]["by_method"][mname][
                    f"delta_{int(delta_mm):+d}mm"] = {
                    "pos_err_mean": pos_err.mean().item(),
                    "pos_err_std":  pos_err.std().item(),
                    "succ_2cm": s2, "succ_5cm": s5, "succ_10cm": s10,
                    "frac_A": frac_A,
                    "ee_path_len_mean_m": ee_path_len,
                    "ee_excess_mean": ee_excess,
                    "vel_mean": vel,
                    "viol": viol,
                    "max_orig_to_mod_drift_m": max_drift,
                    "mean_orig_to_mod_drift_m": mean_drift,
                }

            out_path.write_text(json.dumps(results, indent=2))
        print()                                                                # blank between links

    # ---- Markdown summary ----
    _write_markdown_summary(results, md_path)
    print(f"\nSaved JSON: {out_path}")
    print(f"Saved MD:   {md_path}")


def _write_markdown_summary(results: dict, md_path: Path) -> None:
    """Write paper-readable markdown tables to md_path."""
    cfg = results["config"]
    lines = []
    lines.append("# Franka Link-Morphology Transfer — Eval Results\n")
    lines.append(f"- z_e = {cfg['z_eval']}, n = {cfg['n']} per (link, δ, method)")
    lines.append(f"- Ours guidance α = ({cfg['alpha_g']}, {cfg['alpha_s']}, {cfg['alpha_v']})")
    lines.append(f"- DP-C guidance α = ({cfg['dp_c_alpha_g']}, {cfg['dp_c_alpha_s']}, {cfg['dp_c_alpha_v']})")
    lines.append(f"- Ours reverse steps = {cfg['ours_sample_steps']}, DP inference steps = {cfg['dp_inference_steps']}\n")
    lines.append("Sampling uses the original Franka manifold (no model adaptation). "
                 "Ground-truth EE position is computed under the *modified* Franka.\n")

    deltas = cfg["deltas_mm"]
    methods = ["BC", "DP-canonical", "DP-A", "DP-C", "Projected",
               "Ours-Analytic", "Ours-V2"]

    for link_key in cfg["links"]:
        spec = results["link_specs"][link_key]
        lines.append(f"## Link `{link_key}` — {spec['name']}\n")
        lines.append(f"Base length |xyz[{spec['axis']}]| = {abs(spec['base']):.4f} m. "
                     f"Joint = `{spec['joint']}`.\n")

        for metric_key, metric_label, scale, fmt in [
            ("pos_err_mean", "pos_err mean (mm)", 1000.0, ".1f"),
            ("succ_5cm", "succ@5cm (%)", 100.0, ".1f"),
            ("succ_10cm", "succ@10cm (%)", 100.0, ".1f"),
            ("ee_excess_mean", "EE excess (× straight)", 1.0, ".2f"),
            ("max_orig_to_mod_drift_m", "max F_orig→F_mod drift (mm)", 1000.0, ".1f"),
        ]:
            header = "| method | " + " | ".join(f"δ={d:+.0f}mm" for d in deltas) + " |"
            sep = "|" + "---|" * (len(deltas) + 1)
            lines.append(f"### {metric_label}\n")
            lines.append(header); lines.append(sep)
            for m in methods:
                row = [m]
                for d in deltas:
                    cell = results["by_link"][link_key]["by_method"][m].get(
                        f"delta_{int(d):+d}mm", {}).get(metric_key)
                    if cell is None:
                        row.append("—")
                    else:
                        row.append(f"{cell * scale:{fmt}}")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

    md_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
