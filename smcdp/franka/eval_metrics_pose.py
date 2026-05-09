"""Comprehensive pose-extended evaluation metrics — implements `metric.md`.

Computes the full metric set required for Ours-V2 (Method A) vs baselines comparison:

  Primary:
    - pos_err mean/median/std/max  (cm)
    - rot_err mean/median/std/max  (deg)
    - pos_succ@5cm, pos_succ@2cm
    - rot_succ@5deg, rot_succ@10deg
    - pose_succ@(5cm,5°), pose_succ@(5cm,10°), pose_succ@(2cm,5°)

  Self-model pose consistency:
    - max/mean manifold position gap (mm)
    - max/mean manifold rotation gap (deg)
    - quaternion norm error (mean/max), invalid_quat_rate

  Multimodality:
    - frac_A_gen, frac_A_demo, mode_frac_err, mode_collapse, frac_between
    - sliced W_1^q  (joint trajectory space)

  Trajectory quality:
    - E_vel, E_acc, E_p_dot, E_omega

  Physical feasibility:
    - joint_viol_rate, e_jl, joint_margin

Compatible with both Ours-V2 (chart-form score-net + retraction) and baselines
(BC, DP-*, Projected).  Caller passes `samples` with shape (B, H+1, ambient_dim_pose)
where ambient_dim_pose = n_q + 7 (storage form) + n_z.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from smcdp.lie_se3 import (
    log_relative_Rp, log_SO3, quat_to_R, pose7_to_Rp,
)


def _safe_median(t: Tensor) -> float:
    return torch.median(t).item()


def compute_pose_metrics(
    arm,                                       # EmbodimentPoseGraphManifold (or subclass)
    samples: Tensor,                           # (B, H+1, ambient_dim_pose)
    T_target: Tensor,                          # (B, 7)  storage form (quat 4 + p 3)
    *,
    x_demo: Optional[Tensor] = None,           # (B, H+1, ambient_dim_pose) — demo reference
    n_w1_dirs: int = 64,
    sigma_R: float = 0.1,
    quat_tol: float = 0.01,
) -> dict:
    """Compute full pose-extended metric set.  See `metric.md` for definitions."""
    device = samples.device
    dtype = samples.dtype
    B, H1, _ = samples.shape
    n_q = arm.n_q

    # ---- 1. Primary endpoint metrics ----
    x_H = samples[:, -1, :]
    q_H, q_R_H, p_H, _ = arm.split_x(x_H)
    R_H = quat_to_R(q_R_H)
    R_target, p_target = pose7_to_Rp(T_target)
    e = log_relative_Rp(R_H, p_H, R_target, p_target)            # (B, 6) body-frame twist
    e_p = e[..., :3].norm(dim=-1)                                 # (B,) m
    e_R = e[..., 3:].norm(dim=-1)                                 # (B,) rad

    # success thresholds
    th_5cm  = 0.05
    th_2cm  = 0.02
    th_5deg = math.radians(5.0)
    th_10deg = math.radians(10.0)

    primary = {
        "pos_err_mean_cm":   100.0 * e_p.mean().item(),
        "pos_err_median_cm": 100.0 * _safe_median(e_p),
        "pos_err_std_cm":    100.0 * e_p.std().item(),
        "pos_err_max_cm":    100.0 * e_p.max().item(),
        "rot_err_mean_deg":   math.degrees(e_R.mean().item()),
        "rot_err_median_deg": math.degrees(_safe_median(e_R)),
        "rot_err_std_deg":    math.degrees(e_R.std().item()),
        "rot_err_max_deg":    math.degrees(e_R.max().item()),
        "pos_succ_5cm":  (e_p < th_5cm).float().mean().item(),
        "pos_succ_2cm":  (e_p < th_2cm).float().mean().item(),
        "rot_succ_5deg":  (e_R < th_5deg).float().mean().item(),
        "rot_succ_10deg": (e_R < th_10deg).float().mean().item(),
        "pose_succ_5cm_5deg":  ((e_p < th_5cm)  & (e_R < th_5deg)).float().mean().item(),
        "pose_succ_5cm_10deg": ((e_p < th_5cm)  & (e_R < th_10deg)).float().mean().item(),
        "pose_succ_2cm_5deg":  ((e_p < th_2cm)  & (e_R < th_5deg)).float().mean().item(),
    }

    # ---- 2. Manifold gap (per-step, per-trajectory) ----
    # arm.constraint returns (..., 6) twist g = Log_SE3(T_φ^{-1} T)
    g_all = arm.constraint(samples.reshape(B * H1, -1)).reshape(B, H1, 6)
    g_p = g_all[..., :3].norm(dim=-1)                             # (B, H+1) m
    g_R = g_all[..., 3:].norm(dim=-1)                             # (B, H+1) rad

    manifold = {
        "manif_pos_max_mm":   1000.0 * g_p.max().item(),
        "manif_pos_mean_mm":  1000.0 * g_p.mean().item(),
        "manif_rot_max_deg":  math.degrees(g_R.max().item()),
        "manif_rot_mean_deg": math.degrees(g_R.mean().item()),
        "manif_pose_combined_max":  (g_p.pow(2)/(arm.sigma_p**2) + g_R.pow(2)/(sigma_R**2)).sqrt().max().item(),
    }

    # ---- 3. Quaternion validity ----
    q_R_all = samples[..., n_q : n_q + 4]                         # (B, H+1, 4)
    q_norm = q_R_all.norm(dim=-1)                                  # (B, H+1)
    q_norm_err = (q_norm - 1.0).abs()
    quat = {
        "quat_norm_mean": q_norm_err.mean().item(),
        "quat_norm_max":  q_norm_err.max().item(),
        "invalid_quat_rate": (q_norm_err > quat_tol).float().mean().item(),
    }

    # ---- 4. Multimodality ----
    h_mid = H1 // 2
    q1_mid_gen = samples[:, h_mid, 0]                              # 1st joint at trajectory midpoint
    frac_A_gen = (q1_mid_gen > 0).float().mean().item()
    if x_demo is not None:
        q1_mid_demo = x_demo[:, h_mid, 0]
        frac_A_demo = (q1_mid_demo > 0).float().mean().item()
    else:
        frac_A_demo = 0.5                                          # designed bimodal balance
    multimodal = {
        "frac_A_gen": frac_A_gen,
        "frac_A_demo": frac_A_demo,
        "mode_frac_err": abs(frac_A_gen - frac_A_demo),
        "mode_collapse": float(frac_A_gen > 0.9 or frac_A_gen < 0.1),
        "frac_between": (q1_mid_gen.abs() < 0.05).float().mean().item(),
    }

    # ---- 5. Sliced W_1 in joint-trajectory space ----
    if x_demo is not None and B >= 8:
        q_gen_flat  = samples[..., :n_q].reshape(B, H1 * n_q)      # (B, H+1, n_q) → flat
        q_demo_flat = x_demo[..., :n_q].reshape(x_demo.shape[0], H1 * n_q)
        m = min(B, x_demo.shape[0])
        ig = torch.randperm(B, device=device)[:m]
        id_ = torch.randperm(x_demo.shape[0], device=device)[:m]
        q_gen_flat = q_gen_flat[ig]
        q_demo_flat = q_demo_flat[id_]
        dirs = torch.randn(n_w1_dirs, q_gen_flat.shape[-1], device=device, dtype=dtype)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        proj_gen  = q_gen_flat  @ dirs.T                           # (m, n_dir)
        proj_demo = q_demo_flat @ dirs.T
        proj_gen_sorted, _  = proj_gen.sort(0)
        proj_demo_sorted, _ = proj_demo.sort(0)
        W1_q = (proj_gen_sorted - proj_demo_sorted).abs().mean().item()
        multimodal["W1_q"] = W1_q
    else:
        multimodal["W1_q"] = float("nan")

    # ---- 6. Trajectory smoothness ----
    q_traj = samples[..., :n_q]                                    # (B, H+1, n_q)
    q_diff = q_traj[:, 1:, :] - q_traj[:, :-1, :]                  # (B, H, n_q)
    E_vel  = q_diff.pow(2).sum(-1).mean().item()
    if H1 > 2:
        q_acc  = q_traj[:, 2:, :] - 2*q_traj[:, 1:-1, :] + q_traj[:, :-2, :]
        E_acc  = q_acc.pow(2).sum(-1).mean().item()
    else:
        E_acc = 0.0
    p_traj = samples[..., n_q + 4 : n_q + 7]
    p_diff = p_traj[:, 1:, :] - p_traj[:, :-1, :]
    E_p_dot = p_diff.pow(2).sum(-1).mean().item()
    R_traj = quat_to_R(samples[..., n_q : n_q + 4])                # (B, H+1, 3, 3)
    R_inv_R_next = R_traj[:, :-1, :, :].transpose(-1, -2) @ R_traj[:, 1:, :, :]
    omega = log_SO3(R_inv_R_next.reshape(-1, 3, 3)).reshape(B, H1 - 1, 3)
    E_omega = omega.pow(2).sum(-1).mean().item()
    smoothness = {"E_vel": E_vel, "E_acc": E_acc, "E_p_dot": E_p_dot, "E_omega": E_omega}

    # ---- 7. Physical feasibility ----
    if hasattr(arm, "joint_limits"):
        q_lo, q_hi = arm.joint_limits(device=device, dtype=dtype)  # (n_q,) each
        viol_per_traj = ((q_traj < q_lo) | (q_traj > q_hi)).any(-1).any(-1)  # (B,)
        joint_viol_rate = viol_per_traj.float().mean().item()
        excess_lo = (q_lo - q_traj).clamp(min=0)
        excess_hi = (q_traj - q_hi).clamp(min=0)
        e_jl = (excess_lo.pow(2) + excess_hi.pow(2)).sum(-1).mean().item()
        m_lo = (q_traj - q_lo) / (q_hi - q_lo).clamp(min=1e-6)
        m_hi = (q_hi - q_traj) / (q_hi - q_lo).clamp(min=1e-6)
        m_min = torch.minimum(m_lo, m_hi).min().item()
        physical = {
            "joint_viol_rate": joint_viol_rate,
            "e_jl": e_jl,
            "joint_margin_min": m_min,
        }
    else:
        physical = {"joint_viol_rate": float("nan"), "e_jl": float("nan"), "joint_margin_min": float("nan")}

    return {
        **primary,
        **manifold,
        **quat,
        **multimodal,
        **smoothness,
        **physical,
    }


# Concise summary for log printing
PRIMARY_KEYS = [
    "pos_err_mean_cm", "rot_err_mean_deg",
    "pose_succ_5cm_5deg", "pose_succ_5cm_10deg",
    "manif_pos_max_mm", "manif_rot_max_deg",
    "mode_frac_err", "joint_viol_rate",
]


def format_row(z_e: float, m: dict) -> str:
    return (f"{z_e:>5.2f} | "
            f"{m['pos_err_mean_cm']:>6.2f} | "
            f"{m['rot_err_mean_deg']:>6.2f} | "
            f"{m['pose_succ_5cm_5deg']:>6.2%} | "
            f"{m['pose_succ_5cm_10deg']:>6.2%} | "
            f"{m['manif_pos_max_mm']:>7.3f} | "
            f"{m['manif_rot_max_deg']:>7.3f} | "
            f"{m['mode_frac_err']:>5.2f} | "
            f"{m['joint_viol_rate']:>5.1%}")


def format_header() -> str:
    return f"{'z_e':>5} | {'p(cm)':>6} | {'R(°)':>6} | {'5/5°':>6} | {'5/10°':>6} | {'g_p mm':>7} | {'g_R °':>7} | {'mfe':>5} | {'jvio':>5}"
