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
    # ---- mode classifier (auto-detect from q_rest_A/B if provided) ----
    # In Tier 0 demos q_rest_A[0]=+0.6, q_rest_B[0]=-0.6 so q[0] is the natural
    # mode discriminator; in Tier 2 v5 q_rest_A/B differ in q[3]/q[5] instead,
    # so a fixed q[0]>0 rule misclassifies.  Pass q_rest_A/B (the trainer's
    # `--q-rest-A`/`-B` lists) and we auto-pick the joint with largest |A−B|.
    q_rest_A: Optional[list] = None,
    q_rest_B: Optional[list] = None,
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

    # ---- chart-aware physical-q recovery (joint_limit_extension v4.1) ----
    # When `arm` is a BoundedChartPoseManifold, samples[..., :n_q] stores u
    # (chart coord), not physical q.  For metrics that require physical q
    # (mode capture, W_1^q, joint feasibility), apply ψ via arm.physical_q.
    # For unwrapped manifold (v4) or IdentityChart, physical_q is identity.
    if hasattr(arm, "chart"):
        q_phys = arm.physical_q(samples)                            # (B, H+1, n_q)
        q_phys_demo = arm.physical_q(x_demo) if x_demo is not None else None
        is_bounded = True
    else:
        q_phys = samples[..., :n_q]                                 # v4: chart slot IS q
        q_phys_demo = x_demo[..., :n_q] if x_demo is not None else None
        is_bounded = False

    # ---- 4. Multimodality (in physical q-chart for fair v4↔v4.1 comparison) ----
    # Auto-pick the mode-discriminating joint from q_rest_A/B.  Falls back to
    # joint 0 with threshold 0 (legacy Tier 0 behaviour) when q_rests are unavailable.
    h_mid = H1 // 2
    if q_rest_A is not None and q_rest_B is not None:
        import numpy as _np
        A_arr = _np.asarray(q_rest_A, dtype=float)
        B_arr = _np.asarray(q_rest_B, dtype=float)
        diff = _np.abs(A_arr - B_arr)
        mode_joint = int(diff.argmax())
        mode_thresh = 0.5 * (A_arr[mode_joint] + B_arr[mode_joint])
        mode_A_above = bool(A_arr[mode_joint] > mode_thresh)
    else:
        mode_joint = 0
        mode_thresh = 0.0
        mode_A_above = True

    q_mid_gen = q_phys[:, h_mid, mode_joint]                        # mode joint at midpoint
    if mode_A_above:
        frac_A_gen = (q_mid_gen > mode_thresh).float().mean().item()
    else:
        frac_A_gen = (q_mid_gen < mode_thresh).float().mean().item()

    if q_phys_demo is not None:
        q_mid_demo = q_phys_demo[:, h_mid, mode_joint]
        if mode_A_above:
            frac_A_demo = (q_mid_demo > mode_thresh).float().mean().item()
        else:
            frac_A_demo = (q_mid_demo < mode_thresh).float().mean().item()
    else:
        frac_A_demo = 0.5                                          # designed bimodal balance

    multimodal = {
        "frac_A_gen": frac_A_gen,
        "frac_A_demo": frac_A_demo,
        "mode_frac_err": abs(frac_A_gen - frac_A_demo),
        "mode_collapse": float(frac_A_gen > 0.9 or frac_A_gen < 0.1),
        "frac_between": ((q_mid_gen - mode_thresh).abs() < 0.05).float().mean().item(),
        "mode_joint_idx": mode_joint,
        "mode_threshold": float(mode_thresh),
    }

    # ---- 5. Sliced W_1 in physical joint-trajectory space (v4.1 §13 caveat) ----
    if q_phys_demo is not None and B >= 8:
        q_gen_flat  = q_phys.reshape(B, H1 * n_q)
        q_demo_flat = q_phys_demo.reshape(q_phys_demo.shape[0], H1 * n_q)
        m = min(B, q_phys_demo.shape[0])
        ig = torch.randperm(B, device=device)[:m]
        id_ = torch.randperm(q_phys_demo.shape[0], device=device)[:m]
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
    # Spec v4.1 §13 requests both u-chart and q-chart smoothness when bounded.
    chart_slot = samples[..., :n_q]                                # u (bounded) or q (unwrapped)
    # Physical q-chart smoothness (default, baseline parity per v4.1 §12.3)
    q_diff = q_phys[:, 1:, :] - q_phys[:, :-1, :]
    E_vel  = q_diff.pow(2).sum(-1).mean().item()
    if H1 > 2:
        q_acc  = q_phys[:, 2:, :] - 2 * q_phys[:, 1:-1, :] + q_phys[:, :-2, :]
        E_acc  = q_acc.pow(2).sum(-1).mean().item()
    else:
        E_acc = 0.0
    smoothness = {"E_vel": E_vel, "E_acc": E_acc}
    # u-chart smoothness diagnostic (only meaningful for bounded chart)
    if is_bounded:
        u_diff = chart_slot[:, 1:, :] - chart_slot[:, :-1, :]
        smoothness["E_vel_u"] = u_diff.pow(2).sum(-1).mean().item()
        if H1 > 2:
            u_acc = chart_slot[:, 2:, :] - 2 * chart_slot[:, 1:-1, :] + chart_slot[:, :-2, :]
            smoothness["E_acc_u"] = u_acc.pow(2).sum(-1).mean().item()
        else:
            smoothness["E_acc_u"] = 0.0
    p_traj = samples[..., n_q + 4 : n_q + 7]
    p_diff = p_traj[:, 1:, :] - p_traj[:, :-1, :]
    E_p_dot = p_diff.pow(2).sum(-1).mean().item()
    R_traj = quat_to_R(samples[..., n_q : n_q + 4])                # (B, H+1, 3, 3)
    R_inv_R_next = R_traj[:, :-1, :, :].transpose(-1, -2) @ R_traj[:, 1:, :, :]
    omega = log_SO3(R_inv_R_next.reshape(-1, 3, 3)).reshape(B, H1 - 1, 3)
    E_omega = omega.pow(2).sum(-1).mean().item()
    smoothness["E_p_dot"] = E_p_dot
    smoothness["E_omega"] = E_omega

    # ---- 7. Physical feasibility (v4.1 §13: viol = 0 by construction when bounded) ----
    if hasattr(arm, "joint_limits"):
        q_lo, q_hi = arm.joint_limits(device=device, dtype=dtype)  # (n_q,) each
        viol_per_traj = ((q_phys < q_lo) | (q_phys > q_hi)).any(-1).any(-1)  # (B,)
        joint_viol_rate = viol_per_traj.float().mean().item()
        # Sample-wise exact effective_succ (diagnostic_plan.md §2.1):
        # joint_safe = ~viol_per_traj, then eff_succ = (pose_ok & joint_safe).mean().
        # The product form `succ * (1 - jvio)` is an independence-assumption
        # approximation; pose-failure and joint-violation are typically positively
        # correlated (a violated trajectory is also likelier to miss the target),
        # so the product overestimates effective success.
        joint_safe = (~viol_per_traj)
        pose_ok_55  = ((e_p < th_5cm) & (e_R < th_5deg))
        pose_ok_510 = ((e_p < th_5cm) & (e_R < th_10deg))
        pose_ok_55_2cm = ((e_p < th_2cm) & (e_R < th_5deg))
        primary["eff_pose_succ_5cm_5deg"]  = (pose_ok_55  & joint_safe).float().mean().item()
        primary["eff_pose_succ_5cm_10deg"] = (pose_ok_510 & joint_safe).float().mean().item()
        primary["eff_pose_succ_2cm_5deg"]  = (pose_ok_55_2cm & joint_safe).float().mean().item()
        primary["eff_pose_succ_5cm_5deg_product"]  = primary["pose_succ_5cm_5deg"]  * (1.0 - joint_viol_rate)
        primary["eff_pose_succ_5cm_10deg_product"] = primary["pose_succ_5cm_10deg"] * (1.0 - joint_viol_rate)
        excess_lo = (q_lo - q_phys).clamp(min=0)
        excess_hi = (q_phys - q_hi).clamp(min=0)
        e_jl = (excess_lo.pow(2) + excess_hi.pow(2)).sum(-1).mean().item()
        m_lo = (q_phys - q_lo) / (q_hi - q_lo).clamp(min=1e-6)
        m_hi = (q_hi - q_phys) / (q_hi - q_lo).clamp(min=1e-6)
        m_min = torch.minimum(m_lo, m_hi).min().item()
        physical = {
            "joint_viol_rate": joint_viol_rate,
            "e_jl": e_jl,
            "joint_margin_min": m_min,
        }
        # v4.1 §13 saturation diagnostic: ||u||_inf percentiles, only for bounded
        if is_bounded:
            u_inf = chart_slot.abs().reshape(-1, n_q).max(-1).values  # (B*H1,)
            sorted_u = u_inf.sort().values
            n = sorted_u.numel()
            def pct(p):
                idx = max(0, min(n - 1, int(round(p * (n - 1)))))
                return sorted_u[idx].item()
            physical["u_inf_p50"] = pct(0.50)
            physical["u_inf_p90"] = pct(0.90)
            physical["u_inf_p99"] = pct(0.99)
    else:
        physical = {"joint_viol_rate": float("nan"), "e_jl": float("nan"),
                     "joint_margin_min": float("nan")}

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


# =====================================================================
# joint_limit_extension v5.1 required metrics (spec §13)
# =====================================================================


@torch.no_grad()
def compute_reference_marginal_match(
    sde,                                         # PoseChartOUSDE
    tau_0: Tensor,                                # (B, H+1, ambient_dim_pose) — demo trajectories
    *,
    r: Optional[float] = None,                    # default tf (forward to r=K)
) -> dict:
    """Reference-marginal match diagnostic (v5.1 §13).

    Measures how well the forward marginal p_K matches the IK-free reference
    p_ref = N(0, Ḡ_Q^{-1}) — required by spec to justify the IK-free reference
    distribution choice for the chosen β_f.  Reports:

      - KL(emp(u_K) || p_ref)   in Gaussian closed form
      - W_2^2 approximation     (mean²-distance + trace gap on the diagonal)
      - empirical mean / variance (chart-coordinate-wise)
      - alpha(K), sigma²(K)     — finite-K mismatch indicators

    Per spec §8.2.4: the IK-free claim rests on p_∞ = N(0, Ḡ_Q^{-1}) being
    structurally exact; this metric checks how closely p_K (at the chosen β_f)
    approaches that stationary.  Smaller is better (β_f = 20 should yield
    α(K) ≈ 0.007 and KL ≪ 1).
    """
    from smcdp.trajectories_pose import traj_forward_ou_chart_pose
    manifold = sde.manifold
    schedule = sde.schedule
    n_q = manifold.n_q
    B, H1, _ = tau_0.shape
    device, dtype = tau_0.device, tau_0.dtype

    r_val = schedule.tf if r is None else float(r)
    r_b = torch.full((B,), r_val, device=device, dtype=dtype)
    tau_r = traj_forward_ou_chart_pose(sde, tau_0, r_b)
    u_r = tau_r[..., :n_q]                                                       # (B, H+1, n_q)
    u_flat = u_r.reshape(B * H1, n_q)                                            # samples for marginal estimate

    emp_mean = u_flat.mean(dim=0)                                                # (n_q,)
    emp_cov = (
        (u_flat - emp_mean.unsqueeze(0)).unsqueeze(-1)
        @ (u_flat - emp_mean.unsqueeze(0)).unsqueeze(-2)
    ).mean(dim=0)                                                                # (n_q, n_q)

    # Reference distribution Ḡ_Q^{-1} for KL.  For gbar_mode='identity' this is
    # I_{n_q}, so Ḡ_Q = I and log det Ḡ_Q = 0.  For other modes the SDE
    # container exposes gbar_apply / gbar_inv_apply etc. — extend here when
    # those modes are wired.
    if sde.gbar_mode == "identity":
        # KL(N(μ, Σ) || N(0, I))  =  ½ ( tr(Σ) + ‖μ‖² − n − log det Σ )
        sign, logdet_Sigma = torch.linalg.slogdet(emp_cov + 1e-8 * torch.eye(n_q, device=device, dtype=dtype))
        kl = 0.5 * (
            torch.diagonal(emp_cov).sum() + emp_mean.pow(2).sum() - n_q
            - sign * logdet_Sigma
        )
        # W_2^2 lower bound for diagonal-only approximation:
        #   ‖μ‖² + tr(Σ + I − 2 √Σ)  ≈  ‖μ‖² + Σ_i (√σ_ii − 1)²
        diag_var = torch.diagonal(emp_cov).clamp(min=0.0)
        w2_sq = emp_mean.pow(2).sum() + (diag_var.sqrt() - 1.0).pow(2).sum()
    else:
        # Hook for non-identity Ḡ_Q (v5.1 ablation modes "origin", "data_mean").
        raise NotImplementedError(
            f"reference-marginal KL not wired for gbar_mode='{sde.gbar_mode}'"
        )

    alpha_K = schedule.alpha(torch.tensor(r_val, device=device, dtype=dtype)).item()
    sigma2_K = schedule.sigma2(torch.tensor(r_val, device=device, dtype=dtype)).item()

    return {
        "r": r_val,
        "alpha_r": alpha_K,
        "sigma2_r": sigma2_K,
        "kl_emp_vs_pref": float(kl.item()),
        "w2sq_emp_vs_pref": float(w2_sq.item()),
        "emp_mean_norm": float(emp_mean.norm().item()),
        "emp_var_mean": float(torch.diagonal(emp_cov).mean().item()),
        "emp_var_min": float(torch.diagonal(emp_cov).min().item()),
        "emp_var_max": float(torch.diagonal(emp_cov).max().item()),
    }


@torch.no_grad()
def compute_chart_saturation(
    samples: Tensor,                              # (B, H+1, ambient_dim_pose) — generated trajectories
    n_q: int,
) -> dict:
    """Chart-saturation diagnostic on ‖u‖_∞ percentiles (v5.1 §13).

    Spec recommendation: healthy operating regime ‖u‖_∞ ≤ 2; saturation
    onset > 3 (sech² < 0.01 → near-degenerate D_ψ).  Reported alongside
    p50 / p90 / p99 over the sampled trajectories.
    """
    u = samples[..., :n_q]
    u_inf = u.abs().reshape(-1, n_q).max(-1).values                              # (B*H1,)
    sorted_u = u_inf.sort().values
    n = sorted_u.numel()

    def pct(p: float) -> float:
        idx = max(0, min(n - 1, int(round(p * (n - 1)))))
        return float(sorted_u[idx].item())

    return {
        "u_inf_p50": pct(0.50),
        "u_inf_p90": pct(0.90),
        "u_inf_p99": pct(0.99),
        "u_inf_max": float(sorted_u[-1].item()),
        "saturation_rate": float((u_inf > 3.0).float().mean().item()),
    }
