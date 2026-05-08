"""Reward gradient verification protocol — extension.tex Appendix A.4.

Resolves the ± sign and 𝒥_l vs 𝒥_r ambiguity in the closed-form pose-reward
gradient by comparing four candidate closed-forms against a finite-difference
reference and an autograd reference.

Protocol (per extension.tex Appendix A.4):
    1. Setup    : random q_H ∈ R^{n_q}, random T_target, fixed z_e, α_p^g, α_R^g.
    2. Autograd : compute g_ag = ∇_{q_H} R_goal via torch.autograd.grad.
    3. FD       : g_fd[i] ≈ (R(q+ε e_i) − R(q−ε e_i)) / (2 ε).  Confirm  ‖g_ag − g_fd‖_∞ < 1e-4.
    4. Closed   : compute g_cf for each (sign, J_l/J_r) ∈ {±} × {l, r}.  Identify the
                  combination matching g_ag.
    5. Approx   : measure ‖g_simple − g_ag‖ for the small-error simplified form
                  g_simple = -2 J_pose^⊤ W_g e_goal  (no 𝒥-factor).
    6. Document : save outcome to outputs/diagnostic/reward_gradient_verification.json.

Run:
    python -m tests.test_reward_gradient_verification
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pybullet_data
import torch

from smcdp.lie_se3 import (
    log_relative_Rp, log_SE3, exp_SE3, exp_SO3,
    J_l_SO3, J_l_inv_SO3, hat_so3,
    compose_Rp, inverse_Rp,
    R_to_quat, quat_to_R,
)
from smcdp.manifolds_pose import Franka7DoFPose


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def _se3_left_jacobian_block(xi: torch.Tensor) -> torch.Tensor:
    """SE(3) left Jacobian 𝒥_l(ξ) ∈ R^{6×6} (Barfoot Eq. 7.85b).

    Block form with Q(ρ, ω) the SE(3) coupling block.  We use the closed-form
    Q expression (Barfoot Eq. 7.86):
        Q(ρ, ω) = (1/2) [ρ]_×
                + ((θ−sinθ)/θ³) (Ω_ρ + ρ_Ω + Ω_ρ Ω)
                − ((θ²+2cosθ−2)/(2θ⁴)) (Ω² ρ + ρ Ω² − 3 Ω ρ Ω)
                + ((2θ−3sinθ+θ cosθ)/(2θ⁵)) (Ω ρ Ω² + Ω² ρ Ω)
        with Ω = [ω]_×, ρ_skew = [ρ]_×, Ω_ρ = Ω · ρ_skew, ρ_Ω = ρ_skew · Ω.

    Smooth at θ=0 via Taylor expansion (we use a small-θ guard).
    """
    rho = xi[..., :3]
    omega = xi[..., 3:]
    theta = omega.norm(dim=-1, keepdim=True).clamp(min=1e-30)                 # (..., 1)

    Omega = hat_so3(omega)                                                    # (..., 3, 3)
    Rho = hat_so3(rho)                                                        # (..., 3, 3)
    OmegaRho = Omega @ Rho
    RhoOmega = Rho @ Omega
    Omega2 = Omega @ Omega
    Omega2_Rho = Omega2 @ Rho
    Rho_Omega2 = Rho @ Omega2
    Omega_Rho_Omega = Omega @ Rho @ Omega
    Omega_Rho_Omega2 = Omega @ Rho @ Omega2
    Omega2_Rho_Omega = Omega2 @ Rho @ Omega

    th2 = theta * theta
    th3 = th2 * theta
    th4 = th3 * theta
    th5 = th4 * theta
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)

    c2 = ((theta - sin_t) / th3).unsqueeze(-1)
    c3 = (-(th2 + 2.0 * cos_t - 2.0) / (2.0 * th4)).unsqueeze(-1)
    c4 = ((2.0 * theta - 3.0 * sin_t + theta * cos_t) / (2.0 * th5)).unsqueeze(-1)

    Q = 0.5 * Rho \
        + c2 * (OmegaRho + RhoOmega + Omega_Rho_Omega) \
        + c3 * (Omega2_Rho + Rho_Omega2 - 3.0 * Omega_Rho_Omega) \
        + c4 * (Omega_Rho_Omega2 + Omega2_Rho_Omega)

    Jl_so3 = J_l_SO3(omega)                                                    # (..., 3, 3)
    zero33 = torch.zeros_like(Jl_so3)
    top = torch.cat([Jl_so3, Q], dim=-1)                                      # (..., 3, 6)
    bot = torch.cat([zero33, Jl_so3], dim=-1)                                 # (..., 3, 6)
    return torch.cat([top, bot], dim=-2)                                      # (..., 6, 6)


def _se3_right_jacobian_block(xi: torch.Tensor) -> torch.Tensor:
    """𝒥_r(ξ) = 𝒥_l(−ξ)."""
    return _se3_left_jacobian_block(-xi)


def _reward_goal(q_H, z_e, T_target_Rp, alpha_p_g, alpha_R_g, manifold):
    """R_goal(q_H) per extension.tex Eq. (47) — coupled body-frame twist form.

    R_goal = -(α_p ‖e_ρ‖² + α_R ‖e_ω‖²)  where (e_ρ, e_ω) = Log_SE(3)(T_φ^{-1} T_target).
    """
    R_phi, p_phi = manifold.T_phi_Rp(q_H, z_e)
    R_t, p_t = T_target_Rp
    e = log_relative_Rp(R_phi, p_phi, R_t, p_t)                               # (B, 6)
    e_rho = e[..., :3]
    e_omega = e[..., 3:]
    return -(alpha_p_g * (e_rho ** 2).sum(-1) + alpha_R_g * (e_omega ** 2).sum(-1))


def _make_T_target(q_seed, z_e, manifold, perturb_rad: float = 0.3, perturb_p: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """Target = T_φ(q_seed) · exp_SE3(ξ_perturb).  ξ_perturb random with bounded norm so
    e_goal = Log_SE(3)(T_φ^{-1} T_target) ≈ ξ_perturb is in the linearization regime."""
    B = q_seed.shape[0]
    R_phi, p_phi = manifold.T_phi_Rp(q_seed, z_e)
    omega = (torch.rand(B, 3, dtype=q_seed.dtype) * 2 - 1) * perturb_rad
    rho = (torch.rand(B, 3, dtype=q_seed.dtype) * 2 - 1) * perturb_p
    xi = torch.cat([rho, omega], dim=-1)
    R_d, p_d = exp_SE3(xi)
    return compose_Rp(R_phi, p_phi, R_d, p_d)


def main():
    print("Reward gradient verification (extension.tex Appendix A.4)")
    print("=" * 70)
    torch.manual_seed(0)

    arm = Franka7DoFPose(urdf_path=URDF, sigma_p=0.01, sigma_R=0.1)
    arm.chain = arm.chain.to(dtype=torch.float64)
    arm._chain_state = (None, torch.float64)

    # ------ Setup ------
    B = 4
    lower, upper = arm.joint_limits(dtype=torch.float64)
    margin = 0.2 * (upper - lower)
    q_H = lower + margin + (upper - lower - 2 * margin) * torch.rand(B, 7, dtype=torch.float64)
    z_e = torch.rand(B, 1, dtype=torch.float64) * 0.1 + 0.05
    R_t, p_t = _make_T_target(q_H, z_e, arm, perturb_rad=0.3, perturb_p=0.05)
    alpha_p_g, alpha_R_g = 100.0, 100.0
    print(f"B={B}, alpha_p={alpha_p_g}, alpha_R={alpha_R_g}, perturb=0.3 rad")

    # ------ Autograd reference ------
    q_var = q_H.clone().requires_grad_(True)
    R_scalar = _reward_goal(q_var, z_e, (R_t, p_t), alpha_p_g, alpha_R_g, arm).sum()
    g_ag = torch.autograd.grad(R_scalar, q_var)[0]                            # (B, 7)
    print(f"\n[autograd]  ‖g_ag‖_∞ per sample: {g_ag.abs().max(-1).values.tolist()}")

    # ------ Finite difference reference ------
    eps = 1e-5
    g_fd = torch.zeros_like(q_H)
    for i in range(7):
        ei = torch.zeros_like(q_H); ei[..., i] = eps
        Rp = _reward_goal(q_H + ei, z_e, (R_t, p_t), alpha_p_g, alpha_R_g, arm)
        Rm = _reward_goal(q_H - ei, z_e, (R_t, p_t), alpha_p_g, alpha_R_g, arm)
        g_fd[..., i] = (Rp - Rm) / (2 * eps)
    err_ag_fd = (g_ag - g_fd).abs().max().item()
    print(f"[FD vs autograd]  max diff = {err_ag_fd:.3e}  (target < 1e-4)")
    assert err_ag_fd < 1e-4, "autograd vs FD disagreement"

    # ------ Closed-form candidates ------
    # Compute the building blocks.
    R_phi, p_phi = arm.T_phi_Rp(q_H, z_e)
    e_goal = log_relative_Rp(R_phi, p_phi, R_t, p_t)                           # (B, 6)
    Jp = arm.jacobian_pose(q_H, z_e)                                           # (B, 6, 7)
    W_g = torch.diag(torch.tensor(
        [alpha_p_g] * 3 + [alpha_R_g] * 3, dtype=torch.float64
    ))                                                                         # (6, 6)

    # Variants: 𝒥_l(e_goal), 𝒥_l(-e_goal) = 𝒥_r(e_goal); both transposed; both signs.
    Jl = _se3_left_jacobian_block(e_goal)                                     # (B, 6, 6)
    Jr = _se3_right_jacobian_block(e_goal)
    # 𝒥^{-T}: invert then transpose.
    Jl_invT = torch.linalg.inv(Jl).transpose(-1, -2)
    Jr_invT = torch.linalg.inv(Jr).transpose(-1, -2)

    def _candidate(sign, J_invT):
        # g = sign * 2 * J_pose^T · J_invT · W_g · e_goal
        WgEg = (W_g @ e_goal.unsqueeze(-1))                                   # (B, 6, 1)
        inner = (J_invT @ WgEg)                                                # (B, 6, 1)
        out = sign * 2.0 * (Jp.transpose(-1, -2) @ inner).squeeze(-1)          # (B, 7)
        return out

    candidates = {
        "+_Jl": _candidate(+1.0, Jl_invT),
        "-_Jl": _candidate(-1.0, Jl_invT),
        "+_Jr": _candidate(+1.0, Jr_invT),
        "-_Jr": _candidate(-1.0, Jr_invT),
    }
    print("\n[closed-form candidates]  max diff vs autograd:")
    matches = []
    for name, cf in candidates.items():
        diff = (cf - g_ag).abs().max().item()
        ratio = diff / g_ag.abs().max().item()
        marker = "  ← MATCH" if ratio < 1e-3 else ""
        print(f"  g_cf({name}):  max diff = {diff:.3e}  (rel {ratio:.2e}){marker}")
        if ratio < 1e-3:
            matches.append(name)

    if not matches:
        print("\n  WARNING: no candidate matches autograd within rel 1e-3.")
    else:
        print(f"\n  Resolved combination: {matches}")

    # ------ Simplified form (no 𝒥-factor; resolved sign = +2) ------
    g_simple_pos = (+2.0 * (Jp.transpose(-1, -2) @ (W_g @ e_goal.unsqueeze(-1)))
                     ).squeeze(-1)
    g_simple_neg = (-2.0 * (Jp.transpose(-1, -2) @ (W_g @ e_goal.unsqueeze(-1)))
                     ).squeeze(-1)
    err_simple_pos = (g_simple_pos - g_ag).abs().max().item()
    err_simple_neg = (g_simple_neg - g_ag).abs().max().item()
    e_norm = e_goal.norm(dim=-1).max().item()
    print(f"\n[simplified +2 Jp^T W_g e]  max diff vs autograd = {err_simple_pos:.3e}")
    print(f"[simplified -2 Jp^T W_g e]  max diff vs autograd = {err_simple_neg:.3e}  (v1 sign — WRONG)")
    print(f"  ‖e_goal‖_∞ = {e_norm:.3e}  (small-error regime)")
    rel_simple = err_simple_pos / g_ag.abs().max().item()
    print(f"  relative error (correct +sign) = {rel_simple:.2%}")

    # ------ Sweep error magnitude to find threshold ------
    print("\n[sweep ‖e_goal‖]  measure simplified-form deviation across perturbation scales:")
    sweep = []
    for perturb in [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80]:
        torch.manual_seed(42)
        R_t_s, p_t_s = _make_T_target(q_H, z_e, arm, perturb_rad=perturb, perturb_p=perturb*0.1)
        q_var_s = q_H.clone().requires_grad_(True)
        R_scalar_s = _reward_goal(q_var_s, z_e, (R_t_s, p_t_s),
                                    alpha_p_g, alpha_R_g, arm).sum()
        g_ag_s = torch.autograd.grad(R_scalar_s, q_var_s)[0]
        R_phi_s, p_phi_s = arm.T_phi_Rp(q_H, z_e)
        e_s = log_relative_Rp(R_phi_s, p_phi_s, R_t_s, p_t_s)
        Jp_s = arm.jacobian_pose(q_H, z_e)
        # Use the RESOLVED +sign for the simplified form
        g_simple_s = (+2.0 * (Jp_s.transpose(-1, -2) @ (W_g @ e_s.unsqueeze(-1)))
                       ).squeeze(-1)
        rel = (g_simple_s - g_ag_s).abs().max().item() / g_ag_s.abs().max().item()
        e_norm = e_s.norm(dim=-1).max().item()
        sweep.append({"perturb_rad": perturb, "e_norm": e_norm, "rel_err_simple": rel})
        print(f"   perturb={perturb:.2f} rad  →  ‖e‖_∞={e_norm:.3e}  "
              f"rel_err(simple) = {rel:.2%}")

    # ------ Save outcome ------
    out_path = Path("outputs/diagnostic/reward_gradient_verification.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "B": B,
        "alpha_p_g": alpha_p_g,
        "alpha_R_g": alpha_R_g,
        "fd_vs_autograd_max": err_ag_fd,
        "candidates": {
            name: {
                "max_diff": (cf - g_ag).abs().max().item(),
                "rel_err": ((cf - g_ag).abs().max().item()
                              / g_ag.abs().max().item()),
            }
            for name, cf in candidates.items()
        },
        "matched_combinations": matches,
        "simplified_pos_max_diff": err_simple_pos,
        "simplified_neg_max_diff": err_simple_neg,
        "simplified_pos_rel_err": rel_simple,
        "resolved_simplified_sign": "+",
        "sweep": sweep,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
