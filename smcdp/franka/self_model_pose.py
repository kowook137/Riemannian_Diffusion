"""Stage-1 pose self-model on Franka 7-DoF (extension.tex Sec. 1).

Mirrors `smcdp/franka/self_model.py` but learns a se(3) residual twist
ξ_φ(q, z_e) = (ρ_φ, ω_φ) ∈ R^6 instead of a world-frame Δ_φ ∈ R^3:

    T_φ(q, z_e) = T_analytic(q, z_e) · exp_SE(3)(ξ_φ(q, z_e)^∧)

Stage-1 trains ξ_φ to minimise the pose error twist
    e_self(q, T_true, z_e) = Log_SE(3)(T_φ(q, z_e)^{-1} · T_true) = (e_ρ, e_ω)
weighted as  w_p ‖e_ρ‖² + w_R ‖e_ω‖²,  with separate Frobenius regularizers
on  ∂_q ρ_φ, ∂_q ω_φ.

Position-only fallback: setting ω_φ ≡ 0 (e.g. via `output_omega=False`)
recovers translation-only behaviour with ρ_φ as a body-frame translation
residual; the world-frame Δ_φ used in the position-only framework relates
via Δ_φ(q, z_e) = R_analytic(q, z_e) · ρ_φ(q, z_e).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from smcdp.lie_se3 import (
    exp_SE3, log_SE3,
    compose_Rp, inverse_Rp, log_relative_Rp,
    adjoint_inverse_Rp,
)
from smcdp.manifolds_pose import Franka7DoFPose, EmbodimentPoseGraphManifold


class PoseResidualMLP(nn.Module):
    """Residual MLP ξ_φ : (q, z_e) ↦ R^6 ≅ se(3).

    Architecture mirrors `DeltaResidualMLP` (Stage-1 position-only): N hidden
    layers × hidden width, Softplus activations (smooth → autograd-friendly),
    final layer initialised small so ξ_φ ≈ 0 at the start of training (the
    worst-case fallback is the analytic T_analytic).
    """

    def __init__(
        self,
        n_q: int = 7,
        n_z: int = 1,
        hidden: int = 128,
        n_layers: int = 3,
        activation: type = nn.Softplus,
        final_init_scale: float = 1e-3,
        output_omega: bool = True,
    ):
        super().__init__()
        self.output_omega = bool(output_omega)
        self.out_dim = 6 if self.output_omega else 3
        in_dim = n_q + n_z
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden))
            layers.append(activation())
            d = hidden
        last = nn.Linear(d, self.out_dim)
        with torch.no_grad():
            last.weight.mul_(final_init_scale)
            last.bias.zero_()
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, q: Tensor, z_e: Tensor) -> Tensor:
        h = torch.cat([q, z_e], dim=-1)
        out = self.net(h)
        if self.output_omega:
            return out                                                   # (..., 6)
        zeros = torch.zeros_like(out)                                    # (..., 3)
        return torch.cat([out, zeros], dim=-1)                           # (..., 6)


def pose_self_model_loss(
    residual_net: PoseResidualMLP,
    q: Tensor,
    z_e: Tensor,
    T_true_Rp: tuple[Tensor, Tensor],
    fk_analytic_Rp,                                                      # callable (q, z) -> (R, p)
    w_p: float = 1.0,
    w_R: float = 1.0,
    beta_p: float = 1e-3,
    beta_R: float = 1e-3,
) -> tuple[Tensor, dict]:
    """Stage-1 pose self-model loss (extension.tex Eq. (5)).

    Inputs:
        q          (B, n_q)       chart sample
        z_e        (B, n_z)       embodiment context
        T_true_Rp  ((B,3,3),(B,3))  ground-truth pose
        fk_analytic_Rp  callable (q, z) → (R_analytic, p_analytic)

    Loss:
        L = w_p ‖e_ρ‖² + w_R ‖e_ω‖²
            + β_p ‖∂_q ρ_φ‖_F²  +  β_R ‖∂_q ω_φ‖_F²
    where  (e_ρ, e_ω) = Log_SE(3)( T_φ^{-1} · T_true ).

    All operations are vmap-safe ((R, p) form, no quaternion roundtrip).
    """
    R_t, p_t = T_true_Rp
    # Build T_φ in (R, p) form: T_φ = T_analytic · exp_SE(3)(ξ_φ)
    R_a, p_a = fk_analytic_Rp(q, z_e)                                    # (B, 3, 3), (B, 3)
    xi = residual_net(q, z_e)                                            # (B, 6)
    R_d, p_d = exp_SE3(xi)                                               # (B, 3, 3), (B, 3)
    R_phi, p_phi = compose_Rp(R_a, p_a, R_d, p_d)

    # Body-frame error twist
    e = log_relative_Rp(R_phi, p_phi, R_t, p_t)                          # (B, 6)
    e_rho = e[..., :3]
    e_omega = e[..., 3:]

    fit_p = (e_rho ** 2).sum(-1).mean()
    fit_R = (e_omega ** 2).sum(-1).mean()
    loss = w_p * fit_p + w_R * fit_R

    info = {"fit_p": fit_p.item(), "fit_R": fit_R.item()}

    if beta_p > 0.0 or beta_R > 0.0:
        # Smoothness regularizers on ∂_q ξ_φ (split into ∂_q ρ_φ, ∂_q ω_φ).
        def _xi_pair(q_s: Tensor, z_s: Tensor) -> Tensor:
            return residual_net(q_s.unsqueeze(0), z_s.unsqueeze(0))[0]
        Jq = torch.func.vmap(torch.func.jacrev(_xi_pair, argnums=0))(q, z_e)  # (B, 6, n_q)
        # Split into ρ-block and ω-block.
        Jq_rho = Jq[..., :3, :]
        Jq_omega = Jq[..., 3:, :]
        smooth_p = (Jq_rho ** 2).sum(dim=(-1, -2)).mean()
        smooth_R = (Jq_omega ** 2).sum(dim=(-1, -2)).mean()
        loss = loss + beta_p * smooth_p + beta_R * smooth_R
        info["smooth_p"] = smooth_p.item()
        info["smooth_R"] = smooth_R.item()

    info["loss"] = loss.item()
    return loss, info


class LearnedSelfModelFranka7DoFPose(Franka7DoFPose):
    """Franka 7-DoF pose self-model with frozen learned se(3) residual ξ_φ.

        T_φ(q, z_e) = T_analytic(q, z_e) · exp_SE(3)(ξ_φ(q, z_e)^∧)

    `T_phi_Rp` is overridden to compose the analytic FK with the learned
    residual.  `jacobian_pose` falls back to autograd through `T_phi_Rp` (parent
    `EmbodimentPoseGraphManifold` default), since the closed-form analytic
    Jacobian no longer applies once a learned residual is added.
    """

    def __init__(
        self,
        residual_net: PoseResidualMLP,
        urdf_path: str,
        end_link: str = "panda_hand",
        tool_z_max: float = 0.20,
        sigma_p: float = 0.01,
        sigma_R: float = 0.1,
        metric: str = "riemannian",
        joint_limit_margin_frac: float = 0.10,
        link_perturb_dl3: float = 0.0,
        link_perturb_dl5: float = 0.0,
    ):
        super().__init__(
            urdf_path=urdf_path, end_link=end_link, tool_z_max=tool_z_max,
            sigma_p=sigma_p, sigma_R=sigma_R, metric=metric,
            joint_limit_margin_frac=joint_limit_margin_frac,
            link_perturb_dl3=link_perturb_dl3,
            link_perturb_dl5=link_perturb_dl5,
        )
        self.residual_net = residual_net
        for p in self.residual_net.parameters():
            p.requires_grad_(False)

    def fk_analytic_Rp(self, q: Tensor, z_e: Tensor) -> tuple[Tensor, Tensor]:
        return Franka7DoFPose.T_phi_Rp(self, q, z_e)

    def T_phi_Rp(self, q: Tensor, z_e: Tensor) -> tuple[Tensor, Tensor]:
        R_a, p_a = self.fk_analytic_Rp(q, z_e)
        xi = self.residual_net(q, z_e)                                   # (..., 6)
        R_d, p_d = exp_SE3(xi)
        return compose_Rp(R_a, p_a, R_d, p_d)

    def jacobian_pose(self, q: Tensor, z_e: Tensor) -> Tensor:
        """Body-frame J_pose for the learned self-model.

        Decompose T_φ = T_a · T_d (analytic · residual) and apply the body-frame
        velocity-twist composition rule:

            ξ_φ^body = Ad_{T_d^{-1}} · ξ_a^body  +  ξ_d^body
            J_pose^body(q, z) = Ad_{T_d^{-1}}(q, z) · J_a^body(q, z) + J_d^body(q, z)

        with:
          - J_a^body   = closed-form Franka body Jacobian (Franka7DoFPose).
          - J_d^body   = body-frame Jacobian of the *residual* map
                          T_d_Rp(q, z) := exp_SE(3)(ξ_φ(q, z))  via autograd
                          (the residual is pure-torch / vmap-safe).
          - Ad_{T_d^{-1}}: SE(3) adjoint of T_d^{-1}, closed form from (R_d, p_d).

        The hybrid is REQUIRED: a direct autograd over the full T_phi_Rp would
        try to vmap-differentiate through pytorch_kinematics, which uses
        in-place ops and fails under vmap.
        """
        # --- analytic part: closed-form J_a^body ---
        J_a = Franka7DoFPose.jacobian_pose(self, q, z_e)                 # (B, 6, 7)

        # --- residual part: body-frame Jacobian of T_d via autograd ---
        residual_net = self.residual_net

        def T_d_Rp(q_var: Tensor, z_var: Tensor) -> tuple[Tensor, Tensor]:
            xi = residual_net(q_var, z_var)                              # (..., 6)
            return exp_SE3(xi)                                            # (..., 3, 3), (..., 3)

        q_base = q.detach()
        Rd_base, pd_base = T_d_Rp(q_base, z_e)
        Rd_inv, pd_inv = inverse_Rp(Rd_base, pd_base)
        Rd_inv = Rd_inv.detach()
        pd_inv = pd_inv.detach()

        def xi_per_sample(q_var_s: Tensor, z_s: Tensor,
                          R_inv_s: Tensor, p_inv_s: Tensor) -> Tensor:
            R_var, p_var = T_d_Rp(q_var_s.unsqueeze(0), z_s.unsqueeze(0))
            R_var = R_var.squeeze(0)
            p_var = p_var.squeeze(0)
            R_rel = R_inv_s @ R_var
            p_rel = (R_inv_s @ p_var.unsqueeze(-1)).squeeze(-1) + p_inv_s
            return log_SE3(R_rel, p_rel)

        J_d = torch.func.vmap(
            torch.func.jacrev(xi_per_sample, argnums=0),
            in_dims=(0, 0, 0, 0),
        )(q, z_e, Rd_inv, pd_inv)                                        # (B, 6, 7)

        # --- frame correction: Ad_{T_d^{-1}} ---
        # Recompute (R_d, p_d) (graph-attached) for the adjoint factor.
        xi = residual_net(q, z_e)
        R_d, p_d = exp_SE3(xi)
        Ad_inv = adjoint_inverse_Rp(R_d, p_d)                            # (B, 6, 6)

        return Ad_inv @ J_a + J_d                                        # (B, 6, 7)
