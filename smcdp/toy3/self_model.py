"""Stage 1 — learned residual self-model and the manifold built on it.

Per Idea_formulation §7.1:
    g_φ(q, p_ee, z_e) = p_ee − FK_analytic(q, z_e) − Δ_φ(q, z_e)
    Δ_φ network: 3-layer MLP, hidden 128, Softplus, small init.

The learned manifold (Idea §7.3 Stage 2) is then
    M_φ(z_e) = {(q, p) : p = FK_analytic(q, z_e) + Δ_φ(q, z_e)}
which is a graph manifold with embodiment context z_e — i.e. an instance of
EmbodimentGraphManifold whose F is the analytic FK plus the frozen learned Δ_φ.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from smcdp.manifolds import EmbodimentGraphManifold, EmbodimentNLinkPlanarArm


class DeltaResidualMLP(nn.Module):
    """Residual MLP Δ_φ : (q, z_e) ↦ R^{n_p}.

    Architecture mirrors Idea §7.1 — 3 hidden layers, width 128, Softplus
    activation (smooth → autograd-friendly Jacobian + Hessian).  Final layer
    initialised small so Δ_φ ≈ 0 at the start of training (worst-case fall-back
    is the analytic FK).
    """

    def __init__(
        self,
        n_q: int = 2,
        n_p: int = 2,
        n_z: int = 1,
        hidden: int = 128,
        n_layers: int = 3,
        activation: type = nn.Softplus,
        final_init_scale: float = 1e-3,
    ):
        super().__init__()
        in_dim = n_q + n_z
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden))
            layers.append(activation())
            d = hidden
        last = nn.Linear(d, n_p)
        with torch.no_grad():
            last.weight.mul_(final_init_scale)
            last.bias.zero_()
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, q: Tensor, z_e: Tensor) -> Tensor:
        h = torch.cat([q, z_e], dim=-1)
        return self.net(h)


def self_model_loss(
    delta_net: DeltaResidualMLP,
    q: Tensor,
    z_e: Tensor,
    p_true: Tensor,
    fk_analytic_fn,                              # callable (q, z_e) -> p_analytic
    smoothness_weight: float = 1e-3,
) -> tuple[Tensor, dict]:
    """Stage-1 loss (§7.2):

        L_self = E ‖FK_analytic(q, z_e) + Δ_φ(q, z_e) − p_true‖²
                 +  β · E ‖∂_q Δ_φ‖²_F

    The Frobenius-norm regulariser keeps Δ_φ smooth in q; this stabilises the
    induced J_F = J_FK + ∂Δ_φ/∂q used to build G(q, z_e).
    """
    p_analytic = fk_analytic_fn(q, z_e)
    delta = delta_net(q, z_e)
    p_pred = p_analytic + delta
    fit = ((p_pred - p_true) ** 2).sum(-1).mean()

    # smoothness regulariser via vmap-jacrev over q (z_e fixed per sample)
    if smoothness_weight > 0.0:
        def _delta_pair(q_s: Tensor, z_s: Tensor) -> Tensor:
            return delta_net(q_s.unsqueeze(0), z_s.unsqueeze(0))[0]
        jac_q = torch.func.vmap(torch.func.jacrev(_delta_pair, argnums=0))(q, z_e)
        smooth = (jac_q ** 2).sum(dim=(-1, -2)).mean()                 # ‖J‖_F²
        loss = fit + smoothness_weight * smooth
        return loss, {"fit": fit.item(), "smooth": smooth.item()}
    return fit, {"fit": fit.item()}


class LearnedSelfModelArm(EmbodimentGraphManifold):
    """Graph manifold whose F is the analytic FK + learned residual Δ_φ.

        F(q, z_e) = FK_analytic(q, z_e) + Δ_φ(q, z_e)

    The Δ_φ MLP is treated as FROZEN here (Stage 1 is finished before the score
    net is trained).  We deliberately use autograd for jacobian_F instead of a
    closed form, because that is exactly the differentiable-self-model design
    Idea_formulation §2 prescribes (autograd over a learned smooth function).
    """

    def __init__(
        self,
        delta_net: DeltaResidualMLP,
        l1: float = 1.0,
        l2_base: float = 1.0,
        metric: str = "riemannian",
    ):
        super().__init__(n_q=2, n_p=2, n_z=1, metric=metric)
        self.l1 = float(l1)
        self.l2_base = float(l2_base)
        self.delta_net = delta_net
        # Stage-1 model is fixed during Stage 5; freeze parameters so the score
        # net's gradients don't leak into the self-model.
        for p in self.delta_net.parameters():
            p.requires_grad_(False)

    def fk_analytic(self, q: Tensor, z_e: Tensor) -> Tensor:
        q1 = q[..., 0]
        q2 = q[..., 1]
        l2_eff = self.l2_base + z_e[..., 0]
        c1 = torch.cos(q1)
        s1 = torch.sin(q1)
        c12 = torch.cos(q1 + q2)
        s12 = torch.sin(q1 + q2)
        return torch.stack([self.l1 * c1 + l2_eff * c12,
                            self.l1 * s1 + l2_eff * s12], dim=-1)

    def F(self, q: Tensor, z_e: Tensor) -> Tensor:
        return self.fk_analytic(q, z_e) + self.delta_net(q, z_e)


class LearnedSelfModelNLinkArm(EmbodimentNLinkPlanarArm):
    """N-link planar arm with learned residual Δ_φ — generalises LearnedSelfModelArm.

    Inherits the analytic FK + closed-form Jacobian of EmbodimentNLinkPlanarArm
    and overrides F to add the LEARNED residual:

        F(q, z_e) = analytic_FK_N(q, z_e) + Δ_φ(q, z_e)

    `jacobian_F` falls back to the autograd default (inherited from
    EmbodimentGraphManifold), which differentiates through both the analytic
    FK and the residual MLP — exactly the Idea §2.1 paradigm B
    (differentiable function + autograd-derived geometry).
    """

    def __init__(
        self,
        delta_net: DeltaResidualMLP,
        link_lengths_base,
        metric: str = "riemannian",
    ):
        super().__init__(link_lengths_base=link_lengths_base, metric=metric)
        self.delta_net = delta_net
        # Stage-1 self-model frozen during Stage-5 score training (Idea §7.3).
        for p in self.delta_net.parameters():
            p.requires_grad_(False)

    def fk_analytic(self, q: Tensor, z_e: Tensor) -> Tensor:
        # Analytic part of the inherited closed-form FK.
        return EmbodimentNLinkPlanarArm.F(self, q, z_e)

    def F(self, q: Tensor, z_e: Tensor) -> Tensor:
        return self.fk_analytic(q, z_e) + self.delta_net(q, z_e)

    def jacobian_F(self, q: Tensor, z_e: Tensor) -> Tensor:
        """∂F/∂q = ∂F_analytic/∂q + ∂Δ_φ/∂q.

        The analytic part has a closed form (inherited from
        EmbodimentNLinkPlanarArm) — fast, no autograd needed.  The residual
        part requires autograd through the learned MLP; we vmap+jacrev only
        over the residual to avoid recomputing the analytic-FK Jacobian
        through autograd.

        This override is REQUIRED for the by-construction tangent property
        of the SMCDP framework: the lift_chart_to_tangent uses J_F to map
        chart-coord scores into ambient tangent vectors.  Without including
        ∂Δ_φ/∂q, the lifted vectors fail J_g · v = 0 once Δ_φ has any
        non-zero gradient (i.e. after Stage-1 training), breaking Idea §4.4.
        """
        # Analytic part — closed form via parent class
        Jf_analytic = EmbodimentNLinkPlanarArm.jacobian_F(self, q, z_e)
        # Residual part — autograd through frozen Δ_φ
        def delta_at(q_s: Tensor, z_s: Tensor) -> Tensor:
            return self.delta_net(q_s.unsqueeze(0), z_s.unsqueeze(0))[0]
        Jf_residual = torch.func.vmap(torch.func.jacrev(delta_at, argnums=0))(q, z_e)
        return Jf_analytic + Jf_residual
