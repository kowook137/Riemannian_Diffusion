"""Stage-1 learned residual self-model on Franka 7-DoF (Idea_formulation §7).

Pattern mirrors smcdp/toy3/self_model.py:
    F(q, z_e)  =  FK_analytic(q, z_e)  +  Δ_φ(q, z_e)
    M_φ(z_e)   =  {(q, p) : p = F(q, z_e)}    (graph manifold)

Stage 1 trains Δ_φ to minimise  ‖p_true − F(q, z_e)‖²  on a chart-uniform
joint-config dataset; Stage 5+ freezes Δ_φ and uses this manifold for the
score-net.

`jacobian_F` is OVERRIDDEN here: the parent's autograd default would
differentiate through pytorch_kinematics' `chain.jacobian` under vmap, which
is incompatible.  We instead reuse Franka7DoF's closed-form analytic Jacobian
and add the residual ∂Δ_φ/∂q via vmap+jacrev on the MLP only (pure torch,
vmap-OK).  This is REQUIRED for the by-construction tangent property of the
SMCDP framework: lift_chart_to_tangent uses J_F to lift chart scores into
ambient T_xM, and J_g · v = 0 holds only when J_F includes ∂Δ_φ/∂q.
"""
from __future__ import annotations

import torch
from torch import Tensor

from smcdp.manifolds import Franka7DoF
from smcdp.toy3.self_model import DeltaResidualMLP


class LearnedSelfModelFranka7DoF(Franka7DoF):
    """Franka7DoF graph manifold with frozen learned residual Δ_φ.

        F(q, z_e) = analytic_FK(q, z_e) + Δ_φ(q, z_e)

    Δ_φ is a small MLP (default 3 layers × 128, Softplus activation,
    small final-layer init) shared across the Stage-1 dataset.  After Stage-1
    fitting it is frozen; Stage 5+ trains the score net on this learned
    manifold.

    Idea_formulation §7.3: this is the "learned self-model manifold" used
    throughout the killer experiment.
    """

    def __init__(
        self,
        delta_net: DeltaResidualMLP,
        urdf_path: str,
        end_link: str = "panda_hand",
        tool_z_max: float = 0.20,
        metric: str = "riemannian",
        joint_limit_margin_frac: float = 0.10,
    ):
        super().__init__(
            urdf_path=urdf_path, end_link=end_link, tool_z_max=tool_z_max,
            metric=metric, joint_limit_margin_frac=joint_limit_margin_frac,
        )
        self.delta_net = delta_net
        for p in self.delta_net.parameters():
            p.requires_grad_(False)

    def fk_analytic(self, q: Tensor, z_e: Tensor) -> Tensor:
        return Franka7DoF.F(self, q, z_e)

    def F(self, q: Tensor, z_e: Tensor) -> Tensor:
        return self.fk_analytic(q, z_e) + self.delta_net(q, z_e)

    def jacobian_F(self, q: Tensor, z_e: Tensor) -> Tensor:
        """∂F/∂q  =  ∂F_analytic/∂q  +  ∂Δ_φ/∂q.

        Analytic part: closed form via pytorch_kinematics body Jacobian + tool
                        offset cross term (Franka7DoF.jacobian_F).
        Residual part: vmap+jacrev over the MLP only — Δ_φ is pure torch ops,
                        so vmap is safe.  Combining the two avoids a vmap+jacrev
                        through pytorch_kinematics (which fails due to in-place
                        ops in chain.jacobian).
        """
        Jf_analytic = Franka7DoF.jacobian_F(self, q, z_e)
        def delta_at(q_s: Tensor, z_s: Tensor) -> Tensor:
            return self.delta_net(q_s.unsqueeze(0), z_s.unsqueeze(0))[0]
        Jf_residual = torch.func.vmap(torch.func.jacrev(delta_at, argnums=0))(q, z_e)
        return Jf_analytic + Jf_residual
