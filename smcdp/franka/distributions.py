"""Franka-specific limiting / data distributions.

Why a Franka-specific subclass of WrappedNormalEmbodiment?

The parent class's `_grad_log_det_G_chart` uses `vmap(jacrev(...))` to compute
∂_q (½ log det G(q, z_e)).  For planar arms this works because `jacobian_F`
is a closed-form torch expression (cumsum + sin/cos), but for Franka7DoF the
closed-form `jacobian_F` calls `pytorch_kinematics.SerialChain.jacobian`,
whose internals contain in-place ops that are incompatible with vmap.

Plain `torch.autograd.grad` (no vmap) DOES work through `chain.jacobian` —
the in-place issue is specific to vmap's batched-tensor abstraction, not
to autograd itself.  We therefore compute the SAME mathematical quantity
∂_q (½ log det G(q, z_e)) via plain autograd over the batch sum.  This is
exactly equivalent (no approximation) to the parent vmap+jacrev path —
each sample's ½ log det G(q_i, z_i) only depends on its own q_i, so summing
and back-propagating gives the per-sample gradient correctly.
"""
from __future__ import annotations

import torch
from torch import Tensor

from smcdp.toy3.distributions import WrappedNormalEmbodiment


class WrappedNormalFranka7DoF(WrappedNormalEmbodiment):
    """WrappedNormal limiting on Franka7DoF — full Riemannian potential.

        U(q, z_e)  =  ½ ‖q − μ_q‖² / γ²  +  ½ log det G(q, z_e)

    grad_U is the full Riemannian gradient identical to the parent class;
    only the implementation of `_grad_log_det_G_chart` is overridden to use
    plain autograd instead of vmap+jacrev (parent's default).  Result is
    mathematically identical — see module-level docstring for justification.
    """

    def _grad_log_det_G_chart(self, q: Tensor, z: Tensor) -> Tensor:
        """∂_q (½ log det G(q, z_e))  via plain autograd (no vmap).

        Each sample's logdet depends only on its own q row, so
        ∂(Σ_i logdet_i)/∂q_j = ∂logdet_j/∂q_j — i.e., autograd over the sum
        produces the per-sample gradient.  No vmap / no approximation.

        We wrap the computation in `torch.enable_grad()` because this method is
        often called inside a `@torch.no_grad()` context (e.g. forward GRW),
        which globally disables autograd otherwise.  The returned tensor is
        detached from the outer graph (create_graph=False).
        """
        with torch.enable_grad():
            q_leaf = q.detach().clone().requires_grad_(True)
            G = self.manifold.G(q_leaf, z)                          # (B, n_q, n_q)
            half_logdet = 0.5 * torch.linalg.slogdet(G).logabsdet   # (B,)
            grad = torch.autograd.grad(
                half_logdet.sum(), q_leaf,
                create_graph=False, retain_graph=False,
            )[0]
        return grad.detach()
