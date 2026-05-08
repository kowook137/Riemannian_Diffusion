"""Score network for Riemannian SGM.

Output is a tangent vector at x, i.e. a section of TM.  We follow the
chart-output + manifold-lift parametrisation used in RSGM's
CanonicalGenerator / ParallelTransportGenerator: the MLP outputs a
chart-coordinate vector s_q ∈ R^{intrinsic_dim}, and the manifold's
lift_chart_to_tangent (= J_H(x) for a graph manifold) maps it into T_xM ⊂ R^d.
By construction this guarantees the score lies in the tangent bundle — no
projection step needed.

For Sphere1 the lift is a ↦ a · perp(x); for graph manifolds it will be
a ↦ (a, J_F(q) a).  The same network code works for both.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from smcdp.manifolds import Manifold


def sinusoidal_time_embedding(t: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
    """Standard transformer-style sinusoidal embedding of scalar timesteps."""
    half = dim // 2
    device, dtype = t.device, t.dtype
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=device, dtype=dtype) / max(half - 1, 1)
    )
    args = t.unsqueeze(-1) * freqs                                    # (..., half)
    emb = torch.cat([args.sin(), args.cos()], dim=-1)                 # (..., 2*half)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class _SinAct(nn.Module):
    """Pointwise sin activation, as used by RSGM's `Concat` MLP (act='sin')."""
    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)


_ACTIVATIONS = {
    "silu": nn.SiLU,
    "softplus": nn.Softplus,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sin": _SinAct,
}


class ChartScoreNet(nn.Module):
    """MLP score network with chart-coord output, lifted to ambient T_xM by the manifold.

    Mirrors RSGM's CanonicalGenerator + Concat-MLP pattern: the network ingests
    [x, t] (ambient point concatenated with raw scalar time, OR a sinusoidal time
    embedding), produces a chart-coordinate score s_q ∈ R^{intrinsic_dim}, and
    delegates the lift to ambient T_xM to the manifold's lift_chart_to_tangent.

    Defaults match RSGM's `architecture/concat.yaml`:
        hidden=512, n_layers=5, activation='sin', time_embedding='raw'.
    """

    def __init__(
        self,
        manifold: Manifold,
        hidden: int = 512,
        n_layers: int = 5,
        t_embed_dim: int = 64,
        activation: str = "sin",
        time_embedding: str = "raw",         # 'raw' | 'sinusoidal'
        final_init_scale: float = 1.0,        # 1.0 = Haiku/PyTorch default; <1 dampens initial score
    ):
        super().__init__()
        self.manifold = manifold
        self.time_embedding = time_embedding
        self.t_embed_dim = t_embed_dim

        if time_embedding == "raw":
            in_dim = manifold.ambient_dim + 1
        elif time_embedding == "sinusoidal":
            in_dim = manifold.ambient_dim + t_embed_dim
        else:
            raise ValueError(f"unknown time_embedding '{time_embedding}'")
        out_dim = manifold.intrinsic_dim

        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation '{activation}'")
        act_layer = _ACTIVATIONS[activation]

        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden))
            layers.append(act_layer())
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

        if final_init_scale != 1.0:
            with torch.no_grad():
                self.net[-1].weight.mul_(final_init_scale)
                self.net[-1].bias.zero_()

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if self.time_embedding == "raw":
            te = t.unsqueeze(-1)                                         # (B, 1)
        else:
            te = sinusoidal_time_embedding(t, self.t_embed_dim)          # (B, t_embed_dim)
        h = torch.cat([x, te], dim=-1)
        s_chart = self.net(h)                                            # (B, intrinsic_dim)
        s_ambient = self.manifold.lift_chart_to_tangent(x, s_chart)      # (B, ambient_dim)
        return s_ambient


class ScaledScoreFn(nn.Module):
    """RSGM-style score-fn wrapper (`get_score_fn` with std_trick + residual_trick).

    Two compositional tricks applied on top of the raw network output:

      std_trick:      score = net(x, t) / σ(t)
        Network is trained to output σ(t)·∇_M log p_t (well-bounded across t).
        Evaluation undoes the scaling so the reverse SDE sees the true score.

      residual_trick: score += 2·b_fwd(x, t) / β(t)
        Ensures `net = 0` at initialization makes the reverse SDE preserve the
        SDE's limiting distribution.  Critical when the forward drift is non-zero
        (Langevin case); a no-op for Brownian (b_fwd = 0).

        Derivation: with score = net/σ + 2·b_fwd/β, the reverse drift becomes
            b_rev = -b_fwd + β · score
                  = -b_fwd + β · (net/σ + 2 b_fwd/β)
                  = -b_fwd + β · net/σ + 2 b_fwd
                  = b_fwd + β · net/σ
        At init (net=0): b_rev = b_fwd, so reverse SDE = forward SDE in distribution
        ⇒ the limiting distribution is fixed.
    """

    def __init__(self, net: ChartScoreNet, sde,
                 std_trick: bool = True, residual_trick: bool = True):
        super().__init__()
        self.net = net
        self.sde = sde
        self.std_trick = std_trick
        self.residual_trick = residual_trick

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        out = self.net(x, t)
        if self.std_trick:
            sigma = self.sde.schedule.proxy_std(t).clamp(min=1e-6)
            out = out / sigma.unsqueeze(-1)
        if self.residual_trick:
            b_fwd = self.sde.drift(x, t)                                  # (B, d) ∈ T_xM
            beta = self.sde.schedule.beta(t).clamp(min=1e-12)
            out = out + 2.0 * b_fwd / beta.unsqueeze(-1)
        return out
