"""Chart parameterizations for joint-bounded manifolds (joint_limit_extension v4.1).

A `Chart` maps an unbounded computational coordinate `u ∈ R^n_q` to the physical
joint configuration `q = ψ(u) ∈ (q_min, q_max)` via a smooth diffeomorphism.

Default implementation: `TanhBoundedChart` with
    ψ_i(u_i) = q_mid_i + (q_range_i / 2) · tanh(u_i)
    ψ⁻¹(q)_i = atanh( 2(q_i - q_mid_i) / q_range_i )            (with η-clip safety)
    D_ψ(u)_i = (q_range_i / 2) · sech²(u_i)                       (diagonal Jacobian)

The chart is decoupled from the manifold class so that v4 (unbounded chart, identity
ψ) and v4.1 (bounded chart) share the same downstream code paths via dependency
injection. Setting `IdentityChart` recovers v4 behavior exactly.
"""
from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor


class Chart(Protocol):
    """Chart interface: a smooth diffeomorphism u → q with a known diagonal Jacobian.

    All methods are vmap-safe and operate on arbitrary batch shapes; the last
    dimension is `n_q` (the joint dimension).
    """

    n_q: int

    def psi(self, u: Tensor) -> Tensor:
        """u → q. Shape preserved."""
        ...

    def psi_inv(self, q: Tensor, *, eps: float = 1e-3) -> Tensor:
        """q → u with η-clip safety (init-only, see v4.1 §10.3)."""
        ...

    def D_psi_diag(self, u: Tensor) -> Tensor:
        """Diagonal entries of ∂ψ/∂u. Shape (..., n_q). Always positive."""
        ...

    def joint_limits(self, *, device=None, dtype=None) -> tuple[Tensor, Tensor]:
        """(q_min, q_max). For IdentityChart these are ±∞ in spirit; we return
        large finite numbers as a sentinel."""
        ...


class TanhBoundedChart:
    """Bounded chart via element-wise tanh diffeomorphism (v4.1 default).

    ψ(u) = q_mid + (q_range / 2) tanh(u)  ∈ (q_min, q_max)

    Construct from per-joint limits (q_min, q_max). Stores tensors on CPU; lazily
    moves to the device of the input on each call.
    """

    def __init__(self, q_min: Tensor, q_max: Tensor):
        if q_min.shape != q_max.shape or q_min.dim() != 1:
            raise ValueError(
                f"q_min, q_max must be 1-D tensors of equal shape; "
                f"got {tuple(q_min.shape)} vs {tuple(q_max.shape)}"
            )
        if not torch.all(q_max > q_min):
            raise ValueError("Require q_max > q_min element-wise")
        self.n_q = int(q_min.shape[0])
        # Store as buffers (CPU); _to dispatches on call site
        self._q_min = q_min.detach().clone()
        self._q_max = q_max.detach().clone()
        self._q_mid = 0.5 * (self._q_min + self._q_max)
        self._q_range = self._q_max - self._q_min

    def _broadcast(self, u: Tensor) -> tuple[Tensor, Tensor]:
        """Move stored constants to u's device/dtype and broadcast to u's shape."""
        q_mid = self._q_mid.to(device=u.device, dtype=u.dtype)
        q_range = self._q_range.to(device=u.device, dtype=u.dtype)
        return q_mid, q_range

    def psi(self, u: Tensor) -> Tensor:
        if u.shape[-1] != self.n_q:
            raise ValueError(f"Expected last-dim {self.n_q}, got {u.shape[-1]}")
        q_mid, q_range = self._broadcast(u)
        return q_mid + 0.5 * q_range * torch.tanh(u)

    def psi_inv(self, q: Tensor, *, eps: float = 1e-3) -> Tensor:
        """Inverse map with η-clipping. Init-only safety per v4.1 §10.3.

        Computes η = 2(q - q_mid)/q_range, clips to [-1+eps, 1-eps], then atanh.
        Out-of-range q values silently get clipped (warning emitted in debug
        builds via assertion). For typical usage `q` should already be inside
        (q_min, q_max).
        """
        if q.shape[-1] != self.n_q:
            raise ValueError(f"Expected last-dim {self.n_q}, got {q.shape[-1]}")
        q_mid, q_range = self._broadcast(q)
        eta = 2.0 * (q - q_mid) / q_range
        eta = eta.clamp(min=-1.0 + eps, max=1.0 - eps)
        return torch.atanh(eta)

    def D_psi_diag(self, u: Tensor) -> Tensor:
        """Diagonal of ∂ψ/∂u: (q_range/2) sech²(u). Always > 0."""
        if u.shape[-1] != self.n_q:
            raise ValueError(f"Expected last-dim {self.n_q}, got {u.shape[-1]}")
        _, q_range = self._broadcast(u)
        # sech²(u) = 1 - tanh²(u) = 1 / cosh²(u). Use 1 - tanh² for numerical
        # stability across the range (tanh² ∈ [0, 1)).
        sech_sq = 1.0 - torch.tanh(u).pow(2)
        return 0.5 * q_range * sech_sq

    def joint_limits(self, *, device=None, dtype=None) -> tuple[Tensor, Tensor]:
        return (self._q_min.to(device=device, dtype=dtype),
                self._q_max.to(device=device, dtype=dtype))


class IdentityChart:
    """Identity chart ψ(u) = u, D_ψ = I. Recovers v4 behavior.

    `joint_limits` returns ±large_value as a sentinel; the unbounded chart has
    no actual limits in the chart formulation. Downstream code that consumes
    joint_limits should handle this case (or just use violates_limits from the
    underlying manifold, which uses the URDF limits).
    """

    def __init__(self, n_q: int, sentinel: float = 1e6):
        self.n_q = int(n_q)
        self._sentinel = float(sentinel)

    def psi(self, u: Tensor) -> Tensor:
        return u

    def psi_inv(self, q: Tensor, *, eps: float = 0.0) -> Tensor:
        return q

    def D_psi_diag(self, u: Tensor) -> Tensor:
        return torch.ones_like(u)

    def joint_limits(self, *, device=None, dtype=None) -> tuple[Tensor, Tensor]:
        return (
            torch.full((self.n_q,), -self._sentinel, device=device, dtype=dtype),
            torch.full((self.n_q,), +self._sentinel, device=device, dtype=dtype),
        )


def make_chart_from_manifold(
    manifold,
    *,
    bounded: bool,
    device=None,
    dtype=None,
) -> Chart:
    """Factory: build a chart whose joint limits match the manifold's URDF.

    bounded=False  →  IdentityChart (v4)
    bounded=True   →  TanhBoundedChart with limits from `manifold.joint_limits()`
    """
    n_q = int(manifold.n_q)
    if not bounded:
        return IdentityChart(n_q=n_q)
    if not hasattr(manifold, "joint_limits"):
        raise AttributeError(
            f"Bounded chart requires manifold with joint_limits(); "
            f"got {type(manifold).__name__}"
        )
    q_min, q_max = manifold.joint_limits(device=device, dtype=dtype)
    if q_min.shape[0] != n_q:
        raise ValueError(f"manifold.joint_limits returned shape {q_min.shape}, "
                          f"expected ({n_q},)")
    return TanhBoundedChart(q_min, q_max)
