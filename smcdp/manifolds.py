"""Embedded Riemannian manifolds M ⊂ R^d.

The Manifold ABC provides the dispatch surface that lets SDE / sampler / loss
code be manifold-agnostic.  Concrete manifolds plug in closed-form (or
numerical) implementations of exp / log / projection / sampling.

Convention (matches Lee, "Introduction to Riemannian Manifolds"):
  - exp(x, v): exp_x(v),   v ∈ T_xM,    returns point on M
  - log(x, y): log_x(y),   x, y ∈ M,    returns tangent vector at x toward y

Toy 1 implements Sphere1; Toy 2/3 will add GraphManifold sharing this API.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Manifold(ABC):
    """Embedded Riemannian manifold M ⊂ R^{ambient_dim} of dimension intrinsic_dim."""

    ambient_dim: int
    intrinsic_dim: int

    @abstractmethod
    def exp(self, x: Tensor, v: Tensor) -> Tensor:
        """exp_x(v).  x: (..., d), v: (..., d) ∈ T_xM.  Returns (..., d) on M."""

    @abstractmethod
    def log(self, x: Tensor, y: Tensor) -> Tensor:
        """log_x(y) ∈ T_xM.  Tangent vector at x pointing toward y."""

    @abstractmethod
    def proj_to_tangent(self, x: Tensor, v: Tensor) -> Tensor:
        """Project ambient vector v ∈ R^d onto T_xM."""

    @abstractmethod
    def lift_chart_to_tangent(self, x: Tensor, a: Tensor) -> Tensor:
        """Lift chart-coordinate vector a ∈ R^{intrinsic_dim} to ambient T_xM via J_H(x).

        For graph manifolds, this is a ↦ (a, J_F(q) a).  For Sphere1, a ↦ a · perp(x).
        """

    @abstractmethod
    def random_normal_tangent(self, x: Tensor) -> Tensor:
        """Sample tangent Gaussian at x with chart covariance G(x)^{-1}, returned in
        ambient form so that ‖ξ‖²_{ambient} matches the Riemannian norm."""

    @abstractmethod
    def random_uniform(self, n: int, device=None, dtype=torch.float32) -> Tensor:
        """Sample n points uniformly on M (compact M only)."""

    @abstractmethod
    def squared_norm(self, x: Tensor, v: Tensor) -> Tensor:
        """Riemannian ‖v‖²_x for v ∈ T_xM."""

    @abstractmethod
    def belongs(self, x: Tensor, atol: float = 1e-5) -> Tensor:
        """Boolean tensor indicating which points satisfy the manifold constraint."""

    @abstractmethod
    def constraint(self, x: Tensor) -> Tensor:
        """g_φ(x) — vector-valued constraint, zero iff x ∈ M."""

    # Optional fast path for forward marginal sampling p_t(· | x_0).
    # Default raises so SDE code can dispatch via hasattr-check.
    has_analytic_marginal: bool = False

    def marginal_sample(self, x_0: Tensor, t: Tensor, schedule) -> Tensor:  # pragma: no cover
        raise NotImplementedError


class Sphere1(Manifold):
    """S^1 = {x ∈ R^2 : ‖x‖ = 1}, intrinsic dim 1.

    Chart: angle θ; embedding H(θ) = (cos θ, sin θ).
    Induced metric G(θ) = J_H^T J_H = sin²θ + cos²θ = 1, so chart-Eucl ≡ chart-G ≡ ambient.
    Parallelizable (single global frame E(θ) = (-sin θ, cos θ) = perp(x)).
    Compact ⇒ Brownian on S¹ targets uniform.
    """

    ambient_dim = 2
    intrinsic_dim = 1
    has_analytic_marginal = True

    @staticmethod
    def _perp(x: Tensor) -> Tensor:
        # Rotate by +90°: (x_0, x_1) ↦ (-x_1, x_0).  Lies in T_xS^1 by construction.
        return torch.stack([-x[..., 1], x[..., 0]], dim=-1)

    def exp(self, x: Tensor, v: Tensor) -> Tensor:
        # Geodesic: exp_x(v) = cos(‖v‖) x + sin(‖v‖) v̂.   ‖v‖ here is ambient = chart.
        norm = v.norm(dim=-1, keepdim=True)
        # safe normalization for v=0
        v_hat = torch.where(norm > 1e-12, v / norm.clamp(min=1e-12), torch.zeros_like(v))
        return torch.cos(norm) * x + torch.sin(norm) * v_hat

    def log(self, x: Tensor, y: Tensor) -> Tensor:
        # Signed shortest-arc angle from x to y, lifted along perp(x).
        perp = self._perp(x)
        cos_a = (x * y).sum(-1)
        sin_a = (perp * y).sum(-1)
        angle = torch.atan2(sin_a, cos_a)
        return angle.unsqueeze(-1) * perp

    def proj_to_tangent(self, x: Tensor, v: Tensor) -> Tensor:
        # Remove radial component.
        return v - (v * x).sum(-1, keepdim=True) * x

    def lift_chart_to_tangent(self, x: Tensor, a: Tensor) -> Tensor:
        # a: (..., 1) chart-coord scalar; lift to (..., 2) ambient via perp(x) = J_H.
        return a * self._perp(x)

    def random_normal_tangent(self, x: Tensor) -> Tensor:
        # G = 1 ⇒ standard normal in chart.  Lift to ambient via perp(x).
        a = torch.randn(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
        return self.lift_chart_to_tangent(x, a)

    def random_uniform(self, n: int, device=None, dtype=torch.float32) -> Tensor:
        theta = 2 * math.pi * torch.rand(n, device=device, dtype=dtype)
        return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    def squared_norm(self, x: Tensor, v: Tensor) -> Tensor:
        # G = I in ambient (and equivalently in chart since J_H^T J_H = 1).
        return (v * v).sum(-1)

    def belongs(self, x: Tensor, atol: float = 1e-5) -> Tensor:
        return (x.norm(dim=-1) - 1.0).abs() < atol

    def constraint(self, x: Tensor) -> Tensor:
        return ((x * x).sum(-1, keepdim=True) - 1.0)

    # ---- chart helpers ----
    def to_angle(self, x: Tensor) -> Tensor:
        return torch.atan2(x[..., 1], x[..., 0])

    def from_angle(self, theta: Tensor) -> Tensor:
        return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    # ---- exact transition-kernel score (Sturm–Liouville on S^1) ----
    def heat_kernel_score(self, x_t: Tensor, x_0: Tensor, tau: Tensor,
                          n_modes: int = 20) -> Tensor:
        """Exact ∇_{x_t} log p_{t|0}(x_t | x_0) for Brownian on S^1 in ambient form.

        Uses the Sturm–Liouville expansion of the heat kernel on S^1:
            p(θ_t | θ_0, τ) ∝ 1 + 2·Σ_{n=1}^N exp(−n²τ) cos(n(θ_t − θ_0))
        ⇒ ∂_{θ_t} log p = −2·Σ n exp(−n²τ) sin(n·diff) / [1 + 2·Σ exp(−n²τ) cos(n·diff)]

        This replaces the Varadhan asymptotic Log_{x_t}(x_0)/t in DSM, eliminating
        the asymptotic bias and reducing per-sample target variance.
        """
        diff = self.to_angle(x_t) - self.to_angle(x_0)                       # (B,)
        ks = torch.arange(1, n_modes + 1, device=diff.device, dtype=diff.dtype)  # (M,)
        decay = torch.exp(-ks ** 2 * tau.unsqueeze(-1))                       # (B, M)

        kdiff = ks.unsqueeze(0) * diff.unsqueeze(-1)                          # (B, M)
        cos_part = (decay * torch.cos(kdiff)).sum(-1)                         # (B,)
        sin_part = (decay * ks.unsqueeze(0) * torch.sin(kdiff)).sum(-1)       # (B,)

        p_unnorm = 1.0 + 2.0 * cos_part
        dp_unnorm = -2.0 * sin_part
        score_chart = dp_unnorm / p_unnorm.clamp(min=1e-30)                   # (B,)
        return self.lift_chart_to_tangent(x_t, score_chart.unsqueeze(-1))

    # ---- analytic forward marginal (closed-form on S^1) ----
    def marginal_sample(self, x_0: Tensor, t: Tensor, schedule) -> Tensor:
        # For Brownian on S^1 with diffusion √β(t), the heat kernel started at x_0 is
        # the wrapped Gaussian at x_0 with variance ∫_0^t β(s) ds.  Sample by drawing
        # a tangent Gaussian δθ ~ N(0, I(t)) and exp_{x_0}(δθ · perp(x_0)).
        I = schedule.integral(t).clamp(min=1e-12)              # (B,)
        a = torch.randn_like(t) * I.sqrt()                       # (B,)
        v = self.lift_chart_to_tangent(x_0, a.unsqueeze(-1))    # (B, 2)
        return self.exp(x_0, v)


# =====================================================================
# Graph manifolds  M = {(q, p) ∈ R^{n_q+n_p} : p = F(q)}
# =====================================================================


class GraphManifold(Manifold):
    """Embedded manifold of the form  M = {(q, p) : p = F(q)},  q ∈ R^{n_q}.

    Self-model in the SMCDP framework (Idea §2.3) is a graph manifold with
    F(q) = FK_analytic(q) + Δ_φ(q).  Here we keep F abstract; subclasses provide
    F and (optionally) a closed-form ∂F/∂q for speed.

    Geometry — automatic from the embedding:
        H(q) = (q, F(q))                      embedding map R^{n_q} → R^d
        J_H(q) = [I; J_F(q)]                  embedding Jacobian
        G(q)  = J_H^T J_H = I + J_F^T J_F     induced metric in chart
        T_xM  = Im(J_H(q)) = ker(J_g(x))      tangent space at x = (q, p)
                                              with J_g = [-J_F, I]

    Norm equivalence (Idea §3.3):  for v = J_H(q) a ∈ T_xM ⊂ R^d,
        ‖v‖²_ambient = v^T v = a^T G(q) a = ‖a‖²_G

    metric mode:
      - 'riemannian'    : tangent Gaussian ~ N(0, G^{-1}) in chart;
                          squared norm uses ambient = a^T G a in chart.
                          This is the genuine induced Riemannian metric.
      - 'chart_euclidean': pretend G = I in chart.  Tangent Gaussian ~ N(0, I);
                          squared norm = a^T a (chart-Euclidean).
                          Toy approximation per Idea §3.3 (sub-phase 2a).
    """

    def __init__(self, n_q: int, n_p: int, metric: str = "riemannian"):
        if metric not in ("riemannian", "chart_euclidean"):
            raise ValueError(f"unknown metric mode '{metric}'")
        self.n_q = n_q
        self.n_p = n_p
        self.metric = metric

    @property
    def ambient_dim(self) -> int:                                       # type: ignore[override]
        return self.n_q + self.n_p

    @property
    def intrinsic_dim(self) -> int:                                     # type: ignore[override]
        return self.n_q

    @abstractmethod
    def F(self, q: Tensor) -> Tensor:
        """Forward map q ∈ R^{n_q} ↦ p ∈ R^{n_p}."""

    def jacobian_F(self, q: Tensor) -> Tensor:
        """∂F/∂q ∈ R^{n_p × n_q}.  Default uses autograd; override for closed-form."""
        # vmap over the batch dimension; jacrev returns (n_p, n_q) for each q
        return torch.func.vmap(torch.func.jacrev(self.F))(q)

    # ---- derived geometric objects ----
    def H(self, q: Tensor) -> Tensor:
        return torch.cat([q, self.F(q)], dim=-1)

    def G(self, q: Tensor) -> Tensor:
        Jf = self.jacobian_F(q)                                          # (..., n_p, n_q)
        eye = torch.eye(self.n_q, device=q.device, dtype=q.dtype)
        return eye + Jf.transpose(-1, -2) @ Jf                           # (..., n_q, n_q)

    def G_chol(self, q: Tensor) -> Tensor:
        """Lower-triangular L with L L^T = G(q).  Used to sample N(0, G^{-1}) tangent noise."""
        return torch.linalg.cholesky(self.G(q))

    # ---- Manifold ABC implementation ----
    def lift_chart_to_tangent(self, x: Tensor, a: Tensor) -> Tensor:
        q = x[..., : self.n_q]
        Jf = self.jacobian_F(q)                                          # (..., n_p, n_q)
        Jfa = (Jf @ a.unsqueeze(-1)).squeeze(-1)                         # (..., n_p)
        return torch.cat([a, Jfa], dim=-1)                               # (..., n_q+n_p)

    def exp(self, x: Tensor, v: Tensor) -> Tensor:
        # Graph retraction (Idea §5.2): take the q-component of v, advance, re-lift via H.
        # NOT an isometry; not the true Riemannian exp.  Cheap and stays on M by construction.
        q = x[..., : self.n_q]
        delta_q = v[..., : self.n_q]
        return self.H(q + delta_q)

    def log(self, x: Tensor, y: Tensor) -> Tensor:
        # Graph-retraction inverse: lift the chart difference (q_y − q_x) back to T_xM.
        # Requires y ∈ M.  At small distances this matches the Varadhan-asymptotic Log
        # used by RSGM's DSM target (q-difference is the leading-order Log on M).
        q_x = x[..., : self.n_q]
        q_y = y[..., : self.n_q]
        return self.lift_chart_to_tangent(x, q_y - q_x)

    def proj_to_tangent(self, x: Tensor, v: Tensor) -> Tensor:
        # Orthogonal projection of an ambient vector onto T_xM:
        # solve  a* = arg min_a ‖(a, J_F a) − v‖²,  i.e. a* = G^{-1} (v_q + J_F^T v_p).
        q = x[..., : self.n_q]
        v_q = v[..., : self.n_q]
        v_p = v[..., self.n_q :]
        Jf = self.jacobian_F(q)
        rhs = v_q + (Jf.transpose(-1, -2) @ v_p.unsqueeze(-1)).squeeze(-1)
        L = self.G_chol(q)
        a = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
        return self.lift_chart_to_tangent(x, a)

    def random_normal_tangent(self, x: Tensor) -> Tensor:
        q = x[..., : self.n_q]
        z = torch.randn(*q.shape, device=q.device, dtype=q.dtype)
        if self.metric == "riemannian":
            # ξ_q ~ N(0, G^{-1}).  Cov(L^{-T} z) = L^{-T} I L^{-1} = G^{-1} where G = L L^T.
            L = self.G_chol(q)
            a = torch.linalg.solve_triangular(
                L.transpose(-1, -2), z.unsqueeze(-1), upper=True
            ).squeeze(-1)
        else:
            a = z
        return self.lift_chart_to_tangent(x, a)

    def squared_norm(self, x: Tensor, v: Tensor) -> Tensor:
        if self.metric == "riemannian":
            # ‖v‖²_ambient = a^T G a for v ∈ T_xM (auto-G-weighted).
            return (v * v).sum(-1)
        # Chart-Euclidean: drop the p-part and use ‖a‖².
        a = v[..., : self.n_q]
        return (a * a).sum(-1)

    def belongs(self, x: Tensor, atol: float = 1e-5) -> Tensor:
        q = x[..., : self.n_q]
        p = x[..., self.n_q :]
        return (p - self.F(q)).norm(dim=-1) < atol

    def constraint(self, x: Tensor) -> Tensor:
        q = x[..., : self.n_q]
        p = x[..., self.n_q :]
        return p - self.F(q)

    # No closed-form uniform measure on a non-compact graph manifold; subclasses can
    # provide a chart-bounded box for use as a sampling prior.
    has_analytic_marginal = False

    def random_uniform(self, n: int, device=None, dtype=torch.float32) -> Tensor:  # pragma: no cover
        raise NotImplementedError(
            "GraphManifold over R^{n_q} has no canonical uniform measure; "
            "override random_uniform or use marginal_sample / data prior."
        )


class TwoLinkArm(GraphManifold):
    """Planar 2-link arm forward-kinematics manifold M ⊂ R^4.

    q = (q_1, q_2) ∈ R^2 (joint angles, treated as a Euclidean chart).
    F(q) = (l_1 cos q_1 + l_2 cos(q_1+q_2),  l_1 sin q_1 + l_2 sin(q_1+q_2)).
    Singular at q_2 = 0 or ±π (J_F drops rank); avoid with a bounded q-box for now.
    """

    def __init__(
        self,
        l1: float = 1.0,
        l2: float = 1.0,
        metric: str = "riemannian",
        q_box_half: float = 1.5,        # box for chart-uniform sampling prior
    ):
        super().__init__(n_q=2, n_p=2, metric=metric)
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.q_box_half = float(q_box_half)

    def F(self, q: Tensor) -> Tensor:
        q1 = q[..., 0]
        q2 = q[..., 1]
        c1 = torch.cos(q1)
        s1 = torch.sin(q1)
        c12 = torch.cos(q1 + q2)
        s12 = torch.sin(q1 + q2)
        px = self.l1 * c1 + self.l2 * c12
        py = self.l1 * s1 + self.l2 * s12
        return torch.stack([px, py], dim=-1)

    def jacobian_F(self, q: Tensor) -> Tensor:
        # Closed form (avoids autograd cost).
        q1 = q[..., 0]
        q2 = q[..., 1]
        s1 = torch.sin(q1)
        c1 = torch.cos(q1)
        s12 = torch.sin(q1 + q2)
        c12 = torch.cos(q1 + q2)
        l1, l2 = self.l1, self.l2
        # rows: [∂p_x/∂q,  ∂p_y/∂q],  cols: [∂/∂q_1, ∂/∂q_2]
        row0 = torch.stack([-l1 * s1 - l2 * s12, -l2 * s12], dim=-1)     # (..., 2)
        row1 = torch.stack([l1 * c1 + l2 * c12,  l2 * c12], dim=-1)
        return torch.stack([row0, row1], dim=-2)                          # (..., 2, 2)

    def random_uniform(self, n: int, device=None, dtype=torch.float32) -> Tensor:
        # Uniform in the q-box, lifted to M.  This is uniform in chart, NOT on M
        # w.r.t. the Riemannian volume form — adequate as a sampling prior for Toy 2.
        q = (2 * torch.rand(n, 2, device=device, dtype=dtype) - 1) * self.q_box_half
        return self.H(q)


class NLinkPlanarArm(GraphManifold):
    """Planar N-link arm forward-kinematics manifold  M = {(q, p) : p = FK(q)} ⊂ R^{N+2}.

    q ∈ R^N (joint angles), p ∈ R^2 (end-effector position).  Generalises
    `TwoLinkArm` to arbitrary link counts with closed-form FK and Jacobian.

    Forward kinematics:
        s_k = q_1 + q_2 + … + q_k       (cumulative joint angle)
        F(q) = ( Σ_{k=1}^{N} l_k cos(s_k),   Σ_{k=1}^{N} l_k sin(s_k) )

    Jacobian (∂p/∂q_j depends only on links k ≥ j since q_j shifts s_k for k ≥ j):
        ∂p_x/∂q_j = − Σ_{k=j}^{N} l_k sin(s_k)
        ∂p_y/∂q_j = + Σ_{k=j}^{N} l_k cos(s_k)
    Implemented via a single suffix-cumulative-sum (cheap, fully vectorised).

    Redundancy:
        N > 2  ⇒  J_F is 2×N with rank ≤ 2 ⇒ null space dim N − 2 ≥ 1.
        Multiple joint configurations reach the same end-effector position,
        which is the source of the multi-modal demo distributions tested in
        Toy 3.5 / killer experiment (Idea_formulation §15).
    """

    def __init__(
        self,
        link_lengths: list[float] | tuple[float, ...] | torch.Tensor,
        metric: str = "riemannian",
        q_box_half: float = 1.5,
    ):
        link_lengths = torch.as_tensor(list(link_lengths), dtype=torch.float32)
        if link_lengths.ndim != 1 or link_lengths.numel() < 2:
            raise ValueError("link_lengths must be a 1D iterable of length ≥ 2")
        super().__init__(n_q=link_lengths.numel(), n_p=2, metric=metric)
        # Buffer-like: keep on CPU and transfer when needed to avoid stale device state
        self.link_lengths = link_lengths
        self.q_box_half = float(q_box_half)

    def _link_lengths(self, like: Tensor) -> Tensor:
        return self.link_lengths.to(device=like.device, dtype=like.dtype)

    def F(self, q: Tensor) -> Tensor:
        l = self._link_lengths(q)
        s = torch.cumsum(q, dim=-1)                       # (..., N)
        px = (l * torch.cos(s)).sum(-1)
        py = (l * torch.sin(s)).sum(-1)
        return torch.stack([px, py], dim=-1)

    def jacobian_F(self, q: Tensor) -> Tensor:
        # Closed form: suffix-sum of l_k sin(s_k) and l_k cos(s_k) gives column j.
        l = self._link_lengths(q)
        s = torch.cumsum(q, dim=-1)
        l_sin_s = l * torch.sin(s)                        # (..., N)
        l_cos_s = l * torch.cos(s)
        # Suffix sums (Σ_{k=j}^{N-1}) via reversed cumsum.
        sx_suffix = torch.flip(torch.cumsum(torch.flip(l_sin_s, [-1]), dim=-1), [-1])
        sy_suffix = torch.flip(torch.cumsum(torch.flip(l_cos_s, [-1]), dim=-1), [-1])
        # rows: [∂p_x/∂q, ∂p_y/∂q]
        return torch.stack([-sx_suffix, sy_suffix], dim=-2)   # (..., 2, N)

    def random_uniform(self, n: int, device=None, dtype=torch.float32) -> Tensor:
        q = (2 * torch.rand(n, self.n_q, device=device, dtype=dtype) - 1) * self.q_box_half
        return self.H(q)


# =====================================================================
# Embodiment-aware graph manifolds
#
#   M_φ(z_e) = {(q, p) ∈ R^{n_q+n_p} : p = F(q, z_e)},   z_e ∈ R^{n_z}
#
# State layout (Toy 3+):  x = (q, p, z_e)  ∈  R^{n_q + n_p + n_z}
#   - z_e is *frozen* per sample throughout the SDE (tangent component ≡ 0)
#   - F and J_F take (q, z_e); the manifold deforms with z_e
#   - induced metric  G(q, z_e) = I + J_F(q, z_e)^T J_F(q, z_e)
#   - This realises Idea_formulation §3 (z_e as manifold-deformation parameter)
#     via state augmentation rather than threading a `context` argument through
#     every SDE/sampler/loss method — the substrate code does not need changes.
# =====================================================================


class EmbodimentGraphManifold(Manifold):
    """Embodiment-parametrised graph manifold (Idea §2.3, §3).

    Same as GraphManifold but F depends on an embodiment context z_e that is
    carried as the trailing block of x.  Concrete subclasses implement F(q, z_e)
    (and optionally a closed-form ∂F/∂q at fixed z_e).
    """

    def __init__(self, n_q: int, n_p: int, n_z: int, metric: str = "riemannian"):
        if metric not in ("riemannian", "chart_euclidean"):
            raise ValueError(f"unknown metric mode '{metric}'")
        self.n_q = n_q
        self.n_p = n_p
        self.n_z = n_z
        self.metric = metric

    @property
    def ambient_dim(self) -> int:                                       # type: ignore[override]
        return self.n_q + self.n_p + self.n_z

    @property
    def intrinsic_dim(self) -> int:                                     # type: ignore[override]
        return self.n_q

    # ---- subclass contract ----
    @abstractmethod
    def F(self, q: Tensor, z: Tensor) -> Tensor:
        """F(q, z) → p ∈ R^{n_p}.  q: (..., n_q), z: (..., n_z), returns (..., n_p)."""

    def jacobian_F(self, q: Tensor, z: Tensor) -> Tensor:
        """∂F/∂q at fixed z, shape (..., n_p, n_q).  Default uses autograd."""
        def F_pair(q_single: Tensor, z_single: Tensor) -> Tensor:
            return self.F(q_single.unsqueeze(0), z_single.unsqueeze(0))[0]
        return torch.func.vmap(torch.func.jacrev(F_pair, argnums=0))(q, z)

    # ---- state split helpers ----
    def split_x(self, x: Tensor):
        n_q, n_p = self.n_q, self.n_p
        q = x[..., :n_q]
        p = x[..., n_q : n_q + n_p]
        z = x[..., n_q + n_p :]
        return q, p, z

    def make_x(self, q: Tensor, z: Tensor) -> Tensor:
        return torch.cat([q, self.F(q, z), z], dim=-1)

    # ---- derived geometric objects ----
    def G(self, q: Tensor, z: Tensor) -> Tensor:
        Jf = self.jacobian_F(q, z)
        eye = torch.eye(self.n_q, device=q.device, dtype=q.dtype)
        return eye + Jf.transpose(-1, -2) @ Jf

    def G_chol(self, q: Tensor, z: Tensor) -> Tensor:
        return torch.linalg.cholesky(self.G(q, z))

    # ---- Manifold ABC implementation ----
    def lift_chart_to_tangent(self, x: Tensor, a: Tensor) -> Tensor:
        q, _, z = self.split_x(x)
        Jf = self.jacobian_F(q, z)
        Jfa = (Jf @ a.unsqueeze(-1)).squeeze(-1)
        zeros_z = torch.zeros_like(z)                                    # z_e tangent ≡ 0 (frozen)
        return torch.cat([a, Jfa, zeros_z], dim=-1)

    def exp(self, x: Tensor, v: Tensor) -> Tensor:
        # Graph retraction with z_e held fixed.
        q, _, z = self.split_x(x)
        delta_q = v[..., : self.n_q]
        return self.make_x(q + delta_q, z)

    def log(self, x: Tensor, y: Tensor) -> Tensor:
        # Assumes x, y on the same M_φ(z_e) (i.e. matching z_e).
        q_x, _, _ = self.split_x(x)
        q_y, _, _ = self.split_x(y)
        return self.lift_chart_to_tangent(x, q_y - q_x)

    def proj_to_tangent(self, x: Tensor, v: Tensor) -> Tensor:
        q, _, z = self.split_x(x)
        v_q = v[..., : self.n_q]
        v_p = v[..., self.n_q : self.n_q + self.n_p]
        Jf = self.jacobian_F(q, z)
        rhs = v_q + (Jf.transpose(-1, -2) @ v_p.unsqueeze(-1)).squeeze(-1)
        L = self.G_chol(q, z)
        a = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
        return self.lift_chart_to_tangent(x, a)

    def random_normal_tangent(self, x: Tensor) -> Tensor:
        q, _, z = self.split_x(x)
        rand = torch.randn(*q.shape, device=q.device, dtype=q.dtype)
        if self.metric == "riemannian":
            L = self.G_chol(q, z)
            a = torch.linalg.solve_triangular(
                L.transpose(-1, -2), rand.unsqueeze(-1), upper=True
            ).squeeze(-1)
        else:
            a = rand
        return self.lift_chart_to_tangent(x, a)

    def squared_norm(self, x: Tensor, v: Tensor) -> Tensor:
        if self.metric == "riemannian":
            # v_z ≡ 0 by construction so ambient ‖v‖² = ‖(a, J_F a)‖² = a^T G a.
            return (v * v).sum(-1)
        a = v[..., : self.n_q]
        return (a * a).sum(-1)

    def belongs(self, x: Tensor, atol: float = 1e-5) -> Tensor:
        q, p, z = self.split_x(x)
        return (p - self.F(q, z)).norm(dim=-1) < atol

    def constraint(self, x: Tensor) -> Tensor:
        q, p, z = self.split_x(x)
        return p - self.F(q, z)

    has_analytic_marginal = False

    def random_uniform(self, n: int, device=None, dtype=torch.float32) -> Tensor:  # pragma: no cover
        raise NotImplementedError(
            "Embodiment graph manifold has no canonical uniform measure; "
            "subclass should override or use a Langevin SDE with explicit limiting."
        )


class EmbodimentTwoLinkArm(EmbodimentGraphManifold):
    """2-link planar arm with a tool extension as embodiment context.

    z_e ∈ R is the tool-length increment added to link 2:  effective ℓ_2 = ℓ_2_base + z_e.
    F(q, z_e) is the analytic forward kinematics with this extended length:

        F(q, z) = (ℓ_1 cos q_1 + (ℓ_2 + z) cos(q_1 + q_2),
                   ℓ_1 sin q_1 + (ℓ_2 + z) sin(q_1 + q_2))

    This matches Idea_formulation §15.1 killer-experiment description (tool-length
    perturbation as embodiment context).  No learned residual yet — that comes in
    LearnedSelfModelArm (built on top of this analytic FK + Δ_φ MLP).
    """

    def __init__(
        self,
        l1: float = 1.0,
        l2_base: float = 1.0,
        metric: str = "riemannian",
    ):
        super().__init__(n_q=2, n_p=2, n_z=1, metric=metric)
        self.l1 = float(l1)
        self.l2_base = float(l2_base)

    def F(self, q: Tensor, z: Tensor) -> Tensor:
        q1 = q[..., 0]
        q2 = q[..., 1]
        l2_eff = self.l2_base + z[..., 0]
        c1 = torch.cos(q1)
        s1 = torch.sin(q1)
        c12 = torch.cos(q1 + q2)
        s12 = torch.sin(q1 + q2)
        px = self.l1 * c1 + l2_eff * c12
        py = self.l1 * s1 + l2_eff * s12
        return torch.stack([px, py], dim=-1)

    def jacobian_F(self, q: Tensor, z: Tensor) -> Tensor:
        # Closed form (matches TwoLinkArm but with effective ℓ_2 depending on z).
        q1 = q[..., 0]
        q2 = q[..., 1]
        l1 = self.l1
        l2 = self.l2_base + z[..., 0]
        s1 = torch.sin(q1)
        c1 = torch.cos(q1)
        s12 = torch.sin(q1 + q2)
        c12 = torch.cos(q1 + q2)
        row0 = torch.stack([-l1 * s1 - l2 * s12, -l2 * s12], dim=-1)
        row1 = torch.stack([l1 * c1 + l2 * c12,   l2 * c12], dim=-1)
        return torch.stack([row0, row1], dim=-2)


class EmbodimentNLinkPlanarArm(EmbodimentGraphManifold):
    """N-link planar arm with the LAST link's length as embodiment context.

    Generalises EmbodimentTwoLinkArm to N ≥ 2 links.  z_e ∈ R is the tool-length
    increment added to link N (the end-effector tool):  effective ℓ_N = ℓ_N_base + z_e.

    Forward kinematics with cumulative angles  s_k = q_1 + … + q_k:
        F(q, z) = ( Σ_{k<N} ℓ_k cos(s_k) + (ℓ_N + z) cos(s_N),
                    Σ_{k<N} ℓ_k sin(s_k) + (ℓ_N + z) sin(s_N) )

    Jacobian: same suffix-cumulative-sum pattern as `NLinkPlanarArm`, but ℓ_N is
    replaced by  ℓ_N_eff(z) = ℓ_N_base + z  in the contributions involving the
    last link.

    Redundancy:  N > 2  ⇒  J_F ∈ R^{2 × N} has null space dim ≥ N − 2 (kinematic
    redundancy).  When combined with bimodal IK demos this gives the killer-
    experiment multi-modal trajectory test (Idea §15.1).
    """

    def __init__(
        self,
        link_lengths_base: list[float] | tuple[float, ...] | torch.Tensor,
        metric: str = "riemannian",
    ):
        link_lengths_base = torch.as_tensor(list(link_lengths_base), dtype=torch.float32)
        if link_lengths_base.ndim != 1 or link_lengths_base.numel() < 2:
            raise ValueError("link_lengths_base must be a 1D iterable of length ≥ 2")
        n_q = link_lengths_base.numel()
        super().__init__(n_q=n_q, n_p=2, n_z=1, metric=metric)
        self.link_lengths_base = link_lengths_base

    def _link_lengths_eff(self, z: Tensor, like: Tensor) -> Tensor:
        """Effective link lengths ℓ(z): ℓ_k for k < N, ℓ_N + z for k = N.
        Returned shape broadcasts to match q = (..., N).  Uses out-of-place
        ops so it composes with autograd / vmap.
        """
        l_base = self.link_lengths_base.to(device=like.device, dtype=like.dtype)  # (N,)
        target_shape = (*like.shape[:-1], self.n_q)
        # broadcast first N-1 entries unchanged
        l_kept = l_base[: self.n_q - 1].expand(*like.shape[:-1], self.n_q - 1)
        # last entry: ℓ_N_base + z_e[..., 0]
        l_last = (l_base[self.n_q - 1] + z[..., 0]).unsqueeze(-1)             # (..., 1)
        return torch.cat([l_kept, l_last], dim=-1)                             # (..., N)

    def F(self, q: Tensor, z: Tensor) -> Tensor:
        l_eff = self._link_lengths_eff(z, q)                 # (..., N)
        s = torch.cumsum(q, dim=-1)                          # (..., N)
        px = (l_eff * torch.cos(s)).sum(-1)
        py = (l_eff * torch.sin(s)).sum(-1)
        return torch.stack([px, py], dim=-1)

    def jacobian_F(self, q: Tensor, z: Tensor) -> Tensor:
        # Closed form via suffix-sum trick (same as NLinkPlanarArm), with ℓ_N
        # replaced by ℓ_N_base + z_e in the last entry.
        l_eff = self._link_lengths_eff(z, q)                 # (..., N)
        s = torch.cumsum(q, dim=-1)
        l_sin_s = l_eff * torch.sin(s)
        l_cos_s = l_eff * torch.cos(s)
        sx_suffix = torch.flip(torch.cumsum(torch.flip(l_sin_s, [-1]), dim=-1), [-1])
        sy_suffix = torch.flip(torch.cumsum(torch.flip(l_cos_s, [-1]), dim=-1), [-1])
        return torch.stack([-sx_suffix, sy_suffix], dim=-2)  # (..., 2, N)


# =====================================================================
# 7-DoF Franka Panda  (Idea_formulation §15.1 Phase 4 killer experiment)
# =====================================================================


class Franka7DoF(EmbodimentGraphManifold):
    """7-DoF Franka Panda position-only graph manifold.

    State: x = (q, p, z_e) ∈ R^{7+3+1}.
        q ∈ R^7    — joint angles
        p ∈ R^3    — tool-tip 3D position (world frame)
        z_e ∈ R    — tool-tip offset along panda_hand body z-axis (Idea §15.1)

    Forward map (kinematics + tool extension):
        pos_hand(q), R_hand(q) := SE(3) of panda_hand ← pytorch_kinematics
        F(q, z_e) = pos_hand(q) + R_hand(q) @ [0, 0, z_e]

    Closed-form Jacobian (avoids vmap-in-pytorch_kinematics issues):
        Let J_lin, J_ang ∈ R^{3×7} be the body Jacobian rows of panda_hand
        (linear / angular parts of geometric Jacobian, returned by
        chain.jacobian(q)).  Tool offset in world frame is
            o(q, z_e) = R_hand(q) @ [0,0,z_e] = z_e · R_hand[:, 2]   (third column)
        Then
            ∂F/∂q_i  =  J_lin[:, i]  +  ω_i × o(q, z_e)
        where ω_i = J_ang[:, i] is the angular Jacobian's i-th column.

    Joint limits (URDF panda):
        q_lower / q_upper accessible as buffers, used for chart-uniform
        sampling and (later) the Projected baseline's constraint set.
    """

    def __init__(
        self,
        urdf_path: str,
        end_link: str = "panda_hand",
        tool_z_max: float = 0.20,
        metric: str = "riemannian",
        joint_limit_margin_frac: float = 0.10,
    ):
        super().__init__(n_q=7, n_p=3, n_z=1, metric=metric)
        # Lazy import keeps the rest of the package independent of pytorch_kinematics.
        import pytorch_kinematics as pk
        # Suppress URDF parser warnings about <material> / <contact> tags.
        import logging
        _pk_log_level = logging.getLogger("pytorch_kinematics").level
        logging.getLogger("pytorch_kinematics").setLevel(logging.ERROR)
        try:
            with open(urdf_path) as f:
                urdf_str = f.read()
            chain = pk.build_serial_chain_from_urdf(urdf_str, end_link)
        finally:
            logging.getLogger("pytorch_kinematics").setLevel(_pk_log_level)
        self.chain = chain
        self.urdf_path = str(urdf_path)
        self.end_link = str(end_link)
        self.tool_z_max = float(tool_z_max)

        lower, upper = chain.get_joint_limits()
        self.q_lower = torch.as_tensor(list(lower), dtype=torch.float32)
        self.q_upper = torch.as_tensor(list(upper), dtype=torch.float32)
        self.joint_limit_margin_frac = float(joint_limit_margin_frac)

        self._chain_state = None  # (device, dtype) of last `.to()` call

    # ---- internal helpers ----
    def _ensure_chain(self, like: Tensor) -> None:
        """pytorch_kinematics chain holds internal buffers; sync device/dtype on first use."""
        target = (like.device, like.dtype)
        if self._chain_state != target:
            self.chain = self.chain.to(device=like.device, dtype=like.dtype)
            self._chain_state = target

    def _fk_hand(self, q_flat: Tensor):
        """Forward kinematics for panda_hand. q_flat: (B, 7) → (pos (B,3), R (B,3,3))."""
        m = self.chain.forward_kinematics(q_flat).get_matrix()   # (B, 4, 4)
        return m[..., :3, 3], m[..., :3, :3]

    # ---- subclass contract ----
    def F(self, q: Tensor, z: Tensor) -> Tensor:
        batch_shape = q.shape[:-1]
        q_flat = q.reshape(-1, 7)
        z_flat = z.reshape(-1, 1)
        self._ensure_chain(q_flat)
        pos, R = self._fk_hand(q_flat)
        # Tool offset in world frame = z_e · R[:, :, 2]  (third column = world image of body z-axis)
        offset_world = R[..., :, 2] * z_flat                       # (B, 3)
        p = pos + offset_world
        return p.reshape(*batch_shape, 3)

    def jacobian_F(self, q: Tensor, z: Tensor) -> Tensor:
        batch_shape = q.shape[:-1]
        q_flat = q.reshape(-1, 7)
        z_flat = z.reshape(-1, 1)
        self._ensure_chain(q_flat)
        # 6D geometric body Jacobian: rows 0:3 = linear (∂pos_hand/∂q), rows 3:6 = angular (ω_i).
        J_full = self.chain.jacobian(q_flat)                       # (B, 6, 7)
        m = self.chain.forward_kinematics(q_flat).get_matrix()     # (B, 4, 4)
        R = m[..., :3, :3]
        J_lin = J_full[:, :3, :]                                   # (B, 3, 7)
        J_ang = J_full[:, 3:, :]                                   # (B, 3, 7)
        # offset in world frame (broadcast over the 7 joint columns)
        offset_world = (R[..., :, 2] * z_flat).unsqueeze(-1).expand_as(J_ang)  # (B, 3, 7)
        # ∂(R · n)/∂q_i = ω_i × (R · n).  cross over the 3-vector axis (dim=1).
        cross_term = torch.cross(J_ang, offset_world, dim=1)       # (B, 3, 7)
        J_F = J_lin + cross_term                                   # (B, 3, 7)
        return J_F.reshape(*batch_shape, 3, 7)

    # ---- chart-uniform sampling within (margin-shrunken) joint limits ----
    def random_uniform(self, n: int, device=None, dtype=torch.float32) -> Tensor:
        lower = self.q_lower.to(device=device, dtype=dtype)
        upper = self.q_upper.to(device=device, dtype=dtype)
        margin = self.joint_limit_margin_frac * (upper - lower)
        lo = lower + margin
        hi = upper - margin
        q = lo + (hi - lo) * torch.rand(n, 7, device=device, dtype=dtype)
        z = torch.rand(n, 1, device=device, dtype=dtype) * self.tool_z_max
        return self.make_x(q, z)

    # ---- joint-limit utilities (for Projected baseline / eval metric) ----
    def joint_limits(self, device=None, dtype=torch.float32) -> tuple[Tensor, Tensor]:
        return (self.q_lower.to(device=device, dtype=dtype),
                self.q_upper.to(device=device, dtype=dtype))

    def violates_limits(self, q: Tensor) -> Tensor:
        lower = self.q_lower.to(device=q.device, dtype=q.dtype)
        upper = self.q_upper.to(device=q.device, dtype=q.dtype)
        return (q < lower).any(-1) | (q > upper).any(-1)
