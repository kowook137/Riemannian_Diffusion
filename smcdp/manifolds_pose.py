"""Pose-extended embodiment graph manifolds (extension.tex v2 + joint_limit_extension v4.1).

Reference: extension.tex (Sec. 1–4 for self-model + manifold + metric).
v4.1 adds: BoundedChartPoseManifold wrapper (joint_limit_extension.tex §3–§5).

State layout (storage form):
    x = (q, q_R, p, z_e) ∈ R^{n_q + 4 + 3 + n_z}
        q       ∈ R^{n_q}      joint configuration (chart coordinate)
        q_R     ∈ S^3 ⊂ R^4    end-effector quaternion (x, y, z, w; matches roma)
        p       ∈ R^3          end-effector position (world frame)
        z_e     ∈ R^{n_z}      embodiment context (frozen per sample)

Tangent layout (trivialized form):
    v = (v_q, v_ξ, v_z) ∈ R^{n_q + 6 + n_z}
        v_q     ∈ R^{n_q}      chart-velocity
        v_ξ     ∈ se(3) ≅ R^6  body-frame pose twist (ρ, ω)
        v_z ≡ 0  embodiment frozen.

NOTE the dimensional asymmetry: points use SE(3)-storage (7 reals/pose) while
tangents use se(3)-trivialized (6 reals/pose).  This is intentional and matches
extension.tex Sec. 3 ("trivialized tangent / configuration manifold dimension"
distinct from raw matrix-embedding dimension).  We expose `ambient_dim` as the
*point* dimension for compatibility with callers that read it; tangent shapes
are documented at each method.

Geometric forward map (extension.tex Eq. (1)):
    T_φ(q, z_e) = T_analytic(q, z_e) · exp_SE(3)(ξ_φ(q, z_e)^∧)

Induced metric (extension.tex Eq. (12)):
    G_pose = I_{n_q} + J_pose^⊤ W J_pose,    W = diag(W_p I_3, W_R I_3)

Implementation:
- Subclasses provide `T_phi_Rp(q, z)` returning (R, p) ∈ SO(3) × R^3 — vmap-safe
  (no quaternion ops).  This is the ONLY abstract method.
- `jacobian_pose` defaults to autograd-through-log_relative_Rp via
  `lie_se3.jacobian_pose_autograd`; subclasses can override with closed forms.
- `T_phi_storage` (storage-form 7-vector) is built on top by quaternion
  conversion at the boundary.
"""
from __future__ import annotations

from abc import abstractmethod

import torch
from torch import Tensor

from smcdp.manifolds import Manifold
from smcdp.lie_se3 import (
    Rp_to_pose7, pose7_to_Rp,
    log_SE3,
    compose_Rp, inverse_Rp, log_relative_Rp,
    jacobian_pose_autograd,
    quat_to_R, R_to_quat,
)


class EmbodimentPoseGraphManifold(Manifold):
    """Pose-extended embodiment graph manifold (extension.tex Sec. 2).

    Subclass contract: implement `T_phi_Rp(q, z) -> (R, p)`.
    """

    def __init__(
        self,
        n_q: int,
        n_z: int,
        sigma_p: float = 0.01,
        sigma_R: float = 0.1,
        metric: str = "riemannian",
        tikhonov_frac: float = 0.0,
    ):
        if metric not in ("riemannian", "chart_euclidean"):
            raise ValueError(f"unknown metric mode '{metric}'")
        self.n_q = n_q
        self.n_z = n_z
        # Pose-tangent weights — extension.tex Sec. 4 (W = diag(W_p I_3, W_R I_3)).
        self.sigma_p = float(sigma_p)
        self.sigma_R = float(sigma_R)
        self.W_p = float(sigma_p) ** -2
        self.W_R = float(sigma_R) ** -2
        self.metric = metric
        # Fix 2 (noise_stationary_fix.md): adaptive Tikhonov on G_pose.
        # 0.0 → fixed jitter only (legacy behavior).  Set > 0 (e.g. 1e-2) to
        # add λ(q) = tikhonov_frac · tr(G)/n_q before Cholesky.
        self.tikhonov_frac = float(tikhonov_frac)

    # ---------------- dim properties ----------------
    @property
    def ambient_dim(self) -> int:                                       # type: ignore[override]
        # *Point* dimension (storage form): n_q + 7 + n_z.
        return self.n_q + 7 + self.n_z

    @property
    def intrinsic_dim(self) -> int:                                     # type: ignore[override]
        return self.n_q

    @property
    def tangent_dim(self) -> int:
        # Trivialized tangent dim: n_q + 6 + n_z.
        return self.n_q + 6 + self.n_z

    # ---------------- subclass contract ----------------
    @abstractmethod
    def T_phi_Rp(self, q: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        """Forward map  q (..., n_q), z (..., n_z) → (R (..., 3, 3), p (..., 3)).

        Must be vmap-safe (no quaternion roundtrip, no in-place ops, no
        data-dependent control flow).  Used inside the autograd-critical paths
        (J_pose, DSM target, reward gradients).
        """

    # Storage-form variant — boundary use only (calls roma quat ops).
    def T_phi_storage(self, q: Tensor, z: Tensor) -> Tensor:
        """T_φ in storage form (q_R, p) ∈ R^7.  NOT vmap-safe (uses R_to_quat)."""
        R, p = self.T_phi_Rp(q, z)
        return Rp_to_pose7(R, p)

    # ---------------- W matrix helpers ----------------
    def _W_diag(self, like: Tensor) -> Tensor:
        """W = diag(W_p I_3, W_R I_3) ∈ R^6 (returned as a length-6 vector)."""
        return torch.tensor(
            [self.W_p, self.W_p, self.W_p, self.W_R, self.W_R, self.W_R],
            device=like.device, dtype=like.dtype,
        )

    # ---------------- geometric Jacobian ----------------
    def jacobian_pose(self, q: Tensor, z: Tensor) -> Tensor:
        """Body-frame geometric Jacobian J_pose(q, z) ∈ R^{(...) × 6 × n_q}.

        Default implementation: autograd through `log_relative_Rp` with z fixed
        per sample.  Subclasses (e.g. Franka7DoFPose) may override with a
        closed-form expression for speed.
        """
        q_base = q.detach()
        R_base, p_base = self.T_phi_Rp(q_base, z)
        R_inv, p_inv = inverse_Rp(R_base, p_base)
        R_inv = R_inv.detach()
        p_inv = p_inv.detach()

        def xi_per_sample(q_var_s: Tensor, z_s: Tensor,
                          R_inv_s: Tensor, p_inv_s: Tensor) -> Tensor:
            R_var, p_var = self.T_phi_Rp(q_var_s.unsqueeze(0), z_s.unsqueeze(0))
            R_var = R_var.squeeze(0)
            p_var = p_var.squeeze(0)
            R_rel = R_inv_s @ R_var
            p_rel = (R_inv_s @ p_var.unsqueeze(-1)).squeeze(-1) + p_inv_s
            return log_SE3(R_rel, p_rel)

        return torch.func.vmap(
            torch.func.jacrev(xi_per_sample, argnums=0),
            in_dims=(0, 0, 0, 0),
        )(q, z, R_inv, p_inv)

    def G_pose(self, q: Tensor, z: Tensor) -> Tensor:
        """G_pose = I + J_pose^⊤ W J_pose ∈ R^{n_q × n_q}."""
        Jp = self.jacobian_pose(q, z)                                    # (..., 6, n_q)
        W = self._W_diag(q).unsqueeze(-1)                                # (6, 1)
        eye = torch.eye(self.n_q, device=q.device, dtype=q.dtype)
        return eye + Jp.transpose(-1, -2) @ (W * Jp)

    def G_pose_chol(self, q: Tensor, z: Tensor, jitter: float = 1e-4,
                    tikhonov_frac: float | None = None) -> Tensor:
        """Cholesky of G_pose with optional adaptive Tikhonov + jitter retry.

        Per `noise_stationary_fix.md` Fix 2: adaptive Tikhonov is a principled
        upgrade of the ad-hoc fixed jitter 1e-4.  Replaces fixed `jitter` with
            λ(q) = tikhonov_frac · tr(G(q)) / n_q
        which scales with G's magnitude (so always proportional to G's
        diagonal, regardless of σ_p choice).

        - `tikhonov_frac = None`:  read from `self.tikhonov_frac` (default 0.0
            → fixed jitter only, backward-compat).
        - `tikhonov_frac > 0.0`:  adaptive Tikhonov + final jitter as safety belt.

        Note: additive λI cannot fundamentally reduce cond(I + J^T W J) when
        max eig is dominated by W_p · ‖J‖² ≫ 1 (that is Fix 1's role: loosen
        σ_p).  Tikhonov primarily helps off-distribution q where G's
        diagonal grows.
        """
        if tikhonov_frac is None:
            tikhonov_frac = self.tikhonov_frac
        G = self.G_pose(q, z)
        eye = torch.eye(self.n_q, device=q.device, dtype=q.dtype)
        if tikhonov_frac > 0.0:
            tr_G = torch.diagonal(G, dim1=-2, dim2=-1).sum(-1)               # (...,)
            lam = (tikhonov_frac * tr_G / self.n_q).unsqueeze(-1).unsqueeze(-1)
            G = G + lam * eye
        for j in (jitter, jitter * 10.0, jitter * 100.0):
            try:
                return torch.linalg.cholesky(G + j * eye)
            except torch._C._LinAlgError:
                continue
        return torch.linalg.cholesky(G + (jitter * 1000.0) * eye)

    # ---------------- state split / assemble ----------------
    def split_x(self, x: Tensor):
        n_q = self.n_q
        q = x[..., :n_q]
        q_R = x[..., n_q : n_q + 4]
        p = x[..., n_q + 4 : n_q + 7]
        z = x[..., n_q + 7 :]
        return q, q_R, p, z

    def make_x(self, q: Tensor, z: Tensor) -> Tensor:
        T7 = self.T_phi_storage(q, z)                                    # (..., 7)
        return torch.cat([q, T7, z], dim=-1)

    def H(self, q: Tensor, z: Tensor) -> Tensor:
        """Embedding map H_φ^pose(q, z_e) — extension.tex Eq. (8)."""
        return self.make_x(q, z)

    def split_v(self, v: Tensor):
        """Split trivialized tangent v ∈ R^{n_q + 6 + n_z}."""
        n_q = self.n_q
        v_q = v[..., :n_q]
        v_xi = v[..., n_q : n_q + 6]
        v_z = v[..., n_q + 6 :]
        return v_q, v_xi, v_z

    # ---------------- Manifold ABC ----------------
    def lift_chart_to_tangent(self, x: Tensor, a: Tensor) -> Tensor:
        """a ∈ R^{n_q} → (a, J_pose · a, 0) ∈ R^{n_q + 6 + n_z}.

        Implements extension.tex Eq. (10) (J_H^pose).  Output is in trivialized
        tangent form: the pose-block is the body-frame velocity twist ξ ∈ se(3),
        NOT a quaternion-velocity.
        """
        q, _, _, z = self.split_x(x)
        Jp = self.jacobian_pose(q, z)
        xi = (Jp @ a.unsqueeze(-1)).squeeze(-1)                          # (..., 6)
        zero_z = torch.zeros_like(z)
        return torch.cat([a, xi, zero_z], dim=-1)

    def exp(self, x: Tensor, v: Tensor) -> Tensor:
        """Graph retraction (extension.tex Eq. (24)):
            Retr_x(v) = (q + v_q, T_φ(q + v_q, z_e)).
        v_ξ and v_z are unused — recovery via T_φ keeps the result on M^pose
        exactly (manifold adherence by construction)."""
        q, _, _, z = self.split_x(x)
        delta_q = v[..., : self.n_q]
        return self.make_x(q + delta_q, z)

    def log(self, x: Tensor, y: Tensor) -> Tensor:
        """Graph-retraction inverse: lift the chart difference (q_y − q_x).

        Assumes both x, y on the same M_φ^pose(z_e) (same z_e).  Matches the
        Varadhan-asymptotic Log used in the chart-form DSM target.
        """
        q_x, _, _, _ = self.split_x(x)
        q_y, _, _, _ = self.split_x(y)
        return self.lift_chart_to_tangent(x, q_y - q_x)

    def proj_to_tangent(self, x: Tensor, v: Tensor) -> Tensor:
        """Orthogonal projection (under W̃ = diag(I_{n_q}, W)) of v ∈ R^{n_q+6+n_z}
        onto Im(J_H^pose).  Solves the normal equations:
            G_pose · a* = v_q + J_pose^⊤ W v_ξ
        and returns the lifted projection.
        """
        q, _, _, z = self.split_x(x)
        v_q = v[..., : self.n_q]
        v_xi = v[..., self.n_q : self.n_q + 6]
        Jp = self.jacobian_pose(q, z)
        W = self._W_diag(q)                                              # (6,)
        rhs = v_q + (Jp.transpose(-1, -2) @ (W.unsqueeze(-1) * v_xi.unsqueeze(-1))
                      ).squeeze(-1)
        L = self.G_pose_chol(q, z)
        a_star = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
        return self.lift_chart_to_tangent(x, a_star)

    def random_normal_tangent(self, x: Tensor) -> Tensor:
        """Riemannian Gaussian tangent at x with chart covariance G_pose^{-1};
        returned in lifted (trivialized ambient) form so that
        ‖v‖²_{W̃} = a^⊤ G_pose a (norm equivalence).
        """
        q, _, _, z = self.split_x(x)
        rand = torch.randn(*q.shape, device=q.device, dtype=q.dtype)
        if self.metric == "riemannian":
            L = self.G_pose_chol(q, z)
            a = torch.linalg.solve_triangular(
                L.transpose(-1, -2), rand.unsqueeze(-1), upper=True
            ).squeeze(-1)
        else:
            a = rand
        return self.lift_chart_to_tangent(x, a)

    def squared_norm(self, x: Tensor, v: Tensor) -> Tensor:
        """Riemannian: ‖v‖²_{W̃} = ‖v_q‖² + v_ξ^⊤ W v_ξ for v ∈ T_xM lifted form.
        Chart-Euclidean: drop the pose-block, use ‖v_q‖²."""
        v_q = v[..., : self.n_q]
        if self.metric == "chart_euclidean":
            return (v_q * v_q).sum(-1)
        v_xi = v[..., self.n_q : self.n_q + 6]
        W = self._W_diag(v)                                              # (6,)
        return (v_q * v_q).sum(-1) + ((v_xi * v_xi) * W).sum(-1)

    def belongs(self, x: Tensor, atol: float = 1e-5) -> Tensor:
        q, q_R, p, z = self.split_x(x)
        R = quat_to_R(q_R)
        R_phi, p_phi = self.T_phi_Rp(q, z)
        # SE(3) error twist; both translation (m) and rotation (rad) compared
        # against atol.
        xi = log_relative_Rp(R_phi, p_phi, R, p)
        return xi.norm(dim=-1) < atol

    def constraint(self, x: Tensor) -> Tensor:
        """g_φ(x, z_e) = Log_{SE(3)}(T_φ(q)^{-1} T) ∈ se(3).  Zero iff x ∈ M_φ^pose."""
        q, q_R, p, z = self.split_x(x)
        R = quat_to_R(q_R)
        R_phi, p_phi = self.T_phi_Rp(q, z)
        return log_relative_Rp(R_phi, p_phi, R, p)                       # (..., 6)

    has_analytic_marginal = False

    def random_uniform(self, n: int, device=None, dtype=torch.float32) -> Tensor:  # pragma: no cover
        raise NotImplementedError(
            "EmbodimentPoseGraphManifold has no canonical uniform measure; "
            "subclasses should provide a chart-bounded sampler."
        )


# =====================================================================
# 7-DoF Franka Panda — pose self-model manifold (analytic FK only)
# =====================================================================


class Franka7DoFPose(EmbodimentPoseGraphManifold):
    """Franka 7-DoF pose self-model with analytic FK and zero residual.

    Identical kinematics to `Franka7DoF` (position-only) — same URDF, same hand
    link, same tool-extension semantics — but produces a full SE(3) pose:
        T_analytic(q, z_e) = T_hand(q) · diag(I_3, (0,0,z_e))
        i.e.  R_analytic = R_hand(q),
              p_analytic = p_hand(q) + z_e · R_hand(q)[:, 2].

    Subclass `LearnedSelfModelFranka7DoFPose` adds a learned se(3) residual.
    """

    def __init__(
        self,
        urdf_path: str,
        end_link: str = "panda_hand",
        tool_z_max: float = 0.20,
        sigma_p: float = 0.01,
        sigma_R: float = 0.1,
        metric: str = "riemannian",
        joint_limit_margin_frac: float = 0.10,
        tikhonov_frac: float = 0.0,
        link_perturb_dl3: float = 0.0,
        link_perturb_dl5: float = 0.0,
    ):
        super().__init__(n_q=7, n_z=1, sigma_p=sigma_p, sigma_R=sigma_R,
                          metric=metric, tikhonov_frac=tikhonov_frac)
        import pytorch_kinematics as pk
        import logging
        _pk_log_level = logging.getLogger("pytorch_kinematics").level
        logging.getLogger("pytorch_kinematics").setLevel(logging.ERROR)
        try:
            if abs(link_perturb_dl3) > 1e-9 or abs(link_perturb_dl5) > 1e-9:
                # diagnostic_plan §B: zero-shot link-length OOD eval.  Modify the
                # URDF in memory before chain construction.  Same n_q / joint
                # limits, only link 3 / link 5 lengths change.
                from smcdp.franka.franka_link_perturb import make_link_perturbed_urdf
                urdf_str = make_link_perturbed_urdf(
                    urdf_path,
                    dl_3=float(link_perturb_dl3),
                    dl_5=float(link_perturb_dl5),
                )
            else:
                with open(urdf_path) as f:
                    urdf_str = f.read()
            chain = pk.build_serial_chain_from_urdf(urdf_str, end_link)
        finally:
            logging.getLogger("pytorch_kinematics").setLevel(_pk_log_level)
        self.chain = chain
        self.link_perturb_dl3 = float(link_perturb_dl3)
        self.link_perturb_dl5 = float(link_perturb_dl5)
        self.urdf_path = str(urdf_path)
        self.end_link = str(end_link)
        self.tool_z_max = float(tool_z_max)

        lower, upper = chain.get_joint_limits()
        self.q_lower = torch.as_tensor(list(lower), dtype=torch.float32)
        self.q_upper = torch.as_tensor(list(upper), dtype=torch.float32)
        self.joint_limit_margin_frac = float(joint_limit_margin_frac)
        self._chain_state = None

    # ---- internal helpers ----
    def _ensure_chain(self, like: Tensor) -> None:
        target = (like.device, like.dtype)
        if self._chain_state != target:
            self.chain = self.chain.to(device=like.device, dtype=like.dtype)
            self._chain_state = target

    def _fk_hand(self, q_flat: Tensor):
        m = self.chain.forward_kinematics(q_flat).get_matrix()           # (B, 4, 4)
        return m[..., :3, 3], m[..., :3, :3]

    # ---- subclass contract ----
    def T_phi_Rp(self, q: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        batch_shape = q.shape[:-1]
        q_flat = q.reshape(-1, 7)
        z_flat = z.reshape(-1, 1)
        self._ensure_chain(q_flat)
        pos, R = self._fk_hand(q_flat)
        offset_world = R[..., :, 2] * z_flat                              # (B, 3)
        p = pos + offset_world
        R_out = R.reshape(*batch_shape, 3, 3)
        p_out = p.reshape(*batch_shape, 3)
        return R_out, p_out

    def jacobian_pose(self, q: Tensor, z: Tensor) -> Tensor:
        """Closed-form body-frame J_pose for analytic Franka FK.

        We get the *spatial-frame* (world) geometric Jacobian J_full from
        pytorch_kinematics:
            J_lin: ∂p_hand / ∂q ∈ R^{3 × 7}
            J_ang: ω_world / ∂q  ∈ R^{3 × 7}
        Tool-extended position adds  o = z_e · R_hand[:, 2]:
            ∂p_F / ∂q = J_lin + (J_ang × o)            (chain rule; cross over 3-axis)
        Body-frame conversion: a body-frame twist (ρ_b, ω_b) is related to
        the spatial-frame (linear v_s = ∂p / ∂q  q̇, angular ω_s) by
            ω_b = R^⊤ ω_s,    ρ_b = R^⊤ v_s    (extension.tex Sec. 3 conv.)
        """
        batch_shape = q.shape[:-1]
        q_flat = q.reshape(-1, 7)
        z_flat = z.reshape(-1, 1)
        self._ensure_chain(q_flat)
        J_full = self.chain.jacobian(q_flat)                              # (B, 6, 7) spatial
        m = self.chain.forward_kinematics(q_flat).get_matrix()
        R = m[..., :3, :3]                                                # (B, 3, 3)
        J_lin = J_full[:, :3, :]                                          # (B, 3, 7)
        J_ang = J_full[:, 3:, :]                                          # (B, 3, 7) world-frame ω
        offset_world = (R[..., :, 2] * z_flat).unsqueeze(-1).expand_as(J_ang)  # (B, 3, 7)
        cross_term = torch.cross(J_ang, offset_world, dim=1)              # (B, 3, 7)
        J_v_world = J_lin + cross_term                                    # (B, 3, 7)
        # Body-frame: J_v_body = R^⊤ J_v_world,   J_ω_body = R^⊤ J_ang
        Rt = R.transpose(-1, -2)                                          # (B, 3, 3)
        J_v_body = Rt @ J_v_world                                         # (B, 3, 7)
        J_w_body = Rt @ J_ang                                             # (B, 3, 7)
        J_pose = torch.cat([J_v_body, J_w_body], dim=-2)                  # (B, 6, 7)
        return J_pose.reshape(*batch_shape, 6, 7)

    def random_uniform(self, n: int, device=None, dtype=torch.float32) -> Tensor:
        lower = self.q_lower.to(device=device, dtype=dtype)
        upper = self.q_upper.to(device=device, dtype=dtype)
        margin = self.joint_limit_margin_frac * (upper - lower)
        lo = lower + margin
        hi = upper - margin
        q = lo + (hi - lo) * torch.rand(n, 7, device=device, dtype=dtype)
        z = torch.rand(n, 1, device=device, dtype=dtype) * self.tool_z_max
        return self.make_x(q, z)

    def joint_limits(self, device=None, dtype=torch.float32) -> tuple[Tensor, Tensor]:
        return (self.q_lower.to(device=device, dtype=dtype),
                self.q_upper.to(device=device, dtype=dtype))

    def violates_limits(self, q: Tensor) -> Tensor:
        lower = self.q_lower.to(device=q.device, dtype=q.dtype)
        upper = self.q_upper.to(device=q.device, dtype=q.dtype)
        return (q < lower).any(-1) | (q > upper).any(-1)


# =====================================================================
# Joint-limit bounded chart wrapper (joint_limit_extension v4.1)
# =====================================================================


class BoundedChartPoseManifold(EmbodimentPoseGraphManifold):
    """Bounded-chart wrapper around an EmbodimentPoseGraphManifold (v4.1 §3–§5).

    Operates in a u-chart `u ∈ R^{n_q}` with a smooth diffeomorphism
    `q = ψ(u) ∈ (q_min, q_max)` (e.g., element-wise tanh).  All chart-level
    operations accept and return u (not q); the underlying physical manifold is
    accessed transparently via ψ.

    Storage layout (chart slot stores u, T-block stores T_φ(ψ(u))):
        x = (u, q_R, p, z_e) ∈ R^{n_q + 4 + 3 + n_z}
    Same shape as v4 (`EmbodimentPoseGraphManifold`); only the chart slot
    interpretation differs.  Use `physical_q(x)` to recover q = ψ(u) when
    needed.

    Choice A semantics (v4.1 §4 default):
        J^Q(u, z_e) = J_pose(ψ(u), z_e) · D_ψ(u)        (pulled-back Jacobian)
        J_H^{Q,A}   = [I_{n_q} ; J^Q]                    (tangent map)
        G_Q^A       = I_{n_q} + (J^Q)^T W J^Q ≽ I        (lower-bounded by identity)
        Retr^Q_{(u,T)}(δu) = (u + δu, T_φ(ψ(u + δu), z_e))
        ψ auto-enforces q ∈ (q_min, q_max); no clipping during sampling.

    Backward compatibility:
        Pass an `IdentityChart` (ψ(u) = u, D_ψ = I) to recover the v4 unbounded
        formulation exactly — every method reduces to its base-class form.

    Parameters
    ----------
    base_manifold
        Concrete `EmbodimentPoseGraphManifold` subclass providing
        `T_phi_Rp(q, z) -> (R, p)` and (optionally) a closed-form
        `jacobian_pose(q, z) -> J ∈ R^{6×n_q}`.
    chart
        `Chart` instance (e.g., `TanhBoundedChart` or `IdentityChart` from
        `smcdp.charts`) with `psi(u)`, `psi_inv(q, eps)`, `D_psi_diag(u)`.
    lambda_floor
        Tikhonov floor `λ_floor` for the regularized G_Q^A (v4.1 §5.1).
        Default 1e-4; the floor is structurally less critical under Choice A
        (since G_Q^A ≽ I globally) but retained for arithmetic underflow
        safety.

    Notes
    -----
    Inherits `_W_diag`, `split_x` (returns first slot as u, not q), `make_x`,
    `H`, `split_v`, `lift_chart_to_tangent`, `exp`, `log`, `proj_to_tangent`,
    `random_normal_tangent`, `squared_norm`, `belongs`, `constraint`, and
    `T_phi_storage` from the parent.  These automatically inherit chart-aware
    semantics because they call `self.jacobian_pose`, `self.G_pose`,
    `self.T_phi_Rp` (all overridden below).
    """

    def __init__(
        self,
        base_manifold: "EmbodimentPoseGraphManifold",
        chart,                                                  # smcdp.charts.Chart
        *,
        lambda_floor: float = 1e-4,
    ):
        if base_manifold.n_q != chart.n_q:
            raise ValueError(
                f"chart.n_q ({chart.n_q}) != base_manifold.n_q ({base_manifold.n_q})"
            )
        # Parent __init__ initializes n_q, n_z, sigma_p, sigma_R, W_p, W_R,
        # metric, tikhonov_frac.  We mirror the base_manifold's settings so
        # downstream code reads the right values via self.{attr}.
        super().__init__(
            n_q=base_manifold.n_q,
            n_z=base_manifold.n_z,
            sigma_p=base_manifold.sigma_p,
            sigma_R=base_manifold.sigma_R,
            metric=base_manifold.metric,
            tikhonov_frac=base_manifold.tikhonov_frac,
        )
        self.base = base_manifold
        self.chart = chart
        # v4.1 §5.1: λ_floor — additive floor on the adaptive Tikhonov term.
        self.lambda_floor = float(lambda_floor)

    # ---------------- chart-aware overrides (v4.1 §3–§5) ----------------

    def physical_q(self, x: Tensor) -> Tensor:
        """Extract physical joint configuration q = ψ(u) from ambient state.

        Storage layout has u (not q) in the chart slot.  Use this when joint
        limits, kinematics validity, or the physical interpretation of q is
        needed downstream.
        """
        u = x[..., : self.n_q]
        return self.chart.psi(u)

    def T_phi_Rp(self, u: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        """T_φ(ψ(u), z_e) → (R, p).  Wraps base.T_phi_Rp via ψ.

        Spec v4.1 §1: the self-model T_φ is a function of physical q; we
        compose with the chart map ψ here.
        """
        q = self.chart.psi(u)
        return self.base.T_phi_Rp(q, z)

    # T_phi_storage inherited from parent — calls our T_phi_Rp + Rp_to_pose7.

    def jacobian_pose(self, u: Tensor, z: Tensor) -> Tensor:
        """Pulled-back body-frame Jacobian J^Q(u, z_e) ∈ R^{... × 6 × n_q}.

        Spec v4.1 §4:  J^Q(u, z_e) = J_pose(ψ(u), z_e) · D_ψ(u).
        Right-multiplication by the diagonal D_ψ scales each column j of
        J_pose by (q_range_j / 2) · sech²(u_j).  At u → ∞ (chart boundary),
        D_ψ → 0 and J^Q → 0 element-wise; this is the geometric source of
        the boundary degeneracy that Choice A's identity floor in G_Q^A
        protects against (G_Q^A → I, not 0).
        """
        q = self.chart.psi(u)
        J = self.base.jacobian_pose(q, z)                   # (..., 6, n_q)
        D_diag = self.chart.D_psi_diag(u)                    # (..., n_q)
        # Right-multiply by diag(D_ψ): scale column j by D_ψ[..., j].
        # (J · diag(D))[i, j] = J[i, j] · D[j].
        return J * D_diag.unsqueeze(-2)

    def G_pose(self, u: Tensor, z: Tensor) -> Tensor:
        """Choice A induced metric G_Q^A(u, z_e) ∈ R^{n_q × n_q} (v4.1 §5).

            G_Q^A = I_{n_q} + (J^Q)^T W J^Q
                  = I_{n_q} + D_ψ^T J_pose^T W J_pose D_ψ.

        Lower-bounded by I_{n_q} globally because the identity term is
        independent of D_ψ.  At chart boundary D_ψ → 0 → G_Q^A → I (well-
        conditioned, not degenerate).  Same shape as base's G_pose.
        """
        Jq = self.jacobian_pose(u, z)                        # (..., 6, n_q) — already J^Q
        W = self._W_diag(u).unsqueeze(-1)                    # (6, 1)
        eye = torch.eye(self.n_q, device=u.device, dtype=u.dtype)
        return eye + Jq.transpose(-1, -2) @ (W * Jq)

    def G_pose_chol(
        self,
        u: Tensor,
        z: Tensor,
        jitter: float = 1e-4,
        tikhonov_frac: float | None = None,
        lambda_floor: float | None = None,
    ) -> Tensor:
        """Cholesky of G_Q^{A,reg} with adaptive Tikhonov + floor (v4.1 §5.1).

            G_Q^{A,reg}(u, z_e) = G_Q^A(u, z_e) + λ_Q(u, z_e) · I_{n_q}
            λ_Q(u, z_e)         = c_λ · tr(G_Q^A) / n_q + λ_floor

        Under Choice A the floor is structurally less critical (since
        G_Q^A ≽ I), but retained for arithmetic underflow safety.  Falls back
        to the parent's jitter-retry on rare Cholesky failures.
        """
        if tikhonov_frac is None:
            tikhonov_frac = self.tikhonov_frac
        if lambda_floor is None:
            lambda_floor = self.lambda_floor
        G = self.G_pose(u, z)
        eye = torch.eye(self.n_q, device=u.device, dtype=u.dtype)
        if tikhonov_frac > 0.0 or lambda_floor > 0.0:
            tr_G = torch.diagonal(G, dim1=-2, dim2=-1).sum(-1)         # (...,)
            lam = (tikhonov_frac * tr_G / self.n_q + lambda_floor)
            lam = lam.unsqueeze(-1).unsqueeze(-1)
            G = G + lam * eye
        for j in (jitter, jitter * 10.0, jitter * 100.0, jitter * 1000.0):
            try:
                return torch.linalg.cholesky(G + j * eye)
            except torch._C._LinAlgError:
                continue
        # Per-element fallback for pathological batch entries (NaN/Inf in G,
        # or numerical degeneration past additive ridge tolerance — happens at
        # chart saturation |u|>>3 where D_psi→0).  Replace the bad slice with
        # ridge·I so cholesky_solve returns a well-defined (1/ridge)·∂R
        # direction.  Required for v5.1 anti-saturation R_u path (G4-G8).
        G_test = G + (jitter * 1000.0) * eye
        info = torch.linalg.cholesky_ex(G_test).info                  # (...,)
        bad = info.ne(0)
        if bad.any():
            ridge = max(float(lambda_floor), 1.0)
            G_safe = torch.where(
                bad.view(*bad.shape, 1, 1), ridge * eye, G_test,
            )
        else:
            G_safe = G_test
        return torch.linalg.cholesky(G_safe)

    # ---------------- limits / sanity (v4.1 §13) ----------------

    def joint_limits(self, device=None, dtype=None) -> tuple[Tensor, Tensor]:
        """(q_min, q_max) — physical joint limits inherited from chart."""
        return self.chart.joint_limits(device=device, dtype=dtype)

    def violates_limits(self, u: Tensor) -> Tensor:
        """For TanhBoundedChart, this is False at any finite u by construction
        (v4.1 §13: viol(τ) = 0 by construction).  We compute ψ(u) and check
        against the base manifold's URDF limits as a sanity check; if this
        ever returns True, it indicates a numerical issue (e.g., u became
        non-finite).
        """
        q = self.chart.psi(u)
        if hasattr(self.base, "violates_limits"):
            return self.base.violates_limits(q)
        # Fallback: use chart limits directly
        q_lo, q_hi = self.chart.joint_limits(device=q.device, dtype=q.dtype)
        return (q < q_lo).any(-1) | (q > q_hi).any(-1)

    # ---------------- sampling utilities ----------------

    def random_uniform(self, n: int, device=None, dtype=torch.float32) -> Tensor:
        """Uniform sample on M_φ^Q.

        Strategy: sample q uniformly via the base manifold (which knows the
        URDF margin convention), then convert q → u via ψ⁻¹ with η-clip
        safety (init-only, v4.1 §10.3).  T-block (= T_φ(q) = T_φ(ψ(u))) and
        z-block are inherited unchanged.
        """
        if not hasattr(self.base, "random_uniform"):
            raise NotImplementedError(
                f"BoundedChartPoseManifold.random_uniform requires base.random_uniform; "
                f"base = {type(self.base).__name__}"
            )
        x_base = self.base.random_uniform(n, device=device, dtype=dtype)
        q_unif = x_base[..., : self.n_q]
        u_unif = self.chart.psi_inv(q_unif, eps=1e-3)
        # Replace chart slot only; T-block + z-block are already T_φ(q), z.
        return torch.cat([u_unif, x_base[..., self.n_q:]], dim=-1)

    # ---------------- delegation for backbone-specific attrs ----------------

    def _ensure_chain(self, like: Tensor) -> None:
        """Delegate chain device/dtype migration to base (Franka-specific)."""
        if hasattr(self.base, "_ensure_chain"):
            self.base._ensure_chain(like)

    def __getattr__(self, name: str):
        """Fallback: delegate unknown attributes to base manifold.

        Only triggered when standard attribute lookup fails (i.e., the wrapper
        does not define `name` directly).  Allows downstream code that reads
        e.g. `arm.urdf_path` or `arm.tool_z_max` to work transparently when
        wrapping a Franka7DoFPose.

        Caveat: setting attributes on the wrapper still hits the wrapper's
        __dict__ (Python doesn't route __setattr__ through __getattr__), so
        e.g. `arm.tikhonov_frac = ...` updates the wrapper, not the base.
        Mirror sets explicitly if both layers must stay in sync.
        """
        # Avoid recursion: __getattr__ is only called when the attr isn't
        # found normally.  Forward to base's __getattribute__.
        if name in ("base", "chart", "lambda_floor"):
            raise AttributeError(name)
        try:
            return getattr(self.__dict__["base"], name)
        except KeyError:
            raise AttributeError(name)


# Convenience factory --------------------------------------------------------


def wrap_with_bounded_chart(
    base_manifold: "EmbodimentPoseGraphManifold",
    *,
    lambda_floor: float = 1e-4,
) -> "BoundedChartPoseManifold":
    """Convenience factory: wrap a base manifold with a TanhBoundedChart whose
    limits come from the manifold itself.  Equivalent to::

        chart = make_chart_from_manifold(base, bounded=True, ...)
        wrapped = BoundedChartPoseManifold(base, chart, lambda_floor=lambda_floor)
    """
    from smcdp.charts import make_chart_from_manifold
    chart = make_chart_from_manifold(base_manifold, bounded=True)
    return BoundedChartPoseManifold(base_manifold, chart, lambda_floor=lambda_floor)
