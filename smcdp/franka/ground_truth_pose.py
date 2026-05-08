"""Ground-truth Franka with simulated pose compliance — extension.tex Sec. 1.

Generalises `TrueFrankaCompliance` (position-only) by adding a smooth
SE(3)-valued residual: the "true" robot's end-effector pose deviates from the
analytic FK by a small body-frame twist ξ_true(q, z_e) = (ρ_true, ω_true):

    T_true(q, z_e) = T_analytic(q, z_e) · exp_SE(3)(ξ_true(q, z_e)^∧)

Position residual ρ_true follows the same physically-motivated compliance
pattern as `TrueFrankaCompliance` (gravity sag, elbow drift, calibration bias),
mapped from world-frame Δ to body-frame ρ via Δ = R_analytic · ρ.

Rotation residual ω_true is a smooth (q, z_e)-dependent twist with magnitude
in the 1–3° range, approximating wrist compliance + tool-mounting rotational
calibration error.

Stage-1 trains ξ_φ ∈ R^6 to recover (ρ_true, ω_true).  Reference targets
(see extension.tex Sec. 12.2 verification status):
    position: ~0.29 mm (analogous to position-only 55.8× improvement)
    rotation: ~1° (~0.017 rad) on synthetic 1-3° ground truth.
"""
from __future__ import annotations

import logging

import torch
from torch import Tensor

import pytorch_kinematics as pk

from smcdp.lie_se3 import exp_SO3, exp_SE3, compose_Rp


class TrueFrankaCompliancePose:
    """Analytic FK + 6D body-frame compliance perturbation.

        (R_true, p_true) = compose( T_analytic(q, z_e), exp_SE(3)(ξ_true(q, z_e)) )

    Position residual (body-frame ρ_true ∈ R^3):
        body-frame mapping of the existing world-frame compliance pattern.
        ρ_true = R_analytic^⊤ · Δ_world,  with Δ_world matching `TrueFrankaCompliance`.

    Rotation residual (body-frame ω_true ∈ R^3, ‖ω_true‖ in 1–3° range):
        ω_true_x = K_R · (1 + K_tool · z_e/0.2) ·  A_x · sin(q_4)
        ω_true_y = K_R · (1 + K_tool · z_e/0.2) ·  A_y · cos(q_2) · sin(q_6)
        ω_true_z = K_R                            ·  A_z · cos(q_3)

    Defaults give ‖ω_true‖ ≈ 1–3° (≈ 0.017–0.052 rad).
    """

    def __init__(
        self,
        urdf_path: str,
        end_link: str = "panda_hand",
        # Position compliance (matches TrueFrankaCompliance defaults)
        K_grav: float = 0.025,
        K_offset: float = 0.005,
        K_tool: float = 0.20,
        # Rotation compliance (1–3° range)
        K_R: float = 0.025,                                              # ~1.4° base scale (rad)
        A_x: float = 1.0,
        A_y: float = 1.0,
        A_z: float = 0.5,
        K_tool_R: float = 0.5,
    ):
        _lvl = logging.getLogger("pytorch_kinematics").level
        logging.getLogger("pytorch_kinematics").setLevel(logging.ERROR)
        try:
            with open(urdf_path) as f:
                urdf_str = f.read()
            self.chain = pk.build_serial_chain_from_urdf(urdf_str, end_link)
        finally:
            logging.getLogger("pytorch_kinematics").setLevel(_lvl)
        self.K_grav = float(K_grav)
        self.K_offset = float(K_offset)
        self.K_tool = float(K_tool)
        self.K_R = float(K_R)
        self.A_x = float(A_x)
        self.A_y = float(A_y)
        self.A_z = float(A_z)
        self.K_tool_R = float(K_tool_R)
        self._chain_state = None

    def _ensure_chain(self, like: Tensor) -> None:
        target = (like.device, like.dtype)
        if self._chain_state != target:
            self.chain = self.chain.to(device=like.device, dtype=like.dtype)
            self._chain_state = target

    def fk_analytic_Rp(self, q: Tensor, z_e: Tensor) -> tuple[Tensor, Tensor]:
        """Analytic Franka FK (matches Franka7DoFPose.T_phi_Rp): (R, p)."""
        batch_shape = q.shape[:-1]
        q_flat = q.reshape(-1, 7)
        z_flat = z_e.reshape(-1, 1)
        self._ensure_chain(q_flat)
        m = self.chain.forward_kinematics(q_flat).get_matrix()
        pos = m[..., :3, 3]
        R = m[..., :3, :3]
        offset_world = R[..., :, 2] * z_flat
        p = pos + offset_world
        return R.reshape(*batch_shape, 3, 3), p.reshape(*batch_shape, 3)

    def delta_world(self, q: Tensor, z_e: Tensor) -> Tensor:
        """World-frame position compliance (same as TrueFrankaCompliance.delta_true)."""
        q2 = q[..., 1]
        q4 = q[..., 3]
        z = z_e[..., 0]
        amp = self.K_grav * (1.0 + self.K_tool * z / 0.20)
        sag_z = -amp * torch.cos(q2) * (0.5 + 0.5 * torch.cos(q4))
        drift_y = 0.5 * amp * torch.sin(q4)
        off_x = self.K_offset * torch.ones_like(q2)
        return torch.stack([off_x, drift_y, sag_z], dim=-1)

    def omega_true(self, q: Tensor, z_e: Tensor) -> Tensor:
        """Body-frame rotation residual ω_true(q, z_e) ∈ R^3 (rad)."""
        q2 = q[..., 1]
        q3 = q[..., 2]
        q4 = q[..., 3]
        q6 = q[..., 5]
        z = z_e[..., 0]
        amp = self.K_R * (1.0 + self.K_tool_R * z / 0.20)
        wx = amp * self.A_x * torch.sin(q4)
        wy = amp * self.A_y * torch.cos(q2) * torch.sin(q6)
        wz = self.K_R * self.A_z * torch.cos(q3)
        return torch.stack([wx, wy, wz], dim=-1)

    def xi_true(self, q: Tensor, z_e: Tensor) -> Tensor:
        """Full body-frame twist ξ_true = (ρ_true, ω_true) ∈ R^6."""
        R_a, _ = self.fk_analytic_Rp(q, z_e)
        delta_w = self.delta_world(q, z_e)
        # Body-frame ρ_true = R_analytic^T · Δ_world (pre-residual frame).
        # Strictly, ξ_true is the body-frame twist *of the residual right-multiply*,
        # which translates the analytic frame by R_analytic·ρ_true.  So the
        # induced world-frame translation is R_a·ρ_true; we want this to equal
        # delta_w.  Hence ρ_true = R_a^T · delta_w.
        rho_true = (R_a.transpose(-1, -2) @ delta_w.unsqueeze(-1)).squeeze(-1)
        omega_true = self.omega_true(q, z_e)
        return torch.cat([rho_true, omega_true], dim=-1)

    def T_true_Rp(self, q: Tensor, z_e: Tensor) -> tuple[Tensor, Tensor]:
        """Ground-truth pose (R_true, p_true) = T_analytic · exp_SE(3)(ξ_true)."""
        R_a, p_a = self.fk_analytic_Rp(q, z_e)
        xi = self.xi_true(q, z_e)
        R_d, p_d = exp_SE3(xi)
        return compose_Rp(R_a, p_a, R_d, p_d)
