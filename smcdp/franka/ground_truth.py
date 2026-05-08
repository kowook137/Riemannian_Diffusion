"""Ground-truth Franka with simulated compliance / calibration error.

Idea_formulation §7.1, §15.1: the "true" robot deviates from analytic FK by an
unknown smooth function of (q, z_e).  Stage-1 trains Δ_φ to recover this; the
learned manifold M_φ then carries the score-net training in Stage 5+.

The compliance model is intentionally simple but physically motivated:
gravity-loaded shoulder/elbow sag + tool-length-dependent flex of the wrist.
It is NOT a ground-truth pin-down (real-world compliance is more complex);
it serves as a controllable target for the self-model fitting test.
"""
from __future__ import annotations

import logging

import torch
from torch import Tensor

import pytorch_kinematics as pk


class TrueFrankaCompliance:
    """Analytic FK + compliance perturbation, used as ground truth for Stage 1.

        p_true(q, z_e) = FK_analytic(q, z_e) + Δ_true(q, z_e)

    where Δ_true encodes:
      - gravity-induced shoulder sag: amplitude depends on cos(q_2) (lever arm in
        the horizontal plane) and tool length (1 + 4 z_e/0.2)
      - elbow droop modulated by sin(q_4)
      - small calibration offset (constant ≠ 0)

    All terms are smooth; magnitude is controlled by `K_grav`, `K_offset` so the
    Δ_true norm stays in the millimetre to centimetre range typical of soft-robot
    compliance / calibration errors (see Idea_formulation §10 C2).
    """

    def __init__(
        self,
        urdf_path: str,
        end_link: str = "panda_hand",
        K_grav: float = 0.025,
        K_offset: float = 0.005,
        K_tool: float = 0.20,                  # how strongly z_e amplifies sag
    ):
        # log suppression around pytorch_kinematics URDF parser
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
        self._chain_state = None

    def _ensure_chain(self, like: Tensor) -> None:
        target = (like.device, like.dtype)
        if self._chain_state != target:
            self.chain = self.chain.to(device=like.device, dtype=like.dtype)
            self._chain_state = target

    def fk_analytic(self, q: Tensor, z_e: Tensor) -> Tensor:
        batch_shape = q.shape[:-1]
        q_flat = q.reshape(-1, 7)
        z_flat = z_e.reshape(-1, 1)
        self._ensure_chain(q_flat)
        m = self.chain.forward_kinematics(q_flat).get_matrix()
        pos = m[..., :3, 3]
        R = m[..., :3, :3]
        offset_world = R[..., :, 2] * z_flat                     # tool tip = hand_pos + z_e R[:,:,2]
        return (pos + offset_world).reshape(*batch_shape, 3)

    def delta_true(self, q: Tensor, z_e: Tensor) -> Tensor:
        """Compliance perturbation Δ_true(q, z_e) ∈ R^3.

        Designed to be smooth and (q, z_e)-dependent so Stage-1 has a non-trivial
        target, but small enough that the analytic FK is a good initial guess.
        """
        # broadcasting: q (..., 7), z_e (..., 1) → (..., 3)
        q2 = q[..., 1]
        q4 = q[..., 3]
        z = z_e[..., 0]
        # gravity sag along world -z axis; lever arm proxy = cos(q_2) (shoulder pitch)
        # amplified by tool length (longer tool ⇒ more droop)
        amp = self.K_grav * (1.0 + self.K_tool * z / 0.20)
        sag_z = -amp * torch.cos(q2) * (0.5 + 0.5 * torch.cos(q4))      # negative (drops)
        # elbow-driven lateral drift along world +y
        drift_y = 0.5 * amp * torch.sin(q4)
        # small constant calibration bias along +x (mimics encoder offset)
        off_x = self.K_offset * torch.ones_like(q2)
        return torch.stack([off_x, drift_y, sag_z], dim=-1)

    def p_true(self, q: Tensor, z_e: Tensor) -> Tensor:
        return self.fk_analytic(q, z_e) + self.delta_true(q, z_e)
