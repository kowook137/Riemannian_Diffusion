"""Ground-truth 2-link arm with synthetic compliance.

Used ONLY for data generation and final evaluation (capture quality of Δ_φ).
The score-based model and the manifold class never see Δ_true directly — they
only have access to (q, p_true, z_e) tuples sampled by self-exploration.

Compliance model (synthetic but reasonable for a compliant tool extension):

  p_true(q, z_e) = FK_analytic(q, l_2 + z_e) + Δ_true(q, z_e)

  Δ_true(q, z_e) = K_grav · z_e² · sin(q_1 + q_2) · n̂_perp(q_1 + q_2)
                 + K_offset · z_e · n̂_along(q_1 + q_2)

where n̂_along = (cos α, sin α) along the tool axis (α = q_1 + q_2),
      n̂_perp  = (−sin α, cos α) perpendicular to it,
      K_grav   = bend coefficient (gravity-induced sag, scales with z_e²),
      K_offset = small along-axis stretch (calibration drift, scales with z_e).

This is the kind of effect analytic FK CANNOT capture and that the residual
Δ_φ must learn from data.
"""
from __future__ import annotations

import torch
from torch import Tensor


class TrueArmCompliance:
    """Synthetic ground-truth FK with a closed-form compliance Δ_true(q, z_e).

    Defines `p_true(q, z_e)`: the true end-effector position (the data
    generator's only output for self-exploration).  Subclasses are not expected
    — this is pure synthetic data generation.
    """

    def __init__(
        self,
        l1: float = 1.0,
        l2_base: float = 1.0,
        K_grav: float = 0.30,            # tool sag coeff  (units: 1/length)
        K_offset: float = 0.05,          # along-axis stretch coeff
    ):
        self.l1 = float(l1)
        self.l2_base = float(l2_base)
        self.K_grav = float(K_grav)
        self.K_offset = float(K_offset)

    # ---- analytic FK with extended tool length ----
    def fk_analytic(self, q: Tensor, z_e: Tensor) -> Tensor:
        q1 = q[..., 0]
        q2 = q[..., 1]
        l2_eff = self.l2_base + z_e[..., 0]
        c1 = torch.cos(q1)
        s1 = torch.sin(q1)
        c12 = torch.cos(q1 + q2)
        s12 = torch.sin(q1 + q2)
        px = self.l1 * c1 + l2_eff * c12
        py = self.l1 * s1 + l2_eff * s12
        return torch.stack([px, py], dim=-1)

    # ---- synthetic compliance residual ----
    def delta_true(self, q: Tensor, z_e: Tensor) -> Tensor:
        alpha = q[..., 0] + q[..., 1]
        c_a, s_a = torch.cos(alpha), torch.sin(alpha)
        z = z_e[..., 0]
        # gravity-like sag: amplitude K_grav · z² · sin(alpha) · n̂_perp
        sag_mag = self.K_grav * (z ** 2) * torch.sin(alpha)
        sag = torch.stack([-s_a * sag_mag, c_a * sag_mag], dim=-1)
        # along-axis offset: amplitude K_offset · z · n̂_along
        offset_mag = self.K_offset * z
        offset = torch.stack([c_a * offset_mag, s_a * offset_mag], dim=-1)
        return sag + offset

    def p_true(self, q: Tensor, z_e: Tensor) -> Tensor:
        return self.fk_analytic(q, z_e) + self.delta_true(q, z_e)


class TrueNLinkArmCompliance:
    """N-link ground-truth FK with synthetic compliance (generalises TrueArmCompliance).

    z_e ∈ R extends ONLY the last link's length:  ℓ_N_eff = ℓ_N_base + z_e.
    The compliance form depends on the end-effector orientation
        α = q_1 + q_2 + … + q_N    (cumulative sum of all joint angles),
    consistent with the 2-link special case (α = q_1 + q_2 there).

    Δ_true(q, z_e) = K_grav · z_e² · sin(α) · n̂_perp(α)
                   + K_offset · z_e · n̂_along(α)
    with the same n̂_along = (cos α, sin α), n̂_perp = (−sin α, cos α) as the
    2-link case.  The same compliance "shape" — sag perpendicular to the tool
    axis + along-axis stretch — applied at the end of an N-link chain.

    Used for ground-truth data generation and Stage-1 evaluation; never used
    by the model code (the model only sees the (q, p_true, z_e) triples).
    """

    def __init__(
        self,
        link_lengths_base,
        K_grav: float = 0.30,
        K_offset: float = 0.05,
    ):
        link_lengths_base = list(link_lengths_base)
        if len(link_lengths_base) < 2:
            raise ValueError("link_lengths_base must have length ≥ 2")
        self.link_lengths_base = link_lengths_base
        self.n_q = len(link_lengths_base)
        self.K_grav = float(K_grav)
        self.K_offset = float(K_offset)

    @property
    def l1(self) -> float:
        return self.link_lengths_base[0]

    @property
    def l2_base(self) -> float:
        # last-link base length (called l2_base for backward-compat with 2-link Stage-1 ckpts)
        return self.link_lengths_base[-1]

    def _link_lengths_eff(self, q: Tensor, z_e: Tensor) -> Tensor:
        n_q = q.shape[-1]
        l_base = torch.as_tensor(self.link_lengths_base,
                                 device=q.device, dtype=q.dtype)
        l_kept = l_base[: n_q - 1].expand(*q.shape[:-1], n_q - 1)
        l_last = (l_base[n_q - 1] + z_e[..., 0]).unsqueeze(-1)
        return torch.cat([l_kept, l_last], dim=-1)

    def fk_analytic(self, q: Tensor, z_e: Tensor) -> Tensor:
        l_eff = self._link_lengths_eff(q, z_e)
        s = torch.cumsum(q, dim=-1)
        px = (l_eff * torch.cos(s)).sum(-1)
        py = (l_eff * torch.sin(s)).sum(-1)
        return torch.stack([px, py], dim=-1)

    def delta_true(self, q: Tensor, z_e: Tensor) -> Tensor:
        alpha = q.sum(dim=-1)                                   # end-effector orientation
        c_a, s_a = torch.cos(alpha), torch.sin(alpha)
        z = z_e[..., 0]
        sag_mag = self.K_grav * (z ** 2) * torch.sin(alpha)
        sag = torch.stack([-s_a * sag_mag, c_a * sag_mag], dim=-1)
        offset_mag = self.K_offset * z
        offset = torch.stack([c_a * offset_mag, s_a * offset_mag], dim=-1)
        return sag + offset

    def p_true(self, q: Tensor, z_e: Tensor) -> Tensor:
        return self.fk_analytic(q, z_e) + self.delta_true(q, z_e)


def generate_self_exploration_dataset(
    n: int,
    truth: TrueArmCompliance,
    z_e_range: tuple[float, float] = (0.0, 0.3),
    q_box: tuple[float, float] = (-1.2, 1.2),
    seed: int = 0,
    device=None,
    dtype=torch.float32,
) -> dict:
    """Sample (q_i, p_true_i, z_e_i) by uniform exploration.

    Returns a dict with tensors  q (n, 2), z_e (n, 1), p_true (n, 2).
    Used for Stage-1 (self-model) training data.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    q_lo, q_hi = q_box
    z_lo, z_hi = z_e_range
    q = q_lo + (q_hi - q_lo) * torch.rand(n, 2, generator=g, dtype=dtype)
    z_e = z_lo + (z_hi - z_lo) * torch.rand(n, 1, generator=g, dtype=dtype)
    if device is not None:
        q = q.to(device); z_e = z_e.to(device)
    p_true = truth.p_true(q, z_e)
    return {"q": q, "z_e": z_e, "p_true": p_true}
