"""Trajectory-level Riemannian SGM on the product manifold  M_φ(z_e)^{H+1}.

Implements Idea_formulation §3.4 (product manifold), §4.2 (component-wise
forward), §4.4 (per-timestep chart→ambient lift), §4.6 (sum-of-DSM loss),
§5.4 (per-component reverse GRW with retraction).

Mathematical contract (faithfully):

    Trajectory state :  τ = (x_0, x_1, …, x_H) ∈ M_φ^{H+1} ⊂ R^{(H+1)·d}
    Tangent space    :  T_τ M_φ^{H+1} = ⊕_h T_{x_h} M_φ                  (§3.4)
    Product metric   :  ⟨u, v⟩_τ = Σ_h ⟨u_h, v_h⟩_{x_h}                  (§3.4)

    Forward SDE      :  dX_{h,t} = b(X_{h,t}) dt + √β(t) dB^M_{h,t}      (§4.2)
                        — components iid given drift/diffusion, B^M_{h,t}'s independent
    Reverse SDE      :  dY_{h,τ} = (-b + β·s_h) dτ + √β dB̃^M_{h,τ}        (§4.2 + Thm 1)

    Loss (Varadhan)  :  ℒ = E_{r, τ_0, τ_r} [ Σ_h ‖s_h − Log(x_{h,r}, x_{h,0})/τ‖²_{x_{h,r}} ]   (§4.6)
                        with τ = ∫_0^r β(s)ds  (Brownian rescaled time).

The substrate code is a thin wrapper: trajectory-shape (B, H+1, d) tensors are
flattened to (B·(H+1), d) for the per-point manifold operations in `manifolds.py`
and `sde.py`, and reshaped back.  No new mathematics — just product-manifold
bookkeeping.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from smcdp.manifolds import GraphManifold, EmbodimentGraphManifold
from smcdp.sde import LangevinSDE
from smcdp.score_net import sinusoidal_time_embedding, _ACTIVATIONS


# ---------------------------------------------------------------------------
# Trajectory data distributions
# ---------------------------------------------------------------------------


class LinearChartTrajectoryDist:
    """Linear-interpolation chart trajectory data on a (non-embodiment) graph manifold.

    Each demo τ = (x_0, …, x_H) is parameterised by a random (q_start, q_end);
    in chart we set
        q_h = q_start + (h / H) · (q_end − q_start),    h = 0, …, H.
    The trajectory is then lifted to the manifold by H_φ per timestep,
        x_h = (q_h, F(q_h)) ∈ M_φ.
    The endpoints are sampled iid as q_⋆ ∼ N(μ_q, σ_endpoint² I) in chart.
    """

    def __init__(
        self,
        manifold: GraphManifold,
        H: int,
        mean_q,
        scale_endpoint: float = 0.3,
    ):
        self.manifold = manifold
        self.H = int(H)
        if isinstance(mean_q, list):
            mean_q = torch.tensor(mean_q)
        assert mean_q.numel() == manifold.n_q
        self.mean_q = mean_q
        self.scale_endpoint = float(scale_endpoint)

    def sample(self, n: int, device=None, dtype=torch.float32) -> Tensor:
        H1 = self.H + 1
        n_q = self.manifold.n_q
        mu = self.mean_q.to(device=device, dtype=dtype)
        q_start = mu + self.scale_endpoint * torch.randn(n, n_q, device=device, dtype=dtype)
        q_end = mu + self.scale_endpoint * torch.randn(n, n_q, device=device, dtype=dtype)
        s = torch.linspace(0.0, 1.0, H1, device=device, dtype=dtype).view(1, H1, 1)  # (1, H+1, 1)
        q_traj = q_start.unsqueeze(1) + s * (q_end - q_start).unsqueeze(1)             # (n, H+1, n_q)
        # lift each point — flatten and call H, then reshape
        q_flat = q_traj.reshape(-1, n_q)
        x_flat = self.manifold.H(q_flat)                                                # (n*(H+1), d)
        return x_flat.reshape(n, H1, self.manifold.ambient_dim)


# ---------------------------------------------------------------------------
# Bimodal redundant-IK trajectory data (Step C — kinematic-redundancy multi-modal)
# ---------------------------------------------------------------------------


def _two_link_ik(w_xy: Tensor, l1: float, l2: float, branch_up: Tensor):
    """2-link planar IK with explicit branch selection (elbow-up vs -down).

    Given a target wrist position w = (w_x, w_y) reachable by a 2-link arm with
    link lengths (l_1, l_2), returns one of the two IK solutions:
        elbow-up   :  q_2 > 0
        elbow-down :  q_2 < 0
    Both solutions share the same wrist position but cluster q_1, q_2 differently.

    w_xy      : (..., 2)
    branch_up : (...,) bool tensor — True ⇒ elbow-up
    Returns   : (q_1, q_2),  each (...)
    """
    r2 = (w_xy ** 2).sum(-1)
    cos_q2 = (r2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_q2 = cos_q2.clamp(-1.0, 1.0)
    q2_abs = torch.acos(cos_q2)                                              # (...,) ≥ 0
    q2 = torch.where(branch_up, q2_abs, -q2_abs)
    alpha = torch.atan2(w_xy[..., 1], w_xy[..., 0])
    sin_q2 = torch.sin(q2)
    cos_q2_eval = torch.cos(q2)
    beta = torch.atan2(l2 * sin_q2, l1 + l2 * cos_q2_eval)
    q1 = alpha - beta
    return q1, q2


class BimodalRedundantTrajectoryDist:
    """Bimodal demonstration on a 3-link planar arm via dual-branch 2-link IK.

    Each demo τ = (x_0, …, x_H) is built as:
      1. Sample end-effector trajectory:  p_start, p_end ∼ N(μ_⋆, jitter_p² I);
         linearly interpolate to get p_h, h = 0…H.
      2. Sample one IK branch ∈ {up, down} per trajectory (Bernoulli, P(up) configurable).
      3. Choose wrist position on the radial line through p, at distance |p| − l_3:
            w_h = p_h · (|p_h| − l_3) / |p_h|.
         Then 2-link IK to w_h with the chosen branch yields (q_1_h, q_2_h);
         the wrist angle s_3 = atan2(p_h_y, p_h_x), giving q_3_h = s_3 − q_1_h − q_2_h.
      4. Lift via H_φ.

    Both branches reach the SAME p-trajectory, so end-effector statistics are
    bimodally identical; only the chart-q distribution is bimodal — the
    canonical kinematic-redundancy multi-modal test.

    Returns from `sample()`:
        x         : (n, H+1, 5)  — trajectory tensor on M
        branch_up : (n,) bool    — ground-truth branch label (for mode-coverage eval)
    """

    def __init__(
        self,
        manifold,                     # NLinkPlanarArm with 3 links
        H: int,
        mu_p_start=(1.6, 0.6),
        mu_p_end=(1.6, -0.6),
        jitter_p: float = 0.05,
        branch_p_up: float = 0.5,
    ):
        # Sanity: 3-link planar arm
        assert getattr(manifold, "n_q", None) == 3 and getattr(manifold, "n_p", None) == 2
        assert hasattr(manifold, "link_lengths") and manifold.link_lengths.numel() == 3
        self.manifold = manifold
        self.H = int(H)
        self.mu_p_start = torch.as_tensor(mu_p_start, dtype=torch.float32)
        self.mu_p_end = torch.as_tensor(mu_p_end, dtype=torch.float32)
        self.jitter_p = float(jitter_p)
        self.branch_p_up = float(branch_p_up)

    def _ik_3link(self, p_traj: Tensor, branch_up_traj: Tensor) -> Tensor:
        """3-link branch-conditional IK along an end-effector trajectory.

        p_traj         : (n, H+1, 2)
        branch_up_traj : (n, H+1) bool
        Returns q_traj : (n, H+1, 3)
        """
        l1, l2, l3 = (float(x) for x in self.manifold.link_lengths.tolist())
        r_p = p_traj.norm(dim=-1, keepdim=True).clamp(min=1e-6)              # (n, H+1, 1)
        w = p_traj * ((r_p - l3) / r_p)                                      # (n, H+1, 2)
        q1, q2 = _two_link_ik(w, l1, l2, branch_up_traj)
        s3 = torch.atan2(p_traj[..., 1], p_traj[..., 0])                     # (n, H+1)
        q3 = s3 - q1 - q2
        return torch.stack([q1, q2, q3], dim=-1)

    def sample(self, n: int, device=None, dtype=torch.float32):
        H1 = self.H + 1
        mu_s = self.mu_p_start.to(device=device, dtype=dtype)
        mu_e = self.mu_p_end.to(device=device, dtype=dtype)
        p_start = mu_s + self.jitter_p * torch.randn(n, 2, device=device, dtype=dtype)
        p_end = mu_e + self.jitter_p * torch.randn(n, 2, device=device, dtype=dtype)
        s = torch.linspace(0.0, 1.0, H1, device=device, dtype=dtype).view(1, H1, 1)
        p_traj = p_start.unsqueeze(1) + s * (p_end - p_start).unsqueeze(1)   # (n, H+1, 2)

        # one branch per trajectory, broadcast across H+1
        branch_up = torch.rand(n, device=device) < self.branch_p_up           # (n,) bool
        branch_up_traj = branch_up.unsqueeze(1).expand(n, H1)                 # (n, H+1)
        q_traj = self._ik_3link(p_traj, branch_up_traj)                       # (n, H+1, 3)

        q_flat = q_traj.reshape(-1, 3)
        x_flat = self.manifold.H(q_flat)
        x = x_flat.reshape(n, H1, self.manifold.ambient_dim)
        return x, branch_up

    def sample_x(self, n: int, device=None, dtype=torch.float32) -> Tensor:
        """Match `sample(n) → x` API of other distributions (drops branch label)."""
        x, _ = self.sample(n, device=device, dtype=dtype)
        return x


class BimodalRedundantTrajectoryDistEmb:
    """Bimodal demonstration on a 3-link redundant arm WITH embodiment context z_e.

    Combines `BimodalRedundantTrajectoryDist` (kinematic-redundancy bimodal IK)
    with `EmbodimentNLinkPlanarArm` (z_e = last-link tool-extension).  Each demo
    samples one z_e ∈ [z_min, z_max] (frozen across the trajectory) and one
    branch ∈ {up, down}, then computes IK to a fixed end-effector trajectory
    using effective ℓ_3 = ℓ_3_base + z_e.

    The end-effector target trajectory is independent of z_e (the demonstrator
    aims for the same workspace target regardless of tool length); only the
    joint configurations adapt to z_e × branch.  Returned x carries z_e in
    its trailing slot so the EmbodimentGraphManifold operations work as-is.

    Sample(n) returns (x, branch_up, z_e):
        x         : (n, H+1, ambient_dim)
        branch_up : (n,) bool
        z_e       : (n, 1) per-trajectory embodiment value
    """

    def __init__(
        self,
        manifold,                       # EmbodimentNLinkPlanarArm with 3 links
        H: int,
        mu_p_start=(1.6, 0.6),
        mu_p_end=(1.6, -0.6),
        jitter_p: float = 0.05,
        z_e_range: tuple[float, float] = (0.0, 0.3),
        branch_p_up: float = 0.5,
    ):
        # Sanity: 3-link embodiment-aware planar arm
        assert getattr(manifold, "n_q", None) == 3 and getattr(manifold, "n_p", None) == 2
        assert getattr(manifold, "n_z", None) == 1
        assert hasattr(manifold, "link_lengths_base") and manifold.link_lengths_base.numel() == 3
        self.manifold = manifold
        self.H = int(H)
        self.mu_p_start = torch.as_tensor(mu_p_start, dtype=torch.float32)
        self.mu_p_end = torch.as_tensor(mu_p_end, dtype=torch.float32)
        self.jitter_p = float(jitter_p)
        self.z_lo, self.z_hi = z_e_range
        self.branch_p_up = float(branch_p_up)

    def _ik_3link_emb(
        self,
        p_traj: Tensor,           # (n, H+1, 2)
        branch_up_traj: Tensor,   # (n, H+1) bool
        z_e: Tensor,              # (n, 1) per trajectory
    ) -> Tensor:
        """3-link IK to end-effector p with branch + tool-length z_e."""
        l1 = float(self.manifold.link_lengths_base[0])
        l2 = float(self.manifold.link_lengths_base[1])
        l3_base = float(self.manifold.link_lengths_base[2])
        # effective tool length per trajectory; broadcast across timesteps
        H1 = p_traj.shape[1]
        l3_eff = (l3_base + z_e[..., 0]).unsqueeze(1).expand(-1, H1)            # (n, H+1)
        # wrist on the radial line through p, at distance |p| − ℓ_3_eff
        r_p = p_traj.norm(dim=-1, keepdim=True).clamp(min=1e-6)                  # (n, H+1, 1)
        w = p_traj * ((r_p - l3_eff.unsqueeze(-1)) / r_p)
        # 2-link IK to wrist (ℓ_1, ℓ_2 are unaffected by z_e)
        q1, q2 = _two_link_ik(w, l1, l2, branch_up_traj)
        # wrist angle = angle of p from origin (since w on radial line through p)
        s3 = torch.atan2(p_traj[..., 1], p_traj[..., 0])
        q3 = s3 - q1 - q2
        return torch.stack([q1, q2, q3], dim=-1)                                  # (n, H+1, 3)

    def sample(self, n: int, device=None, dtype=torch.float32):
        H1 = self.H + 1
        mu_s = self.mu_p_start.to(device=device, dtype=dtype)
        mu_e = self.mu_p_end.to(device=device, dtype=dtype)
        p_start = mu_s + self.jitter_p * torch.randn(n, 2, device=device, dtype=dtype)
        p_end = mu_e + self.jitter_p * torch.randn(n, 2, device=device, dtype=dtype)
        s = torch.linspace(0.0, 1.0, H1, device=device, dtype=dtype).view(1, H1, 1)
        p_traj = p_start.unsqueeze(1) + s * (p_end - p_start).unsqueeze(1)        # (n, H+1, 2)

        z_e = self.z_lo + (self.z_hi - self.z_lo) * torch.rand(
            n, self.manifold.n_z, device=device, dtype=dtype
        )                                                                          # (n, 1)
        branch_up = torch.rand(n, device=device) < self.branch_p_up                # (n,)
        branch_up_traj = branch_up.unsqueeze(1).expand(n, H1)                      # (n, H+1)

        q_traj = self._ik_3link_emb(p_traj, branch_up_traj, z_e)                   # (n, H+1, 3)

        # broadcast z_e across H+1 (frozen per trajectory) and lift via make_x
        z_traj = z_e.unsqueeze(1).expand(n, H1, self.manifold.n_z).contiguous()
        q_flat = q_traj.reshape(-1, 3)
        z_flat = z_traj.reshape(-1, self.manifold.n_z)
        x_flat = self.manifold.make_x(q_flat, z_flat)
        x = x_flat.reshape(n, H1, self.manifold.ambient_dim)
        return x, branch_up, z_e

    def sample_x(self, n: int, device=None, dtype=torch.float32) -> Tensor:
        x, _, _ = self.sample(n, device=device, dtype=dtype)
        return x


class LinearChartTrajectoryDistEmb:
    """Linear-interpolation chart trajectories on an EmbodimentGraphManifold.

    Each trajectory carries a single embodiment context z_e (frozen across H+1
    timesteps), reflecting the physical reality that the robot's hardware does
    not change during a single execution.
    """

    def __init__(
        self,
        manifold: EmbodimentGraphManifold,
        H: int,
        mean_q,
        scale_endpoint: float = 0.3,
        z_e_range: tuple[float, float] = (0.0, 0.3),
    ):
        self.manifold = manifold
        self.H = int(H)
        if isinstance(mean_q, list):
            mean_q = torch.tensor(mean_q)
        assert mean_q.numel() == manifold.n_q
        self.mean_q = mean_q
        self.scale_endpoint = float(scale_endpoint)
        self.z_lo, self.z_hi = z_e_range

    def sample(self, n: int, device=None, dtype=torch.float32, z_e=None) -> Tensor:
        H1 = self.H + 1
        n_q = self.manifold.n_q
        n_z = self.manifold.n_z
        mu = self.mean_q.to(device=device, dtype=dtype)
        q_start = mu + self.scale_endpoint * torch.randn(n, n_q, device=device, dtype=dtype)
        q_end = mu + self.scale_endpoint * torch.randn(n, n_q, device=device, dtype=dtype)
        s = torch.linspace(0.0, 1.0, H1, device=device, dtype=dtype).view(1, H1, 1)
        q_traj = q_start.unsqueeze(1) + s * (q_end - q_start).unsqueeze(1)            # (n, H+1, n_q)

        if z_e is None:
            z_e = self.z_lo + (self.z_hi - self.z_lo) * torch.rand(
                n, n_z, device=device, dtype=dtype
            )
        # broadcast z_e over H+1 (frozen per trajectory)
        z_traj = z_e.unsqueeze(1).expand(-1, H1, -1).contiguous()                      # (n, H+1, n_z)

        q_flat = q_traj.reshape(-1, n_q)
        z_flat = z_traj.reshape(-1, n_z)
        x_flat = self.manifold.make_x(q_flat, z_flat)                                  # (n*(H+1), d)
        return x_flat.reshape(n, H1, self.manifold.ambient_dim)


# ---------------------------------------------------------------------------
# Trajectory score net (flat-MLP coupling whole trajectory)
# ---------------------------------------------------------------------------


class TrajectoryScoreNet(nn.Module):
    """Score net  s_θ : (τ, t) ↦ section of T M_φ^{H+1}, in ambient form per timestep.

    Per Idea §4.2 (tail) and §4.4: the network may couple components to learn
    trajectory smoothness, but each component is lifted via the SAME chart→ambient
    map J_H(x_h).  Implementation:

        flatten     τ shape (B, H+1, d) → (B, (H+1)·d)
        condition   on time t (raw scalar concat or sinusoidal embed)
        MLP         → chart-coord output (B, (H+1)·n_q)
        unflatten   to (B, H+1, n_q)
        lift        per-timestep via manifold.lift_chart_to_tangent → (B, H+1, d)

    The lift carries the manifold's full geometric structure (J_F, embodiment z_e
    extracted from x) so coupling is purely *temporal*; geometry stays per-timestep.
    """

    def __init__(
        self,
        manifold,
        H: int,
        hidden: int = 512,
        n_layers: int = 5,
        t_embed_dim: int = 64,
        activation: str = "sin",
        time_embedding: str = "raw",
        final_init_scale: float = 1.0,
    ):
        super().__init__()
        self.manifold = manifold
        self.H = int(H)
        self.t_embed_dim = t_embed_dim
        self.time_embedding = time_embedding

        H1 = self.H + 1
        flat_dim = H1 * manifold.ambient_dim
        out_dim = H1 * manifold.intrinsic_dim

        if time_embedding == "raw":
            in_dim = flat_dim + 1
        elif time_embedding == "sinusoidal":
            in_dim = flat_dim + t_embed_dim
        else:
            raise ValueError(f"unknown time_embedding '{time_embedding}'")

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

    def forward(self, tau: Tensor, t: Tensor) -> Tensor:
        # τ: (B, H+1, d),  t: (B,)
        B, H1, d = tau.shape
        tau_flat = tau.reshape(B, H1 * d)
        if self.time_embedding == "raw":
            t_emb = t.unsqueeze(-1)                                          # (B, 1)
        else:
            t_emb = sinusoidal_time_embedding(t, self.t_embed_dim)
        h_input = torch.cat([tau_flat, t_emb], dim=-1)
        out_flat = self.net(h_input)                                         # (B, H1*n_q)
        s_chart = out_flat.reshape(B, H1, self.manifold.intrinsic_dim)

        # Per-timestep chart→ambient lift (uses each x_h's J_F and, for embodiment
        # manifolds, each x_h's frozen z_e — all extracted from `tau` by the
        # manifold's lift_chart_to_tangent).
        tau_pts = tau.reshape(B * H1, d)
        s_chart_pts = s_chart.reshape(B * H1, self.manifold.intrinsic_dim)
        s_amb_pts = self.manifold.lift_chart_to_tangent(tau_pts, s_chart_pts)
        return s_amb_pts.reshape(B, H1, d)


class TrajectoryScoreNetUNet(nn.Module):
    """ConditionalUnet1D-based trajectory score net (DP [Chi et al. 2023] arch in our framework).

    Replaces the flat-MLP TrajectoryScoreNet with the temporal-CNN architecture
    from Diffusion Policy.  The motivation (Idea_formulation §15.1, killer-
    experiment scale) is that 7-DoF + horizon ≥ 16 trajectories have rich
    temporal structure that flat MLPs lack the inductive bias for; CNN's
    locality + FiLM conditioning + time-axis translation equivariance match
    manipulation-trajectory smoothness.

    Plumbing (faithful to Idea §4.2 / §4.4 / §15.1):
        Input         :  q_traj ∈ R^{B × (H+1) × n_q}    (chart coords sliced from τ)
        Global cond   :  concat(z_e, goal_cond) ∈ R^{B × (n_z + goal_cond_dim)}
                          - z_e: frozen-per-trajectory tool offset (Embodiment manifolds)
                          - goal_cond: §15.1 goal-conditional input (e.g. p_target ∈ R^3)
        Diffusion t   :  scalar in [t0, tf], scaled by `t_scale` to match DP's
                          SinusoidalPosEmb frequency range.
        UNet output   :  chart-coord score s_q ∈ R^{B × (H+1) × n_q}
        Lift          :  per-timestep manifold.lift_chart_to_tangent → T_τ M^{H+1}

    Horizon constraint:  H+1 must be divisible by 2^(len(down_dims)-1).

    `goal_cond_dim=0` (default) reproduces the unconditional case.  When > 0,
    the score net's `forward(tau, t, goal_cond=...)` requires `goal_cond` of
    shape (B, goal_cond_dim).
    """

    def __init__(
        self,
        manifold,
        H: int,
        down_dims: tuple = (128, 256, 512),
        diffusion_step_embed_dim: int = 256,
        n_groups: int = 8,
        kernel_size: int = 3,
        cond_predict_scale: bool = False,
        t_scale: float = 1000.0,
        goal_cond_dim: int = 0,
        cond_injection: str = "global",   # "global" (default) or "channel" (Phase 6.1)
    ):
        super().__init__()
        # Lazy import: keeps the rest of the package independent of the
        # diffusion_policy package being importable.
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        self.manifold = manifold
        self.H = int(H)
        H1 = self.H + 1
        downsample_factor = 2 ** (len(down_dims) - 1)
        if H1 % downsample_factor != 0:
            raise ValueError(
                f"H+1 ({H1}) must be divisible by 2^(len(down_dims)-1) = "
                f"{downsample_factor}.  Either pad the horizon or reduce down_dims depth."
            )
        self.t_scale = float(t_scale)

        n_z = getattr(manifold, "n_z", 0)
        self.has_embodiment = bool(n_z > 0)
        self.goal_cond_dim = int(goal_cond_dim)
        if cond_injection not in ("global", "channel"):
            raise ValueError(f"cond_injection must be 'global' or 'channel', got {cond_injection}")
        self.cond_injection = cond_injection

        if self.cond_injection == "global":
            # Original: cond goes into global_cond, FiLM-modulates each ResBlock
            global_dim = (n_z if self.has_embodiment else 0) + self.goal_cond_dim
            input_dim_unet = manifold.n_q
        else:
            # Channel concat: broadcast cond across H+1 and concat as input channels
            # (Phase 6.1 — each timestep directly sees the target)
            global_dim = 0
            input_dim_unet = manifold.n_q + self.goal_cond_dim + (n_z if self.has_embodiment else 0)

        self.unet = ConditionalUnet1D(
            input_dim=input_dim_unet,
            global_cond_dim=(global_dim if global_dim > 0 else None),
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

    def forward(self, tau: Tensor, t: Tensor, goal_cond: Tensor | None = None) -> Tensor:
        # τ: (B, H+1, d),  t: (B,),  goal_cond: (B, goal_cond_dim) | None
        B, H1, d = tau.shape
        n_q = self.manifold.n_q
        n_p = self.manifold.n_p
        q_traj = tau[..., :n_q]                                                # (B, H+1, n_q)
        if self.has_embodiment:
            z_traj = tau[..., n_q + n_p:]                                       # (B, H+1, n_z)
            z_global = z_traj[:, 0, :]                                          # (B, n_z) — constant per traj
        else:
            z_traj = None
            z_global = None

        if self.goal_cond_dim > 0:
            if goal_cond is None:
                raise ValueError(
                    f"goal_cond_dim={self.goal_cond_dim} but goal_cond not provided"
                )
            if goal_cond.shape != (B, self.goal_cond_dim):
                raise ValueError(
                    f"goal_cond shape {tuple(goal_cond.shape)} != ({B}, {self.goal_cond_dim})"
                )

        t_scaled = t * self.t_scale                                             # (B,)

        if self.cond_injection == "global":
            if self.goal_cond_dim > 0:
                full_cond = (torch.cat([z_global, goal_cond], dim=-1)
                             if z_global is not None else goal_cond)
            else:
                full_cond = z_global
            s_chart = self.unet(q_traj, t_scaled, global_cond=full_cond)         # (B, H+1, n_q)
        else:
            # channel-concat injection: broadcast cond across H+1 and append as channels
            channel_inputs = [q_traj]
            if self.goal_cond_dim > 0:
                p_target_traj = goal_cond.unsqueeze(1).expand(-1, H1, -1)        # (B, H+1, goal_cond_dim)
                channel_inputs.append(p_target_traj)
            if self.has_embodiment:
                # z_traj already (B, H+1, n_z) — frozen across timesteps but
                # the channel layout is the same to keep the network input clean
                channel_inputs.append(z_traj)
            x_input = torch.cat(channel_inputs, dim=-1)                          # (B, H+1, input_dim_unet)
            out_full = self.unet(x_input, t_scaled, global_cond=None)            # (B, H+1, input_dim_unet)
            s_chart = out_full[..., :n_q]                                        # extract n_q channels as score

        # Per-timestep chart→ambient lift (uses each x_h's J_F and frozen z_e)
        tau_pts = tau.reshape(B * H1, d)
        s_chart_pts = s_chart.reshape(B * H1, n_q)
        s_amb_pts = self.manifold.lift_chart_to_tangent(tau_pts, s_chart_pts)
        return s_amb_pts.reshape(B, H1, d)


class TrajectoryScaledScoreFn(nn.Module):
    """std_trick + residual_trick wrapper for a trajectory score net.

    Applied per-timestep (broadcast over the H+1 axis):
        score = (net / σ(t)) + 2 · b_fwd / β(t)
    Identical to ScaledScoreFn for single points; just shape-aware.
    """

    def __init__(self, net: TrajectoryScoreNet, sde,
                 std_trick: bool = True, residual_trick: bool = True):
        super().__init__()
        self.net = net
        self.sde = sde
        self.std_trick = std_trick
        self.residual_trick = residual_trick

    def forward(self, tau: Tensor, t: Tensor, goal_cond: Tensor | None = None) -> Tensor:
        # `goal_cond` is forwarded to `self.net` if it has goal_cond_dim > 0;
        # ignored by older nets (TrajectoryScoreNet) since they don't accept it.
        try:
            out = self.net(tau, t, goal_cond=goal_cond)                        # (B, H+1, d)
        except TypeError:
            # Backward-compat: flat-MLP TrajectoryScoreNet forward(tau, t)
            out = self.net(tau, t)
        B, H1, d = tau.shape
        if self.std_trick:
            sigma = self.sde.schedule.proxy_std(t).clamp(min=1e-6).view(B, 1, 1)
            out = out / sigma
        if self.residual_trick:
            tau_flat = tau.reshape(B * H1, d)
            t_flat = t.unsqueeze(1).expand(B, H1).reshape(-1)
            b_fwd_flat = self.sde.drift(tau_flat, t_flat)
            b_fwd = b_fwd_flat.reshape(B, H1, d)
            beta = self.sde.schedule.beta(t).clamp(min=1e-12).view(B, 1, 1)
            out = out + 2.0 * b_fwd / beta
        return out


def _maybe_goal_cond(score_fn, tau, t, goal_cond):
    """Call score_fn forwarding goal_cond if it accepts the kwarg, else without."""
    try:
        return score_fn(tau, t, goal_cond=goal_cond)
    except TypeError:
        # Older score nets that don't accept goal_cond
        return score_fn(tau, t)


# ---------------------------------------------------------------------------
# Trajectory forward / reverse GRW (component-wise, vectorised)
# ---------------------------------------------------------------------------


@torch.no_grad()
def traj_forward_grw(
    sde: LangevinSDE,
    tau_0: Tensor,
    t_target: Tensor,
    n_steps: int,
) -> Tensor:
    """Forward GRW on the product manifold (Idea §4.2 component-wise).

    Each timestep is diffused independently with the SAME forward drift function
    (Langevin: -½ β ∇U ; Brownian: 0) and an INDEPENDENT Brownian on M.  All
    H+1 components share the same SDE time t_target per trajectory.
    """
    B, H1, d = tau_0.shape
    schedule = sde.schedule
    # broadcast t_target over the H+1 axis
    t_per_pt = t_target.unsqueeze(1).expand(B, H1).reshape(-1)               # (B*H1,)
    tau_flat = tau_0.reshape(B * H1, d)
    dt = (t_per_pt - schedule.t0) / n_steps                                  # (B*H1,)
    x = tau_flat
    for k in range(n_steps):
        t_k = schedule.t0 + k * dt
        b_fwd = sde.drift(x, t_k)
        sigma = sde.diffusion(t_k)
        z = sde.manifold.random_normal_tangent(x)
        v = b_fwd * dt.unsqueeze(-1) + sigma.unsqueeze(-1) * z * dt.abs().sqrt().unsqueeze(-1)
        x = sde.manifold.exp(x, v)
    return x.reshape(B, H1, d)


def _ee_anchor_guidance(
    tau: Tensor,
    p_anchor: Tensor,
    manifold,
    alpha: float,
    h_indices: list[int] | None,
) -> Tensor:
    """Compute Riemannian EE-anchor tangent at specified timesteps.

    Adds  $-\\alpha \\cdot G^{-1} \\partial_q R(q, p_\\text{anchor})$  lifted to ambient,
    where $R = \\tfrac12 \\|F(q, z_e) - p_\\text{anchor}\\|^2$.  Negates so the score
    moves q-component in the direction that DECREASES EE error to the anchor.
    Other timesteps are zero (we only constrain the requested h's).

    Used for both:
      - Endpoint goal-pull at h=H (with p_anchor = p_target)
      - Start anchor at h=0 (with p_anchor = p_start)

    Returns a tangent tensor of shape (B, H+1, d) to ADD to the model score.
    """
    B, H1, d = tau.shape
    n_q = manifold.n_q
    n_p = manifold.n_p
    if h_indices is None:
        h_indices = [H1 - 1]
    out = torch.zeros_like(tau)
    for h in h_indices:
        q_h = tau[:, h, :n_q]
        z_h = tau[:, h, n_q + n_p:]
        with torch.enable_grad():
            q_leaf = q_h.detach().clone().requires_grad_(True)
            p_h = manifold.F(q_leaf, z_h)
            err_sq = ((p_h - p_anchor) ** 2).sum(-1)                                 # (B,)
            chart_grad = torch.autograd.grad(err_sq.sum(), q_leaf,
                                             create_graph=False, retain_graph=False)[0]
        a = -alpha * chart_grad.detach()                                              # (B, n_q)
        L = manifold.G_chol(q_h, z_h)
        a_riem = torch.cholesky_solve(a.unsqueeze(-1), L).squeeze(-1)
        lift = manifold.lift_chart_to_tangent(tau[:, h, :], a_riem)
        out[:, h, :] = lift
    return out


def _goal_residual_guidance(tau, p_target, manifold, alpha, h_indices):
    """Backward-compat alias for endpoint-pull.  Use _ee_anchor_guidance directly."""
    return _ee_anchor_guidance(tau, p_target, manifold, alpha, h_indices)


def _smoothness_guidance(
    tau: Tensor,
    manifold,
    alpha_vel: float,
    alpha_acc: float,
) -> Tensor:
    """Trajectory-smoothness guidance via velocity / acceleration penalties.

    Penalty (negated; gradient of -R points in descent direction):
      R_vel(τ)  = ½ Σ_{h=0..H-1} ‖q_{h+1} − q_h‖²
      R_acc(τ)  = ½ Σ_{h=1..H-1} ‖q_{h+1} − 2 q_h + q_{h-1}‖²

    The gradient w.r.t. q at each timestep is computed analytically (cheap),
    converted to Riemannian via G^{-1}, and lifted to ambient.  Acts on q-block
    only; z-block stays 0 (frozen).

    Returns a tangent tensor of shape (B, H+1, d) to ADD to the model score.
    """
    B, H1, d = tau.shape
    n_q = manifold.n_q
    n_p = manifold.n_p
    q_traj = tau[..., :n_q]                                                          # (B, H+1, n_q)

    # ∂R_vel/∂q_h:
    #   for h=0:        −(q_1 − q_0)
    #   for 0<h<H:      −(q_{h+1} − q_h) + (q_h − q_{h-1}) = 2q_h − q_{h+1} − q_{h-1}
    #   for h=H:         (q_H − q_{H-1})
    grad_vel = torch.zeros_like(q_traj)
    grad_vel[:, 0, :]   = -(q_traj[:, 1, :] - q_traj[:, 0, :])
    grad_vel[:, -1, :]  =  (q_traj[:, -1, :] - q_traj[:, -2, :])
    if H1 > 2:
        grad_vel[:, 1:-1, :] = (
            2 * q_traj[:, 1:-1, :] - q_traj[:, 2:, :] - q_traj[:, :-2, :]
        )

    # ∂R_acc/∂q_h:  see standard finite-difference Hessian; compute via shifted differences
    grad_acc = torch.zeros_like(q_traj)
    if H1 >= 3:
        # Acc residuals r_h = q_{h+1} − 2 q_h + q_{h-1} for h ∈ [1, H-1]
        # ∂(½ Σ r_h²)/∂q_k = Σ_h ∂r_h/∂q_k · r_h
        # ∂r_h/∂q_k:  +1 if k=h+1, −2 if k=h, +1 if k=h-1, else 0
        r = q_traj[:, 2:, :] - 2 * q_traj[:, 1:-1, :] + q_traj[:, :-2, :]            # (B, H-1, n_q)
        # contribution from h = k-1 (i.e. r_{k-1}): coefficient +1 → grad += r_{k-1} for k≥1
        grad_acc[:, 1:-1, :] += r * (-2.0)                                            # k = h's
        if H1 > 2:
            grad_acc[:, 0, :]   += r[:, 0, :] * (+1.0)                                # k = h-1 for h=1
            grad_acc[:, -1, :]  += r[:, -1, :] * (+1.0)                               # k = h+1 for h=H-1
            grad_acc[:, 2:, :]  += r * (+1.0)                                         # k = h+1 for general
            grad_acc[:, :-2, :] += r * (+1.0)                                         # k = h-1 for general

    chart_grad = alpha_vel * grad_vel + alpha_acc * grad_acc                          # (B, H+1, n_q)
    a = -chart_grad                                                                    # descent

    # Convert to Riemannian per timestep
    out = torch.zeros_like(tau)
    if alpha_vel == 0.0 and alpha_acc == 0.0:
        return out
    q_flat = q_traj.reshape(-1, n_q)
    z_flat = tau[..., n_q + n_p:].reshape(-1, manifold.n_z)
    L = manifold.G_chol(q_flat, z_flat)                                                # (B*H1, n_q, n_q)
    a_flat = a.reshape(-1, n_q).unsqueeze(-1)
    a_riem = torch.cholesky_solve(a_flat, L).squeeze(-1).reshape(B, H1, n_q)
    # Lift each timestep
    tau_pts = tau.reshape(-1, d)
    lift_pts = manifold.lift_chart_to_tangent(tau_pts, a_riem.reshape(-1, n_q))
    out = lift_pts.reshape(B, H1, d)
    return out


@torch.no_grad()
def traj_reverse_grw(
    sde: LangevinSDE,
    score_fn,                           # callable (τ: (B, H+1, d), t: (B,), goal_cond=None) -> (B, H+1, d) ∈ T_τ M^{H+1}
    tau_T: Tensor,
    n_steps: int,
    eps: float = 1e-3,
    return_history: bool = False,
    goal_cond: Tensor | None = None,
    guidance_scale: float = 0.0,        # CFG: 0 = cond only, w > 0 = (1+w)·cond − w·uncond
    goal_residual_alpha: float = 0.0,   # Phase 5.3: explicit goal-pulling at sampling time
    goal_residual_h: list[int] | None = None,
    p_start: Tensor | None = None,      # (B, 3) for start-anchor guidance
    start_anchor_alpha: float = 0.0,
    start_anchor_h: list[int] | None = None,
    smoothness_alpha_vel: float = 0.0,  # quadratic vel penalty across all h
    smoothness_alpha_acc: float = 0.0,  # quadratic acc penalty across all h
):
    """Reverse-time GRW on the product manifold (Idea §5.4).

    `goal_cond` (if provided, shape (B, goal_cond_dim)) is forwarded to the
    score function each step — used for §15.1 goal-conditional sampling.

    `guidance_scale` (CFG, Ho & Salimans 2022): if > 0, two score evaluations
    per step:  score_cond  (with goal_cond)  and  score_uncond  (zeros cond).
    Final score = (1 + w) * score_cond − w * score_uncond.  Both lie in the
    tangent bundle, so the linear combination remains tangent.

    `goal_residual_alpha` (Phase 5.3 of diagnostic_plan): if > 0, adds
    `-alpha · G⁻¹ · ∂_q ½‖F(q_h, z_e) − p_target‖²` lifted to ambient at every
    reverse step.  This is an explicit Riemannian goal-pulling term that
    bypasses any weakness in the learned conditioning by using the analytic
    relationship between q and end-EE.  Applied at timesteps in
    `goal_residual_h` (default: only the last timestep H, the trajectory end).
    """
    schedule = sde.schedule
    B, H1, d = tau_T.shape
    device, dtype = tau_T.device, tau_T.dtype

    ts = torch.linspace(schedule.tf, schedule.t0 + eps,
                        n_steps + 1, device=device, dtype=dtype)
    history = [] if return_history else None
    tau = tau_T
    for k in range(n_steps):
        t_k = ts[k].expand(B)                                                # (B,)
        dtau = ts[k] - ts[k + 1]                                             # > 0 (scalar)
        beta = schedule.beta(t_k)                                            # (B,)

        # forward drift per-timestep (flatten/unflatten)
        tau_flat = tau.reshape(B * H1, d)
        t_flat = t_k.unsqueeze(1).expand(B, H1).reshape(-1)
        b_fwd_flat = sde.drift(tau_flat, t_flat)
        b_fwd = b_fwd_flat.reshape(B, H1, d)

        # score (per-timestep, possibly with cross-time coupling internally)
        if goal_cond is not None and guidance_scale != 0.0:
            score_cond = _maybe_goal_cond(score_fn, tau, t_k, goal_cond)
            null_cond = torch.zeros_like(goal_cond)
            score_uncond = _maybe_goal_cond(score_fn, tau, t_k, null_cond)
            score = (1.0 + guidance_scale) * score_cond - guidance_scale * score_uncond
        else:
            score = _maybe_goal_cond(score_fn, tau, t_k, goal_cond)          # (B, H+1, d)

        # explicit Riemannian endpoint-pull guidance (Phase 5.3) — uses goal_cond
        # as analytic target IF its first 3 dims are p_target (default convention).
        # Caller must ensure goal_cond[..., :3] == p_target.
        if goal_residual_alpha > 0.0 and goal_cond is not None:
            p_target_an = goal_cond[..., :3]
            score = score + _ee_anchor_guidance(
                tau, p_target_an, sde.manifold,
                alpha=goal_residual_alpha,
                h_indices=goal_residual_h,
            )

        # start-anchor guidance — pulls q_0 such that F(q_0, z_e) ≈ p_start
        if start_anchor_alpha > 0.0 and p_start is not None:
            sa_h = start_anchor_h if start_anchor_h is not None else [0]
            score = score + _ee_anchor_guidance(
                tau, p_start, sde.manifold,
                alpha=start_anchor_alpha,
                h_indices=sa_h,
            )

        # smoothness guidance — vel/acc penalty across all timesteps
        if smoothness_alpha_vel > 0.0 or smoothness_alpha_acc > 0.0:
            score = score + _smoothness_guidance(
                tau, sde.manifold,
                alpha_vel=smoothness_alpha_vel,
                alpha_acc=smoothness_alpha_acc,
            )

        # tangent noise per-timestep
        z_flat = sde.manifold.random_normal_tangent(tau_flat)
        z = z_flat.reshape(B, H1, d)

        beta_b = beta.view(B, 1, 1)
        reverse_drift = -b_fwd + beta_b * score
        diffusion = beta_b.sqrt() * z

        v = reverse_drift * dtau + diffusion * dtau.sqrt()                   # (B, H+1, d)
        v_flat = v.reshape(B * H1, d)
        tau = sde.manifold.exp(tau_flat, v_flat).reshape(B, H1, d)

        if return_history:
            history.append(tau)

    if return_history:
        return tau, torch.stack(history, dim=0), ts
    return tau


# ---------------------------------------------------------------------------
# Trajectory DSM-Varadhan loss
# ---------------------------------------------------------------------------


def traj_dsm_varadhan_loss(
    score_fn,                           # callable (τ, t, goal_cond=None) -> (B, H+1, d) ∈ T_τ M^{H+1}
    sde: LangevinSDE,
    tau_0: Tensor,
    eps: float = 1e-3,
    weight: str = "sigma2",
    n_grw_steps: int = 20,
    goal_cond: Tensor | None = None,
    cond_drop_prob: float = 0.0,        # CFG: probability to replace goal_cond with null (zeros)
    endpoint_weight: float = 1.0,       # Phase 5.2: extra weight on h=H timestep (1.0 = uniform)
) -> Tensor:
    """ℒ_traj = E_{r, τ_0, τ_r} [ w(r) · Σ_h ‖s_h − Log(x_{h,r}, x_{h,0})/τ‖²_{x_{h,r}} ].

    `τ` here is the Brownian rescaled time τ(r) = ∫_0^r β(s) ds  (RSGM `varhadan_exp`
    convention, faithful to §4.6).  Per-timestep ‖·‖² uses the manifold's metric
    (ambient = G-weighted in chart for `riemannian` mode; chart-Eucl for the toy
    approximation mode).  Time-weighting w(r) is applied to the SUMMED squared
    error — equivalent to applying it per-timestep before summation.
    """
    B, H1, d = tau_0.shape
    device, dtype = tau_0.device, tau_0.dtype
    schedule = sde.schedule

    r = eps + (schedule.tf - eps) * torch.rand(B, device=device, dtype=dtype)
    tau_r = traj_forward_grw(sde, tau_0, r, n_grw_steps)

    # CFG cond dropout: with prob `cond_drop_prob` per sample, replace goal_cond
    # with a null (zeros) token so the score net learns BOTH p(τ|cond) and
    # p(τ|null) jointly.  At sampling time, guidance scales the cond/uncond
    # difference (Ho & Salimans 2022, https://arxiv.org/abs/2207.12598).
    if goal_cond is not None and cond_drop_prob > 0.0:
        drop_mask = torch.rand(B, device=device) < cond_drop_prob              # (B,)
        if drop_mask.any():
            goal_cond = goal_cond.clone()
            goal_cond[drop_mask] = 0.0

    # per-timestep Varadhan target  (Log/τ in Brownian rescaled time)
    tau_brown = schedule.integral(r).clamp(min=1e-12).view(B, 1, 1)
    tau_r_flat = tau_r.reshape(B * H1, d)
    tau_0_flat = tau_0.reshape(B * H1, d)
    log_flat = sde.manifold.log(tau_r_flat, tau_0_flat)
    target = log_flat.reshape(B, H1, d) / tau_brown

    # score (B, H+1, d)
    score = _maybe_goal_cond(score_fn, tau_r, r, goal_cond)
    diff = score - target

    # per-timestep squared norm, summed over H+1
    sq_per_pt = sde.manifold.squared_norm(tau_r_flat, diff.reshape(B * H1, d))   # (B*H1,)
    sq_per_pt_h = sq_per_pt.reshape(B, H1)                                        # (B, H+1)

    # Phase 5.2: per-timestep weight w_h with endpoint emphasis
    if endpoint_weight != 1.0:
        h_weights = torch.ones(H1, device=device, dtype=dtype)
        h_weights[-1] = float(endpoint_weight)
        sq_per_traj = (sq_per_pt_h * h_weights).sum(-1)                           # (B,)
    else:
        sq_per_traj = sq_per_pt_h.sum(-1)                                          # (B,)

    if weight == "sigma2":
        w = schedule.proxy_std(r) ** 2
    elif weight == "beta":
        w = schedule.beta(r)
    elif weight == "none":
        w = torch.ones_like(r)
    else:
        raise ValueError(f"unknown weight '{weight}'")
    return (w * sq_per_traj).mean()
