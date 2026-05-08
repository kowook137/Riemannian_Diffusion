"""Published baselines for the §15.2 killer-experiment comparison.

Each baseline follows its canonical published formulation as faithfully as
possible at the scope of our 3-link toy (no real-image observations).

References
----------
[Pomerleau89]   Pomerleau, "ALVINN: An Autonomous Land Vehicle In a Neural Network",
                NIPS 1989.   — Behavior Cloning origin.
[Florence22]    Florence et al. "Implicit Behavioral Cloning", CoRL 2022.
                — multi-modal-aware BC; we use the simpler explicit form (mode-
                  collapse exposure is the goal of this baseline).
[Ho20]          Ho, Jain & Abbeel, "Denoising Diffusion Probabilistic Models",
                NeurIPS 2020.   — DDPM forward / reverse, ε-prediction, MSE loss.
[Nichol21]      Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic
                Models", ICML 2021.   — cosine β-schedule.
[Chi23]         Chi et al. "Diffusion Policy: Visuomotor Policy Learning via
                Action Diffusion", RSS 2023.   — DDPM on action chunks (= our
                trajectory τ); receding-horizon control conditioned on
                observations.  We thread `z_e` as the conditioning observation.
[Christopher24] Christopher et al. "Projected Generative Diffusion Models for
                Constraint Satisfaction", NeurIPS 2024.   — ambient generative
                diffusion + per-step projection onto a constraint manifold.

Implementation notes
--------------------
* DDPM uses a *discrete* T-step Markov chain with αₜ, β̄ₜ.  The score-SDE form
  (Song21) is mathematically equivalent.  We use the score-SDE/Yang-Song
  formulation throughout for a single training-time codepath; the resulting
  policy is the diffusion-policy baseline of [Chi23] up to a continuous-time
  reformulation.
* ε-prediction (DDPM convention) and score-prediction are related by
  s = − ε / σ(t).  We train the network to predict ε directly (so it matches
  [Ho20] / [Chi23]) and convert to score for the reverse step.
* Architecture: [Chi23] uses a 1D-temporal CNN UNet for long action chunks
  (H ≥ 16).  For our short H+1 = 9 chunks we keep the same flat-MLP backbone
  used by all our other models, so the comparison isolates the diffusion-vs-
  manifold-Riemannian effect from architectural-bias differences.  If the
  reviewer wants the canonical 1D-CNN backbone, swap `_FlatTrajMLP` for one;
  the rest of the pipeline is invariant.
* β-schedule: linear (matches DDPM/Diffusion Policy default; cosine optional).
* Sampling: reverse SDE Euler–Maruyama (equivalent to DDPM ancestral sampling
  in expectation; DDIM is the deterministic limit).
* Projected baseline: ambient (q, p) state, ε-prediction in ambient, project p
  onto the LEARNED manifold M_φ(z_e) by replacing p ← F_φ(q, z_e) after every
  reverse step (per-[Christopher24]).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from smcdp.score_net import sinusoidal_time_embedding, _ACTIVATIONS
from smcdp.sde import LinearBetaSchedule


# ===========================================================================
# Backbone — flat-MLP, intentionally same as our manifold-aware score net so
# that the diffusion-vs-manifold comparison isolates the framework difference,
# not architectural bias.   See module docstring.
# ===========================================================================


class _FlatTrajMLP(nn.Module):
    """Flat-MLP score / ε predictor for trajectories.

    Input  τ ∈ R^{B × (H+1) × d_state},  t ∈ R^B,  ctx ∈ R^{B × n_ctx} optional
    Output prediction tensor of the same shape as τ (interpreted as ε in
    [Ho20]/[Chi23] convention).
    """

    def __init__(
        self,
        d_state: int,
        H: int,
        hidden: int = 512,
        n_layers: int = 5,
        t_embed_dim: int = 64,
        activation: str = "sin",
        time_embedding: str = "raw",
        ctx_dim: int = 0,
        final_init_scale: float = 1.0,
    ):
        super().__init__()
        self.d_state = d_state
        self.H = int(H)
        self.t_embed_dim = t_embed_dim
        self.time_embedding = time_embedding
        self.ctx_dim = ctx_dim

        H1 = self.H + 1
        flat_dim = H1 * d_state
        out_dim = H1 * d_state

        if time_embedding == "raw":
            in_dim = flat_dim + 1 + ctx_dim
        elif time_embedding == "sinusoidal":
            in_dim = flat_dim + t_embed_dim + ctx_dim
        else:
            raise ValueError(time_embedding)

        if activation not in _ACTIVATIONS:
            raise ValueError(activation)
        act = _ACTIVATIONS[activation]

        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden))
            layers.append(act())
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        if final_init_scale != 1.0:
            with torch.no_grad():
                self.net[-1].weight.mul_(final_init_scale)
                self.net[-1].bias.zero_()

    def forward(self, tau: Tensor, t: Tensor, ctx: Tensor | None = None) -> Tensor:
        B, H1, d = tau.shape
        flat = tau.reshape(B, H1 * d)
        if self.time_embedding == "raw":
            t_emb = t.unsqueeze(-1)
        else:
            t_emb = sinusoidal_time_embedding(t, self.t_embed_dim)
        feats = [flat, t_emb]
        if self.ctx_dim > 0:
            assert ctx is not None and ctx.shape[-1] == self.ctx_dim
            feats.append(ctx)
        h = torch.cat(feats, dim=-1)
        out = self.net(h)
        return out.reshape(B, H1, d)


# ===========================================================================
# Diffusion Policy [Chi23] — DDPM/score-SDE on (chart-Eucl) trajectories
#
# Standard DDPM ε-prediction in the Yang-Song-equivalent score-SDE form:
#   forward (chart-Eucl Brownian, like a VP-SDE with linear β):
#       q_t = q_0 + σ(t) · z,              z ∼ N(0, I)
#   training loss (ε-prediction, [Ho20] eq.14 simplified):
#       ℓ(θ) = E_{t, q_0, z}  ‖ ε_θ(q_t, t, ctx) − z ‖²
#       (re-weighted by σ²(t) to give a likelihood-aligned objective; this
#        matches RSGM `like_w=False` exactly so the comparison is consistent.)
#   sampling (reverse SDE Euler–Maruyama; DDIM is the deterministic limit):
#       q_{t-dτ} = q_t + β(t)·s_θ(q_t, t)·dτ + √(β(t)·dτ) · z̃,  s_θ = − ε_θ / σ(t).
# ===========================================================================


def diffusion_policy_marginal(
    q_0: Tensor, t: Tensor, schedule: LinearBetaSchedule
) -> tuple[Tensor, Tensor, Tensor]:
    """Forward marginal q_t and the ground-truth noise z used for ε-prediction.

    q_t = q_0 + σ(t) · z,    z ∼ N(0, I).
    Returns (q_t, z, σ(t))   so the trainer can construct the ε-pred loss.
    """
    sigma = schedule.proxy_std(t)
    z = torch.randn_like(q_0)
    sigma_b = sigma.view(-1, *([1] * (q_0.ndim - 1)))
    return q_0 + sigma_b * z, z, sigma


def diffusion_policy_loss(
    eps_net: _FlatTrajMLP,
    schedule: LinearBetaSchedule,
    q_0: Tensor,
    eps: float = 1e-3,
    ctx: Tensor | None = None,
    weight: str = "sigma2",
) -> Tensor:
    """ε-prediction MSE loss matching [Ho20]/[Chi23] in continuous-time form.

    weight :  'sigma2' (default; aligns with RSGM `like_w=False` and DDPM
                ELBO-aligned weighting), 'beta', or 'none'.
    """
    B = q_0.shape[0]
    device, dtype = q_0.device, q_0.dtype
    t = eps + (schedule.tf - eps) * torch.rand(B, device=device, dtype=dtype)
    q_t, z, sigma = diffusion_policy_marginal(q_0, t, schedule)
    eps_pred = eps_net(q_t, t, ctx)
    diff = eps_pred - z                                                       # ‖ε_θ − z‖²
    sq = (diff * diff).sum(dim=tuple(range(1, q_0.ndim)))
    if weight == "sigma2":
        w = sigma ** 2
    elif weight == "beta":
        w = schedule.beta(t)
    elif weight == "none":
        w = torch.ones_like(t)
    else:
        raise ValueError(weight)
    return (w * sq).mean()


@torch.no_grad()
def diffusion_policy_reverse(
    eps_net: _FlatTrajMLP,
    schedule: LinearBetaSchedule,
    q_T: Tensor,
    n_steps: int,
    eps: float = 1e-3,
    ctx: Tensor | None = None,
) -> Tensor:
    """Reverse-time SDE Euler–Maruyama.  Convert ε-pred to score: s = −ε/σ."""
    device, dtype = q_T.device, q_T.dtype
    ts = torch.linspace(schedule.tf, schedule.t0 + eps,
                        n_steps + 1, device=device, dtype=dtype)
    q = q_T
    for k in range(n_steps):
        t_k = ts[k].expand(q.shape[0])
        dtau = ts[k] - ts[k + 1]
        beta = schedule.beta(t_k).view(-1, *([1] * (q.ndim - 1)))
        sigma = schedule.proxy_std(t_k).clamp(min=1e-6).view(-1, *([1] * (q.ndim - 1)))
        eps_pred = eps_net(q, t_k, ctx)
        score = -eps_pred / sigma
        z = torch.randn_like(q)
        drift = beta * score
        diffusion = beta.sqrt() * z
        q = q + drift * dtau + diffusion * dtau.sqrt()
    return q


def diffusion_policy_prior(
    n: int, H: int, n_q: int, device, dtype=torch.float32, scale: float = 1.0
) -> Tensor:
    """Standard-normal prior in chart-Euclidean.  N(0, I); a wider `scale` is
    fine because the reverse SDE will denoise it."""
    return scale * torch.randn(n, H + 1, n_q, device=device, dtype=dtype)


# ===========================================================================
# Projected Diffusion [Christopher24] / SafeDiffuser style
#
# Same DDPM/score-SDE in AMBIENT (q, p) space; after every reverse step the
# state is projected onto the learned manifold M_φ(z_e) by replacing p with
# F_φ(q, z_e).  This realises "ambient generation + post-hoc projection" — the
# pattern of [Christopher24], with our learned g_φ as the constraint set.
# ===========================================================================


def projected_loss(
    eps_net: _FlatTrajMLP,
    schedule: LinearBetaSchedule,
    x_0: Tensor,                              # ambient demos (B, H+1, n_q+n_p)
    eps: float = 1e-3,
    ctx: Tensor | None = None,
    weight: str = "sigma2",
) -> Tensor:
    """Same ε-prediction loss as Diffusion Policy but in ambient (q, p) space.

    [Christopher24] applies projection only at sampling time, not training.
    Training is unconstrained ambient diffusion on the demo's ambient states.
    """
    return diffusion_policy_loss(eps_net, schedule, x_0, eps=eps, ctx=ctx, weight=weight)


@torch.no_grad()
def projected_reverse(
    eps_net: _FlatTrajMLP,
    schedule: LinearBetaSchedule,
    x_T: Tensor,
    n_steps: int,
    eps: float = 1e-3,
    ctx: Tensor | None = None,
    project_fn=None,                          # callable (x_amb, ctx) → projected x_amb
) -> Tensor:
    """Reverse SDE in ambient + per-step projection ([Christopher24] §3)."""
    device, dtype = x_T.device, x_T.dtype
    ts = torch.linspace(schedule.tf, schedule.t0 + eps,
                        n_steps + 1, device=device, dtype=dtype)
    x = x_T
    for k in range(n_steps):
        t_k = ts[k].expand(x.shape[0])
        dtau = ts[k] - ts[k + 1]
        beta = schedule.beta(t_k).view(-1, *([1] * (x.ndim - 1)))
        sigma = schedule.proxy_std(t_k).clamp(min=1e-6).view(-1, *([1] * (x.ndim - 1)))
        eps_pred = eps_net(x, t_k, ctx)
        score = -eps_pred / sigma
        z = torch.randn_like(x)
        drift = beta * score
        diffusion = beta.sqrt() * z
        x = x + drift * dtau + diffusion * dtau.sqrt()
        if project_fn is not None:
            x = project_fn(x, ctx)            # post-step projection onto M_φ
    return x


# ===========================================================================
# Behavior Cloning [Pomerleau89] — explicit deterministic policy
#
# Standard supervised regression: ctx → q-trajectory.   The conditioning is
# what the demonstrator would *see* at execution time: end-effector goal +
# embodiment context.  Multi-modal data ⇒ the deterministic regressor outputs
# the conditional mean — this is the canonical mode-averaging failure mode
# that [Florence22] addresses with implicit (energy-based) BC.  We use the
# explicit form to expose the failure cleanly.
# ===========================================================================


class BCTrajectoryPredictor(nn.Module):
    """Deterministic q-trajectory regressor:  ctx ↦ q ∈ R^{(H+1) × n_q}."""

    def __init__(
        self,
        n_q: int,
        H: int,
        ctx_dim: int,
        hidden: int = 512,
        n_layers: int = 5,
        activation: str = "sin",
    ):
        super().__init__()
        H1 = H + 1
        out_dim = H1 * n_q
        self.n_q = n_q
        self.H = H
        if activation not in _ACTIVATIONS:
            raise ValueError(activation)
        act = _ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        d = ctx_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden))
            layers.append(act())
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, ctx: Tensor) -> Tensor:
        B = ctx.shape[0]
        out = self.net(ctx)
        return out.reshape(B, self.H + 1, self.n_q)


def bc_loss(net: BCTrajectoryPredictor, ctx: Tensor, q_demo: Tensor) -> Tensor:
    """Mean-squared regression loss [Pomerleau89]."""
    pred = net(ctx)
    return ((pred - q_demo) ** 2).mean()


# ===========================================================================
# Diffusion Policy [Chi23] — using the OFFICIAL implementation
#
# ConditionalUnet1D + DDPMScheduler from real-stanford/diffusion_policy and
# huggingface diffusers, exactly as in [Chi23].  The horizon must be a power
# of 2 (down/upsample by stride-2 at each level), so we use H+1 = 16 in all
# downstream comparisons.
# ===========================================================================
try:
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    _HAS_OFFICIAL_DP = True
except Exception:                              # pragma: no cover
    _HAS_OFFICIAL_DP = False


def make_official_diffusion_policy(
    *,
    n_q: int,
    global_cond_dim: int,
    down_dims: list[int] | None = None,
    diffusion_step_embed_dim: int = 128,
    n_train_timesteps: int = 100,
    beta_schedule: str = "squaredcos_cap_v2",
    prediction_type: str = "epsilon",
    clip_sample: bool = True,
    cond_predict_scale: bool = False,
):
    """Build the canonical [Chi23] DDPM policy: ConditionalUnet1D + DDPMScheduler.

    Defaults match the published config (`squaredcos_cap_v2` β-schedule, ε-pred,
    clipped sample).  `down_dims` is reduced from [256,512,1024] to a smaller
    stack that's appropriate for the toy-scope joint-action dimensionality.
    """
    if not _HAS_OFFICIAL_DP:
        raise RuntimeError(
            "Official diffusion_policy + diffusers not importable. "
            "Install with `pip install -e baselines_external/diffusion_policy && pip install diffusers einops`."
        )
    if down_dims is None:
        down_dims = [128, 256, 512]
    model = ConditionalUnet1D(
        input_dim=n_q,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=down_dims,
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=cond_predict_scale,
    )
    scheduler = DDPMScheduler(
        num_train_timesteps=n_train_timesteps,
        beta_schedule=beta_schedule,
        clip_sample=clip_sample,
        prediction_type=prediction_type,
    )
    return model, scheduler


def official_dp_loss(model, scheduler, q_traj: Tensor, ctx: Tensor) -> Tensor:
    """Standard DDPM ε-prediction MSE  ([Chi23] training loop):
        sample t  ~  Uniform{0, …, T-1}
        sample z  ~  N(0, I)
        q_t = scheduler.add_noise(q_0, z, t)
        ε̂   = model(q_t, t, global_cond=ctx)
        loss = MSE(ε̂, z)
    """
    B = q_traj.shape[0]
    device = q_traj.device
    z = torch.randn_like(q_traj)
    t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device, dtype=torch.long)
    q_t = scheduler.add_noise(q_traj, z, t)
    eps_hat = model(q_t, t, global_cond=ctx)
    return ((eps_hat - z) ** 2).mean()


@torch.no_grad()
def official_dp_sample(model, scheduler, batch_size: int, horizon: int, n_q: int,
                       ctx: Tensor, device, n_inference_steps: int | None = None) -> Tensor:
    """DDPM ancestral sampling with ε-prediction model ([Chi23] §III.B)."""
    if n_inference_steps is None:
        n_inference_steps = scheduler.config.num_train_timesteps
    scheduler.set_timesteps(n_inference_steps, device=device)
    q = torch.randn(batch_size, horizon, n_q, device=device)
    for t in scheduler.timesteps:
        eps_hat = model(q, t, global_cond=ctx)
        q = scheduler.step(model_output=eps_hat, timestep=t, sample=q).prev_sample
    return q


# ---------------------------------------------------------------------------
# DP with channel-concat conditioning  (DP-A variant — architecture parity with Ours-V2)
# ---------------------------------------------------------------------------
# Standard DP (above) injects cond as global_cond → FiLM modulation.  For
# fairness comparison with Ours-V2 (which uses channel-concat), we provide a
# variant that broadcasts cond across timesteps and concatenates as input
# channels, then ε-predicts only the q-channels.


def channel_concat_dp_loss(model, scheduler, q_traj: Tensor, ctx_per_h: Tensor) -> Tensor:
    """ε-prediction loss with channel-concat cond.

    Input layout per timestep: [q_h, ctx_h] where ctx_h is the broadcast cond.
    Noise is added only to the q-block; cond channels remain as constants.
    Loss compares predicted ε on q-channels to the added noise.

    q_traj    : (B, H+1, n_q)
    ctx_per_h : (B, H+1, ctx_dim)  — broadcast cond per timestep
    """
    B, H1, n_q = q_traj.shape
    device = q_traj.device
    z = torch.randn_like(q_traj)
    t = torch.randint(0, scheduler.config.num_train_timesteps, (B,),
                      device=device, dtype=torch.long)
    q_t = scheduler.add_noise(q_traj, z, t)
    x_input = torch.cat([q_t, ctx_per_h], dim=-1)                    # (B, H+1, n_q + ctx_dim)
    eps_full = model(x_input, t, global_cond=None)                   # (B, H+1, n_q + ctx_dim)
    eps_pred_q = eps_full[..., :n_q]                                  # ε on q-channels only
    return ((eps_pred_q - z) ** 2).mean()


@torch.no_grad()
def channel_concat_dp_sample(model, scheduler, batch_size: int, horizon: int, n_q: int,
                              ctx_per_h: Tensor, device,
                              n_inference_steps: int | None = None) -> Tensor:
    """DDPM ancestral sampling for the channel-concat DP variant.

    Cond is fixed across timesteps and concatenated as input channels at every
    reverse step.  Only q is denoised; cond channels are passed through.
    """
    if n_inference_steps is None:
        n_inference_steps = scheduler.config.num_train_timesteps
    scheduler.set_timesteps(n_inference_steps, device=device)
    q = torch.randn(batch_size, horizon, n_q, device=device)
    for t in scheduler.timesteps:
        x_input = torch.cat([q, ctx_per_h], dim=-1)
        eps_full = model(x_input, t, global_cond=None)
        eps_pred_q = eps_full[..., :n_q]
        q = scheduler.step(model_output=eps_pred_q, timestep=t, sample=q).prev_sample
    return q


# ---------------------------------------------------------------------------
# DP-C — DP-A + classifier guidance (sampling-time, no retrain)
# ---------------------------------------------------------------------------
# Apply Dhariwal-Nichol-style classifier guidance with reward gradients during
# DDPM ancestral sampling.  Reward = α_g R_goal + α_s R_start + α_v R_vel,
# with R_goal at h=H, R_start at h=0, R_vel across all h.  Mirrors Ours-V2's
# sampling-time analytic guidance, but in flat Euclidean q-space (no G^{-1}).


def official_dp_sample_guided(
    model, scheduler, batch_size: int, horizon: int, n_q: int,
    ctx: Tensor, device, n_inference_steps: int | None = None,
    *,
    fk_fn,
    z_e_per_traj: Tensor,
    p_target: Tensor | None = None,
    alpha_g: float = 0.0,
    h_indices_goal: list[int] | None = None,
    p_start_anchor: Tensor | None = None,
    alpha_s: float = 0.0,
    h_indices_start: list[int] | None = None,
    alpha_v: float = 0.0,
):
    """DP-B: Canonical Chi23 DP (global_cond) + classifier guidance via reward gradients.

    Same guidance schema as channel_concat_dp_sample_guided but uses global_cond
    architecture (canonical Chi23).  Isolates the contribution of the architecture
    (global vs channel) given identical sampling-time guidance.
    """
    if n_inference_steps is None:
        n_inference_steps = scheduler.config.num_train_timesteps
    scheduler.set_timesteps(n_inference_steps, device=device)
    q = torch.randn(batch_size, horizon, n_q, device=device)
    if h_indices_goal is None:
        h_indices_goal = [horizon - 1]
    if h_indices_start is None:
        h_indices_start = [0]

    for t in scheduler.timesteps:
        with torch.no_grad():
            eps_pred_q = model(q, t, global_cond=ctx)

        grad = torch.zeros_like(q)
        if alpha_g > 0.0 and p_target is not None:
            for h in h_indices_goal:
                q_h_leaf = q[:, h, :].detach().clone().requires_grad_(True)
                with torch.enable_grad():
                    p_h = fk_fn(q_h_leaf, z_e_per_traj)
                    err_sq = ((p_h - p_target) ** 2).sum(-1).sum()
                    g_h = torch.autograd.grad(err_sq, q_h_leaf,
                                              create_graph=False, retain_graph=False)[0]
                grad[:, h, :] = grad[:, h, :] + alpha_g * g_h.detach()
        if alpha_s > 0.0 and p_start_anchor is not None:
            for h in h_indices_start:
                q_h_leaf = q[:, h, :].detach().clone().requires_grad_(True)
                with torch.enable_grad():
                    p_h = fk_fn(q_h_leaf, z_e_per_traj)
                    err_sq = ((p_h - p_start_anchor) ** 2).sum(-1).sum()
                    g_h = torch.autograd.grad(err_sq, q_h_leaf,
                                              create_graph=False, retain_graph=False)[0]
                grad[:, h, :] = grad[:, h, :] + alpha_s * g_h.detach()
        if alpha_v > 0.0 and horizon >= 2:
            grad_vel = torch.zeros_like(q)
            grad_vel[:, 0, :]   = 2.0 * (q[:, 0, :]   - q[:, 1, :])
            grad_vel[:, -1, :]  = 2.0 * (q[:, -1, :]  - q[:, -2, :])
            if horizon > 2:
                grad_vel[:, 1:-1, :] = 2.0 * (2.0 * q[:, 1:-1, :] - q[:, :-2, :] - q[:, 2:, :])
            grad = grad + alpha_v * grad_vel

        eps_guided = eps_pred_q + grad
        with torch.no_grad():
            q = scheduler.step(model_output=eps_guided, timestep=t, sample=q).prev_sample
    return q


def channel_concat_dp_sample_guided(
    model, scheduler, batch_size: int, horizon: int, n_q: int,
    ctx_per_h: Tensor, device, n_inference_steps: int | None = None,
    *,
    fk_fn,                                                           # callable F(q_h: (B,n_q), z_e: (B,1)) -> (B,3)
    z_e_per_traj: Tensor,                                            # (B, 1)
    p_target: Tensor | None = None,                                  # (B, 3) — endpoint goal
    alpha_g: float = 0.0,
    h_indices_goal: list[int] | None = None,
    p_start_anchor: Tensor | None = None,                            # (B, 3) — start anchor
    alpha_s: float = 0.0,
    h_indices_start: list[int] | None = None,
    alpha_v: float = 0.0,                                            # velocity smoothness
):
    """DDPM ancestral sampling with classifier guidance via reward gradients.

    At each step:
      ε_θ predicts noise (no grad needed for model's params),
      reward gradient g = ∇_q R(q_t, c) is computed analytically
      (R_goal/R_start use F via autograd; R_vel is closed-form),
      ε_guided = ε_θ + g       (Euclidean classifier guidance scaling absorbed in α)
    """
    if n_inference_steps is None:
        n_inference_steps = scheduler.config.num_train_timesteps
    scheduler.set_timesteps(n_inference_steps, device=device)
    q = torch.randn(batch_size, horizon, n_q, device=device)
    if h_indices_goal is None:
        h_indices_goal = [horizon - 1]
    if h_indices_start is None:
        h_indices_start = [0]

    for t in scheduler.timesteps:
        with torch.no_grad():
            x_input = torch.cat([q, ctx_per_h], dim=-1)
            eps_full = model(x_input, t, global_cond=None)
            eps_pred_q = eps_full[..., :n_q]

        # Compute reward-gradient corrections in chart-q space (Euclidean).
        grad = torch.zeros_like(q)

        if alpha_g > 0.0 and p_target is not None:
            for h in h_indices_goal:
                q_h_leaf = q[:, h, :].detach().clone().requires_grad_(True)
                with torch.enable_grad():
                    p_h = fk_fn(q_h_leaf, z_e_per_traj)
                    err_sq = ((p_h - p_target) ** 2).sum(-1).sum()
                    g_h = torch.autograd.grad(err_sq, q_h_leaf,
                                              create_graph=False, retain_graph=False)[0]
                grad[:, h, :] = grad[:, h, :] + alpha_g * g_h.detach()

        if alpha_s > 0.0 and p_start_anchor is not None:
            for h in h_indices_start:
                q_h_leaf = q[:, h, :].detach().clone().requires_grad_(True)
                with torch.enable_grad():
                    p_h = fk_fn(q_h_leaf, z_e_per_traj)
                    err_sq = ((p_h - p_start_anchor) ** 2).sum(-1).sum()
                    g_h = torch.autograd.grad(err_sq, q_h_leaf,
                                              create_graph=False, retain_graph=False)[0]
                grad[:, h, :] = grad[:, h, :] + alpha_s * g_h.detach()

        if alpha_v > 0.0 and horizon >= 2:
            # ∂(∑ ‖q_{h+1} - q_h‖²)/∂q_h: 2(q_0-q_1) at h=0, 2(q_H-q_{H-1}) at h=H,
            # 2(2 q_h - q_{h-1} - q_{h+1}) for 0<h<H.
            grad_vel = torch.zeros_like(q)
            grad_vel[:, 0, :]   = 2.0 * (q[:, 0, :]   - q[:, 1, :])
            grad_vel[:, -1, :]  = 2.0 * (q[:, -1, :]  - q[:, -2, :])
            if horizon > 2:
                grad_vel[:, 1:-1, :] = 2.0 * (2.0 * q[:, 1:-1, :] - q[:, :-2, :] - q[:, 2:, :])
            grad = grad + alpha_v * grad_vel

        # ε guidance: ε_guided = ε_θ + grad  (descent direction in error-increasing reward sense)
        eps_guided = eps_pred_q + grad

        with torch.no_grad():
            q = scheduler.step(model_output=eps_guided, timestep=t, sample=q).prev_sample
    return q
