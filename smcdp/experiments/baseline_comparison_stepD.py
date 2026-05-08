"""§15.2 baseline comparison at Step D-2 conditions.

Trains 4 baselines on the SAME demo data as Step D-2 (3-link bimodal IK
trajectories with z_e variation) and evaluates each with the SAME standardised
metric suite.  Ours = Step D-2 ckpt (loaded, not retrained).

Baselines (Idea §15.2):
    1. BC                   — deterministic q-trajectory predictor (mode-collapses)
    2. Vanilla DSM          — chart-Eucl trajectory diffusion, no z_e
    3. Vanilla DSM + z_e    — chart-Eucl + z_e conditioning
    4. Projected DSM + z_e  — ambient (q, p) DSM with post-hoc projection onto
                              the LEARNED M_φ(z_e) (uses Stage-1 Δ_φ)
    6. Step D-1             — our framework with Δ_φ = 0 (analytic FK)  ← already done
    7. Ours = Step D-2      — full framework                              ← already done

Uniform metrics per z_e (Idea §15.4):
    • mode coverage          frac(branch=up) for model vs demo  (target = 0.5)
    • mode-averaging fraction (|q_2_mid| < 0.5)
    • per-mode chart-q sliced-W₁ vs the demo branch
    • physical reach error   ‖p_actual_at_h=H − target_p_end‖
                              with p_actual = TrueArmCompliance.p_true(q, z_e)
                              [represents what really happens when the joint
                              command is executed on the compliant real arm]
    • adherence to LEARNED manifold  (only ours / projected baseline)
    • adherence to TRUE manifold     mean_h ‖p_in_state − p_true(q, z_e)‖
                                     (only models with p in state)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from smcdp.sde import LinearBetaSchedule, LangevinSDE
from smcdp.toy3.ground_truth import TrueNLinkArmCompliance
from smcdp.toy3.self_model import DeltaResidualMLP, LearnedSelfModelNLinkArm
from smcdp.toy3.distributions import WrappedNormalEmbodiment
from smcdp.trajectories import (
    BimodalRedundantTrajectoryDistEmb,
    TrajectoryScoreNet,
    TrajectoryScaledScoreFn,
    traj_reverse_grw,
)
from smcdp.baselines import (
    _FlatTrajMLP,
    diffusion_policy_loss,
    diffusion_policy_reverse,
    diffusion_policy_prior,
    projected_loss,
    projected_reverse,
    BCTrajectoryPredictor,
    bc_loss,
    make_official_diffusion_policy,
    official_dp_loss,
    official_dp_sample,
)


# =============================================================================
# Standard evaluation suite — all baselines + ours go through the same metrics.
# =============================================================================


def standardised_eval(
    *,
    name: str,
    q_gen: torch.Tensor,                 # (n, H+1, n_q)  — generated joint trajectory
    q_data: torch.Tensor,                # (n, H+1, n_q)  — demo joint trajectory
    branch_data: torch.Tensor,           # (n,) bool      — demo branch label
    z_e: torch.Tensor,                   # (n, 1)         — embodiment per traj
    truth: TrueNLinkArmCompliance,
    target_end: torch.Tensor,            # (2,) p_target endpoint
    learned_p_at: torch.Tensor | None = None,  # (n, H+1, 2) — model's p in state, if any
    n_q: int = 3,
    n_dir: int = 64,
) -> dict:
    """Compute the unified metric set listed at top of file."""
    n, H1, _ = q_gen.shape
    device = q_gen.device
    h_mid = H1 // 2

    # Mode classification (sign of q_2 at midpoint)
    q2_mid_data = q_data[:, h_mid, 1]
    q2_mid_gen = q_gen[:, h_mid, 1]
    mode_up_data = q2_mid_data > 0
    mode_up_gen = q2_mid_gen > 0
    frac_up_data = mode_up_data.float().mean().item()
    frac_up_gen = mode_up_gen.float().mean().item()
    between = (q2_mid_gen.abs() < 0.5).float().mean().item()

    # Per-mode chart-q sliced-W₁
    dirs = torch.randn(n_dir, n_q, device=device)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)

    def _per_mode_w1(qa: torch.Tensor, qb: torch.Tensor) -> float:
        m = min(qa.shape[0], qb.shape[0])
        if m < 8:
            return float("nan")
        ia = torch.randperm(qa.shape[0], device=device)[:m]
        ib = torch.randperm(qb.shape[0], device=device)[:m]
        a = qa[ia]; b = qb[ib]
        ws = []
        for h in range(H1):
            pa = (a[:, h] @ dirs.T).sort(dim=0).values
            pb = (b[:, h] @ dirs.T).sort(dim=0).values
            ws.append((pa - pb).abs().mean().item())
        return sum(ws) / len(ws)

    w1_up = _per_mode_w1(q_gen[mode_up_gen], q_data[mode_up_data])
    w1_dn = _per_mode_w1(q_gen[~mode_up_gen], q_data[~mode_up_data])
    w1_mean = float("nan") if any(map(lambda x: x != x, [w1_up, w1_dn])) else 0.5 * (w1_up + w1_dn)

    # Physical reach error: simulate q-execution on TRUE compliant arm
    # p_actual_h = TrueArmCompliance.p_true(q_h, z_e)
    z_per_pt = z_e.unsqueeze(1).expand(-1, H1, -1).reshape(-1, 1)
    p_actual_flat = truth.p_true(q_gen.reshape(-1, n_q), z_per_pt)
    p_actual = p_actual_flat.reshape(n, H1, 2)
    reach_err = (p_actual[:, -1] - target_end.to(device)).norm(dim=-1).mean().item()

    # Adherence to TRUE manifold:  if model carries a p in state, compare against p_true
    if learned_p_at is not None:
        g_truth = (learned_p_at - p_actual).norm(dim=-1).mean().item()
    else:
        g_truth = float("nan")

    return {
        "name": name,
        "frac_up_data": frac_up_data,
        "frac_up_gen": frac_up_gen,
        "frac_between_modes": between,
        "w1_up": w1_up,
        "w1_dn": w1_dn,
        "w1_mean": w1_mean,
        "reach_err": reach_err,
        "g_truth_meanH": g_truth,
    }


# =============================================================================
# Baseline trainers — each returns a callable `sample_q(n, z_e_batch) -> q-traj`
# =============================================================================


def train_vanilla(
    *, data, schedule, n_q, H, device,
    steps, batch, lr, ema, hidden, n_layers, eps,
    use_z_e: bool,
) -> tuple:
    """Train vanilla chart-Eucl DSM on q-trajectories.  Returns (sample_fn, info)."""
    ctx_dim = 1 if use_z_e else 0
    net = _FlatTrajMLP(d_state=n_q, H=H, hidden=hidden, n_layers=n_layers,
                       ctx_dim=ctx_dim, time_embedding="raw", activation="sin").to(device)
    ema_net = _FlatTrajMLP(d_state=n_q, H=H, hidden=hidden, n_layers=n_layers,
                           ctx_dim=ctx_dim, time_embedding="raw", activation="sin").to(device)
    ema_net.load_state_dict(net.state_dict())
    for p in ema_net.parameters():
        p.requires_grad_(False)

    optim = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    losses_log: list[float] = []
    for step in tqdm(range(steps), desc=f"vanilla{'+z_e' if use_z_e else ''}"):
        x_demo, _, z_e_demo = data.sample(batch, device=device)
        q_demo = x_demo[..., :n_q]
        ctx = z_e_demo if use_z_e else None
        loss = diffusion_policy_loss(net, schedule, q_demo, eps=eps, ctx=ctx)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        with torch.no_grad():
            for pe, pn in zip(ema_net.parameters(), net.parameters()):
                pe.mul_(ema).add_(pn, alpha=(1.0 - ema))
        losses_log.append(loss.item())

    @torch.no_grad()
    def sample_q(n, z_e_batch, n_sample_steps=200):
        q_T = diffusion_policy_prior(n, H, n_q, device=device, scale=1.5)
        ctx = z_e_batch if use_z_e else None
        q_gen = diffusion_policy_reverse(ema_net, schedule, q_T,
                                         n_steps=n_sample_steps, eps=eps, ctx=ctx)
        return q_gen

    return sample_q, {"net": net, "ema": ema_net, "losses": losses_log}


def train_official_dp(
    *, data, n_q, H, device,
    steps, batch, lr, ema, eps,
):
    """Train the OFFICIAL Diffusion Policy [Chi23]: ConditionalUnet1D + DDPMScheduler.
    Conditioning: z_e (1-D global_cond).  Output: q-trajectory (n_q dims).
    """
    H1 = H + 1
    model, scheduler = make_official_diffusion_policy(
        n_q=n_q, global_cond_dim=1,
        down_dims=[128, 256, 512],
        n_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
    )
    model = model.to(device)

    # EMA copy
    ema_model, ema_scheduler = make_official_diffusion_policy(
        n_q=n_q, global_cond_dim=1,
        down_dims=[128, 256, 512],
        n_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
    )
    ema_model = ema_model.to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad_(False)

    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    losses_log: list[float] = []
    for step in tqdm(range(steps), desc="diffusion-policy [Chi23]"):
        x_demo, _, z_e_demo = data.sample(batch, device=device)
        q_demo = x_demo[..., :n_q]
        loss = official_dp_loss(model, scheduler, q_demo, ctx=z_e_demo)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        with torch.no_grad():
            for pe, pn in zip(ema_model.parameters(), model.parameters()):
                pe.mul_(ema).add_(pn, alpha=(1.0 - ema))
        losses_log.append(loss.item())

    @torch.no_grad()
    def sample_q(n, z_e_batch, n_sample_steps=100):
        return official_dp_sample(ema_model, ema_scheduler, batch_size=n,
                                  horizon=H1, n_q=n_q, ctx=z_e_batch,
                                  device=device, n_inference_steps=n_sample_steps)

    return sample_q, {"model": model, "ema": ema_model, "losses": losses_log}


def train_projected(
    *, data, schedule, arm, n_q, H, device,
    steps, batch, lr, ema, hidden, n_layers, eps,
) -> tuple:
    """Train ambient (q, p) chart-Eucl DSM with post-hoc projection onto M_φ.

    Conditioning: z_e (1-dim).  State: ambient (q, p) ∈ R^{n_q+2}.
    """
    d_state = n_q + 2
    ctx_dim = 1
    net = _FlatTrajMLP(d_state=d_state, H=H, hidden=hidden, n_layers=n_layers,
                       ctx_dim=ctx_dim, time_embedding="raw", activation="sin").to(device)
    ema_net = _FlatTrajMLP(d_state=d_state, H=H, hidden=hidden, n_layers=n_layers,
                           ctx_dim=ctx_dim, time_embedding="raw", activation="sin").to(device)
    ema_net.load_state_dict(net.state_dict())
    for p in ema_net.parameters():
        p.requires_grad_(False)

    optim = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    losses_log: list[float] = []
    for step in tqdm(range(steps), desc="projected+z_e"):
        x_demo, _, z_e_demo = data.sample(batch, device=device)
        # demo state is (q, p, z_e) but we strip z_e here — z_e is conditioning
        x_amb = x_demo[..., : n_q + 2]
        loss = projected_loss(net, schedule, x_amb, eps=eps, ctx=z_e_demo)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        with torch.no_grad():
            for pe, pn in zip(ema_net.parameters(), net.parameters()):
                pe.mul_(ema).add_(pn, alpha=(1.0 - ema))
        losses_log.append(loss.item())

    # projection function: replace p with F_φ(q, z_e=ctx)
    def project_to_M(x_amb: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # x_amb: (n, H+1, n_q+2)   ctx (z_e): (n, 1)
        n, H1, _ = x_amb.shape
        q_part = x_amb[..., :n_q]
        z_per_pt = ctx.unsqueeze(1).expand(-1, H1, -1).reshape(-1, 1)
        with torch.no_grad():
            p_proj = arm.F(q_part.reshape(-1, n_q), z_per_pt).reshape(n, H1, 2)
        return torch.cat([q_part, p_proj], dim=-1)

    @torch.no_grad()
    def sample_q_and_p(n, z_e_batch, n_sample_steps=200):
        x_T = diffusion_policy_prior(n, H, d_state, device=device, scale=1.5)
        x_T = project_to_M(x_T, z_e_batch)             # start on M
        x_gen = projected_reverse(ema_net, schedule, x_T,
                                  n_steps=n_sample_steps, eps=eps,
                                  ctx=z_e_batch, project_fn=project_to_M)
        q_gen = x_gen[..., :n_q]
        p_gen = x_gen[..., n_q : n_q + 2]
        return q_gen, p_gen

    return sample_q_and_p, {"net": net, "ema": ema_net, "losses": losses_log}


def train_bc(
    *, data, n_q, H, device,
    steps, batch, lr, ema, hidden, n_layers,
) -> tuple:
    """Deterministic BC: ctx (z_e + start/end p) → q-trajectory."""
    ctx_dim = 1 + 2 + 2                                                       # z_e + p_start + p_end
    net = BCTrajectoryPredictor(n_q=n_q, H=H, ctx_dim=ctx_dim,
                                hidden=hidden, n_layers=n_layers,
                                activation="sin").to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    losses_log: list[float] = []
    for step in tqdm(range(steps), desc="BC"):
        x_demo, _, z_e_demo = data.sample(batch, device=device)
        q_demo = x_demo[..., :n_q]
        # Condition on z_e + p_start + p_end (the demo's start/end of end-effector traj)
        p_start = x_demo[:, 0, n_q : n_q + 2]
        p_end = x_demo[:, -1, n_q : n_q + 2]
        ctx = torch.cat([z_e_demo, p_start, p_end], dim=-1)
        loss = bc_loss(net, ctx, q_demo)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        losses_log.append(loss.item())

    @torch.no_grad()
    def sample_q(n, z_e_batch, p_start_batch, p_end_batch):
        ctx = torch.cat([z_e_batch, p_start_batch, p_end_batch], dim=-1)
        return net(ctx)

    return sample_q, {"net": net, "losses": losses_log}


# =============================================================================
# Main: train baselines, load ours/D-1, run unified eval
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-ckpt", type=str, default="outputs/toy3p5_stage1/delta_phi.pt")
    p.add_argument("--stepD2-ckpt", type=str,
                   default="outputs/toy3p5_stepD2_H16/ckpt_riemannian.pt")
    p.add_argument("--stepD1-ckpt", type=str,
                   default="outputs/toy3p5_stepD_H16/ckpt_riemannian.pt")
    p.add_argument("--H", type=int, default=15,
                   help="trajectory horizon H (H+1 timesteps); 15 → H+1=16 "
                        "to satisfy ConditionalUnet1D's power-of-2 horizon constraint.")
    p.add_argument("--steps", type=int, default=15_000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=5)
    p.add_argument("--beta-0", type=float, default=0.001)
    p.add_argument("--beta-f", type=float, default=6.0)
    p.add_argument("--eps", type=float, default=2e-4)
    p.add_argument("--n-eval-per-z", type=int, default=2048)
    p.add_argument("--z-eval", type=float, nargs="+", default=[0.00, 0.15, 0.30, 0.45])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="outputs/baselines_stepD")
    p.add_argument(
        "--baselines", type=str, nargs="+",
        default=["dp_official", "projected", "bc"],
        help="which baselines to train (subset of "
             "{dp_official, vanilla, vanilla_z, projected, bc}). "
             "dp_official = real-stanford/diffusion_policy + diffusers DDPMScheduler.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out_dir={out_dir}  baselines={args.baselines}")

    # ---- shared setup (re-used Stage-1 ckpt for the manifold) ----
    s1_ck = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
    s1 = s1_ck["args"]
    n_q = len(s1["link_lengths_base"])
    print(f"Stage-1 improvement={s1_ck['metrics']['improvement_factor']:.1f}x  "
          f"link_lengths_base={s1['link_lengths_base']}")

    delta_net = DeltaResidualMLP(n_q=n_q, n_p=2, n_z=1,
                                 hidden=s1["hidden"], n_layers=s1["n_layers"]).to(device)
    delta_net.load_state_dict(s1_ck["delta_net_state"])
    delta_net.eval()
    arm = LearnedSelfModelNLinkArm(delta_net=delta_net,
                                   link_lengths_base=s1["link_lengths_base"],
                                   metric="riemannian")
    truth = TrueNLinkArmCompliance(link_lengths_base=s1["link_lengths_base"],
                                   K_grav=s1["K_grav"], K_offset=s1["K_offset"])

    schedule = LinearBetaSchedule(beta_0=args.beta_0, beta_f=args.beta_f, t0=0.0, tf=1.0)
    limiting = WrappedNormalEmbodiment(arm, mean_q=[0.0, 0.0, 0.0], scale=1.5,
                                       z_e_range=(0.0, 0.30))
    sde = LangevinSDE(arm, schedule, limiting)
    data = BimodalRedundantTrajectoryDistEmb(
        arm, H=args.H,
        mu_p_start=(1.6, 0.6), mu_p_end=(1.6, -0.6),
        jitter_p=0.05, z_e_range=(0.0, 0.30), branch_p_up=0.5,
    )

    # ---- train requested baselines ----
    samplers = {}     # name -> sample_fn (varies in signature; standardised below)
    train_times = {}

    if "dp_official" in args.baselines:
        t0 = time.time()
        sf, info = train_official_dp(data=data, n_q=n_q, H=args.H, device=device,
                                     steps=args.steps, batch=args.batch, lr=args.lr,
                                     ema=args.ema, eps=args.eps)
        samplers["dp_official"] = ("q-only-cond", sf)
        train_times["dp_official"] = time.time() - t0

    if "vanilla" in args.baselines:
        t0 = time.time()
        sf, info = train_vanilla(data=data, schedule=schedule, n_q=n_q, H=args.H, device=device,
                                 steps=args.steps, batch=args.batch, lr=args.lr, ema=args.ema,
                                 hidden=args.hidden, n_layers=args.n_layers, eps=args.eps,
                                 use_z_e=False)
        samplers["vanilla"] = ("q-only", sf)
        train_times["vanilla"] = time.time() - t0

    if "vanilla_z" in args.baselines:
        t0 = time.time()
        sf, info = train_vanilla(data=data, schedule=schedule, n_q=n_q, H=args.H, device=device,
                                 steps=args.steps, batch=args.batch, lr=args.lr, ema=args.ema,
                                 hidden=args.hidden, n_layers=args.n_layers, eps=args.eps,
                                 use_z_e=True)
        samplers["vanilla_z"] = ("q-only-cond", sf)
        train_times["vanilla_z"] = time.time() - t0

    if "projected" in args.baselines:
        t0 = time.time()
        sf, info = train_projected(data=data, schedule=schedule, arm=arm, n_q=n_q, H=args.H,
                                   device=device, steps=args.steps, batch=args.batch, lr=args.lr,
                                   ema=args.ema, hidden=args.hidden, n_layers=args.n_layers,
                                   eps=args.eps)
        samplers["projected"] = ("ambient-cond", sf)
        train_times["projected"] = time.time() - t0

    if "bc" in args.baselines:
        t0 = time.time()
        sf, info = train_bc(data=data, n_q=n_q, H=args.H, device=device, steps=args.steps,
                            batch=args.batch, lr=args.lr, ema=args.ema,
                            hidden=args.hidden, n_layers=args.n_layers)
        samplers["bc"] = ("bc-cond", sf)
        train_times["bc"] = time.time() - t0

    # ---- load OURS (Step D-2) and Step D-1 (oracle analytic, baseline 6) ----
    def load_traj_score_ckpt(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        a = ck["args"]
        net = TrajectoryScoreNet(arm, H=args.H,
                                 hidden=a["hidden"], n_layers=a["n_layers"],
                                 t_embed_dim=64, activation=a["activation"],
                                 time_embedding=a["time_embedding"]).to(device)
        net.load_state_dict(ck["ema_state"])
        net.eval()
        sde_local = LangevinSDE(
            arm, LinearBetaSchedule(beta_0=a["beta_0"], beta_f=a["beta_f"]),
            WrappedNormalEmbodiment(arm, mean_q=a["limiting_mean_q"],
                                    scale=a["limiting_scale"],
                                    z_e_range=(a["z_min"], a["z_max"])),
        )
        score_fn = TrajectoryScaledScoreFn(net, sde_local)

        @torch.no_grad()
        def sample(n, z_e_batch, n_sample_steps=200):
            d = arm.ambient_dim
            H1 = args.H + 1
            z_lim = z_e_batch.unsqueeze(1).expand(-1, H1, -1).reshape(n * H1, -1)
            tau_T = sde_local.limiting.sample(n * H1, device=device, z_e=z_lim).reshape(n, H1, d)
            tau_gen = traj_reverse_grw(sde_local, score_fn, tau_T,
                                       n_steps=n_sample_steps, eps=a["eps"])
            return tau_gen[..., :n_q], tau_gen[..., n_q : n_q + 2]
        return sample

    samplers["d1_oracle"] = ("ambient-cond-noresidual", load_traj_score_ckpt(args.stepD1_ckpt))
    samplers["d2_ours"] = ("ambient-cond-residual", load_traj_score_ckpt(args.stepD2_ckpt))

    # ---- standardised eval per z_e ----
    H1 = args.H + 1
    n_eval = args.n_eval_per_z
    p_target_end = data.mu_p_end.to(device=device)
    all_metrics = {}

    for z_val in args.z_eval:
        z_tensor = torch.full((n_eval, 1), z_val, device=device)
        # ground-truth demo at this z_e: build by sampling
        x_demo_z, branch_z, _ = data.sample(n_eval, device=device, dtype=torch.float32)        # ignore z_e returned (we override)
        # rebuild data trajectories at the FIXED z_val
        z_traj_data = z_tensor.unsqueeze(1).expand(-1, H1, -1).contiguous()
        p_start = data.mu_p_start.to(device=device) + data.jitter_p * torch.randn(n_eval, 2, device=device)
        p_end = data.mu_p_end.to(device=device) + data.jitter_p * torch.randn(n_eval, 2, device=device)
        s = torch.linspace(0, 1, H1, device=device).view(1, H1, 1)
        p_traj_data = p_start.unsqueeze(1) + s * (p_end - p_start).unsqueeze(1)
        branch_data = torch.rand(n_eval, device=device) < data.branch_p_up
        branch_traj = branch_data.unsqueeze(1).expand(n_eval, H1)
        q_data = data._ik_3link_emb(p_traj_data, branch_traj, z_tensor)        # (n, H+1, 3)

        for name, (kind, sf) in samplers.items():
            if kind == "q-only":
                q_gen = sf(n_eval, z_tensor)
                p_at = None
            elif kind == "q-only-cond":
                q_gen = sf(n_eval, z_tensor)
                p_at = None
            elif kind == "ambient-cond":
                q_gen, p_gen = sf(n_eval, z_tensor)
                p_at = p_gen
            elif kind == "ambient-cond-noresidual":
                q_gen, p_gen = sf(n_eval, z_tensor)
                p_at = p_gen
            elif kind == "ambient-cond-residual":
                q_gen, p_gen = sf(n_eval, z_tensor)
                p_at = p_gen
            elif kind == "bc-cond":
                q_gen = sf(n_eval, z_tensor, p_start, p_end)
                p_at = None
            else:
                raise ValueError(kind)

            metrics = standardised_eval(
                name=name,
                q_gen=q_gen, q_data=q_data, branch_data=branch_data,
                z_e=z_tensor, truth=truth, target_end=p_target_end,
                learned_p_at=p_at, n_q=n_q,
            )
            all_metrics.setdefault(name, {})[float(z_val)] = metrics

    # ---- print summary table ----
    rows_order = list(samplers.keys())
    print("\n" + "=" * 110)
    print("Summary — frac_up / averaging / W₁_mean / reach_err / g_truth (mean over h)")
    print("=" * 110)
    header = f"{'baseline':<28} | " + " | ".join(f"{f'z={z:.2f}':>22}" for z in args.z_eval)
    print(header)
    print("-" * len(header))
    for r in rows_order:
        cells = []
        for z in args.z_eval:
            m = all_metrics[r][float(z)]
            cells.append(
                f"f_up={m['frac_up_gen']:.2f} avg={m['frac_between_modes']:.2f} "
                f"W₁={m['w1_mean']:.2f} R={m['reach_err']:.3f}"
            )
        print(f"{r:<28} | " + " | ".join(f"{c:>22}" for c in cells))
    print("=" * 110)

    with open(out_dir / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    with open(out_dir / "train_times_sec.json", "w") as f:
        json.dump(train_times, f, indent=2)
    print(f"\nresults saved to {out_dir}/all_metrics.json")


if __name__ == "__main__":
    main()
