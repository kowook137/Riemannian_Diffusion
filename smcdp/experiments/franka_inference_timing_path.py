"""Inference-time and shortest-path comparison across all methods.

For each method (BC, DP-canonical, DP-A, DP-C, Projected, Ours-Analytic, Ours-V2):
  1. Time the sampling for fixed batch size (warmup + multiple repeats, std reported).
  2. Compute end-effector and joint-space path lengths and straight-line ratios
     (how close each generated trajectory is to a straight-line shortest path).

All methods evaluated at z_e=0.10 (in-distribution) with the same target/start
distribution and the same demo reference, so that only the method differs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import pybullet_data

from smcdp.manifolds import Franka7DoF
from smcdp.toy3.self_model import DeltaResidualMLP
from smcdp.franka.self_model import LearnedSelfModelFranka7DoF
from smcdp.franka.distributions import WrappedNormalFranka7DoF
from smcdp.franka.demo_gen import FrankaBimodalReachingDemo
from smcdp.sde import LangevinSDE, LinearBetaSchedule
from smcdp.trajectories import (
    TrajectoryScoreNetUNet, TrajectoryScaledScoreFn, traj_reverse_grw,
)
from smcdp.baselines import (
    BCTrajectoryPredictor, make_official_diffusion_policy,
    official_dp_sample, channel_concat_dp_sample,
    channel_concat_dp_sample_guided,
)


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def _build_arms(device: str, perfect_fk: bool, stage1_args: dict, tool_z_max: float = 0.30):
    """Build (analytic, learned-or-analytic) Franka arms used for FK / metrics."""
    arm_a = Franka7DoF(urdf_path=URDF, end_link="panda_hand", tool_z_max=tool_z_max)
    arm_a._ensure_chain(torch.zeros(1, 7, device=device))
    if perfect_fk:
        return arm_a, arm_a
    delta_net = DeltaResidualMLP(n_q=7, n_p=3, n_z=1,
                                  hidden=stage1_args["hidden"],
                                  n_layers=stage1_args["n_layers"],
                                  activation=torch.nn.Softplus,
                                  final_init_scale=1e-3).to(device)
    s1_ck = torch.load("outputs/franka_stage1/delta_phi.pt",
                       map_location=device, weights_only=False)
    delta_net.load_state_dict(s1_ck["delta_net_state"])
    delta_net.eval()
    arm = LearnedSelfModelFranka7DoF(
        delta_net=delta_net, urdf_path=URDF, end_link="panda_hand",
        tool_z_max=tool_z_max,
    )
    arm._ensure_chain(torch.zeros(1, 7, device=device))
    return arm_a, arm


def _time_sampler(sample_fn, *, warmup: int = 2, repeats: int = 5):
    """Run sample_fn warmup+repeats times, return (mean_sec, std_sec)."""
    for _ in range(warmup):
        sample_fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sample_fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    t = torch.tensor(times)
    return t.mean().item(), t.std().item()


def _path_metrics(p_traj: torch.Tensor, q_traj: torch.Tensor) -> dict:
    """End-effector and joint-space path-length stats per trajectory.

    p_traj: (n, H+1, 3), q_traj: (n, H+1, n_q)
    Returns means over n.
    """
    seg_p = (p_traj[:, 1:] - p_traj[:, :-1]).norm(dim=-1)                    # (n, H)
    path_len_ee = seg_p.sum(dim=1)                                            # (n,)
    straight_ee = (p_traj[:, -1] - p_traj[:, 0]).norm(dim=-1)                 # (n,)

    seg_q = (q_traj[:, 1:] - q_traj[:, :-1]).norm(dim=-1)
    path_len_q = seg_q.sum(dim=1)
    straight_q = (q_traj[:, -1] - q_traj[:, 0]).norm(dim=-1)

    eps = 1e-9
    straightness_ee = (straight_ee / (path_len_ee + eps)).clamp(max=1.0)
    excess_ee = (path_len_ee / (straight_ee + eps))
    straightness_q = (straight_q / (path_len_q + eps)).clamp(max=1.0)
    excess_q = (path_len_q / (straight_q + eps))

    return {
        "ee_path_len_mean_m":   path_len_ee.mean().item(),
        "ee_path_len_std_m":    path_len_ee.std().item(),
        "ee_straight_mean_m":   straight_ee.mean().item(),
        "ee_straightness_mean": straightness_ee.mean().item(),
        "ee_straightness_std":  straightness_ee.std().item(),
        "ee_excess_mean":       excess_ee.mean().item(),
        "ee_excess_std":        excess_ee.std().item(),
        "q_path_len_mean":      path_len_q.mean().item(),
        "q_path_len_std":       path_len_q.std().item(),
        "q_straight_mean":      straight_q.mean().item(),
        "q_straightness_mean":  straightness_q.mean().item(),
        "q_excess_mean":        excess_q.mean().item(),
    }


# ---------------------------------------------------------------------------
# Per-method runners.  Each returns (sample_callable, q_traj, p_traj_for_metrics)
# where the callable is what we time, q_traj/p_traj are produced once for path
# metrics (n trajectories, H+1 timesteps).
# ---------------------------------------------------------------------------


def _run_bc(ckpt_path: str, n: int, device: str, ctx: torch.Tensor, z_e: torch.Tensor):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["args"]; s1 = ck["stage1_args"]
    perfect_fk = bool(a.get("perfect_fk", False))
    arm_a, arm = _build_arms(device, perfect_fk, s1)
    H1 = a["H"] + 1
    model = BCTrajectoryPredictor(
        n_q=7, H=a["H"], ctx_dim=ctx.shape[-1],
        hidden=a["bc_hidden"], n_layers=a["bc_layers"],
        activation="sin",
    ).to(device)
    state = ck["ema_state"] if "ema_state" in ck else ck["model_state"]
    model.load_state_dict(state); model.eval()

    @torch.no_grad()
    def _sample():
        return model(ctx)

    q_gen = _sample()
    z_flat = z_e.unsqueeze(1).expand(-1, H1, -1).reshape(-1, 1)
    p_gen = arm.F(q_gen.reshape(-1, 7), z_flat).reshape(n, H1, 3)
    return _sample, q_gen, p_gen, arm, arm_a, a


def _run_dp(ckpt_path: str, n: int, device: str, ctx: torch.Tensor, z_e: torch.Tensor,
            n_inference_steps: int, channel: bool):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["args"]; s1 = ck["stage1_args"]
    perfect_fk = bool(a.get("perfect_fk", False))
    arm_a, arm = _build_arms(device, perfect_fk, s1)
    H1 = a["H"] + 1; n_q = 7
    if channel:
        model, scheduler = make_official_diffusion_policy(
            n_q=n_q + ctx.shape[-1], global_cond_dim=None,
            down_dims=list(a["down_dims"]),
            diffusion_step_embed_dim=a["diff_step_embed"],
            n_train_timesteps=a["dp_train_timesteps"],
        )
    else:
        model, scheduler = make_official_diffusion_policy(
            n_q=n_q, global_cond_dim=ctx.shape[-1],
            down_dims=list(a["down_dims"]),
            diffusion_step_embed_dim=a["diff_step_embed"],
            n_train_timesteps=a["dp_train_timesteps"],
        )
    model = model.to(device)
    model.load_state_dict(ck["ema_state"]); model.eval()

    if channel:
        ctx_per_h = ctx.unsqueeze(1).expand(-1, H1, -1)
        @torch.no_grad()
        def _sample():
            return channel_concat_dp_sample(
                model, scheduler, batch_size=n, horizon=H1, n_q=n_q,
                ctx_per_h=ctx_per_h, device=device,
                n_inference_steps=n_inference_steps,
            )
    else:
        @torch.no_grad()
        def _sample():
            return official_dp_sample(
                model, scheduler, batch_size=n, horizon=H1, n_q=n_q,
                ctx=ctx, device=device,
                n_inference_steps=n_inference_steps,
            )

    q_gen = _sample()
    z_flat = z_e.unsqueeze(1).expand(-1, H1, -1).reshape(-1, 1)
    p_gen = arm.F(q_gen.reshape(-1, 7), z_flat).reshape(n, H1, 3)
    return _sample, q_gen, p_gen, arm, arm_a, a, model, scheduler


def _run_dp_c(ckpt_path: str, n: int, device: str, ctx: torch.Tensor, z_e: torch.Tensor,
              p_targets: torch.Tensor, p_starts: torch.Tensor,
              n_inference_steps: int, alpha_g: float, alpha_s: float, alpha_v: float):
    """DP-C: channel ckpt + classifier guidance (re-uses DP-A ckpt)."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["args"]; s1 = ck["stage1_args"]
    perfect_fk = bool(a.get("perfect_fk", False))
    arm_a, arm = _build_arms(device, perfect_fk, s1)
    H1 = a["H"] + 1; n_q = 7
    model, scheduler = make_official_diffusion_policy(
        n_q=n_q + ctx.shape[-1], global_cond_dim=None,
        down_dims=list(a["down_dims"]),
        diffusion_step_embed_dim=a["diff_step_embed"],
        n_train_timesteps=a["dp_train_timesteps"],
    )
    model = model.to(device)
    model.load_state_dict(ck["ema_state"]); model.eval()

    ctx_per_h = ctx.unsqueeze(1).expand(-1, H1, -1)
    grh = list(range(H1 - H1 // 2, H1)); sah = [0]

    def _sample():
        return channel_concat_dp_sample_guided(
            model, scheduler, batch_size=n, horizon=H1, n_q=n_q,
            ctx_per_h=ctx_per_h, device=device,
            n_inference_steps=n_inference_steps,
            fk_fn=arm.F, z_e_per_traj=z_e,
            p_target=p_targets, alpha_g=alpha_g, h_indices_goal=grh,
            p_start_anchor=p_starts, alpha_s=alpha_s, h_indices_start=sah,
            alpha_v=alpha_v,
        )

    q_gen = _sample()
    z_flat = z_e.unsqueeze(1).expand(-1, H1, -1).reshape(-1, 1)
    p_gen = arm.F(q_gen.reshape(-1, 7), z_flat).reshape(n, H1, 3)
    return _sample, q_gen, p_gen, arm, arm_a, a


def _run_projected(ckpt_path: str, n: int, device: str, ctx: torch.Tensor, z_e: torch.Tensor,
                   n_inference_steps: int):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["args"]; s1 = ck["stage1_args"]
    perfect_fk = bool(a.get("perfect_fk", False))
    arm_a, arm = _build_arms(device, perfect_fk, s1)
    H1 = a["H"] + 1; n_q, n_p = 7, 3
    model, scheduler = make_official_diffusion_policy(
        n_q=n_q + n_p, global_cond_dim=ctx.shape[-1],
        down_dims=list(a["down_dims"]),
        diffusion_step_embed_dim=a["diff_step_embed"],
        n_train_timesteps=a["dp_train_timesteps"],
    )
    model = model.to(device)
    model.load_state_dict(ck["ema_state"]); model.eval()

    @torch.no_grad()
    def _sample():
        return official_dp_sample(
            model, scheduler, batch_size=n, horizon=H1, n_q=n_q + n_p,
            ctx=ctx, device=device, n_inference_steps=n_inference_steps,
        )

    x_amb = _sample()
    q_gen = x_amb[..., :n_q]
    z_flat = z_e.unsqueeze(1).expand(-1, H1, -1).reshape(-1, 1)
    # Reported path uses post-projection p (Christopher24 final state).
    p_proj = arm.F(q_gen.reshape(-1, 7), z_flat).reshape(n, H1, 3)
    return _sample, q_gen, p_proj, arm, arm_a, a


def _run_ours(ckpt_path: str, n: int, device: str, p_targets: torch.Tensor,
              p_starts: torch.Tensor, z_e: torch.Tensor, n_steps: int,
              alpha_goal: float, alpha_start: float, alpha_vel: float):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["args"]
    # Ours-Analytic ckpts have no learned residual → no stage1_args.
    has_stage1 = "stage1_args" in ck
    s1 = ck["stage1_args"] if has_stage1 else {}
    perfect_fk = (not has_stage1) or bool(a.get("perfect_fk", False))
    z_eval_last = a["z_eval"][-1] if "z_eval" in a else a.get("z_max", 0.20)
    arm_a, arm = _build_arms(device, perfect_fk, s1,
                              tool_z_max=max(a["z_max"], z_eval_last) + 0.05)
    if "metric" in a:
        arm.metric = a["metric"]                                              # ensure metric matches training
    schedule = LinearBetaSchedule(beta_0=a["beta_0"], beta_f=a["beta_f"], t0=0.0, tf=1.0)
    limiting = WrappedNormalFranka7DoF(arm, mean_q=list(a["limiting_mean_q"]),
                                       scale=a["limiting_scale"],
                                       z_e_range=(a["z_min"], a["z_max"]))
    sde = LangevinSDE(arm, schedule, limiting)
    H1 = a["H"] + 1; d = arm.ambient_dim

    use_p_start = bool(a.get("use_p_start_cond", False))
    GOAL_DIM = 6 if use_p_start else 3
    cond_inj = a.get("cond_injection", "global")
    net = TrajectoryScoreNetUNet(
        arm, H=a["H"], down_dims=tuple(a["down_dims"]),
        diffusion_step_embed_dim=a["diff_step_embed"],
        n_groups=a["unet_groups"], kernel_size=a["unet_kernel"],
        t_scale=a["t_scale"], goal_cond_dim=GOAL_DIM,
        cond_injection=cond_inj,
    ).to(device)
    net.load_state_dict(ck["ema_state"]); net.eval()
    score_fn = TrajectoryScaledScoreFn(net, sde)

    if use_p_start:
        cond_for_eval = torch.cat([p_targets, p_starts], dim=-1)
    else:
        cond_for_eval = p_targets

    grh = list(range(H1 - H1 // 2, H1)); sah = [0]
    z_lim = z_e.unsqueeze(1).expand(-1, H1, -1).reshape(n * H1, -1)

    def _sample():
        tau_T = limiting.sample(n * H1, device=device, z_e=z_lim).reshape(n, H1, d)
        with torch.no_grad():
            return traj_reverse_grw(
                sde, score_fn, tau_T, n_steps=n_steps, eps=a["eps"],
                goal_cond=cond_for_eval, guidance_scale=0.0,
                goal_residual_alpha=alpha_goal, goal_residual_h=grh,
                p_start=p_starts,
                start_anchor_alpha=alpha_start, start_anchor_h=sah,
                smoothness_alpha_vel=alpha_vel,
                smoothness_alpha_acc=0.0,
            )

    tau = _sample()
    q_gen = tau[..., :7]
    p_gen = tau[..., 7:10]
    return _sample, q_gen, p_gen, arm, arm_a, a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=256, help="trajectories for path metrics")
    p.add_argument("--n-time-batch", type=int, default=64,
                   help="batch size used for inference timing")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--z-eval", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str,
                   default="outputs/diagnostic/inference_timing_path.json")
    p.add_argument("--dp-inference-steps", type=int, default=100)
    p.add_argument("--ours-sample-steps", type=int, default=200)
    p.add_argument("--alpha-g", type=float, default=100.0)
    p.add_argument("--alpha-s", type=float, default=100.0)
    p.add_argument("--alpha-v", type=float, default=5.0)
    p.add_argument("--dp-c-alpha-g", type=float, default=30.0)
    p.add_argument("--dp-c-alpha-s", type=float, default=30.0)
    p.add_argument("--dp-c-alpha-v", type=float, default=3.0)
    args = p.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    torch.manual_seed(args.seed)
    n = args.n; n_t = args.n_time_batch; H = 15; H1 = H + 1

    box_lo = torch.tensor([0.40, -0.05, 0.40], device=device)
    box_hi = torch.tensor([0.50,  0.05, 0.50], device=device)
    torch.manual_seed(args.seed + 1000)
    p_targets = box_lo + (box_hi - box_lo) * torch.rand(n, 3, device=device)
    torch.manual_seed(args.seed + 2000)
    p_starts = box_lo + (box_hi - box_lo) * torch.rand(n, 3, device=device)
    z_e = torch.full((n, 1), args.z_eval, device=device)
    ctx = torch.cat([p_targets, p_starts, z_e], dim=-1)

    # Timing-only smaller batch (so we measure per-batch latency rather than total work)
    p_targets_t = p_targets[:n_t]; p_starts_t = p_starts[:n_t]; z_e_t = z_e[:n_t]
    ctx_t = torch.cat([p_targets_t, p_starts_t, z_e_t], dim=-1)

    methods = {}

    def _record(name, sample_fn_full, sample_fn_timing, q_traj, p_traj, ctx_keys: dict):
        mean_t, std_t = _time_sampler(sample_fn_timing,
                                       warmup=args.warmup, repeats=args.repeats)
        path = _path_metrics(p_traj, q_traj)
        # Reach error + success for the same n trajectories.
        p_end = p_traj[:, -1]
        pos_err = (p_end - p_targets).norm(dim=-1)
        path["pos_err_mean"] = pos_err.mean().item()
        path["succ_at_5cm"]  = (pos_err < 0.05).float().mean().item()
        path["succ_at_10cm"] = (pos_err < 0.10).float().mean().item()
        methods[name] = {
            "inference_time_per_batch_mean_s": mean_t,
            "inference_time_per_batch_std_s":  std_t,
            "batch_size_for_timing":           n_t,
            "throughput_traj_per_s":           n_t / mean_t,
            **path,
            **ctx_keys,
        }
        print(f"{name:>20s}  time={mean_t*1e3:7.1f}±{std_t*1e3:5.1f}ms/batch  "
              f"thru={n_t/mean_t:7.1f} traj/s  "
              f"ee_path={path['ee_path_len_mean_m']*100:5.1f}cm  "
              f"ee_str={path['ee_straightness_mean']:.3f}  "
              f"ee_excess={path['ee_excess_mean']:.3f}  "
              f"q_str={path['q_straightness_mean']:.3f}  "
              f"pos_err={path['pos_err_mean']*100:5.1f}cm")

    print(f"\nz_e={args.z_eval}  n={n}  timing_batch={n_t}  "
          f"DP_steps={args.dp_inference_steps}  Ours_steps={args.ours_sample_steps}\n")
    print(f"{'method':>20s}  {'time (ms/batch)':>18s}  {'throughput':>14s}  "
          f"{'EE path':>7s}  {'EE str':>7s}  {'EE exc':>9s}  {'q str':>6s}  {'pos_err':>9s}")

    # ----- Demo reference -----
    arm_a_ref = Franka7DoF(urdf_path=URDF, end_link="panda_hand", tool_z_max=0.30)
    arm_a_ref._ensure_chain(torch.zeros(1, 7, device=device))
    s1_ck = torch.load("outputs/franka_stage1/delta_phi.pt",
                       map_location=device, weights_only=False)
    delta_net = DeltaResidualMLP(n_q=7, n_p=3, n_z=1,
                                  hidden=s1_ck["args"]["hidden"],
                                  n_layers=s1_ck["args"]["n_layers"],
                                  activation=torch.nn.Softplus,
                                  final_init_scale=1e-3).to(device)
    delta_net.load_state_dict(s1_ck["delta_net_state"]); delta_net.eval()
    arm_ref = LearnedSelfModelFranka7DoF(
        delta_net=delta_net, urdf_path=URDF, end_link="panda_hand", tool_z_max=0.30)
    arm_ref._ensure_chain(torch.zeros(1, 7, device=device))
    demo = FrankaBimodalReachingDemo(
        manifold=arm_ref, ik_arm=arm_a_ref, H=H,
        q_rest_A=[ 0.6, -0.3, 0.0, -1.7, 0.0, 1.4, 0.0],
        q_rest_B=[-0.6, -0.3, 0.0, -1.7, 0.0, 1.4, 0.0],
        p_box_lo=(0.40, -0.05, 0.40), p_box_hi=(0.50, 0.05, 0.50),
        z_e_range=(args.z_eval, args.z_eval),
        branch_p_A=0.5, jitter_q=0.0, n_ik_steps=120,
    )
    x_demo, _, _, _, _ = demo.sample(n, device=device,
                                      p_target=p_targets, p_start=p_starts)
    q_demo = x_demo[..., :7]
    p_demo = x_demo[..., 7:10]
    demo_path = _path_metrics(p_demo, q_demo)
    methods["Demo"] = {
        "inference_time_per_batch_mean_s": float("nan"),
        "inference_time_per_batch_std_s":  float("nan"),
        "throughput_traj_per_s":           float("nan"),
        **demo_path,
        "note": "IK demo reference (n_ik_steps=120, jitter=0)",
    }
    print(f"{'Demo (reference)':>20s}                                      "
          f"            ee_path={demo_path['ee_path_len_mean_m']*100:5.1f}cm  "
          f"ee_str={demo_path['ee_straightness_mean']:.3f}  "
          f"ee_excess={demo_path['ee_excess_mean']:.3f}  "
          f"q_str={demo_path['q_straightness_mean']:.3f}")

    # ----- BC -----
    fn_full, q, ptraj, _, _, _ = _run_bc(
        "outputs/franka_baseline_bc/ckpt.pt", n, device, ctx, z_e)
    fn_t, q_t, _, arm_t, _, _ = _run_bc(
        "outputs/franka_baseline_bc/ckpt.pt", n_t, device, ctx_t, z_e_t)
    _record("BC", fn_full, fn_t, q, ptraj,
            ctx_keys={"n_inference_steps": 1, "ckpt": "franka_baseline_bc"})

    # ----- DP-canonical (global cond) -----
    fn_full, q, ptraj, _, _, _, _, _ = _run_dp(
        "outputs/franka_baseline_dp_official/ckpt.pt", n, device, ctx, z_e,
        n_inference_steps=args.dp_inference_steps, channel=False)
    fn_t, _, _, _, _, _, _, _ = _run_dp(
        "outputs/franka_baseline_dp_official/ckpt.pt", n_t, device, ctx_t, z_e_t,
        n_inference_steps=args.dp_inference_steps, channel=False)
    _record("DP-canonical", fn_full, fn_t, q, ptraj,
            ctx_keys={"n_inference_steps": args.dp_inference_steps,
                      "ckpt": "franka_baseline_dp_official"})

    # ----- DP-A (channel cond, no guidance) -----
    fn_full, q, ptraj, _, _, _, _, _ = _run_dp(
        "outputs/franka_baseline_dp_official_channel/ckpt.pt", n, device, ctx, z_e,
        n_inference_steps=args.dp_inference_steps, channel=True)
    fn_t, _, _, _, _, _, _, _ = _run_dp(
        "outputs/franka_baseline_dp_official_channel/ckpt.pt", n_t, device, ctx_t, z_e_t,
        n_inference_steps=args.dp_inference_steps, channel=True)
    _record("DP-A", fn_full, fn_t, q, ptraj,
            ctx_keys={"n_inference_steps": args.dp_inference_steps,
                      "ckpt": "franka_baseline_dp_official_channel"})

    # ----- DP-C (channel cond + classifier guidance) -----
    fn_full, q, ptraj, _, _, _ = _run_dp_c(
        "outputs/franka_baseline_dp_official_channel/ckpt.pt", n, device,
        ctx, z_e, p_targets, p_starts,
        n_inference_steps=args.dp_inference_steps,
        alpha_g=args.dp_c_alpha_g, alpha_s=args.dp_c_alpha_s,
        alpha_v=args.dp_c_alpha_v)
    fn_t, _, _, _, _, _ = _run_dp_c(
        "outputs/franka_baseline_dp_official_channel/ckpt.pt", n_t, device,
        ctx_t, z_e_t, p_targets_t, p_starts_t,
        n_inference_steps=args.dp_inference_steps,
        alpha_g=args.dp_c_alpha_g, alpha_s=args.dp_c_alpha_s,
        alpha_v=args.dp_c_alpha_v)
    _record("DP-C", fn_full, fn_t, q, ptraj,
            ctx_keys={"n_inference_steps": args.dp_inference_steps,
                      "alpha_g": args.dp_c_alpha_g, "alpha_s": args.dp_c_alpha_s,
                      "alpha_v": args.dp_c_alpha_v,
                      "ckpt": "franka_baseline_dp_official_channel + guidance"})

    # ----- Projected -----
    fn_full, q, ptraj, _, _, _ = _run_projected(
        "outputs/franka_baseline_projected/ckpt.pt", n, device, ctx, z_e,
        n_inference_steps=args.dp_inference_steps)
    fn_t, _, _, _, _, _ = _run_projected(
        "outputs/franka_baseline_projected/ckpt.pt", n_t, device, ctx_t, z_e_t,
        n_inference_steps=args.dp_inference_steps)
    _record("Projected", fn_full, fn_t, q, ptraj,
            ctx_keys={"n_inference_steps": args.dp_inference_steps,
                      "ckpt": "franka_baseline_projected"})

    # ----- Ours-Analytic -----
    fn_full, q, ptraj, _, _, _ = _run_ours(
        "outputs/franka_traj_unet_v2_analytic/ckpt_riemannian.pt", n, device,
        p_targets, p_starts, z_e,
        n_steps=args.ours_sample_steps,
        alpha_goal=args.alpha_g, alpha_start=args.alpha_s, alpha_vel=args.alpha_v)
    fn_t, _, _, _, _, _ = _run_ours(
        "outputs/franka_traj_unet_v2_analytic/ckpt_riemannian.pt", n_t, device,
        p_targets_t, p_starts_t, z_e_t,
        n_steps=args.ours_sample_steps,
        alpha_goal=args.alpha_g, alpha_start=args.alpha_s, alpha_vel=args.alpha_v)
    _record("Ours-Analytic", fn_full, fn_t, q, ptraj,
            ctx_keys={"n_steps": args.ours_sample_steps,
                      "alpha_g": args.alpha_g, "alpha_s": args.alpha_s,
                      "alpha_v": args.alpha_v,
                      "ckpt": "franka_traj_unet_v2_analytic"})

    # ----- Ours-V2 (learned residual) -----
    fn_full, q, ptraj, _, _, _ = _run_ours(
        "outputs/franka_traj_unet_v2/ckpt_riemannian.pt", n, device,
        p_targets, p_starts, z_e,
        n_steps=args.ours_sample_steps,
        alpha_goal=args.alpha_g, alpha_start=args.alpha_s, alpha_vel=args.alpha_v)
    fn_t, _, _, _, _, _ = _run_ours(
        "outputs/franka_traj_unet_v2/ckpt_riemannian.pt", n_t, device,
        p_targets_t, p_starts_t, z_e_t,
        n_steps=args.ours_sample_steps,
        alpha_goal=args.alpha_g, alpha_start=args.alpha_s, alpha_vel=args.alpha_v)
    _record("Ours-V2", fn_full, fn_t, q, ptraj,
            ctx_keys={"n_steps": args.ours_sample_steps,
                      "alpha_g": args.alpha_g, "alpha_s": args.alpha_s,
                      "alpha_v": args.alpha_v,
                      "ckpt": "franka_traj_unet_v2"})

    out_path.write_text(json.dumps(methods, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
