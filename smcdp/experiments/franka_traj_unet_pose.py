"""Stage 5 (pose-extended) — Ours-UNet GOAL-CONDITIONAL trajectory diffusion
on Franka 7-DoF with FULL SE(3) target conditioning  (extension.tex Sec. 6–8).

Pose extension of `franka_traj_unet.py`:
  - Manifold      : LearnedSelfModelFranka7DoFPose  (analytic FK + frozen ξ_φ)
  - Conditioning  : c = (T_start, T_target, z_e),  T_• ∈ R^7 storage form
                    → goal_cond_dim = 14
  - Score         : chart-form output via TrajectoryScoreNetUNetPose
  - Loss          : traj_dsm_pose_loss  (chart-G DSM, extension.tex Eq. (37))
  - Sampler       : traj_reverse_grw_pose (chart-form, retraction via H_φ^pose)

Run:
    python -m smcdp.experiments.franka_traj_unet_pose \
        --stage1-pose-ckpt outputs/franka_stage1_pose/xi_phi.pt
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from tqdm import tqdm
import pybullet_data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from smcdp.sde import LinearBetaSchedule
from smcdp.manifolds_pose import Franka7DoFPose
from smcdp.franka.self_model_pose import (
    PoseResidualMLP, LearnedSelfModelFranka7DoFPose,
)
from smcdp.franka.demo_gen_pose import FrankaBimodalReachingDemoPose
from smcdp.trajectories_pose import (
    TrajectoryScoreNetUNetPose, TrajectoryScaledScoreFnPose,
    PoseLangevinSDE,
    traj_dsm_pose_loss, traj_reverse_grw_pose,
)
from smcdp.lie_se3 import (
    log_relative_Rp, quat_to_R, R_to_quat, exp_SE3, compose_Rp,
    pose7_to_Rp,
)


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-pose-ckpt", type=str,
                   default="outputs/franka_stage1_pose/xi_phi.pt",
                   help="Stage-1 pose self-model ckpt (xi_phi.pt).")
    p.add_argument("--H", type=int, default=15)
    # demo
    p.add_argument("--q-rest-A", type=float, nargs=7,
                   default=[+0.6, -0.3, 0.0, -1.7, 0.0, 1.4, 0.0])
    p.add_argument("--q-rest-B", type=float, nargs=7,
                   default=[-0.6, -0.3, 0.0, -1.7, 0.0, 1.4, 0.0])
    p.add_argument("--p-box-lo", type=float, nargs=3, default=[0.40, -0.05, 0.40])
    p.add_argument("--p-box-hi", type=float, nargs=3, default=[0.50, +0.05, 0.50])
    p.add_argument("--branch-p-A", type=float, default=0.5)
    p.add_argument("--jitter-q", type=float, default=0.05)
    p.add_argument("--n-ik-steps", type=int, default=10)
    p.add_argument("--target-perturb-deg", type=float, default=30.0,
                   help="Target rotation perturbation: each axis-angle component "
                         "~ Uniform[-deg, +deg].")
    p.add_argument("--R-anchor-aa", type=float, nargs=3,
                   default=[3.14159265, 0.0, 0.0],
                   help="Default end-effector orientation (axis-angle, rad).")
    p.add_argument("--z-min", type=float, default=0.05)
    p.add_argument("--z-max", type=float, default=0.15)
    # weights for pose metric
    p.add_argument("--sigma-p", type=float, default=0.05,
                   help="Position scale for W_p = σ_p^{-2} (m).  Default 0.05m "
                        "(=5cm) keeps W_p = 400 → cond(G_pose) ~10² — "
                        "numerically stable for Langevin forward (cf. "
                        "noise_stationary_fix.md Fix 1).  Use 0.01 (1cm) only "
                        "after enabling Tikhonov + confining (Fix 2 + 3).")
    p.add_argument("--sigma-R", type=float, default=0.1,
                   help="Rotation scale for W_R = σ_R^{-2} (rad).")
    # net + training
    p.add_argument("--steps", type=int, default=15_000)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--down-dims", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--diff-step-embed", type=int, default=128)
    p.add_argument("--unet-groups", type=int, default=8)
    p.add_argument("--unet-kernel", type=int, default=3)
    p.add_argument("--t-scale", type=float, default=1000.0)
    p.add_argument("--beta-0", type=float, default=0.001)
    p.add_argument("--beta-f", type=float, default=4.0)
    p.add_argument("--eps", type=float, default=2e-4)
    p.add_argument("--n-grw-steps", type=int, default=10)
    p.add_argument("--n-sample-steps", type=int, default=200)
    p.add_argument("--metric", type=str, default="riemannian",
                   choices=["riemannian", "chart_euclidean"])
    p.add_argument("--weight", type=str, default="sigma2",
                   choices=["sigma2", "beta", "none"])
    p.add_argument("--limiting-mean-q", type=float, nargs=7,
                   default=[0.0, -0.3, 0.0, -1.7, 0.0, 1.4, 0.0],
                   help="Legacy single μ_q anchor; ignored when --method-a is set.")
    p.add_argument("--limiting-scale", type=float, default=None,
                   help="σ_K for sampling init (=√τ_brown(K)).  None (default) "
                        "auto-calibrates from the schedule (Method A correct value).")
    p.add_argument("--method-a", action="store_true",
                   help="Method A (modificatin.md): pure-Brownian forward, "
                        "per-trajectory q_init at sampling, brownian-mode "
                        "proxy_std.  Drift OFF + Fix 1 (σ_p=0.05) baseline.")
    p.add_argument("--proxy-std-mode", type=str, default=None,
                   choices=[None, "ou", "brownian"],
                   help="proxy_std calibration: 'brownian' (√I, Method A) or "
                        "'ou' (√(1−exp(−I)), legacy VP-SDE).  None auto-selects "
                        "'brownian' under --method-a, else 'ou'.")
    p.add_argument("--forward-langevin-drift", action="store_true",
                   help="Enable Langevin forward drift -½β G⁻¹∇U "
                        "(extension.tex Eq. 15-17).  Stable when σ_p ≥ 0.05 "
                        "+ Tikhonov regularization (default).  Off by default "
                        "for backward compatibility.")
    # noise_stationary_fix.md Fix 2 + Fix 3 (opt-in)
    p.add_argument("--tikhonov-frac", type=float, default=0.0,
                   help="Fix 2: adaptive Tikhonov on G_pose: "
                        "G ← G + λ(q) I,  λ(q) = c · tr(G)/n_q.  "
                        "0.0 = legacy fixed jitter only.  "
                        "Recommended c ∈ [1e-4, 1e-2] when σ_p ≥ 0.05.  "
                        "noise_stationary_fix.md Sec. 2.2.")
    p.add_argument("--confining-kappa", type=float, default=0.0,
                   help="Fix 3 (Option B): soft anchor-metric confining "
                        "potential strength κ in U_total = (1/2γ²)(q-μ)^T Ĝ "
                        "(q-μ) + κ U_box.  0.0 disables Fix 3 (legacy V).  "
                        "Recommended κ ∈ [1e2, 1e4].  Forward Langevin drift "
                        "is required (--forward-langevin-drift).  "
                        "noise_stationary_fix.md Sec. 2.3.")
    p.add_argument("--confining-epsilon-frac", type=float, default=0.05,
                   help="Fix 3: ε margin = epsilon_frac · (q_max − q_min) "
                        "before the joint-range box potential activates.  "
                        "Default 5%% of range (per doc).")
    p.add_argument("--cond-injection", type=str, default="channel",
                   choices=["global", "channel"])
    p.add_argument("--endpoint-weight", type=float, default=1.0)
    p.add_argument("--cond-drop-prob", type=float, default=0.10)
    p.add_argument("--guidance-scale", type=float, default=0.0)
    # Stage 6' pose reward guidance (extension.tex Sec. 9)
    p.add_argument("--start-alpha-p", type=float, default=0.0,
                   help="Start anchor position weight α_p^s.")
    p.add_argument("--start-alpha-R", type=float, default=0.0,
                   help="Start anchor rotation weight α_R^s.")
    p.add_argument("--goal-alpha-p", type=float, default=0.0,
                   help="Goal anchor position weight α_p^g (per V2 default 100).")
    p.add_argument("--goal-alpha-R", type=float, default=0.0,
                   help="Goal anchor rotation weight α_R^g (per V2 default 100).")
    p.add_argument("--smoothness-alpha-vel", type=float, default=0.0)
    p.add_argument("--smoothness-alpha-acc", type=float, default=0.0)
    p.add_argument("--goal-h-mask", type=str, default="last_only",
                   choices=["all", "last_only", "last_half", "last_quarter"],
                   help="Timestep mask schedule for goal anchor (extension.tex Eq. 49-52).")
    # eval
    p.add_argument("--n-eval-per-z", type=int, default=64)
    p.add_argument("--n-targets-per-z", type=int, default=8)
    p.add_argument("--success-pos", type=float, default=0.02,
                   help="Success threshold on position (m).")
    p.add_argument("--success-rot", type=float, default=0.262,           # 15°
                   help="Success threshold on rotation (rad).")
    p.add_argument("--z-eval", type=float, nargs="+",
                   default=[0.05, 0.10, 0.15, 0.20])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="outputs/franka_traj_unet_pose")
    p.add_argument("--resume-from", type=str, default=None)
    return p.parse_args()


def _load_stage1_pose(arm: Franka7DoFPose, ckpt_path: str, device, dtype):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args1 = ckpt["args"]
    residual_net = PoseResidualMLP(
        n_q=7, n_z=1, hidden=args1["hidden"], n_layers=args1["n_layers"],
        activation=torch.nn.Softplus, final_init_scale=1e-3,
        output_omega=True,
    ).to(device=device, dtype=dtype)
    residual_net.load_state_dict(ckpt["residual_net_state"])
    residual_net.eval()
    print(f"[stage1-pose] loaded ξ_φ from {ckpt_path}: {ckpt['metrics']}")
    return residual_net


def _ema_update(ema_model, model, decay):
    with torch.no_grad():
        for ep, mp in zip(ema_model.parameters(), model.parameters()):
            ep.data.mul_(decay).add_(mp.data, alpha=1.0 - decay)
        for eb, mb in zip(ema_model.buffers(), model.buffers()):
            eb.data.copy_(mb.data)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out_dir={out_dir}")

    # --- manifold ---
    arm_analytic = Franka7DoFPose(
        urdf_path=URDF, end_link="panda_hand", tool_z_max=args.z_max,
        sigma_p=args.sigma_p, sigma_R=args.sigma_R, metric=args.metric,
        tikhonov_frac=args.tikhonov_frac,
    )
    arm_analytic._ensure_chain(torch.zeros(1, 7, device=device))
    residual_net = _load_stage1_pose(arm_analytic, args.stage1_pose_ckpt, device, dtype)
    arm = LearnedSelfModelFranka7DoFPose(
        residual_net=residual_net, urdf_path=URDF, end_link="panda_hand",
        tool_z_max=args.z_max, sigma_p=args.sigma_p, sigma_R=args.sigma_R,
        metric=args.metric,
    )
    # `LearnedSelfModelFranka7DoFPose.__init__` may not forward tikhonov_frac;
    # set the attribute directly so all downstream G_pose_chol calls inherit it.
    arm.tikhonov_frac = float(args.tikhonov_frac)
    arm._ensure_chain(torch.zeros(1, 7, device=device))

    # --- demo gen ---
    target_perturb_rad = args.target_perturb_deg * 3.14159265 / 180.0
    demo = FrankaBimodalReachingDemoPose(
        manifold=arm, ik_arm=arm_analytic, H=args.H,
        q_rest_A=args.q_rest_A, q_rest_B=args.q_rest_B,
        p_box_lo=args.p_box_lo, p_box_hi=args.p_box_hi,
        z_e_range=(args.z_min, args.z_max),
        branch_p_A=args.branch_p_A, jitter_q=args.jitter_q,
        n_ik_steps=args.n_ik_steps,
        R_anchor_axis_angle=args.R_anchor_aa,
        target_perturb_rad=target_perturb_rad,
    )

    # --- SDE + score net ---
    schedule = LinearBetaSchedule(beta_0=args.beta_0, beta_f=args.beta_f, tf=1.0)
    sde = PoseLangevinSDE(
        arm, schedule,
        limiting_q_mean=torch.tensor(args.limiting_mean_q, dtype=dtype),
        limiting_scale=args.limiting_scale,
        forward_langevin_drift=args.forward_langevin_drift,
        confining_kappa=args.confining_kappa,
        confining_epsilon_frac=args.confining_epsilon_frac,
    )
    net = TrajectoryScoreNetUNetPose(
        manifold=arm, H=args.H,
        down_dims=tuple(args.down_dims),
        diffusion_step_embed_dim=args.diff_step_embed,
        n_groups=args.unet_groups, kernel_size=args.unet_kernel,
        cond_predict_scale=False, t_scale=args.t_scale,
        goal_cond_dim=14, cond_injection=args.cond_injection,
    ).to(device=device, dtype=dtype)
    ema_net = copy.deepcopy(net).to(device=device, dtype=dtype)
    for p in ema_net.parameters():
        p.requires_grad_(False)

    # Method A vs legacy proxy_std mode resolution
    if args.proxy_std_mode is None:
        proxy_std_mode = "brownian" if args.method_a else "ou"
    else:
        proxy_std_mode = args.proxy_std_mode
    print(f"[Method A={'ON' if args.method_a else 'OFF'}]  proxy_std_mode={proxy_std_mode}")

    score_fn = TrajectoryScaledScoreFnPose(net, sde, std_trick=True,
                                            proxy_std_mode=proxy_std_mode)
    score_fn_ema = TrajectoryScaledScoreFnPose(ema_net, sde, std_trick=True,
                                                proxy_std_mode=proxy_std_mode)

    optim = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-6)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"score net params: {n_params/1e6:.2f}M")

    if args.resume_from is not None:
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["net"])
        ema_net.load_state_dict(ckpt["ema_net"])
        optim.load_state_dict(ckpt["optim"])
        print(f"[resume] loaded {args.resume_from}")

    # --- training ---
    losses_log: list[float] = []
    pbar = tqdm(range(args.steps), desc="ours-V2 pose train")
    for step in pbar:
        x, branch_A, z_e, T_target, T_start = demo.sample(args.batch, device=device, dtype=dtype)
        goal_cond = torch.cat([T_start, T_target], dim=-1)                    # (B, 14)
        # warmup
        if step < args.warmup_steps:
            for g in optim.param_groups:
                g["lr"] = args.lr * (step + 1) / args.warmup_steps

        loss = traj_dsm_pose_loss(
            score_fn, sde, x, eps=args.eps,
            weight=args.weight, n_grw_steps=args.n_grw_steps,
            goal_cond=goal_cond, cond_drop_prob=args.cond_drop_prob,
            endpoint_weight=args.endpoint_weight,
            proxy_std_mode=proxy_std_mode,
        )
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optim.step()
        _ema_update(ema_net, net, args.ema)

        losses_log.append(loss.item())
        if step % 50 == 0:
            pbar.set_postfix(loss=f"{loss.item():.3e}")

    # --- save ckpt ---
    torch.save({
        "args": vars(args),
        "net": net.state_dict(),
        "ema_net": ema_net.state_dict(),
        "optim": optim.state_dict(),
        "n_params": n_params,
    }, out_dir / "ours_v2_pose.pt")
    print(f"saved {out_dir / 'ours_v2_pose.pt'}")

    # --- eval ---
    ema_net.eval()
    metrics = {"per_z": []}
    for z_val in args.z_eval:
        z_e = torch.full((args.n_eval_per_z, 1), z_val, device=device, dtype=dtype)
        # Sample fresh targets
        x_demo, _, _, T_target, T_start = demo.sample(
            args.n_eval_per_z, device=device, dtype=dtype,
        )
        goal_cond = torch.cat([T_start, T_target], dim=-1)
        # Decode T_start, T_target for Stage 6' anchor guidance.
        T_start_Rp = pose7_to_Rp(T_start) if (args.start_alpha_p > 0 or args.start_alpha_R > 0) else None
        T_target_Rp = pose7_to_Rp(T_target) if (args.goal_alpha_p > 0 or args.goal_alpha_R > 0) else None
        # Build goal mask schedule (extension.tex Eq. 49-52)
        H1 = args.H + 1
        if args.goal_h_mask == "all":
            goal_h = list(range(H1))
        elif args.goal_h_mask == "last_only":
            goal_h = [H1 - 1]
        elif args.goal_h_mask == "last_half":
            goal_h = list(range(H1 // 2, H1))
        else:  # "last_quarter"
            goal_h = list(range(3 * H1 // 4, H1))

        # Method A: per-trajectory q_init from demo's q_0 (T_start의 IK-equivalent
        # since T_start = T_φ(q_demo[0]) by construction).  Legacy: uses single μ_q.
        q_init_eval = (x_demo[:, 0, :arm.n_q].detach().to(device=device, dtype=dtype)
                        if args.method_a else None)
        samples = traj_reverse_grw_pose(
            sde, score_fn_ema, n_samples=args.n_eval_per_z, H=args.H,
            n_steps=args.n_sample_steps, goal_cond=goal_cond, z_e=z_e,
            limiting_q_mean=torch.tensor(args.limiting_mean_q),
            q_init=q_init_eval,
            limiting_scale=args.limiting_scale,                                  # None → auto √τ_brown(K)
            eps=args.eps, device=device, dtype=dtype,
            T_start_Rp=T_start_Rp,
            start_alpha_p=args.start_alpha_p, start_alpha_R=args.start_alpha_R,
            start_h_indices=[0],
            T_target_Rp=T_target_Rp,
            goal_alpha_p=args.goal_alpha_p, goal_alpha_R=args.goal_alpha_R,
            goal_h_indices=goal_h,
            smoothness_alpha_vel=args.smoothness_alpha_vel,
            smoothness_alpha_acc=args.smoothness_alpha_acc,
        )
        # Endpoint pose error
        x_H = samples[:, -1, :]                                                # (B, 15)
        q_H, q_R_H, p_H, _ = arm.split_x(x_H)
        R_H = quat_to_R(q_R_H)
        R_target = quat_to_R(T_target[..., :4])
        p_target = T_target[..., 4:]
        e = log_relative_Rp(R_H, p_H, R_target, p_target)
        e_p = e[..., :3].norm(dim=-1)
        e_R = e[..., 3:].norm(dim=-1)
        succ = ((e_p < args.success_pos) & (e_R < args.success_rot)).float().mean().item()
        manif_adh = arm.constraint(samples.reshape(-1, 15)).norm(dim=-1).max().item()
        metrics["per_z"].append({
            "z_e": z_val,
            "e_p_mean": e_p.mean().item(),
            "e_p_max": e_p.max().item(),
            "e_R_mean": e_R.mean().item(),
            "e_R_max": e_R.max().item(),
            "succ_rate": succ,
            "manif_adh_max": manif_adh,
        })
        print(f"  z_e={z_val:.2f}  pos err mean {e_p.mean():.4e} | "
              f"rot err mean {e_R.mean():.4e} (rad)  succ {succ:.3f}  "
              f"‖g‖_max {manif_adh:.3e}")

    # --- save metrics + plots ---
    import json
    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    losses_t = torch.tensor(losses_log)
    win = max(1, len(losses_t) // 200)
    sm = torch.nn.functional.avg_pool1d(losses_t.view(1, 1, -1),
                                          kernel_size=win, stride=win).flatten()
    ax[0].plot(torch.arange(len(sm)) * win, sm)
    ax[0].set_yscale("log"); ax[0].set_xlabel("step"); ax[0].set_ylabel("loss")
    ax[0].grid(alpha=0.3); ax[0].set_title("V2-pose training loss")

    zs = [m["z_e"] for m in metrics["per_z"]]
    succs = [m["succ_rate"] for m in metrics["per_z"]]
    e_ps = [m["e_p_mean"] for m in metrics["per_z"]]
    e_Rs = [m["e_R_mean"] for m in metrics["per_z"]]
    ax2 = ax[1]
    ax2.plot(zs, succs, "o-", label="success rate")
    ax2.set_xlabel("z_e"); ax2.set_ylabel("success rate"); ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left")
    ax2t = ax2.twinx()
    ax2t.plot(zs, e_ps, "s--", color="tab:orange", label="‖e_p‖")
    ax2t.plot(zs, e_Rs, "^--", color="tab:red", label="‖e_R‖ (rad)")
    ax2t.set_ylabel("error"); ax2t.legend(loc="upper right")
    ax2.set_title("eval per z_e")

    fig.tight_layout()
    fig.savefig(out_dir / "ours_v2_pose.png", dpi=120)
    plt.close(fig)
    print(f"saved {out_dir / 'ours_v2_pose.png'}, {out_dir / 'eval_metrics.json'}")


if __name__ == "__main__":
    main()
