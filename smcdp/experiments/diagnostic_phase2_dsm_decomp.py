"""diagnostic_plan.md Phase 2 — DSM loss 분해 진단 (no retrain).

Diagnoses:
  4.1 Train vs Val DSM loss (overfit / underfit / objective mismatch)
  4.2 DSM loss by diffusion time r (5 bins)  ← reverse-process bottleneck timing
  4.3 DSM loss by target cluster                ← per-condition consistency

판단 기준 (diagnostic_plan §4.4):
  Train ≈ val, both low + task fail   → Objective mismatch
  High at r→1                          → Reverse early-step issue
  High at r→0                          → Final sharpness 부족
  High at specific c                   → Conditioning weak for those targets
"""
from __future__ import annotations

import argparse
import json
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
    TrajectoryScoreNetUNet, TrajectoryScaledScoreFn, traj_dsm_varadhan_loss,
    traj_forward_grw,
)


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="outputs/franka_traj_unet/ckpt_riemannian.pt")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-r-bins", type=int, default=5)
    p.add_argument("--n-target-clusters", type=int, default=4)
    p.add_argument("--out", type=str, default="outputs/diagnostic/phase2.json")
    return p.parse_args()


@torch.no_grad()
def evaluate_loss_on_batch(score_fn, sde, x_data, p_target, eps, weight, n_grw_steps,
                           r_fixed=None):
    """Per-sample DSM-Varadhan loss; if r_fixed given, use that r instead of random."""
    B, H1, d = x_data.shape
    schedule = sde.schedule
    if r_fixed is None:
        r = eps + (schedule.tf - eps) * torch.rand(B, device=x_data.device, dtype=x_data.dtype)
    else:
        r = torch.full((B,), float(r_fixed), device=x_data.device, dtype=x_data.dtype)
    tau_r = traj_forward_grw(sde, x_data, r, n_grw_steps)
    tau_brown = schedule.integral(r).clamp(min=1e-12).view(B, 1, 1)
    tau_r_flat = tau_r.reshape(B * H1, d)
    tau_0_flat = x_data.reshape(B * H1, d)
    log_flat = sde.manifold.log(tau_r_flat, tau_0_flat)
    target = log_flat.reshape(B, H1, d) / tau_brown
    score = score_fn(tau_r, r, goal_cond=p_target)
    diff = score - target
    sq_per_pt = sde.manifold.squared_norm(tau_r_flat, diff.reshape(B * H1, d))
    sq_per_traj = sq_per_pt.reshape(B, H1).sum(-1)
    if weight == "sigma2":
        w = schedule.proxy_std(r) ** 2
    elif weight == "beta":
        w = schedule.beta(r)
    elif weight == "none":
        w = torch.ones_like(r)
    else:
        raise ValueError(weight)
    return (w * sq_per_traj).cpu()                                       # (B,)


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    a = ck["args"]
    s1 = ck["stage1_args"]

    delta_net = DeltaResidualMLP(n_q=7, n_p=3, n_z=1,
                                  hidden=s1["hidden"], n_layers=s1["n_layers"],
                                  activation=torch.nn.Softplus,
                                  final_init_scale=1e-3).to(device)
    stage1_ck = torch.load("outputs/franka_stage1/delta_phi.pt",
                           map_location=device, weights_only=False)
    delta_net.load_state_dict(stage1_ck["delta_net_state"])
    delta_net.eval()

    arm_analytic = Franka7DoF(urdf_path=URDF, end_link="panda_hand",
                              tool_z_max=max(a["z_max"], a["z_eval"][-1]) + 0.05)
    arm_analytic._ensure_chain(torch.zeros(1, 7, device=device))
    arm = LearnedSelfModelFranka7DoF(
        delta_net=delta_net, urdf_path=URDF, end_link="panda_hand",
        tool_z_max=max(a["z_max"], a["z_eval"][-1]) + 0.05, metric=a["metric"],
    )
    arm._ensure_chain(torch.zeros(1, 7, device=device))

    schedule = LinearBetaSchedule(beta_0=a["beta_0"], beta_f=a["beta_f"], t0=0.0, tf=1.0)
    limiting = WrappedNormalFranka7DoF(arm, mean_q=list(a["limiting_mean_q"]),
                                       scale=a["limiting_scale"],
                                       z_e_range=(a["z_min"], a["z_max"]))
    sde = LangevinSDE(arm, schedule, limiting)

    net_ema = TrajectoryScoreNetUNet(
        arm, H=a["H"], down_dims=tuple(a["down_dims"]),
        diffusion_step_embed_dim=a["diff_step_embed"],
        n_groups=a["unet_groups"], kernel_size=a["unet_kernel"],
        t_scale=a["t_scale"], goal_cond_dim=3,
    ).to(device)
    net_ema.load_state_dict(ck["ema_state"]); net_ema.eval()
    score_fn = TrajectoryScaledScoreFn(net_ema, sde)

    # Demo gen
    data = FrankaBimodalReachingDemo(
        manifold=arm, ik_arm=arm_analytic, H=a["H"],
        q_rest_A=list(a["q_rest_A"]), q_rest_B=list(a["q_rest_B"]),
        p_box_lo=tuple(a["p_box_lo"]), p_box_hi=tuple(a["p_box_hi"]),
        z_e_range=(a["z_min"], a["z_max"]),
        branch_p_A=a["branch_p_A"], jitter_q=a["jitter_q"],
        n_ik_steps=a["n_ik_steps"],
    )

    # Sample TRAIN-LIKE distribution (matches training): random box + random z_e
    torch.manual_seed(args.seed)
    x_train, _, _, p_train = data.sample(args.n, device=device)

    # Sample VAL distribution (different seed, but otherwise identical distribution)
    torch.manual_seed(args.seed + 999)
    x_val, _, _, p_val = data.sample(args.n, device=device)

    # ---- 4.1 Train vs Val (random r from same distribution as training) ----
    print(f"Phase 2 — DSM loss decomposition (n={args.n}, weight={a['weight']})\n")
    print("4.1 Train vs Val DSM loss")
    train_losses_random_r = []
    val_losses_random_r = []
    n_repeats = 4
    for _ in range(n_repeats):
        train_losses_random_r.append(evaluate_loss_on_batch(
            score_fn, sde, x_train, p_train, a["eps"], a["weight"], a["n_grw_steps"]
        ))
        val_losses_random_r.append(evaluate_loss_on_batch(
            score_fn, sde, x_val, p_val, a["eps"], a["weight"], a["n_grw_steps"]
        ))
    train_l = torch.cat(train_losses_random_r)
    val_l = torch.cat(val_losses_random_r)
    print(f"  train: mean={train_l.mean():.4f}  median={train_l.median():.4f}  "
          f"std={train_l.std():.4f}  (n={len(train_l)})")
    print(f"  val:   mean={val_l.mean():.4f}  median={val_l.median():.4f}  "
          f"std={val_l.std():.4f}  (n={len(val_l)})")
    train_val_gap = (val_l.mean() - train_l.mean()).item()
    print(f"  val − train gap: {train_val_gap:+.4f}\n")

    # ---- 4.2 Loss by diffusion time r ----
    print(f"4.2 Loss by diffusion time r ({args.n_r_bins} bins)")
    r_edges = torch.linspace(a["eps"], schedule.tf, args.n_r_bins + 1)
    loss_by_r = {}
    for i in range(args.n_r_bins):
        r_mid = 0.5 * (r_edges[i].item() + r_edges[i + 1].item())
        loss_at_r = evaluate_loss_on_batch(
            score_fn, sde, x_val, p_val, a["eps"], a["weight"],
            a["n_grw_steps"], r_fixed=r_mid,
        )
        loss_by_r[f"r={r_mid:.3f}"] = {
            "mean": loss_at_r.mean().item(),
            "median": loss_at_r.median().item(),
        }
        print(f"  r ∈ [{r_edges[i]:.3f}, {r_edges[i+1]:.3f}]  (mid r={r_mid:.3f}):  "
              f"loss mean={loss_at_r.mean():.4f}  median={loss_at_r.median():.4f}")
    print()

    # ---- 4.3 Loss by target cluster ----
    print(f"4.3 Loss by target cluster ({args.n_target_clusters} clusters by p_target spatial)")
    # K-means-lite on p_val using simple grid clustering
    box_lo = torch.tensor(a["p_box_lo"], device=device)
    box_hi = torch.tensor(a["p_box_hi"], device=device)
    # bucket by axis-wise quantile
    p_norm = (p_val - box_lo) / (box_hi - box_lo).clamp(min=1e-6)                 # (B, 3) in [0,1]
    # use first axis to define clusters
    n_clusters = args.n_target_clusters
    cluster_id = (p_norm[:, 0] * n_clusters).clamp(0, n_clusters - 1).long()
    loss_by_cluster = {}
    for c in range(n_clusters):
        mask = (cluster_id == c)
        if mask.sum() < 8:
            continue
        # subset
        x_c = x_val[mask]
        p_c = p_val[mask]
        # accumulate loss across multiple random r
        ls = []
        for _ in range(n_repeats):
            ls.append(evaluate_loss_on_batch(
                score_fn, sde, x_c, p_c, a["eps"], a["weight"], a["n_grw_steps"]
            ))
        ls = torch.cat(ls)
        loss_by_cluster[f"cluster_{c}_x_quantile"] = {
            "n": int(mask.sum()),
            "x_range": [(box_lo[0] + (box_hi[0] - box_lo[0]) * c / n_clusters).item(),
                        (box_lo[0] + (box_hi[0] - box_lo[0]) * (c + 1) / n_clusters).item()],
            "loss_mean": ls.mean().item(),
            "loss_median": ls.median().item(),
        }
        print(f"  cluster {c} (p_x ∈ [{loss_by_cluster[f'cluster_{c}_x_quantile']['x_range'][0]:.3f}, "
              f"{loss_by_cluster[f'cluster_{c}_x_quantile']['x_range'][1]:.3f}], "
              f"n={mask.sum()}):  loss mean={ls.mean():.4f}")

    # ---- Decision ----
    print("\n--- Decision (diagnostic_plan §4.4) ---")
    # Identify pattern
    r_means = [loss_by_r[k]["mean"] for k in sorted(loss_by_r.keys())]
    r_first, r_last = r_means[0], r_means[-1]
    cluster_means = [v["loss_mean"] for v in loss_by_cluster.values()]
    cluster_max = max(cluster_means) if cluster_means else 0
    cluster_min = min(cluster_means) if cluster_means else 0

    notes = []
    if abs(train_val_gap) < 0.10:
        notes.append("Train ≈ Val (no overfitting)")
    elif train_val_gap > 0.10:
        notes.append("OVERFIT: val > train")
    else:
        notes.append("UNDERFIT: train > val (suspicious)")

    if r_last > 2.0 * r_first:
        notes.append(f"HIGH at r→tf (large noise) — reverse early-step issue (r=tf loss "
                     f"{r_last:.3f} vs r=eps {r_first:.3f})")
    elif r_first > 2.0 * r_last:
        notes.append(f"HIGH at r→0 (low noise) — final sharpness 부족")
    else:
        notes.append(f"r-flat (loss similar across r levels)")

    if cluster_means and cluster_max > 1.5 * cluster_min:
        notes.append(f"Cluster-dependent loss (max/min ratio {cluster_max/cluster_min:.2f}) "
                     "— conditioning weak for some targets")
    else:
        notes.append("Cluster-uniform loss")

    if abs(train_val_gap) < 0.10 and r_last < 2.0 * r_first and r_first < 2.0 * r_last:
        notes.append("→ Train≈Val + r-uniform + low DSM loss + task fail = OBJECTIVE MISMATCH "
                     "(DSM accuracy doesn't translate to conditional sample quality)")

    for note in notes:
        print(f"  • {note}")

    summary = {
        "config": {"ckpt": args.ckpt, "n": args.n, "weight": a["weight"]},
        "section_4_1": {
            "train_loss_mean": train_l.mean().item(),
            "val_loss_mean":   val_l.mean().item(),
            "gap":             train_val_gap,
        },
        "section_4_2_loss_by_r": loss_by_r,
        "section_4_3_loss_by_cluster": loss_by_cluster,
        "decision_notes": notes,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
