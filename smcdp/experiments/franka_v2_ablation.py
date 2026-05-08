"""V2 ablation sweep — guidance components on the channel-cond + p_start cond model.

Per user's recommendation, perform the ablation table at sampling time:
  (a) endpoint only (baseline)
  (b) + start anchor
  (c) + smoothness velocity
  (d) + smoothness acceleration
  (e) all four combined; α grid

Metrics per cell:
  e_0   start error
  e_8   middle error
  e_H   end error
  vel   mean ‖q_{h+1} − q_h‖
  acc   mean ‖q_{h+1} − 2q_h + q_{h-1}‖
  frac_A
  succ@2/5/10cm
  max ‖g_φ‖
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
    TrajectoryScoreNetUNet, TrajectoryScaledScoreFn, traj_reverse_grw,
)


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="outputs/franka_traj_unet_v2/ckpt_riemannian.pt")
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--n-sample-steps", type=int, default=200)
    p.add_argument("--z-val", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="outputs/diagnostic/v2_ablation.json")
    return p.parse_args()


def build_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
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

    use_p_start = bool(a.get("use_p_start_cond", False))
    goal_dim = 6 if use_p_start else 3
    cond_inj = a.get("cond_injection", "global")

    net = TrajectoryScoreNetUNet(
        arm, H=a["H"], down_dims=tuple(a["down_dims"]),
        diffusion_step_embed_dim=a["diff_step_embed"],
        n_groups=a["unet_groups"], kernel_size=a["unet_kernel"],
        t_scale=a["t_scale"], goal_cond_dim=goal_dim,
        cond_injection=cond_inj,
    ).to(device)
    net.load_state_dict(ck["ema_state"]); net.eval()
    score_fn = TrajectoryScaledScoreFn(net, sde)
    return arm, arm_analytic, sde, limiting, score_fn, a, use_p_start


def run_one(arm, arm_analytic, sde, limiting, score_fn, args_ck, use_p_start, device,
            args, alpha_goal, alpha_start, alpha_vel, alpha_acc, label):
    H1 = args_ck["H"] + 1
    d = arm.ambient_dim

    box_lo = torch.tensor(args_ck["p_box_lo"], device=device)
    box_hi = torch.tensor(args_ck["p_box_hi"], device=device)
    z_val = args.z_val

    z_tensor = torch.full((args.n, 1), z_val, device=device)
    z_lim = z_tensor.unsqueeze(1).expand(-1, H1, -1).reshape(args.n * H1, -1)

    torch.manual_seed(args.seed + 1000)
    p_targets = box_lo + (box_hi - box_lo) * torch.rand(args.n, 3, device=device)
    torch.manual_seed(args.seed + 2000)
    p_starts = box_lo + (box_hi - box_lo) * torch.rand(args.n, 3, device=device)

    # cond for score_fn
    if use_p_start:
        cond_for_net = torch.cat([p_targets, p_starts], dim=-1)
    else:
        cond_for_net = p_targets

    torch.manual_seed(args.seed + 3000)
    tau_T = limiting.sample(args.n * H1, device=device, z_e=z_lim).reshape(args.n, H1, d)

    grh = list(range(H1 - H1 // 2, H1))                                              # last_half
    sah = [0]                                                                         # h=0 only

    with torch.no_grad():
        tau_gen = traj_reverse_grw(
            sde, score_fn, tau_T,
            n_steps=args.n_sample_steps, eps=args_ck["eps"],
            goal_cond=cond_for_net, guidance_scale=0.0,
            goal_residual_alpha=alpha_goal, goal_residual_h=grh,
            p_start=p_starts, start_anchor_alpha=alpha_start, start_anchor_h=sah,
            smoothness_alpha_vel=alpha_vel,
            smoothness_alpha_acc=alpha_acc,
        )

    # Per-h pos errors (against linear interpolation target)
    q_gen = tau_gen[..., :7]
    z_e_gen = tau_gen[..., 7+3:7+3+1]
    p_gen = arm.F(q_gen.reshape(-1, 7), z_e_gen.reshape(-1, 1)).reshape(args.n, H1, 3)

    s = torch.linspace(0, 1, H1, device=device).view(1, H1, 1)
    p_target_per_h = p_starts.unsqueeze(1) + s * (p_targets - p_starts).unsqueeze(1)

    err_per_h = (p_gen - p_target_per_h).norm(dim=-1)                                # (n, H+1)
    e0 = err_per_h[:, 0]
    e8 = err_per_h[:, H1 // 2]
    eH = err_per_h[:, -1]

    # Smoothness
    vel = (q_gen[:, 1:] - q_gen[:, :-1]).norm(dim=-1)                                # (n, H)
    acc = (q_gen[:, 2:] - 2 * q_gen[:, 1:-1] + q_gen[:, :-2]).norm(dim=-1)           # (n, H-1)

    # Mode + violations + manifold
    q1_mid = q_gen[:, H1 // 2, 0]
    frac_A = (q1_mid > 0).float().mean().item()
    viol = arm.violates_limits(q_gen.reshape(-1, 7)).float().mean().item()
    max_g = arm.constraint(tau_gen.reshape(-1, d)).norm(dim=-1).max().item()

    # Success rates (against endpoint p_target)
    p_end = p_gen[:, -1, :]
    pos_err = (p_end - p_targets).norm(dim=-1)
    succ_2 = (pos_err < 0.02).float().mean().item()
    succ_5 = (pos_err < 0.05).float().mean().item()
    succ_10 = (pos_err < 0.10).float().mean().item()

    return {
        "label": label,
        "alpha": [alpha_goal, alpha_start, alpha_vel, alpha_acc],
        "e0_mean":  e0.mean().item(),  "e0_med":  e0.median().item(),
        "e8_mean":  e8.mean().item(),  "e8_med":  e8.median().item(),
        "eH_mean":  eH.mean().item(),  "eH_med":  eH.median().item(),
        "vel_mean": vel.mean().item(),
        "acc_mean": acc.mean().item(),
        "frac_A": frac_A,
        "viol": viol,
        "max_g": max_g,
        "succ_2cm":  succ_2,
        "succ_5cm":  succ_5,
        "succ_10cm": succ_10,
    }


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed)

    arm, arm_a, sde, limiting, score_fn, a, use_p_start = build_model(args.ckpt, device)
    print(f"Loaded ckpt: {args.ckpt}")
    print(f"  use_p_start_cond = {use_p_start}, cond_injection = {a.get('cond_injection', 'global')}")
    print()

    # Ablation grid
    cells = [
        # (label, alpha_goal, alpha_start, alpha_vel, alpha_acc)
        ("(a) endpoint only",                100, 0,   0, 0),
        ("(b) + start",                      100, 100, 0, 0),
        ("(c) + start + vel",                100, 100, 5, 0),
        ("(d) + start + vel + acc",          100, 100, 5, 5),
        ("(e1) all-strong",                  100, 100, 10, 10),
        ("(e2) endpoint weakened",            50, 100, 5, 5),
        ("(e3) endpoint weakened more",       20, 100, 5, 5),
        ("(f) baseline (no guidance)",         0, 0,   0, 0),
    ]

    print(f"{'cell':>40s}  {'e0':>7s} {'e8':>7s} {'eH':>7s}  "
          f"{'vel':>6s} {'acc':>6s}  {'frac_A':>7s}  "
          f"{'s@2':>5s} {'s@5':>5s} {'s@10':>5s}  {'viol':>5s}  {'max|g|':>9s}")

    rows = []
    for label, ag, as_, av, aa in cells:
        r = run_one(arm, arm_a, sde, limiting, score_fn, a, use_p_start, device,
                    args, ag, as_, av, aa, label)
        rows.append(r)
        print(f"{label:>40s}  "
              f"{r['e0_mean']:>6.3f}m {r['e8_mean']:>6.3f}m {r['eH_mean']:>6.3f}m  "
              f"{r['vel_mean']:>6.3f} {r['acc_mean']:>6.3f}  "
              f"{r['frac_A']:>7.3f}  "
              f"{r['succ_2cm']*100:>4.1f}% {r['succ_5cm']*100:>4.1f}% {r['succ_10cm']*100:>4.1f}%  "
              f"{r['viol']*100:>4.1f}%  {r['max_g']:>9.1e}")

    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
