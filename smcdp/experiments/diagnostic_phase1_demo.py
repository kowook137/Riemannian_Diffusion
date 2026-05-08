"""diagnostic_plan.md Phase 1 — Demo distribution 분석.

Diagnoses:
  2.1 Same-target demo std (q-space, p_ee-space at h=15)
  2.2 Demo target bias (‖mean(p_ee_H) − p_target‖)
  2.3 Per-target demo count
  2.4 IK solver stochasticity (run IK 10× with same target/mode, measure std)

판단 기준 (diagnostic_plan §2.5):
    | std_p_ee at h=15 | target bias | 결론 |
    | < 0.02 m         | < 0.02 m    | Demo OK              → Phase 1.5 |
    | 0.02 - 0.10 m    | < 0.02 m    | Demo 적당 spread     → Phase 1.5 + Phase 2 |
    | > 0.10 m         | any         | Demo spread main     → demo regenerate 우선 |
    | any              | > 0.05 m    | Demo bias main       → IK / demo gen fix 우선 |
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
from smcdp.franka.demo_gen import FrankaBimodalReachingDemo


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="outputs/franka_traj_unet/ckpt_riemannian.pt")
    p.add_argument("--n-targets", type=int, default=12)
    p.add_argument("--n-demos-per-group", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="outputs/diagnostic/phase1.json")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    # ---- load ckpt args (use the same demo configuration as training) ----
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    a = ck["args"]
    s1 = ck["stage1_args"]
    print(f"loaded ckpt args from {args.ckpt}")
    print(f"  H={a['H']}, p_box=[{a['p_box_lo']}, {a['p_box_hi']}], "
          f"z_e=[{a['z_min']}, {a['z_max']}], n_ik_steps={a['n_ik_steps']}")

    delta_net = DeltaResidualMLP(
        n_q=7, n_p=3, n_z=1, hidden=s1["hidden"], n_layers=s1["n_layers"],
        activation=torch.nn.Softplus, final_init_scale=1e-3,
    ).to(device)
    stage1_ck = torch.load("outputs/franka_stage1/delta_phi.pt",
                           map_location=device, weights_only=False)
    delta_net.load_state_dict(stage1_ck["delta_net_state"])
    delta_net.eval()

    arm_analytic = Franka7DoF(urdf_path=URDF, end_link="panda_hand", tool_z_max=0.25)
    arm_analytic._ensure_chain(torch.zeros(1, 7, device=device))
    arm = LearnedSelfModelFranka7DoF(
        delta_net=delta_net, urdf_path=URDF, end_link="panda_hand", tool_z_max=0.25,
    )
    arm._ensure_chain(torch.zeros(1, 7, device=device))

    box_lo = torch.tensor(a["p_box_lo"], device=device)
    box_hi = torch.tensor(a["p_box_hi"], device=device)
    z_e_fixed = 0.5 * (a["z_min"] + a["z_max"])

    # ---- 2.1 / 2.2 / 2.3: per-(target, mode) demo distribution ----
    # Sample n_targets distinct targets, fix z_e, generate n_demos for each (target, mode).
    H1 = a["H"] + 1
    h_end = H1 - 1

    # Pre-sample fixed targets uniformly in p_box
    g = torch.Generator(device=device).manual_seed(args.seed)
    targets = box_lo + (box_hi - box_lo) * torch.rand(
        args.n_targets, 3, generator=g, device=device
    )                                                                              # (T, 3)

    # For each (target, mode_A) and (target, mode_B), generate args.n_demos_per_group demos
    # Strategy: build a demo gen with branch_p_A = 1.0 (all mode A) for mode A,
    # branch_p_A = 0.0 for mode B; pass p_target manually for each demo.
    def make_demo(branch_p_A: float):
        return FrankaBimodalReachingDemo(
            manifold=arm, ik_arm=arm_analytic, H=a["H"],
            q_rest_A=list(a["q_rest_A"]), q_rest_B=list(a["q_rest_B"]),
            p_box_lo=tuple(a["p_box_lo"]), p_box_hi=tuple(a["p_box_hi"]),
            z_e_range=(z_e_fixed, z_e_fixed),
            branch_p_A=branch_p_A, jitter_q=a["jitter_q"],
            n_ik_steps=a["n_ik_steps"],
        )
    demo_A = make_demo(1.0)
    demo_B = make_demo(0.0)

    n_per = args.n_demos_per_group
    per_target_count = []
    within_std_q_A_list, within_std_pee_A_list, bias_A_list = [], [], []
    within_std_q_B_list, within_std_pee_B_list, bias_B_list = [], [], []

    print(f"\nGenerating {n_per} demos × 2 modes × {args.n_targets} targets = "
          f"{n_per * 2 * args.n_targets} demos total ...")

    for ti in range(args.n_targets):
        p_t = targets[ti].unsqueeze(0).expand(n_per, 3).contiguous()                 # (n_per, 3)
        for mode_label, demo in [("A", demo_A), ("B", demo_B)]:
            x, branch_A, _, _ = demo.sample(n_per, device=device, p_target=p_t)
            q = x[..., :7]                                                             # (n_per, H+1, 7)
            # Forward kin to get p_ee at each timestep (use analytic FK with same z_e)
            z_e = torch.full((n_per * H1, 1), z_e_fixed, device=device)
            p_ee = arm_analytic.F(q.reshape(-1, 7), z_e).reshape(n_per, H1, 3)         # (n_per, H+1, 3)
            std_q = q[:, h_end, :].std(dim=0)                                          # (7,)
            std_pee = p_ee[:, h_end, :].std(dim=0)                                     # (3,)
            mean_pee_end = p_ee[:, h_end, :].mean(dim=0)                               # (3,)
            bias = (mean_pee_end - targets[ti]).norm()
            if mode_label == "A":
                within_std_q_A_list.append(std_q)
                within_std_pee_A_list.append(std_pee)
                bias_A_list.append(bias)
            else:
                within_std_q_B_list.append(std_q)
                within_std_pee_B_list.append(std_pee)
                bias_B_list.append(bias)
        per_target_count.append(n_per * 2)

    # ---- 2.4: IK stochasticity test ----
    # Pick one (target, mode_A) and run IK 10 times — but our IK uses random init jitter,
    # so this measures total demo gen stochasticity (jitter + IK convergence margin).
    print("\nIK stochasticity test (same target+mode, 10 demos)...")
    p_t_test = targets[0:1].expand(10, 3).contiguous()
    x_repeat, _, _, _ = demo_A.sample(10, device=device, p_target=p_t_test)
    q_repeat = x_repeat[..., :7]
    z_repeat = torch.full((10 * H1, 1), z_e_fixed, device=device)
    p_repeat = arm_analytic.F(q_repeat.reshape(-1, 7), z_repeat).reshape(10, H1, 3)
    ik_std_q = q_repeat[:, h_end, :].std(dim=0).mean().item()
    ik_std_pee = p_repeat[:, h_end, :].std(dim=0).mean().item()
    ik_bias = (p_repeat[:, h_end, :].mean(0) - targets[0]).norm().item()

    # ---- aggregate ----
    def _stack_mean(lst):
        return torch.stack(lst, dim=0).mean(dim=0)

    summary = {
        "config": {
            "n_targets": args.n_targets,
            "n_demos_per_group": n_per,
            "z_e_fixed": z_e_fixed,
            "p_box": [a["p_box_lo"], a["p_box_hi"]],
            "n_ik_steps": a["n_ik_steps"],
        },
        "mode_A": {
            "within_std_q_at_h_end_per_joint": _stack_mean(within_std_q_A_list).cpu().tolist(),
            "within_std_q_mean":  _stack_mean(within_std_q_A_list).mean().item(),
            "within_std_pee_at_h_end_per_axis": _stack_mean(within_std_pee_A_list).cpu().tolist(),
            "within_std_pee_norm": _stack_mean(within_std_pee_A_list).norm().item(),
            "target_bias_mean": torch.stack(bias_A_list).mean().item(),
            "target_bias_max":  torch.stack(bias_A_list).max().item(),
        },
        "mode_B": {
            "within_std_q_at_h_end_per_joint": _stack_mean(within_std_q_B_list).cpu().tolist(),
            "within_std_q_mean":  _stack_mean(within_std_q_B_list).mean().item(),
            "within_std_pee_at_h_end_per_axis": _stack_mean(within_std_pee_B_list).cpu().tolist(),
            "within_std_pee_norm": _stack_mean(within_std_pee_B_list).norm().item(),
            "target_bias_mean": torch.stack(bias_B_list).mean().item(),
            "target_bias_max":  torch.stack(bias_B_list).max().item(),
        },
        "ik_stochasticity": {
            "ik_std_q_mean":   ik_std_q,
            "ik_std_pee_mean": ik_std_pee,
            "ik_bias":         ik_bias,
        },
    }

    print("\n" + "=" * 70)
    print("Phase 1 Diagnostic Summary")
    print("=" * 70)
    print(f"\nWithin-(target, mode) demo distribution at trajectory END (h={h_end}):")
    for label, mode in [("Mode A", "mode_A"), ("Mode B", "mode_B")]:
        s = summary[mode]
        print(f"  {label}:")
        print(f"    std_q  (rad, mean over joints):  {s['within_std_q_mean']:.4f}")
        print(f"    std_pee (m, ‖·‖ over xyz):        {s['within_std_pee_norm']:.4f}")
        print(f"    target bias (m):  mean={s['target_bias_mean']:.4f}  "
              f"max={s['target_bias_max']:.4f}")
    print(f"\nIK stochasticity (same target+mode, 10 reruns):")
    print(f"  std_q  : {ik_std_q:.4f}")
    print(f"  std_pee: {ik_std_pee:.4f}")
    print(f"  bias   : {ik_bias:.4f}")

    # ---- decision ----
    pee_std_max = max(summary["mode_A"]["within_std_pee_norm"],
                      summary["mode_B"]["within_std_pee_norm"])
    bias_max = max(summary["mode_A"]["target_bias_max"],
                   summary["mode_B"]["target_bias_max"])

    print("\n--- Decision (diagnostic_plan §2.5) ---")
    if bias_max > 0.05:
        verdict = "BIAS critical (>0.05m): Demo bias is main bottleneck → fix IK convergence / demo gen first"
    elif pee_std_max > 0.10:
        verdict = "SPREAD critical (>0.10m): Demo within-mode spread is main bottleneck → demo regenerate"
    elif pee_std_max > 0.02:
        verdict = "MODERATE spread (0.02-0.10m): Demo OK but contributes; proceed to Phase 1.5 + Phase 2"
    else:
        verdict = "Demo SHARP (<0.02m, <0.02m bias): Demo OK; proceed to Phase 1.5"
    print(f"  {verdict}")
    summary["verdict"] = verdict

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
