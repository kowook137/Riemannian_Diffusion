"""Standardised eval for Franka baselines (BC / DP-official / Projected) — same metrics as Ours.

Loads any baseline ckpt, runs reverse / sampling, evaluates with:
  pos_err, succ@{2,5,8,10,15}cm, frac_A, max ‖g_φ‖, W₁_A, W₁_B, vel mean, joint viol.

Compatible with Ours-V2 evaluation (same z_e, same target distribution, same metrics)
for direct comparison.
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
from smcdp.baselines import (
    BCTrajectoryPredictor, make_official_diffusion_policy, official_dp_sample,
    channel_concat_dp_sample,
)


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--baseline", type=str, required=True,
                   choices=["bc", "dp_official", "projected"])
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--n-inference-steps", type=int, default=100)
    p.add_argument("--z-list", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20])
    p.add_argument("--success-radii", type=float, nargs="+",
                   default=[0.02, 0.05, 0.08, 0.10, 0.15])
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.out is None:
        args.out = f"outputs/diagnostic/baseline_{args.baseline}_eval.json"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed)

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    a = ck["args"]
    s1 = ck["stage1_args"]
    print(f"baseline={args.baseline}  ckpt={args.ckpt}")

    # Manifold for evaluation
    arm_a = Franka7DoF(urdf_path=URDF, end_link="panda_hand", tool_z_max=0.25)
    arm_a._ensure_chain(torch.zeros(1, 7, device=device))
    perfect_fk = bool(a.get("perfect_fk", False))
    if perfect_fk:
        arm = arm_a
        print(f"  PERFECT FK regime (eval): arm = Franka7DoF (analytic only)")
    else:
        delta_net = DeltaResidualMLP(n_q=7, n_p=3, n_z=1,
                                      hidden=s1["hidden"], n_layers=s1["n_layers"],
                                      activation=torch.nn.Softplus,
                                      final_init_scale=1e-3).to(device)
        stage1_ck = torch.load("outputs/franka_stage1/delta_phi.pt",
                               map_location=device, weights_only=False)
        delta_net.load_state_dict(stage1_ck["delta_net_state"])
        delta_net.eval()
        arm = LearnedSelfModelFranka7DoF(
            delta_net=delta_net, urdf_path=URDF, end_link="panda_hand", tool_z_max=0.25,
        )
        arm._ensure_chain(torch.zeros(1, 7, device=device))

    H1 = a["H"] + 1
    n_q, n_p = 7, 3
    CTX_DIM = 3 + 3 + 1

    # ---- Load model + scheduler ----
    if args.baseline == "bc":
        model = BCTrajectoryPredictor(
            n_q=n_q, H=a["H"], ctx_dim=CTX_DIM,
            hidden=a["bc_hidden"], n_layers=a["bc_layers"],
            activation="sin",
        ).to(device)
        scheduler = None
    elif args.baseline == "dp_official":
        cond_inj = a.get("cond_injection", "global")
        if cond_inj == "channel":
            model, scheduler = make_official_diffusion_policy(
                n_q=n_q + CTX_DIM, global_cond_dim=None,
                down_dims=list(a["down_dims"]),
                diffusion_step_embed_dim=a["diff_step_embed"],
                n_train_timesteps=a["dp_train_timesteps"],
            )
        else:
            model, scheduler = make_official_diffusion_policy(
                n_q=n_q, global_cond_dim=CTX_DIM,
                down_dims=list(a["down_dims"]),
                diffusion_step_embed_dim=a["diff_step_embed"],
                n_train_timesteps=a["dp_train_timesteps"],
            )
        model = model.to(device)
    elif args.baseline == "projected":
        model, scheduler = make_official_diffusion_policy(
            n_q=n_q + n_p, global_cond_dim=CTX_DIM,
            down_dims=list(a["down_dims"]),
            diffusion_step_embed_dim=a["diff_step_embed"],
            n_train_timesteps=a["dp_train_timesteps"],
        )
        model = model.to(device)

    state = ck["ema_state"] if args.use_ema and "ema_state" in ck else ck["model_state"]
    model.load_state_dict(state)
    model.eval()

    # ---- Demo data for W1 + mode classification reference ----
    box_lo = torch.tensor(a["p_box_lo"], device=device)
    box_hi = torch.tensor(a["p_box_hi"], device=device)

    print(f"\n{'z_e':>5}  {'pos_err':>8} {'med':>8} {'std':>8}  "
          + "  ".join([f"r={r*100:.0f}cm" for r in args.success_radii])
          + f"  {'frac_A':>7}  {'vel':>6}  {'viol%':>5}  {'W1_A':>6}  {'W1_B':>6}  {'max|g|':>9}")

    results = {}
    for z_val in args.z_list:
        torch.manual_seed(args.seed + 1000)
        p_targets = box_lo + (box_hi - box_lo) * torch.rand(args.n, 3, device=device)
        torch.manual_seed(args.seed + 2000)
        p_starts = box_lo + (box_hi - box_lo) * torch.rand(args.n, 3, device=device)
        z_tensor = torch.full((args.n, 1), z_val, device=device)
        ctx = torch.cat([p_targets, p_starts, z_tensor], dim=-1)                    # (n, 7)

        # Demo reference (for W1)
        data_z = FrankaBimodalReachingDemo(
            manifold=arm, ik_arm=arm_a, H=a["H"],
            q_rest_A=list(a["q_rest_A"]), q_rest_B=list(a["q_rest_B"]),
            p_box_lo=tuple(a["p_box_lo"]), p_box_hi=tuple(a["p_box_hi"]),
            z_e_range=(z_val, z_val), branch_p_A=a["branch_p_A"],
            jitter_q=a["jitter_q"], n_ik_steps=a["n_ik_steps"],
        )
        x_data, _, _, _, _ = data_z.sample(args.n, device=device,
                                            p_target=p_targets, p_start=p_starts)

        # ---- Sample per baseline ----
        if args.baseline == "bc":
            with torch.no_grad():
                q_gen = model(ctx)                                                   # (n, H+1, 7)
            # lift to ambient via make_x for fair manifold-adherence comparison
            q_flat = q_gen.reshape(-1, 7)
            z_flat = z_tensor.unsqueeze(1).expand(-1, H1, -1).reshape(-1, 1)
            x_gen = arm.make_x(q_flat, z_flat).reshape(args.n, H1, n_q + n_p + 1)
        elif args.baseline == "dp_official":
            cond_inj = a.get("cond_injection", "global")
            with torch.no_grad():
                if cond_inj == "channel":
                    ctx_per_h = ctx.unsqueeze(1).expand(-1, H1, -1)
                    q_gen = channel_concat_dp_sample(model, scheduler,
                                                      batch_size=args.n, horizon=H1, n_q=n_q,
                                                      ctx_per_h=ctx_per_h, device=device,
                                                      n_inference_steps=args.n_inference_steps)
                else:
                    q_gen = official_dp_sample(model, scheduler,
                                                batch_size=args.n, horizon=H1, n_q=n_q,
                                                ctx=ctx, device=device,
                                                n_inference_steps=args.n_inference_steps)
            q_flat = q_gen.reshape(-1, 7)
            z_flat = z_tensor.unsqueeze(1).expand(-1, H1, -1).reshape(-1, 1)
            x_gen = arm.make_x(q_flat, z_flat).reshape(args.n, H1, n_q + n_p + 1)
        elif args.baseline == "projected":
            with torch.no_grad():
                # Sample ambient (q, p) ∈ R^10 then project per step (already done at end:
                # we project after reverse loop completes, since reverse is inside diffusion_policy
                # internals).  Here we apply final projection.
                x_amb = official_dp_sample(model, scheduler,
                                            batch_size=args.n, horizon=H1, n_q=n_q + n_p,
                                            ctx=ctx, device=device,
                                            n_inference_steps=args.n_inference_steps)
                # x_amb shape (n, H+1, 10). Project: replace p with F_φ(q, z_e).
                q_part = x_amb[..., :n_q]
                z_flat = z_tensor.unsqueeze(1).expand(-1, H1, -1).reshape(-1, 1)
                p_proj = arm.F(q_part.reshape(-1, 7), z_flat).reshape(args.n, H1, 3)
                # Final projected: x = (q, p_proj, z_e) — this is what manifold-adherent samples look like
                z_traj = z_tensor.unsqueeze(1).expand(-1, H1, -1)
                x_gen = torch.cat([q_part, p_proj, z_traj], dim=-1)                   # (n, H+1, 11)
                # ALSO record un-projected ambient EE for "raw" manifold adherence metric
                x_gen_raw = torch.cat([q_part, x_amb[..., n_q:n_q+n_p], z_traj], dim=-1)
                # max|g_φ| on UN-projected (Christopher24 baseline value before final projection)
                max_g_raw = arm.constraint(x_gen_raw.reshape(-1, n_q+n_p+1)).norm(dim=-1).max().item()
                q_gen = q_part

        # ---- Metrics ----
        p_end = x_gen[:, -1, n_q:n_q+n_p]
        pos_err = (p_end - p_targets).norm(dim=-1)
        succ_at = {f"r={r*100:.0f}cm": (pos_err < r).float().mean().item()
                   for r in args.success_radii}

        h_mid = H1 // 2
        q1_mid = q_gen[:, h_mid, 0]
        q1_mid_data = x_data[:, h_mid, 0]
        mode_A_data = q1_mid_data > 0
        mode_A = q1_mid > 0
        frac_A = mode_A.float().mean().item()

        n_dir = 64
        dirs = torch.randn(n_dir, 7, device=device)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        def per_mode_w1(ta, tb):
            m = min(ta.shape[0], tb.shape[0])
            if m < 8: return float("nan")
            ia = torch.randperm(ta.shape[0], device=device)[:m]
            ib = torch.randperm(tb.shape[0], device=device)[:m]
            a_, b_ = ta[ia, :, :7], tb[ib, :, :7]
            ws = []
            for h in range(H1):
                pa = (a_[:, h] @ dirs.T).sort(dim=0).values
                pb = (b_[:, h] @ dirs.T).sort(dim=0).values
                ws.append((pa - pb).abs().mean().item())
            return sum(ws) / len(ws)
        w1A = per_mode_w1(q_gen[mode_A].unsqueeze(-1).repeat(1, 1, 1) if False else q_gen[mode_A].view(-1, H1, 7) if False else q_gen, x_data[mode_A_data])
        # simplified: use per-mode mask directly on gen and data
        if mode_A.any() and mode_A_data.any():
            ta = torch.cat([q_gen[mode_A], torch.zeros(mode_A.sum(), H1, x_data.shape[-1] - 7, device=device)], dim=-1)
            tb = x_data[mode_A_data]
            w1A = per_mode_w1(ta, tb)
        else:
            w1A = float("nan")
        if (~mode_A).any() and (~mode_A_data).any():
            ta = torch.cat([q_gen[~mode_A], torch.zeros((~mode_A).sum(), H1, x_data.shape[-1] - 7, device=device)], dim=-1)
            tb = x_data[~mode_A_data]
            w1B = per_mode_w1(ta, tb)
        else:
            w1B = float("nan")

        vel = (q_gen[:, 1:] - q_gen[:, :-1]).norm(dim=-1).mean().item()
        viol = arm.violates_limits(q_gen.reshape(-1, 7)).float().mean().item()

        # max|g_φ|
        if args.baseline == "projected":
            # Report BOTH raw (un-projected) and projected.  Projected is always 0.
            max_g = max_g_raw
            max_g_post_proj = arm.constraint(x_gen.reshape(-1, n_q + n_p + 1)).norm(dim=-1).max().item()
        else:
            # BC and DP only output q; we lift via make_x → max|g| = 0 by construction.
            # But this is "trivial" because the baseline didn't enforce manifold.
            # The fair comparison is: what does the BASELINE produce?
            # BC/DP output q, then we apply FK to get p; if we use ANALYTIC FK (not learned),
            # this would be a mismatch with M_φ.  Let's compute the gap.
            q_flat = q_gen.reshape(-1, 7)
            z_flat = z_tensor.unsqueeze(1).expand(-1, H1, -1).reshape(-1, 1)
            p_analytic = arm_a.F(q_flat, z_flat)                                     # what baseline assumes
            p_learned = arm.F(q_flat, z_flat)                                        # what M_φ has
            max_g = (p_analytic - p_learned).norm(dim=-1).max().item()
            max_g_post_proj = 0.0  # by lift via make_x

        rad_vals = "  ".join([f"{succ_at[k]*100:>5.1f}%" for k in
                              [f"r={r*100:.0f}cm" for r in args.success_radii]])
        ood = " (OOD)" if (z_val < a["z_min"] or z_val > a["z_max"]) else ""
        print(f"{z_val:>5.3f}  {pos_err.mean().item():>7.4f}m {pos_err.median().item():>7.4f}m "
              f"{pos_err.std().item():>7.4f}m  {rad_vals}  {frac_A:>7.3f}  "
              f"{vel:>6.3f}  {viol*100:>4.1f}%  {w1A:>6.3f}  {w1B:>6.3f}  {max_g:>9.1e}{ood}")

        results[f"z={z_val}"] = {
            "pos_err_mean": pos_err.mean().item(),
            "pos_err_med":  pos_err.median().item(),
            "pos_err_std":  pos_err.std().item(),
            **succ_at,
            "frac_A": frac_A,
            "vel_mean": vel,
            "viol": viol,
            "W1_A": w1A,
            "W1_B": w1B,
            "max_g_raw": max_g,
            "max_g_post_proj": max_g_post_proj,
        }

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
