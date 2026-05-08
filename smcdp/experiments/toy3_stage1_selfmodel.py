"""Toy 3 — Stage 1: train the residual self-model Δ_φ (Idea §7.2, §7.3).

Generates self-exploration data (q_i, p_true_i, z_e_i) ∼ uniform(q_box × z_box)
where p_true uses the synthetic compliance from `toy3.ground_truth`.  Trains
Δ_φ MLP to minimise

    L_self = E ‖FK_analytic + Δ_φ − p_true‖² + β · E ‖∂_q Δ_φ‖²_F

Saves the trained Δ_φ checkpoint for Stage 5.

Run:
    python -m smcdp.experiments.toy3_stage1_selfmodel
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from smcdp.toy3.ground_truth import TrueArmCompliance, generate_self_exploration_dataset
from smcdp.toy3.self_model import DeltaResidualMLP, self_model_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-data", type=int, default=20_000,
                   help="self-exploration dataset size")
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--smoothness", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--l1", type=float, default=1.0)
    p.add_argument("--l2-base", type=float, default=1.0)
    p.add_argument("--K-grav", type=float, default=0.30)
    p.add_argument("--K-offset", type=float, default=0.05)
    p.add_argument("--z-min", type=float, default=0.0)
    p.add_argument("--z-max", type=float, default=0.3)
    p.add_argument("--q-half", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="outputs/toy3_stage1")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out_dir={out_dir}")

    # --- ground truth + data ---
    truth = TrueArmCompliance(
        l1=args.l1, l2_base=args.l2_base,
        K_grav=args.K_grav, K_offset=args.K_offset,
    )
    dset = generate_self_exploration_dataset(
        n=args.n_data,
        truth=truth,
        z_e_range=(args.z_min, args.z_max),
        q_box=(-args.q_half, args.q_half),
        seed=args.seed,
        device=device,
    )
    q_all, ze_all, p_all = dset["q"], dset["z_e"], dset["p_true"]
    print(f"data: {q_all.shape[0]} samples,  q in ±{args.q_half},  z_e in [{args.z_min}, {args.z_max}]")

    # held-out validation slice
    val_size = max(1024, args.n_data // 20)
    val_q  = q_all[-val_size:]
    val_ze = ze_all[-val_size:]
    val_p  = p_all[-val_size:]
    train_q  = q_all[:-val_size]
    train_ze = ze_all[:-val_size]
    train_p  = p_all[:-val_size]
    n_train = train_q.shape[0]

    # --- model + optimiser ---
    delta_net = DeltaResidualMLP(
        n_q=2, n_p=2, n_z=1,
        hidden=args.hidden, n_layers=args.n_layers,
    ).to(device)
    optim = torch.optim.Adam(delta_net.parameters(), lr=args.lr)

    fk_fn = lambda q, z: truth.fk_analytic(q, z)

    # --- training loop ---
    log_train: list[float] = []
    log_val: list[float] = []
    pbar = tqdm(range(args.steps), desc="train Δ_φ")
    for step in pbar:
        idx = torch.randint(0, n_train, (args.batch,), device=device)
        q_b = train_q[idx]
        z_b = train_ze[idx]
        p_b = train_p[idx]
        loss, parts = self_model_loss(
            delta_net, q_b, z_b, p_b, fk_fn,
            smoothness_weight=args.smoothness,
        )
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        log_train.append(parts["fit"])

        if step % 500 == 0:
            with torch.no_grad():
                _, vparts = self_model_loss(
                    delta_net, val_q, val_ze, val_p, fk_fn, smoothness_weight=0.0,
                )
            log_val.append((step, vparts["fit"]))
            pbar.set_postfix(
                train_fit=f"{parts['fit']:.2e}",
                val_fit=f"{vparts['fit']:.2e}",
                smooth=f"{parts.get('smooth', 0):.2e}",
            )

    # --- final evaluation ---
    with torch.no_grad():
        delta_pred_val = delta_net(val_q, val_ze)
        delta_true_val = truth.delta_true(val_q, val_ze)
        delta_err_val = (delta_pred_val - delta_true_val).norm(dim=-1)
        # also: residual fit error after analytic FK
        p_pred_val = truth.fk_analytic(val_q, val_ze) + delta_pred_val
        p_err_val = (p_pred_val - val_p).norm(dim=-1)
        # baseline (Δ_φ ≡ 0): how good is analytic FK alone?
        p_analytic_err_val = (truth.fk_analytic(val_q, val_ze) - val_p).norm(dim=-1)

    print()
    print(f"validation (n = {val_size}):")
    print(f"  ‖p_true − analytic FK‖           = {p_analytic_err_val.mean().item():.4e}  (oracle Δ_φ ≡ 0)")
    print(f"  ‖p_true − analytic FK − Δ_φ‖     = {p_err_val.mean().item():.4e}  (learned)")
    print(f"  ‖Δ_φ − Δ_true‖                   = {delta_err_val.mean().item():.4e}")
    rel = p_err_val.mean().item() / p_analytic_err_val.mean().item()
    print(f"  fit improvement factor           = {1.0/rel:.1f}x  (analytic / learned)")

    # --- plots ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(log_train, lw=0.6)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("train fit MSE")
    axes[0].set_title("Stage 1: Δ_φ training loss")
    axes[0].grid(alpha=0.3)

    if log_val:
        steps_v, vals_v = zip(*log_val)
        axes[0].plot(steps_v, vals_v, "o-", color="C1", lw=1, label="val")
        axes[0].legend()

    # Δ_pred vs Δ_true scatter
    dp = delta_pred_val.cpu().numpy()
    dt = delta_true_val.cpu().numpy()
    axes[1].scatter(dt[:, 0], dp[:, 0], s=4, alpha=0.4, label="x")
    axes[1].scatter(dt[:, 1], dp[:, 1], s=4, alpha=0.4, label="y")
    lo = min(dt.min(), dp.min())
    hi = max(dt.max(), dp.max())
    axes[1].plot([lo, hi], [lo, hi], "k-", lw=0.6, alpha=0.5)
    axes[1].set_xlabel("Δ_true component")
    axes[1].set_ylabel("Δ_φ component")
    axes[1].set_aspect("equal")
    axes[1].legend()
    axes[1].set_title("Δ_φ vs Δ_true (validation)")
    axes[1].grid(alpha=0.3)

    # error vs z_e (does generalization scale with embodiment perturbation?)
    z_axis = val_ze.squeeze(-1).cpu().numpy()
    axes[2].scatter(z_axis, p_err_val.cpu().numpy(), s=4, alpha=0.4, label="learned")
    axes[2].scatter(z_axis, p_analytic_err_val.cpu().numpy(), s=4, alpha=0.4, label="oracle Δ=0")
    axes[2].set_xlabel("z_e (tool length)")
    axes[2].set_ylabel("‖p_true − p_pred‖")
    axes[2].set_yscale("log")
    axes[2].legend()
    axes[2].set_title("residual fit vs embodiment")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Toy 3 Stage 1 — residual self-model on 2-link arm")
    fig.tight_layout()
    fig.savefig(out_dir / "stage1_self_model.png", dpi=120)
    plt.close(fig)

    torch.save({
        "args": vars(args),
        "delta_net_state": delta_net.state_dict(),
        "metrics": {
            "p_err_val": p_err_val.mean().item(),
            "p_analytic_err_val": p_analytic_err_val.mean().item(),
            "delta_err_val": delta_err_val.mean().item(),
            "improvement_factor": 1.0 / rel,
        },
    }, out_dir / "delta_phi.pt")
    print(f"\ncheckpoint saved to  {out_dir / 'delta_phi.pt'}")


if __name__ == "__main__":
    main()
