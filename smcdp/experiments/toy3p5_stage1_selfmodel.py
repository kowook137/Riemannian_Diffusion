"""Toy 3.5 Stage 1 — train residual self-model Δ_φ for the 3-link arm.

Same procedure as `toy3_stage1_selfmodel.py` but with N-link arm
(default 3-link, ℓ_base=[1, 1, 0.5]).  Uses TrueNLinkArmCompliance
(synthetic compliance depending on cumulative end-effector orientation
α = q_1 + q_2 + q_3).

Data: (q_i, p_true_i, z_e_i) ∼ Uniform(q_box × z_box)
Loss: L_self = E ‖FK_analytic + Δ_φ − p_true‖² + β · E ‖∂_q Δ_φ‖²_F

Saves a checkpoint that Step D-2 reuses.

Run:
    python -m smcdp.experiments.toy3p5_stage1_selfmodel
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from smcdp.toy3.ground_truth import (
    TrueNLinkArmCompliance,
    generate_self_exploration_dataset,
)
from smcdp.toy3.self_model import DeltaResidualMLP, self_model_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--link-lengths-base", type=float, nargs="+", default=[1.0, 1.0, 0.5])
    p.add_argument("--n-data", type=int, default=20_000)
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--smoothness", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--K-grav", type=float, default=0.30)
    p.add_argument("--K-offset", type=float, default=0.05)
    p.add_argument("--z-min", type=float, default=0.0)
    p.add_argument("--z-max", type=float, default=0.3)
    p.add_argument("--q-half", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="outputs/toy3p5_stage1")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_q = len(args.link_lengths_base)
    print(f"device={device}  out_dir={out_dir}  n_q={n_q}  links={args.link_lengths_base}")

    truth = TrueNLinkArmCompliance(
        link_lengths_base=args.link_lengths_base,
        K_grav=args.K_grav,
        K_offset=args.K_offset,
    )
    dset = generate_self_exploration_dataset(
        n=args.n_data, truth=truth,
        z_e_range=(args.z_min, args.z_max),
        q_box=(-args.q_half, args.q_half),
        seed=args.seed, device=device,
    )
    # Note: generate_self_exploration_dataset samples q with shape (n, 2) hard-coded
    # for 2-link.  We need n_q-d q.  Use same generator but reshape to n_q.
    # Quick workaround: generate manually here.
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    q_lo, q_hi = -args.q_half, args.q_half
    z_lo, z_hi = args.z_min, args.z_max
    q_all = (q_lo + (q_hi - q_lo) * torch.rand(args.n_data, n_q, generator=g)).to(device)
    ze_all = (z_lo + (z_hi - z_lo) * torch.rand(args.n_data, 1, generator=g)).to(device)
    p_all = truth.p_true(q_all, ze_all)
    print(f"data: n={args.n_data}, q in ±{args.q_half}, z_e in [{args.z_min}, {args.z_max}]")

    val_size = max(1024, args.n_data // 20)
    val_q, val_ze, val_p = q_all[-val_size:], ze_all[-val_size:], p_all[-val_size:]
    train_q, train_ze, train_p = q_all[:-val_size], ze_all[:-val_size], p_all[:-val_size]
    n_train = train_q.shape[0]

    delta_net = DeltaResidualMLP(
        n_q=n_q, n_p=2, n_z=1,
        hidden=args.hidden, n_layers=args.n_layers,
    ).to(device)
    optim = torch.optim.Adam(delta_net.parameters(), lr=args.lr)

    fk_fn = lambda q, z: truth.fk_analytic(q, z)

    log_train, log_val = [], []
    pbar = tqdm(range(args.steps), desc=f"train Δ_φ (N={n_q})")
    for step in pbar:
        idx = torch.randint(0, n_train, (args.batch,), device=device)
        q_b, z_b, p_b = train_q[idx], train_ze[idx], train_p[idx]
        loss, parts = self_model_loss(delta_net, q_b, z_b, p_b, fk_fn,
                                      smoothness_weight=args.smoothness)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        log_train.append(parts["fit"])
        if step % 500 == 0:
            with torch.no_grad():
                _, vp = self_model_loss(delta_net, val_q, val_ze, val_p, fk_fn,
                                        smoothness_weight=0.0)
            log_val.append((step, vp["fit"]))
            pbar.set_postfix(fit=f"{parts['fit']:.2e}",
                             val=f"{vp['fit']:.2e}",
                             smooth=f"{parts.get('smooth', 0):.2e}")

    with torch.no_grad():
        delta_pred = delta_net(val_q, val_ze)
        delta_true = truth.delta_true(val_q, val_ze)
        delta_err = (delta_pred - delta_true).norm(dim=-1)
        p_pred = truth.fk_analytic(val_q, val_ze) + delta_pred
        p_err = (p_pred - val_p).norm(dim=-1)
        p_analytic_err = (truth.fk_analytic(val_q, val_ze) - val_p).norm(dim=-1)
    rel = p_err.mean().item() / p_analytic_err.mean().item()
    print()
    print(f"validation (n = {val_size}):")
    print(f"  ‖p_true − analytic FK‖           = {p_analytic_err.mean():.4e}  (oracle Δ_φ ≡ 0)")
    print(f"  ‖p_true − analytic FK − Δ_φ‖     = {p_err.mean():.4e}  (learned)")
    print(f"  ‖Δ_φ − Δ_true‖                   = {delta_err.mean():.4e}")
    print(f"  fit improvement factor           = {1.0/rel:.1f}x")

    # plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(log_train, lw=0.6)
    if log_val:
        s, v = zip(*log_val)
        axes[0].plot(s, v, "o-", color="C1", lw=1, label="val")
        axes[0].legend()
    axes[0].set_yscale("log"); axes[0].set_xlabel("step")
    axes[0].set_ylabel("train fit MSE"); axes[0].set_title(f"Stage-1 Δ_φ training (N={n_q})")
    axes[0].grid(alpha=0.3)

    dp = delta_pred.cpu().numpy(); dt = delta_true.cpu().numpy()
    axes[1].scatter(dt[:, 0], dp[:, 0], s=4, alpha=0.4, label="x")
    axes[1].scatter(dt[:, 1], dp[:, 1], s=4, alpha=0.4, label="y")
    lo, hi = min(dt.min(), dp.min()), max(dt.max(), dp.max())
    axes[1].plot([lo, hi], [lo, hi], "k-", lw=0.6, alpha=0.5)
    axes[1].set_aspect("equal"); axes[1].set_xlabel("Δ_true component")
    axes[1].set_ylabel("Δ_φ component"); axes[1].set_title("Δ_φ vs Δ_true")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    z_axis = val_ze.squeeze(-1).cpu().numpy()
    axes[2].scatter(z_axis, p_err.cpu().numpy(), s=4, alpha=0.4, label="learned")
    axes[2].scatter(z_axis, p_analytic_err.cpu().numpy(), s=4, alpha=0.4, label="oracle Δ=0")
    axes[2].set_xlabel("z_e"); axes[2].set_ylabel("‖p_true − p_pred‖")
    axes[2].set_yscale("log"); axes[2].legend(); axes[2].grid(alpha=0.3)
    axes[2].set_title("residual fit vs embodiment")

    fig.suptitle(f"Toy 3.5 Stage 1 — N-link self-model  (N={n_q})")
    fig.tight_layout()
    fig.savefig(out_dir / "stage1_self_model.png", dpi=120)
    plt.close(fig)

    torch.save({
        "args": vars(args),
        "delta_net_state": delta_net.state_dict(),
        "metrics": {
            "p_err_val": p_err.mean().item(),
            "p_analytic_err_val": p_analytic_err.mean().item(),
            "delta_err_val": delta_err.mean().item(),
            "improvement_factor": 1.0 / rel,
        },
    }, out_dir / "delta_phi.pt")
    print(f"\ncheckpoint saved to  {out_dir / 'delta_phi.pt'}")


if __name__ == "__main__":
    main()
