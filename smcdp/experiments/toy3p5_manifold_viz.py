"""Visualise the 3-link planar arm's self-model manifold M_φ(z_e) deforming
continuously as the embodiment context z_e varies.

Outputs (all under outputs/figures/toy3p5_manifold_viz/):
  1. arm_pose_sweep.png      — arm pose at fixed q, multiple z_e overlaid
  2. workspace_sweep.png     — reachable EE region for several z_e
  3. manifold_slice.png      — 2-D slice of M_φ at fixed q_1, deforming with z_e
  4. residual_field.png      — Δ_φ(q, z_e) vs Δ_true and analytic-only baseline
  5. arm_animation.gif       — arm pose evolving as z_e: 0 → 0.45
  6. workspace_animation.gif — workspace + manifold slice evolving with z_e
  7. surface_3d_static.png   — 3-D parametric surface of the manifold slice
                                at multiple z_e (overlaid)
  8. surface_3d_animation.gif — same surface deforming continuously (3D rotating)
  9. bundle_3d.png           — manifold "bundle" $\bigcup_{z_e} \mathcal{M}_\phi$
                                stacked along z_e axis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as anim
import numpy as np
import torch

from smcdp.toy3.ground_truth import TrueNLinkArmCompliance
from smcdp.toy3.self_model import DeltaResidualMLP


ART_DIR = Path("outputs/figures/toy3p5_manifold_viz")
ART_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Pure FK helpers (analytic + learned residual + true compliance)
# ---------------------------------------------------------------------------


def link_endpoints(q: torch.Tensor, z_e: torch.Tensor,
                   link_lengths_base=(1.0, 1.0, 0.5)) -> torch.Tensor:
    """Return per-joint cartesian positions [(0,0), p1, p2, p3] for batched q.

    q: (B, 3), z_e: (B, 1)  →  pts: (B, 4, 2).
    Last link length = ℓ_3_base + z_e.
    """
    B = q.shape[0]
    l_base = torch.tensor(link_lengths_base, dtype=q.dtype, device=q.device)
    l_eff = l_base.repeat(B, 1).clone()                                # (B, 3)
    l_eff[:, -1] = l_base[-1] + z_e[:, 0]
    s = torch.cumsum(q, dim=-1)                                         # (B, 3) cumulative
    cs, sn = torch.cos(s), torch.sin(s)
    px = torch.cumsum(l_eff * cs, dim=-1)                               # joint x positions
    py = torch.cumsum(l_eff * sn, dim=-1)
    base = torch.zeros(B, 1, 2, dtype=q.dtype, device=q.device)
    pts = torch.stack([px, py], dim=-1)                                 # (B, 3, 2)
    return torch.cat([base, pts], dim=1)                                # (B, 4, 2)


def fk_analytic(q: torch.Tensor, z_e: torch.Tensor,
                link_lengths_base=(1.0, 1.0, 0.5)) -> torch.Tensor:
    pts = link_endpoints(q, z_e, link_lengths_base)
    return pts[:, -1, :]


def fk_learned(q: torch.Tensor, z_e: torch.Tensor, delta_net) -> torch.Tensor:
    return fk_analytic(q, z_e) + delta_net(q, z_e)


# ---------------------------------------------------------------------------
# 1. Arm pose at fixed q, multiple z_e
# ---------------------------------------------------------------------------


def fig_arm_pose(q_fixed: torch.Tensor, z_list, delta_net=None, truth=None):
    """Side-by-side: analytic + learned (Δ_φ) + true (Δ_true) arm tip at fixed q.

    Each panel overlays the arm at multiple z_e values.  Shows the third link
    elongating (analytic) and the curving sag/offset (learned, true).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Analytic FK only", "Learned manifold $F_\\phi = \\mathrm{FK} + \\Delta_\\phi$",
              "True compliance $F_\\mathrm{true} = \\mathrm{FK} + \\Delta_\\mathrm{true}$"]
    cmap = cm.viridis
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, 2.6); ax.set_ylim(-1.2, 1.6)
        ax.grid(alpha=0.3)
        ax.set_xlabel("$p_x$"); ax.set_ylabel("$p_y$")

    n = len(z_list)
    for i, z_val in enumerate(z_list):
        z = torch.tensor([[z_val]], dtype=torch.float32)
        c = cmap(i / max(n - 1, 1))

        # Analytic
        pts_a = link_endpoints(q_fixed.unsqueeze(0), z).squeeze(0).cpu().numpy()
        axes[0].plot(pts_a[:, 0], pts_a[:, 1], "o-", color=c, lw=2, ms=6,
                     label=f"$z_e$={z_val:.2f}")
        # Learned Δ_φ — same q, but tip moved by Δ_φ
        if delta_net is not None:
            with torch.no_grad():
                p_learned = fk_learned(q_fixed.unsqueeze(0), z, delta_net).squeeze(0).cpu().numpy()
            pts_lp = pts_a.copy(); pts_lp[-1] = p_learned
            axes[1].plot(pts_lp[:, 0], pts_lp[:, 1], "o-", color=c, lw=2, ms=6,
                         label=f"$z_e$={z_val:.2f}")
        # True Δ_true
        if truth is not None:
            with torch.no_grad():
                p_true = truth.p_true(q_fixed.unsqueeze(0), z).squeeze(0).cpu().numpy()
            pts_t = pts_a.copy(); pts_t[-1] = p_true
            axes[2].plot(pts_t[:, 0], pts_t[:, 1], "o-", color=c, lw=2, ms=6,
                         label=f"$z_e$={z_val:.2f}")

    for ax in axes:
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Arm pose at fixed q = {q_fixed.tolist()} as $z_e$ varies\n"
                 "(third link elongates with $z_e$ in analytic; "
                 "$\\Delta_\\phi$ adds compliance sag + offset)", y=1.02)
    fig.tight_layout()
    out = ART_DIR / "arm_pose_sweep.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 2. EE workspace (reachable region) for several z_e
# ---------------------------------------------------------------------------


def fig_workspace(z_list, n_samples=12000, q_half=1.2,
                  delta_net=None, truth=None):
    """Sample q ~ U([-q_half, q_half]^3), compute EE for each z_e, scatter.

    Panel 1: analytic FK only
    Panel 2: learned F_φ
    Panel 3: true F_true
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["$\\mathrm{Im}(F_\\mathrm{analytic}(\\cdot, z_e))$",
              "$\\mathrm{Im}(F_\\phi(\\cdot, z_e))$",
              "$\\mathrm{Im}(F_\\mathrm{true}(\\cdot, z_e))$"]
    cmap = cm.viridis
    n = len(z_list)
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("$p_x$"); ax.set_ylabel("$p_y$")
        ax.grid(alpha=0.3)

    torch.manual_seed(42)
    q = (-q_half + 2 * q_half * torch.rand(n_samples, 3))
    for i, z_val in enumerate(z_list):
        z = torch.full((n_samples, 1), z_val)
        c = cmap(i / max(n - 1, 1))

        p_a = fk_analytic(q, z).cpu().numpy()
        axes[0].scatter(p_a[:, 0], p_a[:, 1], s=1.2, c=[c], alpha=0.4,
                        label=f"$z_e$={z_val:.2f}")
        if delta_net is not None:
            with torch.no_grad():
                p_l = fk_learned(q, z, delta_net).cpu().numpy()
            axes[1].scatter(p_l[:, 0], p_l[:, 1], s=1.2, c=[c], alpha=0.4,
                            label=f"$z_e$={z_val:.2f}")
        if truth is not None:
            with torch.no_grad():
                p_t = truth.p_true(q, z).cpu().numpy()
            axes[2].scatter(p_t[:, 0], p_t[:, 1], s=1.2, c=[c], alpha=0.4,
                            label=f"$z_e$={z_val:.2f}")

    for ax in axes:
        ax.legend(loc="lower right", fontsize=8, markerscale=4)
    fig.suptitle("End-effector workspace = $\\mathrm{Im}(F(\\cdot, z_e))$ — "
                 "expanding shells as $z_e$ grows", y=1.02)
    fig.tight_layout()
    out = ART_DIR / "workspace_sweep.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 3. 2-D manifold slice at fixed q_1, varying (q_2, q_3)
# ---------------------------------------------------------------------------


def fig_manifold_slice(z_list, q1_fixed=0.0, n_grid=80, q_half=1.2,
                       delta_net=None, truth=None):
    """At q_1 fixed, the image of (q_2, q_3) ↦ p ∈ R^2 is a 2-D submanifold
    of the EE plane.  Plot it as a colored mesh; deformation by z_e is
    visible as a continuous warp.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Analytic slice $\\{(q_2, q_3) \\mapsto F_\\mathrm{ana}\\}$, $q_1=$" + f"{q1_fixed:.2f}",
              "Learned slice $F_\\phi$",
              "True slice $F_\\mathrm{true}$"]
    cmap = cm.plasma
    n = len(z_list)
    for ax, title in zip(axes, titles):
        ax.set_title(title); ax.set_aspect("equal")
        ax.set_xlabel("$p_x$"); ax.set_ylabel("$p_y$"); ax.grid(alpha=0.3)

    g = torch.linspace(-q_half, q_half, n_grid)
    q2, q3 = torch.meshgrid(g, g, indexing="ij")
    q2 = q2.reshape(-1); q3 = q3.reshape(-1)
    q1 = torch.full_like(q2, q1_fixed)
    q = torch.stack([q1, q2, q3], dim=-1)                               # (G^2, 3)

    for i, z_val in enumerate(z_list):
        z = torch.full((q.shape[0], 1), z_val)
        c = cmap(i / max(n - 1, 1))
        p_a = fk_analytic(q, z).cpu().numpy()
        axes[0].plot(p_a[:, 0], p_a[:, 1], ",", color=c, alpha=0.45,
                     label=f"$z_e$={z_val:.2f}")
        if delta_net is not None:
            with torch.no_grad():
                p_l = fk_learned(q, z, delta_net).cpu().numpy()
            axes[1].plot(p_l[:, 0], p_l[:, 1], ",", color=c, alpha=0.45,
                         label=f"$z_e$={z_val:.2f}")
        if truth is not None:
            with torch.no_grad():
                p_t = truth.p_true(q, z).cpu().numpy()
            axes[2].plot(p_t[:, 0], p_t[:, 1], ",", color=c, alpha=0.45,
                         label=f"$z_e$={z_val:.2f}")
    for ax in axes:
        # synthetic legend with bigger markers
        h = [plt.Line2D([0], [0], marker="o", linestyle="", color=cmap(i / max(n - 1, 1)),
                         label=f"$z_e$={z_val:.2f}", markersize=6)
             for i, z_val in enumerate(z_list)]
        ax.legend(handles=h, loc="lower right", fontsize=8)
    fig.suptitle(f"2-D manifold slice at $q_1$={q1_fixed:.2f}: "
                 "homeomorphic deformation as $z_e$ varies", y=1.02)
    fig.tight_layout()
    out = ART_DIR / "manifold_slice.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 4. Residual deformation field — Δ_φ(q, z_e) and Δ_true(q, z_e)
# ---------------------------------------------------------------------------


def fig_residual_field(z_list, q1_fixed=0.0, n_grid=20, q_half=1.0,
                       delta_net=None, truth=None):
    """Quiver of Δ_φ(q, z_e) at sampled q grid.  Shows compliance direction +
    magnitude as z_e changes.  Compares learned vs true.
    """
    n = len(z_list)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7.5))
    if n == 1:
        axes = axes.reshape(2, 1)

    g = torch.linspace(-q_half, q_half, n_grid)
    q2, q3 = torch.meshgrid(g, g, indexing="ij")
    q2 = q2.reshape(-1); q3 = q3.reshape(-1)
    q1 = torch.full_like(q2, q1_fixed)
    q = torch.stack([q1, q2, q3], dim=-1)

    for j, z_val in enumerate(z_list):
        z = torch.full((q.shape[0], 1), z_val)
        with torch.no_grad():
            p_a = fk_analytic(q, z).cpu().numpy()
            d_l = (delta_net(q, z) if delta_net is not None else torch.zeros_like(p_a)).cpu().numpy() if delta_net is not None else np.zeros_like(p_a)
            d_t = truth.delta_true(q, z).cpu().numpy() if truth is not None else np.zeros_like(p_a)

        # Subsample the grid for legibility
        stride = max(1, n_grid // 14)
        idx = np.arange(0, n_grid * n_grid).reshape(n_grid, n_grid)[::stride, ::stride].reshape(-1)

        ax = axes[0, j]
        ax.quiver(p_a[idx, 0], p_a[idx, 1], d_l[idx, 0], d_l[idx, 1],
                  scale_units="xy", scale=1.0, color="C0", alpha=0.85, width=0.005)
        ax.set_title(f"$\\Delta_\\phi$  at $z_e$={z_val:.2f}")
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlabel("$p_x$"); ax.set_ylabel("$p_y$")

        ax = axes[1, j]
        ax.quiver(p_a[idx, 0], p_a[idx, 1], d_t[idx, 0], d_t[idx, 1],
                  scale_units="xy", scale=1.0, color="C3", alpha=0.85, width=0.005)
        ax.set_title(f"$\\Delta_\\mathrm{{true}}$ at $z_e$={z_val:.2f}")
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlabel("$p_x$")

    fig.suptitle(f"Compliance field at $q_1$={q1_fixed:.2f} — "
                 "row 1 = learned $\\Delta_\\phi$, row 2 = true $\\Delta_\\mathrm{true}$\n"
                 "(magnitude grows ∝ $z_e^2$ for sag, ∝ $z_e$ for offset)", y=1.02)
    fig.tight_layout()
    out = ART_DIR / "residual_field.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 5. Animation: arm + workspace evolving as z_e: 0 → z_max
# ---------------------------------------------------------------------------


def gif_arm_animation(q_fixed: torch.Tensor, z_max=0.45, n_frames=60,
                      delta_net=None, truth=None):
    """Animate the arm pose and the EE traced as z_e ramps 0 → z_max."""
    z_vals = np.linspace(0.0, z_max, n_frames)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect("equal")
    ax.set_xlim(-0.6, 2.8); ax.set_ylim(-1.2, 1.6)
    ax.grid(alpha=0.3)
    ax.set_xlabel("$p_x$"); ax.set_ylabel("$p_y$")

    line_a, = ax.plot([], [], "o-", color="C0", lw=2.5, ms=7,
                      label="analytic $F_\\mathrm{ana}$")
    line_l, = ax.plot([], [], "o-", color="C2", lw=2.5, ms=7,
                      label="learned $F_\\phi$")
    line_t, = ax.plot([], [], "o-", color="C3", lw=2.5, ms=7,
                      label="true $F_\\mathrm{true}$")
    title = ax.set_title("")
    ax.legend(loc="lower right", fontsize=10)

    # Trace EE tips across z_e
    trail_a, = ax.plot([], [], "-", color="C0", lw=1.0, alpha=0.5)
    trail_l, = ax.plot([], [], "-", color="C2", lw=1.0, alpha=0.5)
    trail_t, = ax.plot([], [], "-", color="C3", lw=1.0, alpha=0.5)
    trail_a_x, trail_a_y, trail_l_x, trail_l_y, trail_t_x, trail_t_y = [], [], [], [], [], []

    def update(frame):
        z_val = z_vals[frame]
        z = torch.tensor([[z_val]], dtype=torch.float32)
        pts_a = link_endpoints(q_fixed.unsqueeze(0), z).squeeze(0).numpy()
        line_a.set_data(pts_a[:, 0], pts_a[:, 1])
        trail_a_x.append(pts_a[-1, 0]); trail_a_y.append(pts_a[-1, 1])
        trail_a.set_data(trail_a_x, trail_a_y)

        if delta_net is not None:
            with torch.no_grad():
                p_l = fk_learned(q_fixed.unsqueeze(0), z, delta_net).squeeze(0).numpy()
            pts_lp = pts_a.copy(); pts_lp[-1] = p_l
            line_l.set_data(pts_lp[:, 0], pts_lp[:, 1])
            trail_l_x.append(p_l[0]); trail_l_y.append(p_l[1])
            trail_l.set_data(trail_l_x, trail_l_y)
        if truth is not None:
            with torch.no_grad():
                p_t = truth.p_true(q_fixed.unsqueeze(0), z).squeeze(0).numpy()
            pts_tp = pts_a.copy(); pts_tp[-1] = p_t
            line_t.set_data(pts_tp[:, 0], pts_tp[:, 1])
            trail_t_x.append(p_t[0]); trail_t_y.append(p_t[1])
            trail_t.set_data(trail_t_x, trail_t_y)
        title.set_text(f"q = {q_fixed.tolist()},  $z_e$ = {z_val:.3f}  "
                       f"(training ≤ 0.30, OOD = 0.45)")
        return line_a, line_l, line_t, trail_a, trail_l, trail_t, title

    anim_obj = anim.FuncAnimation(fig, update, frames=n_frames, blit=False, interval=80)
    out = ART_DIR / "arm_animation.gif"
    anim_obj.save(out, writer="pillow", dpi=110)
    plt.close(fig)
    return out


def gif_workspace_animation(z_max=0.45, n_frames=60, n_samples=4000,
                            q_half=1.2, delta_net=None, truth=None,
                            q1_slice=0.0, n_slice=60):
    """Animate workspace + manifold slice as z_e ramps 0 → z_max."""
    z_vals = np.linspace(0.0, z_max, n_frames)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax in axes: ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax_ws, ax_sl = axes
    ax_ws.set_title("EE workspace $F_\\phi(q, z_e)$ for q ~ U")
    ax_sl.set_title(f"2-D manifold slice  $F_\\phi(q_1={q1_slice:.2f}, q_2, q_3, z_e)$")
    for ax in axes:
        ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)
        ax.set_xlabel("$p_x$"); ax.set_ylabel("$p_y$")

    torch.manual_seed(7)
    q_ws = (-q_half + 2 * q_half * torch.rand(n_samples, 3))

    g = torch.linspace(-q_half, q_half, n_slice)
    q2g, q3g = torch.meshgrid(g, g, indexing="ij")
    q2g = q2g.reshape(-1); q3g = q3g.reshape(-1)
    q1g = torch.full_like(q2g, q1_slice)
    q_sl = torch.stack([q1g, q2g, q3g], dim=-1)

    sc_ws = ax_ws.scatter([], [], s=2.5, c="C2", alpha=0.5)
    sc_sl = ax_sl.scatter([], [], s=2.5, c="C2", alpha=0.5)
    title = fig.suptitle("")

    def update(frame):
        z_val = z_vals[frame]
        z_ws = torch.full((q_ws.shape[0], 1), z_val)
        z_sl = torch.full((q_sl.shape[0], 1), z_val)
        with torch.no_grad():
            p_ws = (fk_learned(q_ws, z_ws, delta_net) if delta_net is not None
                    else fk_analytic(q_ws, z_ws)).numpy()
            p_sl = (fk_learned(q_sl, z_sl, delta_net) if delta_net is not None
                    else fk_analytic(q_sl, z_sl)).numpy()
        sc_ws.set_offsets(p_ws); sc_sl.set_offsets(p_sl)
        # color by z_e for visual continuity
        c = cm.viridis(z_val / max(z_max, 1e-6))
        sc_ws.set_color(c); sc_sl.set_color(c)
        title.set_text(f"$z_e$ = {z_val:.3f}  (training ≤ 0.30, OOD = 0.45)")
        return sc_ws, sc_sl, title

    anim_obj = anim.FuncAnimation(fig, update, frames=n_frames, blit=False, interval=80)
    out = ART_DIR / "workspace_animation.gif"
    anim_obj.save(out, writer="pillow", dpi=110)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 7-9. 3-D parametric-surface visualisations of the manifold slice
# ---------------------------------------------------------------------------


def _surface_grid(q1_fixed: float, n_grid: int, q_half: float, z_val: float,
                  delta_net=None):
    """Return meshgrid arrays (Q2, Q3, PX, PY) for surface plotting.

    Parametrise (q_2, q_3) ∈ [-q_half, q_half]^2 with q_1 fixed.  Compute
    (p_x, p_y) = F_φ(q, z_e) at every grid point; arrays shaped (n_grid, n_grid).
    """
    g = torch.linspace(-q_half, q_half, n_grid)
    Q2, Q3 = torch.meshgrid(g, g, indexing="ij")
    q1 = torch.full_like(Q2, q1_fixed)
    q = torch.stack([q1.reshape(-1), Q2.reshape(-1), Q3.reshape(-1)], dim=-1)
    z = torch.full((q.shape[0], 1), z_val)
    with torch.no_grad():
        p = (fk_learned(q, z, delta_net) if delta_net is not None
             else fk_analytic(q, z))
    PX = p[:, 0].reshape(n_grid, n_grid).numpy()
    PY = p[:, 1].reshape(n_grid, n_grid).numpy()
    return Q2.numpy(), Q3.numpy(), PX, PY


def fig_surface_3d_static(z_list, q1_fixed=0.0, n_grid=40, q_half=1.2,
                          delta_net=None):
    """Faithful graph-of-F embedding of the 2-D slice of M_φ in 3-D.

    The full manifold M_φ = {(q, p) : p = F_φ(q, z_e)} is 3-D in R^5.  Fixing
    q_1 = q1_fixed yields a 2-D slice ⊂ M_φ, parameterised by (q_2, q_3); a
    *faithful* (one-to-one, no fold) embedding into R^3 is the graph
        (q_2, q_3) ↦ (q_2, q_3, p_α(0, q_2, q_3, z_e))
    where p_α is one component of the EE position (α ∈ {x, y}).  Both
    components are needed to fully characterise the slice — we plot them in
    two panels.

    Two panels: graph of p_x and graph of p_y over (q_2, q_3).  Each surface
    is the genuine slice of M_φ (no projection-fold).  As z_e grows the
    surface height p_α(·, ·, z_e) deforms continuously — that *is* the
    manifold deformation.
    """
    from mpl_toolkits.mplot3d import Axes3D                            # noqa: F401
    fig = plt.figure(figsize=(14, 6.5))
    ax_x = fig.add_subplot(121, projection="3d")
    ax_y = fig.add_subplot(122, projection="3d")
    cmap = cm.viridis
    n = len(z_list)
    for i, z_val in enumerate(z_list):
        Q2, Q3, PX, PY = _surface_grid(q1_fixed, n_grid, q_half, z_val, delta_net)
        c = cmap(i / max(n - 1, 1))
        ax_x.plot_surface(Q2, Q3, PX, color=c, alpha=0.45, linewidth=0,
                           antialiased=True, rcount=n_grid, ccount=n_grid)
        ax_y.plot_surface(Q2, Q3, PY, color=c, alpha=0.45, linewidth=0,
                           antialiased=True, rcount=n_grid, ccount=n_grid)
    handles = [plt.Line2D([0], [0], marker="s", linestyle="",
                           color=cmap(i / max(n - 1, 1)),
                           label=f"$z_e$={z_val:.2f}", markersize=10)
               for i, z_val in enumerate(z_list)]
    for ax, comp, label in [(ax_x, "x", "$p_x$"), (ax_y, "y", "$p_y$")]:
        ax.set_xlabel("$q_2$"); ax.set_ylabel("$q_3$"); ax.set_zlabel(label)
        ax.set_title(f"slice $\\mathcal{{M}}_\\phi \\cap \\{{q_1={q1_fixed:.2f}\\}}$ "
                      f"as graph of $p_{comp}(q_2, q_3, z_e)$")
        ax.legend(handles=handles, loc="upper left", fontsize=8)
    fig.suptitle("Faithful 3-D embedding of $\\mathcal{M}_\\phi$ slice "
                  "(graph of $F$ over chart $(q_2, q_3)$) — "
                  "manifold deformation as $z_e$ varies", y=1.02)
    fig.tight_layout()
    out = ART_DIR / "surface_3d_static.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def gif_surface_3d_animation(q_demo: torch.Tensor, z_max=0.45, n_frames=60,
                              q1_fixed=0.0, n_grid=36, q_half=1.2,
                              delta_net=None, truth=None, rotate=True):
    """Animate the *faithful* 2-D slice of M_φ deforming with z_e in 3-D,
    with the physical arm overlaid in a side panel.

    Left panel: graph $(q_2, q_3, p_x(0, q_2, q_3, z_e))$ — slice of
    $\\mathcal{M}_\\phi \\cap \\{q_1 = q_1^\\text{fixed}\\}$ embedded as a
    function graph (one-to-one, no projection fold).  The vertical axis
    $p_x$ deforms continuously as $z_e$ grows — *this is the manifold
    deformation itself*.

    Right panel: physical arm at $q$ = q_demo in the 2-D Cartesian plane,
    with EE tip markers for analytic / learned / true models, animated in
    sync with $z_e$.  This shows what the chart-coordinate change "looks
    like" physically.
    """
    from mpl_toolkits.mplot3d import Axes3D                            # noqa: F401
    z_vals = np.linspace(0.0, z_max, n_frames)

    fig = plt.figure(figsize=(14, 6.5))
    ax3 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)
    ax3.set_xlabel("$q_2$"); ax3.set_ylabel("$q_3$"); ax3.set_zlabel("$p_x$")
    ax3.set_xlim(-q_half, q_half); ax3.set_ylim(-q_half, q_half)
    ax3.set_zlim(-3.0, 3.0)
    ax2.set_aspect("equal"); ax2.grid(alpha=0.3)
    ax2.set_xlim(-0.6, 2.8); ax2.set_ylim(-1.2, 1.6)
    ax2.set_xlabel("$p_x$"); ax2.set_ylabel("$p_y$")

    # Pre-compute initial surface and arm
    title3 = ax3.set_title("")
    title2 = ax2.set_title("")

    # Persistent artists
    artist = {"surf": None, "marker3d": None,
              "arm": None, "tip_a": None, "tip_l": None, "tip_t": None,
              "trail_a": None, "trail_l": None, "trail_t": None}
    trail_x_a, trail_y_a = [], []
    trail_x_l, trail_y_l = [], []
    trail_x_t, trail_y_t = [], []

    def update(frame):
        z_val = float(z_vals[frame])
        z_t = torch.tensor([[z_val]], dtype=torch.float32)

        # ---- Left: 3-D faithful graph (q_2, q_3, p_x) at q_1 = q1_fixed ----
        Q2, Q3, PX, PY = _surface_grid(q1_fixed, n_grid, q_half, z_val, delta_net)
        for k in ("surf", "marker3d"):
            a = artist[k]
            if a is not None:
                try: a.remove()
                except Exception: pass
                artist[k] = None
        artist["surf"] = ax3.plot_surface(
            Q2, Q3, PX, cmap=cm.viridis, alpha=0.7, linewidth=0,
            antialiased=True, rcount=n_grid, ccount=n_grid,
            edgecolor="none", vmin=-3.0, vmax=3.0,
        )
        # Mark the q_demo's location on the slice (only valid if q_demo[0] ≈ q1_fixed,
        # but we still draw it for reference)
        with torch.no_grad():
            p_demo = fk_analytic(q_demo.unsqueeze(0), z_t).squeeze(0).numpy()
        artist["marker3d"] = ax3.scatter(
            [float(q_demo[1])], [float(q_demo[2])], [float(p_demo[0])],
            s=120, c="red", edgecolor="black", zorder=6,
            label="$q$=demo on slice")
        title3.set_text(f"$\\mathcal{{M}}_\\phi \\cap \\{{q_1={q1_fixed:.2f}\\}}$ "
                        f"as graph $(q_2, q_3, p_x)$  |  "
                        f"$z_e$={z_val:.3f}")
        if rotate:
            ax3.view_init(elev=24, azim=-65 + frame * 0.7)

        # ---- Right: physical arm in 2-D ----
        for k in ("arm", "tip_a", "tip_l", "tip_t",
                  "trail_a", "trail_l", "trail_t"):
            a = artist[k]
            if a is not None:
                try: a.remove()
                except Exception: pass
                artist[k] = None
        pts_a = link_endpoints(q_demo.unsqueeze(0), z_t).squeeze(0).numpy()
        artist["arm"] = ax2.plot(pts_a[:, 0], pts_a[:, 1],
                                   "o-", color="black", lw=2.8, ms=7,
                                   zorder=5)[0]
        artist["tip_a"] = ax2.scatter([pts_a[-1, 0]], [pts_a[-1, 1]],
                                        s=80, c="C0", edgecolor="black",
                                        zorder=6, label="analytic tip")
        trail_x_a.append(pts_a[-1, 0]); trail_y_a.append(pts_a[-1, 1])
        artist["trail_a"] = ax2.plot(trail_x_a, trail_y_a, "-", color="C0",
                                       lw=1.0, alpha=0.5)[0]
        if delta_net is not None:
            with torch.no_grad():
                p_l = fk_learned(q_demo.unsqueeze(0), z_t, delta_net).squeeze(0).numpy()
            artist["tip_l"] = ax2.scatter([p_l[0]], [p_l[1]],
                                            s=80, c="C2", edgecolor="black",
                                            zorder=6, label="learned tip")
            trail_x_l.append(p_l[0]); trail_y_l.append(p_l[1])
            artist["trail_l"] = ax2.plot(trail_x_l, trail_y_l, "-",
                                           color="C2", lw=1.0, alpha=0.5)[0]
        if truth is not None:
            with torch.no_grad():
                p_t = truth.p_true(q_demo.unsqueeze(0), z_t).squeeze(0).numpy()
            artist["tip_t"] = ax2.scatter([p_t[0]], [p_t[1]],
                                            s=80, c="C3", edgecolor="black",
                                            zorder=6, label="true tip")
            trail_x_t.append(p_t[0]); trail_y_t.append(p_t[1])
            artist["trail_t"] = ax2.plot(trail_x_t, trail_y_t, "-",
                                           color="C3", lw=1.0, alpha=0.5)[0]
        title2.set_text(f"physical arm at $q$={q_demo.tolist()}  |  "
                         f"$z_e$={z_val:.3f}  "
                         "(blue=analytic, green=learned $\\Delta_\\phi$, "
                         "red=true $\\Delta_\\mathrm{true}$)")

    anim_obj = anim.FuncAnimation(fig, update, frames=n_frames, blit=False,
                                   interval=80)
    out = ART_DIR / "surface_3d_animation.gif"
    anim_obj.save(out, writer="pillow", dpi=110)
    plt.close(fig)
    return out


def fig_bundle_3d(z_list, n_samples=4000, q_half=1.2, delta_net=None):
    """Workspace bundle in $(p_x, p_y, z_e)$ — stacked workspaces along the
    embodiment axis.  Static figure showing the entire family at once."""
    from mpl_toolkits.mplot3d import Axes3D                            # noqa: F401
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    torch.manual_seed(11)
    q = (-q_half + 2 * q_half * torch.rand(n_samples, 3))
    cmap = cm.viridis
    n = len(z_list)
    for i, z_val in enumerate(z_list):
        z = torch.full((n_samples, 1), z_val)
        with torch.no_grad():
            p = (fk_learned(q, z, delta_net) if delta_net is not None
                 else fk_analytic(q, z)).numpy()
        c = cmap(i / max(n - 1, 1))
        ax.scatter(p[:, 0], p[:, 1], np.full(n_samples, z_val),
                   s=2, c=[c], alpha=0.35, depthshade=False)
    ax.set_xlabel("$p_x$"); ax.set_ylabel("$p_y$"); ax.set_zlabel("$z_e$")
    ax.set_title("Workspace bundle $\\{(p, z_e) : p \\in \\mathrm{Im}\\, F_\\phi(\\cdot, z_e)\\}$\n"
                 "expanding shells stacked along the embodiment axis")
    fig.tight_layout()
    out = ART_DIR / "bundle_3d.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="outputs/toy3p5_stage1/delta_phi.pt")
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--no-anim", action="store_true",
                   help="skip GIF generation (faster)")
    args = p.parse_args()

    # --- Load Stage-1 self-model ---
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    link_lengths = list(a["link_lengths_base"])                          # [1.0, 1.0, 0.5]
    delta_net = DeltaResidualMLP(n_q=3, n_p=2, n_z=1,
                                  hidden=a["hidden"], n_layers=a["n_layers"],
                                  activation=torch.nn.Softplus,
                                  final_init_scale=1e-3)
    delta_net.load_state_dict(ck["delta_net_state"])
    delta_net.eval()
    truth = TrueNLinkArmCompliance(link_lengths_base=link_lengths,
                                    K_grav=a["K_grav"], K_offset=a["K_offset"])

    # z_e sweep: 0, 0.10, 0.20, 0.30 (training); 0.45 (OOD)
    z_list = [0.0, 0.10, 0.20, 0.30, 0.45]
    q_demo = torch.tensor([0.4, -0.6, 0.5], dtype=torch.float32)
    print(f"link_lengths_base = {link_lengths},  z_train = [{a['z_min']}, {a['z_max']}]")

    print("(1) Arm pose sweep …")
    p1 = fig_arm_pose(q_demo, z_list, delta_net=delta_net, truth=truth)
    print(f"    → {p1}")

    print("(2) Workspace sweep …")
    p2 = fig_workspace(z_list, n_samples=12000, q_half=a["q_half"],
                       delta_net=delta_net, truth=truth)
    print(f"    → {p2}")

    print("(3) Manifold 2-D slice …")
    p3 = fig_manifold_slice(z_list, q1_fixed=0.0, n_grid=80, q_half=a["q_half"],
                            delta_net=delta_net, truth=truth)
    print(f"    → {p3}")

    print("(4) Residual deformation field …")
    p4 = fig_residual_field([0.0, 0.10, 0.20, 0.30, 0.45], q1_fixed=0.0,
                            n_grid=24, q_half=1.0,
                            delta_net=delta_net, truth=truth)
    print(f"    → {p4}")

    print("(7) 3-D surface (static, multi-z_e) …")
    p7 = fig_surface_3d_static(z_list, q1_fixed=0.0, n_grid=40,
                                q_half=a["q_half"], delta_net=delta_net)
    print(f"    → {p7}")

    print("(8) 3-D workspace bundle …")
    p9 = fig_bundle_3d(z_list, n_samples=3500, q_half=a["q_half"],
                        delta_net=delta_net)
    print(f"    → {p9}")

    if args.no_anim:
        print("\nSkipping GIF generation (--no-anim).")
        return

    print("(9) Arm animation (GIF, 2-D) …")
    p5 = gif_arm_animation(q_demo, z_max=0.45, n_frames=args.frames,
                            delta_net=delta_net, truth=truth)
    print(f"    → {p5}")

    print("(10) Workspace animation (GIF, 2-D) …")
    p6 = gif_workspace_animation(z_max=0.45, n_frames=args.frames,
                                  n_samples=4000, q_half=a["q_half"],
                                  delta_net=delta_net, truth=truth)
    print(f"    → {p6}")

    print("(11) 3-D surface + arm animation (GIF) …")
    p8 = gif_surface_3d_animation(q_demo, z_max=0.45, n_frames=args.frames,
                                   q1_fixed=0.0, n_grid=32,
                                   q_half=a["q_half"],
                                   delta_net=delta_net, truth=truth, rotate=True)
    print(f"    → {p8}")

    print(f"\nAll figures in {ART_DIR}/")


if __name__ == "__main__":
    main()
