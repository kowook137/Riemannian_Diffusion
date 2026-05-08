"""Generate architecture diagrams for the SMCDP framework.

Two figures (PNG) under outputs/figures/architecture/:
  1. self_model_arch.png  — Stage-1 Δ_φ residual MLP (DeltaResidualMLP)
  2. score_net_arch.png   — Stage-2 trajectory score net (TrajectoryScoreNetUNet)
                              wrapping ConditionalUnet1D + chart→ambient lift
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D


ART_DIR = Path("outputs/figures/architecture")
ART_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------


def box(ax, x, y, w, h, text, fc="#ffffff", ec="#333333", fontsize=9,
        weight="normal", boxstyle="round,pad=0.05,rounding_size=0.05"):
    rect = FancyBboxPatch((x, y), w, h, boxstyle=boxstyle,
                           facecolor=fc, edgecolor=ec, linewidth=1.0)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight)


def arrow(ax, x1, y1, x2, y2, color="#444444", style="-|>", lw=1.2,
          rad=0.0):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, mutation_scale=12,
                           color=color, lw=lw,
                           connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(arr)


def label(ax, x, y, text, fontsize=8, color="#555555", ha="center", va="center"):
    ax.text(x, y, text, fontsize=fontsize, color=color, ha=ha, va=va,
            style="italic")


# ---------------------------------------------------------------------------
# Figure 1: Stage-1 self-model Δ_φ (DeltaResidualMLP)
# ---------------------------------------------------------------------------


def fig_self_model_arch(n_q=7, n_z=1, n_p=3, hidden=128, n_layers=3):
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.set_xlim(0, 22); ax.set_ylim(0, 7)
    ax.axis("off")

    # Title
    ax.text(11, 6.5, "Stage-1 self-model  $\\Delta_\\phi$  (DeltaResidualMLP)",
            ha="center", va="center", fontsize=14, weight="bold")
    ax.text(11, 6.05,
            "$F_\\phi(q, z_e) = F_\\mathrm{analytic}(q, z_e) + \\Delta_\\phi(q, z_e)$  —  "
            "$\\mathcal{M}_\\phi(z_e) = \\{(q, p) : p = F_\\phi(q, z_e)\\}$",
            ha="center", va="center", fontsize=10, style="italic", color="#444")

    # Input
    box(ax, 0.2, 3.3, 1.4, 1.0,
        f"$q$\n$\\in \\mathbb{{R}}^{{{n_q}}}$",
        fc="#e8f4ff", fontsize=10)
    box(ax, 0.2, 1.8, 1.4, 1.0,
        f"$z_e$\n$\\in \\mathbb{{R}}^{{{n_z}}}$",
        fc="#e8f4ff", fontsize=10)

    # Concat
    box(ax, 2.0, 2.7, 1.4, 1.2, "concat\n$[q, z_e]$",
        fc="#fff5e6", fontsize=9)
    arrow(ax, 1.6, 3.8, 2.0, 3.5)
    arrow(ax, 1.6, 2.3, 2.0, 3.0)

    # IN-LINE Linear → Softplus → Linear → Softplus → Linear → Softplus → Linear(final)
    in_dim = n_q + n_z
    block_w = 1.7
    sp_w = 1.0
    cur_x = 3.6
    arrow(ax, 3.4, 3.3, cur_x, 3.3)
    dim_seq = [in_dim] + [hidden] * n_layers + [n_p]
    # Draw n_layers (Linear + Softplus) blocks
    for i in range(n_layers):
        # Linear
        box(ax, cur_x, 2.7, block_w, 1.2,
            f"Linear\n$\\to {hidden}$",
            fc="#e6f7e6", fontsize=9)
        cur_x += block_w
        arrow(ax, cur_x, 3.3, cur_x + 0.4, 3.3)
        cur_x += 0.4
        # Softplus
        box(ax, cur_x, 2.7, sp_w, 1.2, "Softplus",
            fc="#fff0f0", fontsize=8)
        cur_x += sp_w
        arrow(ax, cur_x, 3.3, cur_x + 0.4, 3.3)
        cur_x += 0.4
    # Final Linear (no Softplus after)
    box(ax, cur_x, 2.7, block_w + 0.2, 1.2,
        f"Linear\n$\\to {n_p}$\n(init × $10^{{-3}}$)",
        fc="#fde4ff", fontsize=9)
    cur_x += block_w + 0.2

    # Output Δ
    arrow(ax, cur_x, 3.3, cur_x + 0.4, 3.3)
    cur_x += 0.4
    box(ax, cur_x, 2.95, 1.4, 0.7,
        f"$\\Delta_\\phi \\in \\mathbb{{R}}^{{{n_p}}}$",
        fc="#fff8d9", fontsize=10, weight="bold")

    # Below-each-Linear dim labels
    cur_x = 3.6 + block_w
    label(ax, cur_x - block_w / 2, 2.4, f"in: {in_dim}", fontsize=7, color="#666")
    for i in range(n_layers + 1):
        # show output dim on top of each Linear (skip the inner ones for now)
        pass

    # Loss block (below)
    box(ax, 6.0, 0.3, 10.0, 1.1,
        "$\\mathcal{L}_\\mathrm{Stage-1} = "
        "\\mathbb{E}\\,\\|F_\\mathrm{ana}(q,z_e) + \\Delta_\\phi - p_\\mathrm{true}\\|^2 "
        "+ \\beta \\cdot \\mathbb{E}\\,\\|\\partial_q \\Delta_\\phi\\|_F^2$\n"
        f"$\\beta = 10^{{-3}}$  (Frobenius-norm Jacobian regulariser, "
        "smoothness on $\\partial \\Delta_\\phi/\\partial q$)",
        fc="#f6f6f6", fontsize=9)

    # Annotation about hyperparameters
    label(ax, 11, 5.3,
          f"{n_layers} hidden Linear+Softplus blocks  ($d \\to {hidden} \\to {hidden} \\to {hidden}$)  "
          f"+ final Linear ($\\to {n_p}$, init × $10^{{-3}}$)",
          fontsize=9, color="#333")
    label(ax, 11, 4.95,
          "Softplus is between Linears (not 3 separate end-of-net activations).  "
          "Total = 4 Linear, 3 Softplus.",
          fontsize=8, color="#666")

    out = ART_DIR / "self_model_arch.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2: Stage-2 score net  TrajectoryScoreNetUNet  (Ours-V2)
# ---------------------------------------------------------------------------


def fig_score_net_arch(n_q=7, n_p=3, n_z=1, H1=16,
                       down_dims=(128, 256, 512),
                       diffusion_step_embed=256,
                       goal_cond_dim=6,
                       cond_injection="channel"):
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 22); ax.set_ylim(0, 16)
    ax.axis("off")

    # Title
    ax.text(11, 15.4,
            "Stage-2 trajectory score net  $s_\\theta(\\tau, t, c)$  "
            "(TrajectoryScoreNetUNet, Ours-V2)",
            ha="center", va="center", fontsize=15, weight="bold")
    ax.text(11, 14.85,
            "ConditionalUnet1D backbone (Chi23)  +  chart→ambient lift  "
            "($T_\\tau \\mathcal{M}_\\phi^{H+1}$ tangent)",
            ha="center", va="center", fontsize=10, style="italic", color="#444")

    # ----------------- Top inputs -----------------
    box(ax, 0.5, 12.7, 3.3, 1.2,
        f"$\\tau \\in \\mathbb{{R}}^{{B \\times (H+1) \\times d}}$\n"
        f"$d = n_q + n_p + n_z = {n_q + n_p + n_z}$",
        fc="#e8f4ff", fontsize=9, weight="bold")
    box(ax, 4.5, 12.7, 2.5, 1.2,
        f"$t \\in \\mathbb{{R}}^B$\n(diffusion time)",
        fc="#e8f4ff", fontsize=9, weight="bold")
    box(ax, 7.7, 12.7, 5.0, 1.2,
        f"$c = \\mathrm{{goal\\_cond}} \\in \\mathbb{{R}}^{{B \\times {goal_cond_dim}}}$\n"
        "$= (p_\\mathrm{target}, p_\\mathrm{start})$  +  $z_e$ (from $\\tau$)",
        fc="#e8f4ff", fontsize=9, weight="bold")

    # ----------------- Slice + channel concat -----------------
    box(ax, 0.5, 11.0, 3.3, 1.2,
        f"slice  $q_\\mathrm{{traj}} = \\tau[..., :{n_q}]$\n$(B, {H1}, {n_q})$",
        fc="#fff5e6", fontsize=9)
    arrow(ax, 2.15, 12.7, 2.15, 12.2)

    box(ax, 7.7, 11.0, 5.0, 1.2,
        f"broadcast  $c$ across {H1} timesteps\n"
        f"channel-concat: $[q_\\mathrm{{traj}}, p_\\mathrm{{target}}, p_\\mathrm{{start}}, z_e]$\n"
        f"$\\in \\mathbb{{R}}^{{B \\times {H1} \\times ({n_q} + {goal_cond_dim} + {n_z})}}$",
        fc="#fff5e6", fontsize=9)
    arrow(ax, 10.2, 12.7, 10.2, 12.2)

    # ----------------- Diffusion-step encoder -----------------
    box(ax, 13.5, 10.0, 4.5, 2.4,
        "Diffusion-step encoder\n"
        f"SinusoidalPosEmb({diffusion_step_embed})\n"
        f"→ Linear  ({diffusion_step_embed} → {diffusion_step_embed * 4})\n"
        f"→ Mish\n"
        f"→ Linear  ({diffusion_step_embed * 4} → {diffusion_step_embed})",
        fc="#f0e6ff", fontsize=8.5)
    arrow(ax, 5.75, 12.7, 13.5, 11.4, rad=-0.1)
    label(ax, 9, 12.0, "$t \\cdot t_\\mathrm{scale}\\,(=1000)$",
          fontsize=8, color="#666")

    # ----------------- ConditionalUnet1D U-shape -----------------
    # Down path (left to right): [n_q+goal_cond+n_z → 128] → [128 → 256] → [256 → 512]
    # Up path: [1024 → 256] → [512 → 128]  with skip
    # Mid: 2× CRB(512)

    unet_y0 = 4.5
    unet_h = 5.0
    box(ax, 1, unet_y0, 16.5, unet_h,
        "", fc="#fdfdfd", ec="#888", boxstyle="round,pad=0.1,rounding_size=0.1")
    ax.text(9.25, unet_y0 + unet_h - 0.35,
            "ConditionalUnet1D  (Chi23 official)",
            ha="center", va="top", fontsize=11, weight="bold")

    # DOWN path
    down_x = [2.0, 4.6, 7.2]
    down_w = 2.0
    down_dims_full = [n_q + goal_cond_dim + n_z] + list(down_dims)  # [14, 128, 256, 512]
    for i, (x, dim_in, dim_out) in enumerate(zip(down_x, down_dims_full[:-1],
                                                   down_dims_full[1:])):
        box(ax, x, unet_y0 + 3.2, down_w, 1.2,
            f"CondResBlk1D\n$({dim_in}\\to{dim_out})$\n× 2", fc="#e6f7e6", fontsize=8.5)
        if i < len(down_x) - 1:
            box(ax, x + 0.7, unet_y0 + 2.45, 0.6, 0.45, "down ×½",
                fc="#d3f0d3", fontsize=7.2)
            arrow(ax, x + down_w, unet_y0 + 3.8, x + down_w + 0.6, unet_y0 + 3.8)

    # MID
    mid_x = 9.8
    box(ax, mid_x, unet_y0 + 3.2, 2.4, 1.2,
        f"Mid CondResBlk1D\n$({down_dims[-1]}\\to{down_dims[-1]})$\n× 2",
        fc="#fff0c6", fontsize=8.5)
    arrow(ax, down_x[-1] + down_w, unet_y0 + 3.8, mid_x, unet_y0 + 3.8)

    # UP path (mirror)
    up_x = [12.6, 15.0]
    up_dims_pairs = [(down_dims[-1] * 2, down_dims[-2]),
                     (down_dims[-2] * 2, down_dims[-3])]
    for i, (x, (dim_in, dim_out)) in enumerate(zip(up_x, up_dims_pairs)):
        box(ax, x, unet_y0 + 3.2, down_w, 1.2,
            f"CondResBlk1D\n$({dim_in}\\to{dim_out})$\n× 2", fc="#e6f7e6", fontsize=8.5)
        if i < len(up_x) - 1:
            box(ax, x + 0.7, unet_y0 + 2.45, 0.6, 0.45, "up ×2",
                fc="#d3f0d3", fontsize=7.2)
            arrow(ax, x + down_w, unet_y0 + 3.8, x + down_w + 0.6, unet_y0 + 3.8)
        else:
            arrow(ax, x + down_w, unet_y0 + 3.8, x + down_w + 0.4, unet_y0 + 3.8)
    arrow(ax, mid_x + 2.4, unet_y0 + 3.8, up_x[0], unet_y0 + 3.8)

    # Skip connections (curved arrows from down to up)
    skip_pairs = [(0, 1), (1, 0)]                              # down[i] → up[H-1-i]
    for di, ui in skip_pairs:
        x_d = down_x[di] + down_w / 2
        x_u = up_x[ui] + down_w / 2
        arrow(ax, x_d, unet_y0 + 3.2, x_u, unet_y0 + 3.2,
              color="#1f78b4", lw=1.2, rad=-0.4, style="-|>")
    label(ax, 9.25, unet_y0 + 1.5, "skip connections (concat at decoder input)",
          fontsize=8, color="#1f78b4")

    # Final conv
    box(ax, 17.0, unet_y0 + 3.2, 1.6, 1.2,
        f"Conv1d\n$\\to$ {n_q + goal_cond_dim + n_z}",
        fc="#fde4ff", fontsize=8.5)
    arrow(ax, up_x[-1] + down_w, unet_y0 + 3.8, 17.0, unet_y0 + 3.8)

    # Cond input to all blocks
    cond_y = unet_y0 + 0.9
    box(ax, 4, cond_y, 12, 0.8,
        "FiLM-modulation in every CondResBlk1D ← (diff-step embed $\\oplus$ global cond)",
        fc="#f0e6ff", fontsize=8.5, ec="#a78cd8")
    arrow(ax, 15.7, 11.0, 15.5, cond_y + 0.8, color="#a78cd8", style="-|>", lw=1.0)

    # Input arrow into UNet
    arrow(ax, 10.2, 11.0, 2.0, unet_y0 + 4.4, rad=-0.05)

    # ----------------- Output: chart score → ambient lift -----------------
    box(ax, 1, 2.7, 4.5, 1.4,
        f"extract first {n_q} channels  →  $s_\\mathrm{{chart}}$\n"
        f"$\\in \\mathbb{{R}}^{{B \\times {H1} \\times {n_q}}}$",
        fc="#fff8d9", fontsize=9)
    arrow(ax, 17.8, unet_y0 + 3.8, 5.5, 4.1, rad=-0.15)

    # Lift block — draws the geometry
    box(ax, 6.5, 2.0, 9.5, 2.4,
        "Per-timestep chart→ambient lift\n"
        "$s_\\mathrm{lift}^h = J_H \\cdot s_\\mathrm{chart}^h$,  "
        "$J_H = [\\,I_{n_q};\\; J_F(q_h, z_e);\\; 0\\,] \\in \\mathbb{R}^{d \\times n_q}$\n"
        "(stacks $I_{n_q}$ over $q$-block, $J_F$ over $p$-block, zeros over $z$-block)\n"
        "$\\Rightarrow s_\\mathrm{amb}^h \\in T_{\\tau_h}\\,\\mathcal{M}_\\phi(z_e) \\subset \\mathbb{R}^d$\n"
        "$J_F = J_\\mathrm{FK} + \\partial \\Delta_\\phi/\\partial q$  (uses Stage-1 self-model)",
        fc="#e8f4ff", fontsize=9, ec="#1f78b4")
    arrow(ax, 5.5, 3.4, 6.5, 3.4)

    # Final score
    box(ax, 17.0, 2.55, 4.0, 1.6,
        "$s_\\theta(\\tau, t, c)$\n"
        f"$\\in \\mathbb{{R}}^{{B \\times {H1} \\times d}}$\n"
        "$\\subset T_\\tau \\mathcal{M}_\\phi^{H+1}$",
        fc="#d3f0d3", fontsize=10, weight="bold", ec="#2ca02c")
    arrow(ax, 16.0, 3.4, 17.0, 3.4)

    # Below: legend / hyperparameters
    box(ax, 1, 0.2, 20, 1.0,
        "Default hyperparams (Ours-V2):  "
        f"$H+1 = {H1}$,  down_dims = {list(down_dims)},  "
        f"diff_step_embed = {diffusion_step_embed},  n_groups = 8 (FiLM),  kernel_size = 3,  "
        f"cond_injection = '{cond_injection}',  goal_cond_dim = {goal_cond_dim} ($p_\\mathrm{{target}} \\oplus p_\\mathrm{{start}}$),  "
        f"$t_\\mathrm{{scale}} = 1000$\n"
        "Riemannian glue: lift uses $J_F = J_\\mathrm{FK} + \\partial \\Delta_\\phi/\\partial q$; "
        "metric $G = I + J_F^\\top J_F$ for DSM-Varadhan loss + tangent noise + retraction-GRW.",
        fc="#f6f6f6", fontsize=8.5)

    out = ART_DIR / "score_net_arch.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Optional: Conditional Residual Block 1D (zoom-in)
# ---------------------------------------------------------------------------


def fig_cond_res_block():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(7, 5.5, "ConditionalResidualBlock1D  (zoom-in of one block)",
            ha="center", va="center", fontsize=13, weight="bold")
    ax.text(7, 5.05,
            "Conv1dBlock = Conv1d(k=3) → GroupNorm(n=8) → Mish",
            ha="center", va="center", fontsize=9, style="italic", color="#444")

    # Input
    box(ax, 0.4, 2.4, 1.5, 0.9, "$x_\\mathrm{in}$\n$(B, C_\\mathrm{in}, T)$", fc="#e8f4ff", fontsize=9)
    # Conv1dBlock 1
    box(ax, 2.4, 2.4, 2.2, 0.9, "Conv1dBlock\n$C_\\mathrm{in} \\to C_\\mathrm{out}$",
        fc="#e6f7e6", fontsize=9)
    arrow(ax, 1.9, 2.85, 2.4, 2.85)

    # FiLM
    box(ax, 5.0, 1.0, 2.5, 1.4,
        "FiLM modulation\n"
        "$h \\leftarrow \\gamma(c) \\odot h + \\beta(c)$",
        fc="#f0e6ff", fontsize=9)
    box(ax, 5.0, 4.3, 2.5, 0.9,
        "Linear $c \\to 2 C_\\mathrm{out}$\n→ scale $\\gamma$, shift $\\beta$",
        fc="#fff0f0", fontsize=8.5)
    arrow(ax, 6.25, 4.3, 6.25, 2.4)
    arrow(ax, 4.6, 2.85, 5.0, 1.7)

    # Conv1dBlock 2
    box(ax, 8.0, 2.4, 2.2, 0.9, "Conv1dBlock\n$C_\\mathrm{out} \\to C_\\mathrm{out}$",
        fc="#e6f7e6", fontsize=9)
    arrow(ax, 7.5, 1.7, 8.0, 2.85)

    # Residual add + 1x1 conv shortcut
    box(ax, 10.5, 2.4, 1.5, 0.9, "$+$ residual",
        fc="#fde4ff", fontsize=9)
    arrow(ax, 10.2, 2.85, 10.5, 2.85)
    # shortcut
    arrow(ax, 1.15, 2.4, 11.25, 0.7, rad=0.4, color="#888")
    label(ax, 6, 0.55, "1×1 Conv shortcut (if $C_\\mathrm{in} \\neq C_\\mathrm{out}$)",
          fontsize=8, color="#666")
    arrow(ax, 11.25, 0.95, 11.25, 2.4, color="#888")

    # Output
    box(ax, 12.4, 2.4, 1.4, 0.9, "$x_\\mathrm{out}$",
        fc="#fff8d9", fontsize=9, weight="bold")
    arrow(ax, 12.0, 2.85, 12.4, 2.85)

    # Cond input on top
    box(ax, 5.4, 5.5, 1.7, 0.0, "")  # spacer
    label(ax, 5.7, 4.65, "$c$  =  diff_step_embed  $\\oplus$  global_cond",
          fontsize=9, color="#5b3a92", ha="center")

    out = ART_DIR / "cond_res_block.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p1 = fig_self_model_arch(n_q=7, n_z=1, n_p=3, hidden=128, n_layers=3)
    print(f"  → {p1}")
    p2 = fig_score_net_arch(n_q=7, n_p=3, n_z=1, H1=16,
                             down_dims=(128, 256, 512),
                             diffusion_step_embed=256,
                             goal_cond_dim=6,
                             cond_injection="channel")
    print(f"  → {p2}")
    p3 = fig_cond_res_block()
    print(f"  → {p3}")
    print(f"\nAll diagrams in {ART_DIR}/")


if __name__ == "__main__":
    main()
