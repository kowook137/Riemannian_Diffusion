# SMCDP — Self-Model Manifold Diffusion Policy

Riemannian Score-Based Imitation Learning on **learned robot self-model manifolds**.

This repo contains the reference implementation for the framework, the diagnostic + ablation toolchain, and all experiments described in `REPORT.md`.

## Quick start

```bash
# 1. Clone (server side)
git clone https://github.com/kowook137/Riemannian_Diffusion.git smcdp
cd smcdp

# 2. Create env (Python ≥ 3.10) and install Python deps
python -m venv .venv && source .venv/bin/activate         # or conda
pip install -r requirements.txt

# 3. Install official Diffusion Policy (Chi23) as an editable package
git clone https://github.com/real-stanford/diffusion_policy baselines_external/diffusion_policy
pip install -e baselines_external/diffusion_policy

# 4. Install this project
pip install -e .
```

Adjust the CUDA suffix in `requirements.txt` (`+cu128 / +cu121 / +cu118`) to match the server's NVIDIA driver.

## Repo layout

```
smcdp/                       core framework
  manifolds.py               EmbodimentGraphManifold, NLink planar arm, Franka7DoF
  sde.py / sampling.py / losses.py
  trajectories.py            TrajectoryScoreNetUNet (V2), traj_reverse_grw, guidance
  baselines.py               BC, DP-canonical, DP-A, DP-C (channel + guided), Projected
  franka/                    self_model, demo_gen, ground-truth compliance
  toy3/                      3-link planar arm (Stage-1 target), self-model
  experiments/               training + eval + diagnostic scripts
REPORT.md                    detailed write-up of all results (Parts I–VI)
mathematical_formulation.tex framework derivation
Idea_formulation.md          design notes / motivation
Experiment_plan.md           experiment roadmap
```

## Running experiments

Stage-1 self-models (already-trained checkpoints needed — re-train if missing):

```bash
# Franka 7-DoF self-model
python -m smcdp.experiments.franka_stage1_selfmodel
# 3-link planar (toy3.5)
python -m smcdp.experiments.toy3p5_stage1_selfmodel
```

Stage-2 score-net training (V2):

```bash
python -m smcdp.experiments.franka_traj_unet --use-v2
```

Eval / morphology transfer (no training, just inference):

```bash
# main results across all methods (in-dist + OOD z_e)
python -m smcdp.experiments.franka_v2_final_eval
python -m smcdp.experiments.franka_baselines_eval --baseline {bc, dp_official, projected}

# link-length morphology transfer (this fork's new experiment)
python -m smcdp.experiments.franka_link_morph_eval \
    --links d3,d5 \
    --deltas-mm -50 -30 -10 0 10 30 50 100 \
    --out outputs/diagnostic/link_morph_all.json \
    --md-out outputs/diagnostic/link_morph_all.md
```

Visualisations (CPU-only):

```bash
python -m smcdp.experiments.toy3p5_manifold_viz       # 3-link manifold + arm + GIFs
python -m smcdp.experiments.architecture_diagram      # network diagrams
```

## Notes

- Checkpoints (`outputs/**/*.pt`) are **not committed** (too large for GitHub). Re-run training scripts on the server, or transfer via `scp`/`rsync`.
- External clones (`baselines_external/`, `riemannian-score-sde/`) are also excluded — install separately as above.
- See `REPORT.md` for the full results, hyperparameters, and analysis.
