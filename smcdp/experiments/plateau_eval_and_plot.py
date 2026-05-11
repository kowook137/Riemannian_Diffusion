"""Plateau analysis: eval every intermediate ckpt, plot succ-vs-step, detect overfit.

Workflow per method:
  1. Discover  outputs/<run>/step_*/{ours_v2_pose.pt | ckpt.pt}  +  the final ckpt
  2. Run the appropriate eval driver on each (skip if eval_metrics(_full).json exists)
  3. Aggregate (step, pos_err_mean_cm, rot_err_mean_deg, pose_succ_5cm_5deg, ...)
  4. Plot curves; report per-method plateau step + overfit verdict
  5. Final plateau-vs-plateau table

Eval drivers used:
  v5.1   ->  python -m smcdp.experiments.franka_pose_reeval --ckpt <p>
                  --out-name eval_metrics_nsteps_1000.json --n-sample-steps 1000
  DP-*   ->  python -m smcdp.experiments.franka_baselines_pose_eval --ckpt <p>
                  --success-pos 0.05 --success-rot 0.0873
             (writes eval_metrics.json next to ckpt)

Run:
    python -m smcdp.experiments.plateau_eval_and_plot \
        --v51-run outputs/v51_tier2_200k_plateau \
        --dp-raw-run outputs/tier2_dp_raw_200k_plateau \
        --dp-bounded-run outputs/tier2_dp_bounded_200k_plateau \
        --out-dir outputs/plateau_200k_comparison
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_KEYS = ["pos_err_mean_cm", "rot_err_mean_deg",
               "pose_succ_5cm_5deg", "pose_succ_5cm_10deg",
               "mode_frac_err", "joint_viol_rate"]


def _discover_ckpts(run_dir: Path, ckpt_name: str, final_name: str):
    """Return [(step, ckpt_path, eval_dir)] sorted by step.

    Intermediate: <run>/step_NNNNNN/<ckpt_name>
    Final:        <run>/<final_name> -- step inferred from ckpt['args']['steps'].
    """
    out = []
    for sub in sorted(run_dir.glob("step_*")):
        m = re.match(r"step_(\d+)", sub.name)
        if not m:
            continue
        p = sub / ckpt_name
        if p.is_file():
            out.append((int(m.group(1)), p, sub))
    final = run_dir / final_name
    if final.is_file():
        import torch
        ck = torch.load(final, map_location="cpu", weights_only=False)
        step = int(ck["args"]["steps"])
        out.append((step, final, run_dir))
    return sorted(out, key=lambda r: r[0])


def _run_eval_v51(ckpt: Path, out_name: str, force: bool):
    eval_path = ckpt.parent / out_name
    if eval_path.is_file() and not force:
        return eval_path
    cmd = [sys.executable, "-m", "smcdp.experiments.franka_pose_reeval",
           "--ckpt", str(ckpt),
           "--out-name", out_name,
           "--n-sample-steps", "1000"]
    print(f"  $ {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise RuntimeError(f"v5.1 reeval failed (rc={rc}) on {ckpt}")
    return eval_path


def _run_eval_dp(ckpt: Path, force: bool):
    eval_path = ckpt.parent / "eval_metrics.json"
    if eval_path.is_file() and not force:
        return eval_path
    cmd = [sys.executable, "-m", "smcdp.experiments.franka_baselines_pose_eval",
           "--ckpt", str(ckpt),
           "--success-pos", "0.05",
           "--success-rot", "0.0873"]
    print(f"  $ {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise RuntimeError(f"DP eval failed (rc={rc}) on {ckpt}")
    return eval_path


def _aggregate(json_path: Path):
    d = json.load(open(json_path))
    rows = d.get("per_z", [])
    if not rows:
        return {k: float("nan") for k in METRIC_KEYS}
    return {k: sum(r.get(k, 0.0) for r in rows) / len(rows) for k in METRIC_KEYS}


def _eval_method(name: str, run_dir: Path, ckpt_name: str, final_name: str,
                 driver: str, force: bool):
    print(f"\n[{name}] run_dir={run_dir}")
    ckpts = _discover_ckpts(run_dir, ckpt_name, final_name)
    if not ckpts:
        print(f"  (no ckpts found in {run_dir})")
        return []
    rows = []
    for step, ckpt, parent in ckpts:
        print(f"  step={step}  ckpt={ckpt}")
        if driver == "v51":
            jp = _run_eval_v51(ckpt, "eval_metrics_nsteps_1000.json", force)
        else:
            jp = _run_eval_dp(ckpt, force)
        m = _aggregate(jp)
        rows.append({"step": step, **m, "_json": str(jp)})
    return rows


def _plot(curves, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [("pose_succ_5cm_5deg", "succ@(5cm, 5°)"),
              ("pose_succ_5cm_10deg", "succ@(5cm, 10°)"),
              ("pos_err_mean_cm", "pos err mean [cm]"),
              ("rot_err_mean_deg", "rot err mean [°]")]
    for ax, (key, title) in zip(axes.flat, panels):
        for name, rows in curves.items():
            if not rows:
                continue
            xs = [r["step"] for r in rows]
            ys = [r[key] * (100.0 if "succ" in key else 1.0) for r in rows]
            ax.plot(xs, ys, marker="o", label=name)
        ax.set_xlabel("training step")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved {out_path}")


def _diagnose(rows, key="pose_succ_5cm_10deg"):
    """Return (plateau_step, best_val, overfit) where overfit=True if best step
    is strictly before the final step AND final < best - 0.02 (2 pp)."""
    if not rows:
        return None, None, False
    vals = [(r["step"], r[key]) for r in rows]
    best_step, best_val = max(vals, key=lambda x: x[1])
    final_step, final_val = vals[-1]
    overfit = (best_step < final_step) and (best_val - final_val > 0.02)
    return best_step, best_val, overfit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v51-run", type=str, required=True)
    p.add_argument("--dp-raw-run", type=str, required=True)
    p.add_argument("--dp-bounded-run", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="outputs/plateau_comparison")
    p.add_argument("--force", action="store_true",
                   help="Re-run eval even if metric json already exists.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = {
        "v5.1":       _eval_method("v5.1", Path(args.v51_run),
                                   "ours_v2_pose.pt", "ours_v2_pose.pt",
                                   "v51", args.force),
        "DP-raw":     _eval_method("DP-raw", Path(args.dp_raw_run),
                                   "ckpt.pt", "ckpt.pt",
                                   "dp", args.force),
        "DP-bounded": _eval_method("DP-bounded", Path(args.dp_bounded_run),
                                   "ckpt.pt", "ckpt.pt",
                                   "dp", args.force),
    }

    with open(out_dir / "plateau_curves.json", "w") as f:
        json.dump(curves, f, indent=2)
    _plot(curves, out_dir / "plateau_curves.png")

    print("\n" + "=" * 72)
    print("PLATEAU DIAGNOSIS (succ@5cm,10°)")
    print("=" * 72)
    summary = {}
    for name, rows in curves.items():
        bs, bv, of = _diagnose(rows, "pose_succ_5cm_10deg")
        if bs is None:
            print(f"  {name:<12} : (no data)")
            continue
        verdict = "OVERFIT" if of else "OK"
        final = rows[-1]["pose_succ_5cm_10deg"]
        print(f"  {name:<12} : plateau@step {bs:>6d}  best={100*bv:5.1f}%  "
              f"final({rows[-1]['step']})={100*final:5.1f}%  [{verdict}]")
        summary[name] = {"plateau_step": bs, "best_succ510": bv,
                         "final_step": rows[-1]["step"],
                         "final_succ510": final, "overfit": of}
    with open(out_dir / "plateau_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print("FINAL PLATEAU-vs-PLATEAU TABLE")
    print("=" * 72)
    hdr = f"{'method':<12} | {'step':>6} | {'pos cm':>7} | {'rot°':>6} | " \
          f"{'succ55':>7} | {'succ510':>8} | {'mfe':>5} | {'jvio':>5}"
    print(hdr)
    print("-" * len(hdr))
    for name, rows in curves.items():
        if not rows:
            continue
        bs, _, _ = _diagnose(rows, "pose_succ_5cm_10deg")
        r = next(r for r in rows if r["step"] == bs)
        print(f"{name:<12} | {r['step']:>6d} | "
              f"{r['pos_err_mean_cm']:>7.2f} | "
              f"{r['rot_err_mean_deg']:>6.2f} | "
              f"{100*r['pose_succ_5cm_5deg']:>6.1f}% | "
              f"{100*r['pose_succ_5cm_10deg']:>7.1f}% | "
              f"{r['mode_frac_err']:>5.2f} | "
              f"{100*r['joint_viol_rate']:>4.1f}%")


if __name__ == "__main__":
    main()
