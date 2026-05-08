"""Evaluation metrics for Toy 1.

  - manifold_adherence: distribution of |g_φ(x)| over generated samples
  - circular_wasserstein1: 1-Wasserstein on S^1 (sliced 1D distance after
    canonicalising both empirical CDFs to [0, 2π))
"""
from __future__ import annotations

import torch
from torch import Tensor

from smcdp.manifolds import Manifold, Sphere1


def manifold_adherence(manifold: Manifold, x: Tensor) -> dict[str, float]:
    """Returns mean / max / std of |g_φ(x)| over a batch of generated samples."""
    g = manifold.constraint(x).abs().squeeze(-1)
    return {
        "mean": g.mean().item(),
        "max": g.max().item(),
        "std": g.std().item(),
        "frac_in_M_1e-4": (g < 1e-4).float().mean().item(),
        "frac_in_M_1e-2": (g < 1e-2).float().mean().item(),
    }


def circular_wasserstein1(theta_a: Tensor, theta_b: Tensor) -> float:
    """1-Wasserstein distance between two empirical distributions on S^1.

    Each θ ∈ [-π, π).  W_1 on S^1 has the closed form
        W_1(μ, ν) = ∫_0^{2π} | F_μ(θ) − F_ν(θ) − c* | dθ
    where c* is the median of F_μ − F_ν.  Here we approximate via the
    sorted-CDF + best-rotation formulation (see Rabin et al., 2011).
    """
    a = theta_a.detach().cpu().sort().values
    b = theta_b.detach().cpu().sort().values
    assert a.shape == b.shape, "sample sizes must match"
    # cyclic shift that minimises L1 of (a − b) is the median of (a − b)
    diff = a - b
    c = diff.median()
    return (diff - c).abs().mean().item()


def s1_circular_w1(x_a: Tensor, x_b: Tensor) -> float:
    """Convenience wrapper: takes (B, 2) Cartesian samples on S^1."""
    sphere = Sphere1()
    theta_a = sphere.to_angle(x_a)
    theta_b = sphere.to_angle(x_b)
    return circular_wasserstein1(theta_a, theta_b)
