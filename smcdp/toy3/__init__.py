"""Toy 3 — residual self-model on a 2-link arm with synthetic compliance.

Implements Idea_formulation §14 Phase 3 (and §7):
  - True FK has analytic part + a known synthetic compliance Δ_true(q, z_e)
  - Self-model learns Δ_φ ≈ Δ_true via MSE on self-exploration data (Stage 1)
  - Score net trained on the *learned* manifold M_φ(z_e) (Stage 5)
  - z_e (tool length) varied across train/test for embodiment generalisation
"""
