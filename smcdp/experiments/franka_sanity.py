"""Franka7DoF manifold sanity invariants (Idea_formulation §3, §4, §5).

Verifies six invariants that any GraphManifold-based self-model must satisfy:

  (S1)  Closed-form J_F  ==  autograd J_F                          (jacobian_F correct)
  (S2)  retraction stays on M:  ‖p − F(q+δq, z_e)‖ = 0              (graph H is exact)
  (S3)  J_g · v == 0  for v = lift_chart_to_tangent(x, a)           (auto-tangent property)
  (S4)  ‖v‖²_ambient  ==  a^T G(q, z_e) a   for a ∈ R^7              (norm equivalence)
  (S5)  empirical Cov of N(0, G^{-1}) tangent samples ≈ G^{-1}       (chart cov)
  (S6)  exp_x(log_x(y)) ≈ y for nearby (q, z) pairs on M             (log/exp roundtrip)

Run: python -m smcdp.experiments.franka_sanity
"""
from __future__ import annotations

import torch
import pybullet_data

from smcdp.manifolds import Franka7DoF


URDF = f"{pybullet_data.getDataPath()}/franka_panda/panda.urdf"


def make_manifold():
    return Franka7DoF(urdf_path=URDF, end_link="panda_hand", tool_z_max=0.20)


def s1_jacobian_closed_form_vs_autograd():
    M = make_manifold()
    torch.manual_seed(0)
    B = 4
    x = M.random_uniform(B, dtype=torch.float64)
    q, _, z = M.split_x(x)
    Jf_closed = M.jacobian_F(q, z)                                # (B, 3, 7)
    # autograd reference: jacobian over q for each sample (loop, no vmap on pytorch_kinematics)
    Jf_auto = torch.zeros_like(Jf_closed)
    for i in range(B):
        qi = q[i:i+1].clone().requires_grad_(True)
        zi = z[i:i+1]
        p = M.F(qi, zi)                                           # (1, 3)
        for d in range(3):
            grad = torch.autograd.grad(p[0, d], qi, retain_graph=(d < 2))[0]
            Jf_auto[i, d, :] = grad[0]
    err = (Jf_closed - Jf_auto).abs().max().item()
    print(f"  [S1] max |J_F_closed − J_F_autograd|  = {err:.3e}")
    return err < 1e-6


def s2_retraction_stays_on_M():
    M = make_manifold()
    torch.manual_seed(1)
    B = 16
    x = M.random_uniform(B, dtype=torch.float64)
    # take a random tangent step in chart coords
    a = 0.05 * torch.randn(B, 7, dtype=torch.float64)
    v = M.lift_chart_to_tangent(x, a)
    y = M.exp(x, v)
    g = M.constraint(y)                                           # p_y − F(q_y, z_y) should be 0
    err = g.abs().max().item()
    print(f"  [S2] max ‖constraint after retraction‖  = {err:.3e}")
    return err < 1e-7


def s3_lifted_vector_is_tangent():
    """J_g(x) · v = 0  where  J_g = [-J_F, I_p, 0_z]  (Idea §4.4).

    For v = lift(a) = (a, J_F a, 0), J_g · v = −J_F a + J_F a + 0 = 0 by construction.
    This must hold to machine precision for ANY a — it's not a learned property.
    """
    M = make_manifold()
    torch.manual_seed(2)
    B = 4
    x = M.random_uniform(B, dtype=torch.float64)
    q, _, z = M.split_x(x)
    a = torch.randn(B, 7, dtype=torch.float64)
    v = M.lift_chart_to_tangent(x, a)                             # (B, 11)
    Jf = M.jacobian_F(q, z)                                       # (B, 3, 7)
    v_q = v[..., :7]
    v_p = v[..., 7:10]
    v_z = v[..., 10:11]
    Jg_v = -(Jf @ v_q.unsqueeze(-1)).squeeze(-1) + v_p            # n_p block
    err = max(Jg_v.abs().max().item(), v_z.abs().max().item())
    print(f"  [S3] max |J_g · v|  = {err:.3e}")
    return err < 1e-10


def s4_ambient_norm_eq_chart_G_norm():
    M = make_manifold()
    torch.manual_seed(3)
    B = 8
    x = M.random_uniform(B, dtype=torch.float64)
    q, _, z = M.split_x(x)
    a = torch.randn(B, 7, dtype=torch.float64)
    v = M.lift_chart_to_tangent(x, a)
    G = M.G(q, z)                                                 # (B, 7, 7)
    chart_G = (a.unsqueeze(-2) @ G @ a.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    ambient = (v * v).sum(-1)
    err = (ambient - chart_G).abs().max().item()
    print(f"  [S4] max |‖v‖²_ambient − a^T G a|  = {err:.3e}")
    return err < 1e-10


def s5_tangent_gaussian_covariance():
    M = make_manifold()
    torch.manual_seed(4)
    # Single anchor x; sample many tangent vectors at it and compare empirical
    # chart-cov of the q-block to G^{-1}.
    x_anchor = M.random_uniform(1, dtype=torch.float64)
    q_a, _, z_a = M.split_x(x_anchor)
    Ginv_target = torch.linalg.inv(M.G(q_a, z_a))[0]               # (7, 7)
    N = 80_000
    x_rep = x_anchor.expand(N, -1).contiguous()
    v = M.random_normal_tangent(x_rep)                             # (N, 11)
    a_block = v[:, :7]                                             # chart-coord component
    cov_emp = (a_block.t() @ a_block) / N
    err = (cov_emp - Ginv_target).abs().max().item()
    rel = err / Ginv_target.abs().max().item()
    print(f"  [S5] empirical Cov error vs G^-1  abs={err:.3e}  rel={rel:.3e}")
    return rel < 0.05


def s6_log_exp_roundtrip():
    M = make_manifold()
    torch.manual_seed(5)
    B = 16
    x = M.random_uniform(B, dtype=torch.float64)
    # Generate y near x by a small chart step (same z_e to stay on the same M_φ(z_e)).
    q_x, _, z_x = M.split_x(x)
    dq = 0.02 * torch.randn(B, 7, dtype=torch.float64)
    y = M.make_x(q_x + dq, z_x)
    v = M.log(x, y)
    y_round = M.exp(x, v)
    err = (y - y_round).abs().max().item()
    print(f"  [S6] max ‖y − exp(x, log(x,y))‖  = {err:.3e}")
    return err < 1e-9


def main():
    print("Franka7DoF sanity invariants")
    print(f"  URDF: {URDF}")
    print()
    results = {
        "S1": s1_jacobian_closed_form_vs_autograd(),
        "S2": s2_retraction_stays_on_M(),
        "S3": s3_lifted_vector_is_tangent(),
        "S4": s4_ambient_norm_eq_chart_G_norm(),
        "S5": s5_tangent_gaussian_covariance(),
        "S6": s6_log_exp_roundtrip(),
    }
    print()
    for k, v in results.items():
        print(f"  {k}  {'PASS' if v else 'FAIL'}")
    if not all(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
