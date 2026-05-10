# SMCDP — Link-Length Embodiment Generalization Experiment Plan

**Purpose**: Extend the current SE(3) Pose-SMCDP experiment beyond end-effector tool-length variation \(z_e\) to internal robot morphology variation, specifically Franka arm link-length modification.

**Base result**: Current Pose-SMCDP trains on \(z_e \sim \mathrm{Uniform}[0.05,0.15]\), evaluates ID at \(z_e=\{0.05,0.10,0.15\}\), and OOD at \(z_e=0.20\). It achieves ID pose success \(94.27\%\) and OOD pose success \(81.25\%\) under \(\text{pose\_succ@(5cm,5°)}\), with near-zero self-model manifold gap. The current report also shows that standard DP / projected baselines fail at SE(3) pose-conditioned generation, while BC matches pose accuracy but collapses multimodality.  
**Goal of this experiment**: Test whether the self-model manifold framework generalizes to a stronger embodiment shift: internal link-length variation.

---

## 1. Motivation

The current \(z_e\) experiment modifies only the end-effector tool extension. This is a relatively simple embodiment parameter because it changes the pose mostly through a terminal transform.

\[
T_{\text{tip}}(q,z_e)
=
T_{\text{ee}}(q)
T_{\text{tool}}(z_e)
\]

However, internal link-length variation modifies the kinematic chain itself:

\[
T(q;\ell)
=
\prod_i T_i(q_i;\ell_i)
\]

This changes:

- the reachable workspace,
- the end-effector Jacobian,
- IK branch geometry,
- the induced metric \(G_{\text{pose}}\),
- the mapping from joint trajectory to \(SE(3)\) pose trajectory.

Therefore, link-length variation is a stronger test of embodiment-conditioned self-model manifolds than tool-tip extension alone.

The central hypothesis is:

> Pose-SMCDP can extend from end-effector tool-length-conditioned manifolds to internal-link-conditioned self-model manifolds, provided the link-length parameter is included in the embodiment context and the self-model is trained over at least one modified embodiment.

---

## 2. Core Hypothesis

### H1 — Link-conditioned self-model manifold

A link-length-conditioned self-model

\[
T_\phi(q,z)
\in SE(3)
\]

where

\[
z = [z_{\text{tool}}, \delta \ell]
\]

can define a family of pose manifolds:

\[
\mathcal M_\phi^{pose}(z)
=
\{(q,T):T=T_\phi(q,z)\}.
\]

Here:

- \(z_{\text{tool}}\): end-effector tool extension.
- \(\delta \ell\): internal Franka link-length perturbation.

### H2 — Seen-link generalization

If the model is trained on both nominal Franka and one modified-link Franka, it should perform well on both seen morphologies:

\[
\delta \ell \in \{0, +5\%\}.
\]

### H3 — OOD link-length extrapolation

If the self-model learns a smooth link-conditioned manifold family, then it should degrade gracefully on an unseen longer-link morphology:

\[
\delta \ell = +10\%.
\]

### H4 — Advantage over baselines

Compared with BC, DP, and Projected baselines:

- BC may achieve good pose accuracy but collapse multimodal IK branches.
- DP / Projected may fail at target-aware pose generation, as in the current SE(3) pose report.
- Pose-SMCDP should preserve multimodality while maintaining high pose accuracy and exact self-model manifold adherence.

---

## 3. Embodiment Parameterization

### 3.1 Current embodiment parameter

Current Pose-SMCDP uses:

\[
z_e \in \mathbb R
\]

to encode tool extension.

### 3.2 Proposed extended embodiment vector

For link-length variation:

\[
z = [z_{\text{tool}}, \delta\ell]
\]

where:

- \(z_{\text{tool}}\): tool extension, same as current \(z_e\).
- \(\delta\ell\): normalized link-length perturbation.

Example:

\[
\delta \ell = 0
\]

means nominal Franka.

\[
\delta \ell = 0.05
\]

means a \(+5\%\) perturbation of the selected link.

\[
\delta \ell = 0.10
\]

means a \(+10\%\) perturbation, used as OOD.

---

## 4. Which Link to Modify

### 4.1 Recommended first choice

Modify one internal link in the middle or forearm section of the Franka chain.

Recommended candidates:

| Candidate | Reason |
|---|---|
| forearm link length | affects both position and orientation nontrivially |
| link before wrist | changes wrist center location and pose geometry |
| mid-chain link | stronger than tool offset, but not too destructive |

Avoid changing the base link or terminal tool link only, because those are either too global or too close to the current \(z_e\) experiment.

### 4.2 Perturbation magnitude

Use moderate perturbations:

\[
\delta \ell \in \{0\%, +5\%, +10\%\}
\]

Do not start with very large changes such as \(+20\%\), because they may alter the reachable workspace too much and confound the experiment.

---

## 5. Training and Evaluation Splits

### 5.1 Recommended split

Train on nominal and one modified morphology:

\[
\delta \ell_{\text{train}}
\in
\{0\%, +5\%\}.
\]

Evaluate on:

\[
\delta \ell_{\text{eval}}
\in
\{0\%, +5\%, +10\%\}.
\]

| Split | \(\delta \ell\) | Meaning |
|---|---:|---|
| Train / ID | \(0\%\) | nominal Franka |
| Train / ID | \(+5\%\) | seen modified embodiment |
| Eval / ID | \(0\%\) | nominal seen |
| Eval / ID | \(+5\%\) | modified seen |
| Eval / OOD | \(+10\%\) | unseen extrapolated morphology |

### 5.2 Optional interpolation split

If time permits, add:

\[
\delta \ell = +2.5\%
\]

as an interpolation test.

| Split | \(\delta \ell\) | Meaning |
|---|---:|---|
| Interpolation | \(+2.5\%\) | between seen morphologies |

### 5.3 Optional far-OOD split

Only if the \(+10\%\) case is successful:

\[
\delta \ell = +15\%
\]

as stress-test extrapolation.

---

## 6. Tool Extension \(z_{\text{tool}}\) Handling

There are two possible experimental designs.

### Design A — Isolate link-length effect

Keep tool extension fixed.

\[
z_{\text{tool}} = 0.10
\]

Then only link length varies.

This is the cleaner first experiment because it isolates internal morphology variation.

### Design B — Joint tool + link variation

Sample tool extension as in the current report:

\[
z_{\text{tool}}
\sim
\mathrm{Uniform}[0.05,0.15]
\]

and evaluate:

\[
z_{\text{tool}} \in \{0.05,0.10,0.15,0.20\}
\]

together with link-length variation.

This tests combined embodiment generalization but is more complex.

### Recommendation

Start with **Design A**.

Use:

\[
z = [0.10,\delta\ell].
\]

After validating link-length variation alone, extend to Design B if time allows.

---

## 7. Self-Model Definition

The pose self-model becomes:

\[
T_\phi(q,z)
=
T_{\text{analytic}}(q,z)
\exp_{SE(3)}(\xi_\phi(q,z)^\wedge)
\]

where:

\[
z = [z_{\text{tool}},\delta\ell].
\]

The graph manifold is:

\[
\mathcal M_\phi^{pose}(z)
=
\{(q,T):T=T_\phi(q,z)\}.
\]

The induced pose metric is:

\[
G_{\text{pose}}(q,z)
=
I
+
J_{\text{pose}}(q,z)^\top
W
J_{\text{pose}}(q,z),
\]

where:

\[
W=\operatorname{diag}(W_pI_3,W_RI_3).
\]

The pose Jacobian is:

\[
J_{\text{pose}}(q,z)
=
\frac{\partial}{\partial q'}
\bigg|_{q'=q}
\Log_{SE(3)}
\left(
T_\phi(q,z)^{-1}T_\phi(q',z)
\right).
\]

---

## 8. Dataset Generation

### 8.1 Stage-1 self-model dataset

For each morphology:

\[
\delta \ell \in \{0\%,+5\%\}
\]

sample joint configurations:

\[
q \sim \mathcal D_q
\]

inside Franka joint limits.

Generate ground-truth pose:

\[
T_{\text{true}}(q,z)
\]

from the modified Franka kinematic model.

Train residual self-model:

\[
\xi_\phi(q,z)
\]

to minimize pose error:

\[
\mathcal L_{\text{self}}
=
\|p_\phi(q,z)-p_{\text{true}}(q,z)\|^2
+
\lambda_R
\|\Log_{SO(3)}(R_\phi(q,z)^\top R_{\text{true}}(q,z))\|^2.
\]

### 8.2 Stage-2 demonstration dataset

For each morphology:

\[
\delta \ell \in \{0\%,+5\%\}
\]

generate pose-conditioned trajectories.

Condition:

\[
c=(T_{\text{start}},T_{\text{target}},z).
\]

Trajectory:

\[
\tau=(q_0,\dots,q_H).
\]

Use the same bimodal IK branch construction as the current SE(3) pose experiment.

The demo distribution should remain balanced:

\[
P(\text{elbow-up}) \approx 0.5,
\qquad
P(\text{elbow-down}) \approx 0.5.
\]

---

## 9. Models to Compare

### 9.1 Ours — Pose-SMCDP-Link

Use Method A:

- drift-free Brownian forward,
- local Varadhan DSM target,
- condition-aware terminal initialization,
- \(SE(3)\) pose self-model manifold,
- retraction via \(H_\phi^{pose}(q,z)\).

Sampling initialization:

\[
q_h^K
\sim
\mathcal N
\left(
q^{\text{init}},
\sigma_K^2G_{\text{pose}}^{-1}(q^{\text{init}},z)
\right)
\]

where:

\[
q^{\text{init}}\approx IK(T_{\text{start}},z).
\]

### 9.2 BC-link

Deterministic regressor:

\[
c\mapsto q_{0:H}.
\]

Then lift through:

\[
T_h=T_\phi(q_h,z).
\]

Expected behavior:

- good pose accuracy,
- likely mode collapse.

### 9.3 DP-channel-link

Standard Diffusion Policy baseline with channel-concat conditioning.

Condition:

\[
c=(T_{\text{start}},T_{\text{target}},z).
\]

Expected behavior based on current pose report:

- may preserve mode fraction superficially,
- may fail target-aware pose tracking.

### 9.4 Projected-link

Ambient diffusion over:

\[
(q,T)
\]

followed by projection/retraction:

\[
T\leftarrow T_\phi(q,z).
\]

Expected behavior:

- manifold gap near zero after projection,
- target accuracy may remain weak.

### 9.5 Optional DP-canonical-link

Global conditioning version of standard Diffusion Policy.

Include if time allows.

---

## 10. Primary Metrics

### 10.1 Position error

\[
e_p
=
\|p_H-p^*\|_2.
\]

Report in cm.

### 10.2 Rotation geodesic error

\[
e_R
=
\|\Log_{SO(3)}(R_H^\top R^*)\|_2.
\]

Report in degrees.

Quaternion equivalent:

\[
e_R
=
2\cos^{-1}
(|\bar q_H^\top \bar q^*|).
\]

### 10.3 Full pose success

Primary metric:

\[
\text{pose\_succ@(5cm,5°)}
=
\mathbf 1[
e_p<5\text{ cm}
\land
e_R<5^\circ
].
\]

Secondary metric:

\[
\text{pose\_succ@(5cm,10°)}.
\]

### 10.4 Self-model manifold gap

Pose manifold gap:

\[
g_\phi^{pose}(q,T,z)
=
\Log_{SE(3)}
\left(
T_\phi(q,z)^{-1}T
\right).
\]

Report:

\[
\max_h \|g_{\phi,p}(h)\|
\]

and

\[
\max_h \|g_{\phi,R}(h)\|.
\]

Expected:

- Ours: near zero by construction.
- BC lifted through \(H_\phi\): near zero.
- Projected after retraction: near zero.
- Raw ambient DP: may be nonzero if raw \(T\) is evaluated before projection.

### 10.5 Mode fidelity

Mode fraction error:

\[
\text{mode\_frac\_err}
=
|\text{frac}_{up}^{gen}-\text{frac}_{up}^{demo}|.
\]

If demo is balanced:

\[
\text{frac}_{up}^{demo}=0.5.
\]

Then:

\[
\text{mode\_frac\_err}
=
|\text{frac}_{up}^{gen}-0.5|.
\]

Add per-mode trajectory distance if possible:

\[
W_1^{q,\text{mode}}
\]

or sliced Wasserstein.

### 10.6 Joint-limit violation

\[
\text{joint\_viol}
=
\Pr[
\exists h,i:
q_{h,i}<q_{\min,i}
\lor
q_{h,i}>q_{\max,i}
].
\]

### 10.7 Smoothness

Joint velocity energy:

\[
E_{\text{vel}}
=
\frac{1}{H}
\sum_{h=0}^{H-1}
\|q_{h+1}-q_h\|^2.
\]

Joint acceleration energy:

\[
E_{\text{acc}}
=
\frac{1}{H-1}
\sum_{h=1}^{H-1}
\|q_{h+1}-2q_h+q_{h-1}\|^2.
\]

---

## 11. Main Tables

### 11.1 ID link-length result

Train:

\[
\delta\ell\in\{0,+5\%\}.
\]

Evaluate:

\[
\delta\ell\in\{0,+5\%\}.
\]

| Model | \(\delta\ell\) | pos cm ↓ | rot deg ↓ | pose@5/5° ↑ | pose@5/10° ↑ | mode frac err ↓ | manif gap ↓ | jvio ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BC-link | 0% | | | | | | | |
| DP-channel-link | 0% | | | | | | | |
| Projected-link | 0% | | | | | | | |
| Ours-link | 0% | | | | | | | |
| BC-link | +5% | | | | | | | |
| DP-channel-link | +5% | | | | | | | |
| Projected-link | +5% | | | | | | | |
| Ours-link | +5% | | | | | | | |

### 11.2 OOD link-length extrapolation result

Evaluate:

\[
\delta\ell=+10\%.
\]

| Model | pos cm ↓ | rot deg ↓ | pose@5/5° ↑ | pose@5/10° ↑ | OOD drop vs ID ↓ | mode frac err ↓ | manif gap ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| BC-link | | | | | | | |
| DP-channel-link | | | | | | | |
| Projected-link | | | | | | | |
| Ours-link | | | | | | | |

### 11.3 Main summary table

| Model | ID pose@5/5° ↑ | OOD pose@5/5° ↑ | ID pos cm ↓ | OOD pos cm ↓ | ID rot deg ↓ | OOD rot deg ↓ | mode frac err ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| BC-link | | | | | | | |
| DP-channel-link | | | | | | | |
| Projected-link | | | | | | | |
| Ours-link | | | | | | | |

---

## 12. Ablations

### 12.1 Train only nominal, evaluate modified link

This is a far-OOD stress test.

Train:

\[
\delta\ell=0\%.
\]

Evaluate:

\[
\delta\ell=+5\%,+10\%.
\]

Purpose:

- Test whether the model can extrapolate to unseen internal morphology without any link-variation training.

Expected:

- May degrade strongly.
- If successful, very strong evidence.
- If failed, not fatal.

This should be reported as stress-test, not the main result.

### 12.2 Train nominal + modified, evaluate unseen modified

This is the main experiment.

Train:

\[
\delta\ell\in\{0,+5\%\}.
\]

Evaluate:

\[
\delta\ell=+10\%.
\]

Purpose:

- Test whether the link-length-conditioned self-model manifold generalizes across morphology.

### 12.3 Remove link parameter from condition

Train with mixed embodiments but remove \(\delta\ell\) from the condition vector.

Purpose:

- Show that the model needs explicit embodiment conditioning.

Expected:

- degraded pose accuracy,
- worse OOD extrapolation,
- possible target ambiguity.

### 12.4 BC vs Ours mode preservation

Compare BC-link and Ours-link:

- endpoint pose accuracy,
- mode fraction error,
- per-mode distribution.

Expected:

- BC may have strong pose accuracy,
- BC likely collapses modes,
- Ours should preserve multimodality better.

---

## 13. Expected Outcomes

### Outcome A — Best case

Ours-link:

- ID pose success \(>90\%\),
- OOD \(+10\%\) pose success \(>75\%\),
- mode_frac_err significantly lower than BC,
- DP / Projected fail or underperform.

This would strongly support:

> SMCDP can handle internal embodiment changes beyond tool-tip offsets.

### Outcome B — Moderate success

Ours-link:

- ID high success,
- OOD moderate degradation,
- still better multimodality than BC.

This is still useful.

Claim:

> The link-conditioned manifold supports seen morphologies and shows partial OOD extrapolation, but larger internal morphology shifts remain challenging.

### Outcome C — OOD failure

Ours-link succeeds on seen \(0,+5\%\) but fails on \(+10\%\).

This is still interpretable.

Claim:

> Internal link-length variation is a harder morphology shift than tool extension. The framework supports seen link-conditioned manifolds, while stronger OOD extrapolation requires broader morphology training or improved embedding.

### Outcome D — nominal-only train fails on modified link

Not fatal.

Interpretation:

> The framework is not expected to generalize to unseen internal morphology without any embodiment-conditioned self-model data. Link-length variation must be represented in the self-model training distribution.

---

## 14. Paper Positioning

### Strong claim if successful

> Beyond end-effector tool extension, SMCDP can handle internal robot morphology variation by conditioning the self-model manifold on link-length parameters. Training on the nominal arm and one modified embodiment enables generalization to an unseen link-length extrapolation.

### Conservative claim

> We include a preliminary link-length embodiment experiment showing that the self-model manifold can be extended from terminal tool variation to internal kinematic variation. While ID performance remains high, OOD link extrapolation is harder than tool-length extrapolation.

### What not to claim

Do not claim:

- full morphology generalization from nominal Franka only,
- physical feasibility under arbitrary link changes,
- dynamics feasibility,
- collision-free guarantees,
- universal cross-embodiment transfer.

---

## 15. Recommended Execution Order

### Step 1 — Implement modified-link FK / ground truth

- Select one link.
- Add \(\delta\ell\) parameter.
- Verify FK output changes smoothly.
- Verify analytic or finite-difference Jacobian.

### Step 2 — Generate Stage-1 self-model data

For:

\[
\delta\ell\in\{0,+5\%\}
\]

sample \(q\), generate \(T_{\text{true}}(q,z)\).

### Step 3 — Train Stage-1 link-conditioned pose self-model

Train:

\[
T_\phi(q,z)
\]

where:

\[
z=[z_{\text{tool}},\delta\ell].
\]

Evaluate Stage-1 on:

\[
\delta\ell=0,+5,+10\%.
\]

Metrics:

- position error,
- rotation error.

### Step 4 — Generate Stage-2 demos

For:

\[
\delta\ell\in\{0,+5\%\}
\]

generate bimodal pose-conditioned trajectories.

### Step 5 — Train Ours-link

Use Method A.

### Step 6 — Train baselines

Minimum:

- BC-link,
- DP-channel-link,
- Projected-link.

Optional:

- DP-canonical-link.

### Step 7 — Evaluate ID and OOD

Evaluate:

\[
\delta\ell=0,+5,+10\%.
\]

Report:

- pose success,
- pos/rot error,
- mode_frac_err,
- manifold gap,
- joint violation.

### Step 8 — Add optional nominal-only stress test

Train:

\[
\delta\ell=0
\]

only.

Evaluate:

\[
\delta\ell=+5,+10\%.
\]

---

## 16. Minimal Version for Deadline

If time is short, do this:

1. Modify one link.
2. Use fixed tool length:
   \[
   z_{\text{tool}}=0.10.
   \]
3. Train:
   \[
   \delta\ell\in\{0,+5\%\}.
   \]
4. Evaluate:
   \[
   \delta\ell\in\{0,+5,+10\%\}.
   \]
5. Train only:
   - Ours-link,
   - BC-link.
6. Optional:
   - DP-channel-link.

This gives a clean additional embodiment experiment without exploding scope.

---

## 17. Final Recommended Claim

If the experiment succeeds:

> We further evaluate internal morphology variation by modifying a Franka link length. Unlike the current tool-extension experiment, this changes the kinematic chain itself. A link-conditioned Pose-SMCDP trained on the nominal arm and one modified embodiment generalizes to an unseen link-length extrapolation, while preserving self-model pose consistency and multimodal IK-branch diversity.

If it is only partially successful:

> Link-length variation is a stronger embodiment shift than end-effector tool extension. Pose-SMCDP remains stable on seen link-conditioned embodiments and shows partial OOD extrapolation, suggesting a path toward broader morphology-conditioned self-model manifolds.

---

## 18. Summary

The recommended experiment is:

\[
\boxed{
\text{Train on } \delta\ell\in\{0,+5\%\},\quad
\text{evaluate on } \delta\ell\in\{0,+5,+10\%\}.
}
\]

Use:

\[
z=[z_{\text{tool}},\delta\ell].
\]

Prefer:

\[
z_{\text{tool}}=0.10
\]

for the first version to isolate link-length effects.

The key question is not whether the model can magically transfer from nominal Franka to arbitrary morphology.  
The correct question is:

\[
\boxed{
\text{Can the self-model manifold become a morphology-conditioned manifold family beyond terminal tool offsets?}
}
\]

This experiment directly tests that question.