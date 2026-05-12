````markdown
# Experiment Plan — Sparse-Embodiment Generalization with Tool + Link-Length \(z_e\)

**Goal**: few-shot embodiment generalization 실험.  
기존 \(z_e\)가 end-effector/tool length만 의미했다면, 본 실험에서는 \(z_e\)를 **robot embodiment parameter vector**로 확장한다.

\[
z_e =
\begin{bmatrix}
z_{\text{tool}} \\
\Delta l_1 \\
\Delta l_2 \\
\cdots \\
\Delta l_m
\end{bmatrix}
\]

여기서 \(z_{\text{tool}}\)은 EE/tool extension, \(\Delta l_i\)는 selected robot link length perturbation이다.

핵심 질문:

> 제한된 수의 embodiment \(z_e\)에서만 학습한 뒤, unseen \(z_e\), 특히 link length가 변한 embodiment에서도 task success가 유지되는가?

---

## 1. Motivation

기존 report에서 v5.1은 IK-seed 없이 chart-OU SDE와 bounded chart를 사용하여 joint feasibility, manifold adherence, mode preservation을 유지하면서 높은 task success를 달성했다. 특히 chart temperature \(c=2\) 적용 후 v5.1은 DP-bounded와 feasibility-respecting success에서 동률 수준까지 회복했고, \(jvio=0\), \(mfe=0\), manifold gap \(=0\)을 유지했다. :contentReference[oaicite:0]{index=0}

그러나 기존 \(z_e\)는 주로 tool extension, 즉 end-effector length 변화에 집중되어 있었다. 다음 단계는 \(z_e\)를 **embodiment-forming parameter**로 더 강하게 확장하는 것이다.

기존 실험이 다음을 검증했다면:

\[
z_e = z_{\text{tool}}
\]

이번 실험은 다음을 검증한다.

\[
z_e = (z_{\text{tool}}, \Delta l_{\text{link}})
\]

즉, end-effector 길이뿐 아니라 robot kinematic chain 자체가 변했을 때, self-model manifold 기반 score policy가 얼마나 일반화되는지 본다.

---

## 2. Hypothesis

### H1 — Sparse embodiment training generalization

소수의 \(z_e\) 값에서만 학습해도, v5.1 self-model manifold policy는 unseen embodiment에서 task success를 유지한다.

\[
\text{succ}_{\text{unseen}} \approx \text{succ}_{\text{seen}}
\]

단, extrapolation에서는 성능 하락이 있을 수 있다.

---

### H2 — Link-length generalization is harder than tool-length generalization

tool length 변화는 end-effector frame 근처의 비교적 low-rank deformation이다.  
반면 link length 변화는 entire kinematic chain의 Jacobian과 reachable set을 바꾼다.

따라서 예상 난이도:

\[
\text{tool-only OOD} < \text{single-link OOD} < \text{multi-link OOD}
\]

---

### H3 — v5.1 should outperform IK-seeded Method A in embodiment extrapolation

v4.1 / Method A는 per-trajectory IK seed에 의존했고, report의 ABL3에서 \(q^{init}\) 제거 시 성능이 크게 붕괴했다. :contentReference[oaicite:1]{index=1}  
따라서 link length가 변하는 setting에서는 IK seed-free v5.1이 더 정합적인 comparison target이다.

---

### H4 — Bounded chart feasibility remains invariant under \(z_e\)

\(z_e\)가 link length를 바꿔도 bounded chart는 joint coordinate에 대한 diffeomorphism이므로:

\[
q=\psi_c(u)\in(q_{\min},q_{\max})
\]

는 항상 유지되어야 한다.

따라서 모든 embodiment에서:

\[
jvio = 0
\]

이어야 한다.

---

## 3. Embodiment Parameter Definition

### 3.1 Base embodiment vector

최소 실험에서는 다음 3D vector를 사용한다.

\[
z_e =
[z_{\text{tool}},\ \Delta l_3,\ \Delta l_5]
\]

- \(z_{\text{tool}}\): end-effector/tool extension
- \(\Delta l_3\): intermediate link length perturbation
- \(\Delta l_5\): distal link length perturbation

추천 범위:

\[
z_{\text{tool}} \in [0.05,\ 0.20]\ \text{m}
\]

\[
\Delta l_3 \in [-0.03,\ +0.03]\ \text{m}
\]

\[
\Delta l_5 \in [-0.03,\ +0.03]\ \text{m}
\]

처음부터 link를 너무 많이 바꾸지 않는다.  
3D \(z_e\)가 가장 적절한 첫 실험이다.

---

### 3.2 Extended embodiment vector

1차 실험 성공 후 다음으로 확장한다.

\[
z_e =
[z_{\text{tool}},\ \Delta l_2,\ \Delta l_3,\ \Delta l_5,\ \Delta l_6]
\]

이 경우는 higher-dimensional cross-embodiment generalization 실험이다.

---

## 4. Self-Model Definition

기존 pose self-model은 다음 형태다.

\[
T_\phi(q,z_e)
=
T_{\text{analytic}}(q,z_e)
\cdot
\exp_{\mathrm{SE}(3)}(\xi_\phi(q,z_e)^\wedge)
\]

이번 실험에서는 \(T_{\text{analytic}}\) 자체가 \(z_e\)의 link-length perturbation을 반영해야 한다.

즉:

\[
T_{\text{analytic}}(q,z_e)
=
FK(q;\ l_0+\Delta l(z_e),\ z_{\text{tool}})
\]

그리고 residual self-model은:

\[
\xi_\phi(q,z_e)
\]

가 link perturbation과 tool perturbation에 따른 residual compliance / model error를 학습한다.

---

## 5. Manifold Definition

bounded chart:

\[
q = \psi_c(u)
=
q_{\text{mid}}
+
\frac{q_{\text{range}}}{2}
\tanh(u/c)
\]

pose graph manifold:

\[
\mathcal M_\phi^{b\text{-pose}}(z_e)
=
\{(u,T):T=T_\phi(\psi_c(u),z_e)\}
\]

trajectory product manifold:

\[
\tau
=
(x_0,\dots,x_H)
\in
\left(\mathcal M_\phi^{b\text{-pose}}(z_e)\right)^{H+1}
\]

v5.1에서는 \(u\)-chart에서 chart-space OU score를 학습한다.

\[
du_r
=
-\frac12\beta(r)u_r\,dr
+
\sqrt{\beta(r)}\bar G_Q^{-1/2}dW_r
\]

closed-form transition:

\[
p_{r|0}(u_r|u_0)
=
\mathcal N(\alpha(r)u_0,\sigma^2(r)\bar G_Q^{-1})
\]

score target:

\[
s^{u,*}
=
-\frac{\bar G_Q(u_r-\alpha u_0)}{\sigma^2(r)}
\]

---

## 6. Training Recipe

### 6.1 Primary model

Use:

```text
v5.1 chart-OU
bounded chart
chart_temp c = 2.5
mu_pose = 0
beta_f = 20
n_sample_steps = 1000
save_every = 25k
train_steps = 200k
````

Rationale:

* 기존 report에서 (c=2)는 v5.1 성능 병목을 해결했고, (c=2.5)는 100k 기준 succ@(5cm,10°)와 strict succ@(5cm,5°)가 가장 좋았다. 
* (\mu_{\text{pose}}>0)는 exact OU score objective와 충돌하여 성능을 망쳤으므로 사용하지 않는다. 
* endpoint-relative condition은 보조 효과만 있었으므로 primary recipe에는 넣지 않는다. 필요하면 secondary ablation으로 둔다.

---

### 6.2 Primary v5.1 config

```bash
python -m smcdp.experiments.franka_traj_unet_pose \
  --use-v51 \
  --bounded-chart \
  --chart-temp 2.5 \
  --beta-f 20 \
  --mu-pose 0 \
  --steps 200000 \
  --batch 64 \
  --lr 2e-4 \
  --ema 0.999 \
  --cond-drop 0.10 \
  --save-every 25000 \
  --out-dir outputs/v51_linkze_sparse_c25_200k
```

---

### 6.3 Secondary config: endpoint-relative conditioning

Only after primary recipe.

```bash
python -m smcdp.experiments.franka_traj_unet_pose \
  --use-v51 \
  --bounded-chart \
  --chart-temp 2.5 \
  --endpoint-rel-cond \
  --beta-f 20 \
  --mu-pose 0 \
  --steps 100000 \
  --batch 64 \
  --lr 2e-4 \
  --save-every 25000 \
  --out-dir outputs/v51_linkze_sparse_c25_endpt_100k
```

Purpose:

* endpoint-relative SE(3) error input이 link-length OOD에서 더 도움이 되는지 확인.
* 기존 report에서는 c=2 위에서 endpoint conditioning 효과가 +1.2 pp에 그쳤으므로 primary lever는 아니다. 

---

## 7. Embodiment Split Design

### 7.1 Sparse training set

목표는 “few (z_e)” training이므로 uniform continuous sampling 대신 **discrete sparse embodiment set**으로 학습한다.

#### 3D (z_e = [z_{\text{tool}},\Delta l_3,\Delta l_5])

Define normalized coordinates:

[
\tilde z_i \in {-1,0,+1}
]

where:

[
z_{\text{tool}}:
-1 \mapsto 0.05,\quad
0 \mapsto 0.125,\quad
+1 \mapsto 0.20
]

[
\Delta l_3,\Delta l_5:
-1 \mapsto -0.03,\quad
0 \mapsto 0,\quad
+1 \mapsto +0.03
]

---

### 7.2 Training embodiments

Use 5 sparse training embodiments:

| ID | (z_{\text{tool}}) | (\Delta l_3) | (\Delta l_5) | Type   |
| -: | ----------------: | -----------: | -----------: | ------ |
| T0 |             0.125 |         0.00 |         0.00 | center |
| T1 |              0.05 |        -0.03 |        -0.03 | corner |
| T2 |              0.05 |        +0.03 |        +0.03 | corner |
| T3 |              0.20 |        -0.03 |        +0.03 | corner |
| T4 |              0.20 |        +0.03 |        -0.03 | corner |

This is a sparse diagonal/corner design.
It gives the model exposure to range extremes without full grid coverage.

---

### 7.3 Interpolation test embodiments

Unseen (z_e) inside the convex hull:

| ID | (z_{\text{tool}}) | (\Delta l_3) | (\Delta l_5) | Type          |
| -: | ----------------: | -----------: | -----------: | ------------- |
| I1 |             0.125 |       +0.015 |         0.00 | interpolation |
| I2 |             0.125 |         0.00 |       -0.015 | interpolation |
| I3 |            0.0875 |       -0.015 |       +0.015 | interpolation |
| I4 |            0.1625 |       +0.015 |       -0.015 | interpolation |

---

### 7.4 Extrapolation test embodiments

Outside training support:

| ID | (z_{\text{tool}}) | (\Delta l_3) | (\Delta l_5) | Type                   |
| -: | ----------------: | -----------: | -----------: | ---------------------- |
| E1 |             0.225 |         0.00 |         0.00 | tool extrapolation     |
| E2 |             0.125 |       +0.045 |         0.00 | link extrapolation     |
| E3 |             0.125 |         0.00 |       -0.045 | link extrapolation     |
| E4 |             0.225 |       +0.045 |       -0.045 | combined extrapolation |

---

### 7.5 Optional harder split: leave-one-corner-out

For stronger generalization claim:

* Train on 7 of 8 corners.
* Test on held-out corner.

This evaluates combinatorial embodiment generalization.

Example:

Train on:

[
{-1,+1}^3 \setminus {(+1,+1,+1)}
]

Test on:

[
(+1,+1,+1)
]

This is harder than interpolation but cleaner than arbitrary extrapolation.

---

## 8. Task / Demo Setting

Use the same Tier 2 boundary-active reaching setup from the report, because this is where bounded chart and feasibility matter.

Base Tier 2 setup:

```python
q_rest_A = [0.0, -0.3, 0.0, -0.40, 0.0, +3.40, 0.0]
q_rest_B = [0.0, -0.3, 0.0, -1.50, 0.0, +1.50, 0.0]
p_box = ([0.40, -0.05, 0.40], [0.50, +0.05, 0.50])
jitter_q = 0.05
target_perturb_deg = 20.0
n_ik_steps = 25
ik_alpha = 0.5
ik_alpha_null = 0.25
ik_clamp_to_limits = True
ik_clamp_margin_frac = 0.001
```

Why Tier 2:

* 기존 report에서 Tier 0 interior demos에서는 DP variants가 충분히 강했고, bounded chart의 benefit이 잘 드러나지 않았다.
* Tier 2 boundary-active demos에서는 joint feasibility와 self-model graph manifold 구조가 실제로 의미를 가졌다. 

---

## 9. Training Data Generation

### 9.1 Demo generation per embodiment

For each training embodiment (z_e^{(i)}):

1. Build analytic FK with perturbed link lengths.
2. Generate bimodal reaching demos with the same task distribution.
3. Store:

   * (q_{0:H})
   * (T_{0:H}=T_{\text{true}}(q_{0:H},z_e))
   * (T_{\text{start}})
   * (T_{\text{goal}})
   * (z_e)
   * branch label A/B only for metric diagnostics, not model input.

---

### 9.2 Dataset balance

Training set should be balanced over embodiment and mode.

Recommended:

```text
n_demo_per_embodiment = 2048
n_train_embodiments = 5
total demos = 10240
```

Batch sampling:

```text
sample embodiment uniformly
sample demo uniformly within embodiment
```

This prevents center embodiment from dominating.

---

### 9.3 Stage-1 self-model training

Stage-1 must be trained over the same or wider (z_e) support than Stage-2.

Recommended Stage-1 support:

[
z_{\text{tool}}\in[0.05,0.225]
]

[
\Delta l_3,\Delta l_5\in[-0.045,+0.045]
]

Reason:

* Stage-2 tests extrapolation.
* If Stage-1 self-model is not valid there, policy failure cannot be attributed cleanly to score generalization.

Stage-1 evaluation metrics:

| Metric              | Meaning                               |
| ------------------- | ------------------------------------- |
| pos residual error  | ( |p_{\phi}-p_{\text{true}}| )        |
| rot residual error  | ( d_{SO(3)}(R_\phi,R_{\text{true}}) ) |
| Jacobian error      | ( |J_\phi-J_{\text{true}}|_F )        |
| extrapolation error | error on E1–E4                        |

Stage-1 acceptance threshold:

```text
pos_err_mean < 5 mm on ID/interpolation
pos_err_mean < 10 mm on extrapolation
rot_err_mean < 1.0° on ID/interpolation
rot_err_mean < 2.0° on extrapolation
Jacobian relative error < 10–15%
```

If Stage-1 fails, Stage-2 results are not interpretable.

---

## 10. Baselines

### 10.1 Required baselines

| Method           | Purpose                                       |
| ---------------- | --------------------------------------------- |
| v5.1 c=2.5       | main method                                   |
| DP-bounded c=2.5 | strongest feasibility-respecting baseline     |
| DP-raw           | shows unsafe high-accuracy baseline           |
| BC               | deterministic collapse / lower-bound baseline |

---

### 10.2 Optional baselines

| Method                         | Purpose                                                         |
| ------------------------------ | --------------------------------------------------------------- |
| v5.1 c=2.5 + endpoint_rel_cond | test whether explicit self-model feedback helps link-length OOD |
| Projected DP                   | ambient pose DP + self-model projection                         |
| v4.1 Method A                  | shows IK-seed dependence, but less central                      |

---

### 10.3 Fairness rule

All bounded methods must use the same chart temperature:

[
c=2.5
]

DP-bounded must be compared with the same (c), otherwise chart conditioning is confounded.

---

## 11. Metrics

### 11.1 Primary task metrics

For each embodiment (z_e):

#### Position error

[
e_p
===

|p_{\text{exec}}-p_{\text{goal}}|_2
]

where:

[
T_{\text{exec}}=T_\phi(q_H,z_e)
]

#### Rotation error

[
e_R
===

|\Log_{SO(3)}(R_{\text{exec}}^{-1}R_{\text{goal}})|
]

in degrees.

#### Pose success

Strict:

[
\text{succ}_{5,5}
=================

\mathbb E[
\mathbf 1(e_p<5\text{cm})
\land
\mathbf 1(e_R<5^\circ)
]
]

Relaxed:

[
\text{succ}_{5,10}
==================

\mathbb E[
\mathbf 1(e_p<5\text{cm})
\land
\mathbf 1(e_R<10^\circ)
]
]

Primary success metric:

[
\boxed{\text{succ}_{5,10}}
]

Strict precision metric:

[
\boxed{\text{succ}_{5,5}}
]

---

### 11.2 Feasibility metrics

#### Joint violation

[
jvio
====

\mathbb E[
\mathbf 1(\exists h,i:\ q_{h,i}\notin[q_{\min,i},q_{\max,i}])
]
]

#### Sample-wise effective success

Do not use product approximation as final metric.

[
\boxed{
\text{eff}_{5,10}
=================

\mathbb E[
\mathbf 1(e_p<5\text{cm})
\land
\mathbf 1(e_R<10^\circ)
\land
\mathbf 1(q\in Q)
]
}
]

Likewise:

[
\text{eff}_{5,5}
================

\mathbb E[
\mathbf 1(e_p<5\text{cm})
\land
\mathbf 1(e_R<5^\circ)
\land
\mathbf 1(q\in Q)
]
]

---

### 11.3 Manifold adherence metrics

For methods that output or imply a pose (T):

[
g_T
===

\Log_{SE(3)}
\left(
T_\phi(q,z_e)^{-1}T_{\text{raw}}
\right)
]

Report:

[
g_p=|g_T^{trans}|
]

[
g_R=|g_T^{rot}|
]

For v5.1:

[
g_p\approx 0,\quad g_R\approx 0
]

by construction.

For DP-bounded:

* If pose is evaluated through (T_\phi(q,z_e)), manifold gap is not meaningful unless raw (T) exists.
* If raw (T) exists, report raw-vs-exec gap separately.

---

### 11.4 Mode preservation metrics

Use branch labels only for metrics.

Detect mode-discriminating joint automatically:

[
j^*
===

\arg\max_j |q_{\text{rest},A,j}-q_{\text{rest},B,j}|
]

threshold:

[
\theta
======

\frac{q_{\text{rest},A,j^*}+q_{\text{rest},B,j^*}}{2}
]

Metrics:

| Metric              | Definition                                      |                     |   |
| ------------------- | ----------------------------------------------- | ------------------- | - |
| mode fraction error | (                                               | \hat p_A-p_A^{demo} | ) |
| mode flip rate      | (P(\text{mode}*{init}\neq \text{mode}*{final})) |                     |   |
| per-mode success    | succ for A and B separately                     |                     |   |

Report:

```text
mfe
flip_rate
succ_A
succ_B
pos_A / pos_B
rot_A / rot_B
```

---

### 11.5 Embodiment generalization metrics

Group evaluation by split:

| Split           | Meaning                          |
| --------------- | -------------------------------- |
| Seen            | training embodiments T0–T4       |
| Interp          | unseen but inside train hull     |
| Extrap-tool     | tool length outside train        |
| Extrap-link     | link length outside train        |
| Extrap-combined | both tool and link outside train |

Compute:

[
\Delta_{\text{interp}}
======================

## \text{succ}_{seen}

\text{succ}_{interp}
]

[
\Delta_{\text{extrap}}
======================

## \text{succ}_{seen}

\text{succ}_{extrap}
]

Also report normalized robustness:

[
R_{\text{interp}}
=================

\frac{\text{succ}*{interp}}{\text{succ}*{seen}}
]

[
R_{\text{extrap}}
=================

\frac{\text{succ}*{extrap}}{\text{succ}*{seen}}
]

---

### 11.6 Link-length sensitivity slope

Fit linear regression:

[
e_p =
a_0
+
a_1|\Delta l_3|
+
a_2|\Delta l_5|
+
a_3z_{\text{tool}}
+
\epsilon
]

or for success:

[
\text{logit}(\text{succ})
=========================

b_0
+
b_1|\Delta l_3|
+
b_2|\Delta l_5|
+
b_3z_{\text{tool}}
]

This identifies which embodiment parameter hurts most.

---

### 11.7 Chart diagnostics

Because chart temperature was critical in the report, log:

[
|u|_\infty
]

[
|u/c|_\infty
]

[
|\tanh(u/c)|_{\max}
]

[
D_{\psi,c}\ \text{p1/p5}
]

[
\text{joint margin p1}
]

Report for demo and generated samples separately.

---

### 11.8 Smoothness metrics

Use both (q)-space and pose-space smoothness.

[
E_{\text{vel}}^q
================

\sum_h |q_{h+1}-q_h|^2
]

[
E_{\text{acc}}^q
================

\sum_h |q_{h+1}-2q_h+q_{h-1}|^2
]

Pose smoothness:

[
E_{\text{vel}}^T
================

\sum_h
|\Log(T_h^{-1}T_{h+1})|^2
]

---

## 12. Evaluation Protocol

### 12.1 Per-embodiment sample count

Use:

```text
n_eval_per_z = 128
```

if compute allows. Minimum:

```text
n_eval_per_z = 64
```

Report confidence interval:

[
SE=\sqrt{\frac{p(1-p)}{N}}
]

---

### 12.2 Checkpoints

Evaluate all saved checkpoints:

```text
25k, 50k, 75k, 100k, 125k, 150k, 175k, 200k
```

Select:

* best checkpoint by seen validation eff@5,10
* final 200k
* report both if they differ by >2 pp

Reason: existing report showed v5.1 can peak before final in some settings. 

---

### 12.3 Main table format

| Method | Split | pos cm | rot° | succ@5,5 | succ@5,10 | eff@5,10 | jvio | mfe | gap | (R) |
| ------ | ----- | -----: | ---: | -------: | --------: | -------: | ---: | --: | --: | --: |

Where (R) is robustness ratio:

[
R=\frac{\text{succ}*{split}}{\text{succ}*{seen}}
]

---

## 13. Expected Results

### Expected if v5.1 generalizes well

| Split                  | Expected succ@5,10 |
| ---------------------- | -----------------: |
| Seen                   |             90–95% |
| Interpolation          |             85–95% |
| Tool extrapolation     |             75–90% |
| Link extrapolation     |             65–85% |
| Combined extrapolation |             50–75% |

---

### Failure modes

#### Failure A — Stage-1 self-model fails

Symptoms:

```text
manifold gap remains zero, but exec pose error large
Jacobian error large
all methods degrade similarly
```

Interpretation:

Self-model is not accurate for link-length extrapolation.

---

#### Failure B — v5.1 score fails but self-model is accurate

Symptoms:

```text
Stage-1 error small
DP-bounded succeeds
v5.1 fails
```

Interpretation:

Chart-OU score model did not learn sparse (z_e) conditioning well.

Possible fix:

* richer (z_e) embedding
* endpoint-relative conditioning
* FiLM conditioning
* more training embodiments

---

#### Failure C — all bounded methods fail in link extrapolation

Symptoms:

```text
v5.1, DP-bounded, BC all fail on E2/E3/E4
```

Interpretation:

Task reachability changed too much, or train support too sparse.

Possible fix:

* reduce extrapolation range
* add one or two link-extreme training embodiments
* separate interpolation and extrapolation claims

---

## 14. Ablation Plan

### A1 — Number of training embodiments

Train with:

```text
N_z = 3, 5, 9
```

Example:

| (N_z) | Description              |
| ----: | ------------------------ |
|     3 | center + two corners     |
|     5 | recommended sparse set   |
|     9 | center + 8 corners / LHS |

Metric:

[
\text{succ}*{interp},\quad \text{succ}*{extrap}
]

Goal:

Measure scaling of embodiment coverage.

---

### A2 — Tool-only vs link-only vs combined

Train/evaluate separately:

| Setting   | (z_e)                                     |
| --------- | ----------------------------------------- |
| Tool-only | (z_{\text{tool}})                         |
| Link-only | ((\Delta l_3,\Delta l_5))                 |
| Combined  | ((z_{\text{tool}},\Delta l_3,\Delta l_5)) |

This isolates whether degradation comes from link changes.

---

### A3 — Conditioning architecture

Compare:

| Config                 | Description               |
| ---------------------- | ------------------------- |
| concat (z_e)           | current                   |
| Fourier features (z_e) | better extrapolation?     |
| FiLM (z_e)             | scale/shift UNet features |
| endpoint_rel_cond      | self-model feedback       |

Run only if primary model struggles.

---

### A4 — chart_temp

Because current report shows chart_temp is dominant, test:

[
c\in{2.0,2.5,3.0}
]

Use 100k first.
Proceed to 200k only for best (c).

---

### A5 — DP-bounded fair chart_temp

For every (c) used by v5.1, run DP-bounded with the same (c).

This prevents unfair comparison.

---

## 15. Paper-Ready Claims This Experiment Can Support

### Strong claim if successful

> A v5.1 self-model chart-OU policy trained on only a sparse set of embodiments generalizes to unseen tool and link-length perturbations while preserving joint feasibility and self-model consistency by construction.

---

### More cautious claim

> The bounded self-model graph representation enables structured interpolation across embodiment parameters. Link-length extrapolation is harder than tool-only extrapolation, but v5.1 retains nontrivial task success under sparse embodiment training.

---

### Avoid overclaiming

Do not claim:

```text
cross-embodiment generalization is guaranteed
```

unless all reachable-set/topology assumptions are formally proven.

Use:

```text
structurally encouraged
```

or:

```text
empirically observed under topologically compatible embodiment perturbations
```

---

## 16. Immediate TODO

### Code

* [ ] Extend (z_e) from scalar to vector.
* [ ] Modify analytic FK to accept link-length perturbations.
* [ ] Modify Stage-1 self-model input dimension.
* [ ] Modify demo generator to sample from discrete embodiment set.
* [ ] Save (z_e) vector in dataset.
* [ ] Ensure v5.1 score net accepts vector (z_e).
* [ ] Ensure eval scripts group by embodiment split.
* [ ] Add link-length-specific metrics and tables.
* [ ] Add chart diagnostics with (u/c), not raw (u) only.

---

### Experiments

* [ ] Stage-1 self-model over extended (z_e) support.
* [ ] Verify Stage-1 pose/Jacobian accuracy on seen/interp/extrap embodiments.
* [ ] Train v5.1 c=2.5, 200k, (N_z=5).
* [ ] Train DP-bounded c=2.5, 200k, (N_z=5).
* [ ] Evaluate seen/interp/extrap groups.
* [ ] Run (N_z=3) and (N_z=9) scaling if primary result is promising.
* [ ] Run endpoint_rel_cond only if link extrapolation is weak.

---

## 17. Final Recommended First Run

### First run

```text
Embodiment vector:
z_e = [z_tool, Δl3, Δl5]

Train embodiments:
T0–T4, N_z=5

Model:
v5.1 chart-OU
chart_temp = 2.5
mu_pose = 0
beta_f = 20
steps = 200k
save_every = 25k

Eval:
Seen T0–T4
Interpolation I1–I4
Extrapolation E1–E4
n_eval_per_z = 64 or 128
n_sample_steps = 1000
```

### Primary report table

```text
v5.1 c=2.5 vs DP-bounded c=2.5
grouped by seen / interpolation / tool extrapolation / link extrapolation / combined extrapolation
```

### Decision after first run

| Outcome                                        | Next                                            |
| ---------------------------------------------- | ----------------------------------------------- |
| v5.1 ≈ DP-bounded, better feasibility/manifold | proceed to 3-seed                               |
| v5.1 worse only in extrapolation               | add endpoint_rel_cond / FiLM (z_e)              |
| both fail extrapolation                        | reduce extrap range or add training embodiments |
| Stage-1 error high                             | improve self-model before policy training       |

```
```
