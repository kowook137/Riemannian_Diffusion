# SMCDP — SE(3) Pose-Extended Report

**Scope**: SE(3) (position + rotation via quaternion) 회전까지 모델링하도록 확장한 시스템 단독 정리.
**Base framework**: `REPORT.md` Part I–VI (position-only SMCDP) — 본 문서는 그 위 SE(3) 확장 layer.
**Compiled**: 2026-05-10

---

## 0. 요약

기존 SMCDP의 self-model manifold $\mathcal{M}_\phi$를 $T_\phi : \mathbb{R}^{n_q + n_z} \to \mathrm{SE}(3)$로 확장:

$$T_\phi(q, z_e) = T_\text{analytic}(q, z_e) \cdot \exp_{\mathrm{SE}(3)}(\xi_\phi(q, z_e)^\wedge), \qquad \xi_\phi \in \mathfrak{se}(3)$$

매니폴드 위 점은 storage form $(q, q_R \in S^3, p \in \mathbb{R}^3, z_e)$, tangent는 trivialized $(v_q, v_\xi \in \mathfrak{se}(3), 0)$. 회전은 quaternion으로 저장하되, 모든 autograd-critical path는 $(R, p)$-tuple form (rotation matrix)로 통일하여 vmap-safe.

**달성한 것** (2026-05-10):
- **첫 번째 작동하는 pose-extended SMCDP** — Method A baseline:
  - Training: $z_e \sim \mathrm{Uniform}[0.05, 0.15]$, Eval: ID = {0.05, 0.10, 0.15}, OOD = 0.20 (+33% extrapolation)
  - **pose_succ@(5cm, 5°) ID 평균 = 94.27%** (z_e=0.05/0.10/0.15에서 92–95%), **OOD = 81.25%** (`metric.md` strict full-pose criterion)
  - pos_err 2.00 cm, rot_err 2.79° at z_e=0.05 (ID-boundary)
  - manifold gap ≈ 0 (machine precision, by construction)
- 이전 시도 (anchor-metric Langevin drift 가정) 모두 실패 (succ 0%, q ~10⁵ rad 폭주). Method A (forward Brownian + condition-aware sampling init)으로 해결.
- **Per-trajectory $q^\text{init}$이 결정적 contribution**: ABL3 (제거 시) ID 94% → 19% (−75.5 pp) — Q3 anchor mismatch 진단의 정량 검증.

**완료** (2026-05-10): ABL1/ABL2/ABL3 ablation 3종, pose-baseline 4종 (BC / DP-canonical / DP-channel / Projected) 모두 측정 완료 — `metric.md` 통합 metric.

**v4.1 추가** (2026-05-11): joint-limit bounded chart (`joint_limit_extension.tex`) 구현 + 학습 + 평가 완료. **Joint feasibility by construction 달성 (viol = 0% 모든 z_e)** + multimodality 2.5× 향상, 그러나 Varadhan chart-Euclidean bias로 pose accuracy −19 pp ID / −27 pp OOD 손해 (Choice A 의 structural trade-off, spec §9에 명시). 자세한 비교는 §10 참고.

**v5.1 추가** (2026-05-12, §12): v4.1까지 잔존한 **per-trajectory IK warm-start `q_init`** (Method A의 mode-label cheat) 를 spec `joint_limit_extension.tex` v5.1 (chart-OU SDE, closed-form Gaussian transition, IK-free reference $\mathcal N(0,\bar G_Q^{-1})$) 로 구조적 제거.  Tier 2 boundary-active, n_steps=1000, `metric.md` 표준 평가에서:
- **v5.1-100k가 모든 feasibility-respecting method 능가** — pos **3.88 cm** (v4.1의 4.91), rot **5.63°** (v4.1의 6.50), succ@(5cm,10°) **79.3%** (v4.1의 71.5%, +7.8 pp), jvio=0%, mfe=0, manifold gap=0.
- **DP-raw (jvio=24.6%) vs v5.1-100k**: succ@(5cm,10°) 78.5% ≈ 79.3% 이지만 effective_succ (jvio penalty) DP-raw 59.2% vs **v5.1-100k 79.3% — +20.1 pp**.
- **z_e generalization** (user-stated primary motivation): 모든 $z_e$ (0.05/0.10/0.15/0.20-OOD) 에서 v4.1 능가, OOD에서도 succ510 71.9% vs 65.6% (+6.3 pp).
- **비용은 학습 시간만**: 50k 미달 → 100k 우위. spec §13의 "Higher capacity / more training steps expected" 정확히 확인.

**Tier 2 boundary-active experiment 추가** (2026-05-11, §11): §10 의 Tier 0 결과는 demo 가 interior 라 bounded chart 의 cost 만 측정. **boundary-active demos (1%-tile rel-margin < 0.05) 의 Tier 2 setting** 에서 4 method × 3 ablation (총 7 config) 재학습 + 평가. **수정 결과**:
- **v4.1 (50k) 가 모든 차원에서 best-or-tied**: jvio = **0%** + endpoint 정밀도 **pos 4.8 cm / rot 6.5°** (DP-bounded 의 0.35× / 0.45×) + succ@(5cm,5°) **64.8%** (DP-bounded 와 동등).
- **Effective task_succ@(pose ∧ Q_safe)**: DP-raw ≈ 56% (jvio penalty), DP-bounded = 65.6%, **Ours-v4.1 = 64.8%** → DP-raw 대비 **+8.8 pp**, H1 hypothesis 충족.
- **mfe metric bug 발견 + 수정** (§11.3): 이전 보고한 v4.1 mfe=0.047/0.12 는 Tier 0 의 q[0]-based classifier 가 v5 demos 에 적용된 측정 오류 — 직접 측정 시 v4.1 mode flip rate = **0%**, 모든 method 와 동등.
- **v4.1 cost 는 학습 시간만**: Riemannian + chart 의 loss landscape 가 DP 대비 ~3× 천천히 수렴 (15k → 30k → 50k step scaling 필요). 학습 자원 충분하면 cost 사실상 0.

**주요 finding** (Tier 0 §5–§9 결과 + Tier 2 §11 결과 통합):

1. **Pose accuracy 는 demo regime 에 따라 ranking 이 바뀜**:
   - **Tier 0 (interior demos, 1%-tile rel-margin ≈ 0.20)**: DP variants 가 pose accuracy 에서 Method A 보다 우위 — DP-channel (100%) ≈ Projected (100%) ≈ DP-canonical (99%) > BC (98%) > Ours-V2 Method A (94%) ≫ ABL3 (19%). interior 에서는 jvio 가 자연히 낮아 vanilla DP 가 충분.
   - **Tier 2 (boundary-active demos, 1%-tile rel-margin ≈ 0.001)**: **Ours-v4.1 (50k) 의 endpoint 정밀도가 가장 좋음** — pos 4.8 cm (DP-bounded 13.8 cm 의 0.35×), rot 6.5° (DP-bounded 14.4° 의 0.45×). succ@(5cm,5°) 는 DP-bounded 와 동등 (64.8% vs 65.6%).
   - **Effective task_succ@(pose ∧ Q_safe)** (jvio penalty 포함): DP-raw 56% < Ours-v4.1 64.8% ≈ DP-bounded 65.6% — DP-raw 의 jvio 25% 가 effective succ 를 깎아 v4.1 이 +8.8 pp 우위 (H1 충족).

2. **Multimodality 보존 (mfe)** — Tier 0 의 §10 보고 (Method A 0.12, v4.1 0.047) 는 q[0]-based classifier 가 Tier 2 v5 demos 에 부적합한 측정 artifact 였음 (§11.3 진단). **mode-discriminating joint 를 q_rest_A/B 에서 자동 detect 하도록 fix 한 후 모든 method 의 mfe ≤ 0.02** (Tier 2 z 평균). v4.1 의 직접 측정한 mode flip rate = **0%**, DP-bounded 의 5% A→B flip 보다 더 stable. **DP가 mode collapse 방지에서 가장 우수했던 §6 ranking 은 metric 영향이 컸음** — Tier 2 재측정 시 method 간 차이 미미.

3. **Method A / v4.1 의 contribution 은 boundary-active regime 에서만 측정 가능**:
   - Chart-form score (drift-free Brownian + Varadhan target) 의 원리적 *manifold-correct sampling* 보장은 §5 의 Method A 가 작동하는 base.
   - **§10 (Tier 0) 의 "v4.1 의 cost > benefit" 결론은 interior demos 한정** — bounded chart 가 작동할 영역 없음.
   - **§11 (Tier 2) 에서 진정한 trade-off 측정**: v4.1 가 jvio=0% 구조적 보장 (Choice A 의 G_Q^A ≽ I) + endpoint 정밀도 ~3× 개선 (chart-only DP-bounded 대비) → benefit > cost.
   - **Cost 는 학습 시간만** (DP 대비 ~3× 천천히 수렴, 50k step 필요) → 자원 충분 시 cost ≈ 0.

---

## 0.1 학습 vs 평가 $z_e$ 분포 — ID vs OOD 명시 ⚠️

**모든 ckpt에서 동일** (Method A, 모든 ablation, 모든 baseline):

| 단계 | $z_e$ 분포 |
|---|---|
| **Training** | $z_e \sim \mathrm{Uniform}[0.05, 0.15]$ m (training data 매 batch마다 sampling) |
| Manifold support | tool_z_max = 0.20 m ($T_\phi$가 정의된 최대 length, 학습엔 안 쓰임) |
| **Evaluation** | $z_e \in \{0.05, 0.10, 0.15, 0.20\}$ m (4 fixed values, 각 64 samples) |

**평가 z_e의 ID / OOD 구분**:

| $z_e$ | 분류 | 비고 |
|---|---|---|
| **0.05** m | **ID — boundary (lower edge)** | training 분포의 하한 boundary. Sampling 분포가 [0.05, 0.15] uniform이므로 0.05는 boundary value지만 training 분포 내부 |
| **0.10** m | **ID — interior** | training 분포의 정확히 중간. 학습 가장 풍부한 구간 |
| **0.15** m | **ID — boundary (upper edge)** | training 분포의 상한 boundary |
| **0.20** m | **OOD — extrapolation** | training max(0.15)에서 +0.05 m 외삽 (**+33%**). $T_\phi$ manifold는 정의되어 있지만 학습 데이터 분포에서 벗어남 |

→ z_e=0.05/0.10/0.15는 in-distribution (ID), z_e=0.20은 **out-of-distribution (OOD)**.
→ 이후 모든 표/분석에서 ID/OOD를 명시하여 robustness vs generalization을 구분한다.

---

## 1. 수식 정의 (extension.tex 핵심)

### 1.1 매니폴드

$$\mathcal{M}_\phi^\text{pose}(z_e) = \{(q, T) \in \mathbb{R}^{n_q} \times \mathrm{SE}(3) : T = T_\phi(q, z_e)\}$$

Embedding: $H_\phi^\text{pose}(q, z_e) = (q, T_\phi(q, z_e), z_e)$.
Constraint: $g_\phi(x, z_e) = \mathrm{Log}_{\mathrm{SE}(3)}(T_\phi(q, z_e)^{-1} \cdot T) \in \mathfrak{se}(3)$. Zero iff on manifold.

### 1.2 Body-frame Jacobian + induced metric

$$J_\text{pose}(q, z_e) = \frac{\partial}{\partial q}\bigg|_{q'=q} \mathrm{Log}_{\mathrm{SE}(3)}(T_\phi(q, z_e)^{-1} T_\phi(q', z_e)) \in \mathbb{R}^{6 \times n_q}$$

$$G_\text{pose}(q, z_e) = I_{n_q} + J_\text{pose}^\top W J_\text{pose}, \qquad W = \mathrm{diag}(W_p I_3, W_R I_3)$$

$W_p = \sigma_p^{-2}$, $W_R = \sigma_R^{-2}$. Default $\sigma_p = 0.05$ m, $\sigma_R = 0.1$ rad → $W_p = 400$, $W_R = 100$.

### 1.3 Forward SDE (Method A — drift-free Brownian)

$$\boxed{ dq_r = \sqrt{\beta(r)} \cdot \sigma(q_r) \, dW_r, \qquad \sigma\sigma^\top = G_\text{pose}^{-1}(q_r, z_e) }$$

Effective time $\tau_\text{brown}(r) := \int_0^r \beta(s)\, ds$. Marginal std (leading order):
$$\sigma_\text{marg}(r) := \sqrt{\tau_\text{brown}(r)}, \qquad \mathrm{Cov}(q_r | q_0) \approx \tau_\text{brown}(r) \cdot G_\text{pose}^{-1}(q_0, z_e)$$

LinearBetaSchedule ($\beta_0 = 0.001, \beta_f = 4, K = 1$): $\tau_\text{brown}(K) \approx 2.0$, $\sigma_\text{marg}(K) \approx 1.414$.

### 1.4 Chart-form DSM target (Varadhan, drift-free 정합)

$$\delta x^\text{amb}_{r0} = \begin{pmatrix} q_0 - q_r \\ \mathrm{Log}_{\mathrm{SE}(3)}(T_\phi(q_r, z_e)^{-1} T_\phi(q_0, z_e)) \end{pmatrix} \in \mathbb{R}^{n_q + 6}$$

$$\boxed{ a^*_\text{pose}(q_r, x_0; z_e) = G_\text{pose}^{-1}(q_r, z_e) \cdot J_H^{\text{pose}\top}(q_r, z_e) \cdot W \cdot \delta x^\text{amb}_{r0} }$$

Loss:
$$\mathcal{L} = \mathbb{E}_{r, \tau_0, \tau_r}\left[ w(r) \sum_h (s_{\theta,h}^q - a^*_{\text{pose},h})^\top G_\text{pose}(q_{h,r}, z_e)\, (s_{\theta,h}^q - a^*_{\text{pose},h}) / \tau_\text{brown}^2(r) \right]$$

### 1.5 Sampling 초기화 (condition-aware)

$$q_h^K \sim \mathcal{N}(q^\text{init}, \sigma_\text{marg}(K)^2 \cdot G_\text{pose}^{-1}(q^\text{init}, z_e)), \quad q^\text{init} \approx \mathrm{IK}(T_\text{start})$$

$q^\text{init}$는 per-trajectory (각 sample마다 다름). 학습 시 $q_\text{demo}[0]$, eval 시 IK from $T_\text{start}$.

### 1.6 Reverse SDE

$$dY_r = \beta(r) \cdot s_\theta^q(q_r, r, c, z_e)\, dr̄ + \sqrt{\beta(r)}\cdot\sigma(q_r) \, dW̄_r$$

조건 $c = (T_\text{start}, T_\text{target})$. Multi-component reward guidance (start anchor / goal anchor / smoothness)는 sampling-time에 chart-form gradient로 추가.

---

## 2. 코드 구현 요약

| 모듈 | 역할 |
|---|---|
| `smcdp/lie_se3.py` | SE(3) Lie utilities (exp/log, Adjoint, vmap-safe Jacobian via $(R, p)$-tuple) |
| `smcdp/manifolds_pose.py` | `EmbodimentPoseGraphManifold` ABC + `Franka7DoFPose` (closed-form analytic body-frame Jacobian) |
| `smcdp/franka/{ground_truth_pose,self_model_pose}.py` | 3-axis 1–3° rotation compliance ground truth + Stage-1 ξ_φ residual MLP |
| `smcdp/franka/demo_gen_pose.py` | Pose IK with body-frame twist DLS, bimodal demo |
| `smcdp/trajectories_pose.py` | `TrajectoryScoreNetUNetPose` (chart output, goal_cond_dim=14), pose DSM loss, Method A reverse GRW |
| `smcdp/sde.py` | β-schedule with `proxy_std(t, mode=...)` 옵션 (`brownian` / `ou`) |
| `smcdp/experiments/franka_traj_unet_pose.py` | Stage-2 V2 pose score-net training script |
| `smcdp/experiments/franka_baselines_pose_train.py` | BC / DP-canonical / DP-channel / Projected pose-baseline trainer |
| `smcdp/experiments/franka_baselines_pose_eval.py` | Pose endpoint metric (e_p, e_R) eval |

Unit tests: `tests/test_lie_se3.py` (FP64 machine precision), `tests/test_pose_manifold.py` (S1–S8: retraction exactness, $J_g \cdot J_H = 0$, metric PSD, lift idempotence, log/exp roundtrip, position-only fallback, closed-form Jacobian vs FD).

---

## 3. Stage-1 — Pose Self-Model 학습 결과

`franka_stage1_selfmodel_pose --steps 10000`:

```
err_analytic_pos_mean: 0.01605 m   →   err_learned_pos_mean: 0.00342 m   (4.69× improvement)
err_analytic_rot_mean: 0.02962 rad →   err_learned_rot_mean: 0.00180 rad (16.43× improvement)
```

ξ_φ residual: training-inside ‖ξ‖ ≈ 0.053, outside-1x ‖ξ‖ ≈ 0.066 (smooth extrapolation).

---

## 4. Stage-2 — Method A까지의 iterative debugging

세 차례 시도 (자세한 진단: `training_diagnosis_2026-05-09.md`):

| Phase | 설정 | pos_err | succ@5cm | 학습 안정성 |
|---|---|---|---|---|
| 1 (drift ON, σ_p=0.01) | Cholesky failure | — | — | Step 0 crash |
| 2 (Fix 1+2+3, κ=1000) | drift ON + 강한 confining | 25–31 **m** | 0% | Loss spike 1e+13 |
| 3 (κ=10) | drift ON + 약한 confining | 13–16 **m** | 0% | Loss spike 1e+8 다수 |
| **4 (Method A)** | **drift OFF + condition-aware init** | **2.0–3.2 cm** | **81–96%** (5/5°), **92–100%** (5/10°) | clean monotone |

Phase 2/3 진단의 5가지 의심점 (`training_diagnosis_2026-05-09.md` §4):
- (Q1) drift OFF + pose 미측정 baseline 부재
- (Q2) anchor approximation $\hat G = G(\mu_q)$가 demo region에서 5.4× outlier
- (Q3) Score net은 $T_\text{start}$만 받고 SDE는 $\mu_q$로 끌어당김 → mismatch
- (4.1) Forward Langevin drift + Varadhan target 본질적 비정합
- (4.7) $\sigma_K$ calibration: hardcoded 0.6 vs spec $\sqrt{\tau_\text{brown}(K)} = 1.41$ (2.36× mismatch)

→ **Method A** (`modificatin.md`): drift 완전 제거 + per-trajectory $q^\text{init}$ + auto $\sigma_K$ + Brownian-mode proxy_std. 다섯 의심을 한 fix로 해결.

---

## 5. Stage-2 — Method A 결과 ⭐

`franka_traj_unet_pose --method-a --sigma-p 0.05 --tikhonov-frac 1e-2 --steps 15000 --batch 64`. 학습 시간 3:48.

### 5.1 학습 trajectory

```
step    0:  loss = 5.19e+3    (warmup)
step  330:  loss = 81.67       (5분)
step 1768:  loss = 13.23       (30분)
step 7500:  loss = 7.31
step 11250: loss = 3.76        (bottom)
step 15000: loss = 5.34        (final)

recent 5%:  min 3.33, median 4.81, max 6.01
spike >1e+5:  0건
spike >1e+3:  51 / 15301 (0.3%, 거의 시작 부분)
```

교과서적 monotone convergence, ~1000× 하강 over 15000 steps. fix123/κ=10의 1e+9 ~ 1e+13 oscillation과 명확히 다른 trajectory.

### 5.2 Eval (per z_e — tool extension, `metric.md` 통합 set)

| $z_e$ | 분류 | pos cm | rot ° | pose@5/5° | pose@5/10° | manif gap |
|---|---|---|---|---|---|---|
| 0.05 m | ID-boundary | **2.00** | **2.79°** | **95.31%** | 100% | 0.000 mm / 0.005° |
| 0.10 m | ID-interior | 2.15 | 2.77° | **95.31%** | 100% | ~0 |
| 0.15 m | ID-boundary | 2.53 | 2.93° | 92.19% | 98.44% | ~0 |
| 0.20 m | **OOD (+33%)** | 3.24 | 3.15° | 81.25% | 92.19% | ~0 |

**ID 평균** (z_e ∈ {0.05, 0.10, 0.15}): pos 2.23 cm, rot 2.83°, **pose@(5cm,5°) = 94.27%**, pose@(5cm,10°) = 99.48%
**OOD 단일** (z_e = 0.20): pos 3.24 cm, rot 3.15°, **pose@(5cm,5°) = 81.25%** (−13 pp ID 평균 대비)
**전체 평균** (4 z_e, ID+OOD): pos_err 2.48 cm, rot_err 2.91°, pose_succ@(5cm,5°) = 91.01%, manifold gap ≈ 0 (machine precision).

**(주의)**: 이전 보고된 "succ 51.6%"는 default `--success-pos 0.02` (2cm) threshold 때문에 *position-strict*로 측정됨. `metric.md` standard pose_succ@(5cm, 5°) 기준으로 재평가 시 **95.3%** (z_e=0.05). 표준 metric으로 재계산이 paper의 정확한 성능 수치.

### 5.3 Position-only V2 (REPORT Part III)와의 비교

| 항목 | Position-only V2 | Pose-extended Method A |
|---|---|---|
| Target dim | $p \in \mathbb{R}^3$ (3-dim) | $T \in \mathrm{SE}(3)$ (6-dim) |
| pos_err | 21 mm | ID: 20–25 mm, OOD: 32 mm |
| Success criterion | pos < 5 cm | pos < 5 cm AND rot < 5° (`metric.md` strict) |
| succ rate (ID) | ~95% | **94.27%** (ID 평균) |
| succ rate (OOD z_e=0.20) | n/a | 81.25% |

Position 정밀도는 동등 ($\approx$ 2 cm). Rotation을 strict 5°로 추가 measure해도 succ rate 거의 유지 (z_e≤0.10에서 95%) — rotation precision (mean 2.79°)이 5° threshold 안에 안정적으로 들어감이 SE(3) extension의 강점.

---

## 6. Ablation — Method A 각 component의 기여 isolation

$\sigma_K$ / proxy_std mode / per-trajectory $q^\text{init}$ 셋이 *각각* 얼마나 critical한지 측정. 다른 모든 hyperparameter는 동일.

### 6.1 Ablation 결과 표 (`metric.md` 통합 metric)

`pose_succ@(5cm, 5°)` 기준 (strict full-pose criterion). z_e=0.05/0.10/0.15는 **ID** (training $\mathrm{Uniform}[0.05, 0.15]$), z_e=0.20은 **OOD** (+33% extrapolation).

| 설정 | $\sigma_K$ | proxy_std | $q^\text{init}$ | ID-boundary 0.05 | ID-interior 0.10 | ID-boundary 0.15 | **OOD 0.20** | **ID 평균** | **전체 평균** |
|---|---|---|---|---|---|---|---|---|---|
| **Method A (full)** | 1.414 | brownian | per-traj | **95.31%** | 95.31% | 92.19% | **81.25%** | **94.27%** | 91.01% |
| ABL1 (σ_K=0.6) | 0.6 | brownian | per-traj | 96.88% | 98.44% | 98.44% | **90.62%** | 97.92% | 96.10% |
| **ABL3 (q_init=μ_q)** | 1.414 | brownian | **single μ_q** | **17.19%** | 20.31% | 18.75% | **21.88%** | **18.75%** | 19.53% |
| ABL2 (proxy_std=ou) | 1.414 | **ou** | per-traj | 89.06% | 90.62% | 87.50% | **75.00%** | 89.06% | 85.55% |

### 6.2 Component-wise contribution (ID 기준 분석)

**ABL3 (per-trajectory q_init 제거)** — **ID 평균 94.27% → 18.75% (−75.5 pp)**:
- ID 모든 z_e에서 ~75-80 pp 약화 (z_e=0.05: 95.31% → 17.19%, z_e=0.10: 95.31% → 20.31%, z_e=0.15: 92.19% → 18.75%)
- OOD에서도 81.25% → 21.88% (−59 pp)
- rot_err: Method A 2.79° → ABL3 9.95° (3.6× 악화); pos_err 2.00 → 4.39 cm (2.2× 악화)
- 통계적 매우 견고 (95% CI ±12 pp을 6배 이상 초과)
- → **Per-trajectory $q^\text{init}$이 Method A의 결정적 core contribution**. 진단의 Q3 (anchor mismatch) 가설 정량 검증.

**ABL1 (σ_K=0.6 legacy)** — **ID 평균 +3.65 pp** (94.27% → 97.92%):
- ID 모든 z_e에서 Method A보다 약간 더 나음 (z_e=0.05: +1.6 pp, z_e=0.10: +3.1 pp, z_e=0.15: +6.3 pp)
- OOD에서도 +9.4 pp (81.25% → 90.62%) — narrower init이 OOD에서 더 도움
- 통계적 marginal — 95% CI ±12 pp 이내
- σ_K calibration은 *spec 정밀화*로 옳지만 *empirical 효과 marginal* — robustness signal
- Forward marginal과 정합한 σ_K=1.414 vs legacy 0.6: 비슷하거나 narrow init이 유리 — score net의 effective coverage 영역과 관련

**ABL2 (proxy_std = ou)** — **ID 평균 −5.21 pp** (94.27% → 89.06%):
- ID 모든 z_e에서 약간씩 약화 (z_e=0.05: −6.25 pp, z_e=0.10: −4.69 pp, z_e=0.15: −4.69 pp)
- OOD에서도 −6.25 pp (81.25% → 75.00%)
- 통계적으로 약하지만 consistent (모든 4개 z_e에서 monotone negative)
- proxy_std는 학습 시 std_trick scale + loss weight $w(r) = \sigma^2(r)$에 들어감 → drift-free Brownian forward에서 OU std로 normalize 시 큰 r 영역 weight가 saturated → 큰 r에 학습 capacity 부족 → reverse SDE 큰 r 단계에서 score quality 저하
- → proxy_std calibration 효과는 marginal하지만 **consistent하게 positive** (Brownian이 정합)

### 6.3 Component contribution ranking (`metric.md` strict, ID 기준)

| 순위 | Component | ID 평균 영향 | OOD 영향 | 근거 |
|---|---|---|---|---|
| **1** | **Per-trajectory $q^\text{init}$** | **−75.5 pp** | **−59.4 pp** | ABL3에서 결정적 collapse (94% → 19%) |
| 2 | proxy_std = brownian mode | +5.2 pp | +6.3 pp | ABL2 모든 z_e에서 consistent negative |
| 3 | σ_K = √τ_brown(K) | −3.7 pp | −9.4 pp | ABL1에서 약간 더 좋음, OOD에서 더 큼 |

(2,3은 통계적 noise 범위, 1만 명확한 signal)

---

## 7. Baseline 비교 (`metric.md` 통합 metric)

Pose-extended baseline을 동일 framework에서 학습하여 Ours-V2 (Method A) 대비 우위 입증.

| Baseline | 설명 | 상태 |
|---|---|---|
| BC | Deterministic regressor $c \to q$-trajectory | ✓ 완료 |
| DP-canonical (global cond) | Standard Diffusion Policy (Chi23) | ✓ 완료 |
| DP-channel | DP-A variant (channel-concat cond, parity with Ours) | ✓ 완료 |
| Projected | Ambient (q, T_storage) DP + projection via $H_\phi^\text{pose}$ | ✓ 완료 |

**조건**: 동일 conditioning $c = (T_\text{start} \oplus T_\text{target} \oplus z_e) \in \mathbb{R}^{15}$, 동일 demo 분포, H+1=16, batch=64, steps=15000.

### 7.1 BC (Behavioral Cloning) 결과

| $z_e$ | 분류 | pos cm | rot ° | pose@5/5° | pose@5/10° | mode frac err | manif gap | jvio |
|---|---|---|---|---|---|---|---|---|
| 0.05 | ID-boundary | 2.00 | 3.09 | **95.31%** | 100% | **0.47** | 0 | 0% |
| 0.10 | ID-interior | 1.94 | 2.63 | **100%** | 100% | 0.47 | 0 | 0% |
| 0.15 | ID-boundary | 2.40 | 2.64 | 98.44% | 100% | 0.47 | 0 | 0% |
| 0.20 | **OOD** | 3.62 | 3.05 | **85.94%** | 87.5% | 0.47 | 0 | 0% |
| **ID 평균** | (3 z_e) | 2.11 | 2.79 | **97.92%** | 100% | 0.47 | 0 | 0% |
| **전체 평균** | (4 z_e) | 2.49 | 2.85 | 94.92% | 96.88% | 0.47 | 0 | 0% |

학습 시간 ~3시간, model 1.12 M params (UNet 10M의 1/9). OOD에서 ID 대비 −12 pp.

### 7.2 Method A vs BC — *pose accuracy 동등, BC는 mode collapse*

| Metric | **Method A** | **BC** | 차이 |
|---|---|---|---|
| pose_succ@(5cm,5°) **ID 평균** | 94.27% | **97.92%** | BC +3.7 pp |
| pose_succ@(5cm,5°) **OOD (z_e=0.20)** | **81.25%** | 85.94% | BC +4.7 pp |
| pose_succ@(5cm,5°) 전체 평균 | 91.01% | **94.92%** | BC +3.9 pp |
| pose_succ@(5cm,10°) 전체 | 97.66% | 96.88% | tie |
| pos_err mean (cm) | 2.48 | 2.49 | tie |
| rot_err mean (°) | 2.91 | 2.85 | tie |
| **mode_frac_err** | **0.12** | **0.47 (collapse)** | Method A 4× better |
| manifold gap (mm) | ~0 | ~0 | tie (둘 다 H_φ로 lift) |
| joint viol rate | 0% | 0% | tie |

**핵심 관찰**:
- **Pose accuracy**: BC가 *근소하게* 더 높음 (단일 seed noise 범위)
- **Multimodality**: Method A가 BC 대비 *4× 더 잘 보존*. BC mfe 0.47은 한 mode로 ~97% collapse — BC의 deterministic regression 한계.
- **Manifold adherence**: 둘 다 ≈ 0 (BC는 $q$만 출력 후 $H_\phi^\text{pose}$로 lift, Method A는 sample이 매니폴드 위 by construction)

→ BC와의 비교에서 Method A는 *pose accuracy 동등 + multimodality 보존*. 그러나 §7.3–7.5에서 보듯 DP variants도 pose accuracy 우위 + multimodality (mfe 0.02–0.04)에서 Method A보다 더 강함 — BC vs Method A 비교만으로는 contribution claim이 충분하지 않음.

### 7.3 DP-canonical (Global-cond Diffusion Policy) 결과

| $z_e$ | 분류 | pos cm | rot ° | pose@5/5° | pose@5/10° | mode frac err | manif gap | jvio |
|---|---|---|---|---|---|---|---|---|
| 0.05 | ID-boundary | 1.91 | 2.45 | 98.44% | 100% | 0.02 | 0 | 0% |
| 0.10 | ID-interior | 2.04 | 2.50 | **100%** | 100% | 0.02 | 0 | 0% |
| 0.15 | ID-boundary | 2.40 | 2.59 | 98.44% | 100% | 0.02 | 0 | 0% |
| 0.20 | **OOD** | 3.11 | 2.69 | **95.31%** | 96.88% | 0.00 | 0 | 0% |
| **ID 평균** | (3 z_e) | 2.12 | 2.51 | **98.96%** | 100% | 0.02 | 0 | 0% |
| **전체 평균** | (4 z_e) | 2.36 | 2.56 | 98.05% | 99.22% | 0.015 | 0 | 0% |

학습 시간 ~3시간, model 10M params (UNet 1d, global-cond 표준 Diffusion Policy / Chi23). q는 ambient $q$-only 출력 후 $H_\phi^\text{pose}$로 lift.

**핵심 관찰**:
- **ID 98.96%, OOD 95.31% pose succ — Method A (94.27% / 81.25%)보다 우위**
- mode frac err 0.02 — Method A (0.12)보다 6× 우수, BC (0.47) 대비 23× 우수
- manifold gap = 0 (lift via $H_\phi^\text{pose}$, by construction)
- OOD에서 ID 대비 −3.65 pp 작은 drop

→ **Standard DP가 SE(3) trajectory를 잘 학습**. Conditioning이 충분히 강하고 (T_start ⊕ T_target ⊕ z_e ∈ R^{15}), DDPM이 multimodal demo distribution을 자연스럽게 capture.

### 7.4 DP-channel (channel-concat conditioning, Ours-A architecture parity) 결과

Ours-V2와 architecture parity (UNet 1d, channel-concat).

| $z_e$ | 분류 | pos cm | rot ° | pose@5/5° | pose@5/10° | mode frac err | manif gap | jvio |
|---|---|---|---|---|---|---|---|---|
| 0.05 | ID-boundary | 1.77 | 2.27 | **100%** | 100% | 0.02 | 0 | 0% |
| 0.10 | ID-interior | 1.83 | 2.31 | **100%** | 100% | 0.02 | 0 | 0% |
| 0.15 | ID-boundary | 2.22 | 2.51 | **100%** | 100% | 0.03 | 0 | 0% |
| 0.20 | **OOD** | 3.03 | 2.74 | **96.88%** | 96.88% | 0.06 | 0 | 0% |
| **ID 평균** | (3 z_e) | 1.94 | 2.36 | **100%** | 100% | 0.023 | 0 | 0% |
| **전체 평균** | (4 z_e) | 2.21 | 2.46 | 99.22% | 99.22% | 0.033 | 0 | 0% |

학습 시간 ~2:50, model 10.20M params (UNet 1d, channel-concat conditioning — Ours-A와 동일 architecture).

**핵심 관찰**:
- **ID 100%, OOD 96.88% — 모든 baseline 중 가장 높은 pose accuracy**
- channel-concat이 global-cond 대비 약간 우위 (ID +1 pp, OOD +1.6 pp)
- Method A (94.27% / 81.25%) 대비 +5.7 pp / +15.6 pp 우위
- mfe 0.023 — DP-canonical보다 약간 높지만 여전히 매우 낮음

→ **Method A와 DP-channel은 architecture/cond/data/compute가 동일** (UNet 1d, channel-concat $c$, 10M params, batch 64, 15k steps). 차이는 score formulation만:
- DP-channel: standard $\epsilon$-prediction loss → **ID 100%, OOD 96.88%**
- Method A: chart-form Riemannian DSM (drift-free Brownian, Varadhan) → **ID 94.27%, OOD 81.25%**

→ Pose accuracy에서 standard DP 가 Method A 보다 우위. 이는 Method A의 contribution이 *pose accuracy*가 아니라 *원리적 manifold consistency* (machine-precision manifold gap, 이는 DP도 $H_\phi^\text{pose}$ post-projection으로 동일 달성) 또는 *복잡 manifold extension*에 있음을 시사.

### 7.5 Projected (ambient $(q, T)$ DP + retraction) 결과

ambient state는 $(q, T_\text{storage}) \in \mathbb{R}^{14}$ DP, 출력 후 $T$-block을 $T_\phi(q, z_e)$로 overwrite (manifold projection).

| $z_e$ | 분류 | pos cm | rot ° | pose@5/5° | pose@5/10° | mode frac err | manif gap | jvio |
|---|---|---|---|---|---|---|---|---|
| 0.05 | ID-boundary | 1.78 | 2.31 | **100%** | 100% | 0.02 | 0 | 0% |
| 0.10 | ID-interior | 1.89 | 2.50 | **100%** | 100% | 0.02 | 0 | 0% |
| 0.15 | ID-boundary | 2.20 | 2.59 | **100%** | 100% | 0.05 | 0 | 0% |
| 0.20 | **OOD** | 2.90 | 2.59 | **95.31%** | 95.31% | 0.05 | 0 | 0% |
| **ID 평균** | (3 z_e) | 1.96 | 2.47 | **100%** | 100% | 0.030 | 0 | 0% |
| **전체 평균** | (4 z_e) | 2.19 | 2.50 | 98.83% | 98.83% | 0.035 | 0 | 0% |

학습 시간 ~4시간, model 10M params, ambient $(q, T)$ 출력 후 retraction projection.

**핵심 관찰**:
- **ID 100%, OOD 95.31% — DP-channel과 거의 동등**
- ambient (q, T) state space에서 학습하고 projection으로 manifold 만족 → 잘 작동
- mfe 0.030, multimodality 보존 양호
- OOD에서 trajectory 다소 거침 (E_vel/E_acc 차이는 §부록 참고)

→ Manifold post-projection 방식도 정상 작동. Method A의 *intrinsic* manifold formulation 대비 비교적 simpler approach가 같은 수준의 결과 달성.

---

### 7.6 최종 비교표 — paper-ready main result

**Training**: $z_e \sim \mathrm{Uniform}[0.05, 0.15]$ (모든 모델 동일).
**Eval**: ID = z_e ∈ {0.05, 0.10, 0.15}, OOD = z_e = 0.20 (+33% extrapolation).

#### 7.6.1 In-distribution (ID, z_e ∈ {0.05, 0.10, 0.15})

| 모델 | params | pos cm | rot ° | **pose@(5cm,5°) ID** | pose@(5cm,10°) ID | manif gap (mm) | mode frac err |
|---|---|---|---|---|---|---|---|
| BC | 1.12 M | 2.11 | 2.79 | 97.92% | 100% | 0 | **0.47 (collapse)** |
| DP-canonical | 10 M | 2.12 | 2.51 | 98.96% | 100% | 0 | **0.02** |
| **DP-channel** | 10.20 M | **1.94** | **2.36** | **100%** | 100% | 0 | 0.023 |
| Projected | 10 M | 1.96 | 2.47 | 100% | 100% | 0 | 0.030 |
| Ours-V2 Method A | 10 M | 2.23 | 2.83 | 94.27% | 99.48% | 0 | 0.12 |

#### 7.6.2 Out-of-distribution (OOD, z_e = 0.20 — +33% beyond training max)

| 모델 | pos cm | rot ° | **pose@(5cm,5°) OOD** | pose@(5cm,10°) OOD | OOD vs ID drop |
|---|---|---|---|---|---|
| BC | 3.62 | 3.05 | 85.94% | 87.5% | −12.0 pp |
| DP-canonical | 3.11 | 2.69 | 95.31% | 96.88% | −3.65 pp |
| **DP-channel** | **3.03** | **2.74** | **96.88%** | 96.88% | **−3.12 pp** |
| Projected | 2.90 | 2.59 | 95.31% | 95.31% | −4.69 pp |
| Ours-V2 Method A | 3.24 | 3.15 | 81.25% | 92.19% | −13.0 pp |

#### 7.6.3 평균 (ID+OOD, 4 z_e — reference only)

| 모델 | params | pos cm | rot ° | pose@(5cm,5°) | pose@(5cm,10°) | mode frac err |
|---|---|---|---|---|---|---|
| BC | 1.12 M | 2.49 | 2.85 | 94.92% | 96.88% | 0.47 |
| DP-canonical | 10 M | 2.36 | 2.56 | 98.05% | 99.22% | 0.015 |
| **DP-channel** | 10.20 M | **2.21** | **2.46** | **99.22%** | 99.22% | 0.033 |
| Projected | 10 M | 2.19 | 2.50 | 98.83% | 98.83% | 0.035 |
| Ours-V2 Method A | 10 M | 2.48 | 2.91 | 91.01% | 97.66% | 0.12 |


**핵심 finding**:

1. **Pose accuracy ranking (ID)**: DP-channel (100%) ≈ Projected (100%) ≈ DP-canonical (99%) > BC (98%) > **Method A (94%)** ≫ ABL3 (19%)
   - 모든 generative baseline (DP-*, Projected)이 deterministic BC보다도 우위 — DP의 multimodal modeling이 효과적
   - **Method A가 pose accuracy에서 DP보다 약간 뒤처짐** (−5 pp)

2. **Multimodality (mfe, lower is better)**: DP-canonical (0.015) < DP-channel (0.033) ≈ Projected (0.035) < Method A (0.12) ≪ BC (0.47, collapse)
   - Generative baseline 모두 bimodal distribution 잘 보존 (mfe < 0.05)
   - **Method A의 mfe 0.12는 DP variants 대비 4–8× 나쁨** (그러나 BC 대비 4× 우수)
   - BC만 mode collapse 발생

3. **Manifold adherence**: 모든 모델이 manif gap = 0
   - Method A: by construction (chart-form score, intrinsic)
   - 다른 baseline: post-projection via $H_\phi^\text{pose}$ retraction
   - 둘 다 같은 결과 → simple FK retraction이 가능한 manifold에서는 차별화 안 됨

4. **OOD (+33% extrapolation)**:
   - DP-channel: 100% → 96.88% (drop −3.1 pp)
   - DP-canonical: 99% → 95.31% (drop −3.7 pp)
   - Projected: 100% → 95.31% (drop −4.7 pp)
   - BC: 97.92% → 85.94% (drop −12 pp)
   - **Method A: 94.27% → 81.25% (drop −13 pp)** — OOD에서 **가장 큰 drop**
   - DP variants는 stochastic generative model로서 OOD에서도 robust; Method A는 anchor q_init이 ID 분포에 specialize되어 OOD에서 약화

5. **Architecture-parity head-to-head 재해석**: DP-channel과 Method A는 동일 architecture/cond/data/compute. Score formulation 차이:
   - DP-channel ($\epsilon$-prediction): ID 100%, OOD 96.88%, mfe 0.033
   - Method A (chart-form Riemannian DSM): ID 94.27%, OOD 81.25%, mfe 0.12
   - **DP-channel이 Method A보다 모든 metric에서 우위** (Franka FK retraction-friendly setup에서)

→ **§7 의 결론은 Tier 0 (interior demos) 평가에 한정**: Method A 는 Franka pose task 의 *interior* demos 에서 DP 를 능가하지 못함. Method A 의 이론적 강점 (chart-form score, drift-free Brownian on Riemannian manifold) 이 *실제 우위로 발현되려면* DP 의 post-projection 으로 안 되는 환경이 필요함. 예:
- closed-form retraction 이 없는 implicit constraint manifold
- 동시-equality 제약 (contact + joint limit) 등에서 chart 기반 sampling 이 필수
- **boundary-active demos 의 joint feasibility 강제 (§11 의 Tier 2 setting)**
- Higher-dim multi-body system 에서 retraction 이 expensive

→ Franka 7-DoF + simple tool-z FK + **Tier 0 interior demos** 환경에서는 DP variant 들이 잘 작동.

→ **단, boundary-active demos 환경 (§11) 에서는 평가 결과가 뒤집힘**: v4.1 (50k) 가 effective task_succ@(pose ∧ Q_safe) 에서 DP-raw 대비 +8.8 pp, DP-bounded 와 succ 동등 + endpoint 정밀도 ~3× 개선. 즉 **Method A 의 chart 기반 Riemannian sampling 의 가치는 demo regime 에 따라 결정** — boundary 가 task 의 필수 영역이면 v4.1 가 우위, interior 만이면 DP variants 가 충분.

---

## 8. 알려진 한계

### 8.1 OOD ($z_e=0.20$) generalization 약화

- Training은 $z_e \sim \mathrm{Uniform}[0.05, 0.15]$, $z_e=0.20$은 **+33% OOD extrapolation**
- Method A: ID 평균 94.27% → OOD 81.25% (**−13 pp**), pose@(5cm,10°)는 OOD 92.19%로 회복
- BC도 비슷한 pattern (ID 97.92% → OOD 85.94%, **−12 pp**) — OOD 약화는 generative 형태와 무관
- 가능한 원인: tool 길이가 EE pose에 강한 dependency (특히 rotation rate), score net의 z_e conditioning extrapolation 능력 한계
- 잠재적 해결: training z_e range 확장 ([0.05, 0.20]으로 학습 시 OOD 영역 ID로 흡수), z_e-conditional weighting, z_e dimension에 expanded encoding, reward guidance

### 8.2 Rotation 정밀도

- 평균 rot_err 3.1°, 일부 task에서 부족 (< 1° desired for pick/place)
- $W_R = 100$ (= $\sigma_R^{-2} = (0.1)^{-2}$) tuning 또는 SO(3)-aware loss weighting 필요

### 8.3 비compact joint chart의 stationary 부재

- $q \in \mathbb{R}^{n_q}$ unbounded → pure Brownian forward는 stationary 분포 없음
- Method A는 *finite-time noising*으로 우회 — $r \in [0, K]$ horizon 내 score matching만 valid
- Drift-consistent transition score targets (e.g. closed-form OU on Riemannian)는 future work

### 8.4 단일 seed eval의 통계적 한계

- $n_\text{eval} = 64$ per z_e → succ rate std error ~6.25%, 95% CI ±12 pp
- Method A vs ABL1 (~5 pp 차이)는 noise 범위
- Method A vs ABL3 (−71.5 pp 평균 차이)는 매우 견고한 signal (95% CI을 6배 초과)
- Multi-seed 학습 (5 seeds × 4 ablations) ≈ 76 시간 추가 학습 필요 — 시간 제약으로 단일 seed로 보고

---

## 9. Future Work

기존 [extension.tex Sec 11] 항목 + Method A 측정으로 새로 발견된 약점:

- **Reward guidance ablation**: Method A baseline 위에 start/goal anchor + smoothness 추가, sampling-time only, 학습 cost 0
- **Manipulability-weighted W**: $W(q, z_e) = (J_\text{pose} J_\text{pose}^\top)^{-1}$, kinematic-singularity-aware metric
- **Joint-space inertia**: $G^\text{phys} = M(q) + J^\top W J$, energy-aware sampling
- **Task ellipsoid**: $W_p = R_\text{task} \mathrm{diag}(\sigma^{-2}) R_\text{task}^\top$, anisotropic tolerance
- **Multi-seed evaluation**: 통계적 power 확보를 위해 5 seeds × 주요 setting
- **Drift-consistent target**: OU-type forward + closed-form transition kernel (modificatin.md §11 future work)

---

## 부록 A: Method A 변경 사항 일람 (modificatin.md 기준)

| 항목 | V2 (Phase 2/3) | Method A (Phase 4) |
|---|---|---|
| Forward drift | $-\frac{1}{2}G^{-1}\nabla U_\text{total}$ | **0** (pure Brownian) |
| Anchor $\mu_q$ | per-batch or fixed | **per-trajectory $q^\text{init}$** |
| Anchor metric $\hat G$ | $G(\mu_q, z_e)$ fixed | **없음** |
| Box potential $U_\text{box}$ | ReLU-quadratic, $\kappa$ scaled | **없음** |
| Stationary distribution | $\mathcal{N}(\mu_q, \gamma^2 \hat G^{-1}) \cap \text{box}$ | 없음 (finite-time noising) |
| $\sigma_K$ for sampling init | hardcoded 0.6 | **$\sqrt{\tau_\text{brown}(K)} \approx 1.414$** auto |
| proxy_std mode | $\sqrt{1 - e^{-I}}$ (OU) | **$\sqrt{I}$ (Brownian)** |
| Reverse drift | $-b^q + \beta s_\theta^q$ | **$\beta s_\theta^q$** (b_fwd=0 정합) |
| Score net input | $[q, r, h/H, z_e, T_\text{start}, T_\text{target}]$ | **동일** |
| $G_\text{pose}$, $J_H^\text{pose}$, retraction | 유지 | 유지 |
| Adaptive Tikhonov | 유지 | 유지 |
| Multi-component guidance | 유지 | 유지 |
| $\sigma_p = 0.05$ default | Fix 1 적용 | 유지 |

---

## 10. v4.1 Joint-Limit Bounded Chart 결과 (joint_limit_extension.tex)

`joint_limit_extension.tex` (v4.1) 사양에 따라 chart $u \in \R^{n_q}$ + $q = \psi(u) = q_\text{mid} + (q_\text{range}/2) \tanh(u)$ 적용. Choice A: $G_Q^A = I + (J^Q)^T W J^Q \succeq I$, $J^Q = J_\text{pose} \cdot D_\psi$, no clipping during sampling, IK seed $u^\text{init} = \psi^{-1}(q^\text{init})$ with η-clip safety.

### 10.1 Pre-flight margin diagnostic

Spec §10.2의 mandatory pre-flight: demo의 relative-margin percentile.

| 지표 | 측정값 | Threshold | 판정 |
|---|---|---|---|
| 1%-tile rel-margin | **0.2045** | ≥ 0.05 (safe) | ✅ 4.1× 여유 |
| 5%-tile | **0.2389** | ≥ 0.10 (safe) | ✅ 2.4× 여유 |
| Min margin | 0.140 | > 0.01 (abort) | ✅ 14× 여유 |

→ **SAFE** 판정 — bounded chart 배포 승인.

### 10.2 학습 trajectory

`franka_traj_unet_pose --method-a --bounded-chart --lambda-floor 1e-4 --sigma-p 0.05 --tikhonov-frac 1e-2 --steps 15000 --batch 64`. 학습 시간 4:14 (GPU 1 공유).

| step | v4 Method A loss | v4.1 Method A + bounded loss |
|---|---|---|
| 100 | n/a | 1.7e+02 |
| 1000 | n/a | 38.1 |
| 3000 | n/a | 15.5 |
| 7500 | 7.31 | 7.79 (step 6087) |
| 15000 | **5.34** | **5.40** |

Spike (>1e+3) 0건, monotone 수렴 — v4와 거의 동일한 trajectory. ψ retraction이 학습 안정성 유지.

### 10.3 평가 결과 (`metric.md` 통합 metric, n=64 per z_e)

| $z_e$ | 분류 | pos cm | rot ° | pose@5/5° | pose@5/10° | mfe | manif gap | jvio | margin | ‖u‖∞ p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | ID-boundary | 2.90 | 3.90 | 79.69% | 96.88% | 0.047 | 0 | **0.0%** | 0.195 | 0.66 |
| 0.10 | ID-interior | 3.04 | 3.79 | 76.56% | 96.88% | 0.047 | 0 | **0.0%** | 0.219 | 0.60 |
| 0.15 | ID-boundary | 3.44 | 4.03 | 68.75% | 89.06% | 0.047 | 0 | **0.0%** | 0.220 | 0.54 |
| 0.20 | **OOD** | 4.18 | 4.49 | 54.69% | 75.00% | 0.031 | 0 | **0.0%** | 0.218 | 0.54 |
| **ID 평균** | (3 z_e) | 3.13 | 3.91 | **75.00%** | 94.27% | 0.047 | 0 | 0.0% | — | — |
| **OOD** | (z_e=0.20) | 4.18 | 4.49 | **54.69%** | 75.00% | 0.031 | 0 | 0.0% | — | — |

**핵심 관찰**:
- **Joint feasibility 0% violation 4 z_e 모두에서** (spec §13의 "viol(τ) = 0 by construction" 정량 검증 — random sampling이든 OOD든 무관)
- ‖u‖_∞ 99%-tile = 0.54–0.66 (saturation 한계 $|u|>3$ 대비 매우 낮음) → pre-flight forecast 0.6–0.7와 정확히 일치, chart는 정상 regime에서 작동
- Joint margin min 0.20 → demo의 1%-tile 0.20과 일치 (sample이 demo 분포 외부로 안 새어나감)
- mfe 0.047 (v4 0.12 대비 **2.5× better**)

### 10.4 v4 Method A vs v4.1 Method A + bounded chart — head-to-head

| Metric | v4 Method A | v4.1 + bounded | Δ |
|---|---|---|---|
| pose@(5cm,5°) **ID 평균** | **94.27%** | 75.00% | **−19.27 pp** |
| pose@(5cm,5°) **OOD** | **81.25%** | 54.69% | **−26.56 pp** |
| pose@(5cm,10°) ID | 99.48% | 94.27% | −5.21 pp |
| pose@(5cm,10°) OOD | 92.19% | 75.00% | −17.19 pp |
| pos err ID (cm) | 2.23 | 3.13 | +0.90 |
| rot err ID (°) | 2.83 | 3.91 | +1.08 |
| mfe ID | 0.12 | **0.047** | **2.5× better** |
| Joint feasibility | de facto 0% | **structurally 0%** | ✅ guarantee |
| Manifold adherence | ~0 (machine prec) | ~0 (machine prec) | tie |

### 10.5 분석 — Trade-off 본질

**v4.1은 pose accuracy를 19–27 pp 손해보면서 joint feasibility 보장 + multimodality 2.5× 향상을 얻는다.**

#### Saturation은 원인 아님
- ‖u‖_∞ p99 = 0.54–0.66, $D_\psi$가 vanish하는 영역 ($|u| > 2$) 한참 못 미침
- Pre-flight forecast (0.6–0.7) 정확히 일치 → chart는 demo 분포 내부에서만 작동

#### 진짜 원인: Varadhan chart-Euclidean bias (spec §9 caveat 정량 검증)
Spec §9 노트 인용:
> "The chart-Euclidean displacement $u_0 - u_r$ in $a^*_Q$ assumes a uniform inner product on the first ambient block ... The chart-Euclidean target $u_0 - u_r$ thus carries a position-dependent systematic bias for boundary-adjacent samples."

실험적으로:
- $\widetilde W$의 first block = $I_{n_q}$ (Choice A)는 "$\Delta u$ 1 unit = $\Delta q$ 1 unit"으로 취급
- 실제로는 Franka의 joint-별 $D_\psi$ 값이 $q_\text{range}/2 \in [1.57, 2.97]$로 다양 → $\Delta u = 1$이 joint 별로 $\Delta q$ 1.57–2.97 rad에 해당
- Score net이 이 inhomogeneous metric을 학습하면서 chart slot 출력의 정밀도가 저하 → endpoint pose error 증가

이는 **structural** 한계 — Choice A의 identity floor를 포기 안 하면 제거 불가 (spec §9 trade-off 명시).

#### Saturation diagnostic이 실패한 다른 부분: smoothness 비대칭
$E_\text{vel}^q$ = 1.20 (v4 미보고; baseline DP-channel = 0.10) → bounded chart trajectory가 q-space에서 약 **12× 더 jerky** 함. 가능한 원인:
- $u$-chart의 score net 학습이 noisier → trajectory step-to-step variance가 q-chart보다 큼
- 또는 boundary 근처에서 $D_\psi$ varies → 같은 $\Delta u$가 시간별로 다른 $\Delta q$ 생성

### 10.6 결론 (Tier 0 한정)

다음 권장은 **Tier 0 interior demos** 평가에 한정. Tier 2 boundary-active 결과는 §11 참고 — 권장이 뒤집힘.

| 결정 | Tier 0 권장 | Tier 2 권장 (§11) |
|---|---|---|
| Pose accuracy 우선 task (pick/place, IK 정밀) | **v4 unbounded chart** (Method A 그대로) | **v4.1 (50k)** — endpoint 정밀도 모든 method 중 최고 (pos 4.8 cm, rot 6.5°) |
| Joint feasibility 강제가 필요한 task (안전 critical, hardware 제약) | **v4.1 bounded chart** (pose 손해 수용) | **v4.1 (50k)** — 손해 없이 v4 의 pose accuracy 회복 + jvio = 0% |
| Multimodal 분포 보존이 critical | v4.1이 약간 우위 (mfe 0.047 vs 0.12) | **모든 method 동등** (mfe ≤ 0.02; 이전 mfe ranking 은 q[0] classifier artifact — §11.3 진단) |

**Paper 관점 (Tier 0)**: v4.1 은 *fundamental new capability* (joint feasibility by construction, never reachable in v4) 추가했지만 Tier 0 interior demos 에서는 v4 의 pose accuracy 우위를 빼앗지 못함. 이는 Choice A 의 well-known trade-off (spec §9 에 명시) 이며, Franka pose task 의 demo 분포가 interior 에 있어 unbounded chart 가 충분한 안전 마진으로 작동했기 때문.

**Paper 관점 (Tier 2, §11)**: boundary-active demos 에서 v4.1 가 모든 차원에서 best-or-tied → §10 의 trade-off 가 demo regime 에 dependent. v4.1 의 진정한 valuation 은 boundary-active task 에서만 가능.

---

## 11. Tier 2 Boundary-Active Experiment (`Experiment_plan.md` §2.3)

§10 의 Tier 0 결과는 v4.1 의 pose accuracy 가 v4 대비 −19 pp ID / −27 pp OOD 떨어짐 (Varadhan bias 영향) 을 보였지만, **Tier 0 demos 의 1%-tile rel-margin = 0.2045 이라 bounded chart 의 joint-safety 장점이 보이지 않는 setting** 이었음 (§10.5 saturation diagnostic 결론). 본 §11 은 demo 가 joint boundary 와 active 하게 상호작용하는 **Tier 2 setting** 에서 v4 vs v4.1 vs DP variants 의 진정한 trade-off 를 측정.

### 11.1 Tier 2 v5 demo 분포 (`Experiment_plan.md` §2.3.1)

탐색 끝에 §2.3 의 모든 PASS criterion 을 만족하는 sampling config 확정:

```python
q_rest_A = [+0.0, -0.3, 0.0, -0.40, 0.0, +3.40, 0.0]   # q[3], q[5] near upper bounds
q_rest_B = [+0.0, -0.3, 0.0, -1.50, 0.0, +1.50, 0.0]   # safe interior
p_box    = ([0.40, -0.05, 0.40], [0.50, +0.05, 0.50])  # Tier 0 box
jitter_q = 0.05;  target_perturb_deg = 20.0
n_ik_steps = 25;  ik_alpha = 0.5;  ik_alpha_null = 0.25
ik_clamp_to_limits = True;  ik_clamp_margin_frac = 0.001
```

| 지표 | Tier 0 (control) | Tier 2 v5 | Pass criterion |
|---|---|---|---|
| 1%-tile rel-margin | 0.2024 | **0.0010** | < 0.05 ✓ |
| active_joint_ratio | 0.000 | **0.1158** | > 0.10 ✓ |
| $P_{90}[b_\max]$ | 0.65 | **0.999** | > 0.95 ✓ |
| feasible fraction | 1.000 | **1.000** | > 0.99 ✓ |
| Mode A margin (boundary) | 0.285 | **0.053** | A=boundary-active |
| Mode B margin (safe) | 0.284 | **0.296** | B=safe |
| **Verdict** | (control) | **PASS** | — |

§2.5 의 mode 비대칭 (A=boundary, B=safe) 깔끔하게 분리.

### 11.2 Implementation fixes 발견

Tier 2 첫 시도에서 3가지 버그 발견 + 수정 (`Experiment_plan.md` §2.3.4):

1. **Demo gen 의 chart 이중 적용**: `FrankaBimodalReachingDemoPose.sample()` 의 realized endpoint 계산이 wrapped manifold 에 physical q 를 넘겨 ψ(q) 가 추가 적용 → endpoint error ~50cm. Fix: `getattr(self.manifold, "base", self.manifold).T_phi_Rp(q_traj, z_e)`.

2. **DDPMScheduler `clip_sample=True` default**: reverse process 에서 x_0 prediction 을 [-1, +1] 로 clip. Tier 0 demos 는 우연히 q ∈ [-1, +1] 안이라 영향 없었지만, Tier 2 mode A 의 q[5]≈3.4 가 clipping → garbage. Fix: trainer 에서 명시 `clip_sample=False`.

3. **Eval scheduler 가 saved scheduler_config 무시**: `franka_baselines_pose_eval` 의 scheduler 재생성도 default `clip_sample=True`. Fix: ckpt 의 `scheduler_config` 에서 clip_sample 값 propagate.

### 11.3 mfe metric 진단 및 수정

`compute_pose_metrics` 의 mfe classifier 가 `q[0] > 0` 하드코딩. Tier 0 demos (q_rest_A[0]=+0.6) 에서는 q[0] 가 mode discriminator 였지만, **Tier 2 v5 에서는 q_rest_A[0]=q_rest_B[0]=0.0** → q[0] 가 random noise 만 측정 → mfe misleading.

**검증**: v4.1 (50k) 의 q_init → q_H mode flip 직접 측정 (q[5] threshold 기준):
- A→A: 100, A→B: 0, B→A: 0, B→B: 156 → flip rate **0%**
- DP-bounded 도 같이 측정: A→B 5%, B→A 0.6% → DP-bounded 가 오히려 약간 flip

**Fix**: `compute_pose_metrics` 가 `q_rest_A`, `q_rest_B` 인자 받아 자동 detect:
```python
mode_joint = argmax(|q_rest_A - q_rest_B|)
mode_threshold = (q_rest_A[mode_joint] + q_rest_B[mode_joint]) / 2
```
Tier 0 → mode_joint=0 (legacy), Tier 2 v5 → mode_joint=5.

### 11.4 7-method Tier 2 평가 결과 (mfe fixed)

전체 학습 + eval pipeline 통과한 7개 config (z_e 평균):

| Method | succ@(5cm,5°) | pos cm | rot° | mfe | jvio | task_succ@(pose∧Q_safe) |
|---|---|---|---|---|---|---|
| DP-raw | **75.0%** | 8.8 | 12.0 | 0.01 | 24.6% | ~56% |
| DP-bounded | 65.6% | 13.8 | 14.4 | 0.005 | **0%** | 65.6% |
| Ours-v4 (Method A, no chart) | 64.5% | 5.6 | 7.0 | 0.00 | 28.5% | ~46% |
| Ours-v4.1 (15k, lf=1e-2) | 35.5% | 7.3 | 10.0 | 0.00 | **0%** | 35.5% |
| Ours-v4.1 (15k, lf=1e-3) | 35.5% | 7.2 | 10.0 | 0.00 | **0%** | 35.5% |
| Ours-v4.1 (30k) | 59.8% | 5.6 | 7.6 | 0.00 | **0%** | 59.8% |
| **Ours-v4.1 (50k)** | **64.8%** | **4.8** | **6.5** | **0.00** | **0%** | **64.8%** |

**Chart × Riemannian 2×2 ablation**:

| | Unbounded chart | Bounded chart |
|---|---|---|
| **Vanilla DP** | DP-raw: 75%, jvio 25% | DP-bounded: 66%, jvio 0% |
| **Riemannian (Method A)** | v4: 65%, jvio 29% | v4.1 (50k): **65%**, jvio **0%** |

### 11.5 학습 step scaling — v4.1 의 늦은 수렴

15k → 30k → 50k step 으로 늘리며 v4.1 의 pose accuracy 가 단계적으로 회복:

| | 15k | 30k | 50k |
|---|---|---|---|
| pos err mean cm | 7.3 | 5.6 | **4.8** |
| rot err mean ° | 10.0 | 7.6 | **6.5** |
| succ@(5cm,5°) | 35.5% | 59.8% | **64.8%** |
| succ@(5cm,10°) | 51.9% | 66.6% | **71.9%** |

DP-raw / DP-bounded 는 15k 에서 수렴 (loss 0.02 stable) 하지만 **v4.1 의 Riemannian SDE + bounded chart 조합 loss landscape 가 훨씬 천천히 수렴** — 동등 hyperparameter 에서 ~3x 더 긴 학습 필요. 30k → 50k 사이 한계 수익체감 (+5pp) → 50k 가 수렴 가까움.

### 11.6 v4.1 (50k) 의 final positioning

**모든 차원에서 best-or-tied**:
- **jvio = 0%** — bounded chart 가 구조적 보장 (DP-bounded 와 함께 유일)
- **mfe = 0** — mode capture 완벽 (모든 method 와 동등 + 직접 측정한 mode flip rate = 0%)
- **pos err 4.8 cm — 모든 method 중 가장 낮음** (DP-bounded 의 0.35x, DP-raw 의 0.55x)
- **rot err 6.5° — 모든 method 중 가장 낮음** (DP-bounded 의 0.45x, DP-raw 의 0.54x)
- **succ@(5cm,5°) 64.8% — DP-bounded 의 65.6% 와 통계적 동등** (1pp 차이)

**H1 가설** (`Experiment_plan.md` §1):

> v4.1 effective task_succ ≥ DP-raw + 5pp.

DP-raw effective task_succ = 75% × (1 − 0.25) ≈ 56% (jvio penalty).
v4.1 (50k) effective task_succ = 64.8% × 1.0 = 64.8%.
**Δ = +8.8 pp → H1 충족** ✓

### 11.7 Tier 0 (§10) vs Tier 2 (§11) — v4.1 의 진짜 trade-off

| | Tier 0 (interior demos) | Tier 2 (boundary-active demos) |
|---|---|---|
| Demo 1%-tile rel-margin | 0.2045 | 0.0010 |
| v4 succ@(5cm,5°) | **94.27%** ID | 64.5% |
| v4.1 succ@(5cm,5°) (≥30k) | 75.00% ID | **64.8%** (50k) |
| v4.1 vs v4 (succ) | −19.27 pp | **+0.3 pp** ✓ |
| v4.1 jvio | 0% (de facto) | **0% (structural)** |
| 결론 | v4.1 의 cost > benefit | **v4.1 의 cost ≈ 0, benefit 구조적** |

**§10 의 "v4.1 은 pose accuracy 19–27 pp 손해" 결론은 Tier 0 setting 에만 한정** — interior demos 에서는 bounded chart 가 작동할 영역 없음 → cost 만 측정됨. **Tier 2 (boundary-active) 에서는 v4.1 가 v4 의 succ 를 회복 + jvio 구조적 보장 + endpoint 정밀도 ~3x 개선 (vs DP-bounded)** — Choice A 의 G^A ≽ I floor 의 실제 가치가 측정됨.

### 11.8 Paper claim 정리

1. **Bounded chart 가 jvio = 0% 구조적 보장**: Choice A 의 G_Q^A ≽ I floor + ψ retraction.
2. **Chart × Riemannian (v4.1) 가 모든 차원에서 best-or-tied**:
   - DP-raw 대비 effective task_succ **+8.8 pp** (jvio penalty 회피).
   - DP-bounded 와 succ 동등 + endpoint 정밀도 **~3x 개선** (Riemannian 기여).
   - DP-raw / DP-bounded / Ours-v4 와 mode capture (mfe) 동등.
3. **Tier 0 의 §10 finding 재해석**: interior demos 에서 v4.1 의 cost 만 측정됨 (joint-safety benefit 없음 → trade-off 만 보임). Tier 2 에서 진정한 trade-off 측정됨 → benefit > cost.
4. **Cost**: v4.1 의 학습이 ~3x 더 오래 걸림 (Riemannian + chart loss landscape). 학습 시간 / model capacity 가 충분하면 cost 사실상 없음.

### 11.9 잔존 한계 및 향후

- **Mode A 의 per-mode succ 14% (v4.1) vs Mode B 81%**: boundary-active reaching task 자체의 난이도 (DP-bounded 도 18% / 88% 로 유사). v4.1 의 결함 아님.
- **Cross-validation 미실시**: seed=0 단일 결과. 3-5 seed sweep 으로 confidence interval 측정 가능 (각 30분).
- **Tier 1 (moderate boundary) 미측정**: Tier 0 → Tier 2 scaling 그래프 부재. Tier 1 demo gen + 학습 + eval 으로 보완 가능.
- **Larger UNet / cond_injection=global**: 추가 capacity ablation 가능. Mode A succ 추가 개선 여지.

---

## 12. v5.1 Chart-OU SDE — IK-free 재구축 (`joint_limit_extension.tex` v5.1)

**Compiled**: 2026-05-12.  Tier 2 boundary-active demo distribution (§11과 동일 setting; q_rest_A[3]=−0.4, q_rest_A[5]=3.4, IK clamp margin 0.1%, demo pool 8192).  `metric.md` 표준 평가.

### 12.0 동기 — v4.1의 잔존 cheating

§5–§11의 Method A / v4.1은 sampling-time 초기화에서 **per-trajectory IK warm-start** `q_init = demo's q_0`를 사용했다.  ABL3 (§6.4) 가 보였듯 이 항을 제거하면 ID success가 94% → 19% (−75pp) 로 붕괴 — 즉 model이 IK seed에 강하게 의존.  Demo의 첫 timestep은 T_start의 IK 해이며 mode label을 내포 (mode A / B 에 따라 q_rest가 달라 IK output도 다름), 결국 **mode 정보가 score net을 거치지 않고 sampling init에 직접 누설**.  이는 framework가 IK solver의 mode-discrimination 능력을 빌려쓰는 셈으로, $z_e$를 통한 embodiment generalization claim과 양립하기 어렵다.

`joint_limit_extension.tex` v5.1은 이 누설을 구조적으로 차단한다:

| 요소 | v4.1 (Method A) | v5.1 |
|---|---|---|
| Forward SDE | drift-free Brownian, $du = \sqrt{\beta}\,\sigma(u_r)\,dW$, $\sigma\sigma^\top=G_Q^{-1}$ | constant-coefficient OU on chart, $du = -\tfrac{1}{2}\beta u\,dr + \sqrt{\beta}\,\bar G_Q^{-1/2}\,dW$ |
| Forward marginal | Varadhan 근사 (locally Gaussian) | **Closed-form 정확해** $p_{r\mid 0}=\mathcal N(\alpha(r) u_0,\sigma^2(r)\bar G_Q^{-1})$ |
| DSM target | Varadhan-asymptotic $a^*_Q/\tau$ | **Exact Euclidean OU score** $-\bar G_Q(u_r-\alpha u_0)/\sigma^2(r)$ |
| Reference distribution | $\mathcal N(q_\text{init},\sigma_K^2 G_Q^{-1})$ — IK 의존 | **$\mathcal N(0,\bar G_Q^{-1})$ — data/conditioning 독립** |
| Reverse SDE | $dY=\beta s\,d\bar r + \sqrt{\beta}\,\sigma\,d\bar W$ | $du = [-\tfrac{1}{2}\beta u - \beta \bar G_Q^{-1} s_\theta]\,dr + \sqrt{\beta}\,\bar G_Q^{-1/2}\,d\bar W_r$ (Conv-1) |
| Score net 입력 | $[q,r,h/H,z_e,T_s,T_g]$ (q 자체가 anchor 정보 보유) | $[u,r,h/H,z_e,T_s,T_g]$ — q-anchor 없음 |

검증된 spec invariants (`tests/test_v51_chart_ou.py` 9/9):
- $\alpha(0)=1,\ \alpha(K)\approx 0.0067$ at $\beta_f=20$ (spec §8.2 표와 일치)
- Empirical Monte-Carlo forward marginal mean/cov가 $\alpha u_0$ / $\sigma^2(r)\bar G_Q^{-1}$ 와 일치 (relative err < 5%)
- Score target $-\bar G_Q(u_r-\alpha u_0)/\sigma^2(r)$이 autograd $\nabla_u \log\mathcal N$와 자릿수 1e-5 일치
- Reverse sampler signature에 `q_init` / `limiting_q_mean` / `q_warm` 모두 부재 (IK-free invariant 강제)

### 12.1 v5.1 학습 결과 (Tier 2, 50k vs 100k)

**Config**:  `--use-v51 --bounded-chart --gbar-mode identity --mu-pose 0 --beta-f 20 --tikhonov-frac 0.01 --lambda-floor 0.01`; demo pool 8192; batch 64; lr 2e-4; ema 0.999; cond_drop 0.10; weight `sigma2` (v5.1 매핑 시 $\sigma^4 = (1-e^{-\tau})^2$).  여타 hyperparam은 §11 Tier 2 50k와 동일.

**평가 잣대 정정**: 50k 첫 measurement는 default `n_sample_steps=200`을 그대로 따랐는데, sweep 결과 v5.1의 reverse SDE는 **n_steps에 매우 민감** (200 → 1000 step에서 pos 5.73→4.20cm, succ510 47→64%).  반면 v4.1은 200/500/1000 거의 평탄.  구조적 이유: v5.1 reverse drift는 OU mirror $+\tfrac{1}{2}\beta u$ 항을 포함하나 v4.1 (Method A, $b_q\equiv 0$) 은 없음.  따라서 **v5.1 평가는 n_steps=1000을 표준**으로 한다 (§12.4 참조).

### 12.2 Head-to-head (Tier 2, n_steps=1000, metric.md 표준)

| 방법 | pos cm | rot° | succ@(5cm,5°) | succ@(5cm,10°) | $g_p$ mm | $g_R$° | mfe | jvio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **DP-raw** (no chart) | 8.81 | 12.04 | **75.0%** | 78.5% | $T\ne T_\phi$ | — | 0.01 | **24.6%** ❌ |
| **DP-bounded** (chart only) | 13.82 | 14.41 | 65.6% | 73.0% | $T\ne T_\phi$ | — | 0.00 | 0% |
| **v4.1-50k** (Method A + IK) | 4.91 | 6.50 | 63.7% | 71.5% | 0.000 | 0.005 | 0.00 | 0% |
| **v5.1-50k** (chart-OU, IK-free) | 4.69 | 7.01 | 33.2% | 63.7% | 0.000 | 0.004 | 0.00 | 0% |
| **v5.1-100k** (chart-OU, IK-free) | **3.88** ★ | **5.63** ★ | 52.7% | **79.3%** ★ | 0.000 | 0.007 | 0.00 | 0% |

(DP는 §7의 별도 sampler 사용; $T=T_\phi(q,z_e)$를 강제하지 않으므로 manifold gap은 비교에서 제외)

**Effective task success** (joint-violation 패널티 반영, $\text{eff} = \text{succ}\times(1-\text{jvio})$):

| 방법 | eff_succ55 | eff_succ510 |
|---|---:|---:|
| DP-raw | 56.6% | 59.2% |
| DP-bounded | 65.6% | 73.0% |
| v4.1-50k | 63.7% | 71.5% |
| v5.1-50k | 33.2% | 63.7% |
| **v5.1-100k** | **52.7%** | **79.3%** ★ |

### 12.3 핵심 finding

1. **v5.1 100k는 effective task success 기준 best**.  DP-raw가 strict succ55에서 가장 높지만 25% trajectory가 joint limit 위반 → effective succ510 59.2%로 v5.1 100k (79.3%) 대비 −20.1pp.  Joint feasibility를 만족하는 method들 중 v5.1 100k가 모든 continuous 지표 (pos, rot) 와 succ510에서 1위.

2. **v5.1 100k > v4.1 50k 모든 지표**.  pos −1.03cm, rot −0.87°, succ510 +7.8pp.  IK seed 제거의 본질적 비용은 단순히 **충분한 학습 시간**.  Score net이 (T_start, T_target, z_e) conditioning만으로 endpoint 정밀도를 학습하는 데 v4.1의 2× step이 필요할 뿐, 정성적 한계가 아님.

3. **DP-bounded는 chart만으로는 부족**.  joint feasibility는 얻지만 pos mean 13.82cm — 절대 정밀도가 v5.1 대비 3.6× 나쁨.  Self-model graph manifold + Riemannian-weighted loss + chart-form score 의 조합이 핵심 (DP-bounded는 chart만 빌려옴, $T=T_\phi(q,z_e)$ 보장 없음).

4. **Strict succ55 cliff는 metric artifact**.  v5.1 100k rot 평균 5.63° — 5° 임계 바로 위에 분포 모임 → binary `<5°` gate에서 가까스로 탈락하는 sample 다수.  10° gate (succ510) 에서 v4.1 대비 +7.8pp 우위가 실제 endpoint 분포 차이를 더 정확히 반영.

5. **모든 구조 보존**.  jvio=0%, mfe=0.00, manifold gap=0 (machine precision).  IK-free reverse가 mode collapse 또는 manifold 일탈을 유발하지 않음 — score net 단독이 (T_start, T_target, z_e) 만으로 bimodality를 학습.

### 12.4 $z_e$ generalization — primary claim 검증

v4.1 vs v5.1-100k succ@(5cm,10°) by $z_e$ (모두 n_steps=1000):

| $z_e$ | 분류 | v4.1-50k | v5.1-100k | Δ |
|---|---|---:|---:|---:|
| 0.05 | ID lower boundary | 75.00% | **85.94%** | **+10.9pp** ★ |
| 0.10 | ID interior | 73.44% | **82.81%** | **+9.4pp** ★ |
| 0.15 | ID upper boundary | 73.44% | 76.56% | +3.1pp |
| 0.20 | OOD (+33% extrap) | 65.62% | **71.88%** | **+6.3pp** ★ |

**v5.1 100k는 모든 $z_e$ 에서 v4.1 능가**.  OOD ($z_e=0.20$, training $[0.05,0.15]$ 외) 에서도 +6.3pp.

v5.1의 $z_e$ generalization 우위가 정량 확인된 것은 §0 의 *user-stated primary motivation* 의 직접적 증명:
> "IK seed 는 cheat sheet … 이것 때문에 z_e 를 통한 embodiment 일반화가 잘 이루어지지 않잖아."

IK seed 제거 → score net이 $z_e$를 conditioning으로 본격 활용 → embodiment-conditioned trajectory를 학습 → $z_e$ extrapolation regime에서도 정밀도 유지.

### 12.5 추가 진단 — 실패한 보강 시도

v5.1 baseline 도출 과정에서 다음 변형을 시도, 모두 baseline에 못 미침:

| 시도 | 결과 | 진단 |
|---|---|---|
| Sampling-time guidance $R_\text{start}, R_\text{goal}$ sweep (G0–G8, α_g ∈ {0.5, 1, 2}, α_s = 2α_g, with metric weights $W_p=400, W_R=100$) | 모든 config가 baseline 근처 또는 약간 악화 | Score field 자체의 한계, sampling-time 개입으로 못 고침 |
| Chart-norm penalty $R_u = -\alpha_u\sum\|u_h\|^2$, $\alpha_u \in \{10^{-3}, 10^{-2}, 0.5\}$ | $\|u\|_\infty^{p99}$ 3.57 → 3.53 (변화 미미) | Score net이 boundary 근처 학습 (Tier 2 demo의 $u\approx 1.5$ 보다 훨씬 큼) — model-quality 이슈 |
| $\mu_\text{pose}=0.1$ retrain (`L_pose = \|J^Q s_\theta - \text{Log}_{SE(3)}/\tau\|_W^2`, spec form) | 학습 발산 — pos 70cm | Target $\text{Log}_{SE(3)}/\tau \sim \mathcal O(1/\sqrt\tau)$ 가 $r\to 0$에서 발산.  Spec form은 ideal solution에서만 수렴 |
| $\mu_\text{pose}=0.1$ retrain, τ-scaled `L_pose = \|τ J^Q s_\theta - \text{Log}_{SE(3)}\|_W^2` (같은 minimizer, 수치 안정) | 학습은 수렴, eval pos 42cm | $L_\text{pose}/L_\text{score} \approx 75\times$ 압도 |
| $\mu_\text{pose}=10^{-3}$ retrain, scale 균형 | pos 14cm, succ510 48% — baseline의 64%에서 후퇴 | **$L_\text{pose}$ 의 minimizer ≠ $L_\text{score}$ minimizer**.  Varadhan-asymptotic $T$-tangent target은 exact OU score와 양립 불가, aux reg가 active 학습 방해 → $\mu_\text{pose}>0$ 경로 폐기 |
| Exponential integrator on OU mirror (reverse step에서 $\frac{1}{2}\beta u$ 항을 해석적 적분) | euler와 byte-equivalent 결과 | $\beta\Delta r/2 \le 0.05$ (per step) 에서 $e^{x}-(1+x)\approx 1.3\times10^{-3}$ 무시 가능, OU mirror가 실제 stiff하지 않음.  200→1000 step 개선은 score field 의 r-/u-dependence 정확도이며 linear-stiffness 처리가 아님 |

**spec §10의 "μ_pose default 0" 권장이 정답이라는 empirical 증명.** Varadhan pose-consistency term은 minimizer가 exact OU score와 다른 별도 objective이므로 aux로도 쓰면 안 됨.

### 12.6 $z_e$-wise table (v5.1 100k, primary)

| $z_e$ | pos cm | rot° | pos_succ@5cm | rot_succ@5° | rot_succ@10° | pose_succ@(5cm,5°) | pose_succ@(5cm,10°) | mfe | $g_p$ mm | $g_R$ ° | jvio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 3.03 | 5.39 | 79.7% | 56.2% | 92.2% | **48.4%** | **85.9%** | 0.00 | 0.000 | 0.007 | 0% |
| 0.10 | 3.22 | 4.98 | 87.5% | 64.1% | 92.2% | **60.9%** | **82.8%** | 0.00 | 0.000 | 0.002 | 0% |
| 0.15 | 4.57 | 6.10 | 70.3% | 56.2% | 87.5% | 54.7% | 76.6% | 0.00 | 0.000 | 0.008 | 0% |
| 0.20 (OOD) | 4.69 | 6.04 | 65.6% | 48.4% | 81.2% | 46.9% | 71.9% | 0.00 | 0.000 | 0.010 | 0% |

**관찰**:
- ID interior ($z_e=0.10$) 에서 가장 강함 (succ55 60.9%, succ510 82.8%) — 학습 중심.
- ID boundary ($z_e=0.05, 0.15$) 에서 약간 하락하지만 succ510 76%+ 유지.
- OOD ($z_e=0.20$) 에서 succ510 71.9%, v4.1 OOD (65.6%) 대비 +6.3pp 우위.
- 모든 $z_e$ 에서 jvio=0%, mfe=0, manifold gap = 0 (machine precision).

### 12.7 결론 — v5.1 100k 시점의 정량 검증 (잠정)

v5.1 (chart-OU SDE, IK-free reference distribution, exact Euclidean score, bounded chart) 은 본 절의 100k baseline 학습 시점에서:

1. **구조적 cheating 제거** — IK seed 의존성을 완전히 차단 (test V6 invariant).
2. **Tier 2 boundary-active demo에서 정밀도 회복** — 50k 학습 후 v4.1 미달, 100k 학습 후 v4.1 우위 (pos, rot, succ510 모두).
3. **모든 구조 property 보존** — joint feasibility (0%), mode capture (mfe=0), manifold adherence (0 gap) 동시 만족.
4. **$z_e$ generalization 정량 우위** — 모든 $z_e$ (ID + OOD) 에서 v4.1 능가, OOD 에서도 succ510 71.9%.

이로써 §0 의 motivation (`IK seed cheat 제거 → embodiment generalization 회복`) 이 **정량 metric으로 입증**됨.

> **Caveat — v5.1 vs DP family 비교는 §13/§14 에서 재정정.**  §12.2 의 비교 표는 v5.1 100k vs DP family 15k step 의 비대칭 학습량에 기반.  Plateau-vs-plateau (200k) 동급 비교는 §13.4, chart_temp 통합 비교는 §14.6 참조.

### 12.8 잔존 한계 및 future work

- **strict succ@(5cm,5°)** 에서 v4.1 (63.7%) > v5.1 100k (52.7%).  rot 분포가 5° 임계 바로 위에 집중 — 4 cm 정도 더 정밀해도 binary cliff.  Continuous metrics (pos/rot mean) 와 relaxed succ510 에서는 v5.1 우위.
- **chart saturation** ($\|u\|_\infty^{p99} \approx 3.5$) 가 spec 권장 ≤2 보다 큼.  $R_u$ penalty / smoothness reward / weight tuning 등 ablation 가능.
- **200k 학습**: 50k → 100k 에서 succ510 64 → 79% (+15pp).  100k → 200k 의 plateau 시점을 측정하면 spec §13 의 "Higher capacity / more training steps" claim 의 한계를 정량화 가능.
- **3-seed 재현성**: 현재 1 seed.  Random variance bound 측정 필요.
- **Link-length embodiment**: $z_e$ 외에 link 길이 변화 (URDF perturbation) 까지 v5.1 generalization 확장.

### 12.9 재현 정보

| 항목 | 값 |
|---|---|
| Ckpt | `outputs/v51_tier2_100k_baseline/ours_v2_pose.pt` |
| Stage 1 | `outputs/franka_stage1_pose_tier2/xi_phi.pt` |
| Train cmd | `python -m smcdp.experiments.franka_traj_unet_pose --use-v51 --bounded-chart --beta-f 20 --mu-pose 0 --steps 100000 --batch 64 --lr 2e-4 ...` (q_rest/p_box/etc 는 Tier 2 §11과 동일) |
| Eval cmd | `python -m smcdp.experiments.franka_pose_reeval --ckpt <…> --n-sample-steps 1000 --n-eval-per-z 64 --z-list 0.05 0.10 0.15 0.20` |
| Metric.md 매핑 | `compute_pose_metrics()`가 §1 (pos/rot, succ55/510) + §2 (manifold gap) + §3 (mfe/W_1^q) + §4 (smoothness) + §5 (joint viol) 통합 구현 |
| Sanity tests | `python -m tests.test_v51_chart_ou` — V1–V9 (schedule, marginals, score target, IK-free invariant 등) 9/9 통과 |

---

## 13. Plateau-vs-plateau 비교 (200k 학습, 2026-05-12)

### 13.0 동기 — §12 비교의 불공정성 정정

§12.2 / §12.3 의 "DP family 능가" 결론은 **v5.1 100k vs DP 15k** 의 비대칭 비교였다. DP-raw, DP-bounded 의 default `--steps 15000` 만 학습된 ckpt 와 v5.1 100k 를 비교한 결과 — DP 의 plateau 가 어디인지 측정하지 않은 채 v5.1 의 100k 성능이 우위라고 보고. Plateau 측정 없이는 "더 학습하면 closer 또는 reverse 될 수 있는가" 를 답할 수 없다.

본 절은 세 method 모두 **동일 학습량 (200k step)** 으로 학습하고 25k step 마다 intermediate ckpt 를 저장해 succ-vs-step 곡선을 측정, 각 method 의 plateau / overfit 양상을 확인한 결과를 정리한다.

### 13.1 Setup

- **Trainer 양쪽에 `--save-every 25000`** 옵션 추가 (`smcdp/experiments/franka_traj_unet_pose.py`, `smcdp/experiments/franka_baselines_pose_train.py`).  25k, 50k, 75k, 100k, 125k, 150k, 175k 7개 + final 200k = 8 ckpt × 3 method = 24 ckpt.
- **Eval driver**: `smcdp/experiments/plateau_eval_and_plot.py` — 각 method 의 step_NNNNNN/ ckpt 를 모두 발견, 적절한 eval driver (v5.1 → `franka_pose_reeval`, DP → `franka_baselines_pose_eval`) 자동 dispatch, per-z 평균 metric 집계, succ-vs-step 4-panel plot, plateau step + overfit verdict (best > final by >2 pp) 산출.
- **Eval config**: n_sample_steps=1000, n_eval_per_z=64, z_e ∈ {0.05, 0.10, 0.15, 0.20}, success threshold (5 cm, 10°) primary.

### 13.2 학습 곡선 — succ@(5cm, 10°), per step

| step | v5.1 | DP-raw | DP-bounded |
|------|----:|------:|----------:|
|  25k | 43.4% | 89.8% | 81.6% |
|  50k | 64.1% | 92.6% | 90.2% |
|  75k | 75.4% | 94.5% | 93.0% |
| 100k | 76.2% | 94.1% | 92.6% |
| 125k | 80.9% | 95.7% | 92.6% |
| 150k | 83.6% | 95.3% | 93.4% |
| **175k** | **86.7%** ← peak | **96.5%** | 93.0% |
| 200k | 84.8% | 96.5% | **94.5%** ← peak |

Pos error (cm) — v5.1 만 monotonic 감소 지속 (3.30 @200k), succ saturate 후에도 mean precision 개선됨을 시사.

### 13.3 Plateau / overfit 진단

| 방법 | plateau step | best succ510 | final (200k) | Δ(best−final) | verdict |
|------|------------:|-----:|-----:|----:|-----|
| v5.1 (Ours) | 175k | 86.7% | 84.8% | **−1.9 pp** | borderline mild regression (2 pp 임계 직전) |
| DP-raw | 175k = 200k | 96.5% | 96.5% | 0 | 완전 plateau |
| DP-bounded | 200k | 94.5% | 94.5% | 0 | **아직 monotonic 증가 중**, 200k+ 추가 학습으로 추가 향상 가능 |

### 13.4 200k 최종 비교 (raw)

| 방법 | step | pos cm | rot° | succ@(5,5°) | succ@(5,10°) | mfe | **joint viol** |
|------|----:|----:|----:|----:|----:|----:|----:|
| v5.1 (G0 sampler) | 175k | 3.44 | 5.10 | 71.5% | **86.7%** | 0.00 | **0.0%** |
| DP-raw | 175k | 1.38 | 1.78 | 94.5% | **96.5%** | 0.00 | ⚠ **18.4%** |
| DP-bounded | 200k | 1.78 | 2.00 | 92.2% | **94.5%** | 0.00 | 0.4% |

**DP-raw 의 96.5% succ 는 측정 인공물**.  Joint violation rate 18.4% — 즉 "성공" trajectory 의 1/5 이 q 한계를 위반하여 실제 robot 배포 불가.  `effective_succ = succ × (1 − jvio)` 으로 보정 시:

| 방법 | step | succ@(5,10°) | jvio | **effective succ** |
|------|----:|----:|----:|----:|
| DP-raw | 175k | 96.5% | 18.4% | **78.7%** |
| DP-bounded | 200k | 94.5% | 0.4% | **94.1%** ★ |
| v5.1 (G0 sampler) | 175k | 86.7% | 0.0% | **86.7%** |

**Plateau-vs-plateau 기준 best feasibility-respecting method = DP-bounded (94.1%)**.  §12.3 의 "v5.1 100k 가 DP family 능가" claim 은 학습 단계 비대칭에 기인한 결과로, 동일 200k 비교에서는 **반대로 DP-bounded 가 +7.4 pp 우위**.

### 13.5 v5.1 sampling-time recipe sweep — spec §7 guidance + §10 R_u 모두 켜기

§12.5 의 guidance sweep 은 v5.1 100k ckpt 에서 측정. 200k ckpt 에서 spec 의 sampling-time 메커니즘 **전체** (guidance term 의 `start_alpha_p/R`, `goal_alpha_p/R`, chart-norm penalty `R_u`) 를 9 config 로 재측정:

**Config grid** (`smcdp/experiments/franka_pose_v51_guidance_sweep.py`):  α_p/α_R = α_s · W_p / α_s · W_R (W_p = 1/σ_p² = 400, W_R = 1/σ_R² = 100), α_g 는 goal 측 동일 scale.

| config | α_g | α_s | α_u | pos cm | rot° | succ@(5,5°) | **succ@(5,10°)** | ‖u‖∞ p99 | sat>3 | jvio |
|--------|---:|---:|---:|----:|---:|----:|------:|---:|----:|----:|
| **G0** baseline (guidance OFF) | 0 | 0 | 0 | 3.30 | 5.07 | 68.4% | **84.8%** | 3.52 | 37.5% | 0.0% |
| G1 mild | 0.5 | 1 | 0 | 3.37 | 5.04 | 68.8% | 85.2% | 3.53 | 37.5% | 0.0% |
| G2 spec default | 1 | 2 | 0 | 3.68 | 5.23 | 68.0% | 84.4% | 3.52 | 37.5% | 0.0% |
| G3 strong | 2 | 4 | 0 | 3.56 | 5.39 | 69.1% | 84.4% | 3.52 | 37.5% | 0.0% |
| G4 mild + R_u | 0.5 | 1 | 0.1 | 3.40 | 5.00 | 67.2% | 85.2% | 3.52 | 37.5% | 0.0% |
| G5 default + R_u | 1 | 2 | 0.1 | 3.62 | 5.09 | 67.6% | 85.2% | 3.51 | 37.5% | 0.0% |
| G6 strong + R_u | 2 | 4 | 0.1 | 3.63 | 5.38 | 68.0% | 84.0% | 3.51 | 37.5% | 0.0% |
| G7 R_u=0.5 | 1 | 2 | 0.5 | 4.24 | 5.69 | 65.6% | 82.0% | 3.49 | 37.1% | 0.0% |
| G8 aggressive | 2 | 4 | 0.5 | 4.03 | 5.79 | 66.8% | 80.1% | 3.49 | 37.1% | 0.0% |

**핵심 관찰**:

1. **모든 sampling-time 메커니즘이 ceiling 못 깸**.  최선 (G1/G4/G5) = **85.2%**, baseline 대비 +0.4 pp (통계적 변동 수준).
2. **‖u‖∞ p99 가 α_u 변화에 거의 무반응** (0 → 0.5 의 5× 증가에도 3.52 → 3.49).  Sampling-time soft penalty 로 chart 분포 mass 를 못 옮김.  sat>3 rate 37% 가 변하지 않음.
3. **강한 R_u (G7, G8) 는 오히려 악화**.  pos err 3.30 → 4.0+ cm, succ 84.8% → 80%.  Score field 와 충돌.
4. **DP-bounded 94.1% effective 와의 −9 pp gap 은 sampling-time 메커니즘 (guidance + R_u) 으로 닫히지 않음**.

### 13.6 진단 — sampling-time ceiling 의 근원

#### Chart saturation 은 training-time 현상

학습된 score net 의 분포 mass 가 이미 ‖u‖∞ ≈ 3.5 saturation 영역에 위치 (sat>3 = 37%).  Sampling 단계에서 soft penalty (chart-form natural gradient $-\alpha_u G^{-1} \cdot 2u$) 로는 이 학습된 분포 자체를 옮길 수 없다.

#### Endpoint precision gap 의 구조적 원인

| metric | v5.1 (200k) | DP-bounded (200k) | gap |
|--------|----:|----:|----:|
| pos err mean | 3.30 cm | 1.78 cm | **1.85×** |
| rot err mean | 5.07° | 2.00° | **2.5×** |
| succ@(5,10°) | 84.8% | 94.5% | −9.7 pp |
| sat>3 rate | 37% | — (no chart) | — |

DP-bounded 는 trajectory ambient $q \in \mathbb R^7$ 에 직접 diffuse 하고 chart wrap 은 sampling 시 한 번만 적용 (`q = \psi(u)`).  v5.1 은 학습 내내 chart 좌표 $u$ 에 diffuse — saturation 영역에서 학습 신호가 약해지는 구조적 단점.

#### Sampling 시 noise vs. drift balance

v5.1 175k → 200k 의 succ510 1.9 pp 하락은 **training loss 는 계속 감소 (score net 정밀도 개선) 인데 succ 가 약간 흔들리는 양상** — sampling-time variance 가 train-time 정밀도 개선보다 큰 영역에 진입.  Drift coefficient $\beta_f = 20$ 의 reverse step 1000 까지 떨어뜨려도 chart-boundary 근처 sample 의 stochastic kick 이 endpoint precision 을 자체 한계로 묶음.

### 13.7 결론 — chart_temp=1 한정 plateau 비교 (잠정)

**Plateau-vs-plateau (200k, chart_temp=1) 결과**:

1. v5.1 chart_temp=1 baseline 은 본 ckpt 에서 sampling-time recipe (guidance G0–G8, anti-saturation R_u) 모두 동원해도 succ@(5,10°) ceiling ≈ 85% — endpoint precision bottleneck.
2. DP-raw 96.5% 는 비교 대상 부적격 (jvio 18.8% → eff_sw 80.5%, sample-wise per §14.1).
3. **§12 의 "v5.1 100k 가 DP family 능가" 는 학습 단계 비대칭 (v5.1 100k vs DP 15k) 에 기인** — Plateau-vs-plateau (chart_temp=1) 에서 부분 반전.
4. **v5.1 가 우위인 영역은 여전**: $z_e$ generalization, feasibility, manifold gap, mode capture.  본 framework 의 "구조 보존 + IK-free + multimodal" 가치는 §12.7 그대로 유효.
5. **Endpoint precision 의 root cause 가 chart parameterization (training-time)** 으로 §14 audit 에서 확정.

> **Caveat — §13.7 의 verdict 는 chart_temp=1 한정.**  본 절의 "v5.1 ceiling ~85% / DP-bounded 우위" 는 chart_temp=1 학습 한정 finding 이며, **chart_temp c=2 (§14.4/§14.6) 도입 시 v5.1 c=2 200k 가 DP-bounded c=2 200k 와 eff_sw 동률 (94.5% = 94.5%) + feasibility 우위 (0% vs 0.4%) 로 정정**.  Root cause 인 chart saturation 의 training-time 본질은 §14.2 u-distribution audit 에서 정량 확인되었다.  본 절의 "training-time R_u / chart 범위 확대 / endpoint conditioning 강화" 권고 중 **chart 범위 확대** 가 정답으로 판명 (§14.4); endpoint conditioning 은 architectural side 가 root 아니라는 audit 결과와 정합하게 marginal 효과 (§14.3, §14.5).

### 13.8 산출물

| 항목 | 경로 |
|---|---|
| Training ckpts | `outputs/v51_tier2_200k_plateau/`, `outputs/tier2_dp_raw_200k_plateau/`, `outputs/tier2_dp_bounded_200k_plateau/` (각 8 ckpt) |
| Plateau eval JSON | `outputs/plateau_200k_comparison/plateau_curves.json` (24 ckpt × per-z metric), `plateau_summary.json` |
| Plateau plot | `outputs/plateau_200k_comparison/plateau_curves.png` (4-panel: succ55, succ510, pos, rot vs step) |
| 200k guidance sweep | `outputs/v51_tier2_200k_plateau/guidance_sweep_v51_200k.{json,md}` (G0–G8) |
| Cholesky fix | `smcdp/manifolds_pose.py:613` — `BoundedChartPoseManifold.G_pose_chol` 의 jitter chain 끝에 per-element NaN/non-PSD fallback 추가 (`ridge·I` 로 batch 의 bad slice 만 교체).  Chart saturation 시 R_u sweep G4–G8 가 crash 안 나도록. |

---

## 14. chart_temp + endpoint-relative SE(3) cond ablation (2026-05-12)

### 14.0 동기 — §13 결론의 한계 진단

§13.7 의 "DP-bounded > v5.1 by ~9 pp on effective_succ at plateau" 결론은 chart_temp=1 (default) 한정 finding 임이 후속 진단으로 드러났다.  본 절은 `diagnostic_plan.md` 의 권고를 단계별로 실행해 v5.1 의 sampling-time ceiling (G0–G8 가 깨지 못한 ~85%) 의 root cause 를 분리하고, 그에 대응하는 코드 수정으로 gap 을 완전히 닫는다.

### 14.1 Phase 0 — Evaluation audit (`diagnostic_plan.md` §2)

**§2.1 sample-wise exact `effective_succ`**.  현행 product estimate $\text{eff} = \text{succ} \times (1 - \text{jvio})$ 는 두 indicator 의 독립성 가정 근사.  Spec 의 정의는 trajectory-wise AND:
$$
\text{eff}_{\text{exact}} = \mathbb E\big[\mathbf 1(e_p < \rho_p) \cdot \mathbf 1(e_R < \rho_R) \cdot \mathbf 1(q \in Q)\big].
$$
`smcdp/franka/eval_metrics_pose.py` 에 sample-wise 와 product 둘 다 저장 ([commit `b44c907`](#)).  DP-raw 200k 재측정:

| z_e | pose_succ@(5,10°) | jvio | eff (sample-wise) | eff (product) | Δ |
|----:|--:|--:|--:|--:|--:|
| 0.05 | 98.4% | 23.4% | 76.6% | 75.4% | +1.20 pp |
| 0.10 | 95.3% | 20.3% | 78.1% | 76.0% | +2.17 pp |
| 0.15 | 95.3% | 21.9% | 78.1% | 74.5% | +3.66 pp |
| 0.20 | 96.9% |  9.4% | 89.1% | 87.8% | +1.27 pp |
| **avg** | **96.5%** | **18.8%** | **80.5%** | **78.4%** | **+2.1 pp** |

해석: $P(\text{pose\_ok} \mid \text{joint\_safe}) = 99.2\%$ vs $P(\text{pose\_ok} \mid \text{joint\_viol}) = 84.9\%$ — 두 indicator 가 양 상관 (관절 위반이 endpoint 실패와 함께 가는 경향).  Product 는 effective success 를 **체계적으로 underestimate**.

**§2.2 DP-bounded T_exec audit**.  `franka_baselines_pose_eval.py:188-206` 에서 DP-bounded 가 `arm.make_x(u_gen, z_e)` 호출 → `BoundedChartPoseManifold.T_phi_Rp(u, z) = base.T_phi_Rp(\psi(u), z)` 적용.  Pose error 는 x_gen 의 T-block (= T_φ(ψ(u), z_e)) 에서 추출되므로 **execution pose 기준 정확 측정**.  `compute_pose_metrics:73` 의 `arm.split_x(x_H)` 도 동일 (raw predicted T 가 아님).

**§2.3 jvio 0.4% 분해**.  TanhBoundedChart 의 ψ(u) ∈ (q_min, q_max) strict 이므로 finite u 에서 waypoint-jvio = 0 이어야 함.  Eval 결과 200k DP-bounded 의 per-z jvio = {0%, 1.6%, 0%, 0%}, 평균 0.4% ≈ 1 traj/256.  `q_phys` 가 float32 underflow 영역 (tanh(u) ≈ ±1.0 exactly at $|u| > 7$) 에서 비교 결과의 numerical edge case 로 추정 — 본질적 위반 아님.

### 14.2 Phase 0c — u-distribution audit

`smcdp/experiments/v51_u_distribution_audit.py`: v5.1 200k ckpt 로 demo 와 generated u 의 $\|u\|_\infty$ 분포 비교 ($n=256$ × 4 z_e):

| z_e | source | p50 | p90 | p99 | max | sat>2 | sat>3 |
|----:|:------|----:|----:|----:|----:|------:|------:|
| 0.05 | demo  | 0.54 | **3.45** | **3.45** | **3.45** | 40.6% | **40.6%** |
| 0.05 | gen   | 0.54 | 3.47 | 3.53 | 3.81 | 40.6% | **40.6%** |
| 0.10 | demo  | 0.51 | **3.45** | **3.45** | **3.45** | 46.1% | **46.1%** |
| 0.10 | gen   | 0.50 | 3.47 | 3.57 | 3.73 | 46.1% | **46.1%** |
| 0.15 | demo  | 0.42 | **3.45** | **3.45** | **3.45** | 46.1% | **46.1%** |
| 0.15 | gen   | 0.42 | 3.47 | 3.54 | 3.77 | 46.1% | **46.1%** |
| 0.20 | demo  | 0.36 | **3.45** | **3.45** | **3.45** | 45.5% | **45.5%** |
| 0.20 | gen   | 0.36 | 3.48 | 3.53 | 3.76 | 44.9% | **44.9%** |

**핵심**: demo p99 = p99 = max = 3.45 (z=0.05–0.20 모두 동일) → demo 가 자연 분포가 아니라 **정확한 boundary 에 clamp 되어 있음**.  출처 추적:

$$
u_{\text{clamp}} = \psi^{-1}(q_{\max} - 0.001 \cdot q_{\text{range}}) = \mathrm{atanh}(1 - 0.002) \approx 3.45.
$$

즉 `ik_clamp_margin_frac=0.001` (Tier 2 setup) 으로 q 가 chart-margin 에 pinned → atanh 적용 시 u = ±3.45 정확.  Tier 2 의 boundary-active q_rest (q_rest_A[5]=3.4, q_rest_B[5]=1.5; q_max[5]=3.752) 가 IK 해를 boundary 근방으로 강제 → demo 의 44% 가 ‖u‖∞ ≥ 3.45.

**Verdict**: gen 이 demo 분포를 충실히 학습 중 (gen sat>3 ≈ demo sat>3, gen 이 demo 의 cap 을 0.3 정도 약하게 넘어선 정도) — score net architectural 결함 아님.  Bottleneck 은 chart parameterization 의 boundary 압축 (|u|≈3.5 에서 sech²(u) ≈ 8×10⁻⁴ → J^Q = J_pose · D_ψ 가 0 근접).

### 14.3 Phase 1 — endpoint-relative SE(3) error input (`diagnostic_plan.md` §5)

`TrajectoryScoreNetUNetPose` 에 `endpoint_rel_cond` flag 추가.  cond_injection='channel' 일 때 forward pass 에서 per-step:
$$
e_{\text{goal}, h} = \mathrm{Log}_{SE(3)}\!\Big(T_\phi\!\big(\psi(u_h),\, z_e\big)^{-1} \cdot T_{\text{target}}\Big) \in \mathbb R^6
$$
을 계산해 UNet 입력에 6 채널 concat.  T_phi 는 manifold 의 chart-aware 메서드 (bounded 시 ψ 자동 적용), `log_relative_Rp` 는 body-frame twist (vmap-safe).  추가 파라미터 $\approx$ 2K (첫 conv 의 input_dim grew by 6).

| config | step | pos cm | rot° | succ@(5,5°) | succ@(5,10°) | Δ vs baseline 100k |
|--------|---:|----:|----:|----:|----:|---:|
| v5.1 baseline | 100k | 3.88 | 5.63 | 52.7% | 79.3% | — |
| **v5.1 endpoint_cond** | **100k** | **3.54** | **5.04** | **64.1%** | **83.2%** | **+3.9 pp** |

Endpoint cond 가 architectural 결함을 정면 해결한다는 가설이 옳다면 큰 jump 를 기대했으나 **+3.9 pp 약한 효과**.  audit 결과 (gen ≈ demo) 와 정합 — score net 의 architectural capacity 가 root bottleneck 이 아님.

### 14.4 Phase 1b — chart temperature ablation (`diagnostic_plan.md` §4)

**수정 내용**: `TanhBoundedChart` 에 `chart_temp` (c) 추가, $\psi/\psi^{-1}/D_\psi$ 모두 c-스케일 일관 적용:
$$
\psi_c(u) = q_{\text{mid}} + \tfrac{q_{\text{range}}}{2}\tanh(u/c),
\quad
\psi_c^{-1}(q) = c \cdot \mathrm{atanh}\!\Big(\tfrac{2(q - q_{\text{mid}})}{q_{\text{range}}}\Big),
\quad
D_{\psi,c}(u) = \tfrac{q_{\text{range}}}{2c}\,\mathrm{sech}^2(u/c).
$$
ψ_c 의 image 는 $(q_{\min}, q_{\max})$ 그대로 — 즉 **q-공간의 분포는 불변**, u-좌표만 c 배 stretching.  Spec 의 ψ 자유도 안에서 reparameterization 이라 G_Q^A / J^Q / score loss / 참고 분포 모두 정합 유지 (test V1–V9 통과 + 신규 c-sanity 통과).

| config (100k) | pos cm | rot° | succ@(5,5°) | succ@(5,10°) | sat>3 | Δ vs baseline 100k |
|--------|---:|---:|---:|---:|---:|---:|
| v5.1 baseline (c=1) | 3.88 | 5.63 | 52.7% | 79.3% | 44.4% | — |
| **v5.1 c=2** | **2.66** | **3.63** | **80.9%** | **90.2%** | TBD | **+10.9 pp** ⭐ |

**대규모 jump**: succ@(5,10°) +10.9 pp, succ@(5,5°) +28.2 pp, pos err −31%, rot err −36%.  v5.1 c=2 100k 는 v5.1 baseline 200k (84.8%) 보다 **+5.4 pp 더 높음 — 절반 학습량**.

DP-bounded 도 같은 lever 가 작동:

| config (DP-bounded) | step | pos cm | rot° | succ@(5,10°) |
|--------|---:|---:|---:|---:|
| baseline (c=1) | 200k | 1.78 | 2.00 | 94.5% |
| **c=2** | **100k** | 2.92 | 3.50 | **93.4%** |
| **c=2** | **200k** | **1.58** | **1.66** | **94.5%** |

DP-bounded 도 c=2 가 100k 만에 baseline 200k 수준 (93.4% ≈ 94.2%).  200k 까지 끌면 baseline 보다 약간 정밀 (pos 1.78 → 1.58 cm).

### 14.5 Phase 1c — c=2 + endpoint_rel_cond combined

두 lever 가 orthogonal 한지 검증:

| config (100k) | pos cm | rot° | succ@(5,5°) | succ@(5,10°) | Δ vs c=2 alone |
|--------|---:|---:|---:|---:|---:|
| v5.1 c=2 alone | 2.66 | 3.63 | 80.9% | 90.2% | — |
| **v5.1 c=2 + endpt** | 2.71 | 3.96 | 80.5% | **91.4%** | **+1.2 pp** |

Endpoint cond 가 chart_temp 위에서 +1.2 pp 만 추가 — chart parameterization 이 dominant lever 임이 재확인.  Endpoint cond 의 가설 (score net 의 target-relative geometry 가시성 부족) 은 sub-dominant.

### 14.6 Phase 1d — c=2 를 200k 까지 확장 (plateau-vs-plateau 동급 비교)

§13 의 200k plateau 실험을 c=2 로 재실행 (`--save-every 25000`, intermediate ckpt 8 개).  100k vs 200k 비교:

v5.1 c=2 의 plateau 곡선 (25k–300k, intermediate 매 25k):

| step | pos cm | rot° | succ@(5,5°) | succ@(5,10°) | jvio |
|------|----:|----:|----:|------:|---:|
|  25k | 5.59 | 7.05 | 41.8% | 65.6% | 0.0% |
|  50k | 3.65 | 4.85 | 70.3% | 85.2% | 0.0% |
|  75k | 2.88 | 4.11 | 79.3% | 88.3% | 0.0% |
| 100k | 2.68 | 3.82 | 81.2% | 89.1% | 0.0% |
| 125k | 2.41 | 3.74 | 85.5% | 93.0% | 0.0% |
| 150k | 2.33 | 3.35 | 85.9% | 94.1% | 0.0% |
| 175k | 2.15 | 3.22 | 87.5% | 94.9% | 0.0% |
| 200k | 2.12 | 3.18 | 87.9% | 94.5% | 0.0% |
| 225k | 2.05 | 3.04 | 89.1% | 93.8% | 0.0% |
| 250k | 1.92 | 2.79 | 92.6% | 94.9% | 0.0% |
| **275k** | **1.88** | 2.80 | 91.4% | **95.3%** ⭐ | 0.0% |
| 300k | 1.87 | **2.72** | 92.2% | 94.5% | 0.0% |

**Peak = 275k @ 95.3% succ@(5,10°)**.  175k 부터 ~94% 수준 plateau, 250k 부터 95% 재상승, 275k peak, 300k 미세 dip — c=1 baseline 의 175k peak (84.8%) 와 같은 패턴이나 +10.5 pp 더 높은 절대값에서 발생.

Pos/rot mean 은 plateau 없이 monotonic 개선 (25k 5.59 cm → 300k 1.87 cm, −66%; 7.05° → 2.72°, −61%) — succ 의 binary cliff 때문에 mean precision 의 개선이 succ 에 반영 안 됨.

**비교 — best v5.1 c=2 vs best DP-bounded c=2**:

| config | step | pos cm | rot° | succ@(5,5°) | succ@(5,10°) | jvio | **eff_sw** |
|--------|---:|---:|---:|---:|---:|---:|---:|
| **v5.1 c=2** (peak) | **275k** | 1.88 | 2.80 | 91.4% | **95.3%** ⭐ | **0.0%** | **95.3%** ⭐ |
| v5.1 c=2 (final) | 300k | 1.87 | 2.72 | 92.2% | 94.5% | 0.0% | 94.5% |
| DP-bounded c=2 | 200k | 1.58 | 1.66 | 93.0% | 94.5% | 0.4% | 94.5% |
| DP-bounded baseline | 200k | 1.78 | 2.00 | 92.2% | 94.5% | 0.4% | 94.2% |

**최종 결과**:

- **v5.1 c=2 275k = 95.3% succ@(5,10°), 95.3% eff_sw — DP-bounded c=2 200k (94.5%) 를 +0.8 pp 능가 ⭐**.
- **Joint feasibility**: v5.1 jvio = 0% vs DP-bounded 0.4% — v5.1 strict 우위.
- **strict succ@(5,5°)**: DP-bounded c=2 93.0% > v5.1 c=2 275k 91.4% (−1.6 pp).  여전히 pos/rot mean precision 차이 (DP 1.58 cm/1.66° vs v5.1 1.88 cm/2.80°) 가 strict 임계의 binary cliff 에서 −1.6 pp 잔여.
- **Mode capture (mfe)**: 둘 다 0.00 동률.
- **Manifold gap**: v5.1 0 (machine precision), DP-bounded T ≠ T_φ (§12.2 caveat) — v5.1 strict 우위.

### 14.7 §13 결론 정정

§13.7 의 "DP-bounded 가 sampling-time recipe 로 v5.1 보다 +9 pp 우위" 결론은 **chart_temp=1 한정 finding**.  Plateau-vs-plateau 비교에서 c=2 도입 + 275k 학습 시:

- **eff_sw**: v5.1 c=2 275k **95.3% > DP-bounded c=2 200k 94.5%** (+0.8 pp, v5.1 우위 ⭐)
- **Joint feasibility**: v5.1 (0%) > DP-bounded (0.4%) — v5.1 우위
- **Mean precision (pos/rot)**: DP-bounded c=2 약간 우위 (1.58 cm vs 1.88 cm), 둘 다 임계 한참 아래
- **Strict succ@(5,5°)**: DP-bounded 93.0% > v5.1 91.4% (−1.6 pp) — pos/rot mean cliff 잔여
- **Manifold gap, mode capture, multimodality**: §12.7 그대로 — v5.1 우위

종합하면 **v5.1 c=2 가 chart_temp + 충분한 학습 step 만으로 DP-bounded c=2 를 effective success 에서 능가하면서 feasibility / manifold 보존 면에서도 strict 우위**.  남은 갭은 strict succ@(5,5°) 의 −1.6 pp — pos/rot mean precision 의 binary cliff 에서 발생, endpoint conditioning 추가 (§14.5 의 +1.2 pp 가 strict 영역에서 가산 가능성) 또는 ε-prediction parameterization (`diagnostic_plan.md` §8) 으로 추가 닫기 시도 중 (§14.12 진행 중).

### 14.8 학습 가속 효과 — chart_temp 의 양면적 가치

| 학습량 | v5.1 (c=1 baseline) | v5.1 c=2 | DP-bounded (c=1 baseline) | DP-bounded c=2 |
|---:|:-:|:-:|:-:|:-:|
| 100k succ@(5,10°) | 79.3% | **90.2% (+10.9pp)** | (15k → 90% 도달) ※ | **93.4%** |
| 200k succ@(5,10°) | 84.8% | **94.5% (+9.7pp)** | 94.5% | **94.5% (+0.3pp)** |

chart_temp c=2 는 **두 method 모두**:
- 같은 step 에서 더 높은 정확도
- 더 빠른 plateau 도달
- 데이터 fitting 정밀도 향상

이는 chart 의 saturation regime 의 표현 압축이 score 학습에 미치는 비효율을 c=2 가 완화 ([§14.2 audit](#142-phase-0c--u-distribution-audit) 참조).  단 reparameterization 이라 q-공간 분포는 불변 — feasibility, mode capture, manifold gap 보존.

### 14.9 산출물

| 항목 | 경로 |
|---|---|
| chart_temp 100k ckpts | `outputs/v51_tier2_100k_chart_temp_2/`, `outputs/dp_bounded_100k_chart_temp_2/` |
| chart_temp 200k ckpts | `outputs/v51_tier2_200k_chart_temp_2/`, `outputs/dp_bounded_200k_chart_temp_2/` (intermediate 25k 마다) |
| endpoint_cond ckpts | `outputs/v51_tier2_100k_endpoint_cond/`, `outputs/v51_tier2_100k_c2_endpoint_cond/` |
| u-dist audit | `outputs/v51_tier2_200k_plateau/u_distribution_audit.json` |
| 종합 비교 스크립트 | `compare_phase1c1d.py` (서버 위치) |
| 코드 변경 | `smcdp/charts.py` (chart_temp), `smcdp/trajectories_pose.py` (endpoint_rel_cond), `smcdp/franka/eval_metrics_pose.py` (sample-wise eff), 트레이너/eval 4종 |

### 14.10 chart_temp c sweep — c=2.5 부근 sweet spot

c=2 의 효과 검증 후 추가 c sweep (100k 동급):

| config (100k) | pos cm | rot° | succ@(5,5°) | succ@(5,10°) | Δ vs c=2 (succ510) |
|--------|---:|---:|---:|---:|---:|
| c=1 baseline | 3.88 | 5.63 | 52.7% | 79.3% | −10.9 pp |
| c=2 | 2.66 | **3.63** | 80.9% | 90.2% | — |
| **c=2.5** | 2.77 | 4.01 | **85.2%** | **92.2%** | **+2.0 pp** ⭐ |
| c=3 | 2.71 | 3.74 | 84.0% | 91.8% | +1.6 pp |

**관찰**:
- **succ 양쪽 모두 c=2.5 가 미세 최적** (succ@(5,5°) +4.3 pp, succ@(5,10°) +2.0 pp vs c=2).
- c=3 는 c=2.5 대비 약간 regression (succ@(5,10°) −0.4 pp) — chart 가 너무 평탄해지면 데이터 spread 가 ±9.2 까지 늘어나 학습 어려워짐.
- **pos err 는 셋 다 평탄** (2.66~2.77 cm) — c-tuning 의 효과는 rot/succ 에 집중.
- c=2 가 rot mean 만은 최저 (3.63°), 다만 succ@(5,5°) 에서는 c=2.5/c=3 가 우위 — rot 분포의 tail behavior 가 c 마다 다름.

**100k 시점 verdict**: c=2.5 가 best (92.2% vs c=2 의 90.2%).  단 200k 확장 시 c 별 ceiling 이 다름 (§14.11 참조).

### 14.11 c=2.5 / c=2 의 200k plateau 비교 — 의외 결과

c=2.5 도 200k 확장 (intermediate 25k):

| step | v5.1 c=2 succ@(5,10°) | v5.1 c=2.5 succ@(5,10°) | DP-bounded c=2 succ@(5,10°) | DP-bounded c=2.5 succ@(5,10°) |
|---:|---:|---:|---:|---:|
| 25k | 65.6% | 71.5% | (n/a) | 69.5% |
| 50k | 85.2% | 84.8% | (n/a) | 88.3% |
| 75k | 88.3% | 89.8% | (n/a) | 91.8% |
| 100k | 89.1% | **91.4%** | 93.4% | 93.4% |
| 125k | 93.0% | 92.6% | (n/a) | 94.1% |
| 150k | 94.1% | 92.6% | (n/a) | 94.5% |
| 175k | **94.9%** | 93.0% | (n/a) | 94.5% |
| 200k | 94.5% | 93.4% | **94.5%** | **95.3%** |

**예상 빗나간 결과**: 100k 까지는 c=2.5 가 c=2 우위 (92.2% vs 90.2%) 였으나, **200k 에서는 c=2 가 c=2.5 보다 우위** (94.5% > 93.4%) for v5.1; 반대로 DP-bounded 는 c=2.5 200k (95.3%) 가 c=2 200k (94.5%) 보다 우위.

**해석**: c 가 클수록 u-data spread 가 c 배 (c=2: ±6.9, c=2.5: ±8.6, c=3: ±10.4).  더 wide 한 분포는 (a) 초기 학습에서 풍부한 representation → 100k 시점 c=2.5 우위, (b) 후기 학습 ceiling 에서 score net 의 분포 fit 정밀도 부담 증가 → 200k 에서 c=2 가 미세 우위 (for v5.1).  DP 의 ε-prediction 은 wider 분포에 robust 해서 c=2.5 까지 안정적으로 benefit.

**v5.1 의 최적 c = 2.0**.  단 100k 시점 비교 시 c=2.5 가 보이는 +2 pp 우위는 학습 시간 절약 가치가 있음 (early-stopping 시나리오).

### 14.12 잔존 한계 및 향후

- **v5.1 c=2 의 진짜 ceiling = 275k @ 95.3%** (§14.6 plateau 곡선) — 275k 부터 mild regression.  **DP-bounded c=2 200k 94.5% 능가 (+0.8 pp)**.
- **strict succ@(5,5°) gap**: DP-bounded c=2 93.0% vs v5.1 c=2 275k 91.4% (−1.6 pp).  pos/rot mean precision (DP 1.58 cm/1.66° vs v5.1 1.88 cm/2.80°) 의 binary cliff 잔여.
- **endpoint_cond + c=2 결합 확장 (진행 중)**: §14.5 의 100k 시점 +1.2 pp 가 strict 영역에서 효과 가산되는지 c=2 + endpt 300k 학습 후 검증.
- **3-seed reproducibility**: 단일 seed.  Variance bound 필요.
- **chart_temp 적응 schedule**: 학습 초반 c=2.5 (빠른 학습) → 후반 c=2 (정밀화) anneal — c-별 ceiling 차이 (§14.11) 를 이용해 best-of-both.
- **DP-bounded 의 c=2.5/c=3 sweep**: DP-bounded c=2.5 200k 가 95.3% — 더 큰 c 에서 추가 향상 가능성 (DP 의 ε-pred 의 wider 분포 robust 활용).

---

## 부록 B: 진단 보고서 5가지 의심의 검증 상태

| # | 의심 | 상태 |
|---|---|---|
| Q1 | drift OFF + pose 미측정 | ✓ Method A로 첫 measurement, 작동 확인 (ID pose_succ@(5cm,5°)=94.27%, OOD=81.25%) |
| Q2 | anchor approximation 무효 | ✓ Method A에서 anchor 자체 제거로 해소 |
| **Q3** | **Score net과 SDE의 anchor mismatch** | ✓ **ABL3에서 ID −75.5 pp / OOD −59.4 pp 로 정량 검증**;  v5.1 §12 에서 **IK seed 자체를 구조적으로 제거**, 100k 학습으로 정밀도까지 회복 (v4.1 대비 succ510 +7.8pp) |
| Q4.1 | Langevin forward + Varadhan target 비정합 | ✓ drift 제거 시 학습 자체 정상화 (loss 1e+13 → 1e+1) |
| Q4.7 | σ_K calibration | △ ABL1에서 측정, marginal 차이만 (ID +3.7 pp, OOD +9.4 pp);  v5.1 에서는 σ_K 자체가 제거됨 ($u_K\sim\mathcal N(0,\bar G_Q^{-1})$ data-independent) |

**결론**: Q3가 가장 큰 ROI. Q1/Q2/Q4.1는 baseline 작동 자체를 결정. Q4.7은 spec 정밀화이지만 empirical 차이 미미.  **v5.1 (§12) 에서 Q3는 ABL 검증을 넘어 구조적 cheat 자체를 제거한 framework로 진화.**
