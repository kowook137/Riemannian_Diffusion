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

**주요 finding**:
1. **Pose accuracy ranking** (ID 평균): DP-channel (100%) ≈ Projected (100%) ≈ DP-canonical (99%) > BC (98%) > **Ours-V2 Method A (94%)** ≫ ABL3 (19%). DP variants가 pose accuracy에서 Method A보다 우위.
2. **Multimodality 보존 (mfe)**: DP-canonical 0.02, DP-channel 0.025, Projected 0.04, **Method A 0.12**, BC 0.47 (collapse). DP가 mode collapse 방지에서 가장 우수.
3. **Method A의 남은 contribution**: chart-form score (drift-free Brownian + Varadhan target)이 *원리적으로 manifold-correct sampling*을 보장. Franka의 simple FK retraction 환경에서는 DP+post-projection으로도 비슷한 결과 가능. *복잡한 manifold (e.g. constraint manifold without closed-form retraction)*에서 차별화 가능성 — future work.

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

→ **Paper의 narrative 재구성 필요**: Method A는 Franka pose task에서 DP를 능가하지 못함. Method A의 이론적 강점 (chart-form score, drift-free Brownian on Riemannian manifold)이 *실제 우위로 발현되려면* DP의 post-projection으로 안 되는 환경이 필요함. 예:
- closed-form retraction이 없는 implicit constraint manifold (e.g. SE(3) sub-manifold from contact constraints)
- 동시-equality 제약 (contact + joint limit) 등에서 chart 기반 sampling이 필수
- Higher-dim multi-body system에서 retraction이 expensive
→ Franka 7-DoF + simple tool-z FK는 그러한 환경이 *아니므로* DP variant들이 잘 작동.

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

### 10.6 결론

| 결정 | 권장 |
|---|---|
| Pose accuracy 우선 task (pick/place, IK 정밀) | **v4 unbounded chart** (Method A 그대로) |
| Joint feasibility 강제가 필요한 task (안전 critical, hardware 제약) | **v4.1 bounded chart** (이 형식 채택, pose accuracy 손해 수용) |
| Multimodal 분포 보존이 critical | v4.1이 약간 우위 (mfe 0.047 vs 0.12) |

**Paper 관점**: v4.1은 *fundamental new capability* (joint feasibility by construction, never reachable in v4) 추가했지만 v4의 pose accuracy 우위를 빼앗지 못함. 이는 Choice A의 well-known trade-off (spec §9에 명시)이며, Franka pose task의 demo 분포가 충분히 interior에 있기에 unbounded chart가 충분한 안전 마진으로 작동하기 때문.

---

## 부록 B: 진단 보고서 5가지 의심의 검증 상태

| # | 의심 | 상태 |
|---|---|---|
| Q1 | drift OFF + pose 미측정 | ✓ Method A로 첫 measurement, 작동 확인 (ID pose_succ@(5cm,5°)=94.27%, OOD=81.25%) |
| Q2 | anchor approximation 무효 | ✓ Method A에서 anchor 자체 제거로 해소 |
| **Q3** | **Score net과 SDE의 anchor mismatch** | ✓ **ABL3에서 ID −75.5 pp / OOD −59.4 pp 로 정량 검증** |
| Q4.1 | Langevin forward + Varadhan target 비정합 | ✓ drift 제거 시 학습 자체 정상화 (loss 1e+13 → 1e+1) |
| Q4.7 | σ_K calibration | △ ABL1에서 측정, marginal 차이만 (ID +3.7 pp, OOD +9.4 pp) |

**결론**: Q3가 가장 큰 ROI. Q1/Q2/Q4.1는 baseline 작동 자체를 결정. Q4.7은 spec 정밀화이지만 empirical 차이 미미.
