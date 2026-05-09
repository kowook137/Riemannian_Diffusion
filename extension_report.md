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
  - **pose_succ@(5cm, 5°) = 95.31% (z_e=0.05), 91.0% 평균** (`metric.md` 기준 strict full-pose criterion)
  - pos_err 2.00 cm, rot_err 2.79° at z_e=0.05
  - manifold gap ≈ 0 (machine precision, by construction)
- 이전 시도 (anchor-metric Langevin drift 가정) 모두 실패 (succ 0%, q ~10⁵ rad 폭주). Method A (forward Brownian + condition-aware sampling init)으로 해결.
- **Per-trajectory $q^\text{init}$이 결정적 contribution**: ABL3 (제거 시) 91% → 19.5% (-71.5 pp) — Q3 anchor mismatch 진단의 정량 검증.

**진행 중** (2026-05-10): ABL2 (proxy_std=ou), pose-baseline 비교 (BC, DP-canonical, DP-channel, Projected) — `metric.md` 통합 metric. 결과 도착 순서대로 §6, §7에 누적 업데이트.

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

| $z_e$ | pos cm | rot ° | pose@5/5° | pose@5/10° | manif gap |
|---|---|---|---|---|---|
| 0.05 m | **2.00** | **2.79°** | **95.31%** | 100% | 0.000 mm / 0.005° |
| 0.10 m | 2.15 | 2.77° | **95.31%** | 100% | ~0 |
| 0.15 m | 2.53 | 2.93° | 92.19% | 98.44% | ~0 |
| 0.20 m | 3.24 | 3.15° | 81.25% | 92.19% | ~0 |

**평균** (4개 z_e): pos_err 2.48 cm, rot_err 2.91°, **pose_succ@(5cm,5°) = 91.0%**, manifold gap ≈ 0 (machine precision).

**(주의)**: 이전 보고된 "succ 51.6%"는 default `--success-pos 0.02` (2cm) threshold 때문에 *position-strict*로 측정됨. `metric.md` standard pose_succ@(5cm, 5°) 기준으로 재평가 시 **95.3%** (z_e=0.05). 표준 metric으로 재계산이 paper의 정확한 성능 수치.

### 5.3 Position-only V2 (REPORT Part III)와의 비교

| 항목 | Position-only V2 | Pose-extended Method A |
|---|---|---|
| Target dim | $p \in \mathbb{R}^3$ (3-dim) | $T \in \mathrm{SE}(3)$ (6-dim) |
| pos_err | 21 mm | 20–32 mm (z_e=0.05–0.20) |
| Success criterion | pos < 5 cm | pos < 5 cm AND rot < 5° (`metric.md` strict) |
| succ rate | ~95% | 81–95% (z_e=0.05–0.20) — **average 91%** |

Position 정밀도는 동등 ($\approx$ 2 cm). Rotation을 strict 5°로 추가 measure해도 succ rate 거의 유지 (z_e≤0.10에서 95%) — rotation precision (mean 2.79°)이 5° threshold 안에 안정적으로 들어감이 SE(3) extension의 강점.

---

## 6. Ablation — Method A 각 component의 기여 isolation

$\sigma_K$ / proxy_std mode / per-trajectory $q^\text{init}$ 셋이 *각각* 얼마나 critical한지 측정. 다른 모든 hyperparameter는 동일.

### 6.1 Ablation 결과 표 (`metric.md` 통합 metric, ABL2 측정 중)

`pose_succ@(5cm, 5°)` 기준 (strict full-pose criterion).

| 설정 | $\sigma_K$ | proxy_std | $q^\text{init}$ | z_e=0.05 | z_e=0.10 | z_e=0.15 | z_e=0.20 | **평균** |
|---|---|---|---|---|---|---|---|---|
| **Method A (full)** | 1.414 | brownian | per-traj | **95.31%** | 95.31% | 92.19% | 81.25% | **91.01%** |
| ABL1 (σ_K=0.6) | 0.6 | brownian | per-traj | 96.88% | 98.44% | 98.44% | 90.62% | 96.10% |
| **ABL3 (q_init=μ_q)** | 1.414 | brownian | **single μ_q** | **17.19%** | 20.31% | 18.75% | 21.88% | **19.53%** |
| ABL2 (proxy_std=ou) | 1.414 | **ou** | per-traj | 89.06% | 90.62% | 87.50% | 75.00% | 85.55% |

### 6.2 Component-wise contribution

**ABL3 (per-trajectory q_init 제거)** — **−78.1 pp at z_e=0.05** (95.31% → 17.19%):
- 모든 z_e에서 Method A 대비 ~70-80 pp 약화 (평균 91.01% → 19.53%, **−71.5 pp**)
- rot_err: Method A 2.79° → ABL3 9.95° (3.6× 악화); pos_err 2.00 → 4.39 cm (2.2× 악화)
- 통계적 매우 견고 (95% CI ±12 pp을 6배 이상 초과)
- → **Per-trajectory $q^\text{init}$이 Method A의 결정적 core contribution**. 진단의 Q3 (anchor mismatch) 가설 정량 검증.

**ABL1 (σ_K=0.6 legacy)** — **+1.6 pp at z_e=0.05** (95.31% → 96.88%, 평균 +5 pp):
- 모든 z_e에서 Method A보다 약간 더 나음
- 통계적 marginal — 95% CI ±12 pp 이내
- σ_K calibration은 *spec 정밀화*로 옳지만 *empirical 효과 marginal* — robustness signal
- Forward marginal과 정합한 σ_K=1.414 vs legacy 0.6이 비슷하거나 약간 더 narrow init이 유리 — score net의 effective coverage 영역과 관련 (paper에서 추가 분석 가능)

**ABL2 (proxy_std = ou)** — **−5.46 pp 평균** (91.01% → 85.55%):
- 모든 z_e에서 약간씩 약화 (−6.25 pp at z_e=0.05, −6.25 pp at z_e=0.20)
- 통계적으로 약하지만 consistent (모든 4개 z_e에서 monotone negative)
- proxy_std는 학습 시 std_trick scale + loss weight $w(r) = \sigma^2(r)$에 들어감 → drift-free Brownian forward에서 OU std로 normalize 시 큰 r 영역 weight가 saturated → 큰 r에 학습 capacity 부족 → reverse SDE 큰 r 단계에서 score quality 저하
- → proxy_std calibration 효과는 marginal하지만 **consistent하게 positive** (Brownian이 정합)

### 6.3 Component contribution ranking (`metric.md` strict criterion)

| 순위 | Component | 평균 영향 | 근거 |
|---|---|---|---|
| **1** | **Per-trajectory $q^\text{init}$** | **−71.5 pp** | ABL3에서 결정적 collapse (95% → 19%) |
| 2 | proxy_std = brownian mode | +5.5 pp | ABL2 모든 z_e에서 consistent negative |
| 3 | σ_K = √τ_brown(K) | −5.1 pp | ABL1에서 약간 더 좋음, robustness signal |

(2,3은 통계적 noise 범위, 1만 명확한 signal)

---

## 7. Baseline 비교 (`metric.md` 통합 metric)

Pose-extended baseline을 동일 framework에서 학습하여 Ours-V2 (Method A) 대비 우위 입증.

| Baseline | 설명 | 상태 |
|---|---|---|
| BC | Deterministic regressor $c \to q$-trajectory | ✓ 완료 |
| DP-canonical (global cond) | Standard Diffusion Policy (Chi23) | 진행 중 (GPU 0) |
| DP-channel | DP-A variant (channel-concat cond, parity with Ours) | 대기 (GPU 1) |
| Projected | Ambient (q, T_storage) DP + projection via $H_\phi^\text{pose}$ | 대기 (GPU 1) |

**조건**: 동일 conditioning $c = (T_\text{start} \oplus T_\text{target} \oplus z_e) \in \mathbb{R}^{15}$, 동일 demo 분포, H+1=16, batch=64, steps=15000.

### 7.1 BC (Behavioral Cloning) 결과

| $z_e$ | pos cm | rot ° | pose@5/5° | pose@5/10° | mode frac err | manif gap | jvio |
|---|---|---|---|---|---|---|---|
| 0.05 | 2.00 | 3.09 | **95.31%** | 100% | **0.47** | 0 | 0% |
| 0.10 | 1.94 | 2.63 | **100%** | 100% | 0.47 | 0 | 0% |
| 0.15 | 2.40 | 2.64 | 98.44% | 100% | 0.47 | 0 | 0% |
| 0.20 | 3.62 | 3.05 | 85.94% | 87.5% | 0.47 | 0 | 0% |
| **평균** | **2.49** | **2.85** | **94.92%** | **96.88%** | **0.47** | **0** | **0%** |

학습 시간 ~3시간, model 1.12 M params (UNet 10M의 1/9).

### 7.2 Method A vs BC — *pose accuracy 동등, multimodality 차이*

| Metric | **Method A** | **BC** | 차이 |
|---|---|---|---|
| pose_succ@(5cm,5°) avg | 91.01% | **94.92%** | BC +3.9 pp |
| pose_succ@(5cm,10°) avg | 97.66% | 96.88% | tie |
| pos_err mean (cm) | 2.48 | 2.49 | tie |
| rot_err mean (°) | 2.91 | 2.85 | tie |
| **mode_frac_err** | **0.12** | **0.47** | **Method A 4× better** |
| manifold gap (mm) | ~0 | ~0 | tie (둘 다 H_φ로 lift) |
| joint viol rate | 0% | 0% | tie |

**핵심 관찰**:
- **Pose accuracy**: BC가 *근소하게* 더 높음 (단일 seed noise 범위)
- **Multimodality**: Method A가 *4× 더 잘 보존*. BC의 mode_frac_err = 0.47은 demo bimodal balance (0.5)에서 한 mode로 ~97% collapse한 것.
- **Manifold adherence**: 둘 다 ≈ 0 (BC는 $q$만 출력 후 $H_\phi^\text{pose}$로 lift, Method A는 sample이 매니폴드 위 by construction)

→ Paper의 contribution은 "**pose accuracy가 우위**"가 아니라 "**같은 pose accuracy를 *bimodal distribution을 보존하면서* 달성**". `metric.md` §7.1이 예측한 BC의 mode collapse 약점이 정확히 정량 검증됨.

**남은 baseline**: DP-canonical (mode capture 능력 있음, 진짜 강한 비교), DP-channel (Ours architecture parity), Projected (manifold gap 차이 측정).

---

## 8. 알려진 한계

### 8.1 z_e robustness 약화

- z_e=0.20 (longest tool, large lever arm)에서 pose_succ@(5cm,5°) 81.25% — Method A의 z_e=0.05의 95% 대비 −14 pp
- relaxed threshold pose_succ@(5cm,10°)는 92.19%로 z_e robustness가 *rotation*에서 더 두드러짐
- 가능한 원인: tool 길이가 EE pose에 강한 dependency (특히 rotation rate), score net의 z_e conditioning이 이 dependency를 충분히 학습 못함
- 잠재적 해결: reward guidance (Stage 6'), z_e-conditional weighting, 또는 z_e dimension에 expanded encoding

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

## 부록 B: 진단 보고서 5가지 의심의 검증 상태

| # | 의심 | 상태 |
|---|---|---|
| Q1 | drift OFF + pose 미측정 | ✓ Method A로 첫 measurement, 작동 확인 (pose_succ@(5cm,5°)=91%) |
| Q2 | anchor approximation 무효 | ✓ Method A에서 anchor 자체 제거로 해소 |
| **Q3** | **Score net과 SDE의 anchor mismatch** | ✓ **ABL3에서 −71.5 pp 평균 (z_e=0.05에선 −78.1 pp) 로 정량 검증** |
| Q4.1 | Langevin forward + Varadhan target 비정합 | ✓ drift 제거 시 학습 자체 정상화 (loss 1e+13 → 1e+1) |
| Q4.7 | σ_K calibration | △ ABL1에서 측정, marginal 차이만 (robustness signal) |

**결론**: Q3가 가장 큰 ROI. Q1/Q2/Q4.1는 baseline 작동 자체를 결정. Q4.7은 spec 정밀화이지만 empirical 차이 미미.
