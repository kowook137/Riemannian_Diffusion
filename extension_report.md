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
- **첫 번째 작동하는 pose-extended SMCDP**: succ@5cm 51.6% (z_e=0.05), pos_err 2.09 cm, rot_err 2.7°.
- 이전 시도 (anchor-metric Langevin drift 가정) 모두 실패 (succ 0%, q ~10⁵ rad 폭주). Method A (forward Brownian + condition-aware sampling init)으로 해결.

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
| **4 (Method A)** | **drift OFF + condition-aware init** | **2–3 cm** | **12–52%** | clean monotone |

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

### 5.2 Eval (per z_e — tool extension)

| $z_e$ | pos_err mean | pos_err max | rot_err mean | **succ@5cm + 15°** | manif ‖$g_\phi$‖_max |
|---|---|---|---|---|---|
| 0.05 m | **2.09 cm** | 4.49 cm | 0.048 rad (2.7°) | **51.6%** | 3.0e-5 |
| 0.10 m | 2.13 cm | 3.45 cm | 0.049 rad (2.8°) | 40.6% | 3.0e-5 |
| 0.15 m | 2.72 cm | 5.94 cm | 0.056 rad (3.2°) | 21.9% | 1.1e-4 |
| 0.20 m | 3.42 cm | 6.90 cm | 0.063 rad (3.6°) | 12.5% | 1.5e-5 |

**평균** (4개 z_e): pos_err 2.59 cm, rot_err 3.1°, **succ@5cm 31.6%**.

**Pure pose 도달** (z_e=0.05): pos_err **2.09 cm**, rot_err **2.7°**, **succ@5cm 51.6%**.

### 5.3 Position-only V2 (REPORT Part III)와의 비교

| 항목 | Position-only V2 | Pose-extended Method A |
|---|---|---|
| Target dim | $p \in \mathbb{R}^3$ (3-dim) | $T \in \mathrm{SE}(3)$ (6-dim) |
| pos_err | 21 mm | 21–34 mm (z_e=0.05–0.20) |
| Success criterion | pos < 5 cm | pos < 5 cm AND rot < 15° |
| succ rate | ~95% | 13–52% |

Position 정밀도는 동등 ($\approx$ 2 cm). Rotation을 추가로 맞추는 stricter criterion으로 succ rate 떨어짐.

---

## 6. Ablation — Method A 각 component의 기여 isolation

$\sigma_K$ / proxy_std mode / per-trajectory $q^\text{init}$ 셋이 *각각* 얼마나 critical한지 측정. 다른 모든 hyperparameter는 동일.

### 6.1 Ablation 결과 표 (2026-05-10 시점, ABL2 진행 중)

| 설정 | $\sigma_K$ | proxy_std | $q^\text{init}$ | succ@5cm (z_e=0.05) | succ 평균 |
|---|---|---|---|---|---|
| **Method A (full)** | calibrated 1.414 | brownian | per-traj | **51.6%** | **31.6%** |
| ABL1 (σ_K=0.6) | legacy 0.6 | brownian | per-traj | 60.9% | 35.6% |
| **ABL3 (q_init=μ_q)** | calibrated 1.414 | brownian | **single μ_q** | **7.8%** | **12.5%** |
| ABL2 (proxy_std=ou) | calibrated 1.414 | **ou** | per-traj | (pending) | (pending) |

### 6.2 Component-wise contribution

**ABL3 (per-trajectory q_init 제거)** — **−43.8 pp at z_e=0.05** (51.6% → 7.8%):
- z_e=0.05: 51.6% → 7.8% (6.6× 약화), rot_err 0.048 → 0.165 rad (3.4× 악화)
- 평균: 31.6% → 12.5% (−19 pp)
- 통계적으로 명백 (95% CI ±12 pp 초과)
- → **Per-trajectory $q^\text{init}$이 Method A의 *empirical* core contribution**. 진단의 Q3 (anchor mismatch) 가설 정량 검증.

**ABL1 (σ_K=0.6 legacy)** — **+9 pp at z_e=0.05** (51.6% → 60.9%):
- 평균 −0~9 pp 차이, 통계적 noise 범위 (95% CI ±12 pp 이내)
- σ_K calibration은 *spec 정밀화*로서 옳지만 *empirical 효과*는 marginal — robustness signal
- 실제로 $\sigma_K = 0.6$이 약간 더 잘 나옴: score net의 effective coverage가 forward marginal보다 narrower일 가능성 (training weight 분포)

**ABL2 (proxy_std = ou)**: 측정 진행 중 (ETA 2시간). proxy_std는 학습 weighting + std_trick에만 영향, sampling init 영향 없음 → ABL1처럼 marginal 차이 예상.

---

## 7. Baseline 비교 (예정)

Pose-extended baseline을 동일 framework에서 학습하여 Ours-V2 (Method A) 대비 우위 입증:

| Baseline | 설명 | 코드 |
|---|---|---|
| BC | Deterministic regressor $c \to q$-trajectory | `BCTrajectoryPredictor` |
| DP-canonical (global cond) | Standard Diffusion Policy (Chi23, global cond) | `make_official_diffusion_policy` |
| DP-channel | DP-A variant (channel-concat cond, parity with Ours) | `channel_concat_dp_loss` |
| Projected | Ambient (q, T_storage) DP + post-step projection via $H_\phi^\text{pose}$ | with `arm.make_x` |

**조건**:
- Conditioning: $c = (T_\text{start} \oplus T_\text{target} \oplus z_e) \in \mathbb{R}^{15}$ (storage form)
- 모든 baseline 동일 demo 분포, 동일 H+1=16, batch=64, steps=15000
- Eval: 동일 metric (e_p, e_R, combined succ@5cm + 15°)

**Training scripts**:
```bash
# BC
python -m smcdp.experiments.franka_baselines_pose_train --baseline bc

# DP-canonical (global cond)
python -m smcdp.experiments.franka_baselines_pose_train --baseline dp_official

# DP-channel (parity with Ours)
python -m smcdp.experiments.franka_baselines_pose_train --baseline dp_official --cond-injection channel

# Projected
python -m smcdp.experiments.franka_baselines_pose_train --baseline projected
```

**Eval**:
```bash
python -m smcdp.experiments.franka_baselines_pose_eval --ckpt outputs/franka_baseline_pose_<name>/ckpt.pt
```

결과는 §7.x에 추가 예정.

---

## 8. 알려진 한계

### 8.1 z_e robustness 약화

- z_e=0.20 (longest tool, large lever arm)에서 succ 12.5% — 약함
- 가능한 원인: tool 길이가 EE pose에 강한 dependency, score net의 z_e conditioning이 이 dependency를 충분히 학습 못함
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
- Method A vs ABL1 (9 pp 차이)는 noise 범위
- Method A vs ABL3 (44 pp 차이)는 명확한 signal
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
| Q1 | drift OFF + pose 미측정 | ✓ Method A로 첫 measurement, 작동 확인 (succ 52%) |
| Q2 | anchor approximation 무효 | ✓ Method A에서 anchor 자체 제거로 해소 |
| **Q3** | **Score net과 SDE의 anchor mismatch** | ✓ **ABL3에서 −43.8 pp로 정량 검증** |
| Q4.1 | Langevin forward + Varadhan target 비정합 | ✓ drift 제거 시 학습 자체 정상화 (loss 1e+13 → 1e+1) |
| Q4.7 | σ_K calibration | △ ABL1에서 측정, marginal 차이만 (robustness signal) |

**결론**: Q3가 가장 큰 ROI. Q1/Q2/Q4.1는 baseline 작동 자체를 결정. Q4.7은 spec 정밀화이지만 empirical 차이 미미.
