# SMCDP Pose-Extended Stage-2 학습 진단 보고서

**작성일**: 2026-05-09
**대상**: Franka 7-DoF SE(3) pose-extended SMCDP V2 score-net 학습
**상태**: 미해결 (현재 κ=10 재학습 진행 중, ~80% 완료, 수렴 의심스러움)

---

## 1. 요약 (Executive Summary)

SE(3) pose-extended 확장의 Stage-2 (V2 score-net) 학습이 **세 차례 시도 모두 충분히 수렴하지 않음**. Stage-1 (pose self-model)은 정상 수렴 (4.7× pos / 16.4× rot improvement)했으나, Stage-2가:

- Loss가 수렴 target (~1e-1) 대비 **6-7 자릿수 큰 상태**에서 정체
- 마지막 시도(`fix123`)는 sample이 Franka 도달 범위 **수십 미터 밖**으로 폭주, succ rate 0%
- 진단 결과 **forward Langevin drift + Varadhan DSM target의 수학적 불일치**가 핵심 원인으로 의심됨
- 현재 시도(`κ=10`)는 catastrophic spike는 막았으나 여전히 절대 loss 큼

---

## 2. 현재 발생 중인 문제 (구체적 수치)

### 2.1 학습 곡선 비교

| 시도 | Final loss | Loss range during training | Spikes >1e+10 | Eval succ@5cm | Eval pos_err |
|---|---|---|---|---|---|
| Drift OFF + position-only V2 | ~1e-2 | 안정 | 0 | ~95% | ~21mm |
| **Drift OFF + pose extension** | **측정 안 됨** | — | — | — | — |
| **fix123** (Fix 1+2+3, κ=1e3) | 2.04e+8 | 1e+8 ~ 1e+13 | 다수 (peak 1e+13) | **0%** (모든 z_e) | **25-31 m** |
| **κ=10** (현재 진행 중, step 12279/15000) | 7.49e+5 | 5e+4 ~ 3e+7 | **0건** | (학습 중) | (학습 중) |

### 2.2 fix123 실패 — 구체적 파괴

진단 스크립트로 학습된 ckpt를 실제 sampling 돌려본 결과:

```
sampled q range:    [-117,575,  +129,373]  rad   ← 미친 값
Franka q range:     [-3.14,     +3.82]     rad   ← 정상 한계
joint-range violation rate (per-element): 94.9%
joint-range violation rate (per-trajectory): 95.7%
max excess past upper joint limit: 129,370 rad  (= 20,000 회전)
endpoint position ‖p‖: [10.13, 35.53] m         ← Franka 도달 ~0.85 m
manifold adherence ‖g_φ‖_max: 1.12e-4           ← 수학적으로는 정상
```

→ **매니폴드 정의(g_φ = 0)는 만족되지만**, q가 unphysical 영역으로 폭주하면서 pytorch_kinematics가 fp32 catastrophic cancellation으로 의미 없는 결과 반환.

### 2.3 fix123 학습 loss 분석

```
step 0:        7.68e+9   (시작부터 이미 큼)
step 150:      1.23e+10  (peak로 올라감)
step 1500:     4.42e+5   (잠깐 하강 — 5 자릿수 떨어짐)
step 3750:     7.10e+6   (다시 폭발)
step 7500:     4.65e+7
step 11250:    9.40e+6
step 15000:    2.04e+8
```

- 학습 *시작부터* loss가 ~10⁹–10¹⁰ 사이에서 wildly oscillating
- 15000 step 내내 안정화되지 않음, 최대 spike **~10¹³ (10조)**
- 한 번도 "정상 학습 phase"에 진입한 적 없음

### 2.4 κ=10 (현재 진행 중) 학습 loss

step 12279 (전체의 81%) 시점 통계:

```
percentile (시간순):
   0%  (step 0):       2.72e+7
   5%  (step ~600):    2.71e+6
  25%  (step ~3000):   4.11e+5
  50%  (step ~6000):   8.70e+5
  75%  (step ~9000):   2.00e+5
  95%  (step ~11500):  4.36e+5
  end (step 12279):    7.49e+5

recent 20% window:  min=5.56e+4, median=4.23e+5, max=3.03e+7
spikes > 1e+8:       102 / 12525  (0.8%)
spikes > 1e+10:      0   / 12525
```

**해석:**
- ✅ Catastrophic spike (>1e+10) 없음 — fix123과 명확히 다른 안정성
- ✅ 추세는 하강 (2.7e+7 → ~4e+5)
- ⚠️ 절대 loss 여전히 5 자릿수 큼 (수렴 target ~1e-1 대비)
- ⚠️ Recent 20% 변동폭 5.6e+4 ~ 3.0e+7 (3 자릿수) — 수렴 신호 약함

### 2.5 수학적 reference: 정상 loss scale

Chart-G DSM loss (extension.tex Eq. 37):
$$\mathcal{L} = \mathbb{E}_r \left[ w(r) \cdot \sum_h (s^q_h - a^*_h)^\top G_\text{pose} (s^q_h - a^*_h) / \tau^2_\text{brown}(r) \right]$$

이론적 minimum (perfect score net):
- $w(r) = \sigma^2(r) \approx \tau_\text{brown}(r)$ (proxy std)
- $G$ max eigenvalue $\sim 6 \times 10^4$ (W_p=400 dominated)
- Diff $\to 0$ → loss $\to 0$
- 실험적 baseline (position-only V2): **~1e-2**

따라서 현재 4.97e+5는 baseline 대비 **~7 자릿수 위**.

---

## 3. 시도해본 해결책과 그 결과

### 3.0 Phase 0: Drift OFF baseline 측정 — 실시 안 됨

**기록 확인** (서버 로그 + git log):
- 첫 pose Stage-2 시도 (`outputs/franka_traj_unet_pose/`, 2026-05-08): step 0에서 즉시 Cholesky failure로 crash
  - 당시 σ_p=0.01 (1cm), W_p=10⁴, cond(G_pose)~10⁵, 일부 demo q에서 fp32 precision으로 G가 non-PD
  - 이 시점에는 forward Langevin drift가 ON이었음 (commit `fd8cf2c` 이후)
- Cholesky 실패 → adb591a에서 forward drift opt-in (default OFF)으로 변경 → **그러나 OFF로 별도 학습 measurement 없음**, Fix 1 (σ_p=0.05) 적용과 Langevin drift ON을 함께 commit (48d430b)
- 이후 모든 시도(`fix123`, `kappa10`)는 drift ON

→ **Drift OFF + pose extension의 baseline은 한 번도 측정된 적 없음**. 이는 보고서의 중요한 누락사항. position-only V2의 21mm는 다른 framework 전용.

### 3.1 Phase 1: Forward Langevin drift 활성화 (실패)

**시도:** extension.tex Eq. 15-17 그대로 구현
- $b^q = -\frac{1}{2} G^{-1} \nabla U^\text{pose}$ 
- $U^\text{pose} = \frac{1}{2\gamma^2}\|q-\mu\|^2 + \frac{1}{2}\log\det G$

**결과:**
- W_p = σ_p⁻² = 10⁴ (σ_p=0.01 m, 1 cm)에서 cond(G_pose) ~10⁵
- G⁻¹ null-space 방향으로 drift 누적
- q가 Franka joint range 밖으로 이탈
- pytorch_kinematics가 NaN 반환 → G_pose가 non-PD → Cholesky failure

**결정:** `forward_langevin_drift=False` (default OFF)로 회피. 학습은 되지만 reviewer-defendable한 수학적 정합성 부재.

### 3.2 Phase 2: noise_stationary_fix.md 도입 (Fix 1+2+3, κ=1e3)

**시도:** 세 가지 수치 안정화 fix를 동시 적용
- **Fix 1**: σ_p 0.01 → 0.05 (W_p 10⁴ → 400, cond ~10²)
- **Fix 2**: Adaptive Tikhonov $\lambda(q) = c \cdot \text{tr}(G)/n_q$, c=1e-2
- **Fix 3**: Anchor-metric soft confining
  - $U_\text{total} = \frac{1}{2\gamma^2}(q-\mu)^\top \hat G (q-\mu) + \kappa U_\text{box}(q)$
  - $\hat G = G(\mu_q)$ per-batch
  - $\kappa = 1000$, $\epsilon = 0.05 \cdot (q_\text{max} - q_\text{min})$
- **Forward Langevin drift ON**

**결과:** 학습 catastrophic 실패
- Loss 1e+9 ~ 1e+13 oscillation, 수렴 안 함
- Eval: q ~ ±10⁵ rad, pos_err 25-31 m, succ 0%

**원인 분석 (κ=1e3 box potential의 drift step 계산):**

```
Box gradient at q=q_max:    2·κ·ε = 2·10³·0.3 = 600         (chart units)
G⁻¹ in null-space:           ~1.0  (J^T W J 영향 없는 redundancy 방향)
G⁻¹·∇U_box (null direction): ~600
Drift step per substep:      ½·β·dr·600 = ½·4·0.05·600 = 60 rad
Franka joint range:          ~6 rad
```

→ **한 substep당 drift가 joint range의 10배** → q가 미친듯이 ricochet → marginal q_r 분포 long-tail → score net 학습 불가능.

doc의 권장값 κ ∈ [1e², 1e⁴]은 다른 normalization 가정에서 나온 듯. 현 코드 setting (ε=0.3 rad, β=4, G⁻¹의 null-space ≈ 1)에서는 부적합.

### 3.3 Phase 3: κ=10으로 축소 (완료, **여전히 실패**)

**시도:** Fix 1+2+3 + Langevin drift ON, **κ=1000 → κ=10** (100배 축소)

**예상 drift step:** `½·4·0.05·2·10·0.3 = 0.6 rad/substep` — joint range의 10% 정도, 안정적.

**최종 결과 (step 15000 완료):**

| 항목 | 값 | 평가 |
|---|---|---|
| Final loss | 4.42e+4 | 여전히 5 자릿수 위 |
| Recent 5% (last ~750 samples) | min 1.4e+4, median 1.4e+5, max 8.9e+6 | 변동성 큼 |
| Catastrophic spike (>1e+10) | 0건 | ✅ cascade 폭발 차단 |
| Spike >1e+8 | 153/15301 (1%) | 가끔 발생 |

**Eval (per z_e):**

| z_e | pos_err mean | pos_err max | rot_err | succ@5cm | manif ‖g‖ |
|---|---|---|---|---|---|
| 0.05 | **15.2 m** | 34.9 m | 2.05 rad | **0%** | 1.78e-4 |
| 0.10 | 14.6 m | 35.4 m | 2.16 rad | 0% | 1.92e-4 |
| 0.15 | 16.2 m | 39.8 m | 2.16 rad | 0% | 1.08e-4 |
| 0.20 | 13.3 m | 33.8 m | 2.15 rad | 0% | 1.99e-4 |

**Sample q-range diagnostic:**
```
sampled q ∈ [-8271, +7168] rad     ← Franka [-3.14, +3.82] 의 1000× 밖
per-element 위반: 95.0%
max excess past upper: 7165 rad (≈1140 회전)
manifold ‖g‖_max: 0.000 (수학적 정합 ✓)
```

**fix123 vs κ=10 비교:**

| 지표 | fix123 (κ=1000) | κ=10 | 변화 |
|---|---|---|---|
| q 폭주 magnitude | ±10⁵ rad | ±10⁴ rad | 14× 작아짐 (여전히 1000× past) |
| pos_err mean | 25–31 m | 13–16 m | 50% 줄어듬 (여전히 도달 불가) |
| **succ@5cm** | **0%** | **0%** | **변화 없음** |
| Loss spike >1e+10 | 다수 | 0건 | 안정성만 개선 |

**해석**: κ tuning은 **증상의 강도**를 약화시켰을 뿐, **근본 메커니즘은 그대로**. 둘 다 사용 불가능한 모델. 학습 자체는 fix123 대비 안정화됐으나 score net이 의미 있게 수렴하지 못함.

---

## 4. 의심되는 핵심 문제 (Root Cause Hypotheses)

### 4.1 [의심 1] Forward Langevin drift + Varadhan DSM target의 수학적 불일치

**Chart-form DSM target** (extension.tex Eq. 35):
$$a^*_\text{pose} = G^{-1} \cdot \left[(q_0 - q_r) + J^\top W \cdot \text{Log}_\text{SE(3)}(T_\phi(q_r)^{-1} T_\phi(q_0))\right]$$

이 target은 **Varadhan asymptotics**를 가정 — *pure Brownian forward*에서만 정확:
- 작은 $r$ (small diffusion time)에서 $\log p(q_r | q_0; r) \approx -\frac{d^2(q_r, q_0)}{2\tau_\text{brown}}$
- 이때 chart score $\approx \text{Log}(q_0)/\tau_\text{brown}$

Forward에 Langevin drift가 추가되면:
- Marginal $p(q_r | q_0; r)$이 더 이상 heat kernel 형태가 아님
- 정확한 score는 drift 항도 포함해야 함
- $a^*/\tau_\text{brown}$이 *근사 score*가 되어, 큰 $r$에서 systematic bias 발생

**현 코드는 forward를 Langevin으로 바꾸면서 DSM target은 Varadhan 형식 그대로 사용** → score net이 잘못된 target에 fit하려 시도 → loss 안 떨어짐.

### 4.2 [의심 2] Anchor metric Ĝ가 q ≠ μ에서 부정확

Fix 3의 $\hat G = G(\mu_q)$는 **fixed at anchor**. q가 μ에서 멀어지면:
- Drift $-\frac{1}{2} G(q)^{-1} \cdot \gamma^{-2} \hat G (q-\mu)$가 *비대칭*
- $G(q)^{-1} \hat G \neq I$ when $q \neq \mu$
- Forward marginal이 이론적 stationary $\mathcal{N}(\mu, \gamma^2 \hat G^{-1})$와 *demo 분포* 사이에서 **interpolation이 안 맞음**
- Score net이 학습할 marginal이 명확히 정의 안 됨

doc 자체도 Sec 2.3.5에서 인정: "$\hat G$ approximation은 $q$가 $\mu_q$에서 멀어질수록 부정확. $\gamma$가 충분히 작으면 무시 가능". 그러나 demo q와 μ 간 거리가 $\gamma = 0.6$ 보다 큼.

### 4.3 [의심 3] Score net 출력 분포가 학습 target과 불일치

`TrajectoryScaledScoreFnPose`에서 `std_trick=True`로:
$$s_\text{net}(q_r, r) = \text{net}(q_r, r) / \sigma(r), \quad \sigma(r) = \text{proxy\_std}(r)$$

target은 $a^*/\tau_\text{brown}$. 작은 $r$에서:
- $\sigma(r) \to 0$
- net output을 매우 작은 값으로 나눔 → 수치 불안정
- target 자체도 $1/\tau_\text{brown}$ → 비슷하게 큼

이 두 큰 값의 **diff에 G_pose-norm**을 잡으면 오차가 증폭. 학습 가능한 형태이지만 numerical conditioning이 매우 까다로움.

### 4.4 [의심 4] σ_p=0.05도 여전히 클 수 있음

현재 W_p=400, cond(G) ~10². doc은 Layer B 수치 문제만 정량화했지만:
- Score net이 chart-form 7-dim 출력을 G-weighted norm으로 학습 — G의 max eig ~6e+4
- 이 weight 분포가 loss landscape를 매우 *anisotropic*하게 만듦
- AdamW가 isotropic step을 가정하므로 특정 방향으로만 잘 학습되고 나머지는 underfit

### 4.5 [의심 5] Loss weighting w(r) = sigma²의 부적절성

```python
if weight == "sigma2":
    w = schedule.proxy_std(r) ** 2
```

이 weighting은 *큰 r*을 강조 — Varadhan 근사가 가장 부정확한 영역. Pure Brownian에서는 OK였지만 drift가 들어가면 큰 r에서 target bias가 더 크고, weighting이 그쪽을 강조 → 학습 misdirection.

---

## 4.6 [의심 6 — 추가 진단으로 확인됨] Anchor approximation이 demo 분포에서 명확히 깨짐

**질문**: $\hat G = G(\mu_q)$ approximation은 $q \approx \mu_q$일 때만 valid. 실제 demo q는 $\mu_q$에서 얼마나 떨어져 있는가?

**측정 결과 (B=1024 demo trajectories, σ_p=0.05, γ=0.6):**

| 거리 척도 | 값 |
|---|---|
| chart-Euclidean $\|q_\text{demo} - \mu_q\|$ | mean 1.097, max 2.32, p99 1.67 |
| $\|q_\text{demo} - \mu_q\| / \gamma$ | mean **1.83**, p99 **2.79** |
| $\hat G$ eigenvalues | min 1.00, max 441.5, cond 442 |
| $\|q_\text{demo} - \mu_q\|_{\hat G}$ (Mahalanobis) | mean **8.61**, max **16.84** |
| $\|q_\text{demo} - \mu_q\|_{\hat G} / \gamma$ | mean **14.4**, max **28.1** |
| Stationary expected $\|q-\mu\|_{\hat G}$ ($\gamma\sqrt{n_q}$) | 1.587 |
| Demo / stationary ratio | **5.4×** |

**해석**:
- Anchor approximation이 valid하려면 demo가 stationary $\mathcal{N}(\mu, \gamma^2 \hat G^{-1})$ 안에 들어와야 함
- 실제로는 demo가 stationary expected의 **5.4×** 거리에 위치 — 명확한 outlier
- doc Sec 2.3.5의 "$\gamma$가 충분히 작으면 무시 가능" 조건이 *반대로* 위배 (γ가 너무 작아서 stationary ellipsoid가 demo를 못 덮음)
- → **Forward SDE의 anchor metric drift는 demo region에서 정확하지 않은 force field를 만듦** → score net 학습 target도 부정확

이건 의심 (4.2)를 *정량적으로 확인*. Fix 3 Option B가 paper에서 가정한 first-order regime이 본 setting에서 성립 안 함.

## 4.7 [의심 7 — 코드 검증으로 확인] Score net과 forward SDE의 anchor 불일치

**질문**: Score net의 conditioning에 $\mu_q$가 들어가는가, 아니면 $T_\text{start}$가 대체하는가?

**코드 확인 (`franka_traj_unet_pose.py:264-265`, `trajectories_pose.py:191-227`):**

```python
# Training loop
x, branch_A, z_e, T_target, T_start = demo.sample(...)
goal_cond = torch.cat([T_start, T_target], dim=-1)  # (B, 14)

# Score net forward (TrajectoryScoreNetUNetPose):
# inputs:
#   - q_traj  (B, H+1, n_q)            # current sample
#   - goal_cond = [T_start, T_target]  # (B, 14)
#   - z_e (embodiment)                 # (B, n_z)
#   - t (diffusion time)
# μ_q는 어디에도 안 들어감.
```

**진단**: $\mu_q$는 forward/reverse SDE의 **limiting drift**에서만 사용 (`PoseLangevinSDE.limiting_q_mean`). Score net 자체는 $\mu_q$를 모르고, **$T_\text{start}$ (= $T_\phi(q_\text{demo}[0])$ — pose form)를 anchor 정보로 사용**.

이는 **fundamental anchor mismatch**:

| 측면 | Forward SDE | Score net |
|---|---|---|
| Anchor 위치 | $\mu_q = [0, -0.3, 0, -1.7, 0, 1.4, 0]$ (chart) | $T_\text{start} = T_\phi(q_\text{demo}[0])$ (pose) |
| Anchor 종류 | 단일 chart-space 점 | per-sample SE(3) point |
| Demo와의 관계 | bimodal 평균 (q_rest_A, q_rest_B의 mid-point) | 한쪽 mode의 trajectory 시작 |

**구체적 차이**:
- demo는 q_rest_A=[+0.6, ...] 또는 q_rest_B=[-0.6, ...] 근처에서 시작 (bimodal)
- $\mu_q = [0, -0.3, ...]$는 *둘 다 아닌* 평균
- score net은 "T_start로부터 trajectory를 reconstructing"을 학습
- 그러나 forward SDE는 q를 *둘 다 아닌* $\mu_q$로 끌어당김
- → 학습된 score는 forward marginal $p(q_r | q_0)$의 score인데, sampling 시 reverse SDE는 $\mu_q$ 중심 stationary에서 시작
- **두 anchor 간 격차가 거의 모든 학습-sampling mismatch의 원인**

**문헌 참조**: Riemannian SGM 표준 framework (De Bortoli 2022 등)에서는 score net이 *limiting distribution의 mean*도 conditioning으로 받거나, limiting distribution을 fixed (e.g. uniform)로 가정. 본 framework는 양쪽도 아닌 hybrid 형태.

## 5. 매니폴드 자체는 정상 (구조 문제 아님)

| 검증 항목 | 결과 |
|---|---|
| Sanity test S1-S8 | ✅ 모두 통과 (machine precision) |
| ‖g_φ‖ on samples | ✅ ~1e-4 (manifold adherence 유지) |
| Stage 1 self-model | ✅ ξ_φ residual smooth (norm 0.053 inside, 0.066 outside) |
| Per-batch Ĝ correctness | ✅ z_e 변화 시 Ĝ가 nontrivial하게 다름 (max diff 28.9 over z=[0.05, 0.20]) |
| 매니폴드 정의의 q-범위 제약 | ❌ 없음 (구조적 한계, 의도된 설계) |

**즉**: 매니폴드 자체는 수학적으로 일관되게 작동. 문제는 **forward SDE / DSM target 정합성** + **하이퍼파라미터 정량 calibration**.

---

## 6. 다음 시도 후보

현재 κ=10 학습 완료 시점 결과에 따라 분기:

### 6.1 κ=10 결과 충분히 좋으면 (예: succ@5cm > 30%)
- Hyperparameter sweep으로 fine-tune (κ ∈ {3, 10, 30}, ε ∈ {0.01, 0.05})
- Forward drift OFF baseline과 비교

### 6.2 κ=10도 succ rate 낮으면
**Option A: DSM target 재유도**
- Forward Langevin drift consistent한 score-matching loss 사용
- Conditional score $\nabla_{q_r} \log p_r(q_r | q_0)$를 OU 근사로 명시적 계산
- 가장 수학적으로 정직, 그러나 구현 부담 큼 (~1주)

**Option B: Forward Langevin drift OFF로 회귀**
- noise_stationary_fix.md 전체 폐기
- Pure Brownian forward + Varadhan target (이게 baseline에서 작동)
- "drift OFF는 mathematical limitation"으로 paper에서 인정
- 가장 실용적, paper writing 측면 약점 인정

**Option C: Box potential 폐기, Anchor metric만 유지**
- $U_\text{total} = \frac{1}{2\gamma^2}(q-\mu)^\top \hat G (q-\mu)$ (no box)
- Joint range는 demo distribution + score net에 맡김
- Box overshoots 원인 자체를 제거
- 실험으로 가능성 검증 필요

**Option D: 학습 자체 stabilization**
- Gradient clipping 더 강하게 (현재 1.0 → 0.1)
- AdamW → SGD with momentum (anisotropic loss landscape에 더 적합할 수 있음)
- Loss weighting 변경 (sigma² → none, large-r 강조 약화)

---

## 7. 다음 시급한 실험: Drift OFF + Fix 1 baseline

세 가지 핵심 발견 (의심 4.6, 4.7, Phase 0)을 종합하면:

1. **Drift OFF + pose extension은 한 번도 측정된 적 없음** — 전체 framework이 작동하는지 자체가 미확인
2. **Anchor approximation이 demo region에서 5.4× outlier** — Fix 3 Option B의 first-order 가정 위배
3. **Score net은 μ_q를 모르고 T_start만 받음** — forward SDE의 limiting drift와 mismatched anchor

**가장 시급한 검증**: 만약 Drift OFF (가장 단순한 setting, Varadhan-asymptotic pure Brownian, anchor mismatch 없음)에서도 학습이 안 된다면, framework 자체에 더 깊은 문제 있음. 작동하면 → 나머지 Fix들의 incremental tuning 문제.

**추천 명령어**:
```bash
CUDA_VISIBLE_DEVICES=0 python -m smcdp.experiments.franka_traj_unet_pose \
  --stage1-pose-ckpt outputs/franka_stage1_pose/xi_phi.pt \
  --sigma-p 0.05 \
  --steps 15000 --batch 64 \
  --out-dir outputs/franka_traj_unet_pose_drift_off_baseline
```

(Fix 1 만 ON, drift / Tikhonov / confining 모두 OFF)

이게 수렴하면:
- pos_err < 1m, succ@5cm > 50% 정도 기대 가능
- 그 위에서 점진적 Fix 추가 (Fix 2 → Fix 3 단계별)

이게 **여전히 실패하면**:
- SE(3) extension framework의 본질적 problem (DSM target 자체, score net 구조, demo distribution mismatch 등)
- noise_stationary_fix.md 폐기하고 framework를 다시 검토

## 8. 현재 진행 상태

- ✅ Phase 0: drift OFF + pose baseline → **never measured**
- ✅ Phase 1: drift ON, no fixes → Cholesky failure
- ✅ Phase 2: Fix 1+2+3, κ=1e3 → catastrophic, succ 0%, q ~10⁵ rad
- ✅ Phase 3: Fix 1+2+3, κ=10 → less catastrophic but still succ 0%, q ~10⁴ rad
- ⏳ Phase 4: drift OFF + Fix 1 baseline (다음 단계)

---

## 8. Reviewer-perspective 요약

paper 작성 시 이 진단을 reviewer에게 어떻게 설명할 것인가:

1. **정직한 limitation**: "Pose-extended SMCDP의 forward Langevin drift는 수치적 안정성 + Varadhan DSM target 정합성 두 측면에서 추가 작업이 필요. 본 논문은 default forward Brownian으로 결과를 보고하되, drift consistent target derivation을 future work로 명시."

2. **Position-only V2의 작동은 strong baseline**: 21mm pos_err, 95% succ rate — pose extension 없이도 framework의 핵심 기여 (graph manifold + chart-form DSM)는 입증됨.

3. **Pose extension의 partial success**: Stage 1 (self-model)은 4.7×/16.4× improvement로 확실히 작동. Stage 2 (score net)는 추가 work 필요.
