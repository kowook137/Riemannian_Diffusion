# Mode Distribution 회복 — 진단 실험 설계

**목적**: Method A 의 mfe 0.12 (DP-* 대비 4–8× 나쁨) 의 root cause 를 분리하고, 가장 효과적인 fix 를 결정한다.

**Compiled**: 2026-05-11
**Reference**: `pose_extension_report.md` §6.2, §7.6, §10.4 (mfe 수치), `boundary_metric_plan.md` §7 (mode metrics)

---

## 0. 배경 — 무엇을 진단하는가

### 0.1 현재 측정된 mfe 수치 정리

`pose_extension_report.md` §7.6.3 + §10.3 통합:

| 모델 | mfe (전체 평균) | 의미 |
|---|---|---|
| DP-canonical | 0.015 | gen frac_A ∈ [0.485, 0.515], 거의 isotropic |
| DP-channel | 0.033 | gen frac_A ∈ [0.467, 0.533] |
| Projected | 0.035 | 비슷 |
| v4.1 bounded | 0.047 | gen frac_A ∈ [0.453, 0.547] |
| **Method A (v4)** | **0.12** | **gen frac_A ∈ {0.38, 0.62}, 1.6× imbalance** |
| BC | 0.47 | 한 mode 만 (collapse) |

Demo 의 frac_A = frac_B = 0.5 (bimodal). Ideal sampler 는 mfe → 0.

### 0.2 핵심 가설

Method A 만이 IK-derived $q^\text{init}$ 를 reference distribution center 로 사용. DP-* 는 standard Gaussian $\mathcal{N}(0, I)$ reference. **이 단일 차이가 mfe 격차의 dominant cause 일 가능성** 이 가장 높다.

### 0.3 5 가지 원인 후보 (배경)

| # | 원인 | 검증 가능성 |
|---|---|---|
| A | IK seed 의 mode-biased selection | 직접 측정 가능 |
| B | Reference distribution σ_K 와 mode coverage trade-off | ABL1 mfe 재측정 |
| C | Score net 의 mode-asymmetric learning | 모델 내부 진단 |
| D | Drift-free Brownian 의 mode-mixing 부재 | offline patch 측정 |
| E | Statistical noise | 배제 (n=256, mfe 0.12 는 5σ 떨어짐) |

본 문서는 **A 와 D 를 가장 낮은 비용으로 분리** 하는 것을 목표.

---

## 1. 실험 1 — IK Mode Bias 직접 측정 (원인 A 검증)

### 1.1 가설

> Method A 의 IK warm-start 가 단일 $q_\text{warm}$ ($q_\text{rest}$) 을 사용하므로, 같은 $T_\text{start}$ 의 두 mode 해 중 하나로 deterministic 하게 편향된다. 결과 IK solution 의 mode 분포가 50/50 이 아니다.

### 1.2 Protocol

#### Step 1: Eval condition 의 IK mode 분포 측정

```python
# eval 에 쓰는 동일 T_start 100 개 sampling
T_start_samples = sample_test_T_start(n=100)

mode_counts = {"A": 0, "B": 0}
for T_start in T_start_samples:
    # 현재 production 코드의 IK 호출 그대로
    q_init = ik_step(T_start, q_warm=q_rest, n_steps=N_default)

    # mode classifier (이미 존재)
    mode = mode_classifier(q_init)  # "A" or "B"
    mode_counts[mode] += 1

frac_A_IK = mode_counts["A"] / 100
print(f"IK frac_A (single warm-start): {frac_A_IK:.3f}")
```

#### Step 2: Multi-warm-start IK 의 mode 분포 측정 (control)

```python
mode_counts_random = {"A": 0, "B": 0}
for T_start in T_start_samples:
    # warm-start 를 demo 의 random subset 에서 sampling
    q_warm = np.random.choice(demo_q_init_pool, 1)[0]
    q_init = ik_step(T_start, q_warm=q_warm, n_steps=N_default)
    mode = mode_classifier(q_init)
    mode_counts_random[mode] += 1

frac_A_IK_random = mode_counts_random["A"] / 100
print(f"IK frac_A (random warm-start): {frac_A_IK_random:.3f}")
```

### 1.3 판정 기준

| frac_A^IK (single warm-start) | 진단 |
|---|---|
| ∈ [0.45, 0.55] | 원인 A 배제. IK 는 mode-unbiased. 다른 원인 (D, C) 가 dominant. |
| ∈ [0.35, 0.45] 또는 [0.55, 0.65] | 원인 A 부분적 기여. mfe 0.12 의 일부 설명 가능. |
| ∉ [0.35, 0.65] | **원인 A 확정**. mfe 의 주된 cause. Solution 1 (mixture IK seed) 가 답. |

### 1.4 비용

- 학습 0
- 컴퓨트: IK 200 회 (≈ 30 초)
- 코드: 새 함수 0, 기존 IK 와 mode classifier 호출만

---

## 2. 실험 2 — Reference 편향 전달 측정 (원인 D 검증)

### 2.1 가설

> 원인 A 가 dominant 가 아니어도, reference distribution 이 mode 간 unbalanced 한 경우 reverse SDE 가 그 편향을 endpoint 까지 전달한다. **현재 Method A 의 score net 이 정확하다면**, reference 를 strict 50/50 mode mixture 로 강제 patch 시 mfe 가 dramatic 하게 떨어져야 한다.

### 2.2 Protocol

#### Step 1: Demo 의 두 mode 의 rest configuration 추출

```python
demo_q_init_pool = collect_demo_q_init()  # 모든 demo 의 첫 timestep q
modes = [mode_classifier(q) for q in demo_q_init_pool]

q_rest_A = mean([q for q, m in zip(demo_q_init_pool, modes) if m == "A"])
q_rest_B = mean([q for q, m in zip(demo_q_init_pool, modes) if m == "B"])
```

#### Step 2: Offline patched sampling

기존 Method A ckpt 그대로. **Sampling 시 reference distribution 만 patch**:

```python
# 기존
# q_init = ik_step(T_start, q_warm=q_rest)

# Patch: 50/50 mixture IK
def patched_q_init(T_start, sample_idx):
    if sample_idx % 2 == 0:
        return ik_step(T_start, q_warm=q_rest_A)
    else:
        return ik_step(T_start, q_warm=q_rest_B)

# 나머지 sampling pipeline 동일
q_init = patched_q_init(T_start, sample_idx)
q_K ~ N(q_init, sigma_K^2 * G^{-1}(q_init))
# reverse SDE 동일
```

#### Step 3: 전체 eval set 에서 mfe 재측정

`metric.md` 통합 eval set: 4 z_e × 64 samples × n_seed=1 = 256 trajectories.

```
mfe_patched = compute_mfe(patched_trajectories, demo_frac_A=0.5)
```

### 2.3 판정 기준

비교: 원본 mfe = 0.12 vs patched mfe.

| Patched mfe | 진단 |
|---|---|
| ≤ 0.03 | **원인 D 확정**. Reference 편향 전달이 dominant. Solution 1 (mixture IK seed) 가 거의 완전한 fix. |
| ∈ (0.03, 0.07] | 원인 D 의 큰 기여. Solution 1 으로 50–80% 개선 예상. |
| ∈ (0.07, 0.10] | 원인 D 부분적. Score net 자체에 mode-asymmetric learning (원인 C) 도 기여. Solution 2 필요. |
| > 0.10 | 원인 D 배제. 학습 자체의 문제 (원인 C). Solution 3 (mode-conditional) 또는 학습 변경 필요. |

### 2.4 비용

- 학습 0
- 컴퓨트: eval 1 회 (256 trajectory) ≈ 2 시간 single GPU
- 코드: sampling pipeline 의 IK 호출만 patch (~10 lines)

---

## 3. 실험 3 — ABL1 mfe 재측정 (원인 B 보정)

### 3.1 가설

> σ_K calibration 이 pose succ 와 mode coverage 사이에 trade-off 가 있다. ABL1 (σ_K = 0.6, narrower) 가 succ 면에서 우수했던 (§6.2) 것이 mode coverage 희생의 결과일 수 있다.

### 3.2 Protocol

기존 ABL1 ckpt 이 있으면 mfe metric 만 계산.

```python
# ABL1 ckpt 로드
ckpt_abl1 = load("franka_pose_method_a_abl1_sigma_K_0.6.pt")

# 동일 eval set 으로 sampling, mfe 계산
trajectories_abl1 = sample(ckpt_abl1, eval_conditions)
mfe_abl1 = compute_mfe(trajectories_abl1, demo_frac_A=0.5)
```

### 3.3 판정 기준

비교: Method A mfe = 0.12 vs ABL1 mfe.

| ABL1 mfe | 진단 |
|---|---|
| ≤ 0.12 | σ_K = 0.6 이 succ + mfe 모두 우위. Method A 의 σ_K = 1.414 spec 정밀화가 mode 면에서 손해. |
| > 0.12 (e.g. 0.20+) | σ_K = 1.414 가 mode coverage 면에서 우위. 현재 spec 유지가 옳음. |
| ≈ 0.12 (variance 내) | σ_K 는 mfe 와 무관. 원인 A/D 만 결정적. |

### 3.4 비용

- 학습 0 (ckpt 재사용)
- 컴퓨트: mfe 계산 (≈ 10 분 — 기존 eval data 재사용 가능)

---

## 4. 실험 4 (Conditional) — Score Net Mode-Asymmetric Learning 검증

### 4.1 트리거 조건

실험 2 의 patched mfe > 0.07 일 때만 실행. 즉 reference 를 strict 50/50 으로 만들어도 mfe 가 남으면, 원인 C (score net 자체) 진단.

### 4.2 Protocol

#### Step 1: Mode-별 validation loss 측정

```python
# Demo 의 두 mode subset
demo_A = [d for d in demos if mode_classifier(d.q[0]) == "A"]
demo_B = [d for d in demos if mode_classifier(d.q[0]) == "B"]

# 학습된 score net 의 loss 측정 (gradient update 없이)
loss_A = compute_dsm_loss(score_net, demo_A, n_samples=200)
loss_B = compute_dsm_loss(score_net, demo_B, n_samples=200)

print(f"Per-mode DSM loss: A={loss_A:.4f}, B={loss_B:.4f}")
print(f"Ratio: {max(loss_A, loss_B) / min(loss_A, loss_B):.2f}×")
```

#### Step 2: Mode-별 endpoint accuracy 측정

Patched sampling (실험 2) 의 trajectory 를 mode A/B 로 split, 각각의 pose succ 비교:

```python
trajectories_patched = sample_with_50_50_mixture(...)
modes_gen = [mode_classifier(traj) for traj in trajectories_patched]

succ_A = pose_succ([t for t, m in zip(trajectories_patched, modes_gen) if m == "A"])
succ_B = pose_succ([t for t, m in zip(trajectories_patched, modes_gen) if m == "B"])

print(f"Per-mode pose_succ: A={succ_A:.2%}, B={succ_B:.2%}")
```

### 4.3 판정 기준

| 측정값 | 진단 |
|---|---|
| loss_A ≈ loss_B (ratio < 1.3×) AND succ_A ≈ succ_B | 원인 C 배제. mfe 잔여는 sampling stochasticity. |
| loss_A / loss_B ≥ 1.5× | Score net 자체가 mode-asymmetric. 학습 시 mode-balanced batch sampling 필요. |
| succ_A vs succ_B 가 > 10 pp 차이 | 한 mode 의 sample 이 manifold 다른 영역에 떨어져 score net coverage 부족. Solution 2 (mixture noising) 필요. |

### 4.4 비용

- 학습 0
- 컴퓨트: validation loss ≈ 10 분, per-mode succ 는 실험 2 의 data 재사용

---

## 5. 의사결정 트리

실험 결과에 따라 Solution 선택:

```
실험 1: frac_A^IK
    ├─ ∈ [0.35, 0.65] → 원인 A 약함
    └─ ∉ [0.35, 0.65] → 원인 A 확정

실험 2: patched mfe
    ├─ ≤ 0.03 → Solution 1 (mixture IK seed) 충분
    ├─ ∈ (0.03, 0.07] → Solution 1 + Solution 2 hybrid 고려
    ├─ ∈ (0.07, 0.10] → Solution 2 (mixture noising training) 필요
    └─ > 0.10 → 실험 4 진행

실험 3: ABL1 mfe
    └─ σ_K 가 mfe 에 영향 있으면 Solution 4 (σ_K adjust) 추가 고려

실험 4 (conditional):
    ├─ loss ratio < 1.3× AND succ tied → Solution 1 + 2 결합으로 해결
    └─ loss / succ 비대칭 → Solution 3 (mode-conditional) 또는 학습 시 mode-balanced sampling
```

---

## 6. Solution 후보 정리

진단 결과에 따라 적용할 fix:

### Solution 1: Mixture IK warm-start

- Sampling 시 $q_\text{warm}$ 을 $\{q_\text{rest}^A, q_\text{rest}^B\}$ 에서 50/50 random 선택
- 학습 0, sampling code 10 lines
- 예상: mfe 0.12 → 0.02–0.05
- 적용 조건: 실험 2 patched mfe ≤ 0.05

### Solution 2: Mixture noising at training

- 학습 시 reference distribution 을 single Gaussian 이 아닌 50/50 mixture 로 변경
- 학습 재실행 필요 (3 시간)
- 예상: mfe → 0.02, 학습-sampling consistency 강화
- 적용 조건: Solution 1 으로 부족 (mfe > 0.05 남음)

### Solution 3: Mode-conditional sampling

- Condition $c$ 에 mode index 추가
- 학습 + sampling 둘 다 변경 (~4 시간)
- 예상: mfe = 0 by construction, 그러나 metric 의미 약화
- 적용 조건: Solution 1/2 모두 실패 시 fallback

### Solution 4: σ_K sweep

- ABL1 (σ_K=0.6), Method A (σ_K=1.414), 추가 (σ_K=1.0) 비교
- 학습 0 (또는 1 회 재학습)
- 적용 조건: 실험 3 에서 ABL1 mfe < 0.10 발견 시

---

## 7. 실험 실행 순서 및 timeline

### Day 0.5 (4 시간)

1. **실험 1** (IK mode bias): 30 분
2. **실험 3** (ABL1 mfe 재측정): 10 분 (ckpt 있는 경우)
3. **실험 2** (offline patched sampling): 2 시간
4. 결과 분석 및 의사결정: 1 시간

### Day 1 (Solution 적용)

진단 결과에 따라:

- **Case Solution 1**: code patch + eval 재실행 (4 시간)
- **Case Solution 2**: 학습 재실행 + eval (5 시간)
- **Case Solution 3**: 학습 + eval + per-mode metric (6 시간)

### Day 1.5 (Cross-validation)

- pose_succ@(5cm, 5°) 가 fix 후 유지되는지 확인
- v4.1 bounded chart 에도 같은 fix 적용 (실험 보고서 §10 결과 일관성 유지)
- per-mode Wasserstein 계산 (mfe 외 distribution 내부 분포까지 확인)

---

## 8. 측정할 metric

### 8.1 Primary

- **mfe** = $|\text{frac}_A^\text{gen} - 0.5|$ (전체 eval set, ID 평균, OOD 분리)
- per-mode pose_succ@(5cm, 5°) — Solution 적용 후 두 mode 모두 demo 수준 유지 확인

### 8.2 Secondary

- **per-mode Wasserstein**: 각 mode 내부에서 generated trajectory 가 demo 분포와 얼마나 가까운가
  - `boundary_metric_plan.md` §7 의 mode metric
  - mfe 만으로는 mode 비율만 보지만 W_1^A, W_1^B 는 mode 내부 분포 fidelity
- **feasible_mode_rate** (v4.1 한정): 선택된 mode 가 joint feasibility 만족하는 비율

### 8.3 Sanity

- pose_succ@(5cm, 5°) ID 평균이 Solution 적용 후 ±2 pp 안에 유지되는가
- manifold gap 0 유지
- v4.1 의 경우 viol = 0% 유지

---

## 9. v4 와 v4.1 둘 다 적용

`pose_extension_report.md` §10 의 v4 vs v4.1 비교 일관성을 위해, 선택된 Solution 을 v4 와 v4.1 양쪽 모두에 적용한다.

**예상 결과 (Solution 1 의 경우)**:

| 모델 | mfe (before) | mfe (after) | pose_succ ID (before) | pose_succ ID (after) |
|---|---|---|---|---|
| v4 Method A | 0.12 | 0.02–0.05 | 94.27% | 92–94% (약간 감소 가능) |
| v4.1 bounded | 0.047 | 0.02–0.04 | 75.00% | 74–76% |

pose_succ 의 약간 감소는 mode mixing 으로 인한 sample diversity 증가가 boundary case 에서 일부 sample 의 endpoint accuracy 를 낮출 가능성 — 측정 후 trade-off 평가.

---

## 10. 성공 기준

### 10.1 Minimal

- Method A (v4) mfe ≤ 0.05 (현재 DP-channel 의 0.033 에 근접)
- pose_succ@(5cm, 5°) ID 평균 ≥ 92% (현재 94.27% 에서 −2 pp 이내)

### 10.2 Strong

- Method A mfe ≤ 0.03 (DP-canonical 의 0.015 에 근접)
- per-mode Wasserstein 이 demo 와 차이 < 5%
- v4.1 bounded chart 에서도 같은 수준 달성

### 10.3 실패 시

mfe 가 0.08 이상 남으면 → score net 자체에 fundamental mode-asymmetric learning. 학습 변경 (Solution 2/3) 또는 가중치 sampling 으로 demo 의 mode balance 학습 강제.

---

## 11. 부록 — 실험 보고서와의 연결

본 진단은 `pose_extension_report.md` 의 다음 항목을 보완:

- §6.2 ABL3 분석에서 "Per-trajectory $q^\text{init}$ 이 결정적 contribution" 확인됨 — 본 실험은 그 $q^\text{init}$ 의 mode-bias 정량화
- §7.6.3 핵심 finding 2 ("Method A mfe 0.12 는 DP variants 대비 4–8× 나쁨") — 본 실험은 그 원인 분리
- §10.4 v4.1 의 mfe 0.047 (v4 대비 2.5× 개선) — 본 실험으로 이 개선이 의도된 효과인지 우연인지 확인 가능

본 진단 결과는 다음 후속 실험의 input:

- `boundary_active_experiment_design.md` 의 Tier 2 실험에서 mode balance 가 task_succ 결정에 더 중요해질 가능성. 본 진단 fix 가 boundary-active experiment 의 사전 조건.