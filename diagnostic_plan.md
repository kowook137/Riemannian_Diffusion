# Task Success Diagnostic Plan

**Goal**: Pos_err 0.30m → 0.10m 도달 (current 7-DoF Franka task)

**Current state**:
- Mean이 cond 따라 이동 (cond pathway 작동)
- Std 0.15-0.24m (분산 큼)
- CFG sweep $w \in \{0, 1, 2, 4, 7, 12\}$, $w=0$이 최선 (CFG amplification 효과 없음)

---

## 0. 진단의 출발점 — Framing

### 0.1 입증된 사실 (Geometric Substrate)

본인 framework의 **geometric substrate**는 검증됨:

**3-DoF planar (D-2)**:
- Mode averaging 0% (vs BC 100%, DP 3-9%)
- W₁ 14-57x improvement
- Pos_err 5-12x improvement vs baselines
- OOD residual learning 30% 추가 개선

**7-DoF Franka substrate**:
- Sanity 6종 모두 machine precision pass
  - $J_F^{\text{closed}} = J_F^{\text{autograd}}$
  - Retraction이 $\Mphi$ 위 유지
  - $J_g \cdot v = 0$ (auto-tangent)
  - Norm equivalence
  - $\N(0, G^{-1})$ sample (1.6% error)
  - log/exp roundtrip exact

**7-DoF Framework mechanism (§10 contributions)**:
- C1: max $\|g_\phi\| = 0$ by construction, DSM loss 50→2 수렴
- C2: Mode collapse 없음 (frac_A 0.40-0.51)
- C3: $z_e$ generalization in-dist + OOD partial

**Self-model accuracy**: 16.2mm → 0.29mm (55.8x improvement)

### 0.2 진단의 정확한 framing

**7-DoF sanity checks indicate that the geometric manifold substrate is correct.** 즉:
- Manifold 정의, tangent bundle, induced metric, retraction 등 **geometric layer** 검증됨
- Mathematical implementation 정확

**그러나 다음은 별도 진단 대상**:
- **Probabilistic modeling layer**: demo distribution을 잘 capture하는지
- **Task-level objective**: DSM loss와 task success 사이 alignment
- **Trajectory smoothness prior**: product manifold가 component-wise이므로 temporal coupling은 score net 학습 의존
- **Score-model capacity**: 7-DoF + sparse conditioning에 sufficient한지
- **Sampling configuration**: discretization이 충분한지

**The remaining 0.30 m task error is likely caused by data sharpness, conditioning strength, score-model capacity, sampling configuration, or task-level objective mismatch — rather than the geometric manifold construction.**

### 0.3 진단의 진짜 의미

이 plan은 **engineering optimization을 위한 systematic exploration**.

본인이 옳은 길에 있고, 남은 건 probabilistic/task layer의 tuning. 가능한 source:
- Score model architecture (conditioning injection 강도)
- Demo distribution (within-mode spread, target bias)
- Hyperparameter (limiting $\gamma$, training schedule, normalization)
- Sampling configuration (K, $\Delta r$)
- Loss design (endpoint weighting, DSM by-r decomposition)

---

## 1. 가능한 Bottleneck

| Layer | 가능성 | Diagnostic 비용 | Layer type |
|---|---|---|---|
| (A) Demo distribution spread / bias | 높음 | 낮음 | Data |
| (B) Conditioning signal weak | 높음 | 중간 | Implementation |
| (C) Sampling discretization 부족 | 중간 | 매우 낮음 | Hyperparameter |
| (D) DSM-task objective mismatch | 중간 | 낮음 | Loss design |
| (E) Score model architecture 한계 | 중간 | 높음 | Implementation |
| (F) Limiting distribution wide | 낮음 | 중간 | Hyperparameter |
| (G) Training 부족 / 수렴 | 낮음 | 낮음 | Hyperparameter |

**모든 bottleneck이 geometric substrate 외부 layer**. 낮은 비용부터 시작 → 원인 빨리 발견.

---

## 2. Phase 1: Demo Distribution 분석 (최우선)

**왜 먼저**: 가장 fundamental. Demo 자체가 spread하거나 target에서 biased면 모델 fix 무의미.

### 2.1 Same-target demo std 측정

```python
dataset = load_demo_dataset()  # [N, 16, 7]

groups = group_by(dataset, ['p_target', 'mode'])

within_mode_std_q = []
within_mode_std_pee = []

for (p_target, mode), demos in groups.items():
    if len(demos) < 5:
        continue
    
    q_traj = stack(demos.q)
    std_q = std(q_traj, axis=0)
    
    p_ee_traj = batch_FK(q_traj)
    std_pee = std(p_ee_traj, axis=0)
    
    within_mode_std_q.append(std_q)
    within_mode_std_pee.append(std_pee)

mean_std_q_at_h15 = mean([s[15] for s in within_mode_std_q], axis=0)
mean_std_pee_at_h15 = mean([s[15] for s in within_mode_std_pee], axis=0)
```

**판단 기준**:
- $\text{std}_{p_{ee}}$ at $h=15$ < 0.02m → Demo는 sharp. 모델 issue (Phase 2로)
- $\text{std}_{p_{ee}}$ at $h=15$ in [0.02, 0.10]m → Demo가 적당. 모델도 부분 issue
- $\text{std}_{p_{ee}}$ at $h=15$ > 0.10m → Demo가 main bottleneck

### 2.2 Demo target bias (NEW)

Std뿐 아니라 mean이 target에서 벗어나는지:

$$
\text{bias}(c, \text{mode}) = \left\| \E_{\text{demo}}[p_{ee, H} \mid c, \text{mode}] - p_{\text{target}} \right\|
$$

```python
target_bias = []
for (p_target, mode), demos in groups.items():
    p_ee_final = batch_FK(stack(demos.q))[:, 15, :]  # [n_demos, 3]
    mean_p_ee_final = mean(p_ee_final, axis=0)
    bias = norm(mean_p_ee_final - p_target)
    target_bias.append(bias)

mean_target_bias = mean(target_bias)
```

**판단 기준**:
- Bias < 0.02m → Demo가 target 잘 도달. Demo OK
- Bias in [0.02, 0.05]m → 약간 bias. IK convergence 부족
- Bias > 0.05m → Demo 자체가 target에서 벗어남. **Critical issue**

**중요**: Bias가 크면 모델이 demo를 정확히 학습해도 task fail. IK convergence threshold 재검토 필요.

### 2.3 Per-target demo count 확인

```python
unique_targets = unique(p_target_list)
demos_per_target = count_per_unique(p_target_list)

print(f"Unique targets: {len(unique_targets)}")
print(f"Demos per target: mean={mean}, min={min}, max={max}")
```

**기준**:
- < 5 → 너무 적음. Sparse target distribution
- > 50 → 충분
- 그 사이 → 보통

### 2.4 IK solver의 stochastic component 확인

같은 (p_target, mode) 조합에 대해 10번 IK 돌려서 결과 std 측정.

확인 사항:
- Random initial joint configuration?
- Null-space bias의 random component?
- Convergence threshold가 sharp한지

### 2.5 Phase 1 결정 분기

| Demo std at end-EE | Demo target bias | 결론 | 다음 step |
|---|---|---|---|
| < 0.02m | < 0.02m | Demo OK | Phase 1.5로 (K sweep) |
| 0.02-0.10m | < 0.02m | Demo 적당 spread | Phase 1.5 → Phase 2 |
| > 0.10m | any | Demo spread main | Demo regenerate 우선 |
| any | > 0.05m | Demo bias main | IK / demo gen fix 우선 |

---

## 3. Phase 1.5: K Sweep Quick Test (NEW)

**왜 일찍**: Retrain 없이 sampling discretization만 테스트. **매우 cheap한 진단**.

### 3.1 Reverse step count sweep

```python
for K in [100, 200, 400, 800, 1600]:
    samples = reverse_sde_sample(model, K_steps=K, n_samples=256)
    pos_err_K = compute_pos_err(samples)
    print(f"K={K}: pos_err={pos_err_K:.3f}m")
```

**판단 기준**:
- K=200 vs K=800에서 pos_err 차이 < 0.02m → Sampling 충분. 다른 issue
- 차이 > 0.05m → Sampling discretization 문제. K 증가가 fix
- K 증가에도 pos_err 평행 → 다른 layer issue (Phase 2로)

### 3.2 EMA / non-EMA model 비교

```python
pos_err_ema = evaluate(model_ema)
pos_err_raw = evaluate(model_raw)
```

차이 크면 EMA 더 길게 또는 다른 EMA decay.

### 3.3 Phase 1.5 결정 분기

| K=200 → K=800 pos_err 변화 | 결론 |
|---|---|
| > 0.05m 감소 | Sampling이 main bottleneck |
| < 0.02m 감소 | Sampling sufficient, Phase 2로 |
| 0.02-0.05m | 부분 contribution |

---

## 4. Phase 2: DSM Loss 분해 진단 (NEW)

**왜 중요**: DSM loss 50→2 수렴이지만 task fail. **DSM loss vs task error의 alignment** 진단.

### 4.1 Train vs Val DSM loss

```python
train_loss = evaluate_dsm(model, train_data)
val_loss = evaluate_dsm(model, val_data)
```

**판단 기준**:
- Train loss 낮고 val loss 높음 → **Overfit**, data sparsity
- 둘 다 높음 → **Underfit**, architecture / optimization 문제
- 둘 다 낮은데 task fail → **Objective mismatch**

### 4.2 Diffusion time $r$별 DSM loss

```python
for r_bin in [0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0]:
    loss_r = evaluate_dsm(model, val_data, r_range=r_bin)
    print(f"r in {r_bin}: DSM loss={loss_r:.3f}")
```

**판단 기준**:
- High-noise time ($r$ → 1)에서 loss 큼 → **Reverse process early step 문제**
- Low-noise time ($r$ → 0)에서 loss 큼 → **Final denoising 부정확**, sharpness 부족
- 모든 $r$에서 균일 → Architecture capacity 한계

### 4.3 Condition / target 별 DSM loss

```python
for target_cluster in target_clusters:
    loss_per_target = evaluate_dsm(model, val_data, condition=target_cluster)
```

**판단 기준**:
- 특정 target cluster에서 loss 큼 → **Conditioning 약함** for those clusters
- 균일하면 conditioning OK

### 4.4 Phase 2 종합 진단 표

| DSM loss pattern | 해석 |
|---|---|
| Train low, val high | Overfit / data sparsity |
| Train ≈ val, both high | Underfit / architecture |
| Train ≈ val, both low + task fail | **Objective mismatch** |
| High at r→1 | Reverse early step 문제 |
| High at r→0 | Final sharpness 부족 |
| High at specific c | Conditioning weak per-c |

---

## 5. Phase 3: Conditioning Signal 분석

### 5.1 Cond/uncond score 차이 measurement

```python
def measure_cond_uncond_diff(model, dataset, n_samples=100):
    diffs_norm = []
    cond_norms = []
    
    for sample in dataset[:n_samples]:
        x_r, c, z_e, r = sample.noised_state, sample.cond, sample.z_e, sample.diffusion_time
        
        with torch.no_grad():
            s_cond = model(r, x_r, c, z_e)
            s_uncond = model(r, x_r, null_token, z_e)
        
        diff = (s_cond - s_uncond).norm()
        cond_norm = s_cond.norm()
        
        diffs_norm.append(diff.item())
        cond_norms.append(cond_norm.item())
    
    relative_magnitude = mean(diffs_norm) / mean(cond_norms)
    return relative_magnitude
```

**기준**:
- Relative magnitude < 0.05 → 매우 약함. Conditioning 거의 무시
- 0.05-0.20 → 약함. CFG 효과 부족 이유
- > 0.30 → 정상

### 5.2 Conditioning 방향성 alignment (NEW)

Magnitude만 보면 부족. **방향성**도 확인:

$$
\cos\left(s_{\text{cond}} - s_{\text{uncond}}, \, \nabla_q R(q, c)\right)
$$

또는 더 직접: $(s_c - s_u)$ 방향으로 한 step 갔을 때 goal error가 줄어드는지.

```python
def measure_cond_direction_alignment(model, samples, target):
    eps = 0.01
    for q, c, z_e in samples:
        s_cond = model(r, q, c, z_e)
        s_uncond = model(r, q, null_token, z_e)
        
        delta = (s_cond - s_uncond)  # chart vector
        
        # Goal error before and after step
        p_before = forward_kin(q)
        err_before = norm(p_before - target)
        
        q_new = q + eps * delta
        p_after = forward_kin(q_new)
        err_after = norm(p_after - target)
        
        delta_err = err_after - err_before  # 음수면 goal error 줄어듦
```

**기준**:
- $\Delta e < 0$ (mean) → Conditioning이 goal 방향. CFG 효과 있어야 함
- $\Delta e \geq 0$ → Conditioning이 강하지만 잘못된 방향 또는 noise

**중요**: 본인의 CFG sweep에서 $w=0$이 최선이었던 것의 진짜 원인. Magnitude는 작거나, 방향이 잘못됐거나, 둘 다.

### 5.3 Target shift 따라 mean shift 비율

이미 진단됨. 비율 1.0이 ideal.

### 5.4 Conditioning ablation

같은 모델로:
- Sample with correct $c$ → mean error
- Sample with random $c$ → mean error
- Sample with $c = \emptyset$ → mean error

**기준**:
- Correct $c$와 random $c$의 차이 작음 → Conditioning 거의 안 쓰임
- 차이 큼 → Conditioning 사용됨, sharpness 부족

### 5.5 FiLM layer activation 분석

ConditionalUnet1D의 FiLM layer:
- $\gamma_c, \beta_c$ output magnitudes
- Layer별 conditioning strength

특정 layer에서 nullified면 architecture issue.

### 5.6 Phase 3 종합 결정 분기

| Cond/uncond magnitude | Direction alignment | 결론 |
|---|---|---|
| < 0.10 | any | Conditioning broken / nullified |
| 0.10-0.30 | $\Delta e < 0$ | Conditioning weak but correct direction |
| 0.10-0.30 | $\Delta e \geq 0$ | Conditioning wrong direction |
| > 0.30 | $\Delta e < 0$ | Conditioning OK, 다른 issue |
| > 0.30 | $\Delta e \geq 0$ | Conditioning strong but wrong (architecture issue) |

---

## 6. Phase 4: Limiting Distribution + 7-DoF 특화 Metric

### 6.1 Forward process trajectory 확인

```python
For r in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    x_r ~ forward_sde(x_0, r)
    measure std(x_r)
```

기대: 점진적으로 limiting까지 spread.

### 6.2 $\gamma$ tightening test (multi-metric)

$\gamma = 0.6 \to 0.3$으로 retrain. **반드시 multi-metric 확인**:

```python
metrics = {
    'pos_err': measure_pos_err(samples),
    'mode_fraction': measure_frac_per_mode(samples),  # mode collapse?
    'diversity_score': measure_diversity(samples),  # within-mode spread
    'max_g_phi': measure_manifold_adherence(samples),  # 0 유지?
    'smoothness': measure_smoothness(samples),  # NEW
    'joint_limit_violation': measure_jl_viol(samples),  # NEW
    'g_condition': measure_G_condition_number(samples),  # NEW
}
```

**판단 기준**:
- Pos_err 감소 + mode 유지 + smoothness OK → $\gamma$ tightening 채택
- Pos_err 감소 but mode collapse → $\gamma$ 너무 tight, 중간값
- Pos_err 변화 없음 → 다른 issue

### 6.3 7-DoF 특화 Metrics (NEW)

#### A. Joint limit violation
$$
\min_h \text{dist}(q_h, [q_{\min}, q_{\max}])
$$
Limit 근처에서 score 불안정 가능. 본인 측정값 1.5-2.1% (이미 OK).

#### B. Trajectory smoothness
$$
\text{vel}: \sum_h \|q_{h+1} - q_h\|^2, \qquad \text{accel}: \sum_h \|q_{h+1} - 2q_h + q_{h-1}\|^2
$$
Pos는 맞아도 trajectory 거칠면 task success 낮음.

#### C. $G$ condition number
$$
\kappa(G) = \frac{\lambda_{\max}(G)}{\lambda_{\min}(G)}
$$
7-DoF에서 metric conditioning이 sampling variance 좌우. 값이 크면 numerical instability.

#### D. Endpoint vs full-trajectory error
- Full trajectory pos_err: $\frac{1}{H+1} \sum_h \|p_h - p_{\text{target}}\|$
- Endpoint pos_err: $\|p_H - p_{\text{target}}\|$

차이 큼 → 모델이 horizon 전체를 맞추느라 endpoint sharpness 약함. **Endpoint loss reweighting fix 후보**.

### 6.4 Phase 4 결정 분기

| Metric | Result | Fix candidate |
|---|---|---|
| Endpoint err >> full traj err | Endpoint sharpness 약함 | Endpoint loss reweighting |
| Smoothness 거침 | Trajectory smoothness 약함 | Smoothness regularization |
| $\kappa(G)$ 큼 (>1000) | Metric conditioning 나쁨 | Manifold parametrization 재검토 |
| $\gamma$ tightening 효과 | Limiting too wide | $\gamma$ 조정 |

---

## 7. Phase 5: Low-Cost Fixes (Architecture 변경 전)

### 7.1 Condition normalization

확인 사항:
- $p_{\text{target}}$ scale (m 단위, 0-1 범위로 normalize?)
- $z_e$ scale (이미 0.05-0.20)
- $q$ scale (rad, joint limit 따라 normalize?)
- $r$ scale ($t \times t_{\text{scale}}=1000$ 적절?)

```python
# Check input statistics
print(f"p_target stats: mean={mean_p}, std={std_p}, range=[{min}, {max}]")
print(f"z_e stats: ...")
print(f"q stats: ...")
```

**Issue 발견 시**:
- Per-channel normalization
- Layer-wise scaling

### 7.2 Endpoint loss reweighting

현재 DSM loss는 trajectory 전체:
$$
\Loss = \sum_{h=0}^{H} \|s_{\theta, h} - a_h^*\|^2_G
$$

Endpoint sharpness를 위해:
$$
\Loss_{\text{weighted}} = \sum_{h=0}^{H} w_h \|s_{\theta, h} - a_h^*\|^2_G, \quad w_H \gg w_h \text{ for } h < H
$$

예: $w_H = 5$, $w_h = 1$ for others.

### 7.3 Goal-conditioned residual guidance

Phase 6 architecture 변경 전, sampling 시 goal residual guidance 추가:
$$
s_{\text{guided}}^q = s_\theta^q + \alpha \cdot G^{-1} \nabla_q \bar{R}(q, p_{\text{target}})
$$

여기서 $\bar R = -\|p_{ee}(q) - p_{\text{target}}\|^2$.

본인 doc §9.2의 reward-based guidance가 정확히 이것. CFG와 별개로 적용 가능.

---

## 8. Phase 6: Architecture 강화 (Last Resort)

위 phase들로 fix 안 되면:

### 8.1 Conditioning channel concat

기존: global_cond [B, 4] → FiLM
강화: q_traj concat with broadcast(p_target) → [B, 16, 7+3]

매 timestep에서 target 정보 직접 access.

### 8.2 Cross-attention

ConditionalUnet1D에 cross-attention block 추가. Cond이 query/key로 작동.

### 8.3 Per-layer FiLM 강화

현재 FiLM이 일부 layer에서만 작동하면 모든 layer에 inject.

---

## 9. 진행 순서

```
Phase 1 (Demo distribution + bias)
    ↓
Phase 1.5 (K sweep — quick, retrain 없음)
    ↓
Phase 2 (DSM loss decomposition — model retrain 없음)
    ↓
Phase 3 (Conditioning magnitude + direction)
    ↓
[중간 결정: Demo 문제? → demo regenerate, return Phase 1]
    ↓
Phase 4 (γ tightening multi-metric + 7-DoF metrics)
    ↓
Phase 5 (Low-cost fixes: normalization, endpoint reweighting, goal residual guidance)
    ↓
Phase 6 (Architecture 강화 — last resort)
```

---

## 10. 결과 정리 Template

| Diagnostic | Result | Interpretation |
|---|---|---|
| Demo std at h=15 (p_ee) | ? m | Demo sharpness |
| Demo target bias | ? m | Demo accuracy |
| Per-target demo count | ? | Data density |
| IK stochastic noise | ? m | Demo gen noise |
| K=200 vs K=800 pos_err diff | ? m | Sampling sufficiency |
| EMA vs raw pos_err diff | ? m | EMA sufficiency |
| DSM train vs val loss | ?, ? | Overfit / underfit |
| DSM loss by r (at r=1) | ? | Reverse early step |
| DSM loss by r (at r=0) | ? | Final sharpness |
| Cond - Uncond / Cond | ? | Conditioning magnitude |
| Cond direction alignment | ? | Conditioning direction |
| Random-c vs correct-c diff | ? m | Conditioning effect |
| Forward SDE std at r=1 | ? | Limiting spread |
| Endpoint vs full traj err | ? m | Endpoint sharpness |
| Smoothness (vel, accel) | ?, ? | Trajectory quality |
| $\kappa(G)$ | ? | Metric conditioning |

---

## 11. Diagnostic 결과별 Fix Path

### 시나리오 A: Demo distribution 문제 (std > 0.10m or bias > 0.05m)

**Fix**:
1. IK solver deterministic하게
2. Per-target demo 수 증가
3. Demo regeneration with tighter convergence

**예상 효과**: Pos_err 0.30 → 0.15-0.20m

### 시나리오 B: Sampling discretization 부족 (K diff 큼)

**Fix**:
1. K = 400-800
2. Step size 자동 schedule

**예상 효과**: Pos_err 부분 개선

### 시나리오 C: DSM-task objective mismatch (loss 낮은데 task fail)

**Fix**:
1. Endpoint loss reweighting
2. Goal residual guidance (Phase 5.3)
3. Per-r loss weighting

**예상 효과**: Pos_err 0.30 → 0.10-0.15m

### 시나리오 D: Conditioning weak/wrong (magnitude < 0.10 또는 direction off)

**Fix options**:
1. Condition normalization (Phase 5.1)
2. Conditioning channel concat (Phase 6.1)
3. Per-layer FiLM 강화

**예상 효과**: Pos_err 0.30 → 0.05-0.10m

### 시나리오 E: Limiting too wide (γ tightening 효과)

**Fix**: $\gamma = 0.3$ + multi-metric 모니터링

**예상 효과**: Pos_err 약간 개선

### 시나리오 F: Endpoint sharpness 부족 (endpoint err >> full)

**Fix**: Endpoint loss reweighting (Phase 5.2)

**예상 효과**: Pos_err 부분 개선

### 시나리오 G: 둘 이상 (가장 가능성 높음)

순차적 fix.

---

## 12. 사전 추측

현재 진단 패턴 (mean 맞음, std 큼, CFG 안 됨)을 보고 추측:

**가능성 높은 시나리오** (확률 순):

1. **Demo within-mode spread + conditioning weak/direction off** (40%)
2. **DSM-task objective mismatch + endpoint sharpness** (25%)
3. **주로 Demo spread or bias** (15%)
4. **주로 Conditioning weak** (10%)
5. **Sampling discretization + 기타** (10%)

**가장 효과적 fix 추측 (low-cost)**:
- Demo bias 확인 + IK fix
- Endpoint loss reweighting
- Condition normalization
- Goal residual guidance

이 셋만 해도 pos_err 0.10-0.15m 가능성 높음.

---

## 13. Implementation Files (제안)

```
experiments/
├── diagnostic_phase1_demo.py
│   ├── analyze_demo_distribution()
│   ├── compute_target_bias()
│   ├── compute_per_target_count()
│   └── ik_stochastic_test()
│
├── diagnostic_phase1_5_sampling.py
│   ├── k_sweep_test()
│   └── ema_comparison()
│
├── diagnostic_phase2_dsm_decomp.py
│   ├── train_val_loss_check()
│   ├── loss_by_diffusion_time()
│   └── loss_by_condition()
│
├── diagnostic_phase3_conditioning.py
│   ├── measure_cond_uncond_diff()
│   ├── measure_direction_alignment()
│   ├── conditioning_ablation()
│   └── analyze_film_activations()
│
├── diagnostic_phase4_limiting_metrics.py
│   ├── forward_sde_trajectory()
│   ├── gamma_tightening_retrain()
│   ├── smoothness_metric()
│   ├── g_condition_number()
│   └── endpoint_vs_full_error()
│
└── diagnostic_phase5_lowcost_fixes.py
    ├── condition_normalization()
    ├── endpoint_loss_reweighting()
    └── goal_residual_guidance()
```

---

## 14. Decision Tree

```
Start
│
├── Geometric substrate correct? ✓ (sanity 6 + C1, C2, C3 입증)
│   → Probabilistic/task layer 진단 대상
│
├── Phase 1: Demo std > 0.10m or bias > 0.05m?
│   └── YES → Demo regenerate (Scenario A)
│   └── NO  → Phase 1.5
│
├── Phase 1.5: K sweep으로 큰 변화?
│   └── YES → Sampling fix (Scenario B)
│   └── NO  → Phase 2
│
├── Phase 2: DSM loss 낮은데 task fail?
│   └── YES → Objective mismatch (Scenario C, F)
│   └── 다른 패턴 → 해당 fix
│
├── Phase 3: Cond/Uncond magnitude < 0.10 or direction wrong?
│   └── YES → Conditioning fix (Scenario D)
│   └── NO  → Phase 4
│
├── Phase 4: γ=0.3 retrain pos_err 크게 줄임 (multi-metric OK)?
│   └── YES → γ tightening (Scenario E)
│   └── NO  → Phase 5
│
├── Phase 5: Low-cost fixes (normalization, endpoint, goal guidance)
│   └── 효과 → 정착
│   └── 부족 → Phase 6
│
└── Phase 6: Architecture 강화 (last resort)
```

---

## 15. 최종 목표

**Task success 도달**:
- pos_err < 0.10m (target box 폭 0.10m 내)
- success@2cm > 50%
- mode capture 유지 (frac_A 0.40-0.60)
- manifold adherence 유지 (max|g_φ| = 0)
- smoothness OK
- joint limit violation < 5%

이 목표 도달 후:
- Baseline 비교 (BC, DP, Projected)
- §16 ablation studies
- Paper writing

---

## 16. 핵심 인사이트 (Framing)

### 16.1 검증된 것 vs 진단 대상

**검증됨 (geometric substrate)**:
- Manifold construction
- Tangent bundle
- Induced metric
- Retraction
- Score lift via $J_H$

**진단 대상 (probabilistic/task layer)**:
- Demo distribution quality
- Conditioning strength + direction
- DSM-task objective alignment
- Sampling sufficiency
- Trajectory smoothness prior
- Score-model capacity for 7-DoF + sparse cond

### 16.2 Engineering optimization의 의미

이 plan은 framework redesign이 아니라 **probabilistic/task modeling layer의 systematic optimization**.

상대적으로 cheap한 fix들이 가능:
- Demo regeneration
- Endpoint loss reweighting
- Goal residual guidance
- Condition normalization
- Conditioning channel concat (if needed)

### 16.3 Paper 관점

본인 framework의 진짜 contribution:
- **Mathematical novelty**: Riemannian SGM on learned manifold
- **Geometric correctness**: by-construction adherence + multi-modal + embodiment
- **Validated**: 3-DoF categorical, 7-DoF mechanism

이게 paper main. Task success는 engineering goal, achievable through this diagnostic.

---

## 17. 진단의 결과로 얻는 것

이 plan을 따르면:

1. **Bottleneck 명확 식별**: 어느 probabilistic/task layer가 main인지
2. **최소 cost로 fix path 결정**: Low-cost fix 우선, architecture 최후
3. **Geometric substrate vs probabilistic layer 명확 구분**: Diagnostic이 framework correctness와 무관함을 명시
4. **Paper writing 준비**: Engineering optimization 결과를 implementation detail section으로

**Geometric manifold substrate는 검증됨. 남은 것은 probabilistic/task modeling layer의 systematic engineering.**