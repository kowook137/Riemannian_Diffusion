# Experiment Plan: Analytic Self-Model + Ablation Study

**목적**: V2 결과 (pos_err 21mm, succ@5cm 99%, succ@10cm 100%)에서 framework의 contribution을 정량 isolate. Paper-grade academic 강도 도달.

---

## 0. 출발점 — 현재 V2 결과

| Metric | V2 (in-dist $z_e=0.10$) |
|---|---|
| pos_err | 0.021 m |
| success@2cm | 50.0% |
| success@5cm | 98.8% |
| success@10cm | 100% |
| frac_A | 0.50 (data 0.50) |
| $W_1^A / W_1^B$ | 0.017 / 0.015 |
| joint violation | 0.0% |
| manifold adherence $\max\|g_\phi\|$ | 0.0 |

이 위에서 추가 실험은 **성능 향상**이 아니라 **원인 분해와 claim isolation**.

---

## 1. 실험의 academic 역할

### 1.1 두 실험 layer의 정확한 분리

본인 framework의 main claim은 두 가지:

**Claim A (Geometric)**: Self-model manifold $\Mphi$ 위 Riemannian diffusion이 Euclidean diffusion보다 우월

**Claim B (Learning)**: Learned residual $\Delta_\phi$가 analytic FK보다 우월

현재 V2는 두 claim이 **entangle**되어 있음. Ours-V2 vs DP-C 비교의 6.4x improvement가:
- Claim A 때문인지
- Claim B 때문인지
- A + B 합인지

분리 안 됨.

### 1.2 Analytic self-model 실험의 핵심 framing

> Perfect FK 실험은 self-model을 제거하는 실험이 아니다. Learned residual $\Delta_\phi$를 제거하고, analytic embodiment-conditioned self-model만 남기는 실험이다.

Perfect FK regime에서:
- $F_\phi = F_{\text{analytic}}$ (residual 제거)
- $\mathcal{M}_{\text{ana}}(z_e) = \{(q, p) : p = F_{\text{analytic}}(q, z_e)\}$ (manifold는 여전히 self-model)
- $G$, $J_H$, retraction, score lift 모두 그대로 (geometric structure 유지)

즉 **geometric framework은 유지하면서 learning component만 제거**. Claim A와 Claim B를 정확히 분리.

---

## 2. Experiment 1: Analytic Self-Model

### 2.1 목적

**핵심 질문**:
> FK가 완전히 정확한 경우에도, self-model manifold 위 Riemannian diffusion policy가 Euclidean DP / Projected diffusion보다 유의미하게 좋은가?

이 실험이 답하면: **Riemannian framework 자체가 가치 있음을 입증** (residual learning과 무관).

### 2.2 Regime 정의

#### Regime A: Perfect FK
$$p_{\text{true}} = F_{\text{analytic}}(q, z_e)$$

- Residual 없음
- Compliance 없음
- Calibration error 없음
- $\Delta_{\text{true}} = 0$
- $F_{\text{analytic}}$가 ground truth

기대: Ours-Learned ≈ Ours-Analytic (consistency 검증)

#### Regime B: Perfect FK + observation/execution noise
$$p_{\text{obs}} = F_{\text{analytic}}(q, z_e) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_p^2 I)$$

Noise level: $\sigma_p \in \{0, 1, 3, 5, 10\}$mm

기대:
- BC degradation > Ours degradation (noise robustness 입증)
- Real robot calibration noise (1-5mm) regime에 직접 mapping

#### Regime C: Systematic residual (current setup)
$$p_{\text{true}} = F_{\text{analytic}}(q, z_e) + \Delta_{\text{true}}(q, z_e)$$

기대: Ours-Learned > Ours-Analytic (learned residual 가치 입증)

### 2.3 비교 method

| Method | Manifold | Residual | 설명 |
|---|---|---|---|
| BC | ✗ | ✗ | Deterministic regression |
| DP-canonical | ✗ | ✗ | Official Diffusion Policy |
| DP-A | ✗ | ✗ | DP + channel-cond + $p_{\text{start}}$ |
| DP-C | ✗ | ✗ | DP-A + Euclidean classifier guidance |
| Projected-Analytic | projection | ✗ | $p \leftarrow F_{\text{analytic}}(q, z_e)$ post-hoc |
| **Ours-Analytic** | ✓ | ✗ | $\mathcal{M}_{\text{ana}}(z_e)$ 위 Riemannian diffusion |
| **Ours-Learned (V2)** | ✓ | ✓ | $\mathcal{M}_\phi(z_e)$ 위 Riemannian diffusion (V2 그대로) |

### 2.4 핵심 비교 논리

| 비교 | 기대 결과 | 의미 |
|---|---|---|
| Ours-Analytic ≈ Ours-Learned (Perfect FK) | 두 방법 거의 동일 | Learned residual 무관 regime에서 framework consistency |
| **Ours-Analytic > DP-C (Perfect FK)** | Ours 우월 | **Riemannian framework 자체 효과 isolate** |
| Ours-Analytic > Projected-Analytic | Ours 우월 | Intrinsic manifold > post-hoc projection |
| Ours-Learned > Ours-Analytic (Residual FK) | Learned 우월 | **Learned residual self-model 가치** |
| Ours-Learned > Ours-Analytic (OOD $z_e$) | 차이 더 큼 | Residual의 OOD 일반화 가치 |

### 2.5 Setup

| 항목 | 값 |
|---|---|
| Robot | Franka 7-DoF |
| Horizon | $H+1 = 16$ |
| $z_e$ training | $[0.05, 0.15]$ |
| $z_e$ test | $\{0.05, 0.10, 0.15\}$ in-dist |
| $z_e$ OOD | $\{0.20, 0.25\}$ |
| Target box | $[0.40, 0.50] \times [-0.05, 0.05] \times [0.40, 0.50]$ m |
| Demo modes | A/B 50:50 |
| IK | DLS + null-space rest bias |
| Training steps | 15k (V2와 동일) |
| Batch size | 64 |
| LR | $2 \times 10^{-4}$ + warmup |
| EMA | 0.999 |
| Seeds | 3개 |

### 2.6 Implementation

**Phase 1**: Perfect FK dataset 생성
```python
# In FrankaBimodalReachingDemo:
# Set Δ_true = 0 (no compliance)
# p_demo = FK_analytic(q_demo, z_e)
```

**Phase 2**: Ours-Analytic 학습
```python
# Same architecture as V2
# Self-model F_φ = FK_analytic (no Δ_φ)
# G_ana = I + J_F_analytic^T J_F_analytic
# Same Riemannian SGM, retraction, lift, multi-component guidance
```

**Phase 3**: Baseline parity
```python
# Same dataset (Perfect FK demos)
# BC, DP-canonical, DP-A, DP-C, Projected-Analytic 모두 retrain
```

**Phase 4**: Evaluation (n=512 per condition, seed 3개)

**Output files**:
- `outputs/franka_traj_unet_v2_analytic/ckpt_riemannian.pt`
- `outputs/franka_baseline_*_perfect/ckpt.pt` (perfect FK trained baselines)
- `outputs/diagnostic/analytic_self_model.json`
- `outputs/diagnostic/noise_robustness.json`

### 2.7 평가 metric

#### 기본 metric (모든 method)

| Metric | 정의 |
|---|---|
| pos_err | $\|F_{\text{true}}(q_H, z_e) - p_{\text{target}}\|$ |
| succ@{2,5,10}cm | Threshold success rate |
| frac_A | Mode A 비율 |
| frac_between_modes | Mode 사이 averaging 비율 |
| $W_1^A, W_1^B$ | Per-mode sliced Wasserstein |
| vel smoothness | $\sum_h \|q_{h+1} - q_h\|^2$ |
| accel smoothness | $\sum_h \|q_{h+1} - 2q_h + q_{h-1}\|^2$ |
| Joint violation | Joint limit violation rate |
| max $\|g\|$ | Manifold adherence |

#### 추가 metric

**Perfect FK**: Analytic manifold adherence
$$g_{\text{ana}}(q, p, z_e) = p - F_{\text{analytic}}(q, z_e)$$

**Residual FK**: True model error
$$\|F_\phi(q, z_e) - F_{\text{true}}(q, z_e)\|$$

**OOD**: Same metrics but evaluated at $z_e \in \{0.20, 0.25\}$

### 2.8 Output table 구조

#### Table A1: Perfect FK main comparison

| Method | pos_err ↓ | succ@2cm ↑ | succ@5cm ↑ | frac_A | between ↓ | $W_1$ ↓ | vel ↓ | viol ↓ | max $\|g_{\text{ana}}\|$ ↓ |
|---|---|---|---|---|---|---|---|---|---|
| BC | | | | | | | | | |
| DP-canonical | | | | | | | | | |
| DP-A | | | | | | | | | |
| DP-C | | | | | | | | | |
| Projected-Analytic | | | | | | | | | |
| **Ours-Analytic** | | | | | | | | | |
| Ours-Learned (V2) | | | | | | | | | |

#### Table A2: Perfect FK + noise robustness

| $\sigma_p$ (mm) | BC pos_err | DP-C pos_err | Projected pos_err | **Ours-Analytic** | Ours-Learned |
|---|---|---|---|---|---|
| 0 | | | | | |
| 1 | | | | | |
| 3 | | | | | |
| 5 | | | | | |
| 10 | | | | | |

같은 table을 succ@5cm, $W_1$, smoothness에 대해서도.

#### Table A3: Residual FK — learned residual value

| Method | pos_err ↓ | succ@5cm ↑ | $W_1$ ↓ | OOD $z_e$ pos_err ↓ | $\|F_\phi - F_{\text{true}}\|$ ↓ |
|---|---|---|---|---|---|
| Ours-Analytic | | | | | (N/A, $\Delta_\phi = 0$) |
| Ours-Learned (V2) | | | | | |

핵심: Ours-Learned > Ours-Analytic, 특히 OOD $z_e$에서.

---

## 3. Experiment 2: Ours Internal Ablation

### 3.1 목적

V2 성능이 어느 component에서 왔는지 분해.

### 3.2 Component decomposition

V2의 6 component:

| Component | 기호 | 설명 |
|---|---|---|
| Riemannian framework | R | $G^{-1}$, tangent lift, retraction |
| Channel conditioning | C | $p_{\text{target}}, p_{\text{start}}, z_e$ broadcast as channels |
| Goal guidance | G | Endpoint target attraction |
| Start guidance | S | Start point anchoring |
| Smoothness guidance | V | Velocity/trajectory smoothness |
| Learned residual | L | $\Delta_\phi$ |

### 3.3 Core ablation table (priority 1)

5개 핵심 ablation:

| Variant | R | C | G | S | V | L | 목적 |
|---|---|---|---|---|---|---|---|
| 1. Ours full V2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 기준점 |
| 2. Ours no guidance | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | Channel cond만의 효과 |
| 3. Ours + goal only | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | Goal guidance 효과 |
| 4. Ours + goal + start | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | Start anchoring 효과 |
| 5. Ours full analytic | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | Learned residual 효과 |

이 5개로 paper에서 다음 질문 답변:
- Guidance 없으면 얼마나 나빠지나?
- Goal guidance만으로 충분한가?
- Start anchoring 진짜 필요한가?
- Smoothness guidance 기여?
- Learned residual 효과 (Perfect FK + Residual FK)?

### 3.4 Extended ablation (priority 2)

추가 4개:

| Variant | R | C | G | S | V | L | 목적 |
|---|---|---|---|---|---|---|---|
| 6. Ours + smooth only | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | Smoothness alone 효과 |
| 7. Ours + start only | ✓ | ✓ | ✗ | ✓ | ✗ | ✓ | Start alone 효과 |
| 8. Ours + start + smooth | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | Trajectory coherence |
| 9. Euclidean chart version | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | Riemannian 제거 |

특히 9번 (Euclidean chart)이 중요. R 제거의 정확한 효과 측정.

### 3.5 V2 ablation table (이미 부분 존재)

V2 ablation table이 이미 있음 (REPORT.md §6.6.2):

| Cell | $\alpha_g/\alpha_s/\alpha_v/\alpha_a$ | e₀ | e_H | succ@2 | succ@5 | succ@10 |
|---|---|---|---|---|---|---|
| (f) no guidance | 0/0/0/0 | 0.080 | 0.080 | 7.0% | 36.7% | 73.0% |
| (a) endpoint only | 100/0/0/0 | 0.073 | 0.044 | 20.7% | 76.6% | 95.3% |
| (b) +start | 100/100/0/0 | 0.044 | 0.043 | 21.9% | 75.8% | 95.3% |
| (c) +start +vel | **100/100/5/0** | **0.024** | **0.022** | **47.7%** | **99.2%** | **100.0%** |
| (d) +start +vel +acc | 100/100/5/5 | 0.028 | 0.035 | 38.3% | 85.9% | 93.0% |

이걸 paper의 ablation table 일부로 활용. **Variant 1, 2, 3, 4가 이미 존재**. Variant 5 (Ours full analytic)와 9 (Euclidean chart)만 새로 진행.

### 3.6 Output table 구조

#### Table B1: Core ablation

| Variant | pos_err ↓ | succ@5cm ↑ | succ@10cm ↑ | $W_1$ ↓ | vel ↓ | max $\|g\|$ ↓ |
|---|---|---|---|---|---|---|
| 1. Ours full V2 | | | | | | |
| 2. Ours no guidance | | | | | | |
| 3. Ours + goal only | | | | | | |
| 4. Ours + goal + start | | | | | | |
| 5. Ours full analytic | | | | | | |

#### Table B2: Extended ablation (선택)

Same table, variants 6-9 포함.

---

## 4. DP Fairness Ablation 통합

### 4.1 이미 존재하는 결과

REPORT.md §11.5에서 검증된 4-way decomposition:

| Lever | pos_err 변화 |
|---|---|
| DP architecture (global → channel + $p_s$) | 333 → 320 mm (4% 감소) |
| DP classifier guidance (Euclidean) | 320 → 132 mm (60% 감소) |
| **Riemannian framework (DP-C → Ours-V2)** | **132 → 21 mm (84% 감소, 6.2x)** |

### 4.2 Analytic self-model 추가로 확장

기존 4-way + Ours-Analytic으로 5-way:

| Step | 변화 | 효과 |
|---|---|---|
| DP-canonical → DP-A | +channel cond | 11mm 감소 |
| DP-A → DP-C | +classifier guidance | 188mm 감소 |
| DP-C → **Ours-Analytic** | **+Riemannian framework (no residual)** | ? mm 감소 ← **NEW** |
| Ours-Analytic → Ours-Learned | +learned residual | ? mm 감소 ← **NEW** |

이 decomposition이 paper의 strongest claim isolation.

### 4.3 Integrated table (paper main)

| Method | Channel cond | Guidance | Riemannian | Residual | pos_err | succ@5cm | $W_1$ | max $\|g\|$ raw |
|---|---|---|---|---|---|---|---|---|
| DP-canonical | ✗ | ✗ | ✗ | ✗ | 333mm | 0% | 0.32 | 25mm |
| DP-A | ✓ | ✗ | ✗ | ✗ | 320mm | 0% | 0.31 | 25mm |
| DP-C | ✓ | Euclidean | ✗ | ✗ | 132mm | 0% | 0.34 | 25mm |
| **Ours-Analytic** | ✓ | Riemannian | ✓ | ✗ | ? | ? | ? | 0 |
| **Ours-Learned (V2)** | ✓ | Riemannian | ✓ | ✓ | 21mm | 99% | 0.017 | 0 |

이 table이 paper의 main results section의 핵심.

---

## 5. 실험 진행 순서

### Phase 1: Perfect FK dataset 생성
- 기존 `FrankaBimodalReachingDemo` 수정 ($\Delta_{\text{true}} = 0$ flag)
- 같은 IK process, 같은 mode 분리
- Validation: $\|p_{\text{demo}} - F_{\text{analytic}}(q_{\text{demo}}, z_e)\| = 0$

**Output**: `outputs/datasets/franka_perfect_fk/`

### Phase 2: Ours-Analytic 학습
- V2와 동일 architecture (channel cond + ConditionalUnet1D)
- $F_\phi = F_{\text{analytic}}$ (residual MLP 사용 안 함)
- $G_{\text{ana}} = I + J_{F_{\text{analytic}}}^T J_{F_{\text{analytic}}}$
- Training: 15k steps, seed 3개

**Output**: `outputs/franka_traj_unet_v2_analytic/ckpt_riemannian_seed{0,1,2}.pt`

### Phase 3: Perfect FK baselines 학습
- BC-perfect, DP-canonical-perfect, DP-A-perfect, DP-C-perfect, Projected-Analytic-perfect
- 같은 dataset (Perfect FK)
- 기존 baseline code 재사용 (true_model=analytic flag)

**Output**: `outputs/franka_baseline_*_perfect/ckpt_seed{0,1,2}.pt`

### Phase 4: Ours internal ablation
- Variant 5 (Ours full analytic) — Phase 2 결과 그대로
- Variant 9 (Euclidean chart version) — 새로 학습

**Output**: `outputs/franka_traj_unet_v2_euclidean/ckpt_seed{0,1,2}.pt`

### Phase 5: Noise robustness sweep
- $\sigma_p \in \{0, 1, 3, 5, 10\}$mm
- Train clean, eval noisy (minimum)
- Optionally: train noisy, eval noisy + train noisy, eval clean

**Output**: `outputs/diagnostic/noise_robustness.json`

### Phase 6: Comprehensive evaluation
- Mode capture test (8 ctxs × 64 samples × all methods)
- OOD generalization (target box shift, $z_e$ extension)
- All metrics 통합 평가

**Output**: 
- `outputs/diagnostic/analytic_self_model_full.json`
- `outputs/diagnostic/ablation_internal.json`
- `outputs/diagnostic/integrated_decomposition.json`

---

## 6. Implementation files (제안)

```
smcdp/experiments/
├── analytic/
│   ├── perfect_fk_dataset.py          # Phase 1
│   ├── train_ours_analytic.py         # Phase 2
│   ├── train_baselines_perfect.py     # Phase 3
│   ├── ablation_internal.py           # Phase 4
│   ├── noise_robustness.py            # Phase 5
│   └── eval_comprehensive.py          # Phase 6
└── analytic/configs/
    ├── ours_analytic.yaml
    ├── baselines_perfect.yaml
    └── ablation.yaml
```

---

## 7. Paper 통합 — 최종 claim 구조

### Claim 1: Analytic self-model manifold value
**Evidence**: Ours-Analytic > DP-C in Perfect FK

> Even when the analytic kinematic self-model is exact, Riemannian diffusion on the embodiment-conditioned self-model manifold improves multimodal trajectory generation over Euclidean diffusion baselines.

### Claim 2: Learned residual self-model value
**Evidence**: Ours-Learned > Ours-Analytic in Residual FK + OOD

> When the analytic self-model is imperfect, the learned residual self-model improves task success and OOD embodiment generalization.

### Claim 3: Riemannian framework value (load-bearing)
**Evidence**: DP-C → Ours-V2 large gap (132 → 21 mm) after architecture/guidance parity

> The improvement is not explained by channel conditioning or sampling-time guidance alone; it remains after matching both in a strengthened Diffusion Policy baseline.

### Claim 4: Guidance contribution (honest)
**Evidence**: Ours full > Ours no guidance

> Task-specific analytic guidance improves reach accuracy and temporal coherence, while the Riemannian framework ensures that this guidance remains compatible with the self-model manifold.

**중요 — 위험한 표현 회피**: "The diffusion model alone solves precise reaching" (이건 false; V2의 task success는 diffusion + guidance hybrid의 결과)

### Claim 5: Noise robustness
**Evidence**: Ours degradation < BC degradation in noise sweep

> Riemannian framework provides robustness against execution/observation noise, indicating real-world deployment relevance.

---

## 8. Reviewer 공격 시나리오 차단

| Reviewer Q | Evidence | Source |
|---|---|---|
| Q1: 왜 framework이 DP보다 좋은가? | DP-C ablation matched parity | §11.5 (existing) |
| Q2: Perfect FK 없이 framework 가치? | Ours-Analytic vs DP-C | Phase 2-3 (NEW) |
| Q3: Learned residual 진짜 필요? | Ours-Learned vs Ours-Analytic in OOD | Phase 6 (NEW) |
| Q4: Real robot validation? | Noise robustness sweep | Phase 5 (NEW) |
| Q5: BC가 task에서 더 좋은데? | Mode collapse + manifold adherence | Existing |
| Q6: V2 task success가 guidance 덕분 아닌가? | Ablation variant 2 (no guidance) | Phase 4 (NEW) |
| Q7: Standard benchmark에서? | Push-T 등 (future work) | Phase 7 (LATER) |

---

## 9. Checklist

### 필수 (Tier 1, paper main)
- [ ] Perfect FK dataset 생성
- [ ] Ours-Analytic 학습 (seed 3)
- [ ] DP-C-perfect 학습 + 평가
- [ ] BC-perfect, DP-canonical-perfect, DP-A-perfect, Projected-Analytic-perfect
- [ ] Ours-Analytic vs Ours-Learned 비교 (Perfect + Residual)
- [ ] Core ablation 5개 (Variants 1-5)
- [ ] Mean ± std 보고 (seed 3)

### 권장 (Tier 2, paper 강화)
- [ ] Noise robustness sweep ($\sigma_p \in \{0,1,3,5,10\}$mm)
- [ ] Extended ablation (Variants 6-9)
- [ ] OOD $z_e = 0.25$
- [ ] Euclidean chart version (Variant 9, R 제거 효과)

### 선택 (Tier 3, future work)
- [ ] Real robot validation
- [ ] SE(3) pose extension
- [ ] Standard benchmark (Push-T, Square, Block-Push)

---

## 10. 시간 추정

| Phase | 작업 | 시간 |
|---|---|---|
| 1 | Perfect FK dataset | 1-2일 |
| 2 | Ours-Analytic 학습 (seed 3) | 1주 |
| 3 | Baselines-perfect (5 method × seed 3) | 2주 |
| 4 | Ours ablation (variants 5, 9) | 1.5주 |
| 5 | Noise robustness | 1주 |
| 6 | Comprehensive eval | 0.5주 |

**Tier 1 only**: 4-5주
**Tier 1 + 2**: 6-7주
**Paper writing 병행**: +4주 → total 10주

---

## 11. Push-T 등 standard benchmark는 이후

### 11.1 Push-T 시점

**Yes, ablation 이후가 정확.** 이유:

1. **Framework contribution을 먼저 isolate**: Ablation 없이 Push-T에서 ours가 좋다고 해도 **왜** 좋은지 불명확.
2. **Push-T setup이 본인 framework에 fit하는지 검증 필요**: 
   - Push-T는 image-based + dynamics task
   - Self-model manifold는 kinematic
   - $z_e$ embodiment context 정의 어려움
   - Domain mismatch 가능성
3. **시간 효율**: Ablation은 기존 setup 그대로, Push-T는 새 setup (vision encoder, simulation 등).

### 11.2 Push-T 대안 검토

본인 framework이 Push-T에 fit하는지 신중 검토 필요. 대안:

**Option A**: Push-T 그대로 — Setup 비용 큼, fit 불확실
**Option B**: DP의 다른 benchmark — Franka-Kitchen, Lift, Square, Block-Push
**Option C**: 본인 setup의 second task — Constraint-critical reach, tool change deployment, self-collision-aware reach

### 11.3 권장 순서

```
Phase 1-6 (Ablation + Analytic) — 4-7주
  ↓
Paper writing 시작
  ↓
[paper 1차 완성]
  ↓
Phase 7 (Standard benchmark, optional)
  - 7a: Push-T or DP benchmark (framework fit 검토)
  - 7b: 또는 본인 setup의 second task
```

Phase 7은 paper의 추가 강화 또는 next paper의 main result.

---

## 12. 핵심 결론

이 plan의 academic 가치:

**(1) Framework contribution을 정량 isolate**:
- Geometric (Claim A) vs Learning (Claim B) 분리
- Riemannian framework이 load-bearing component임을 입증

**(2) Reviewer 공격 미리 차단**:
- DP fairness (existing) + Analytic self-model (new) + Internal ablation (new) + Noise robustness (new)
- 모든 typical attack에 정량 답변

**(3) Paper 강도 maximize**:
- Three-claim structure (geometric + learning + framework)
- Honest assessment (guidance hybrid)
- Robotics relevance (noise robustness)

**(4) Standard benchmark는 이후**:
- Ablation 후 framework fit 검토
- Phase 7로 paper 강화 또는 next paper

---

## 13. 결론 — 이 plan으로 도달할 paper

이 plan 후 paper의 main results 구조:

**§4.1 Mathematical framework** (existing)
- Manifold + metric + score lift + retraction
- Sanity 6 invariants verification

**§4.2 Main experiments — 7-DoF Franka**
- Setup
- Main results vs baselines (Table from REPORT §11.5 + new)
- **Perfect FK isolation (Ours-Analytic vs DP-C)** ← NEW
- **Internal ablation (5 variants)** ← NEW
- OOD generalization (existing + noise robustness NEW)

**§4.3 Controlled experiment — 3-DoF planar (D-2)**
- Categorical superiority (existing)

**§5 Discussion**
- Limitations (task simplicity 정직)
- Future work (Push-T, real robot, SE(3))

이 구조에서 본인 paper는 **mechanism-level + benchmark-level + ablation-level** 모두 균형. ML 학회 (NeurIPS, ICML, ICLR) paper로서 매우 강함.

본인 framework은 이미 옳다. 이 plan은 **그것을 reviewer가 공격할 수 없게 입증**하는 작업.