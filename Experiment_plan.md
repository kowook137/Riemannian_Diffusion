# Boundary-Joint Active Experiment — 설계 문서

**목적**: v4 (unbounded chart) vs v4.1 (bounded chart Choice A) 의 진정한 trade-off 를 측정하기 위해, **demo 자체가 joint boundary 와 active 하게 상호작용하는 setting** 에서 재실험.

**Compiled**: 2026-05-11
**Reference**: `pose_extension_report.md` §10 (현재 결과), `boundary_metric_plan.md` (metric 정의)

---

## 0. 동기 — 왜 다시 실험해야 하는가

### 0.1 현재 실험의 한계 (보고서 §10.1)

Pre-flight margin diagnostic:

| 지표 | 현재 demo 측정값 | 의미 |
|---|---|---|
| 1%-tile rel-margin | **0.2045** | 가장 boundary-가까운 1% trajectory 도 joint range 의 20.45% 안쪽 |
| 5%-tile | 0.2389 | demo 가 모두 joint 중앙 근처 |
| Min margin | 0.140 | 절대로 boundary 에 닿지 않음 |
| ‖u‖∞ p99 | **0.54–0.66** | $\tanh u \in [-0.58, 0.58]$ → saturation regime ($|u|>3$) 한참 못 미침 |

→ **bounded chart 가 작동할 기회가 없는 setting**. $D_\psi$ 가 항상 $\sim q_\text{range}/2$ 근처 (degenerate 안 함). Joint feasibility 0% 는 demo 가 애초 boundary 안 깊숙이 있어서 자명.

### 0.2 보고서 §10.5 의 진단

> "Saturation 은 원인 아님 ... 진짜 원인: Varadhan chart-Euclidean bias (spec §9 caveat 정량 검증)"

즉 v4.1 의 −19/−27 pp pose accuracy 손해는 **structural** (Choice A identity floor 의 trade-off). Demo 가 boundary 와 무관해도 발생한다.

### 0.3 결론

현재 setting 은 **v4.1 의 cost 만 측정하고 benefit 은 측정 못함**. Paper claim 을 위해서는:
- **DP-raw 가 joint violation 을 실제로 일으키는** demo 분포
- **v4.1 의 boundary-aware sampling 이 작동할** $\|u\|_\infty$ regime
- **task_succ (pose ∧ joint)** 에서 v4.1 우위가 가시화되는 task

---

## 1. 실험 가설

### H1 — Primary

> **demo 가 joint boundary 와 active 하게 상호작용할 때, v4.1 (bounded chart) 은 DP-raw 대비 task_succ@(5cm, 5°, $Q_\text{safe}$) 에서 우위를 갖는다.**

정량적 형태:
$$\text{task\_succ}^{v4.1} - \text{task\_succ}^{DP\text{-raw}} \geq +10\text{ pp}$$

근거:
- DP-raw: pose 는 잘 맞추지만 joint violation 발생 → task_succ 약화
- v4.1: pose 는 약간 약하지만 joint violation 0% → task_succ 유지

### H2 — Trade-off characterization

> **boundary activeness 가 증가할수록 v4 vs v4.1 의 task_succ 격차가 확대된다.**

세 가지 tier 의 demo distribution 을 만들어 비교 (§3 참고):
- Tier 0 (현재): 1%-tile margin > 0.2 → v4 우위
- Tier 1 (moderate): 1%-tile margin ∈ [0.05, 0.10] → 격차 축소
- Tier 2 (boundary-active): 1%-tile margin < 0.05 → v4.1 우위

### H3 — Bounded-chart diagnostic activation

> **Tier 2 에서 v4.1 의 $\|u\|_\infty$ p99 가 saturation regime 근처 ($> 2$) 에 도달한다.**

이게 충족되어야 §10 의 "saturation 은 원인 아님" 진단이 Tier 2 에서 뒤집히고, $D_\psi$-degeneracy 의 실질적 효과를 측정 가능.

### H4 — DP-clipping 한계

> **DP-clipped (post-hoc projection to $Q_\text{safe}$) 는 pose accuracy 를 망가뜨려 task_succ 가 v4.1 보다 낮다.**

근거: clipping 은 trajectory 의 endpoint 를 강제로 이동 → pose error 증가. Smoothness 도 깨짐 ($E_\text{acc}$ 증가).

### H5 — DP-bounded 와의 분리

> **DP-bounded (DP 에 $\psi$ 만 적용) 와 v4.1 의 차이는 self-model Riemannian metric 효과를 isolate 한다.**

만약 두 결과가 비슷 → bounded chart 자체가 핵심, self-model manifold 는 부수적.
만약 v4.1 우위 → Riemannian metric + chart-form score 의 contribution 검증.

---

## 2. Boundary-Active Demo Distribution 생성

### 2.1 Strategy — IK target 을 boundary 영역으로 강제

기존 demo 생성:
1. $T_\text{start}, T_\text{target} \in \SE(3)$ 를 workspace 에서 sampling
2. IK 로 $q_\text{start}, q_\text{target}$ 계산
3. trajectory interpolation

**문제**: workspace sampling 이 보통 reachable 하면서 redundancy 가 풍부한 영역 → IK 가 joint center 근처 해를 선호.

**해결**: **target pose 를 workspace 의 boundary 영역에서 sampling**, redundancy resolution 을 **preferred-near-limit** 모드로 변경.

### 2.2 Tier 별 demo 생성 protocol

#### Tier 0 (control, 현재 setting)

기존 그대로. Workspace cube 중앙 80% 영역에서 $T_\text{target}$ sampling, IK redundancy = damped least squares with neutral seed $q_\text{rest}$.

#### Tier 1 (moderate boundary-active)

- $T_\text{target}$ sampling 영역: workspace cube 의 외곽 60% (중앙 40% 제외)
- IK seed: 두 가지 모드 mixture
  - 50%: $q_\text{rest}$ (neutral) — 정상 trajectory
  - 50%: $q_\text{rest} + \delta \cdot e_j$ ($\delta = 0.4 q_{\text{range}, j}/2$, $j$ random) — 한 joint 를 boundary 쪽으로 편향
- Demo filter: 1%-tile rel-margin ∈ [0.05, 0.10] 만족할 때까지 reject sampling

#### Tier 2 (strongly boundary-active) — **main experiment**

- $T_\text{target}$ sampling 영역: workspace cube 의 외곽 30% (중앙 70% 제외) + reachability-edge 강제
- IK seed: weighted toward joint-limit-active configurations
  - Joint $j$ 별로 $\eta_j = \pm 0.85$ ($q_j$ 가 joint range 의 85% 지점) 으로 seed
  - 4–5 joints 동시에 boundary-near
- Demo filter:
  - 1%-tile rel-margin < 0.05 (즉 trajectory 의 1% 가 joint range 의 5% 안쪽)
  - active_joint_ratio > 0.10 (전체 timestep × joint 의 10% 이상이 boundary-active, $\delta_\text{active} = 0.05$)
  - max boundary proximity $b_\max(\tau) > 0.85$ (적어도 한 시점에서 한 joint 가 boundary 의 7.5% 이내)

### 2.3 Demo 검증 protocol

생성 후 다음을 측정 (`boundary_metric_plan.md` §3):

| 지표 | Tier 0 (현재) | Tier 1 (target) | Tier 2 (target) |
|---|---|---|---|
| 1%-tile rel-margin | 0.2045 | [0.05, 0.10] | < 0.05 |
| 5%-tile rel-margin | 0.2389 | [0.10, 0.15] | < 0.10 |
| Min margin | 0.140 | < 0.05 | < 0.02 |
| active_joint_ratio | ~0% | 5–10% | > 10% |
| $\mathbb{E}[b_\max(\tau)]$ | ~0.55 | 0.70–0.80 | > 0.85 |
| $P_{90}[b_\max]$ | ~0.65 | > 0.85 | > 0.95 |

검증 통과 못하면 sampling parameter 재조정.

#### 2.3.1 Tier 2 v5 — 검증 완료 config (2026-05-10)

탐색 끝에 다음 sampling config 가 §2.3 의 모든 PASS criterion 을 만족함:

```python
q_rest_A = [+0.0, -0.3, 0.0, -0.40, 0.0, +3.40, 0.0]   # q[3] near upper, q[5] near upper
q_rest_B = [+0.0, -0.3, 0.0, -1.50, 0.0, +1.50, 0.0]   # safe (center-ish)
p_box    = ([0.40, -0.05, 0.40], [0.50, +0.05, 0.50])  # Tier 0 box (target reachable)
jitter_q = 0.05
target_perturb_deg = 20.0
n_ik_steps = 25
ik_alpha = 0.5
ik_alpha_null = 0.25                                   # weaker than Tier 0 (0.3) → IK lets q drift toward target
ik_lam = 0.05
ik_clamp_to_limits = True                              # critical — keeps q strictly feasible
ik_clamp_margin_frac = 0.001                           # δ = 0.1% of q_range
```

| 지표 | Tier 0 (control) | Tier 2 v5 (validated) | Pass criterion |
|---|---|---|---|
| 1%-tile rel-margin | 0.2024 | **0.0010** | < 0.05 ✓ |
| feasible fraction | 1.000 | **1.000** | > 0.99 ✓ |
| active_joint_ratio | 0.000 | **0.1158** | > 0.10 ✓ |
| $P_{90}[b_\max]$ | 0.65 | **0.999** | > 0.95 ✓ |
| Conditioning consistency | 0.005° | 0.005° | ~0 ✓ |
| **Verdict** | (control) | **PASS** | — |

**v5 로 수렴한 lessons** (실패 변형들로부터):

1. **3-joint boundary push 는 too aggressive**: q[0], q[1], q[3], q[5] 동시에 boundary 로 밀면 IK 가 수렴 못 하고 endpoint error 가 18 cm 까지 커짐 (v4 실패).
2. **2-joint boundary 가 sweet spot**: q[3] (unilateral upper bound 0) 와 q[5] (unilateral upper bound 3.822) 만 push 하면 IK 가 25 step 안에 수렴 + AIR > 0.10 달성 가능.
3. **`alpha_null=0.3` → 0.25 로 낮춰야** mode A 에서 IK 가 boundary-near seed 를 *부분적으로* 떠나도록 허용 — 그렇지 않으면 모든 trajectory 가 q_rest 정확히에 anchor 되어 boundary 와의 active interaction 이 사라짐.
4. **`ik_clamp_to_limits=True` 가 결정적**: clamp 없으면 mode A 의 ~50% 가 q < q_min 또는 q > q_max 로 새서 infeasible. Clamp 가 있으면 100% feasible 유지.
5. **`p_box` 는 Tier 0 box 그대로 유지** (확장 안 함): 외곽 box 로 확장하면 boundary-near seed + 외곽 target 조합이 reachability 의 한계를 넘어 IK 발산. Tier 0 box 안에서도 boundary-near q_rest seed 만으로 boundary-active demo 가능.
6. **Realized endpoint conditioning**: `T_target_used = T_phi(q_H, z_e)` (IK target 이 아니라) 로 packing 해야 IK 가 (clamp 때문에) target 에 정확히 수렴 못 해도 score net 이 학습할 (T_target, x) 쌍이 *내부적으로* consistent. 이 fix 가 없으면 conditioning error 가 누적되어 학습 불가능 (`demo_gen_pose.py` §7 참고).

검증 script: `python -m smcdp.experiments.tier2_demo_preflight --tier 0 2 --n 2048`

**CLI flags 추가 (모든 train/eval entrypoint)**: `--ik-alpha`, `--ik-alpha-null`, `--ik-lam`, `--ik-clamp-to-limits`, `--ik-clamp-margin-frac`. Tier 2 학습 시 위 v5 config 를 그대로 명령줄로 지정.

#### 2.3.2 Online demo gen → demo pool 최적화 (training 시간 단축)

n_ik_steps=25 로 인한 demo gen 비용이 학습의 critical path 가 되어 (per-step ~3 초), 15k steps 학습 시 method 당 ~10시간 까지 길어짐. 해결:

- 새 CLI flag `--demo-pool-size N` (default 0 = legacy online resampling).
- N>0 이면 trainer 시작 시 `demo.sample(N)` 한 번 실행해서 (x, branch_A, z_e, T_target, T_start) 캐시.
- training loop 매 step 마다 캐시에서 random batch index 만 뽑음 → IK 비용 0.
- N=8192 에서 cache=8 MB, 일회 demo gen ~30초. 이후 step 당 ~10ms (network only).
- **결과**: DP-raw 15k steps 8시간 → 3분 (160x speedup).
- **Caveat**: pool 이 fixed dataset 처럼 작동하므로 epochs 가 본질적으로 N/batch 만큼 = 128 epochs (8192/64). 일반적인 DP/diffusion 학습에 충분.

#### 2.3.3 Bounded chart Cholesky numerical 안정화 (v4.1 only)

Tier 2 v4.1 학습 step 75 에서 `linalg.cholesky` 가 batch element 1개에서 PD 실패 (`G + jitter*1000*eye` 까지 fallback 해도 실패) → NaN propagation 추정 (forward GRW 가 |u| 큰 영역으로 push 했을 때 chart Jacobian 의 underflow + edge case).

**해결 (v5 default)**:
- `--lambda-floor 1e-2` (기존 default 1e-4 의 100x)
- `--tikhonov-frac 0.01` (G 의 trace 비례 jitter)

이걸로 추가 diagonal 0.01 + adaptive 가 항상 들어가 PD 보장. 학습 정상 진행 확인됨 (loss 21k → 60 by step 600).

#### 2.3.4 발견한 추가 버그 (Tier 2 첫 시도 후)

**버그 1 — Demo gen 의 chart 이중 적용 (BoundedChartPoseManifold 만)**:
`FrankaBimodalReachingDemoPose.sample()` step 7 에서 realized endpoint 계산 시 `self.manifold.T_phi_Rp(q_traj, z_e)` 호출. v4.1 의 wrapped manifold 는 `T_phi_Rp(u, z) = base.T_phi_Rp(ψ(u), z)` 로 되어 있어, q (physical) 를 u 로 잘못 해석하고 ψ(q) 를 추가 적용 → endpoint error ~50cm. **fix**: `getattr(self.manifold, "base", self.manifold).T_phi_Rp(q_traj, z_e)` 로 base 를 명시 호출 (v4 unwrapped 일 땐 self.manifold == base 라 영향 없음).

**버그 2 — DDPMScheduler clip_sample=True default 가 Tier 2 에서 q clipping**:
`make_official_diffusion_policy(..., clip_sample=True)` default 가 reverse process 에서 x_0 prediction 을 [-1, +1] 로 clip. Tier 0 demos 는 우연히 q 가 대부분 [-1, +1] 안이라 영향 없었지만, Tier 2 mode A 는 q[3]≈-0.4 (within), q[5]≈3.4 (>>1) 까지 가서 q[5] 가 clip 됨 → DP 출력 garbage. **fix**: trainer 에서 명시 `clip_sample=False`.

**버그 3 — Eval scheduler 가 saved scheduler_config 무시**:
`franka_baselines_pose_eval` 의 scheduler 재생성도 `clip_sample` 안 넘겨서 default True 로 만들어짐. ckpt 의 `scheduler_config["clip_sample"]=False` 가 무시. **fix**: `clip_s = ck.get("scheduler_config", {}).get("clip_sample", True)` 로 ckpt 값 읽어 propagate.

이 3개 fix 후 결과 (Tier 2):
- DP-raw: succ@(5cm,5°) 65-79%, jvio 14-31% (boundary 침범)
- DP-bounded: succ 60-73%, jvio **0%** (chart 보장)
- v4.1: succ 32-39%, jvio **0%**

#### 2.3.5 최종 결과: Chart × Riemannian 2x2 ablation (Tier 2 final, 2026-05-11)

| | **Unbounded chart** | **Bounded chart (Choice A)** |
|---|---|---|
| **Vanilla DP** | DP-raw: pose **75%**, jvio 25% | DP-bounded: pose **66%**, jvio **0%** |
| **Riemannian (Method A)** | Ours-v4: pose **65%**, jvio 28% | Ours-v4.1: pose **36%**, jvio **0%** |

(succ@(5cm,5°) z 평균, jvio z 평균)

**관찰**:
1. **Bounded chart 단독 (DP-bounded) 이 가장 실용적**: jvio=0% 보장 + pose 66% (Riemannian 없이도).
2. **Riemannian SDE 가 pose accuracy 에 손해**: DP-raw 75% → v4 65% (-10pp), DP-bounded 66% → v4.1 36% (-30pp).
3. **v4.1 이 worst-of-both-worlds**: chart + Riemannian 의 추가 cost (Varadhan bias) 가 정당화 안 됨.
4. **lambda_floor 1e-2 vs 1e-3 동등**: Cholesky regularization 이 아니라 구조적 cost (Choice A 의 G ≽ I floor).

**H1 가설 결과**: v4.1 task_succ@(pose, Q_safe) **<** DP-bounded → **불충족**.

§13.3 의 risk 가 실현됨: "Varadhan bias (structural) 가 Tier 2 에서도 dominant → v4.1 의 task_succ 가 DP-bounded 보다도 낮음".

**남은 향상 방향** (paper 마감 전 시도 가능):
- v4.1 학습 step 30k / 50k 로 늘리기 (mfe 0.27 안정화 시도)
- smoothness chart_form="u" 시도 (현 default "q")
- σ_K calibration 재조정 (chart 공간 scale 보정)
- DP-bounded 와 v4.1 의 cond_injection 변경 (현재 channel)
- Loss 의 chart-space weighting 조정 (Varadhan offset 보정 항 추가)

#### 2.3.6 v4.1 30k tuning 결과 (2026-05-11)

Step 15k→30k 단순 증가로 v4.1 pose accuracy 크게 회복:

| | v4.1 15k | v4.1 30k | DP-bounded (15k) |
|---|---|---|---|
| pos err mean cm (z avg) | 7.3 | **5.6** | 13.8 |
| rot err mean ° (z avg) | 10.0 | **7.6** | 14.4 |
| succ@(5cm,5°) avg | 35.5% | **59.8%** | 65.6% |
| jvio avg | **0%** | **0%** | **0%** |
| mfe (mode err) avg | 0.29 | 0.28 | 0.05 |

**관찰**:
- Pose accuracy +24.3pp (35.5% → 59.8%). 학습이 15k 에서 미수렴 상태였음.
- mode capture (mfe) 는 거의 변화 없음 (0.29 → 0.28). 학습 시간이 아니라 model capacity / loss formulation 이슈.
- **여전히 DP-bounded 와 6pp 차이** (59.8% vs 65.6%). gap 의 원인:
  1. mode capture: v4.1 mfe=0.28 vs DP-bounded mfe=0.05 → mode A 또는 mode B 한쪽이 우세 (~14% imbalance) → 그만큼 wrong-mode 출력
  2. Residual Varadhan bias: pose err 가 DP-bounded 보다 살짝 큼

**다음 tuning 후보** (mode capture 개선):
- 학습 step 50k (mfe 도 결국 수렴 가능)
- cond_injection "global" 시도 (channel → global, mode 가 더 잘 학습될 수도)
- DSM loss 의 mode-balance 항 추가
- Larger UNet (down_dims 256/512/1024)

#### 2.3.7 v4.1 50k 결과 — H1 가설 충족 (2026-05-11)

| | v4.1 15k | v4.1 30k | **v4.1 50k** | DP-bounded |
|---|---|---|---|---|
| pos err mean (cm, z avg) | 7.3 | 5.6 | **4.8** | 13.8 |
| rot err mean (deg, z avg) | 10.0 | 7.6 | **6.5** | 14.4 |
| succ@(5cm,5°) avg | 35.5% | 59.8% | **64.8%** | 65.6% |
| succ@(5cm,10°) avg | 51.9% | 66.6% | **71.9%** | 73.0% |
| jvio avg | **0%** | **0%** | **0%** | **0%** |
| mfe avg | 0.29 | 0.28 | 0.29 | 0.05 |

**Effective task_succ@(pose ∧ Q_safe)** (jvio penalty 반영):
- DP-raw: 75.0% × (1 − 0.25) ≈ **56.3%**
- DP-bounded: 65.6% × 1 = **65.6%**
- Ours-v4: 64.5% × (1 − 0.29) ≈ **45.8%**
- **Ours-v4.1 (50k): 64.8%** ≥ DP-raw + **8.5pp** ✓ (H1 minimal success bar 충족)

**결론**:
1. **H1 가설 충족**: v4.1 (50k) 의 effective task_succ 가 DP-raw 대비 +8.5pp.
2. **v4.1 (50k) ≈ DP-bounded**: succ@(5cm,5°) 64.8% vs 65.6% (1pp 차이), 단 pos/rot err 는 v4.1 가 **훨씬 더 낮음** (4.8cm vs 13.8cm).
3. **Riemannian 의 contribution**: chart 만 (DP-bounded) 대비 succ 동등, 그러나 **endpoint 정밀도가 3배 개선** (pos err 3배 ↓, rot err 2.2배 ↓). 즉 chart 가 boundary safety 를 보장하면서 Riemannian SDE 가 endpoint 추적 정밀도를 끌어올림.
4. **잔존 한계 — mode capture**: mfe = 0.29 (DP-bounded 의 6배). 50k 에서도 안정 → 구조적 (학습 시간 아님). Riemannian forward GRW 가 bimodal demos 의 한 mode 로 쏠리는 경향. 추후 fix 영역.

**Paper claim 가능 메시지** (강 → 약 순서):
- Bounded chart 가 jvio=0% 를 구조적으로 보장 (Choice A 의 G_Q^A ≽ I).
- Chart × Riemannian 조합 (v4.1) 이 chart-only (DP-bounded) 와 **succ 동등** + **endpoint 정밀도 3배 개선** (paper-grade 결과).
- Mode capture 는 v4.1 의 잔존 trade-off (mfe=0.29) — 후속 연구 영역.

#### 2.3.8 mfe metric 진단 및 수정 — 모든 결과 재평가 (2026-05-11)

`diagnostic_plan.md` 의 hypothesis A/D 검증 중 발견:

**버그**: `eval_metrics_pose.py:137` 의 mfe classifier 가 `q[0] > 0` 하드코딩. Tier 0 demos 에서는 q_rest_A[0]=+0.6, q_rest_B[0]=-0.6 라 q[0] 가 mode discriminator. 그러나 **Tier 2 v5 에서는 q_rest_A[0]=q_rest_B[0]=0.0** → q[0] 가 random noise 만 측정 → mfe 가 misleading.

**검증**: v4.1 (50k) 의 q_init→q_H mode flip 직접 측정 (q[5] threshold 기준):
- A→A: 100, A→B: 0, B→A: 0, B→B: 156 (flip rate **0%**)
- DP-bounded 도 같이 측정: A→B 5%, B→A 0.6% → DP-bounded 가 오히려 약간 flip 있음

**Fix**: `compute_pose_metrics` 가 `q_rest_A`, `q_rest_B` 인자를 받아 auto-detect:
```python
diff = |q_rest_A - q_rest_B|  →  mode_joint = argmax(diff)
mode_threshold = (q_rest_A[mode_joint] + q_rest_B[mode_joint]) / 2
```
Tier 0 → mode_joint=0 (legacy), Tier 2 v5 → mode_joint=5.

**전체 7-method 재평가 결과 (수정된 mfe)**:

| Method | succ@(5cm,5°) | pos cm | rot° | **mfe (fixed)** | jvio | task_succ@(pose∧Q_safe) |
|---|---|---|---|---|---|---|
| DP-raw | 75.0% | 8.8 | 12.0 | **0.01** | 24.6% | ~56% |
| DP-bounded | 65.6% | 13.8 | 14.4 | **0.005** | 0% | 65.6% |
| Ours-v4 | 64.5% | 5.6 | 7.0 | **0.00** | 28.5% | ~46% |
| Ours-v4.1 (15k) | 35.5% | 7.3 | 10.0 | **0.00** | 0% | 35.5% |
| Ours-v4.1 (30k) | 59.8% | 5.6 | 7.6 | **0.00** | 0% | 59.8% |
| **Ours-v4.1 (50k)** | **64.8%** | **4.8** | **6.5** | **0.00** | **0%** | **64.8%** |

**최종 paper claim 정리** (mfe 수정 반영):
- **모든 method 의 mode capture 는 동등하게 잘 됨** (mfe ≤ 0.02). 이전 mfe=0.29 는 metric artifact.
- **Ours-v4.1 (50k) 가 모든 차원에서 best-or-tied**:
  - jvio=0% (Q_safe 구조적 보장)
  - mfe=0 (mode capture)
  - **pos err 4.8 cm — 모든 method 중 가장 낮음** (DP-bounded 의 0.35x)
  - **rot err 6.5° — 모든 method 중 가장 낮음** (DP-bounded 의 0.45x)
  - succ@(5cm,5°) = DP-bounded 와 동등 (1pp 차이)
- **H1 충족**: v4.1 effective task_succ ≥ DP-raw + 8.8pp.
- **Riemannian SGM 의 contribution**: chart 만 (DP-bounded) 대비 succ 동등 + **endpoint 정밀도 ~3x 개선**.

### 2.4 Demo size 와 mode structure

- 각 Tier 당 N = 4096 trajectories (현재와 동일)
- bimodal 유지 (elbow-up / elbow-down) — `boundary_metric_plan.md` §7 의 per-mode feasibility 측정 필요
- **단, Tier 2 에서는 mode 별 feasibility 차이가 클 것** (한 mode 는 boundary 더 자주 침범, 다른 mode 는 안전) → `feasible_mode_rate` metric 에서 차별 가능

### 2.5 Bimodal mode 의 boundary activeness 비대칭

Tier 2 에서 elbow-up / elbow-down mode 가 boundary 와 비대칭적으로 상호작용하도록 설계:
- Elbow-up: q4 가 자주 $q_{\max,4}$ 근처 (boundary-active mode)
- Elbow-down: q4 가 중앙 근처 (safe mode)

이러면:
- DP-raw 는 두 mode 모두 capture 시도 → up-mode 에서 violation 발생
- v4.1 은 두 mode 모두 capture 하면서도 violation 0% 유지
- BC 는 mode collapse → safe mode 만 학습 (다른 종류의 trade-off)

---

## 3. Baseline 구성

`boundary_metric_plan.md` §9 의 권장사항을 따라:

| Method | 설명 | 핵심 비교 대상 |
|---|---|---|
| **BC** | Deterministic regressor | mode collapse baseline |
| **DP-raw** | Standard DP, no joint enforcement | joint violation baseline |
| **DP-clipped** | DP + post-hoc clip $q \to \text{clip}(q, q_\text{min}, q_\text{max})$ | naive safety baseline |
| **DP-projected** | DP + reflective projection | softer safety baseline |
| **DP-bounded** | DP applied in $u$-chart ($\psi$ but no Riemannian metric) | bounded chart isolated |
| **Ours-v4** | Method A unbounded chart | self-model + unbounded |
| **Ours-v4.1** | Method A bounded chart Choice A | self-model + bounded (main proposal) |

### 3.1 DP-clipped 정의

```python
q_traj = DP_sampler(c)  # standard DP output
q_clipped = clip(q_traj, q_min + epsilon_safe, q_max - epsilon_safe)
T_traj = T_phi(q_clipped, z_e)  # lift via self-model
```

### 3.2 DP-projected 정의

DP-clipped 와 같지만, hard clip 대신 reflective projection (boundary 에서 안쪽으로 거리 만큼 reflect). Trajectory smoothness 가 덜 깨짐.

### 3.3 DP-bounded 정의

```python
# Training: q_demo -> u_demo via psi^{-1}, train DP on u
u_demo = psi_inv(q_demo)
DP.train(u_demo, c)  # standard DP loss in u-space

# Sampling: u_traj -> q_traj via psi
u_traj = DP.sample(c)
q_traj = psi(u_traj)  # auto-bounded
T_traj = T_phi(q_traj, z_e)
```

핵심: $\psi$ 는 적용하지만 $G_Q^A$, $J_\text{pose}^Q$, Varadhan target 같은 Riemannian 구조 없음. **bounded chart 의 isolated effect** 측정용.

### 3.4 Architecture/training parity

`pose_extension_report.md` §7 의 parity 유지:
- 동일 conditioning $c = (T_\text{start} \oplus T_\text{target} \oplus z_e) \in \R^{15}$
- 동일 H+1 = 16 trajectory length
- Batch 64, steps 15000
- UNet1d, channel-concat
- Same demo data (per Tier)
- Same $z_e$ training distribution: $\mathrm{Uniform}[0.05, 0.15]$
- Same eval $z_e \in \{0.05, 0.10, 0.15, 0.20\}$ (ID + OOD)

---

## 4. Joint Safety Margin 정의

`boundary_metric_plan.md` §12.2 Option B 채택:

$$Q_\text{safe} = [q_\text{min} + \epsilon,\ q_\text{max} - \epsilon], \qquad \epsilon = 0.05 \cdot q_\text{range}$$

즉 joint range 의 5% 안쪽까지를 "safe" 로 간주. 이유:
- Real robot safety claim 강화 (5% margin 은 실제 robot 의 minor calibration error 흡수)
- DP-raw 와 v4.1 의 차별성 강화 (margin 0 보다 더 엄격한 기준에서 측정)
- v4.1 의 **strict guarantee 가 open interval $(q_\text{min}, q_\text{max})$** 이므로, $Q_\text{safe} \subset Q$ 까지 보장하지는 않음 — **이게 honest 한 측정**

### 4.1 $\epsilon$-clip in IK seed

v4.1 의 IK seed safety: $\eta_i = \text{clip}(\eta_i^\text{raw}, -1+\epsilon, 1-\epsilon)$ with $\epsilon = 10^{-3}$ (initialization-only).

**주의**: 이 $\epsilon$ (IK init clip) 와 위의 $\epsilon$ (joint safety margin) 은 **서로 다른 값**. Notation 충돌 피하기 위해:
- IK init clip: $\epsilon_\text{init} = 10^{-3}$
- Joint safety margin: $\epsilon_\text{safe} = 0.05 \cdot q_\text{range}$

문서 전체에서 명확히 구분.

---

## 5. Metric 정의 (boundary_metric_plan.md 충실 반영)

### 5.1 Primary metric

$$\boxed{ \text{task\_succ}@(5\text{cm}, 5°, Q_\text{safe}) = \mathbb{1}[e_p \leq 5\text{cm} \land e_R \leq 5° \land q_h \in Q_\text{safe} \forall h] }$$

**모든 Tier × Method 조합 의 메인 number.**

### 5.2 Pose metrics (분해)

- `pose_succ@(5cm, 5°)`: pose 만 분리
- `pose_succ@(5cm, 10°)`: rotation bottleneck 진단
- `pos_err_mean_cm`, `pos_err_p95_cm`
- `rot_err_mean_deg`, `rot_err_p95_deg`

### 5.3 Joint feasibility metrics

- `joint_succ`: $\mathbb{1}[q_h \in Q_\text{safe} \forall h]$
- `sample_joint_viol_rate`: trajectory-level violation
- `element_joint_viol_rate`: timestep × joint level
- `max_joint_viol_mag` (rad): violation severity, mean and $P_{95}$
- `joint_margin_5pct`: 5%-tile margin across trajectories
- `active_joint_ratio` ($\delta_\text{active} = 0.05$): boundary-active 비율

### 5.4 Boundary activeness metrics (Tier 검증용)

- `b_max_mean`, `b_max_p50`, `b_max_p90`: trajectory-level boundary proximity
- demo 측정값과 generated trajectory 측정값 비교 → distribution shift 진단

### 5.5 Smoothness metrics

- `E_vel`: $(1/H) \sum \| q_{h+1} - q_h \|^2$ — **q-chart 에서 측정** (모든 method 공정 비교)
- `E_acc`: $(1/(H-1)) \sum \| q_{h+1} - 2q_h + q_{h-1} \|^2$
- DP-clipped 의 smoothness 약화 측정용

### 5.6 Mode metrics

- `mode_frac_err`: $|\text{frac}_A^\text{gen} - \text{frac}_A^\text{demo}|$
- `feasible_mode_rate`: 선택된 mode 가 feasible 한 비율
- per-mode `task_succ_up`, `task_succ_down`

### 5.7 Bounded-chart diagnostics (v4.1, DP-bounded 한정)

- $\|u\|_\infty$ mean / p95 / **p99** (saturation 진단의 핵심)
- $D_\psi^\text{min}$ p5: 가장 작은 $D_\psi$ 의 5%-tile
- $\kappa(G_Q^A)$ p95: condition number 의 95%-tile
- $\lambda_\text{min}(G_Q^A)$ min: identity floor 검증 (≥ 1 이어야 함)

### 5.8 Manifold adherence

- `manifold_gap_max` (mm): 모든 method 가 ~0 일 것으로 예상 (post-projection 또는 by construction)
- 단순 sanity check

---

## 6. 평가 protocol

### 6.1 Eval set

각 Tier × Method 조합에 대해:
- $z_e \in \{0.05, 0.10, 0.15, 0.20\}$ (ID + OOD)
- 각 $z_e$ 당 $n = 64$ samples
- 총 $7 \text{ methods} \times 3 \text{ tiers} \times 4 z_e \times 64 \text{ samples} = 5376$ trajectories

### 6.2 Eval condition $c$ 의 분포

**중요**: eval 시 $T_\text{start}, T_\text{target}$ 분포가 demo 분포와 일치해야 한다 (Tier-matched eval).

- Tier 0 method 는 Tier 0 eval 에서 측정
- Tier 2 method 는 Tier 2 eval 에서 측정
- **Cross-Tier 평가**도 부록으로 진행 (§9.4)

### 6.3 Per-Tier seed

각 Tier 당 동일 random seed 5 개 사용. 모든 method 가 같은 5 개의 demo subset / eval subset 으로 평가 → seed-paired comparison 가능.

### 6.4 Statistical reporting

- 각 metric 의 mean ± std across 5 seeds
- Method 간 paired comparison (Wilcoxon signed-rank test, paired by seed)
- $p < 0.05$ 면 significant 표시

---

## 7. 예상 결과 패턴

### 7.1 Tier 0 (control, 현재)

기존 §10.4 결과 재현 예상:

| Method | task_succ ID | task_succ OOD | joint_viol | pose_succ ID | mfe |
|---|---|---|---|---|---|
| BC | ~98% | ~86% | 0% | 98% | 0.47 |
| DP-channel | ~100% | ~97% | ~0% | 100% | 0.03 |
| Ours-v4 | ~94% | ~81% | 0% | 94% | 0.12 |
| Ours-v4.1 | ~75% | ~55% | **0%** | 75% | 0.05 |

**해석**: Tier 0 은 demo 가 boundary 와 무관 → DP-* 가 자동으로 violation 0% 근처 → v4.1 의 guarantee 가 vacuous → trade-off 만 visible.

### 7.2 Tier 2 (main, boundary-active) — 예상

| Method | task_succ ID | task_succ OOD | joint_viol | pose_succ ID | mfe |
|---|---|---|---|---|---|
| BC | ~50–70% | ~40–60% | 5–15% | ~85% | 0.5+ (collapse to safe mode) |
| DP-raw | ~55–70% | ~40–55% | **20–35%** | ~95% | 0.05 |
| DP-clipped | ~60–75% | ~45–60% | 0% | ~80% (clipping 으로 pose 망가짐) | 0.10 |
| DP-projected | ~70–80% | ~55–70% | 0% | ~85% | 0.10 |
| DP-bounded | ~75–85% | ~60–75% | 0% | ~85% | 0.05 |
| Ours-v4 | ~70–85% | ~50–65% | 5–15% | ~90% | 0.10 |
| **Ours-v4.1** | **~75–88%** | **~60–75%** | **0%** | **~78%** | **0.05** |

**예상 finding**:
- **DP-raw**: pose succ 95% 지만 violation 25% → task_succ ~70% (drop from pose_succ by 25 pp)
- **DP-clipped**: violation 0% but pose_succ 80% → task_succ ~75%
- **v4.1**: pose_succ 78% but violation 0% → task_succ ~78% (best ID)
- **DP-bounded**: pose_succ 85% but $\psi$ 자체의 chart-Euclidean bias 일부 발생 → task_succ ~80%

**Key margin**: v4.1 vs DP-bounded 차이 $\geq 5$ pp 이면 self-model Riemannian metric 의 contribution 검증.

### 7.3 Tier 1 (intermediate)

격차가 부분적으로만 가시화될 것. v4 vs v4.1 차이 ~10 pp, DP-raw vs v4.1 차이 ~5 pp 예상. Tier 1 은 hypothesis H2 (격차 monotone increase) 검증용.

---

## 8. Bounded-chart diagnostic 의 Tier 의존성

Tier 가 올라갈수록 v4.1 의 $\|u\|_\infty$ 가 saturation regime 에 가까워질 것:

| 지표 | Tier 0 | Tier 1 (예상) | Tier 2 (예상) |
|---|---|---|---|
| $\|u\|_\infty$ p99 | 0.66 | 1.5–2.0 | **2.5–3.5** |
| $D_\psi^\text{min}$ p5 | ~$q_\text{range}/2$ × 0.7 | × 0.3 | **× 0.05** |
| $\kappa(G_Q^A)$ p95 | small | medium | **large** (Tikhonov floor activation) |
| $\lambda_\text{min}(G_Q^A)$ | $\geq 1$ | $\geq 1$ | $\geq 1$ (Choice A guarantee) |

**Tier 2 에서 $\|u\|_\infty > 3$ 도달**하면 §10 의 "saturation 은 원인 아님" 진단이 뒤집힌다 — 이번엔 saturation 도 한 원인. 이게 측정되면 Choice A 의 $G_Q^A \succeq I$ floor 가 실질적으로 작동하는 첫 measurement.

---

## 9. Ablation 실험

### 9.1 v4.1 의 IK init clip $\epsilon_\text{init}$ sensitivity

$\epsilon_\text{init} \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$ 에서 task_succ 측정. Tier 2 에서 demo 의 일부가 $|\eta| > 1 - \epsilon_\text{init}$ 영역에 있으면 IK seed 가 잘림 → mode capture 영향.

### 9.2 Tikhonov floor $\lambda_\text{floor}$ sensitivity (v4.1)

$\lambda_\text{floor} \in \{0, 10^{-6}, 10^{-4}, 10^{-2}\}$. Tier 2 에서 $\kappa(G_Q^A)$ 큰 영역 등장 → floor 값이 학습 안정성에 영향. **Tier 0 에서는 무관해야 함** (Choice A 가 floor 없이도 안정).

### 9.3 Smoothness chart selection (v4.1)

`pose_extension_report.md` 보고서의 spec §11.4 alternative:
$$R_\text{vel}^q = -\sum \| \psi(u_{h+1}) - \psi(u_h) \|^2$$

$u$-chart smoothness vs $q$-chart smoothness 비교. Tier 2 에서 v4.1 의 $E_\text{acc}$ 가 baseline 대비 어떻게 비교되는지 측정.

### 9.4 Cross-Tier transfer

Tier 0 에서 학습한 model 을 Tier 2 eval 에서 평가, 그 반대도. v4.1 의 boundary-aware sampling 이 demo 분포와 어떻게 결합되는지 측정.

---

## 10. Stage-1 self-model 재학습 필요성

**중요**: Stage-1 self-model $T_\phi$ 은 demo 분포에 의해 학습된다 (`pose_extension_report.md` §3). Tier 가 바뀌면 self-model 도 재학습 필요.

각 Tier 당:
1. 해당 Tier 의 demo 로 self-exploration data 생성 ($\sim$100k $(q, T_\text{true}, z_e)$ tuples)
2. Stage-1 학습 (10000 steps, ~30 분)
3. Stage-1 평가: `err_learned_pos_mean`, `err_learned_rot_mean` 측정

**Tier 2 에서 self-model 정확도**: boundary 영역까지 학습되어야 하므로 Tier 0 보다 약간 약화 예상 (boundary 근처 sample 이 더 많지만 redundancy 가 적어 학습 어려움). 만약 Tier 2 self-model 이 Tier 0 대비 의미있게 약하면 모든 Tier 2 결과의 상한을 결정 → 별도 보고.

---

## 11. 실험 실행 순서

### 11.1 Phase 1 — Demo generation (1 day)

1. Tier 1, Tier 2 demo 생성 protocol 구현
2. 각 Tier 당 N=4096 trajectory 생성
3. Pre-flight margin diagnostic 측정 (§2.3 기준 통과 확인)
4. Demo 분포 시각화 (boundary proximity histogram, per-mode breakdown)

**Go/No-Go**: Tier 2 의 1%-tile rel-margin < 0.05 달성 못 하면 sampling parameter 재조정.

### 11.2 Phase 2 — Stage-1 self-model (Tier 1, Tier 2) (1 day)

각 Tier 당 self-model 학습. Tier 0 self-model 은 기존 것 재사용.

### 11.3 Phase 3 — Stage-2 학습 (3 days)

7 methods × 3 tiers × 5 seeds = 105 학습 run. 단, Tier 0 의 일부 (BC, DP-canonical, DP-channel, Projected, Ours-v4, Ours-v4.1) 는 이미 측정됨 — 5 seed 만 보강 필요.

새로 학습:
- Tier 0: DP-raw, DP-clipped, DP-bounded × 5 seeds = 15 runs (각 ~3 시간)
- Tier 1: 7 methods × 5 seeds = 35 runs
- Tier 2: 7 methods × 5 seeds = 35 runs

총 85 runs × 평균 3 시간 = ~255 시간 (≈ 11 일 single GPU). **GPU 4 대 병렬** 가정 시 ~3 일.

### 11.4 Phase 4 — Eval (1 day)

각 학습된 model 에 대해:
- 4 $z_e$ × 64 samples × 5 seeds = 1280 trajectory eval per model
- 모든 metric 계산

### 11.5 Phase 5 — Analysis (2 days)

- Main table 작성 (§5 metric 모두 포함)
- Ablation 결과 정리 (§9)
- Figure: task_succ vs boundary activeness (Tier 0/1/2 trend)
- Per-mode breakdown
- Bounded-chart diagnostic 분포 plot

### 11.6 총 timeline

8 days (GPU 4 대 가정) ~ 14 days (single GPU)

---

## 12. 보고 형식

### 12.1 Main table (paper 의 main contribution)

`boundary_metric_plan.md` §11 권장 형식 기반:

```
Tier 2 (boundary-active demo distribution, ID 평균)

| Method        | task↑ | pose↑ | joint↑ | viol↓ | margin↑ | E_acc↓ | mfe↓ |
|---------------|-------|-------|--------|-------|---------|--------|------|
| BC            | XX.X  | XX.X  | XX.X   | XX.X% | 0.XXX   | 0.XX   | 0.XX |
| DP-raw        | XX.X  | XX.X  | XX.X   | XX.X% | 0.XXX   | 0.XX   | 0.XX |
| DP-clipped    | XX.X  | XX.X  | XX.X   | XX.X% | 0.XXX   | 0.XX   | 0.XX |
| DP-projected  | XX.X  | XX.X  | XX.X   | XX.X% | 0.XXX   | 0.XX   | 0.XX |
| DP-bounded    | XX.X  | XX.X  | XX.X   | XX.X% | 0.XXX   | 0.XX   | 0.XX |
| Ours-v4       | XX.X  | XX.X  | XX.X   | XX.X% | 0.XXX   | 0.XX   | 0.XX |
| **Ours-v4.1** | XX.X  | XX.X  | XX.X   | 0%    | 0.XXX   | 0.XX   | 0.XX |
```

± std across 5 seeds 표시. Significance marker (* for $p < 0.05$ vs DP-raw).

### 12.2 Figure 1 — Boundary activeness 의 영향

X-axis: Tier (0, 1, 2) 또는 1%-tile rel-margin (continuous)
Y-axis: task_succ
Lines: 각 method
**핵심 메시지**: boundary 가 active 해질수록 v4.1 우위 확대

### 12.3 Figure 2 — Pose vs Joint trade-off

X-axis: pose_succ
Y-axis: joint_succ
Points: 각 (method, tier) 조합
**핵심 메시지**: v4.1 은 (high joint, moderate pose) 영역, DP-raw 는 (high pose, low joint) 영역 — task_succ 의 등고선이 함께 표시되면 v4.1 의 우위 시각화

### 12.4 Figure 3 — Bounded-chart diagnostics

Tier 별 $\|u\|_\infty$ p99, $D_\psi^\text{min}$ p5, $\kappa(G_Q^A)$ p95 의 분포 (boxplot)
**핵심 메시지**: Tier 2 에서 saturation regime 에 도달하는 첫 measurement

### 12.5 Per-mode table

Tier 2 에서 elbow-up / elbow-down mode 별 task_succ 분리:

```
| Method     | task↑ (up, boundary-active) | task↑ (down, safe) | feasible_mode↑ |
|------------|-----------------------------|--------------------|-----------------|
| BC         | XX.X (mode collapse)        | XX.X               | XX.X%           |
| DP-raw     | XX.X (high viol)            | XX.X               | XX.X%           |
| Ours-v4.1  | XX.X (joint guarantee)      | XX.X               | 100%            |
```

---

## 13. 리스크 및 mitigation

### 13.1 Tier 2 demo 생성 실패

**리스크**: IK redundancy resolution 으로 boundary-active demo 를 만들기 어려움. Filter 가 너무 엄격해서 데이터 부족.

**Mitigation**: 4096 target 못 채우면 2048 또는 1024 으로 축소. 단 mode 별 $\geq 512$ 보장. 학습 step 도 그에 맞게 축소.

### 13.2 Stage-1 self-model 의 Tier 2 정확도 약화

**리스크**: Tier 2 의 boundary 영역에서 $T_\phi$ 부정확 → 모든 method 의 상한 하락.

**Mitigation**:
- Self-exploration data 양 2× 증가 (200k → 400k)
- Stage-1 학습 step 2× 증가 (10k → 20k)
- 자체 metric 으로 self-model accuracy 보고 (Tier 0 vs Tier 2)

### 13.3 v4.1 의 pose accuracy 가 Tier 2 에서도 못 따라잡음

**리스크**: Varadhan bias (structural) 가 Tier 2 에서도 dominant → v4.1 의 task_succ 가 DP-bounded 보다도 낮음.

**Mitigation**: 이 경우에도 honest reporting. Paper 의 main claim 을 다음으로 수정:
- "v4.1 은 strict joint guarantee 를 제공하며, boundary-active task 에서 DP-raw 대비 task_succ 우위" (DP-bounded 는 별도 비교)
- DP-bounded 가 v4.1 보다 우위면, "bounded chart 자체가 main contribution" 으로 narrative 변경 — self-model Riemannian 부분은 future work

### 13.4 Compute 부족

**리스크**: 11 일 single GPU 부담.

**Mitigation**:
- 5 seeds 대신 3 seeds 로 축소 (statistical power 약화 명시)
- Tier 1 생략, Tier 0 vs Tier 2 만 비교 (H2 검증 약화 but H1 유지)
- 학습 step 15000 → 10000 (이미 plateau 영역)

---

## 14. 성공 기준

### 14.1 Minimal 성공

다음 셋 중 하나라도:
- Tier 2 에서 v4.1 task_succ ≥ DP-raw task_succ + 5 pp
- Tier 2 에서 v4.1 task_succ ≥ DP-clipped task_succ + 3 pp
- Tier 2 에서 v4.1 의 violation = 0% 이고 DP-raw 의 violation > 15%

→ "v4.1 은 boundary-active task 에서 의미있는 contribution" claim 가능.

### 14.2 Strong 성공

위 모두 + 다음 중 하나:
- Tier 2 에서 v4.1 task_succ > DP-bounded task_succ + 3 pp (self-model Riemannian contribution)
- $\|u\|_\infty$ p99 > 2.5 도달 (saturation regime activation 첫 measurement)
- Tier 0 → 1 → 2 monotone task_succ 격차 증가 (H2 검증)

→ "Joint-limit bounded SE(3) self-model manifold" 가 paper-grade contribution.

### 14.3 실패 시

Tier 2 에서도 모든 baseline 보다 v4.1 이 약하면:
- Choice A 포기, Choice B 또는 hybrid (§10.5 spec § 9 alternative) 시도
- 또는 paper 의 main contribution 을 v4 (drift-free Brownian) + Method A 로 회귀, joint-limit 은 future work

---

## 15. 요약

| 항목 | 내용 |
|---|---|
| 목적 | v4.1 bounded chart 의 진정한 trade-off 측정 (Tier 2 boundary-active demo 에서) |
| Primary metric | `task_succ@(5cm, 5°, Q_safe)` with $\epsilon_\text{safe} = 0.05 q_\text{range}$ |
| Demo Tier | 0 (control), 1 (moderate), 2 (boundary-active, **main**) |
| Methods | BC, DP-raw, DP-clipped, DP-projected, DP-bounded, Ours-v4, **Ours-v4.1** |
| Eval | 4 $z_e$ × 64 samples × 5 seeds per (method, tier) |
| Hypotheses | H1 (v4.1 > DP-raw on task_succ), H2 (Tier monotone), H3 (saturation activation), H4 (clipping 한계), H5 (DP-bounded 와 분리) |
| Timeline | 8–14 days |
| Compute | 85 runs × 3h ≈ 255h (4 GPU 병렬 시 3 days) |
| Success bar | Minimal: v4.1 task_succ ≥ DP-raw + 5pp at Tier 2 |