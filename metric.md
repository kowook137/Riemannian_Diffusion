# SMCDP Pose-Extended: Baseline Comparison Metrics

**목적**: Pose-extended SMCDP의 baseline 비교 평가 지표 정의. Task success뿐 아니라 framework의 핵심 claim (self-model pose consistency, multimodality, $z_e$ generalization, trajectory quality) 모두를 측정.

**평가 4축**:

$$\boxed{\text{Task success} \;+\; \text{Self-model pose consistency} \;+\; \text{Multimodal distribution fidelity} \;+\; \text{Trajectory / physical sanity}}$$

---

## 1. Primary metrics (필수)

### 1.1 Position error

End-effector position error:

$$e_p = \|p_H - p^*\|_2$$

단위는 **cm**로 보고.

**보고 항목**:
- mean
- median
- std
- max

**Success thresholds**:

$$\text{pos\_succ@5cm} = \mathbb{1}[e_p < 5\text{ cm}]$$

$$\text{pos\_succ@2cm} = \mathbb{1}[e_p < 2\text{ cm}]$$

### 1.2 Rotation geodesic error

**Quaternion L2 error는 사용 금지**. 반드시 $SO(3)$ geodesic angle.

**Rotation matrix 기준**:

$$e_R = \|\Log_{SO(3)}(R_H^\top R^*)\|_2$$

**Quaternion 기준** (등가):

$$e_R = 2\cos^{-1}(|\bar{q}_H^\top \bar{q}^*|)$$

단위는 **degree**로 보고.

**보고 항목**:
- mean
- median
- std
- max

**Success thresholds**:

$$\text{rot\_succ@5deg} = \mathbb{1}[e_R < 5°]$$

$$\text{rot\_succ@10deg} = \mathbb{1}[e_R < 10°]$$

### 1.3 Pose success (가장 중요한 primary metric)

**Strict full-pose success**:

$$\text{pose\_succ@(5cm,5deg)} = \mathbb{1}[e_p < 5\text{ cm} \;\land\; e_R < 5°]$$

**Relaxed thresholds**:

$$\text{pose\_succ@(5cm,10deg)}, \quad \text{pose\_succ@(2cm,5deg)}$$

**추천 primary reporting**:

| Metric | 역할 |
|---|---|
| pos_succ@5cm | position target 달성 |
| rot_succ@5deg | orientation target 달성 |
| pose_succ@(5cm,5deg) | strict full-pose success |
| pose_succ@(5cm,10deg) | relaxed full-pose success |

**주의**: 기존 결과의 `succ@5cm`이 position-only인지 pose-combined인지 모호. 반드시 분리 보고.

---

## 2. Self-model pose consistency / Manifold adherence

**Framework의 핵심 claim**. Baseline과의 차이를 가장 잘 보여줌.

### 2.1 Pose manifold constraint

$$g_\phi^\text{pose}(x, z_e) = \begin{bmatrix} p - p_\phi(q, z_e) \\ \Log_{SO(3)}(R_\phi(q, z_e)^\top R) \end{bmatrix}$$

### 2.2 Manifold gap metrics

**Position gap**:

$$e_{\text{mani}, p} = \|p - p_\phi(q, z_e)\|$$

**Rotation gap**:

$$e_{\text{mani}, R} = \|\Log_{SO(3)}(R_\phi(q, z_e)^\top R)\|$$

**Trajectory-level**:

$$\max_h e_{\text{mani}, p}(h), \qquad \max_h e_{\text{mani}, R}(h)$$

### 2.3 Weighted pose gap (combined)

$$e_{\text{mani}, \text{pose}} = \sqrt{\frac{1}{\sigma_p^2}\|p - p_\phi(q, z_e)\|^2 + \frac{1}{\sigma_R^2}\|\Log(R_\phi^\top R)\|^2}$$

**보고 항목**:

| Metric | 단위 |
|---|---|
| max manifold position gap | mm 또는 cm |
| mean manifold position gap | mm |
| max manifold rotation gap | deg |
| mean manifold rotation gap | deg |

### 2.4 Baseline별 예상 결과

- **Ours (SMCDP)**: retraction에 의해 $\max\|g_\phi\| \approx 10^{-5}$ 수준 (by construction).
- **DP-pose (ambient output)**: $T \neq T_\phi(q, z_e)$일 수 있음, gap 큼.
- **Projected-pose**: raw output gap과 projected output gap을 둘 다 보고.

### 2.5 Quaternion validity (auxiliary)

Baseline이 quaternion을 직접 출력하는 경우:

$$e_{\text{quat-norm}} = \big| \|\bar{q}\|_2 - 1 \big|$$

**보고 항목**:
- mean quaternion norm error
- max quaternion norm error
- invalid quaternion rate

$$\text{invalid\_quat\_rate} = \Pr[|\|\bar{q}\| - 1| > \epsilon]$$

**주의**:
- Quaternion norm error는 **auxiliary metric**.
- Main rotation metric은 반드시 $SO(3)$ geodesic error.
- Ours는 quaternion을 $SO(3)$ representation으로 normalize하므로 거의 0.

---

## 3. Multimodality / Distribution fidelity metrics

**Redundant kinematics에서 mode collapse 회피**가 framework claim.

### 3.1 Mode fraction

Mode 분류 (예: elbow-up / elbow-down):

$$\text{frac}_{\text{up}} = \frac{\#\text{samples classified as elbow-up}}{N}$$

Demo distribution이 0.5라면:

$$\text{mode frac err} = |\text{frac}_{\text{up}}^{\text{gen}} - 0.5|$$

**보고 항목**:

| Metric | 의미 |
|---|---|
| frac_up | mode balance |
| mode KL | demo mode distribution과의 차이 |
| mode collapse rate | 한 mode에 90% 이상 몰림 |

### 3.2 Between-mode averaging rate

Mode 사이 평균 영역 측정:

$$\text{frac\_between} = \Pr[|q_{2, \text{mid}}| < \epsilon]$$

이 값이 크면 mode averaging 발생. BC-pose에서 deterministic regression의 경우 큼.

### 3.3 Sliced Wasserstein distance

**Joint-space**:

$$W_1^q = W_1(q_{0:H}^{\text{gen}}, q_{0:H}^{\text{demo}})$$

**Pose-space**:

$$W_1^T = W_1^p + \lambda_R W_1^R$$

(rotation은 geodesic distance 기반)

**추천 항목**:
- chart $q$-space sliced $W_1$
- endpoint pose $W_1$
- mode-conditioned $W_1$

---

## 4. Trajectory quality metrics

**Pose target을 맞춰도 trajectory가 흔들리면 안 됨**.

### 4.1 Joint velocity smoothness

$$E_\text{vel} = \frac{1}{H} \sum_{h=0}^{H-1} \|q_{h+1} - q_h\|^2$$

### 4.2 Joint acceleration smoothness

$$E_\text{acc} = \frac{1}{H-1} \sum_{h=1}^{H-1} \|q_{h+1} - 2q_h + q_{h-1}\|^2$$

### 4.3 Pose trajectory smoothness

**Position velocity**:

$$E_{\dot{p}} = \sum_h \|p_{h+1} - p_h\|^2$$

**Angular velocity proxy**:

$$E_\omega = \sum_h \|\Log_{SO(3)}(R_h^\top R_{h+1})\|^2$$

→ Rotation trajectory의 jitter 측정.

---

## 5. Physical feasibility metrics

**Manifold가 physical feasibility를 보장하지 않으므로 별도 보고**.

### 5.1 Joint limit violation

**Rate**:

$$\text{joint\_viol} = \Pr[\exists h, i : q_{h, i} < q_{\min, i} \lor q_{h, i} > q_{\max, i}]$$

**Magnitude**:

$$e_\text{jl} = \sum_{h, i} \max(0, q_{h, i} - q_{\max, i})^2 + \max(0, q_{\min, i} - q_{h, i})^2$$

### 5.2 Joint margin

Joint limit 근접도:

$$m_\text{joint} = \min_{h, i} \frac{\min(q_{h, i} - q_{\min, i},\, q_{\max, i} - q_{h, i})}{q_{\max, i} - q_{\min, i}}$$

이 값이 작으면 limit 근처.

---

## 6. OOD $z_e$ robustness

Pose extension에서 $z_e$별 표가 필수.

### 6.1 $z_e$-wise reporting table

| $z_e$ | pos_err | rot_err | pos_succ | rot_succ | pose_succ | manifold gap | mode frac | joint viol |
|---|---|---|---|---|---|---|---|---|
| 0.05 | | | | | | | | |
| 0.10 | | | | | | | | |
| 0.15 | | | | | | | | |
| 0.20 | | | | | | | | |

**관찰 패턴**: $z_e$ 증가 시 strict pose success가 빠르게 떨어지지만 continuous error는 천천히 변함. 둘 다 보고해야 trade-off 보임.

---

## 7. Baseline별 강조 metrics

### 7.1 BC-pose

Deterministic regression일 가능성 높음.

**주요 비교**:
- pos_err, rot_err
- pose_succ
- mode collapse rate
- $W_1^q$
- trajectory smoothness

**예상**:
- Pose error는 낮을 수 있음
- Multimodal capture 약함
- Per-context mode collapse 가능

### 7.2 DP-pose / Flow-pose

**가장 중요한 baseline**.

#### A. Joint-only DP-pose

Output: $q_{0:H}$. Condition: $T_\text{start}, T_\text{goal}, z_e$. Pose는 FK/self-model로 계산.

**비교 metric**:
- pose_succ
- mode fraction
- $W_1^q$
- $z_e$ OOD
- inference time

(Manifold gap은 자동 0이므로 차이는 metric/guidance/score geometry에서 발생)

#### B. Ambient pose DP

Output: $(q_{0:H}, p_{0:H}, \bar{q}_{0:H})$ 동시 출력.

**비교 metric**:
- $T$ vs $T_\phi(q, z_e)$ gap (manifold consistency)
- Quaternion norm error
- pose_succ
- Mode fidelity

### 7.3 Projected-pose diffusion

**Raw와 projected를 분리 보고**.

| Metric | Raw output | After projection |
|---|---|---|
| pose_succ | ? | ? |
| manifold gap | 클 수 있음 | 작음 |
| trajectory smoothness | 원래 | projection 후 변형 |
| mode fidelity | 원래 | projection 후 왜곡 가능 |

**핵심 관찰**: projection이 success를 올리지만 distribution이나 smoothness를 얼마나 왜곡하는가.

---

## 8. 최종 reporting tables

### 8.1 Main comparison table

| Method | pos_err cm ↓ | rot_err deg ↓ | pose_succ 5cm/5° ↑ | mode frac err ↓ | $W_1^q$ ↓ | max pose gap ↓ | joint viol % ↓ |
|---|---|---|---|---|---|---|---|
| BC-pose | | | | | | | |
| DP-pose (joint) | | | | | | | |
| DP-pose (ambient) | | | | | | | |
| Projected-pose (raw) | | | | | | | |
| Projected-pose (proj) | | | | | | | |
| **Ours (SMCDP)** | | | | | | | |

여기서:

$$\text{mode frac err} = |\text{frac}_\text{up}^\text{gen} - \text{frac}_\text{up}^\text{demo}|$$

$$\text{max pose gap} = \max_h \sqrt{\|p_h - p_\phi(q_h)\|^2 + \lambda_R e_R^2}$$

### 8.2 Secondary table: $z_e$ generalization

| $z_e$ | Method | pos_err | rot_err | pose_succ 5cm/5° | pose_succ 5cm/10° | mode frac | joint viol |
|---|---|---|---|---|---|---|---|
| 0.05 | Ours | | | | | | |
| 0.05 | DP-pose | | | | | | |
| 0.10 | Ours | | | | | | |
| 0.10 | DP-pose | | | | | | |
| 0.15 | ... | | | | | | |
| 0.20 | ... | | | | | | |

### 8.3 Ablation table

| Setting | pos_err | rot_err | pose_succ | mode frac | Interpretation |
|---|---|---|---|---|---|
| Method A (full) | baseline | baseline | baseline | baseline | full design |
| $\sigma_K = 0.6$ | | | | | terminal scale mismatch |
| proxy_std = ou | | | | | training scale mismatch |
| $q_\text{init} = \mu_q$ | | | | | global anchor mismatch |

**기존 ABL3 결과**: $q_\text{init} = \mu_q$에서 평균 success $31.6\% \to 12.5\%$로 떨어짐. Ablation으로 매우 설득력 있는 evidence.

---

## 9. 우선순위 5개 (공간 부족 시)

논문 공간이 부족할 때 다음 5개 우선:

1. **pose_succ@(5cm, 5deg)** — Full pose target success
2. **pos_err mean / rot_err mean** — Continuous error
3. **max pose manifold gap** — Self-model consistency claim
4. **mode fraction error 또는 $W_1^q$** — Multimodality claim
5. **joint violation rate** — Physical feasibility limitation 관리

이 5개로 framework의 핵심 claim 대부분 커버.

---

## 10. 최종 metric set 요약

Quaternion 포함 pose extension baseline 비교에서 framework claim을 모두 살리려면:

| 축 | Metrics |
|---|---|
| **Task success** | $e_p$, $e_R$, pose_succ@(5cm,5°), pose_succ@(5cm,10°) |
| **Self-model pose consistency** | max/mean pose manifold gap, quaternion norm error |
| **Multimodal distribution fidelity** | mode fraction error, $W_1^q$ |
| **Trajectory / physical sanity** | velocity/acceleration smoothness, joint violation rate |
| **OOD generalization** | $z_e$-wise table for all above |

이렇게 잡으면 DP-pose나 Projected-pose가 "pose를 맞출 수 있다" 하더라도 SMCDP가 **pose consistency, mode fidelity, $z_e$-conditioned manifold adherence**에서 어떤 차이를 만드는지 명확히 비교 가능.

---

## 부록 A: 단위 통일 규칙

| Quantity | Unit | 비고 |
|---|---|---|
| Position | cm (display), m (computation) | $\sigma_p = 0.05$m = 5cm |
| Rotation | deg (display), rad (computation) | $\sigma_R$ task-dependent |
| Joint angle | rad | Franka spec |
| Joint velocity | rad / step | Δr-normalized 가능 |
| Manifold gap (position) | mm | small scale 강조 |
| Manifold gap (rotation) | deg | $SO(3)$ geodesic |

## 부록 B: Metric 분류 체크리스트

### Primary (반드시 보고)
- [ ] pos_err mean/median/std/max
- [ ] rot_err mean/median/std/max
- [ ] pos_succ@5cm, pos_succ@2cm
- [ ] rot_succ@5deg, rot_succ@10deg
- [ ] pose_succ@(5cm,5deg)
- [ ] pose_succ@(5cm,10deg)

### Self-model consistency (Ours의 차별성)
- [ ] max manifold position gap
- [ ] mean manifold position gap
- [ ] max manifold rotation gap
- [ ] mean manifold rotation gap
- [ ] quaternion norm error (baseline용)

### Multimodality
- [ ] frac_up (or mode fraction)
- [ ] mode frac err
- [ ] mode collapse rate
- [ ] frac_between
- [ ] $W_1^q$ (sliced)
- [ ] $W_1^T$ (sliced, optional)

### Trajectory quality
- [ ] $E_\text{vel}$
- [ ] $E_\text{acc}$
- [ ] $E_{\dot{p}}$
- [ ] $E_\omega$

### Physical feasibility
- [ ] joint_viol rate
- [ ] $e_\text{jl}$ (magnitude)
- [ ] joint margin

### OOD $z_e$
- [ ] $z_e$-wise table (all primary metrics)

### Inference cost (optional)
- [ ] Inference time per trajectory
- [ ] Sampling steps required