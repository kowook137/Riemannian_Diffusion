# Forward SDE의 수치 안정성 문제: 진단 및 해결 방안

**Context**: SMCDP (Self-Model Conditional Diffusion Policy) framework의 pose-extended formulation에서 forward Langevin drift가 활성화될 때 발생하는 numerical instability 분석 및 해결책.

---

## 1. 문제 진단

### 1.1 관찰된 현상

- Forward Langevin drift를 활성화한 상태에서, $q_r$이 Franka joint range 밖으로 누적 drift됨.
- 이로 인해 `pytorch_kinematics`가 비정상 Jacobian $J_F$를 반환.
- 결과적으로 $G_\text{pose} = I + J_F^\top W J_F$가 non-positive-definite (PD)가 되어 학습/샘플링 붕괴.
- Position-only setting ($W = I$)에서는 발생하지 않음.

### 1.2 임시 우회책의 한계

현재 적용된 우회책: `forward_langevin_drift=False` (default OFF, pure Brownian).

- **장점**: 수치적으로 안정.
- **단점**: 큰 $r$에서 training-eval mismatch. Forward SDE가 mathematically clean하지 않음 (Varadhan-regime 근사).
- **본질**: 문제를 해결한 것이 아니라 회피한 것. Reviewer가 forward process의 mathematical correctness를 물을 때 약점이 됨.

### 1.3 진짜 원인 — 두 layer로 분리

문제는 두 개의 **독립적인** 문제가 합쳐진 것.

#### Layer A: Stationary distribution이 unbounded (Measure-theoretic 문제)

$\mathcal{M}_\phi$가 non-compact이고 boundary가 없으면, Brownian motion + Langevin drift의 stationary measure는

$$\pi(q) \propto \exp(-V(q)) \sqrt{\det G(q)}$$

이 well-defined하려면 potential $V(q)$가 $\|q\|\to\infty$에서 충분히 빨리 자라야 함. 현재 framework는 $V \equiv 0$ (pure Brownian on manifold)이므로 **stationary distribution이 존재하지 않음**. Forward SDE가 무한 시간으로 가면 sample이 무한히 퍼짐.

이는 어떤 $W$를 써도 발생하는 **수학적 구조 문제**.

#### Layer B: Conditioning 폭발 (Numerical 문제)

- $W_p = \sigma_p^{-2} = 10^4$ (mm scale) → $\text{cond}(G_\text{pose}) \sim 10^5$.
- High-condition matrix 연산에서 null-space 방향 (kinematic redundancy)의 drift 항 $\frac{1}{2}\nabla\log\det G$가 numerical drift 누적.
- $q$가 joint range 밖으로 빠져나감.

이는 $W_p$ 크기 때문에 발생하는 **수치 정확도 문제**.

#### 두 layer의 독립성

- Layer A는 measure-theoretic, $W$ 크기와 무관.
- Layer B는 numerical, $W$ 크기에 비례.
- "drift OFF"는 Layer B를 회피하지만 Layer A를 풀지 않음.

### 1.4 Torus 구조 변경의 비현실성

수학적으로 완벽한 구조 (sinusoidal activation + torus 구조 제한)는:
- 주기성 처리, periodic boundary condition 모델링이 복잡.
- 기존 framework의 graph manifold 정의, induced metric, score lift 모두 재작성 필요.
- 시간 제약 외에도 **수학적 모델링 자체의 부담**이 큼.

→ **현재 수학적 구조를 유지하면서 numerical stability를 확보하는 방향**이 합리적.

---

## 2. 해결 방안

세 가지 fix를 우선순위 순으로 적용. **min-effort min-novelty → max-rigor** 순서.

### 2.1 [Fix 1] Tolerance relaxation ($\sigma_p$ 완화)

**목적**: $W_p$의 scale 자체를 줄여서 Layer B의 root cause를 제거.

> **Naming clarification**: 본 fix는 unit conversion이 아님. 기존 코드는 이미 meter unit을 사용 중이며, $\sigma_p = 0.01$ (= 1cm tolerance)이 강한 가중치 $W_p = \sigma_p^{-2} = 10^4$를 만든 것. 따라서 fix는 $\sigma_p$를 (예: 1cm $\to$ 5cm로) 완화하는 것이며, 수학적 구조 변경은 없음.

#### 2.1.1 분석

- 현재 $W_p = \sigma_p^{-2} = 10^4$ ($\sigma_p = 0.01$ m, 즉 1cm tolerance).
- Conditioning $\text{cond}(G_\text{pose}) = \text{cond}(I + J_F^\top W J_F)$에서 $W_p$가 dominant.
- $\sigma_p$를 0.01 $\to$ 0.05 (5cm tolerance)로 완화하면 $W_p$: $10^4 \to 400$, $\text{cond}(G) \sim 10^5 \to 10^3$.
- Task-level 영향: succ@5cm metric은 5cm tolerance와 정합적이므로 large degradation 없을 가능성 높음.

#### 2.1.2 구현

- Config의 `sigma_p` (또는 동등한 hyperparameter) 값만 변경.
- 학습/평가 코드 변경 없음.

#### 2.1.3 비용 / 위험

- **비용**: 5분 (config change).
- **위험**: $\sigma_p$가 너무 크면 task accuracy 손상 (1cm-precision task에는 부적합).
- **기대 효과**: 단독으로 cond이 $10^3$ 이하로 떨어지면 다른 fix 불필요할 수 있음. 가능성 30%.

### 2.2 [Fix 2] Tikhonov regularization on metric (수치 안정화)

**목적**: $G_\text{pose}$의 conditioning을 boundary로 강제하여 Layer B의 numerical drift를 차단.

> **Naming clarification**: 본 fix는 새로운 기법이 아닌 **기존 ad-hoc jitter ($\lambda = 10^{-4}$ I)를 adaptive Tikhonov로 격상**하는 것. 작은 $\lambda$ 한정에서 metric 변화는 무시 가능 ($\|G^\text{reg} - G\|_F / \|G\|_F < 1\%$).

#### 2.2.1 정의

$$G_\text{pose}^\text{reg}(q) = G_\text{pose}(q) + \lambda(q) I, \quad \lambda(q) = c \cdot \frac{\text{tr}(G_\text{pose}(q))}{n_q}, \quad c \approx 10^{-2}$$

기존 fixed jitter $10^{-4}$는 $G$의 trace에 무관하게 일정 → $G$의 magnitude가 클 때 ($W_p$가 클 때) 불충분. Adaptive form은 $G$ scale에 비례하여 항상 적절한 conditioning 보장.

#### 2.2.2 분석

- $\lambda$가 충분히 작으면 manifold geometry는 거의 보존 ($\|G^\text{reg} - G\|_F / \|G\|_F < 1\%$).
- Cond이 effectively bounded: $\text{cond}(G^\text{reg}) \leq \sigma_{\max}/\lambda$.
- $G^{-1}$ 계산이 항상 well-conditioned.

#### 2.2.3 대안

- **Spectral clipping**: $G$의 eigenvalue를 $[\sigma_{\min}, \sigma_{\max}]$로 clip. 더 aggressive지만 안정적. 구현은 SVD 1회 필요.
- 본 framework에서는 Tikhonov가 더 단순하고 미분 가능성도 자연스러움.

#### 2.2.4 구현

- `G_pose = I + JF.T @ W @ JF + lam * I` 한 줄 추가.
- $\lambda$ sweep: $\lambda \in [10^{-4}, 10^{-1}]$, V2 metric (pos_err, succ@5cm)에 영향 측정.
- Sanity 6 invariants가 깨지지 않는 $\lambda$ 범위 결정.

#### 2.2.5 비용 / 위험

- **비용**: 반나절 (sweep 포함).
- **위험**: $\lambda$가 너무 크면 score field가 blurred되어 manifold adherence 손상. Sweep으로 통제.
- **기대 효과**: Fix 1 + Fix 2 결합으로 문제 해결될 가능성 70%.

### 2.3 [Fix 3] Soft confining potential with anchor metric (수학적 정합성 확보)

**목적**: Layer A의 root cause를 mathematical하게 해결. Forward SDE의 stationary distribution이 well-defined하고, **sampling 초기화와 정합**되도록 보장.

#### 2.3.1 Sampling 초기화와의 정합성 요구

본 framework의 reverse sampling 초기화:
$$q_K \sim \mathcal{N}(\mu_q, \gamma^2 G^{-1}(\mu_q))$$

여기서 $\mu_q$는 trajectory의 start config (또는 prior mean), $G^{-1}(\mu_q)$는 anchor point에서의 inverse metric.

**Naive box-confining만 사용하면 mismatch 발생**: $U_\text{box}(q) = \kappa \cdot \text{ReLU-quad}$ 단독으로는 stationary $\pi(q) \propto e^{-U_\text{box}(q)} \sqrt{\det G(q)}$가 joint range 안에서 거의 uniform × $\sqrt{\det G}$ 형태 → sampling 초기화 (Gaussian centered at $\mu_q$)와 distribution mismatch → score model이 학습한 marginal $p_K$와 reverse 시작점 불일치 → 누적 오차.

#### 2.3.2 Anchor metric을 이용한 정합 (Option B)

Sampling 초기화를 그대로 유지하면서 stationary를 정합시키기 위해, **anchor point $\mu_q$에서 evaluated된 fixed metric** $\hat G := G(\mu_q)$를 사용:

$$U_\text{total}(q) = \frac{1}{2\gamma^2} (q - \mu_q)^\top \hat G (q - \mu_q) + \kappa \sum_i \left[ \text{ReLU}(q_i - q_{i,\max} + \epsilon)^2 + \text{ReLU}(q_{i,\min} + \epsilon - q_i)^2 \right]$$

핵심 design choice:
- **$\hat G$는 fixed**: $\mu_q$에서 한 번 계산, forward SDE 동안 $q$에 따라 재계산하지 않음.
- **$\frac{1}{2}\log\det G$ 항은 추가하지 않음**: Sampling이 $G^{-1}$ scaling을 유지하므로 cancel하지 않음.

Stationary distribution:
$$\pi(q) \propto \exp\left( -\frac{1}{2\gamma^2}(q - \mu_q)^\top \hat G (q - \mu_q) - U_\text{box}(q) \right) \cdot \sqrt{\det G(q)}$$

$q \approx \mu_q$ 근처 (즉 sampling 초기 영역)에서 $G(q) \approx \hat G$ → $\sqrt{\det G(q)} \approx \sqrt{\det \hat G}$ (constant) → **stationary가 $\mathcal{N}(\mu_q, \gamma^2 \hat G^{-1})$ × box로 effectively reduce**. 정확히 sampling 초기화와 일치. ✓

#### 2.3.3 Forward SDE 형태

Anchor metric을 사용한 confining drift:

$$dq_r = \left[ \frac{1}{2}\nabla \log \det G(q_r) - \frac{1}{2} G^{-1}(q_r) \nabla U_\text{total}(q_r; \mu_q) \right] dr + dB_r^{\mathcal{M}}$$

여기서:
- $\nabla U_\text{total} = \frac{1}{\gamma^2} \hat G (q - \mu_q) + \nabla U_\text{box}$ (둘 다 elementwise/closed-form).
- $G^{-1}(q_r) \nabla U_\text{total}$: 어차피 forward drift에서 계산하던 항 구조, $\hat G$ 추가 cost는 trajectory당 1회.

#### 2.3.4 효과

- Stationary $\pi$가 well-defined: $\mathcal{N}(\mu_q, \gamma^2 \hat G^{-1}) \cap$ box, $\sqrt{\det G}$ smooth correction.
- **Sampling 초기화와 정합**: reverse SDE 시작점이 stationary에서 sample된 것과 일치.
- Forward Langevin drift를 ON으로 켜도 numerical drift 누적 없음 (box potential이 차단).
- 기존 framework의 $G^{-1}$-based identity (sampling, guidance, tangent noise) 유지 — Riemannian-ness가 forward process에서도 보존.

#### 2.3.5 Mathematical context — paper framing

- Riemannian SGM 문헌의 standard practice (De Bortoli et al. 2022 등이 implicit하게 가정).
- **VP-SDE의 manifold analog**: Euclidean VP-SDE의 linear drift $-\beta(r) q$가 Gaussian stationary를 만들 듯, $-\frac{1}{2} G^{-1} \nabla U_\text{total}$이 manifold에서 anchor-Gaussian × box stationary를 만듦.
- $\hat G = G(\mu_q)$는 paper에서 **"anchor metric" 또는 "reference-point metric"**으로 framing. 매 trajectory의 anchor에서 metric 한 번 평가, forward SDE 동안 fixed.
- $q \approx \mu_q$ 근처에서 $G(q) \approx \hat G$이므로 first-order approximation으로 정당화. $\gamma$가 작을수록 정확.

#### 2.3.6 구현

- $\hat G = G(\mu_q)$: trajectory 시작 시 한 번 계산, batch에 cache.
- $\nabla U_\text{box}$: ReLU-quadratic, 자동미분 trivial.
- $\nabla U_\text{total} = \gamma^{-2} \hat G (q - \mu_q) + \nabla U_\text{box}$: 두 항 합.
- Hyperparameter:
  - $\kappa$: confinement 강도. 권장 시작값 $\kappa \in [10^2, 10^4]$.
  - $\epsilon$: range margin. Franka joint range의 $5\%$ ($\epsilon = 0.05 \cdot (q_{\max} - q_{\min})$) 권장.
  - $\gamma$: 기존 sampling 초기화의 $\gamma$ 값을 그대로 사용. 별도 tuning 불필요.

#### 2.3.7 비용 / 위험

- **비용**: 1-2일 (구현 + tuning + sanity check).
- **위험**:
  - $\kappa$가 너무 크면 manifold 안쪽에서도 distortion. $\epsilon$이 너무 작으면 boundary 진입 시 explosion. → $\kappa$, $\epsilon$ sweep 필요.
  - $\hat G$ approximation은 $q$가 $\mu_q$에서 멀어질수록 부정확. $\gamma$가 충분히 작으면 무시 가능 (paper에서 명시).
- **기대 효과**: Fix 1 + Fix 2 + Fix 3 결합으로 문제 해결 가능성 95%+, **수학적 정합성도 강화** (sampling-stationary alignment 명시적).

---

## 3. 권장 실행 순서

```
[Step 1] σ_p 완화 (5분)
   ↓
   sigma_p: 0.01 → 0.05. G conditioning 측정.
   cond < 10^3이면 이것만으로 해결될 수 있음.
   Sanity 6 + V2 metric 재측정.
   ↓
[Step 2] (해결 안 되면) Adaptive Tikhonov regularization (반나절)
   ↓
   기존 jitter 1e-4 → adaptive λ(q) = c·tr(G)/n_q.
   c sweep (1e-4 ~ 1e-1), 성능 영향 측정.
   Drift ON 상태에서 numerical drift 누적 여부 확인.
   ↓
[Step 3] (그래도 누적되면) Anchor-metric soft confining potential (1-2일)
   ↓
   Ĝ = G(μ_q), U_total = (1/2γ²)(q-μ_q)^T Ĝ (q-μ_q) + κ·U_box.
   κ, ε sweep. Stationary가 sampling 초기화와 정합됨을 verify.
   ↓
[Final] Sanity 6 invariants 재검증
   ↓
   Manifold adherence가 깨지지 않았는지 확인.
   V2 metric (pos_err, succ@5cm, EE excess)이 회복되었는지 확인.
```

---

## 4. Paper writing 측면의 framing

세 fix를 limitation이 아닌 **principled design choice**로 framing 가능.

### 4.1 권장 framing

> Since the self-model manifold $\mathcal{M}_\phi$ is non-compact and embedded in $\mathbb{R}^{n_q + 6}$, naive Riemannian Brownian motion does not admit a stationary distribution. We introduce three principled modifications:
> (i) **adaptive Tikhonov regularization** $G^\text{reg} = G + \lambda(q) I$ on the induced metric to bound the condition number;
> (ii) **anchor-metric soft confining potential**
> $$U_\text{total}(q; \mu_q) = \tfrac{1}{2\gamma^2} (q - \mu_q)^\top \hat G (q - \mu_q) + U_\text{box}(q), \quad \hat G = G(\mu_q),$$
> ensuring the forward SDE admits a well-defined invariant measure;
> (iii) **sampling-stationary alignment**: by using the fixed anchor metric $\hat G$, the resulting stationary distribution near $\mu_q$ matches the sampling initialization $\mathcal{N}(\mu_q, \gamma^2 \hat G^{-1}) \cap \text{box}$, eliminating distribution mismatch at $r=K$.
>
> This is the manifold analog of the linear drift in VP-SDE on $\mathbb{R}^d$, with the anchor-metric structure providing first-order alignment with the reverse sampling initialization while preserving the Riemannian identity ($G^{-1}$-based guidance, tangent noise) of our framework.

### 4.2 효과

- "Drift OFF"는 approximation framing이 되어 reviewer 약점으로 지적되기 쉬움.
- Anchor-metric confinement는 **principled extension**: VP-SDE on manifold의 자연스러운 정의 + sampling-stationary alignment를 명시.
- $\hat G = G(\mu_q)$ approximation은 reviewer가 물을 때 명확한 답변 가능 (first-order, $\gamma$ 작을수록 정확).
- 수학적 정합성이 오히려 강화됨 (forward process의 Riemannian-ness 보존).

---

## 5. 검증 항목

각 fix 적용 후 확인할 metric:

| 항목 | Pass 기준 |
|---|---|
| Sanity 1-6 | machine precision (기존 기준) |
| $\text{cond}(G_\text{pose}^\text{reg})$ | $< 10^3$ |
| Forward SDE drift 누적 ($\|q_r - \mu_q\|$ vs $r$) | bounded, joint range 내 |
| Stationary $\pi$의 empirical 분포 vs $\mathcal{N}(\mu_q, \gamma^2 \hat G^{-1})$ | $W_1$ distance가 $\gamma$ scale 이내 |
| V2 main metric (pos_err) | $\leq 25$ mm (기존 21 mm 대비 약간의 degradation 허용) |
| V2 succ@5cm | $\geq 95\%$ |
| Manifold adherence ($\max\|g_\phi\|$) | $0$ by construction |
| EE excess | $\leq 5.0$ (기존 3.42 대비 약간의 degradation 허용) |

특히 **stationary-sampling alignment**가 Option B의 핵심 design이므로, forward SDE를 충분히 긴 시간 ($r = K$) run한 후 $q_r$의 empirical distribution이 sampling 초기화 분포와 정합되는지 직접 확인 필요.

---

## 6. 요약

- **Layer A (수학적)**: Anchor-metric soft confining potential로 stationary distribution 확보 + sampling 초기화와 정합.
- **Layer B (수치적)**: $\sigma_p$ 완화 + adaptive Tikhonov regularization으로 conditioning 제어.
- **수학적 구조 변경 없이** (torus, sinusoidal 등) 기존 framework의 graph manifold + RSGM 구조 그대로 유지.
- **Option B의 핵심**: 기존 ad-hoc components ($\sigma_p$, jitter)를 단순 격상하면서, 새 구조 (anchor metric)는 **sampling 초기화와의 alignment**라는 명확한 motivation으로 도입. Reviewer-defendable.
- 세 fix의 **누적 적용**으로 forward Langevin drift를 ON으로 사용 가능, paper에서 principled design으로 framing 가능. 기존 V2 결과 (pos_err 21mm 등)와의 connectivity도 유지 (sampling 초기화 변경 없음).