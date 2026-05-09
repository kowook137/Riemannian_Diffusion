# SMCDP Pose-Extended: 방법 A 적용을 위한 수정사항 정리

**목적**: Forward Langevin drift 제거를 통한 forward-target consistency 회복. Manifold adherence와 trajectory quality는 retraction과 multi-component guidance로 보장.

**원칙**: Core geometry (manifold, metric, retraction, lift, guidance) 유지. Forward SDE의 drift 항만 제거. Sampling 초기화를 그에 정합하게 변경.

**작성일**: 2026-05-09

---

## 1. Forward SDE 수정

### 1.1 [제거] Anchor-metric Langevin drift

**v2에 있던 것 (제거)**:

$$U_\text{total}(q; \mu_q, z_e) = \frac{1}{2\gamma^2}(q-\mu_q)^\top \widehat{G}\,(q-\mu_q) + U_\text{box}(q)$$

$$b^q_\text{v2}(q) = -\frac{1}{2}G^{-1}(q,z_e)\,\nabla_q U_\text{total}(q;\mu_q,z_e)$$

**제거되는 구성요소**:
- Anchor $\mu_q$ (per-batch 또는 per-trajectory 어느 정의이든)
- Anchor metric $\widehat{G} = G(\mu_q, z_e)$
- Anchor-Gaussian quadratic term $\frac{1}{2\gamma^2}(q-\mu_q)^\top \widehat{G}(q-\mu_q)$
- Box potential $U_\text{box}(q)$
- 모든 confining drift 계산 (`_anchor_drift_potential_grad` helper 포함)

### 1.2 [유지] Pure Brownian forward on $\mathcal{M}_\phi$

**최종 forward SDE (drift 없음)**:

$$\boxed{ dX_r = dB^\mathcal{M}_r, \qquad X_r \in \mathcal{M}_\phi(z_e) }$$

Chart-coordinate Itô SDE:

$$dq^i_r = \frac{1}{2\sqrt{\det G}}\,\partial_j\!\left(\sqrt{\det G}\, G^{ij}\right) dr + \sigma^i_a(q_r)\,dW^a_r, \qquad \sigma\sigma^\top = G^{-1}$$

이 drift correction은 $G$가 slowly varying할 때 무시 가능. Retraction-based GRW에서는 step-wise tangent Gaussian이 이를 implicit하게 근사.

### 1.3 [유지] Adaptive Tikhonov regularization

수치 안정성을 위해 metric regularization은 유지:

$$G^\text{reg}(q,z_e) = G(q,z_e) + \lambda(q,z_e)\,I, \qquad \lambda(q,z_e) = c_\lambda \cdot \frac{\operatorname{tr} G(q,z_e)}{n_q}$$

$c_\lambda = 10^{-2}$, $\sigma_p = 0.05$ default.

이는 forward SDE 형태와 무관한 numerical conditioning이므로 유지. Drift OFF에서도 $G$의 conditioning은 학습/샘플링 안정성에 영향.

### 1.4 [유지] Trajectory product manifold

$$d\tau_r = dB^{\mathcal{M}^{H+1}}_r, \qquad dX_{h,r} = dB^\mathcal{M}_{h,r}$$

각 timestep이 독립적인 Brownian on $\mathcal{M}_\phi$.

---

## 2. Forward simulation 수정

### 2.1 [수정] Forward step (단일 timestep)

**v2에 있던 것 (수정)**:

```
δq = Δr · b^q(q; μ_q, z_e)  +  √Δr · ξ,    ξ ~ N(0, G^{-1}(q, z_e))
```

**방법 A**:

$$\boxed{ \delta q_k = \sqrt{\Delta r}\,\xi_k, \qquad \xi_k \sim \mathcal{N}(0,\, G^{-1}(q_k, z_e)) }$$

Drift 항 완전 제거. Step은 pure tangent Gaussian noise.

### 2.2 [유지] Retraction

$$q_{k+1} = q_k + \delta q_k, \qquad x_{k+1} = H_\phi^\text{pose}(q_{k+1}, z_e) = (q_{k+1},\, T_\phi(q_{k+1}, z_e))$$

Manifold adherence는 retraction으로 보장 (drift와 무관):

$$g_\phi(x_{k+1}, z_e) = T_\phi(q_{k+1},z_e) - T_\phi(q_{k+1},z_e) = 0 \quad \text{by construction}$$

---

## 3. DSM target 수정

### 3.1 [유지] Varadhan-based local Brownian target

Forward가 pure Brownian이므로 Varadhan asymptotic이 정확히 valid:

$$\nabla_\mathcal{M} \log p_{r|0}(x_r | x_0) \approx \frac{\operatorname{Exp}_{x_r}^{-1}(x_0)}{r}$$

### 3.2 [유지] Pose-extended chart-coordinate target

**Ambient displacement (pose-extended)**:

$$\delta x^\text{amb}_{r0} = \begin{pmatrix} q_0 - q_r \\ \operatorname{Log}_{SE(3)}(T_\phi(q_r,z_e)^{-1}\,T_\phi(q_0,z_e)) \end{pmatrix} \in \mathbb{R}^{n_q + 6}$$

**Chart-coordinate target** (pseudoinverse via $J_H^\top W J_H = G_\text{pose}$):

$$\boxed{ a^*_\text{pose}(q_r, x_0; z_e) = G^{-1}_\text{pose}(q_r, z_e)\,J_H^{\text{pose}\top}(q_r,z_e)\, W\, \delta x^\text{amb}_{r0} }$$

전개:

$$a^*_\text{pose} = G^{-1}_\text{pose}\left[(q_0 - q_r) + W_p \cdot J_F^\top(q_r,z_e)\,(F_\phi(q_0,z_e) - F_\phi(q_r,z_e)) + \mathcal{J}_\text{rot}^\top \cdot W_R \cdot \operatorname{Log}_{SO(3)}(\cdots)\right]$$

여기서 $\mathcal{J}_\text{rot}$은 rotation Jacobian, 정확한 form은 $SE(3)$ Log convention에 의존.

### 3.3 [유지] Chart-G weighted DSM loss

$$\boxed{ \mathcal{L}^\text{chart-}G_\text{traj} = \mathbb{E}_{r, \tau_0, \tau_r}\left[ w(r) \sum_{h=0}^{H} (s_{\theta, h}^q - a^*_{\text{pose}, h})^\top G_\text{pose}(q_{h,r}, z_e)\,(s_{\theta, h}^q - a^*_{\text{pose}, h})\right] }$$

이는 ambient loss와 정확히 equivalent (norm equivalence $\|J_H u\|^2_{W} = u^\top G u$).

### 3.4 [재검토 필요] Loss weighting $w(r)$

v2에서 $w(r) = \sigma^2(r)$이 $\sigma^2$ weighting → 큰 $r$ 강조했음. Drift OFF에서는 Varadhan이 모든 $r$에서 (relatively) 정합적이므로 weighting 부담이 줄어듦. 그러나 std_trick과의 상호작용은 잔존.

**권고**: $w(r)$ 옵션을 ablation으로 비교 (sigma², constant, $1/\tau_\text{brown}$ inverse). Drift OFF baseline (R1)에서 어느 것이 안정적인지 측정.

---

## 4. Sampling 초기화 수정

### 4.1 [수정] Reference distribution

**v2에 있던 것 (수정)**:

$$p_\text{ref}^\text{v2}(\tau | \mu_q, z_e) \propto \prod_{h=0}^H \mathcal{N}(q_h; \mu_q, \gamma^2 \widehat{G}^{-1}) \cdot \mathbb{1}[q_h \in \text{box}]$$

**방법 A** (pure Brownian의 finite-time noising에 대응):

$$\boxed{ p_\text{ref}(\tau | z_e) \propto \prod_{h=0}^{H} \mathcal{N}(q_h;\, q_h^\text{init},\, \sigma_K^2\, G^{-1}(q_h^\text{init}, z_e)) }$$

여기서:
- $q_h^\text{init}$: per-trajectory 초기화 reference (예: demo의 평균 trajectory, 또는 $T_\text{start}$로부터 유도된 IK seed).
- $\sigma_K^2$: forward SDE의 $r=K$ 시점 noise variance scale (schedule에서 결정).

### 4.2 [구현 detail] Initialization seed 선택

세 가지 선택지:

1. **$q_h^\text{init} = q_0^\text{ref}(c)$**: condition $c = (T_\text{start}, T_\text{target})$에서 IK로 seed 생성, 모든 $h$에 같은 값.
2. **$q_h^\text{init} = q_0^{(i)}$ (per-trajectory)**: demo의 $q_0$ 사용 (training 시), test 시 IK seed.
3. **$q_h^\text{init} = $ demo 평균 또는 fixed**: 단순하지만 score net이 학습한 marginal과 mismatch 가능.

**권고**: (1) IK-based seed가 가장 condition-aware하고 score net의 $T_\text{start}$ conditioning과 정합.

### 4.3 [유지] Manifold lift

$$x_{h, K} = H_\phi^\text{pose}(q_{h, K}, z_e) = (q_{h,K},\, T_\phi(q_{h,K}, z_e))$$

Manifold 위 sample로 변환.

---

## 5. Score net architecture (변경 없음)

### 5.1 [유지] Input signature

$$\text{input}_h = [q_h,\, r,\, h/H,\, z_e,\, T_\text{start},\, T_\text{target}]$$

**$\mu_q$ 입력 추가 불필요**: drift가 없으므로 $\mu_q$가 forward dynamics에 영향을 주지 않음. Score net이 학습할 marginal $p_r(q_r | T_\text{start}, T_\text{target}, z_e)$이 $\mu_q$와 무관하게 well-defined.

→ **Q3에서 제기된 conditioning mismatch 문제가 자동 해소**.

### 5.2 [유지] Tangent lift

$$s_\theta^\text{amb}(r, x, c, z_e) = J_H^\text{pose}(q, z_e)\, s_\theta^q(r, q, c, z_e) = \begin{pmatrix} s_\theta^q \\ \mathcal{J}_\text{pose}\, s_\theta^q \end{pmatrix}$$

Tangent verification:

$$J_g^\text{pose}\, s_\theta^\text{amb} = J_g^\text{pose}\, J_H^\text{pose}\, s_\theta^q = 0 \quad \text{by construction}$$

### 5.3 [유지] std trick (필요시)

$s_\theta^q(r, q) = \text{net}(r, q) / \sigma(r)$로 학습 안정화. v2의 conditioning issue는 잔존하지만 drift OFF에서 $a^*$ scale이 더 안정적이므로 영향 감소.

---

## 6. Reverse sampling 수정

### 6.1 [수정] Reverse step (drift 항 제거)

**v2에 있던 것 (수정)**:

$$\mu_{h,k}^q = -b^q(q_{h,k}; \mu_q, z_e) + s_{\theta,h}^q + m_h\, G^{-1}\nabla_{q_h} R_\text{total}$$

**방법 A** (forward drift = 0이므로 reverse drift도 score만):

$$\boxed{ \mu_{h,k}^q = s_{\theta,h}^q(r_k, \tau_k, c, z_e) + m_h \cdot G^{-1}_\text{pose}(q_{h,k}, z_e)\, \nabla_{q_h} R_\text{total}(\tau_k, c) }$$

전체 step:

$$\delta q_{h,k} = \Delta r \cdot \mu_{h,k}^q + \sqrt{\Delta r}\, \xi_{h,k}^q, \qquad \xi_{h,k}^q \sim \mathcal{N}(0,\, G^{-1}_\text{pose}(q_{h,k}, z_e))$$

$$q_{h,k+1} = q_{h,k} + \delta q_{h,k}, \qquad x_{h,k+1} = H_\phi^\text{pose}(q_{h,k+1}, z_e)$$

### 6.2 [유지] Multi-component guidance

$$R_\text{total}(\tau, c) = \alpha_s R_\text{start} + \alpha_g R_\text{goal} + \alpha_v R_\text{vel} + \alpha_a R_\text{acc}$$

각 항:

$$R_\text{start}(\tau, c) = -\|F_\phi(q_0, z_e) - p_\text{start}\|^2$$

$$R_\text{goal}(\tau, c) = -\|F_\phi(q_H, z_e) - p_\text{target}\|^2$$

$$R_\text{vel}(\tau) = -\sum_{h=0}^{H-1} \|q_{h+1} - q_h\|^2$$

$$R_\text{acc}(\tau) = -\sum_{h=1}^{H-1} \|q_{h+1} - 2q_h + q_{h-1}\|^2$$

Pose-extended에서 $R_\text{start}, R_\text{goal}$을 pose distance로 확장 가능:

$$R_\text{goal}^\text{pose}(\tau, c) = -\|p\text{-part of }T_\phi(q_H) - p_\text{target}\|^2 - \alpha_R \|\operatorname{Log}_{SO(3)}(R_\text{target}^\top R(q_H))\|^2$$

### 6.3 [유지] Mask schedule

$$m_h \in \{0, 1\}, \quad m_h^\text{last\_half} = \mathbb{1}[h \geq H/2]$$

Empirically robust, 변경 없음.

### 6.4 [유지] Tangent verification

$$J_g(x_h, z_e) \cdot J_H(q_h, z_e)\,\mu_{h,k}^q = 0 \quad \text{for all components}$$

Score, guidance 모두 chart vector이므로 lift 후 tangent 안에 머무름.

---

## 7. Sanity tests 재정의

### 7.1 [유지] S1-S8 기존 테스트

기존 manifold adherence, metric PD, tangent lift consistency 등은 그대로.

### 7.2 [제거] Anchor-metric 관련 테스트

- Anchor $\widehat{G}$ caching test
- Forward + Reverse drift consistency test
- Box potential gradient test
- $\kappa$, $\epsilon$ hyperparameter sweep test

이들은 drift 제거로 무의미.

### 7.3 [추가] Drift OFF specific test

**Test S9 (신규)**: Forward marginal이 Brownian heat kernel 형태인지 검증.
- $q_0 \to q_r$ via forward SDE, $r \to 0$에서 $\|q_r - q_0\|^2 / r$이 $\operatorname{tr}(G^{-1}(q_0))$에 수렴하는지 측정.

**Test S10 (신규)**: Score target $a^*$의 $r$별 norm 분포.
- 모든 $r$에서 $\|a^*\|$가 발산하지 않는지, $\sigma(r)$로 정규화 후 $O(1)$인지 확인.

**Test S11 (신규)**: Reverse SDE가 drift OFF setup에서 manifold adherence 유지하는지.
- 100 step reverse sampling 후 $\max_h \|g_\phi(x_h, z_e)\| < 10^{-3}$.

---

## 8. Pipeline summary (수정 후)

### 8.1 Framework pipeline (방법 A)

$$\text{Stage 1.} \quad F_\phi = FK_\text{analytic} + \Delta_\phi,\, T_\phi = (F_\phi, R_\phi),\, \min_\phi \mathcal{L}_\text{self}^\text{pose}$$

$$\text{Stage 2.} \quad \mathcal{M}_\phi^\text{pose}(z_e) = \{(q, T) : T = T_\phi(q, z_e)\}$$

$$G_\text{pose} = J_H^{\text{pose}\top} W J_H^\text{pose} = I + W_p J_F^\top J_F + W_R \mathcal{J}_\text{rot}^\top \mathcal{J}_\text{rot}$$

$$G^\text{reg}_\text{pose} = G_\text{pose} + \lambda(q, z_e)\, I$$

$$\text{Stage 3.} \quad \tau_i = (H_\phi^\text{pose}(q_h^{(i)}, z_e))_{h=0}^H \in (\mathcal{M}_\phi^\text{pose})^{H+1}$$

$$\text{Stage 4.} \quad d\tau_r = dB^{\mathcal{M}^{H+1}}_r \quad \text{(pure Brownian, no drift)}$$

$$\text{Stage 5.} \quad \min_\theta \mathcal{L}^\text{chart-}G_\text{traj} \text{ with Varadhan-based } a^*_\text{pose}$$

$$\text{Stage 6.} \quad \tau_K \sim p_\text{ref}(\cdot | z_e, q^\text{init}) \to \tau_0 \quad \text{via retraction-based GRW + multi-component guidance}$$

### 8.2 Application-specific layer

$$c = (T_\text{start}, T_\text{target}, z_e), \quad \text{channel-concat to score net}$$

$$s_\text{guided}^q = s_\theta^q + G^{-1}_\text{traj}\, \nabla_{q_{0:H}} R_\text{total}^\text{pose}(\tau, c)$$

$$R_\text{total}^\text{pose} = \alpha_s R_\text{start}^\text{pose} + \alpha_g R_\text{goal}^\text{pose} + \alpha_v R_\text{vel} + \alpha_a R_\text{acc}$$

---

## 9. 코드 변경 사항 (구현 layer)

### 9.1 [제거] Forward drift code paths

- `_anchor_drift_potential_grad` helper
- `forward_langevin_drift` flag (default OFF에서 hard-OFF로)
- Anchor metric caching (`_compute_anchor_G`, $\widehat{G}$ batch tensor)
- Box potential evaluation
- CLI flags: `--confining-kappa`, `--confining-epsilon-frac`

### 9.2 [유지] Numerical conditioning

- Adaptive Tikhonov: `lam = c_lam * tr(G) / n_q`, `G_reg = G + lam * I`
- CLI flag: `--tikhonov-frac` (기본값 1e-2)
- $\sigma_p = 0.05$ default

### 9.3 [수정] Forward step 함수

`forward_step(q, r, z_e)`:

```
G = compute_G_pose(q, z_e)
G_reg = G + lam(G) * I
xi = sample N(0, G_reg^-1)
q_new = q + sqrt(Δr) * xi
return q_new
```

Drift 계산 완전 제거.

### 9.4 [수정] Sampling 초기화

`sample_initial_q(c, z_e)`:

1. IK seed: $q^\text{init} = \operatorname{IK}(T_\text{start}, z_e)$
2. Per-h sample: $q_h^{(K)} \sim \mathcal{N}(q^\text{init}, \sigma_K^2\, G^{-1}(q^\text{init}, z_e))$
3. Lift: $x_h^{(K)} = H_\phi^\text{pose}(q_h^{(K)}, z_e)$

### 9.5 [수정] Reverse step 함수

`reverse_step(q, r, c, z_e)`:

```
s_q = score_net(q, r, c, z_e) / sigma(r)        # std trick
guidance = G_reg^-1 @ ∇R_total                   # multi-component
mu_q = s_q + mask_h * guidance                   # NO drift term
xi = sample N(0, G_reg^-1)
q_new = q + Δr * mu_q + sqrt(Δr) * xi
return q_new
```

### 9.6 Backward compatibility

V2 코드의 모든 anchor/box/drift code path가 dead code가 되지만, **순수 numerical (Tikhonov, $\sigma_p$)와 manifold ($G$, $J_H$, retraction) code는 그대로 작동**. Position-only V2 결과는 변화 없이 재현 가능.

---

## 10. 검증 항목 (수정 후)

| 항목 | Pass 기준 | 측정 방법 |
|---|---|---|
| Manifold adherence | $\max_h \|g_\phi\| = 0$ by construction | 모든 sample에서 측정 |
| $\operatorname{cond}(G_\text{pose}^\text{reg})$ | $< 10^3$ | Per-batch 측정 |
| Forward marginal Gaussianity (small $r$) | $\|q_r - q_0\|^2 / r \to \operatorname{tr}(G^{-1})$ | S9 test |
| Target $\|a^*\|$ scale | 모든 $r$에서 bounded, $\sigma(r)$로 정규화 시 $O(1)$ | S10 test |
| Stage-2 loss | $\to O(10^{-1})$ 수준 수렴 | Training curve |
| pos_err (pose endpoint) | $\leq 25$ mm | Eval 측정 |
| succ@5cm | $\geq 95\%$ | Eval 측정 |
| EE excess | $\leq 5.0$ | Eval 측정 |
| Joint limit violation | $\leq 5\%$ per-trajectory | Eval 측정, demo distribution이 limit 안에 있다면 자동 만족 예상 |
| Mode capture (bimodal task) | $\|\operatorname{frac}_A - 0.5\| < 0.1$ | Eval 측정, $\mu_q$ mismatch 해소로 개선 예상 |

---

## 11. Paper writing 측면

### 11.1 Method section 핵심 문장

> "The forward process on $\mathcal{M}_\phi^\text{pose}(z_e)$ is pure Riemannian Brownian motion, simulated via retraction-based geodesic random walk with step-wise tangent Gaussian $\xi \sim \mathcal{N}(0, G_\text{pose}^{-1})$. The DSM target follows the Varadhan local Brownian asymptotic $\operatorname{Exp}_{x_r}^{-1}(x_0)/r$, written in chart coordinates as $a^*_\text{pose} = G_\text{pose}^{-1} J_H^{\text{pose}\top} W \delta x^\text{amb}_{r0}$. Trajectory quality is enforced at sampling time through product-manifold Riemannian guidance with start, goal, velocity, and acceleration rewards."

### 11.2 Limitation section

> "Since the joint chart is non-compact, the Brownian forward process does not admit a stationary probability measure. We use finite-time Brownian noising, which suffices for score matching on a finite horizon $r \in [0, K]$. Drift-consistent transition score targets that admit a true stationary reference (e.g., OU-type processes with closed-form transition kernels on Riemannian manifolds) are left for future work."

### 11.3 Contribution claim

> "The core contribution is the $z_e$-conditioned self-model manifold $\mathcal{M}_\phi(z_e)$, the induced task-aware metric $G$, the tangent-lifted score $J_H s^q$, and product-manifold Riemannian guidance, none of which depend on the choice of stationary forward reference."

---

## 12. 한 줄 요약

> **Forward SDE에서 drift 항을 완전 제거하고 pure Brownian으로 회귀, DSM target은 그대로 Varadhan-based, sampling 초기화는 condition-aware IK seed Gaussian으로 변경, manifold/metric/retraction/lift/guidance core는 그대로 유지.**

이게 v2의 mathematical inconsistency를 수식 layer에서 해소하면서 framework의 모든 contribution을 보존하는 가장 minimal하고 정합적인 변경이다.

---

## 부록 A: v2 → 방법 A 변경 사항 일람

| 항목 | v2 | 방법 A |
|---|---|---|
| Forward drift | $-\frac{1}{2}G^{-1}\nabla U_\text{total}$ | $0$ |
| Anchor $\mu_q$ | per-batch 또는 fixed | 없음 |
| Anchor metric $\widehat{G}$ | $G(\mu_q, z_e)$ fixed | 없음 |
| Box potential $U_\text{box}$ | ReLU-quadratic, $\kappa$ scaled | 없음 |
| Forward SDE | Anchor-Langevin + box | Pure Brownian |
| DSM target | Varadhan (mismatched) | Varadhan (consistent) |
| Stationary distribution | $\mathcal{N}(\mu_q, \gamma^2\widehat{G}^{-1}) \cap \text{box}$ | 없음 (finite-time noising) |
| Sampling 초기화 | Anchor-Gaussian × box | IK-seed Gaussian |
| Score net input | $[q, r, h/H, z_e, T_\text{start}, T_\text{target}]$ | 동일 |
| Reverse drift | $-b^q + s_\theta^q + \text{guidance}$ | $s_\theta^q + \text{guidance}$ |
| Tikhonov regularization | 유지 | 유지 |
| $\sigma_p$ default | 0.05 | 0.05 |
| Multi-component guidance | 유지 | 유지 |
| Manifold/metric/retraction/lift | 유지 | 유지 |

---

## 부록 B: 제거되는 코드 구성요소 체크리스트

**Forward drift 관련**:
- [ ] `_anchor_drift_potential_grad` helper 함수
- [ ] `_compute_anchor_G` 함수 (anchor metric 계산)
- [ ] $\widehat{G}$ batch tensor caching
- [ ] Anchor-Gaussian quadratic gradient 계산
- [ ] Box potential gradient 계산 (`U_box` related)
- [ ] `forward_langevin_drift` flag 처리 분기
- [ ] Reverse step의 `-b^q` 항 추가 부분

**CLI / config**:
- [ ] `--confining-kappa` flag
- [ ] `--confining-epsilon-frac` flag
- [ ] `--gamma` parameter (anchor Gaussian scale)
- [ ] `forward_langevin_drift` config option

**Sanity tests**:
- [ ] Anchor $\widehat{G}$ caching test
- [ ] Forward + Reverse drift consistency test (`_anchor_drift_potential_grad` 일관성)
- [ ] Box potential gradient test (경계에서 $2\kappa\epsilon$ 검증)
- [ ] $\kappa$, $\epsilon$ hyperparameter sweep 관련

**유지되는 항목**:
- [x] Adaptive Tikhonov regularization (`lam = c_lam * tr(G) / n_q`)
- [x] $G$, $J_H$, $J_F$ 계산
- [x] Retraction $H_\phi^\text{pose}$
- [x] Tangent lift, score net architecture
- [x] Multi-component reward $R_\text{total}$
- [x] Mask schedule
- [x] $\sigma_p = 0.05$ default

---

## 부록 C: Reviewer 예상 질문 및 답변

**Q1**: "왜 stationary reference가 필요 없다고 주장하는가? Diffusion model은 stationary로 시작해야 reverse가 정합 아닌가?"

**A**: Score matching loss는 transition kernel score $\nabla \log p_{r|0}$을 학습하고, sampling은 학습된 marginal $p_K$에서 시작해서 reverse SDE로 $p_0$ (data distribution)에 도달함. $p_K$가 forward SDE의 true stationary일 필요 없음 — 학습된 marginal과 sampling 초기화 분포가 일치하면 충분. 본 framework는 sampling 초기화를 chart Gaussian으로 잡고, score net이 그에 대응하는 marginal을 학습함. Standard practice.

**Q2**: "Pose extension에서 drift OFF가 작동한다는 직접 증거는?"

**A**: R1 ablation 측정 결과 제시. Position-only V2에서 drift OFF + multi-component guidance가 succ@5cm 99%, pos_err 21mm를 달성한 evidence가 있고, pose extension은 같은 framework structure를 공유하므로 strong induction. R1 결과로 직접 검증.

**Q3**: "Joint limit violation은 어떻게 처리하는가?"

**A**: Manifold definition은 smoothness 위해 joint limit 미포함. Sampling 후 evaluation metric으로 violation rate 보고. Demo distribution이 joint limit 안에 있으면 학습된 score가 demo를 따라가 sample도 안에 있을 가능성 높음 (empirical 검증). 강제가 필요한 deployment에서는 bounded chart parameterization (e.g., $q_i = q_{\text{mid}, i} + (q_{\text{range}, i}/2) \tanh u_i$) 또는 post-hoc clamping. Future work으로 명시.