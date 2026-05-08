# Riemannian Score-Based Imitation Learning on Learned Robot Self-Model Manifolds
## Trajectory Generation on the Tangent Bundle of a Learned Feasibility Manifold under Embodiment Uncertainty

---

## 0. One-line summary

로봇의 self-model을 ambient \(\mathbb{R}^d\) 안의 differentiable implicit embedded manifold \(\mathcal{M}_\phi(z_e) = \{x : g_\phi(x, z_e) = 0\}\)로 학습하고, ambient Euclidean metric이 자동으로 induce하는 Riemannian geometry (induced metric \(G(q, z_e) = J_H^T J_H\)) 위에서 **score-based generative modeling**을 수행한다. Self-model의 미분 구조가 manifold tangent bundle을 정의하며 (feasible velocity bundle), demonstration trajectory는 product manifold \(\mathcal{M}_\phi^{H+1}\) 위 distribution으로 다뤄진다.

본 연구의 핵심 idea: **Vanilla diffusion이 R^d isotropic Gaussian noise에서 데이터로 복원하는 vector field를 학습한다면, 본 연구는 self-model이 정의하는 manifold 위 Riemannian Brownian motion에서 데이터로 복원하는 tangent vector field를 학습한다.** Sampling은 manifold 위에서만 진행되며 (by construction), physical feasibility가 stochastic 학습이 아니라 construction에 의해 보장된다.

본 연구의 위치는 **paradigm 차이**로 정의된다:

1. **vs vanilla diffusion**:
   Vanilla는 R^d Gaussian noise space. 본 연구는 학습된 manifold 위 Riemannian noise space. Diffusion 자체의 geometric structure가 본질적으로 다름.

2. **vs Riemannian SGM (De Bortoli et al. 2022)**:
   RSGM은 known manifold (sphere, SO(3) 등). 본 연구는 robot self-model로 학습된 manifold. RSGM의 learned-manifold robotics instance.

3. **vs Projected Diffusion / SafeDiffuser / Manifold Preserving Diffusion**:
   이들은 ambient에서 생성 후 post-hoc projection. 본 연구는 처음부터 manifold-intrinsic.

4. **vs ATACOM**:
   ATACOM은 known constraint manifold + RL stochastic policy. 본 연구는 learned manifold + diffusion-based imitation learning.

5. **vs NJF**:
   NJF는 Jacobian field 직접 regression (paradigm A). 본 연구는 differentiable function 학습 후 미분 구조와 ambient metric으로부터 Riemannian geometry 자동 derive (paradigm B).

---

## 1. 핵심 문제의식

### 1.1 Vanilla diffusion policy의 feasibility 처리

Vanilla diffusion policy는 R^d 공간의 isotropic Gaussian noise에서 데이터 분포로 복원하는 reverse process를 학습한다. Physical feasibility는 다음 가정에 의존:

> 충분한 demonstration data가 주어지면 model이 feasible action distribution의 support를 학습한다.

본질적 한계:

**(a) Score field 발산**: Manifold 근방에서 score field가 manifold 바깥으로 새어나감. Distribution fitting과 support learning은 다름.

**(b) Stochastic dependency**: Physical feasibility는 task-independent하지만 task-conditional distribution에 implicit으로 학습되므로 비효율.

**(c) Embodiment uncertainty**: Tool change, calibration drift 시 demonstration 분포와 deployment feasible set이 mismatch.

### 1.2 기존 constraint-aware diffusion의 한계

Projected Diffusion Models, Manifold Preserving Guided Diffusion, SafeDiffuser:
- Ambient에서 생성 후 post-hoc projection
- Score field 자체는 ambient
- Per-step projection의 누적 오차
- Manifold-intrinsic distribution recovery의 이론적 정당성 약함

### 1.3 본 연구의 입장

> Diffusion process 자체가 R^d Euclidean이 아니라 학습된 self-model manifold 위에서 정의되어야 한다. Forward noising도, reverse generation도, score field도 모두 manifold-intrinsic해야 한다.

이를 위해:

1. Self-model을 differentiable embedded manifold M ⊂ R^d로 학습
2. Ambient metric이 M 위에 Riemannian geometry 자동 induce: \(G(q, z_e) = J_H^T J_H = I + J_F^T J_F\)
3. 그 미분 구조 (tangent bundle TM)가 feasible velocity bundle
4. Demo trajectory를 product manifold \(\mathcal{M}^{H+1}\) 위 distribution으로
5. Riemannian SGM으로 학습 (forward/reverse SDE on M with metric G)
6. Embodiment context z_e가 M의 deformation parameter

---

## 2. Self-model의 representational choice

### 2.1 Paradigm

**Paradigm B (본 연구)**: Differentiable function learning
- Equality constraint를 미분 가능 함수로 표현
- 미분 구조 (Jacobian, tangent bundle)는 autograd로 derive
- Ambient metric이 Riemannian geometry induce

NJF의 derivative regression paradigm은 (i) manifold framing이 어색, (ii) tangent bundle / Riemannian geometry derive 불가, (iii) ambient metric induced structure 없음.

### 2.2 Implicit embedded manifold

\[
\mathcal{M}_\phi(z_e) = \{x \in \mathbb{R}^d : g_\phi(x, z_e) = 0\}
\]

여기서 \(g_\phi: \mathbb{R}^{d+n_z} \to \mathbb{R}^m\). Codimension은 m, manifold dimension은 d - m.

z_e는 embodiment context (tool length 등)이며 manifold의 deformation parameter.

### 2.3 First instantiation: residual FK form

\[
g_\phi(q, p_{ee}, z_e) = p_{ee} - FK_{\text{analytic}}(q, z_e) - \Delta_\phi(q, z_e)
\]

여기서:
- Tool transform, calibration 등은 \(\Delta_\phi\) residual로 capture
- z_e에 의존 (예: tool length \(\ell\))

이 form은 graph manifold:

\[
\mathcal{M}_\phi(z_e) = \{(q, p_{ee}) : p_{ee} = FK_{\text{analytic}}(q, z_e) + \Delta_\phi(q, z_e)\}
\]

Graph manifold의 chart는 q. 즉 q ∈ R^{n_q}가 manifold의 natural coordinate.

### 2.4 Tangent bundle = feasible velocity bundle

각 점 (q, p_ee) ∈ M에서 tangent space:

\[
T_{(q, p_{ee})} \mathcal{M}_\phi = \{(\dot q, \dot p_{ee}) : \dot p_{ee} = J_{FK,\text{total}}(q, z_e) \dot q\}
\]

여기서 \(J_{FK,\text{total}} = J_{FK,\text{analytic}} + \partial \Delta_\phi / \partial q\).

**핵심**: 이 tangent bundle 자체가 본 연구가 말하는 "feasible velocity bundle"이다. Self-model의 미분 구조로부터 자동으로 따라온다 — 별도 inequality constraint 없이.

Tangent space의 natural parameterization: \(\dot q \in \mathbb{R}^{n_q}\)이 chart coordinate, 그에 대응하는 ambient tangent vector는 \((\dot q, J_{FK,\text{total}} \dot q) = J_H(q, z_e) \dot q\).

---

## 3. 변수와 객체 정의

### 3.1 State

- \(q \in \mathbb{R}^{n_q}\): joint configuration
- \(p_{ee} \in \mathbb{R}^{n_p}\): end-effector / tool-tip pose
- \(z_e \in \mathbb{R}^{n_z}\): embodiment context
- \(o \in \mathbb{R}^{n_o}\): observation (optional)
- \(x = (q, p_{ee}) \in \mathbb{R}^d\), \(d = n_q + n_p\)

### 3.2 Manifold 객체

- \(\mathcal{M}_\phi(z_e) = \{x : g_\phi(x, z_e) = 0\}\), ambient submanifold of R^d
- \(J_g(x, z_e) = \partial g_\phi / \partial x \in \mathbb{R}^{m \times d}\): Jacobian
- \(T_x \mathcal{M}_\phi = \ker J_g(x, z_e) \subset \mathbb{R}^d\): tangent space
- TM: tangent bundle, fiber bundle of all tangent spaces

**Graph manifold case의 explicit form**:

\(g_\phi(q, p, z_e) = p - F_\phi(q, z_e)\)에서 직접:

\[
J_g(q, p, z_e) = \begin{bmatrix} -J_F(q, z_e) & I_{n_p} \end{bmatrix} \in \mathbb{R}^{n_p \times d}
\]

여기서:

\[
J_F(q, z_e) = \frac{\partial F_\phi(q, z_e)}{\partial q} = J_{FK,\text{analytic}}(q, z_e) + \frac{\partial \Delta_\phi(q, z_e)}{\partial q}
\]

Tangent vector \(v = (\dot q, \dot p) \in T_x M\) 조건:

\[
J_g v = -J_F \dot q + \dot p = 0 \implies \dot p = J_F(q, z_e) \dot q
\]

### 3.3 Riemannian structure (induced from ambient embedding)

Ambient metric \(\langle u, v \rangle = u^T v\) on R^d induces metric on M (first fundamental form):

\[
\langle u, v \rangle_{\mathcal{M}, x} = u^T v, \quad u, v \in T_x \mathcal{M}_\phi
\]

**Graph manifold case**: M = {(q, p) : p = F_φ(q, z_e)}에서 chart는 q ∈ R^{n_q}. Embedding map H_φ(q) = (q, F_φ(q, z_e))의 Jacobian:

\[
J_H(q, z_e) = \begin{pmatrix} I_{n_q} \\ J_{F_\phi}(q, z_e) \end{pmatrix} \in \mathbb{R}^{d \times n_q}
\]

Induced metric (chart coordinate q-space에서 표현):

\[
G(q, z_e) = J_H^T J_H = I_{n_q} + J_{F_\phi}(q, z_e)^T J_{F_\phi}(q, z_e)
\]

**Tangent geometric consistency check**: \(J_g J_H = [-J_F, I][I; J_F]^T = -J_F + J_F = 0\). 따라서 \(\text{Im}(J_H) \subseteq \ker J_g\), 그리고 차원이 둘 다 n_q이므로 \(\text{Im}(J_H) = T_x \mathcal{M}_\phi\). Embedding map의 image가 정확히 tangent space.

**핵심**: 이 G(q, z_e)는 본 framework의 자동 산출물이다. Self-model의 미분 (J_{F_φ}) + ambient R^d metric의 induce가 곧 G. 별도 metric 학습 불필요. Core idea의 "ambient metric이 자동으로 Riemannian geometry 정의"가 이 G의 명시적 form.

**Ambient norm vs chart G-norm 등가성**: Tangent vector를 두 표기로 쓸 수 있다.

- **Ambient form**: \(u \in T_x \mathcal{M}_\phi \subset \mathbb{R}^d\) (ambient vector)
- **Chart form**: \(a \in \mathbb{R}^{n_q}\) (chart coordinate)

관계: \(u = J_H(q, z_e) a\). 두 norm은 같은 수치:

\[
\|u\|^2_{\text{ambient}} = u^T u = (J_H a)^T (J_H a) = a^T J_H^T J_H a = a^T G(q, z_e) a = \|a\|^2_G
\]

즉 ambient norm (Euclidean R^d on tangent vector)과 chart G-norm (q-space with induced metric)이 등가. Implementation 시 어느 표기를 쓰느냐만 일관되면 됨.

이로부터 자동 정의:
- Tangent vector length: \(\|u\|^2 = \|a\|^2_G = a^T G(q, z_e) a\)
- Riemannian gradient (chart 표현): \(\nabla_M f = G(q, z_e)^{-1} \nabla_q \bar{f}\), where \(\bar{f}(q) = f(H_\phi(q, z_e))\)
- Ambient form Riemannian gradient: \(\text{grad}_M f = J_H G^{-1} \nabla_q \bar{f}\)
- Brownian motion on M with metric G
- Volume form: \(\sqrt{\det G(q, z_e)} \, dq\)

**Chart-based approximation vs full Riemannian**: 첫 implementation에서 q-space에서 Euclidean Brownian/score를 사용하는 것은 G ≈ I 가정의 chart-based approximation. 진짜 Riemannian SGM은 G(q, z_e)를 명시적으로 사용. 본 framework은 후자를 main으로 하고, chart approximation은 toy implementation의 simplification으로 명시.

### 3.4 Trajectory sample space (product manifold)

Demo trajectory τ = (x_0, x_1, ..., x_H), 각 x_h ∈ M(z_e). Sample space는 product manifold:

\[
\tau \in \mathcal{M}_\phi(z_e)^{H+1} = \mathcal{M}_\phi(z_e) \times \cdots \times \mathcal{M}_\phi(z_e)
\]

Tangent space (product):

\[
T_\tau \mathcal{M}_\phi^{H+1} = T_{x_0} \mathcal{M}_\phi \times T_{x_1} \mathcal{M}_\phi \times \cdots \times T_{x_H} \mathcal{M}_\phi
\]

Product metric:

\[
\langle u, v \rangle_\tau = \sum_{h=0}^H \langle u_h, v_h \rangle_{x_h}
\]

Riemannian SGM은 이 product manifold에 적용된다.

**Notation 구분**:
- h ∈ {0, 1, ..., H}: trajectory index (robot motion timestep)
- r ∈ [0, 1]: diffusion process time (noising/denoising progress)

이 둘은 명확히 다른 객체.

### 3.5 Demo distribution

\[
\tau_i \sim p_{\text{demo}}(\tau | g, z_e), \quad \tau_i \in \mathcal{M}_\phi(z_e)^{H+1}
\]

학습 목표:

\[
p_\theta(\tau | g, z_e) \approx p_{\text{demo}}(\tau | g, z_e)
\]

### 3.6 Tangent bundle as feasible velocity space

각 점 x ∈ M에서 가능한 instantaneous motion direction:

\[
T_x \mathcal{M}_\phi = \{v \in \mathbb{R}^d : J_g(x, z_e) v = 0\}
\]

이 tangent space의 모음 (tangent bundle TM)이 manifold 위 feasible velocity bundle. Score field는 이 tangent bundle의 section:

\[
s_\theta : \mathcal{M}_\phi \to T\mathcal{M}_\phi, \quad s_\theta(x) \in T_x \mathcal{M}_\phi
\]

Diffusion process의 모든 객체 (drift, noise, score)가 이 tangent bundle에 산다.

**Note on terminology**: "feasible velocity bundle"이라는 표현을 쓰지만, 엄밀히는 (i) tangent bundle TM은 manifold 자체의 객체이고, (ii) 그 위 vector field (tangent bundle의 section)가 "velocity field". Score field가 이 vector field의 한 instance.

---

## 4. Riemannian Score-Based Generative Modeling on M

### 4.1 핵심 framework

본 연구는 Riemannian SGM (De Bortoli et al. 2022)을 학습된 manifold M (또는 product manifold M^{H+1})에 적용한다. Diffusion time을 r ∈ [0, 1]로 표기 (trajectory index h와 구분).

**Forward process** (manifold 위 SDE, with induced metric G):

\[
dX_r = b(X_r) dr + dB^M_r
\]

여기서:
- \(B^M_r\)는 manifold M 위 Brownian motion w.r.t. induced metric G
- \(b(X_r) \in T_{X_r} \mathcal{M}_\phi\)는 drift term
- Brownian motion이 metric G에 의존: chart coordinate q-space에서 보면 dq_r에 G(q)^{-1/2} factor가 들어감

Bounded workspace의 robot manifold에 적합한 default: Langevin with wrapped Gaussian target

\[
b(X_r) = -\frac{1}{2} \nabla_M U(X_r), \quad U(x) = d_M(x, \mu)^2 / (2\gamma^2)
\]

여기서 \(\nabla_M\)는 Riemannian gradient (= G^{-1} ∇_q in chart), \(d_M\)는 geodesic distance.

**Reverse process**:

\[
dY_r = \{-b(Y_r) + \nabla_M \log p_{1-r}(Y_r)\} dr + dB^M_r
\]

\(\nabla_M \log p_r\)는 manifold 위 score (tangent bundle의 section). 학습 대상.

**Trajectory-level reverse process** (product manifold):

\[
d\Tau_r = \{-b(\Tau_r) + \nabla_{\mathcal{M}^{H+1}} \log p_{1-r}(\Tau_r | g, z_e)\} dr + dB^{\mathcal{M}^{H+1}}_r
\]

학습할 score:

\[
s_\theta(r, \tau, g, z_e) \approx \nabla_{\mathcal{M}^{H+1}} \log p_r(\tau | g, z_e)
\]

### 4.2 Trajectory-level diffusion (product manifold)

Trajectory τ ∈ M^{H+1}에 대한 diffusion. Forward SDE는 product manifold 위 process:

\[
d\tau_r = b(\tau_r) dr + dB^{\mathcal{M}^{H+1}}_r, \quad \tau_r \in \mathcal{M}_\phi^{H+1}
\]

**Component-wise form**:

\[
dX_{h, r} = b_h(\tau_r) dr + dB^{\mathcal{M}}_{h, r}, \quad h = 0, 1, \ldots, H
\]

가장 단순한 independent noising은 \(b_h(\tau_r) = b(X_{h, r})\) (component별 같은 drift). 더 정교하게는 score network가 trajectory smoothness 같은 component coupling 학습.

Product Brownian motion의 component들은 independent. Product metric은 §3.4 참조.

### 4.3 Vanilla diffusion과의 본질적 차이

| 항목 | Vanilla diffusion | 본 연구 |
|---|---|---|
| Noise space | R^d isotropic Gaussian | Manifold 위 (wrapped) Gaussian with metric G |
| Forward 종착점 | R^d Gaussian | M 위 wrapped Gaussian |
| Score field | \(\nabla \log p_r \in \mathbb{R}^d\) | \(\nabla_M \log p_r \in T_x M\) |
| Effective dimension | d | dim(M) = d - m |
| Manifold adherence | Stochastic learning에 의존 | By construction |
| Embodiment 변화 | Implicit | M의 deformation으로 명시적 |

### 4.4 Score model

Score는 tangent bundle TM의 section:

\[
s_\theta(r, x, g, z_e) \in T_x \mathcal{M}_\phi(z_e)
\]

Trajectory-level의 경우 product tangent space:

\[
s_\theta(r, \tau, g, z_e) \in T_\tau \mathcal{M}_\phi^{H+1}
\]

Implementation (graph manifold에서 chart coordinate q 사용). Network는 chart coordinate로 \(s_\theta^q \in \mathbb{R}^{n_q}\)를 출력하고, ambient tangent vector로 lift:

\[
s_\theta^{\text{amb}}(r, x, g, z_e) = J_H(q, z_e) s_\theta^q(r, q, g, z_e) = \begin{pmatrix} s_\theta^q \\ J_F(q, z_e) s_\theta^q \end{pmatrix}
\]

**Tangent verification**: 이 lift된 vector는 by construction tangent space에 속함:

\[
J_g \, s_\theta^{\text{amb}} = J_g J_H s_\theta^q = 0 \cdot s_\theta^q = 0
\]

(§3.3의 \(J_g J_H = 0\) 결과 사용). 따라서 score model의 output이 자동으로 manifold tangent에 속함 — explicit projection 필요 없음.

**Trajectory-level**:

\[
s_\theta(r, \tau, g, z_e) = (s_{\theta, 0}, s_{\theta, 1}, \ldots, s_{\theta, H}), \quad s_{\theta, h} \in T_{x_h} \mathcal{M}_\phi
\]

각 component가 단일 point case와 동일한 chart-to-ambient lift.

### 4.5 Score matching loss

**Full Riemannian DSM (ambient form)**:

\[
\mathcal{L}_{\text{score}} = \mathbb{E}_{r, x_0, x_r} \left[ \left\| s_\theta(r, x_r, g, z_e) - \frac{\exp_{x_r}^{-1}(x_0)}{r} \right\|^2_{x_r}
\right]
\]

여기서 \(\|\cdot\|^2_{x_r}\)는 manifold 위 norm. Ambient tangent vector u에 대해 \(\|u\|^2_{x_r} = u^T u\) (ambient form).

Varadhan asymptotic (small r): \(\exp_{x_r}^{-1}(x_0) / r \approx\) tangent vector pointing from x_r to x_0.

**Chart coordinate form (등가)**:

Chart에서 score를 \(s_\theta^q(r, q_r, g, z_e) \in \mathbb{R}^{n_q}\)로 출력하고, target chart-coordinate vector를 \(a^*(q_r, q_0)\)로 두면:

\[
\mathcal{L}_{\text{score}}^{\text{chart-G}} = \mathbb{E}_{r, q_0, q_r} \left[ (s_\theta^q - a^*)^T G(q_r, z_e) (s_\theta^q - a^*) \right]
\]

여기서 norm이 G-weighted. Ambient form과 등가:

\[
\|J_H s_\theta^q - J_H a^*\|^2_{\text{ambient}} = (s_\theta^q - a^*)^T G (s_\theta^q - a^*)
\]

Small r에서 \(a^* \approx \text{Log}_{q_r}(q_0) / r\) (manifold logarithm in chart).

**Chart-based approximation** (G ≈ I, 첫 toy implementation의 simplification):

\[
\mathcal{L}_{\text{score}}^{\text{chart-Eucl}} \approx \mathbb{E}_{r, q_0, q_r} \left[ \left\| s_\theta^q(r, q_r, g, z_e) - \frac{q_0 - q_r}{r} \right\|^2 \right]
\]

이건 chart-space Euclidean DSM. Full Riemannian DSM과 다름 (G가 빠짐).

**구분 정리**:
- \(\mathcal{L}_{\text{score}}^{\text{ambient}}\): full Riemannian, ambient norm 사용
- \(\mathcal{L}_{\text{score}}^{\text{chart-G}}\): full Riemannian, chart에서 G-weighted norm. Ambient와 등가.
- \(\mathcal{L}_{\text{score}}^{\text{chart-Eucl}}\): chart approximation, G = I 가정. Toy simplification.

본 framework의 main은 첫 둘 (등가). 셋째는 toy implementation의 명시적 simplification.

### 4.6 Trajectory-level score matching loss (product manifold)

Trajectory \(\tau = (x_0, \ldots, x_H) \in \mathcal{M}^{H+1}\)에 대해 noised trajectory \(\tau_r = (x_{0,r}, \ldots, x_{H,r})\). Target은 component-wise tangent vector:

\[
A^* = \left( \frac{\exp_{x_{0,r}}^{-1}(x_{0,0})}{r}, \ldots, \frac{\exp_{x_{H,r}}^{-1}(x_{H,0})}{r} \right)
\]

**Full Riemannian DSM (trajectory, ambient form)**:

\[
\mathcal{L}_{\text{traj}}^{\text{ambient}} = \mathbb{E} \left[ \sum_{h=0}^H \left\| s_{\theta, h}(r, \tau_r, g, z_e) - \frac{\exp_{x_{h,r}}^{-1}(x_{h,0})}{r} \right\|^2_{x_{h,r}} \right]
\]

Product metric에 의해 sum이 자연스러움.

**Chart-G form (등가)**:

\[
\mathcal{L}_{\text{traj}}^{\text{chart-G}} = \mathbb{E} \left[ \sum_{h=0}^H (s_{\theta, h}^q - a_h^*)^T G(q_{h,r}, z_e) (s_{\theta, h}^q - a_h^*) \right]
\]

**Chart approximation**:

\[
\mathcal{L}_{\text{traj}}^{\text{chart-Eucl}} \approx \mathbb{E} \left[ \sum_{h=0}^H \left\| s_{\theta, h}^q - \frac{q_{h,0} - q_{h,r}}{r} \right\|^2 \right]
\]

---

## 5. Sampling: Retraction-based Geodesic Random Walk

### 5.1 Manifold 위 SDE 시뮬레이션

학습된 manifold에서 exponential map은 closed-form으로 안 나옴. Retraction으로 approximate (RSGM 논문 Algorithm 1):

```
x_0 ~ p_ref (manifold 위 prior, 예: wrapped Gaussian)
For k = 0, ..., N-1:
    Compute drift: b_k = -b(x_k) + s_θ(r_k, x_k, g, z_e)  (tangent vector)
    Sample tangent noise: ξ_k ~ N(0, G(q_k)^{-1})  (Riemannian Gaussian on tangent)
    Tangent step: w_k = Δr · b_k + sqrt(Δr) · ξ_k  ∈ T_{x_k} M
    Retraction: x_{k+1} = Retr_{x_k}(w_k)
```

여기서 r_k는 diffusion time (k는 sampling iteration index).

**Itô SDE in chart coordinates (정확한 form)**:

진짜 Riemannian Brownian motion을 chart q-space에서 표현하면 단순 \(dq_r = dW_r\)가 아니다. Christoffel symbol에 해당하는 geometric drift correction이 들어간다:

\[
dq_r = b^q(q_r) dr - \frac{1}{2} G(q_r)^{-1} \partial_q (\log \det G(q_r)) \, dr + G(q_r)^{-1/2} dW_r
\]

두 번째 항이 metric의 곡률에서 오는 geometric drift. \(G \approx I\) (flat) 또는 G가 거의 constant이면 이 항 무시 가능.

**실용적 처리**:
- 첫 toy implementation: small step size + step-wise Gaussian \(\xi^q \sim \mathcal{N}(0, G^{-1})\)로 근사. Geometric drift correction 무시.
- Itô-correct version: full geometric drift 포함. Method extension.
- RSGM 논문 Algorithm 1은 manifold 위 직접 시뮬레이션이라 chart Itô issue 회피.

### 5.2 Graph retraction (graph manifold case)

Graph manifold M = {(q, p) : p = F_φ(q, z_e)}에서 retraction:

\[
\text{Retr}_{(q, p)}(\delta q, \delta p) = (q + \delta q, F_\phi(q + \delta q, z_e))
\]

즉 tangent step의 q-component만 사용해서 chart에서 step을 밟고, p는 self-model로 다시 lift.

**중요**: 이건 exact orthogonal projection이 **아니다**. Exact projection은:

\[
\text{Proj}_M(\tilde q, \tilde p) = \arg\min_{(q, p) \in M} \|(q, p) - (\tilde q, \tilde p)\|^2
\]

이걸 풀려면 \(q^* = \arg\min_q \|(q, F_\phi(q)) - (\tilde q, \tilde p)\|^2\)의 optimization 필요. Closed-form 아님.

대신 graph retraction은 q를 vertical하게 처리 (chart coordinate 그대로 사용)해서 cheap. RSGM Algorithm 1에서 Geodesic Random Walk가 retraction을 허용 (exp map 대신).

### 5.3 Tangent noise sampling

Tangent space의 chart coordinate가 q이므로, induced metric G(q) 하 Gaussian:

\[
\xi_k = J_H(q_k, z_e) \xi_k^q, \quad \xi_k^q \sim \mathcal{N}(0, G(q_k, z_e)^{-1})
\]

또는 chart-based approximation (G ≈ I):

\[
\xi_k^q \sim \mathcal{N}(0, I_{n_q}), \quad \xi_k = (\xi_k^q, J_{F_\phi}(q_k, z_e) \xi_k^q)
\]

이렇게 sampling된 noise는 by construction tangent에 속함.

### 5.4 Trajectory generation (product manifold sampling)

Conditional trajectory diffusion on product manifold:

\[
\tau \sim p_\theta(\tau | g, z_e), \quad \tau = (x_0, \ldots, x_H), \quad x_h \in \mathcal{M}_\phi(z_e)
\]

방식 (first instantiation):
- Trajectory 전체를 single sample로 보고 product manifold M^{H+1}에서 diffusion
- 각 component가 동시에 noising/denoising
- 각 step에서 trajectory의 모든 point가 manifold 위에 머무름 (per-component retraction)

**Component-wise sampling step**:

각 trajectory time h에 대해:

\[
\delta q_{h, k} = \Delta r \cdot \mu_{h, k}^q + \sqrt{\Delta r} \cdot \xi_{h, k}^q
\]

여기서:
- \(\mu_{h, k}^q = -b^q(q_{h, k}) + s_{\theta, h}^q(r_k, \tau_k, g, z_e)\): chart drift (reverse process)
- \(\xi_{h, k}^q \sim \mathcal{N}(0, G(q_{h, k}, z_e)^{-1})\): Riemannian Gaussian noise

Chart update + retraction:

\[
q_{h, k+1} = q_{h, k} + \delta q_{h, k}
\]

\[
x_{h, k+1} = H_\phi(q_{h, k+1}, z_e) = (q_{h, k+1}, F_\phi(q_{h, k+1}, z_e))
\]

따라서:

\[
\tau_{k+1} = (x_{0, k+1}, x_{1, k+1}, \ldots, x_{H, k+1}) \in \mathcal{M}_\phi(z_e)^{H+1}
\]

**Ambient form**: Component-wise tangent vector \(w_{h, k} = J_H(q_{h, k}, z_e) \delta q_{h, k}\), retraction \(x_{h, k+1} = \text{Retr}_{x_{h, k}}(w_{h, k}) = H_\phi(q_{h, k} + \delta q_{h, k}, z_e)\). Chart form과 등가.

### 5.5 Manifold adherence by construction

Forward, reverse 모든 step:
- Drift b는 tangent vector
- Noise ξ는 tangent vector
- Retraction이 manifold로 다시 mapping

따라서 매 step마다 \(\|g_\phi(x_h)\| \approx 0\) (numerical retraction 오차 내). Vanilla나 ambient projection diffusion과 본질적으로 다름.

---

## 6. Action-level vs Trajectory-level diffusion

본 연구는 두 level을 명시적으로 구분한다.

### 6.1 Trajectory-level (본 연구의 main)

Sample이 trajectory 전체:

\[
\tau \sim p_\theta(\tau | g, z_e), \quad \tau \in \mathcal{M}_\phi^T
\]

- Diffusion이 trajectory distribution on manifold 학습
- SDE의 dX_t/dt가 자동으로 tangent vector (manifold velocity)
- 모든 step이 manifold 위
- Riemannian SGM의 자연스러운 form

### 6.2 Action-level (method extension)

Sample이 한 시점 velocity:

\[
\dot q \sim p_\theta(\dot q | x, g, z_e)
\]

- 한 시점 velocity sampling, trajectory는 sequential rollout
- Manifold tangent의 chart coordinate
- 본 연구의 main은 아니지만 control loop integration에 유용

### 6.3 본 연구의 main claim 위치

> Trajectory-level Riemannian SGM이 본 연구의 main framework. Sample이 manifold 위 위치이고, SDE의 velocity field가 자동으로 self-model의 미분 구조 (tangent bundle)에 부합. Score field가 이 tangent bundle의 section.

### 6.4 학습 / inference pipeline

```
Training:
1. Demo trajectory τ ∈ M (with measurement noise → projection 한 번)
2. Forward: τ를 wrapped Gaussian으로 noising via Riemannian SDE
3. Score model 학습: Riemannian DSM with Varadhan

Inference:
1. Compute manifold M(z_e) via self-model
2. Sample τ_K ~ wrapped Gaussian on M
3. Reverse Riemannian SDE with score s_θ
4. Each step: retraction onto M
5. Return τ_0 (trajectory on M)
```

---

## 7. Self-model architecture

### 7.1 Residual kinematic only

\[
g_\phi(q, p_{ee}, z_e) = p_{ee} - FK_{\text{analytic}}(q, z_e) - \Delta_\phi(q, z_e)
\]

**Δ_φ network**:
```
Input: (q, z_e)
Layers: 3-layer MLP, hidden 128, Softplus activation
Output: Δ ∈ R^{n_p}
Init: small (residual starts near zero)
```

Softplus 선택: smooth (정확한 Jacobian autograd), Hessian 확장 가능.

### 7.2 Self-model 학습

다양한 z_e에서 self-exploration data:

\[
\mathcal{D}_{\text{self}} = \{(q_i, p_{ee, i}, z_{e, i})\}_i
\]

Forward prediction loss:

\[
\mathcal{L}_{\text{self}} = \mathbb{E}_{(q, p_{ee}, z_e) \sim \mathcal{D}_{\text{self}}} \left[ \|FK_{\text{analytic}}(q, z_e) + \Delta_\phi(q, z_e) - p_{ee}\|^2 \right] + \beta \mathbb{E}_q [\|\nabla_q \Delta_\phi\|_F^2]
\]

수렴 기준: kinematic loss < 10^-3, residual smoothness 안정.

### 7.3 Full pipeline (6-stage)

**Stage 1: Self-model learning**

\[
\mathcal{D}_{\text{self}} = \{(q_i, p_i, z_{e, i})\}_{i=1}^N
\]

\[
F_\phi(q, z_e) = FK_{\text{analytic}}(q, z_e) + \Delta_\phi(q, z_e)
\]

\[
\min_\phi \mathcal{L}_{\text{self}} = \mathbb{E}\left[\|F_\phi(q, z_e) - p\|^2\right] + \beta \mathbb{E}\left[\|\nabla_q \Delta_\phi\|_F^2\right]
\]

**Stage 2: Manifold construction**

\[
\mathcal{M}_\phi(z_e) = \{(q, p) : p = F_\phi(q, z_e)\}
\]

\[
H_\phi(q, z_e) = (q, F_\phi(q, z_e)), \quad J_H = \begin{bmatrix} I \\ J_F \end{bmatrix}
\]

\[
G(q, z_e) = J_H^T J_H = I + J_F^T J_F
\]

**Stage 3: Demo projection (preprocessing)**

Raw demo \(\tilde\tau_i = (\tilde q_h, \tilde p_h)_{h=0}^H\) (measurement noise로 manifold에서 약간 벗어남):

\[
q_h = \tilde q_h, \quad p_h = F_\phi(q_h, z_e)
\]

\[
\tau_i = (H_\phi(q_0), \ldots, H_\phi(q_H)) \in \mathcal{M}_\phi(z_e)^{H+1}
\]

이건 graph retraction; 정확한 orthogonal projection 아님 (§5.2 참조).

**Stage 4: Riemannian noising**

\[
\tau_0 \sim p_{\text{demo}}(\tau | g, z_e)
\]

\[
d\tau_r = b(\tau_r) dr + dB^{\mathcal{M}^{H+1}}_r
\]

**Stage 5: Score learning**

\[
s_\theta(r, \tau_r, g, z_e) \approx \nabla_{\mathcal{M}^{H+1}} \log p_r(\tau_r | g, z_e)
\]

Loss 선택:
- Full Riemannian (ambient): \(\mathcal{L}_{\text{traj}}^{\text{ambient}}\) (§4.6)
- Full Riemannian (chart-G): \(\mathcal{L}_{\text{traj}}^{\text{chart-G}}\) (§4.6)
- Chart approximation: \(\mathcal{L}_{\text{traj}}^{\text{chart-Eucl}}\) (§4.6, toy implementation simplification)

**Stage 6: Reverse sampling**

Initialize: \(\tau_K \sim p_{\text{ref}}(\tau | z_e)\) (wrapped Gaussian on M^{H+1})

For \(k = K, K-1, \ldots, 1\), each h ∈ {0, ..., H}:

\[
\delta q_{h, k} = \Delta r \left[-b_h^q(q_{h, k}) + s_{\theta, h}^q(r_k, \tau_k, g, z_e)\right] + \sqrt{\Delta r} \, \xi_{h, k}^q
\]

\[
\xi_{h, k}^q \sim \mathcal{N}(0, G(q_{h, k}, z_e)^{-1})
\]

\[
q_{h, k-1} = q_{h, k} + \delta q_{h, k}, \quad x_{h, k-1} = H_\phi(q_{h, k-1}, z_e)
\]

Return \(\tau_0 = (x_{0, 0}, \ldots, x_{H, 0}) \in \mathcal{M}_\phi(z_e)^{H+1}\).

---

## 8. Goal-conditioned generation

### 8.1 Goal embedding

Goal point g가 manifold M 위 한 점 또는 region:
- Position goal: 직접 M 위 점 (예: tool-tip target position)
- Task descriptor: M 위 distribution

### 8.2 Conditional Riemannian score

\[
s_\theta(r, x, g, z_e) \approx \nabla_M \log p_r(x | g)
\]

또는 guidance 형태:

\[
s_{\text{guided}}(r, x, g, z_e) = s_\theta(r, x) + \alpha \nabla_M R(x, g)
\]

여기서 R(x, g)는 task reward, \(\nabla_M\)는 Riemannian gradient (자동으로 tangent에 속함).

**Chart form (graph manifold)**:

Chart에서 Riemannian gradient는 \(\nabla_M f \leftrightarrow G(q, z_e)^{-1} \nabla_q \bar{f}(q)\) (\(\bar{f}(q) = f(H_\phi(q, z_e))\)). 따라서 chart-coordinate guidance:

\[
s_{\text{guided},h}^q = s_{\theta,h}^q(r, q_h, g, z_e) + \alpha \, G(q_h, z_e)^{-1} \nabla_{q_h} \bar{R}(q_h, g)
\]

\(G^{-1}\) factor가 chart에서 Riemannian gradient를 정확히 표현하기 위해 필요. Chart-based approximation에서는 \(G^{-1} \approx I\)로 두면 \(s_{\text{guided},h}^q \approx s_{\theta,h}^q + \alpha \nabla_{q_h} \bar{R}\).

### 8.3 Trajectory-level guidance (product manifold)

Trajectory τ = (x_0, ..., x_H) ∈ M^{H+1}에 대한 goal-conditioned score는 product tangent vector:

\[
s_\theta(r, \tau, g, z_e) = (s_{\theta,0}, s_{\theta,1}, \ldots, s_{\theta,H}), \quad s_{\theta,h} \in T_{x_h} \mathcal{M}
\]

각 component가 위 chart form으로 처리.

### 8.4 Motion-planning-like behavior

Conditional generation이 start에서 goal까지 manifold-respecting trajectory 생성. Demo distribution이 task-aware motion을 implicit하게 정의.

**Claim 강도**:
- Strong: "manifold-respecting trajectory generation by construction"
- Cautious: "motion-planning-like behavior emergent in demo distribution"
- Avoid: "automatic motion planning" (over-claim)

### 8.5 Generated trajectory를 control로

생성된 trajectory τ = (x_0, ..., x_H), 각 \(x_h = (q_h, p_h)\). 실제 robot 제어는 q_h sequence를 tracking:

\[
\text{control input}_h = \text{controller}(q_h, q_{h+1})
\]

또는 low-level controller가 q_h → q_{h+1} tracking. p_h는 self-model이 \(p_h = F_\phi(q_h, z_e)\)로 자동 결정 (consistency).

**중요한 분리**: 생성된 trajectory가 manifold M 위에 있다는 것은 \(p_h = F_\phi(q_h, z_e)\) 만족. 이는 kinematic consistency w.r.t. learned self-model. Torque/dynamics feasibility는 별도 layer (future work).

---

## 9. 기존 연구와의 차별화

### 9.1 Riemannian SGM (De Bortoli et al. 2022)

본 연구의 가장 직접적 framework reference.

| 항목 | RSGM | 본 연구 |
|---|---|---|
| Manifold | Known (sphere, SO(3), torus) | Learned (residual self-model) |
| 응용 | Density modeling, climate data | Robot imitation learning |
| Embodiment | 없음 | z_e conditioning |
| Goal | 없음 | Conditional generation |
| Multi-modal demo | 부수적 | 핵심 |

본 연구는 RSGM의 framework을 (i) learned manifold, (ii) robot imitation context, (iii) embodiment conditioning, (iv) goal-conditional generation으로 확장.

### 9.2 ATACOM (Liu et al. CoRL 2021, TRO 2024)

| 항목 | ATACOM | 본 연구 |
|---|---|---|
| Constraint manifold | Known analytic | Learned (residual self-model) |
| Learning framework | RL | Imitation learning |
| Policy class | Stochastic Gaussian (single-step) | Riemannian SGM (multi-step) |
| Generation | Action-level | Trajectory-level |
| Manifold framing | Tangent space exploration | Manifold-intrinsic SDE |
| Embodiment context | 다루지 않음 | 핵심 (z_e) |

차이는 (i) learned vs known manifold, (ii) imitation vs RL, (iii) trajectory-level diffusion vs single-step Gaussian, (iv) z_e conditioning.

### 9.3 Projected Diffusion / SafeDiffuser / Manifold Preserving Diffusion

| 항목 | Projected Diffusion 류 | 본 연구 |
|---|---|---|
| Diffusion space | Ambient + post-hoc projection | Manifold-intrinsic |
| Score field | Ambient \(\nabla \log p_t \in \mathbb{R}^d\) | Tangent bundle section \(\nabla_M \log p_t \in T_x M\) |
| Manifold satisfaction | Per-step projection (누적 오차) | By construction |
| Theoretical foundation | Projection theory | Riemannian SDE theory |

### 9.4 NJF (Neural Jacobian Fields)

| 항목 | NJF | 본 연구 |
|---|---|---|
| Self-model | Jacobian regression (paradigm A) | Differentiable function + autograd (paradigm B) |
| Manifold framing | 없음 | 핵심 (level set + ambient embedding) |
| Riemannian structure | 없음 | Ambient metric induced |
| Tangent bundle | 일관성 없음 | 자연스러움 |
| Trajectory generation | IK gradient descent | Riemannian SGM |

### 9.5 ECoMaNN

ECoMaNN은 task constraint manifold를 학습 (implicit form \(g_\phi(x) = 0\)). 본 연구와 implicit form은 같지만:

| 항목 | ECoMaNN | 본 연구 |
|---|---|---|
| Manifold semantics | Task constraint | Robot embodiment feasibility |
| Differential structure 사용 | Normal direction (point projection) | Tangent bundle (Riemannian SGM) |
| 통합 대상 | Sampling-based motion planner | Diffusion-based imitation learning |

### 9.6 Gupta et al. 2026 (Cross-robot transfer, Sci. Robot.)

| 항목 | Gupta et al. | 본 연구 |
|---|---|---|
| 목적 | Cross-robot skill transfer | Diffusion feasibility under embodiment uncertainty |
| Robot kinematic 가정 | Analytic하게 정확히 알려짐 | Calibration drift, compliance 등 uncertainty 다룸 |
| Constraint | Analytic kinematic classification | Learned residual self-model |
| Generation | Globally stable dynamical system | Riemannian SGM |
| Multi-modal | Unimodal | Diffusion's multi-modal |

다른 문제를 다루는 complementary work.

---

## 10. Contribution

### Main Contribution 1: Riemannian SGM on learned robot self-model manifold

Robot self-model을 differentiable embedded manifold M ⊂ R^d로 학습하고, ambient Euclidean metric이 induce하는 Riemannian geometry 위에서 score-based generative modeling을 수행한다. Forward / reverse diffusion, score field, sampling 모두 manifold-intrinsic. Riemannian SGM (De Bortoli 2022)의 robotics에서의 learned-manifold instance.

### Main Contribution 2: Trajectory generation by construction on the manifold

Trajectory 전체가 manifold 위 distribution으로 학습된다. Sampling은 retraction-based geodesic random walk로 매 step manifold에 머무름. Vanilla diffusion이 R^d Gaussian noise space에서 작동하고 ambient projection diffusion이 post-hoc projection으로 manifold에 mapping하는 것과 본질적으로 다름.

### Main Contribution 3: Embodiment-context-aware self-model manifold

\(z_e\) (tool length, payload 등)가 manifold M의 deformation parameter. Tool change, calibration drift 같은 embodiment uncertainty 하에서 self-model이 deployment-time 변화를 capture하고, manifold 자체가 그에 맞게 변형. ATACOM의 fixed analytic constraint, NJF의 z_e-blind self-model, RSGM의 fixed manifold 모두 다루지 않는 차원.

### Supporting Contribution: Empirical validation

Embodiment perturbation regime에서:
- Vanilla diffusion + z_e conditioning 대비 manifold adherence 우월
- Ambient projection diffusion 대비 cumulative projection 오차 없음
- Oracle analytic FK + z_e baseline 대비 residual learning 가치 검증
- Multi-modal trajectory distribution capture

---

## 11. Claim의 위계

### Strongly claim

- Self-model이 정의하는 manifold 위 Riemannian geometry가 ambient embedding으로 자동 정의됨 (별도 metric 학습 불필요)
- Multi-step Riemannian SGM의 모든 step이 manifold 위 (forward, reverse 포함)
- Sampling이 by construction manifold에 머무름 (retraction-based)
- Score field가 tangent bundle의 section
- Embodiment context z_e가 manifold deformation으로 자연스럽게 통합
- Multi-modal trajectory distribution capture

### Cautiously claim

- Embodiment generalization (in-distribution z_e interpolation)
- Motion-planning-like behavior in demo distribution
- Data efficiency from manifold prior
- Compliance / calibration drift 같은 analytic-uncapturable effect의 residual learning

### Avoid claim

- 추론 속도 향상
- Motion planning이 자동 (demo가 task 정의한다는 것 명시)
- 이론적 distribution recovery 보장 (chart-local만)
- 모든 task에서의 우월성

---

## 12. Future work

본 연구의 first instantiation에서 제외된 사항들:

**Inequality constraints (joint limit, manipulability)**: Manifold M에 inequality 추가하면 manifold-with-boundary가 되어 Riemannian SGM이 더 미묘해짐. Boundary 근처 score behavior 분석 필요. 첫 instantiation은 equality (kinematic consistency)만. Inequality는 future work.

**General implicit manifolds**: Graph manifold (residual FK)를 main으로. 일반 g_φ(x) = 0 form은 SVD basis discontinuity 등 issue 다뤄야 함. Method extension.

**Higher-order structure (curvature, Hessian)**: 1차 미분만 사용 (Jacobian, tangent). Curvature-aware sampling은 future work.

**Action chunk-aware feasibility**: Trajectory 내 chart change 처리는 first instantiation에서 simplified.

**Multi-robot transfer**: Gupta et al. 2026의 cross-robot transfer를 본 framework과 결합하는 것은 별도 연구 방향.

---

## 13. 위험 요소

### 13.1 Self-model 학습 quality

Risk: Δ_φ가 ground truth compliance/calibration을 정확히 capture 못함.

대응:
- Residual form이라 worst case에 analytic FK로 환원
- 다양한 z_e에서 self-exploration data
- Validation loss monitoring

### 13.2 Manifold projection의 numerical stability

Risk: Retraction step에서 projection 부정확.

대응: Graph manifold form은 projection이 self-model의 forward call 한 번 (closed-form). 매우 안정. 일반 implicit form은 future work.

### 13.3 Demo가 manifold에 정확히 있지 않음

Risk: Real demo는 measurement noise로 manifold에서 약간 벗어날 수 있음.

대응:
- 학습 전 demo를 manifold로 projection 한 번 적용
- Graph manifold projection은 closed-form
- Projection 후 demo가 manifold 위에 있다고 가정

### 13.4 Bounded workspace의 처리

Risk: Robot manifold가 strictly compact가 아님 (joint limit, workspace boundary).

대응: Wrapped Gaussian target with Langevin dynamics가 bounded region에 자연스럽게 적합. RSGM 논문의 §3.1 두 번째 옵션.

### 13.5 Inference 비용

Risk: Multi-step Riemannian SDE + retraction이 vanilla diffusion보다 무거움.

대응:
- Graph manifold projection은 O(1)
- Score evaluation이 dominant cost (vanilla와 비슷)
- Total inference cost는 vanilla diffusion + manifold projection 정도

### 13.6 Score model의 tangent constraint

Risk: Score model output이 tangent space에 정확히 안 떨어질 수 있음.

대응:
- Graph manifold에서 chart coordinate (q-space)로 reparameterize → 자동으로 tangent
- Lift는 self-model이 처리

### 13.7 Chart Itô SDE의 geometric drift correction

Risk: Chart coordinate에서 Riemannian Brownian motion을 시뮬레이션할 때 Itô-correct form은 geometric drift \(-\frac{1}{2} G^{-1} \partial_q (\log \det G) dr\)을 포함. 첫 implementation에서 이걸 무시하면 sampling distribution이 약간 distort.

대응:
- 첫 toy implementation: small step size + step-wise Gaussian \(\mathcal{N}(0, G^{-1})\). G가 reasonably smooth하면 drift correction의 영향 미미.
- Toy 검증 시 sampling distribution과 demo distribution의 Wasserstein distance 측정.
- Method extension: full Itô-correct chart SDE with \(\partial_q G\) computation.
- 또는 RSGM 논문 Algorithm 1 처럼 manifold 위 직접 시뮬레이션 (chart issue 회피).

---

## 14. 개발 순서

### Phase 0: Design document (현재)

### Phase 1: Toy 1 - S^1 sanity check (2-3주)

S^1 ⊂ R^2 (analytic manifold, 학습 안 함):
- Full Riemannian SGM (induced metric trivial: G = 1) 구현
- Wrapped Gaussian distribution 학습 후 sampling
- Sampling이 S^1에 머무는지 (||g_φ|| < ε)
- Vanilla diffusion + ambient projection과 정량 비교

S^1은 induced metric이 단순 (G = 1)이라 full Riemannian SGM이 chart approximation과 일치. Framework sanity check에 적합.

**Stop and pivot**: framework 자체에 문제 있으면 RSGM 논문 reference 코드 참조.

### Phase 2: Toy 2 - 2-link arm graph manifold (3-4주)

2-link planar arm with analytic FK (학습 안 함):
- M = {(q, p) : p = FK(q)} ⊂ R^4
- Demo trajectory를 product manifold M^{H+1} 위 distribution으로
- 두 sub-phase:
  - **2a (chart-based approximation)**: q-space Euclidean DSM, lift via H_φ. Implementation 단순.
  - **2b (full Riemannian)**: Induced metric G(q) = I + J_FK^T J_FK 사용. Riemannian DSM with G-norm.
- Trajectory generation이 M에 머무는지
- 2a와 2b의 distribution recovery 차이 비교

### Phase 3: Toy 3 - residual FK + compliance (3-4주)

2-link arm with simulated compliance:
- True: p = FK(q) + Δ_true(q, z_e)
- Learned: p = FK(q) + Δ_φ(q, z_e)
- z_e (tool length) 변화 하 self-model 학습
- 학습된 manifold M_φ(z_e)가 ground truth manifold를 capture하는지
- Embodiment perturbation 하 generalization

### Phase 3.5: 3-link planar redundant arm (2-3주, optional)

Multi-modal capture 검증:
- 3-link arm with redundancy (n_q = 3, n_p = 2)
- 같은 end-effector target에 multi-modal solutions
- Riemannian SGM이 multi-modal distribution capture하는지
- 7-DoF로 가기 전 bridge

### Phase 4: 7-DoF + killer experiment (8-10주)

Franka simulation, multi-modal tool-tip reaching with tool length variation.

### Phase 5: Paper writing

---

## 15. Killer experiment 사양

### 15.1 Setting

**Robot**: 7-DOF Franka simulation

**Task**: Multi-modal tool-tip reaching trajectory generation
- Robot이 tool grip, 3D target에 tool-tip 도달
- Redundant kinematics → 같은 target에 multi-modal trajectory
- Demo가 multi-modal solution 보여줌

**Embodiment perturbation**:
- Self-exploration: \(\ell \in \{8, 10, 12, 15, 18\}\)cm
- Demo: \(\ell = 10\)cm
- Test: \(\ell \in [8, 20]\)cm random

**\(z_e = \ell\)** (scalar)

### 15.2 Baselines (7개)

1. **Behavior Cloning** (no self-model)
2. **Vanilla Diffusion Policy**
3. **Diffusion Policy + \(z_e\) conditioning**
4. **Diffusion + ambient projection** (learned g_φ로 post-hoc projection, Projected Diffusion 스타일)
5. **Action-level tangent imitation** (BC version of v6 framework, 본 연구의 trajectory-level과 비교)
6. **Oracle analytic FK + \(z_e\)** (residual term Δ_φ = 0, analytic FK만)
7. **Ours: Riemannian SGM on learned self-model manifold**

### 15.3 Baseline 비교 의미

- vs 1, 2: self-model + Riemannian SGM 가치 전체
- vs 3: z_e conditioning 외 manifold framework 추가 가치
- vs 4: manifold-intrinsic vs ambient + projection
- vs 5: trajectory-level Riemannian SGM vs action-level tangent BC
- vs 6: residual learning이 capture하는 효과 (compliance, calibration)

### 15.4 Metrics

**Task performance**:
- Success rate (target 도달)
- Tool-tip position error

**Manifold adherence**:
- ||g_φ(x_t)|| over generated trajectory
- Cumulative manifold drift

**Multi-modal capture**:
- Diversity score across modes
- Mode coverage

**Robustness**:
- vs tool perturbation (in-distribution / out-of-distribution z_e)
- vs demo count
- vs trajectory length

### 15.5 Expected outcomes

**Primary**:
- Ours가 baseline 4 (ambient projection) 대비 manifold adherence 우월 (cumulative projection 오차 없음)
- Ours가 baseline 6 (oracle analytic) 대비 residual learning이 capture하는 compliance 효과 우월
- Multi-modal trajectory generation이 baseline 5 (action-level)보다 자연스러움

**Secondary**:
- Embodiment perturbation 하 robustness
- Out-of-distribution z_e에서의 graceful degradation

**Failure mode (정직)**:
- Standard regime (no perturbation, simple task)에서는 baseline 3과 비슷
- 명시적 언급

---

## 16. Ablation study

1. **Manifold-intrinsic vs ambient + projection**: 같은 self-model, RSGM (ours) vs Projected Diffusion (baseline 4). 본 연구의 핵심 ablation.
2. **Trajectory-level vs action-level diffusion**: trajectory diffusion (ours) vs action-level qdot diffusion (v6-style)
3. **Analytic-only vs residual self-model**: Δ_φ = 0 vs Δ_φ 학습. Residual의 가치
4. **Implicit g_φ vs derivative regression (NJF-style)**: paradigm B vs A
5. **With/without z_e conditioning**: embodiment context의 역할
6. **Forward process choice**: Brownian (uniform target) vs Langevin (wrapped Gaussian target)

---

## 17. Reviewer 공격 대응

### 공격 1: "Vanilla diffusion + z_e conditioning으로 충분?"

답:
- Vanilla는 R^d Gaussian noise space에서 작동. 본 연구는 학습된 manifold 위 Riemannian noise space.
- Manifold adherence가 stochastic learning에 의존 vs by construction 차이.
- Embodiment 변화 시 demo distribution이 manifold 정보를 충분히 capture 못함, 본 연구의 manifold framework은 demo와 무관하게 작동.
- Killer experiment baseline 3 (z_e conditioning)와 정량 비교.

### 공격 2: "Riemannian SGM (De Bortoli 2022)과 차이?"

답: RSGM은 known manifold (sphere, SO(3), torus)에서 density modeling. 본 연구의 차이:
- Manifold가 self-model로 학습됨 (vs known)
- Robot imitation learning context (vs density modeling)
- Embodiment context z_e conditioning (vs fixed manifold)
- Goal-conditional generation (vs unconditional)
- Multi-modal demonstration (vs marginal density)

본 연구는 RSGM의 framework을 robotics의 learned-manifold + embodiment-aware + goal-conditional context로 확장.

### 공격 3: "Projected Diffusion과 차이?"

답: Projected Diffusion은 ambient에서 생성 후 projection. 본 연구는 manifold-intrinsic.
- Score field가 ambient \(\nabla \log p_t \in \mathbb{R}^d\) vs manifold \(\nabla_M \log p_t \in T_x M\)
- Projection이 post-hoc vs by construction
- 누적 projection 오차 vs 매 step manifold에 머무름
- Killer experiment baseline 4와 정량 비교 (ablation 1)

### 공격 4: "ATACOM은 inequality도 다루고 비슷한 framework"

답: ATACOM과 본 연구는 conceptual core를 일부 공유 (manifold-intrinsic action generation). 차별점:
- Learned (residual) vs known (analytic) manifold
- Imitation learning vs RL
- Multi-step trajectory diffusion vs single-step Gaussian policy
- z_e embodiment context (ATACOM에 없음)
- Multi-modal capture가 자연스러움

### 공격 5: "왜 analytic FK를 쓰지 않나"

답: Killer experiment baseline 6 (oracle analytic + z_e)이 직접 검증. Compliance, calibration drift 같은 analytic-uncapturable effect가 residual로 학습됨. Toy 3 (compliance perturbation)에서 정량 검증.

### 공격 6: "Demo가 manifold에 정확히 있다는 가정은?"

답: Real demo는 measurement noise로 manifold에서 약간 벗어남.
- 학습 전 demo를 manifold로 projection 한 번 적용 (graph manifold form에서 closed-form)
- Projection 후 demo가 manifold 위에 있다고 가정
- 이건 standard preprocessing이며 Riemannian SGM literature에서 공통.

### 공격 7: "Goal-conditional이 motion planning?"

답: 정확한 표현은 "motion-planning-like behavior emergent in demo distribution". Demo가 task-aware motion 정의하므로 학습된 conditional generation이 task-aware하게 행동. Explicit motion planning (collision avoidance, optimality) 아님. Avoid over-claim.

### 공격 8: "이론적 distribution recovery 보장?"

답: RSGM의 chart-local convergence 결과 활용 (RSGM 논문 §4 Theorem 4). Bounded workspace에서 wrapped Gaussian target으로 작동. 일반 case의 이론적 분석은 future work.

### 공격 9: "Inequality (joint limit) 처리는?"

답: Equality (kinematic consistency)만 다룬다는 것 명시. Joint limit, manipulability는 future work. Demo가 자연스럽게 limit/singularity 회피하면 학습된 distribution이 그것을 reflect. Inequality boundary 근처 score behavior 분석은 별도 연구.

### 공격 10: "Gupta et al. 2026과 차이?"

답: Gupta는 cross-robot transfer가 목적, 본 연구는 single robot에서 diffusion feasibility가 목적. 다른 문제, complementary work. Section 9.6 참조.

### 공격 11: "Chart-based approximation은 진짜 Riemannian SGM 아니다"

답: 인정. 첫 toy implementation은 chart-based approximation (q-space Euclidean noise/score, G ≈ I 가정). Full Riemannian (induced metric G with chart Itô-correct SDE) 은 framework의 main이며 method extension. §3.3, §4.5에서 둘을 명확히 구분 명시. Toy 2의 sub-phase 2a (chart) vs 2b (full Riemannian) 비교로 정량 검증.

### 공격 12: "Norm 표기 ambient vs chart의 등가성?"

답: §3.3에 명시. Tangent vector는 ambient form \(u \in T_x M \subset \mathbb{R}^d\) 또는 chart form \(a \in \mathbb{R}^{n_q}\) (with \(u = J_H a\))로 표기 가능. Norm 등가:
\[
\|u\|^2 = u^T u = a^T G(q) a = \|a\|^2_G
\]

Implementation에서 어느 form을 쓰느냐만 일관되면 됨. §4.5에 score matching loss의 ambient/chart-G/chart-Eucl 세 form을 구분 명시.

---

## 18. 최종 정리

### Title

**Riemannian Score-Based Imitation Learning on Learned Robot Self-Model Manifolds: Trajectory Generation with Embodiment Context**

또는 짧게:

**Self-Model Manifold Diffusion**

### Core claim

> Vanilla diffusion policy는 R^d Gaussian noise space에서 데이터로 복원하는 vector field를 학습한다. 본 연구는 학습된 self-model manifold 위 Riemannian Brownian motion에서 데이터로 복원하는 tangent vector field를 학습한다. Self-model이 ambient R^d 안의 differentiable embedded manifold를 정의하고, ambient metric이 자동으로 induce하는 Riemannian geometry 위에서 score-based generative modeling이 진행된다. 미분 구조 (tangent bundle)가 feasible velocity bundle이며, 모든 diffusion step (forward, reverse, score)이 manifold-intrinsic. Sampling은 by construction manifold에 머물고, embodiment context z_e가 manifold의 deformation parameter로 통합된다. Riemannian SGM (De Bortoli 2022)을 robot self-model context로 가져온 framework.

### Abstract draft

> Diffusion policies for robot manipulation rely on R^d Gaussian noise as their reference distribution and learn to reverse a Euclidean diffusion process. This treats physical feasibility as a property to be captured through stochastic learning of the data distribution's support, providing no hard guarantees and requiring extensive demonstration data—particularly under embodiment uncertainty such as tool variation or calibration drift. We propose to instead learn the robot's self-model as a differentiable embedded manifold M ⊂ R^d, on which the ambient Euclidean metric induces a Riemannian geometry. Demonstration trajectories are then treated as samples from a distribution supported on this learned manifold, and we train a Riemannian score-based generative model whose score field lies in the manifold's tangent bundle. By construction, all stages of the diffusion process—forward noising, reverse generation, score field, sampling via retraction-based geodesic random walk—operate intrinsically on M. Embodiment context z_e parameterizes the manifold's deformation, enabling adaptation to deployment-time variation. We distinguish our framework from Riemannian SGM (which assumes known manifolds), Projected Diffusion (ambient generation with post-hoc projection), ATACOM (RL on known constraint manifolds), and Neural Jacobian Fields (direct Jacobian regression). Experiments on multi-modal tool-tip reaching with tool length perturbation demonstrate manifold adherence, embodiment robustness, and multi-modal trajectory capture compared to vanilla diffusion, ambient projection diffusion, action-level tangent imitation, and oracle analytic baselines.

### Boxed core idea

\[
\boxed{
\begin{aligned}
&\text{Self-model } g_\phi(x, z_e) = 0 \\
&\quad \xrightarrow{\text{ambient embedding}} \mathcal{M}_\phi(z_e) \subset \mathbb{R}^d \\
&\quad\quad \xrightarrow{\text{induced metric } G(q, z_e) = J_H^T J_H} (\mathcal{M}_\phi(z_e), G) \text{ Riemannian} \\
&\qquad \xrightarrow{\text{tangent bundle}} TM = \text{feasible velocity bundle} \\
&\qquad\quad \xrightarrow{\text{product manifold}} \mathcal{M}_\phi^{H+1} \text{ for trajectories} \\
&\quad\quad\quad \xrightarrow{\text{Riemannian SGM}} \tau \sim p_\theta(\tau | g, z_e), \quad \tau \in \mathcal{M}_\phi(z_e)^{H+1}
\end{aligned}
}
\]

---

## 19. 다음 단계

1. Toy 1 (S^1) 실험 시작: Riemannian SGM framework sanity check
2. Toy 2 (2-link arm graph manifold) 실험
3. Toy 3 (residual FK + compliance) 실험
4. 7-DoF killer experiment

---