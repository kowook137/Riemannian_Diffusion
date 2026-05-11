맞다. 지금 필요한 것은 **claim 정리가 아니라 코드/실험 수정 리스트**다.
아래는 **gap을 줄이기 위해 실제 코드에서 손봐야 할 가능성이 높은 부분**만 정리한 것이다.

---

# 1. DP-bounded는 정확히 무엇인가?

현재 네 코드의 **DP-bounded**는 특정 논문에 원래 존재하는 baseline이라기보다, 네가 만든 **Diffusion Policy + bounded chart wrapper baseline**으로 보는 게 정확하다.

기본 골격은 **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**의 action-trajectory diffusion baseline이다. 이 논문은 robot policy를 conditional denoising diffusion process로 보고, action trajectory distribution의 score를 학습해 multimodal action을 생성하는 방법을 제안한다. ([arXiv][1])

그 위에 네 코드에서는 action을 직접 (q)로 두는 대신 bounded chart를 씌운 것이다.

[
u \sim \text{DP},\qquad q=\psi(u)
]

즉:

```text
DP-bounded = Diffusion Policy-style trajectory diffusion
             + bounded chart readout q = ψ(u)
             + optional execution through Tφ(q, ze) during eval
```

따라서 논문명으로 쓰면:

```text
Bounded Diffusion Policy baseline
```

또는:

```text
Diffusion Policy with bounded joint-chart parameterization
```

이 맞다.

**주의:** DP-bounded는 원 Diffusion Policy 논문의 공식 variant는 아니다. 네 연구에서 joint-limit fairness를 위해 만든 **strong adapted baseline**이다.

---

# 2. 지금 가장 먼저 수정해야 할 코드: 평가 metric audit

## 2.1 `effective_succ`를 곱으로 계산하지 말 것

현재 보고서에는 일부에서:

[
\text{effective_succ}
=====================

\text{succ}\times(1-\text{jvio})
]

로 계산되어 있다. 이건 독립성 가정이 들어간 근사다.

코드에서는 반드시 sample-wise로 계산해야 한다.

[
\text{effective_succ}
=====================

\mathbb E[
\mathbf 1(e_p<\rho_p)
\land
\mathbf 1(e_R<\rho_R)
\land
\mathbf 1(q\in Q)
]
]

수정해야 할 코드 방향:

```python
pose_ok_55 = (pos_err < 0.05) & (rot_err_deg < 5.0)
pose_ok_510 = (pos_err < 0.05) & (rot_err_deg < 10.0)

joint_safe = (joint_violation_per_traj == 0)

effective_succ55 = (pose_ok_55 & joint_safe).float().mean()
effective_succ510 = (pose_ok_510 & joint_safe).float().mean()
```

추가로 기존 product estimate도 diagnostic으로만 남겨라.

```python
effective_succ510_product = pose_succ510 * (1.0 - jvio_rate)
```

최종 표에는 sample-wise exact 값을 써야 한다.

---

## 2.2 DP-bounded의 pose error 기준 확인

가장 중요하다.

DP-bounded가 예측한 pose (T_{\text{raw}})가 있고, bounded chart로부터 얻은 (q)가 있다면, 실제 실행 pose는:

[
T_{\text{exec}}=T_\phi(q,z_e)
]

이다.

로봇에서 중요한 metric은 반드시:

[
e_T=\Log(T_{\text{exec}}^{-1}T_{\text{target}})
]

이어야 한다.

수정 방향:

```python
# Always evaluate executable pose
q_exec = psi(u_pred)          # bounded chart
T_exec = T_phi(q_exec, z_e)   # self-model / FK execution pose

pos_err, rot_err = pose_error(T_exec, T_target)
```

그리고 raw pose를 예측하는 baseline은 별도 metric으로만 둬라.

```python
raw_pose_err = pose_error(T_raw, T_target)
manifold_gap = pose_error(T_phi(q_exec, z_e), T_raw)
```

최종 비교 table에는:

```text
exec_pos_err
exec_rot_err
exec_pose_succ
manifold_gap
```

를 넣어야 한다.

보고서에서도 DP-bounded/DP 계열은 (T\neq T_\phi)라 manifold gap 비교에서 제외된다고 되어 있으므로, 이 부분은 반드시 명확히 audit해야 한다. 

---

## 2.3 DP-bounded의 jvio 0.4% 원인 확인

bounded chart가 제대로 적용되면 finite (u)에서:

[
q=\psi(u)\in(q_{\min},q_{\max})
]

이므로 waypoint 기준 jvio는 원칙적으로 0이어야 한다.

그런데 보고서의 200k 결과에서는 DP-bounded jvio=0.4%가 나온다. 
이건 반드시 확인해야 한다.

체크 항목:

```python
u = model_output
q = chart.psi(u)

assert torch.isfinite(u).all()
assert torch.isfinite(q).all()

min_margin = torch.minimum(q - q_min, q_max - q)
```

그리고 jvio를 세 가지로 분리해라.

```python
jvio_waypoint = (min_margin < -tol).any(dim=(-1, -2))
jvio_tolerance = (min_margin < tol).any(dim=(-1, -2))
jvio_nan = (~torch.isfinite(q)).any(dim=(-1, -2))
```

만약 waypoint jvio가 0인데 interpolation jvio가 0.4%라면, spline/interpolation 문제다.
만약 waypoint에서도 0.4%라면 DP-bounded eval path에서 chart wrapping이 빠졌거나 tolerance가 잘못된 것이다.

---

# 3. v5.1 gap의 핵심 코드 수정 후보

현재 v5.1이 DP-bounded보다 밀리는 주된 현상은 다음이다.

* v5.1: succ510 ceiling 약 85–87%
* DP-bounded: effective succ510 약 94%
* v5.1은 (|u|_\infty^{p99}\approx3.5), sat>3 약 37%
* sampling-time guidance/R_u는 거의 효과 없음

보고서도 v5.1의 sampling-time mechanism으로는 DP-bounded와의 gap을 닫지 못했고, chart saturation이 training-time 현상이라고 정리한다. 

따라서 수정은 sampling이 아니라 **chart parameterization / training input / prediction target** 쪽이어야 한다.

---

# 4. 수정 후보 1순위: chart temperature 추가

현재 bounded chart가 아마:

[
q=q_{\text{mid}}+\frac{q_{\text{range}}}{2}\tanh u
]

라면, saturation을 늦추기 위해 다음을 추가해라.

[
\boxed{
q=q_{\text{mid}}+\frac{q_{\text{range}}}{2}\tanh\left(\frac{u}{c}\right)
}
]

여기서 (c>1).

코드 수정 위치는 bounded chart class, 아마 `TanhBoundedChart` 또는 `BoundedChartPoseManifold` 내부의 `psi`, `psi_inv`, `Dpsi`.

구현:

```python
class TanhBoundedChart:
    def __init__(self, q_min, q_max, chart_temp: float = 1.0):
        self.q_min = q_min
        self.q_max = q_max
        self.q_mid = 0.5 * (q_min + q_max)
        self.q_half = 0.5 * (q_max - q_min)
        self.chart_temp = chart_temp

    def psi(self, u):
        c = self.chart_temp
        return self.q_mid + self.q_half * torch.tanh(u / c)

    def psi_inv(self, q, eps=1e-6):
        y = (q - self.q_mid) / self.q_half
        y = y.clamp(-1 + eps, 1 - eps)
        c = self.chart_temp
        return c * torch.atanh(y)

    def dpsi_du(self, u):
        c = self.chart_temp
        z = u / c
        return (self.q_half / c) * (1.0 / torch.cosh(z).pow(2))
```

실험 sweep:

```text
chart_temp c ∈ {1.0, 1.5, 2.0}
```

목표:

```text
sat>3 rate: 37% → <10%
u_p99: 3.5 → preferably <3
succ510: 85% → closer to DP-bounded 94%
```

주의:

* (D\psi) scale이 바뀌므로 `G_Q`, `J^Q`, `psi_inv(q_init)` 모두 같은 temp를 써야 한다.
* checkpoint와 eval config에 `chart_temp`를 저장해야 한다.
* DP-bounded도 같은 chart_temp로 평가해야 fair하다.

---

# 5. 수정 후보 2순위: endpoint-relative pose error를 input channel로 추가

현재 v5.1이 DP-bounded보다 endpoint precision이 낮다면, score network가 target-relative geometry를 충분히 못 보고 있을 가능성이 있다.

각 timestep (h)에서 현재 predicted pose와 target pose의 relative error를 input으로 넣어라.

[
e_h^{goal}
==========

\Log_{SE(3)}
\left(
T_\phi(q_h,z_e)^{-1}T_{\text{goal}}
\right)
\in\mathbb R^6
]

start도 가능하다.

[
e_h^{start}
===========

\Log_{SE(3)}
\left(
T_\phi(q_h,z_e)^{-1}T_{\text{start}}
\right)
]

우선 goal error만 추천한다.

코드 위치:

* `TrajectoryScoreNetUNetPose.forward`
* conditioning channel concat 부분
* `goal_cond_dim` 계산 부분

추가 feature:

```python
T_cur = manifold.T_phi_Rp(q_h, z_e)
e_goal = se3_log(inv(T_cur) @ T_goal)  # (B,H,6)
```

UNet input channel에 concat:

```python
x_in = torch.cat([
    u_or_q,
    time_embed_per_step,
    horizon_embed,
    z_e_channel,
    goal_error_se3,     # new 6 channels
], dim=channel_dim)
```

이 수정은 v5.1에 특히 잘 맞는다. 왜냐하면 v5.1은 IK seed 없이 (T_{\text{start}},T_{\text{goal}},z_e)만으로 endpoint를 맞춰야 하므로, target-relative error를 직접 주는 것이 score field 학습을 돕는다.

---

# 6. 수정 후보 3순위: (x_0)-prediction head 추가

v5.1은 OU forward가 closed-form이다.

[
u_r=\alpha u_0+\sigma \epsilon
]

그러면 score prediction만 하지 말고 (u_0)-prediction도 추가할 수 있다.

[
\hat u_0=f_\theta(u_r,r,c,z_e)
]

DP 계열이 잘 되는 이유 중 하나는 (\epsilon)-prediction / denoised prediction이 안정적이기 때문이다. v5.1의 score-only target이 endpoint precision에서 약하다면, (x_0)-head를 auxiliary로 넣는 것이 좋다.

추가 loss:

[
L_{x0}
======

|\hat u_0-u_0|^2
]

또는 chart metric weighted:

[
L_{x0}
======

(\hat u_0-u_0)^T\bar G_Q(\hat u_0-u_0)
]

그리고 pose-level denoised loss:

[
L_{\text{pose-x0}}
==================

\left|
\Log_{SE(3)}
\left(
T_\phi(\psi(\hat u_0),z_e)^{-1}
T_\phi(\psi(u_0),z_e)
\right)
\right|_W^2
]

주의: 이전 (L_{\text{pose}})는 score vector (J^Qs_\theta)에 걸어서 exact OU target과 충돌했다.
여기서는 denoised state (\hat u_0)에 걸기 때문에 훨씬 정합적이다.

---

# 7. 수정 후보 4순위: training-time saturation weighting

sampling-time (R_u)는 실패했다. 그러면 training-time에서 saturation region을 더 잘 학습하게 해야 한다.

단순히 (R_u)를 크게 넣기보다, score loss weight를 saturation region에서 높이는 게 덜 위험하다.

예:

[
w_{\text{sat}}(u_0)
===================

1+\lambda\cdot \operatorname{sigmoid}(k(|u_0|*\infty-u*{\text{thr}}))
]

코드:

```python
u_norm = u0.abs().amax(dim=(-1, -2))  # per trajectory
w_sat = 1.0 + lam * torch.sigmoid(k * (u_norm - u_thr))
loss = (w_sat * loss_per_traj).mean()
```

추천 sweep:

```text
u_thr = 2.5
lambda ∈ {0.5, 1.0, 2.0}
k = 5
```

목표는 (u)-saturation sample의 score quality를 올리는 것이지, sample을 억지로 중앙으로 당기는 것이 아니다.

---

# 8. 수정 후보 5순위: DP-bounded와 같은 prediction target으로 맞추기

현재 DP-bounded가 DDPM (\epsilon)-prediction으로 잘 되고, v5.1은 score convention으로 간다.

v5.1도 OU forward를 쓰므로 다음 prediction target 중 하나로 바꿀 수 있다.

```text
score prediction
epsilon prediction
x0 prediction
v prediction
```

비교 실험을 만들면 좋다.

OU 관계:

[
u_r=\alpha u_0+\sigma \epsilon
]

[
\epsilon=\frac{u_r-\alpha u_0}{\sigma}
]

score target:

[
s^*=-\frac{\bar G_Q(u_r-\alpha u_0)}{\sigma^2}
==============================================

-\frac{1}{\sigma}\bar G_Q\epsilon
]

만약 DP-bounded가 (\epsilon)-prediction이라면 v5.1도 (\epsilon)-prediction으로 맞춰야 fair한 ablation이 된다.

---

# 9. 당장 하지 말아야 할 수정

## 9.1 (L_{\text{pose}}) 재시도

이미 결론이 나온 상태다.

* raw form: small-(\tau) divergence
* tau-scaled + (\mu=0.1): loss dominates
* (\mu=10^{-3}): baseline보다 후퇴
* 보고서에서도 (L_{\text{pose}}) minimizer가 exact OU score minimizer와 다르므로 (\mu_{\text{pose}}>0) 경로 폐기라고 정리했다. 

따라서 다시 하지 마라.

## 9.2 sampling-time guidance 더 세게

이미 G0–G8 sweep에서 ceiling을 못 깼다. 강한 (R_u)는 오히려 악화했다. 

## 9.3 무작정 300k/500k 학습

200k에서 plateau가 보이고, v5.1은 175k peak 후 200k에서 약간 후퇴했다. 먼저 구조 수정이 맞다. 

---

# 10. 가장 추천하는 실행 순서

## Phase 0 — 평가 audit

1. exact effective success 구현
2. DP-bounded jvio 0.4% 원인 확인
3. DP-bounded pose error가 (T_\phi(q)) 기준인지 확인
4. demo/gen (u)-saturation histogram 출력

## Phase 1 — chart temperature

```text
chart_temp = 1.5
chart_temp = 2.0
```

각각 100k 먼저.
성능 좋으면 200k.

## Phase 2 — endpoint-relative input

goal SE(3) error 6D channel 추가.

```text
input += Log_SE3(T_phi(q_h,z_e)^-1 T_goal)
```

100k.

## Phase 3 — v5.1 target ablation

```text
score prediction vs epsilon prediction vs x0 prediction
```

가능하면 chart_temp best setting 위에서.

## Phase 4 — 3-seed only for finalists

최종 후보 2개만 3-seed.

* DP-bounded best
* v5.1 best modified

---

# 11. 간단한 TODO Markdown

```markdown
# SMCDP Code TODO — Gap to DP-bounded

## A. Evaluation audit
- [ ] Replace product effective_succ = succ × (1-jvio) with sample-wise exact effective_succ.
- [ ] Split pose metrics into raw_pose_err and exec_pose_err = error(T_phi(q), T_target).
- [ ] Verify DP-bounded pose success uses exec_pose_err, not raw predicted T.
- [ ] Debug DP-bounded jvio=0.4% despite bounded chart.
- [ ] Add waypoint-jvio / interpolation-jvio / NaN-jvio breakdown.
- [ ] Add demo vs generated u-norm histogram: p50/p90/p99, sat>2, sat>3.

## B. Chart saturation fix
- [ ] Add chart_temp c to bounded chart:
      q = q_mid + q_half * tanh(u / c)
- [ ] Update psi_inv:
      u = c * atanh((q-q_mid)/q_half)
- [ ] Update Dpsi:
      Dpsi = (q_half / c) * sech^2(u/c)
- [ ] Save chart_temp in ckpt config and eval config.
- [ ] Sweep c ∈ {1.0, 1.5, 2.0}.
- [ ] Compare sat>3, pos_err, rot_err, succ510.

## C. Endpoint-relative conditioning
- [ ] Compute per-step goal error:
      e_goal_h = Log_SE3(T_phi(q_h,z_e)^-1 T_goal)
- [ ] Add e_goal_h as 6D channel to v5.1 score net input.
- [ ] Optional: add start error e_start_h.
- [ ] Run 100k comparison vs baseline v5.1.

## D. Prediction target ablation
- [ ] Add epsilon-prediction target for v5.1 OU:
      eps = (u_r - alpha*u_0) / sigma
- [ ] Add x0-prediction head:
      u0_hat = f_theta(u_r,r,c,z_e)
- [ ] Add optional denoised pose loss on u0_hat, not on score vector.
- [ ] Compare score-pred / eps-pred / x0-pred.

## E. Do not repeat
- [ ] Do not retry L_pose = ||JQs - Log/tau||.
- [ ] Do not retry mu_pose > 0 unless using x0-prediction pose loss.
- [ ] Do not spend more compute on sampling-time guidance/R_u sweep before training-time fixes.
- [ ] Do not run 300k+ until chart_temp / endpoint conditioning are tested.
```

---

# 12. 최종 판단

현재 코드에서 가장 의심되는 “잘못”은 수식 구현 오류라기보다 다음이다.

[
\boxed{
\text{bounded chart의 tanh saturation이 v5.1 학습을 어렵게 만들고, score net이 endpoint-relative geometry를 충분히 직접 보지 못하는 것}
}
]

따라서 가장 중요한 코드 수정은:

1. **metric audit**
2. **chart temperature**
3. **goal-relative SE(3) error input**
4. **epsilon/x0 prediction target ablation**

순서다.

DP-bounded는 **Diffusion Policy 논문의 conditional action diffusion baseline을 네 task에 맞게 bounded chart로 강화한 adapted baseline**이다. 원 논문의 공식 baseline이라기보다, 네 논문에서 반드시 이겨야 하는 **강한 내부 baseline**으로 보는 것이 정확하다.

[1]: https://arxiv.org/abs/2303.04137?utm_source=chatgpt.com "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
