````markdown
# SMCDP v5.1 Ours Recipe Search Plan

## 0. 현재 상황 요약

현재 v5.1 baseline은 다음 조건에서 평가되었다.

```text
v5.1 50k baseline
- chart-OU
- IK-free
- guidance off
- μ_pose = 0
- β_f = 20
- Ḡ_Q = I
````

주요 결과:

| Metric             | v4.1 50k + IK seed | v5.1 50k IK-free | 해석                           |
| ------------------ | -----------------: | ---------------: | ---------------------------- |
| pos err 평균         |            4.77 cm |          5.73 cm | +0.96 cm worse               |
| rot err 평균         |              6.46° |            8.15° | +1.69° worse                 |
| succ@(5cm,5°)      |             64.84% |            17.6% | −47.3 pp                     |
| succ@(5cm,10°)     |             71.88% |            47.3% | −24.6 pp                     |
| joint violation    |                 0% |               0% | bounded chart 성공             |
| mode capture error |               0.00 |             0.00 | score net 단독으로 bimodality 보존 |
| manifold gap       |                  0 |                0 | graph retraction 정상          |
| ‖u‖∞ p99           |                n/a |             3.57 | saturation regime, concern   |

핵심 해석:

> v5.1은 IK-free 구조, joint feasibility, learned-manifold adherence, bimodality 보존에는 성공했다.
> 그러나 endpoint pose refinement가 부족하고, chart saturation이 발생한다.

즉, 지금 목적은 baseline 비교가 아니라 **ours recipe 탐색**이다.

---

## 1. 현재 문제 정의

v5.1 baseline의 주요 문제는 두 가지다.

### Problem 1 — Endpoint refinement 부족

strict success가 크게 낮다.

```text
succ@(5cm,5°): 17.6%
succ@(5cm,10°): 47.3%
```

mode capture는 유지되므로, 문제는 branch selection 실패라기보다 다음에 가깝다.

> mode는 맞게 선택하지만, mode 내부에서 endpoint pose precision이 부족하다.

---

### Problem 2 — Chart saturation

현재

[
|u|_\infty^{p99}=3.57
]

이다.

v5.1 spec에서 healthy regime은 대략

[
|u|_\infty \lesssim 2
]

이고, saturation concern은

[
|u|_\infty > 3
]

이다. bounded chart에서 (|u|>3)이면 (D_\psi(u))가 near-singular가 되고, (J_Q=J_{\text{pose}}D_\psi)도 약해진다. v5.1 문서도 saturation regime을 주의해야 한다고 명시한다. 

따라서 ours recipe는 endpoint success만 올리는 것이 아니라, saturation도 같이 제어해야 한다.

---

## 2. 우선순위 결론

지금 바로 할 일은 **50k 재학습이 아니다.**

우선순위는 다음이다.

```text
1. current v5.1 50k checkpoint 고정
2. sampling-time start/goal guidance 활성화
3. chart-norm penalty 추가
4. guidance recipe sweep
5. 그 결과가 부족할 때 μ_pose > 0 재학습
```

이유:

* start/goal guidance는 sampling-time mechanism이다.
* 현재 checkpoint로 바로 sweep 가능하다.
* endpoint refinement 부족을 가장 직접적으로 검증할 수 있다.
* chart saturation도 sampling reward로 먼저 완화할 수 있다.
* 재학습 전에 문제 원인을 분리할 수 있다.

---

## 3. Step 1 — Current 50k checkpoint로 guidance sweep

### 3.1 Reward 구성

Sampling에서 다음 reward를 사용한다.

[
R_{\text{total}}
================

\alpha_s R_{\text{start}}
+
\alpha_g R_{\text{goal}}
+
\alpha_u R_u
+
\alpha_v R_{\text{vel}}
+
\alpha_a R_{\text{acc}}
]

우선 recipe search에서는 핵심적으로 아래 세 항만 본다.

[
R_{\text{start}},\quad R_{\text{goal}},\quad R_u
]

---

### 3.2 Start reward

[
e_{\text{start}}(u_0,z_e)
=========================

\Log_{\mathrm{SE}(3)}
\left(
\widetilde T_\phi(u_0,z_e)^{-1}
T_{\text{start}}
\right)
]

[
R_{\text{start}}
================

*

\left(
\alpha_p^s |e_\rho^s|^2
+
\alpha_R^s |e_\omega^s|^2
\right)
]

---

### 3.3 Goal reward

[
e_{\text{goal}}(u_H,z_e)
========================

\Log_{\mathrm{SE}(3)}
\left(
\widetilde T_\phi(u_H,z_e)^{-1}
T_{\text{target}}
\right)
]

[
R_{\text{goal}}
===============

*

\left(
\alpha_p^g |e_\rho^g|^2
+
\alpha_R^g |e_\omega^g|^2
\right)
]

---

### 3.4 Chart-norm penalty

현재 chart saturation이 있으므로 (R_u)를 반드시 같이 넣는다.

[
R_u
===

*

\sum_{h=0}^{H}
|u_h|^2
]

목적:

[
|u|_\infty^{p99}
]

를 3.0 아래로 낮추는 것.

---

## 4. First sweep grid

우선 (\alpha_s=2\alpha_g)로 고정한다.

v5.1 spec에서도 start anchor를 goal보다 강하게 두는 것을 권장한다. 즉,

[
\alpha_s \ge 2\alpha_g
]

를 기본 recipe로 둔다. 

### First sweep

| Config | (\alpha_g) | (\alpha_s) | (\alpha_u) | 목적                        |
| ------ | ---------: | ---------: | ---------: | ------------------------- |
| G0     |          0 |          0 |          0 | current baseline          |
| G1     |        0.5 |        1.0 |          0 | mild guidance             |
| G2     |        1.0 |        2.0 |          0 | spec default              |
| G3     |        2.0 |        4.0 |          0 | strong guidance           |
| G4     |        0.5 |        1.0 |  (10^{-3}) | mild + anti-saturation    |
| G5     |        1.0 |        2.0 |  (10^{-3}) | default + anti-saturation |
| G6     |        2.0 |        4.0 |  (10^{-3}) | strong + anti-saturation  |
| G7     |        1.0 |        2.0 |  (10^{-2}) | stronger anti-saturation  |
| G8     |        2.0 |        4.0 |  (10^{-2}) | aggressive                |

---

## 5. Selection criteria

Recipe 선택 시 최고 success만 보면 안 된다. saturation을 같이 봐야 한다.

### Primary metrics

| Metric             |                     목표 |
| ------------------ | ---------------------: |
| succ@(5cm,5°)      |         17.6% → 35% 이상 |
| succ@(5cm,10°)     |         47.3% → 60% 이상 |
| pos err            |             5.73 cm 이하 |
| rot err            |               8.15° 이하 |
| mode capture error |                   0 유지 |
| joint violation    |                   0 유지 |
| manifold gap       |                   0 유지 |
| (|u|_\infty^{p99}) | 3.0 이하, ideally 2.5 이하 |

### Selection rule

[
\boxed{
\text{success가 약간 낮더라도 } |u|_\infty^{p99}<3.0 \text{인 config를 우선한다.}
}
]

이유:

* (|u|_\infty^{p99}>3)이면 bounded chart saturation regime이다.
* (D_\psi(u))가 작아지고, pose gradient가 약해질 수 있다.
* 후속 (\mu_{\text{pose}}) 재학습을 해도 saturation이 심하면 endpoint refinement가 어려울 수 있다.

---

## 6. Step 2 — Local ratio sweep

First sweep에서 best config가 나오면, 그 주변에서만 (\alpha_s/\alpha_g) ratio를 본다.

예를 들어 G5가 best라면:

[
\alpha_g=1.0,\quad \alpha_u=10^{-3}
]

를 고정하고,

[
\alpha_s/\alpha_g\in{1,2,4}
]

만 본다.

| Config | (\alpha_g) | (\alpha_s) | (\alpha_u) |
| ------ | ---------: | ---------: | ---------: |
| R1     |        1.0 |        1.0 |  (10^{-3}) |
| R2     |        1.0 |        2.0 |  (10^{-3}) |
| R3     |        1.0 |        4.0 |  (10^{-3}) |

목적:

* start anchor가 너무 약한지 확인
* start anchor가 너무 강해서 trajectory를 망치는지 확인
* endpoint success와 smoothness 사이의 trade-off 확인

---

## 7. Step 3 — Guidance로 부족하면 (\mu_{\text{pose}}) 재학습

Guidance sweep 결과가 다음 중 하나라면 재학습으로 넘어간다.

```text
- succ@(5cm,5°)가 30% 아래
- rot err가 거의 줄지 않음
- guidance를 키우면 saturation만 악화됨
- mode는 맞는데 endpoint만 계속 부정확함
```

그때는 training-time pose consistency signal이 부족한 것이다.

---

## 8. μ_pose 재학습 recipe

첫 재학습은 하나만 한다.

[
\boxed{
\beta_f=20,\quad
\bar G_Q=I,\quad
\mu_{\text{pose}}=0.1,\quad
\tau_{\text{cutoff}}=0.5,\quad
50k
}
]

sampling은 Step 1/2에서 찾은 best guidance setting을 그대로 사용한다.

---

### 8.1 Pose regularizer

v5.1에서 pose term은 exact DSM target이 아니라 auxiliary pose-geometric consistency regularizer로 정의된다. 이는 (T=\widetilde T_\phi(u,z_e))가 (u)의 deterministic graph output이기 때문이다. 

[
\mathcal L_{\text{pose}}
========================

\mathbb E
\left[
\lambda_p(r)
\sum_h
\left|
J_Q(u_{h,r},z_e)s_{\theta,h}^u
------------------------------

\frac{1}{\tau(r)}
\Log_{\mathrm{SE}(3)}
\left(
\widetilde T_\phi(u_{h,r})^{-1}
\widetilde T_\phi(u_{h,0})
\right)
\right|_W^2
\right]
]

Total loss:

[
\mathcal L
==========

\mathcal L_{\text{score}}
+
\mu_{\text{pose}}\mathcal L_{\text{pose}}
]

---

### 8.2 μ_pose sweep

(\mu_{\text{pose}}=0.1)이 효과가 있으면 다음 sweep으로 확장한다.

[
\mu_{\text{pose}}\in{0.05,0.1,0.3}
]

| (\mu_{\text{pose}}) | 예상                                    |
| ------------------: | ------------------------------------- |
|                0.05 | 안정적이지만 약할 수 있음                        |
|                 0.1 | 첫 후보                                  |
|                 0.3 | endpoint 개선 가능, mode/smoothness 손상 위험 |
|                 1.0 | 현재 단계에서는 과함                           |

---

## 9. Step 4 — 그래도 부족하면 (\beta_f=10)

현재 (\beta_f=20)은 매우 clean하다.

[
\alpha(K)\approx0.0067
]

즉, 거의 pure stationary prior에서 복원해야 한다.

장점:

* IK-free stationary reference claim이 가장 깨끗함.
* (p_K)와 (p_{\text{ref}}) mismatch가 작음.

단점:

* denoising 난이도가 높음.
* endpoint precision이 떨어질 수 있음.

만약 guidance + (\mu_{\text{pose}})로도 success가 부족하면:

[
\boxed{\beta_f=10}
]

을 시도한다.

[
\alpha(K)\approx0.082
]

즉, forward terminal distribution에 data dependence가 조금 더 남는다. 학습은 쉬워질 수 있지만, finite-(K) mismatch는 커진다.

따라서 (\beta_f=10) 실험에서는 반드시 다음을 보고해야 한다.

[
W_2(p_K,p_{\text{ref}})
]

또는

[
\mathrm{KL}(p_K|p_{\text{ref}})
]

---

## 10. Step 5 — Best recipe로 longer training

Best recipe가 잡힌 뒤에만 longer training을 한다.

순서:

```text
1. best guidance recipe
2. best μ_pose
3. best β_f
4. 100k training
5. 200k training
```

무작정 100k/200k를 먼저 가면 원인 분리가 안 된다.

---

## 11. Full recipe-search order

최종 순서는 다음이다.

```text
Step 1.
current v5.1 50k checkpoint
→ start/goal guidance + chart-norm penalty sampling sweep

Step 2.
best 주변에서 α_s/α_g local ratio sweep

Step 3.
guidance로 부족하면 μ_pose=0.1, 50k retrain

Step 4.
μ_pose ∈ {0.05, 0.1, 0.3} sweep

Step 5.
그래도 부족하면 β_f=10 ablation

Step 6.
best recipe로 100k / 200k longer training
```

---

## 12. Expected outcomes and interpretation

### Case A — Guidance만으로 success가 크게 회복됨

예:

```text
succ@(5cm,5°): 17.6% → 40%+
succ@(5cm,10°): 47.3% → 60%+
‖u‖∞ p99 < 3.0
mfe = 0
jvio = 0
```

해석:

> v5.1 score model은 mode와 trajectory prior는 잘 학습했고, 부족했던 것은 endpoint refinement였다.
> start/goal guidance가 IK seed의 non-cheating 대체 역할을 한다.

---

### Case B — Guidance를 키우면 success는 오르지만 saturation이 악화됨

예:

```text
succ 상승
‖u‖∞ p99 > 4
```

해석:

> endpoint로 가기 위해 chart boundary를 과도하게 사용하고 있다.
> (R_u) 또는 velocity/acceleration regularization이 필요하다.

대응:

```text
α_u 증가
velocity/acceleration reward 활성화
guidance strength 감소
μ_pose 재학습 검토
```

---

### Case C — Guidance가 거의 안 먹힘

예:

```text
succ@(5cm,5°) < 30%
rot err 감소 없음
```

해석:

> score field 자체가 endpoint pose tangent와 충분히 정렬되지 않았다.
> training-time pose consistency signal이 필요하다.

대응:

```text
μ_pose=0.1 retrain
```

---

### Case D — μ_pose도 부족함

해석:

> β_f=20의 denoising 난이도가 너무 높거나, model capacity/training steps가 부족하다.

대응:

```text
β_f=10 ablation
100k/200k training
network capacity 증가
```

---

## 13. Current recommendation

지금 바로 실행할 것은 다음 하나다.

[
\boxed{
\textbf{current v5.1 50k checkpoint에서 guidance + } R_u \textbf{ sampling sweep}
}
]

구체적으로는 G1–G8을 먼저 돌린다.

가장 먼저 확인해야 할 것은:

```text
1. succ@(5cm,5°)가 35% 이상으로 회복되는가?
2. succ@(5cm,10°)가 60% 이상으로 회복되는가?
3. ‖u‖∞ p99가 3.0 이하로 내려가는가?
4. mfe=0이 유지되는가?
5. jvio=0이 유지되는가?
```

이 결과가 나온 뒤에 (\mu_{\text{pose}}) 재학습 여부를 결정한다.

---

## 14. One-line summary

현재 단계의 최우선 목표는 재학습이 아니라, **IK-free v5.1 checkpoint에서 start/goal guidance와 chart-norm penalty를 켜서 endpoint refinement와 saturation control이 가능한지 확인하는 것**이다.

```
```
