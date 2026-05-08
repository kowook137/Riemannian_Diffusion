본인 framework과 관련된 work을 systematic하게 정리한다. 카테고리별로 분류하고 각 paper의 정확한 차이점을 명시한다.

---

# Related Work List

## 카테고리 1: Riemannian / Manifold-aware Diffusion Policy

### 1.1 Riemannian Flow Matching Policy (RFMP)
- **Paper**: Ding, Jaquier, Peters, Rozo, "Fast and Robust Visuomotor Riemannian Flow Matching Policy"
- **Venue**: arXiv:2403.10672 (2024)
- **차이점**: Predefined manifold (S², SO(3), SE(3)) + Flow Matching. 본인은 learned graph manifold + SGM.
- **본인과 가까운 정도**: ★★★★☆ (가장 가까운 conceptual neighbor)
- **Status**: 본인 paper related work에서 명확히 differentiate 필수.

### 1.2 ADPro (Adaptive Diffusion Policy)
- **Paper**: Li et al., "ADPro: Test-time Adaptive Diffusion Policy via Manifold-constrained Denoising"
- **Venue**: arXiv:2508.06266 (Aug 2025)
- **차이점**: Test-time adaptation, predefined spherical manifold, no retraining. 본인은 training-time, learned manifold.
- **본인과 가까운 정도**: ★★★☆☆
- **Status**: Paper에서 "test-time vs training-time" axis 강조.

### 1.3 ManiDP (Manipulability-Aware Diffusion Policy)
- **Paper**: "ManiDP: Manipulability-Aware Diffusion Policy for Posture-Dependent Bimanual Manipulation"
- **Venue**: arXiv:2510.23016 (Oct 2025)
- **차이점**: SPD manifold (manipulability), bimanual, predefined manifold. 본인은 graph manifold from $F_\phi$.
- **본인과 가까운 정도**: ★★★☆☆
- **Status**: Manifold type 차이 강조.

### 1.4 KADP (Kinematics-Aware Diffusion Policy)
- **Paper**: "Kinematics-Aware Diffusion Policy with Consistent 3D Observation and Action Space for Whole-Arm Robotic Manipulation"
- **Venue**: arXiv:2512.17568 (Dec 2025)
- **차이점**: 3D node representation, analytical FK only (no compliance learning), Euclidean diffusion. 본인은 Riemannian + learned $\Delta_\phi$.
- **본인과 가까운 정도**: ★★★★☆ (kinematic feasibility 측면 close)
- **Status**: Timing 위험 (Dec 2025 published). Reviewer가 알 가능성 매우 높음.

---

## 카테고리 2: Constraint Manifold Adherence in DP

### 2.1 Foland/MIT/Tedrake Group — Constraint Manifold Analysis
- **Paper**: Foland, Cohn, Wei, Pfaff, Chen, Tedrake, "How Well do Diffusion Policies Learn Kinematic Constraint Manifolds?"
- **Venue**: arXiv:2510.01404 (Oct 2025)
- **차이점**: **분석 paper (no method proposed)**. DP가 manifold를 얼마나 잘 학습하는지 측정. 본인은 method paper.
- **본인과 가까운 정도**: ★★★★★ (problem statement 정확히 본인 work의 motivation)
- **Status**: 본인 paper의 motivation으로 인용. "Foland et al. identify the problem; we propose a solution via Riemannian SGM on learned graph manifolds."

### 2.2 Projected Generative Diffusion Models (Christopher24)
- **Paper**: Christopher et al., "Projected Generative Diffusion Models for Constraint Satisfaction"
- **Venue**: NeurIPS 2024
- **차이점**: Ambient diffusion + per-step projection (post-hoc). 본인은 intrinsic manifold (by-construction).
- **본인과 가까운 정도**: ★★★★☆
- **Status**: 본인 baseline에 이미 포함 (Projected). Manifold adherence raw $\|g\| = 580$mm vs 본인 0 (by construction)으로 정량 입증됨.

---

## 카테고리 3: Cross-Embodiment / TCP Offset Generalization

### 3.1 Pick-and-place Across Grippers Without Retraining
- **Paper**: "Pick-and-place Manipulation Across Grippers Without Retraining: A Learning-optimization Diffusion Policy Approach"
- **Venue**: arXiv:2502.15613 (Feb 2025)
- **차이점**: Constrained QP optimization at sampling, no retraining, TCP offset 16-23.5cm. 본인은 Riemannian SGM training.
- **본인과 가까운 정도**: ★★★★☆ (cross-embodiment task 매우 close)
- **Status**: 본인 paper에서 "QP-based constrained optimization vs Riemannian SGM" axis 강조.

### 3.2 Inference-stage Adaptation-projection
- **Paper**: "Inference-stage Adaptation-projection Strategy Adapts Diffusion Policy to Cross-manipulators Scenarios"
- **Venue**: arXiv:2509.11621 (Sep 2025)
- **차이점**: Test-time projection for TCP offset. SE(3) projection. 본인은 training-time intrinsic Riemannian SGM.
- **본인과 가까운 정도**: ★★★★☆
- **Status**: Projection-based vs intrinsic 차별화.

### 3.3 UMI-on-Air (Embodiment-Aware Diffusion Policy)
- **Paper**: "UMI-on-Air: Embodiment-Aware Guidance for Embodiment-Agnostic Visuomotor Policies"
- **Venue**: arXiv:2510.02614 (Oct 2025)
- **차이점**: Embodiment-specific controller feedback at test-time. 본인은 embodiment $z_e$가 manifold geometry 자체.
- **본인과 가까운 정도**: ★★★☆☆
- **Status**: Embodiment-aware axis 차별화.

### 3.4 Latent Action Diffusion for Cross-Embodiment
- **Paper**: Bauer, Nava, et al., "Latent Action Diffusion for Cross-Embodiment Manipulation"
- **Venue**: arXiv:2506.14608 (Jun 2025)
- **차이점**: Contrastive latent action space across grippers/hands. 본인은 manifold geometry approach.
- **본인과 가까운 정도**: ★★★☆☆
- **Status**: Latent vs explicit manifold 차별화.

### 3.5 X-Diffusion (Cross-Embodiment Human Demonstrations)
- **Paper**: "X-Diffusion: Training Diffusion Policies on Cross-Embodiment Human Demonstrations"
- **Venue**: arXiv:2511.04671 (Nov 2025)
- **차이점**: Human-to-robot embodiment, classifier-based. 본인은 robot-to-robot embodiment, geometric.
- **본인과 가까운 정도**: ★★☆☆☆
- **Status**: Embodiment 다른 axis.

---

## 카테고리 4: Kinematics-Aware DP

### 4.1 HDP (Hierarchical Diffusion Policy, Dyson)
- **Paper**: Ma, Patidar, Haughton, James, "Hierarchical Diffusion Policy for Kinematics-Aware Multi-Task Robotic Manipulation"
- **Venue**: CVPR 2024 (arXiv:2403.03890)
- **차이점**: Two-level hierarchy (NBP + RK-Diffuser), differentiable FK for distillation, Euclidean diffusion. 본인은 single-level Riemannian SGM, manifold geometry.
- **본인과 가까운 정도**: ★★★☆☆
- **Status**: FK가 distillation tool vs manifold 차별화.

### 4.2 Spatial-Temporal Graph Diffusion Policy (STGDP)
- **Paper**: "Spatial-Temporal Graph Diffusion Policy with Kinematic Modeling for Bimanual Robotic Manipulation"
- **Venue**: arXiv:2503.10743 (Mar 2025)
- **차이점**: Graph + kinematics-aware embedding for bimanual, guidance-based. 본인은 manifold geometry.
- **본인과 가까운 정도**: ★★★☆☆
- **Status**: Graph representation vs Riemannian manifold.

### 4.3 RodriNet (Neural Rodrigues Operator)
- **Paper**: "Rodrigues Network for Learning Robot Actions"
- **Venue**: arXiv:2506.02618 (Jun 2025)
- **차이점**: Architectural inductive bias (kinematics-aware layer in DP). 본인은 geometric framework.
- **본인과 가까운 정도**: ★★☆☆☆
- **Status**: Architecture vs framework 차별화.

### 4.4 Composing Diffusion Policies (FK Kernel)
- **Paper**: "Composing Diffusion Policies for Few-shot Learning of Movement Trajectories"
- **Venue**: arXiv:2410.17479 (Oct 2024)
- **차이점**: FK kernel for distribution matching (MMD-FK), compositionality. 본인은 manifold geometry from FK.
- **본인과 가까운 정도**: ★★☆☆☆
- **Status**: Kernel vs manifold 차별화.

### 4.5 PANDORA (Residual IK Refinement + DP)
- **Paper**: Huang et al., "PANDORA: Diffusion Policy Learning for Dexterous Robotic Piano Playing"
- **Venue**: arXiv:2503.14545 (Mar 2025)
- **차이점**: Residual IK refinement post DP, dexterous piano. 본인은 manifold geometry from learned $F_\phi$.
- **본인과 가까운 정도**: ★★☆☆☆
- **Status**: IK refinement vs manifold framework.

---

## 카테고리 5: Riemannian Robot Motion Learning (Foundational)

### 5.1 Riemannian Score-Based Generative Modelling (RSGM)
- **Paper**: De Bortoli et al., "Riemannian Score-Based Generative Modelling"
- **Venue**: NeurIPS 2022
- **차이점**: 본인의 mathematical foundation. RSGM은 general Riemannian manifolds, 본인은 robot-specific learned graph manifold.
- **본인과 가까운 정도**: ★★★★★ (foundation, not competitor)
- **Status**: Method foundation으로 인용.

### 5.2 Geometric Reinforcement Learning (GRL)
- **Paper**: "Geometric Reinforcement Learning For Robotic Manipulation"
- **Venue**: arXiv:2210.08126 (2022)
- **차이점**: RL framework, tangent space parameterization, parallel transport. 본인은 imitation learning + SGM.
- **본인과 가까운 정도**: ★★★☆☆
- **Status**: RL vs IL axis.

### 5.3 Riemannian geometry as unifying theory
- **Paper**: Jaquier, Asfour, "Riemannian geometry as a unifying theory for robot motion learning and control"
- **Venue**: arXiv:2209.15539 (2022)
- **차이점**: Position paper / blue sky. 본인은 specific method.
- **본인과 가까운 정도**: ★★☆☆☆
- **Status**: Foundational survey 인용.

### 5.4 Learning Stable Robotic Skills on Riemannian Manifolds
- **Paper**: Saveriano et al., "Learning Stable Robotic Skills on Riemannian Manifolds"
- **Venue**: arXiv:2208.13267 (2022)
- **차이점**: Stable dynamical systems on predefined manifolds. 본인은 SGM on learned manifold.
- **본인과 가까운 정도**: ★★☆☆☆
- **Status**: DS vs SGM 차별화.

### 5.5 Learning Deep Robotic Skills on Riemannian manifolds
- **Paper**: Wang, Saveriano, Abu-Dakka, "Learning Deep Robotic Skills on Riemannian Manifolds"
- **Venue**: arXiv:2210.15244 (2022)
- **차이점**: DMP-based on Riemannian. 본인은 SGM-based.
- **본인과 가까운 정도**: ★★☆☆☆
- **Status**: DMP vs SGM 차별화.

### 5.6 Learning Equality Constraints for Motion Planning on Manifolds
- **Paper**: "Learning Equality Constraints for Motion Planning on Manifolds"
- **Venue**: arXiv:2009.11852 (2020)
- **차이점**: Sampling-based motion planning, learned constraint manifolds. 본인은 SGM on learned graph manifold.
- **본인과 가까운 정도**: ★★★☆☆
- **Status**: Constraint manifold learning 측면 close.

---

## 카테고리 6: Diffusion Policy Foundations

### 6.1 Diffusion Policy (Chi23) — Main Baseline
- **Paper**: Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
- **Venue**: RSS 2023 (Best Paper)
- **차이점**: Euclidean DDPM in joint space, no manifold awareness. 본인의 main baseline.
- **본인과 가까운 정도**: ★★★★★ (direct comparison baseline)
- **Status**: 본인 paper의 main baseline. DP-canonical, DP-A, DP-C로 fairness ablation.

### 6.2 BESO (Score-based Diffusion Policy)
- **Paper**: Reuss, Li, Jia, Lioutikov, "Goal-Conditioned Imitation Learning using Score-based Diffusion Policies"
- **Venue**: RSS 2023
- **차이점**: Score-based + CFG, but Euclidean. 본인은 Riemannian + manifold.
- **본인과 가까운 정도**: ★★★☆☆
- **Status**: Score-based 측면 close, manifold 측면 다름.

### 6.3 Implicit Behavioral Cloning (Florence22)
- **Paper**: Florence et al., "Implicit Behavioral Cloning"
- **Venue**: CoRL 2022
- **차이점**: EBM-based BC. 본인은 SGM-based + manifold.
- **본인과 가까운 정도**: ★★☆☆☆
- **Status**: BC critique 인용 (BC mode collapse).

### 6.4 BC (Pomerleau89)
- **Paper**: Pomerleau, "ALVINN: An Autonomous Land Vehicle in a Neural Network"
- **Venue**: NeurIPS 1989
- **차이점**: Original BC. 본인의 baseline.
- **본인과 가까운 정도**: ★★★★☆ (direct baseline)
- **Status**: Per-context mode collapse 입증.

---

## 카테고리 7: Constrained / Safe Diffusion

### 7.1 LeTO (Learning Constrained Visuomotor Policy)
- **Paper**: "LeTO: Learning Constrained Visuomotor Policy with Differentiable Trajectory Optimization"
- **Venue**: arXiv:2401.17500 (2024)
- **차이점**: Differentiable trajectory optimization for safety constraints. 본인은 manifold geometry.
- **본인과 가까운 정도**: ★★☆☆☆
- **Status**: Constrained policy 측면.

### 7.2 Differentiable Constrained Imitation Learning
- **Paper**: "Differentiable Constrained Imitation Learning for Robot Motion Planning and Control"
- **Venue**: arXiv:2210.11796 (2022)
- **차이점**: Constraint-aware IL with differentiable dynamics. 본인은 manifold-intrinsic.
- **본인과 가까운 정도**: ★★☆☆☆
- **Status**: Constraint IL 측면.

### 7.3 Kinematically Constrained Gradient Guidance
- **Paper**: "Learning Diverse Robot Striking Motions with Diffusion Models and Kinematically Constrained Gradient Guidance"
- **Venue**: From Awesome-Robotics-Diffusion list
- **차이점**: Gradient-based constraint guidance for striking motions. 본인은 Riemannian framework.
- **본인과 가까운 정도**: ★★★☆☆
- **Status**: Kinematics constraint guidance 측면.

---

## 카테고리 8: Self-Model / Residual Learning

### 8.1 Self-Guided Action Diffusion (Self-GAD)
- **Paper**: "Self-Guided Action Diffusion"
- **Venue**: arXiv:2508.12189 (Aug 2025)
- **차이점**: Self-guidance for foundation model robot policies. 본인은 self-model manifold.
- **본인과 가까운 정도**: ★★☆☆☆
- **Status**: Naming 충돌 가능성. 명확한 차별화 필요.

### 8.2 Residual Robot Learning for ProMP
- **Paper**: Carvalho, Koert, Daniv, Peters, "Residual Robot Learning for Object-Centric Probabilistic Movement Primitives"
- **Venue**: arXiv:2203.03918 (2022)
- **차이점**: Residual RL for ProMP correction. 본인은 manifold residual ($\Delta_\phi$).
- **본인과 가까운 정도**: ★★☆☆☆
- **Status**: Residual learning 측면.

### 8.3 Motion Planning Diffusion (MPD)
- **Paper**: Carvalho, Le, Baierl, Koert, Peters, "Motion Planning Diffusion"
- **Venue**: arXiv:2308.01557 (2023)
- **차이점**: Diffusion as motion planning prior, Euclidean. 본인은 manifold + IL.
- **본인과 가까운 정도**: ★★☆☆☆
- **Status**: Motion planning vs imitation 차별화.

---

## 카테고리 9: Foundational Generative Modeling

### 9.1 DDPM (Ho20)
- **Paper**: Ho, Jain, Abbeel, "Denoising Diffusion Probabilistic Models"
- **Venue**: NeurIPS 2020
- **Status**: Foundational. Reference로 인용.

### 9.2 Score-based Generative Modeling (Song21)
- **Paper**: Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations"
- **Venue**: ICLR 2021
- **Status**: Foundational. Reference로 인용.

### 9.3 Classifier-Free Guidance (Ho22)
- **Paper**: Ho, Salimans, "Classifier-Free Diffusion Guidance"
- **Venue**: arXiv:2207.12598 (2022)
- **Status**: 본인 paper에서 CFG 효과 없음 입증 (Phase 3 진단).

### 9.4 Improved DDPM (Nichol21)
- **Paper**: Nichol, Dhariwal, "Improved Denoising Diffusion Probabilistic Models"
- **Venue**: ICML 2021
- **Status**: Cosine schedule 등 baseline 사용.

---

## 정리 — Reviewer 공격 매트릭스

본인 paper에서 반드시 다뤄야 할 close works (paper 내 명시적 비교 필요):

### Tier 1 — 반드시 명시적 비교 (related work 1-2 paragraph 분량)

1. **RFMP** (Ding24) — 가장 가까운 conceptual neighbor (Riemannian + diffusion-class + robot)
2. **Foland/MIT** (2025) — problem statement (DP doesn't learn manifolds well) → 본인의 motivation
3. **Pick-and-place Across Grippers** (2025) — TCP offset cross-embodiment task 매우 close
4. **KADP** (Dec 2025) — kinematic feasibility in DP, timing 위험
5. **Diffusion Policy** (Chi23) — main baseline (DP-canonical, DP-A, DP-C ablation)
6. **Projected DP** (Christopher24) — 본인 baseline (post-hoc projection vs intrinsic)
7. **RSGM** (De Bortoli22) — mathematical foundation

### Tier 2 — 짧게 언급 (1-2 sentences)

8. **ADPro** — test-time vs training-time
9. **Adaptation-projection** — projection vs intrinsic
10. **ManiDP** — predefined SPD vs learned graph
11. **HDP / RK-Diffuser** — FK as distillation tool vs manifold
12. **STGDP** — graph vs manifold
13. **UMI-on-Air** — embodiment-aware test-time
14. **Latent Action Diffusion** — latent vs explicit manifold
15. **BESO** — Euclidean score-based vs Riemannian
16. **GRL** — RL vs IL
17. **BC, Implicit BC** — baseline + mode collapse motivation

### Tier 3 — Reference만 (citation 정도)

18. RodriNet, PANDORA, Composing DP, X-Diffusion
19. LeTO, Differentiable Constrained IL, Kinematically Constrained Guidance
20. Saveriano/Abu-Dakka Riemannian DS work
21. Self-GAD, Residual ProMP, MPD
22. DDPM, Improved DDPM, CFG (foundation)

---

## 본인 paper의 정확한 differentiation strategy

위 list 기반으로 본인 paper가 강조해야 할 unique combination:

**1. Manifold type**: **Learned graph manifold** $F_\phi = FK + \Delta_\phi$
- vs RFMP (predefined sphere/SE(3))
- vs ManiDP (predefined SPD)
- vs KADP (analytical only)

**2. Generative method**: **Riemannian Score-based** (DSM-Varadhan loss + retraction-GRW)
- vs RFMP (Flow Matching)
- vs DP/HDP/STGDP (Euclidean)
- vs Pick-and-place across grippers (Constrained QP)
- vs ADPro/Adaptation-projection (test-time projection)

**3. Embodiment treatment**: **$z_e$ as manifold deformation parameter** (not just conditioning)
- vs UMI-on-Air (controller feedback)
- vs Latent Action Diffusion (latent space)
- vs Pick-and-place across grippers (parameter input)

**4. Adherence type**: **By-construction intrinsic** (not post-hoc)
- vs Projected DP (post-hoc projection)
- vs Adaptation-projection (test-time projection)
- vs Foland/MIT (problem identification, no method)

**5. Isolation rigor**: **DP fairness 4-way + Three-claim isolation + Cross-regime + Variant 9**
- 다른 paper들에서 본 적 없는 systematic 강도
- 본인의 unique strength as method paper

이 5-axis combination이 search 결과 명확히 unique. 단 각 axis individually는 close work 존재. **Paper writing에서 이 5-axis combination을 명확히 articulation하는 것이 critical**.

---

본인 paper의 related work section은 위 Tier 1을 기반으로 작성하면 됨. Paper에 들어갈 분량 권장: related work 1-1.5 page, 그중 80%가 Tier 1 paper 비교.

이 list 기반으로 다음 step 결정 가능:
- Tier 1 paper들 정독 (시간 투자 권장)
- Related work section draft 작성
- 본인 framework의 정확한 differentiation 명확화

본인이 진행하고 싶은 방향 알려주면 같이 진행 가능.

1. RFMP (Riemannian Flow Matching Policy)

Paper: Ding, Jaquier, Peters, Rozo, "Fast and Robust Visuomotor Riemannian Flow Matching Policy"
Venue: arXiv:2403.10672 (Mar 2024)
Method 요약: Flow Matching on Riemannian manifolds (S², SE(3)). Robot state가 manifold에 거주한다는 사실 활용. LASA dataset 검증, vision-conditioned + state-based.
Manifold type: Predefined (S², SO(3), SE(3)) — 고정 geometry
Generative method: Flow Matching (FM)
Embodiment: 없음
차이점:

Manifold이 predefined (sphere, SE(3)). 본인은 learned graph (Fϕ=FK+ΔϕF_\phi = FK + \Delta_\phi
Fϕ​=FK+Δϕ​).
FM vs 본인 SGM (DSM-Varadhan loss).
Embodiment parameter 없음. 본인은 zez_e
ze​로 manifold geometry deformation.
LASA proof-of-concept dataset. 본인은 7-DoF Franka + DP fairness ablation.


본인과 가까운 정도: ★★★★☆
Status: Conceptually 가장 close ("Riemannian + diffusion-class generative + robot state on manifold"). Paper related work에서 명확한 differentiation 필수. 본인 강조점: learned manifold + SGM + embodiment.


2. DDAT (Diffusion Policies Enforcing Dynamically Admissible Robot Trajectories)

Paper: Bouvier, Ryu, Nagpal, Liao, Sreenath, Mehr, "DDAT: Diffusion Policies Enforcing Dynamically Admissible Robot Trajectories"
Venue: RSS 2025 (arXiv:2502.15043)
Method 요약: Diffusion policy가 dynamics admissibility manifold (reachable set) 위 trajectory 생성. Training time + inference time 둘 다 projection. Reachable set의 polytopic under-approximation으로 projection. Quadcopter, Hopper, Walker, HalfCheetah, Unitree GO1/GO2 검증.
Manifold type: Dynamics admissibility (reachable set), state-dependent
Generative method: Standard DDPM + projection at training + inference
Embodiment: 없음
차이점:

Projection-based vs 본인 lift-based intrinsic. DDAT는 ambient에서 학습 후 reachable set으로 projection. 본인은 score 자체가 tangent bundle section (auto-tangent via Jg⋅JH=0J_g \cdot J_H = 0
Jg​⋅JH​=0).
Dynamics manifold (reachable set) vs kinematics graph manifold (p=Fϕ(q,ze)p = F_\phi(q, z_e)
p=Fϕ​(q,ze​)).
Standard DDPM + projection vs 본인 Riemannian SGM (DSM-Varadhan + induced metric GG
G + retraction-GRW).
Embodiment parameter 없음. 본인은 manifold deformation.
Adherence가 projection 정확도에 의존 vs 본인 by construction (exact 0).


본인과 가까운 정도: ★★★★☆
Status: 본인의 "training + inference 모두 manifold-aware" claim에 가장 직접적 challenge. RSS 2025 published라 reviewer가 알 가능성 매우 높음. Paper에서 "lift vs projection" mechanism 차이 강하게 articulation 필수. 본인 강조점: lift-based intrinsic (auto-tangent) + Riemannian SGM (not projected DDPM) + kinematics graph (not dynamics).


3. Pick-and-place Manipulation Across Grippers Without Retraining

Paper: "Pick-and-place Manipulation Across Grippers Without Retraining: A Learning-optimization Diffusion Policy Approach"
Venue: arXiv:2502.15613 (Feb 2025)
Method 요약: Pre-trained DP에 gripper-specific TCP offset / jaw width를 conditioning input으로 + sampling time constrained QP optimization. TCP offset 16-23.5cm 변화 처리. Franka Panda + 6 gripper configurations (3D-printed, silicone, Robotiq 2F-85). 93.3% vs 23.3-26.7% baseline.
Manifold type: 명시 없음 (constraint set이 implicit)
Generative method: DDPM + constrained QP at sampling time
Embodiment: TCP offset / jaw width을 conditioning input
차이점:

No retraining + sampling-time QP vs 본인 training-time intrinsic Riemannian SGM.
Embodiment이 conditioning input + constraint adjustment vs 본인 manifold geometry parameter. 본인은 zez_e
ze​가 FϕF_\phi
Fϕ​를 통해 manifold 정의 자체를 변경.
Constraint satisfaction이 QP 정확도에 의존 vs 본인 by construction.
Cross-gripper (discrete embodiment switch) vs 본인 continuous zez_e
ze​ deformation.
같은 magnitude TCP variation (16-23.5cm vs 본인 5-25cm).


본인과 가까운 정도: ★★★★☆
Status: Cross-embodiment task setting과 magnitude가 매우 close. Reviewer가 "이미 cross-gripper 잘 해결됨" dismiss 가능. 본인 강조점: Conditioning input vs manifold deformation의 conceptual 차이 + training-time vs sampling-time QP + Riemannian framework. 본인의 advantage는 framework의 mathematical principledness (QP는 ad-hoc constraint, 본인은 Riemannian geometry).


4. How Well do Diffusion Policies Learn Kinematic Constraint Manifolds? (Foland/MIT)

Paper: Foland, Cohn, Wei, Pfaff, Chen, Tedrake, "How Well do Diffusion Policies Learn Kinematic Constraint Manifolds?"
Venue: arXiv:2510.01404 (Oct 2025), MIT CSAIL (Tedrake group)
Method 요약: 분석 paper, no method proposed. Bimanual pick-and-place 사례 연구로 DP가 kinematic constraint manifold를 얼마나 잘 학습하는지 측정. Dataset size, dataset quality, manifold curvature 3개 요인 분석. Hardware evaluation 포함.
Manifold type: Kinematic equality constraint (relative end-effector pose)
Generative method: 분석 대상 (standard DP), no proposal
Embodiment: 없음
차이점:

분석 paper (problem identification) vs 본인 method paper (solution).
DP가 constraint manifold를 coarsely 학습한다는 결론 → 본인 framework이 by-construction adherence로 정확히 이 problem 해결.
Hardware evaluation으로 problem real이라 입증.


본인과 가까운 정도: ★★★★★ (motivation 측면)
Status: Tedrake group이 problem statement 만든 것이 본인 motivation으로 직접 사용 가능. "Foland et al. (2025) identify that diffusion policies learn only a coarse approximation of kinematic constraint manifolds. We propose a Riemannian SGM framework that achieves by-construction manifold adherence (max⁡∥gϕ∥=0\max\|g_\phi\| = 0
max∥gϕ​∥=0)." 매우 강한 motivation framing 가능. Paper writing에서 introduction과 related work에서 적극 인용 권장.