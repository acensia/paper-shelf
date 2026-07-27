# Unsupervised Partner Design Enables Robust Ad-hoc Teamwork
## (비지도 파트너 설계를 통한 강건한 Ad-hoc Teamwork)

**저자:** Constantin Ruhdorfer¹, Matteo Bortoletto¹, Victor Oei¹, Anna Penzkofer¹, Andreas Bulling¹
¹Collaborative Artificial Intelligence, University of Stuttgart, Germany
correspondence: constantin.ruhdorfer@vis.uni-stuttgart.de

**게재:** ICML 2026 (43rd ICML, Seoul, South Korea, PMLR 306) · arXiv: [2508.06336](https://arxiv.org/abs/2508.06336) **v2 (2026-06-07)**

> **번역 노트:** 주요 기술 용어(ad-hoc teamwork, learnability, ego agent, curriculum, partner generator 등)는 영어 원문 그대로 유지하고 설명만 한국어로 번역했습니다. 수식과 표는 원문 구조를 보존했으며, 참고문헌 목록(저자·제목)은 원문 그대로 두었습니다.

> **⚠️ 버전 안내 (v1 → v2 주요 변경):** 이 문서는 arXiv **v2(2026-06-07)** 기준 번역입니다. v1 대비 큰 폭의 개정이 있었습니다.
> - **벤치마크 추가:** Level-Based Foraging (LBF) 가 추가되어, 본문이 *고정 환경 실험(§5)* 과 *절차적 생성 실험(§6)* 으로 재편되었습니다.
> - **형식 변경:** 사전 지식의 RL 형식이 Dec-UPOMDP → **under-specified cooperative two-agent stochastic game** 으로 바뀌었고, AHT를 별도 절(§3.2)로 정식화했습니다.
> - **이론 추가:** learnability(return 분산)와 PPO의 기대 정책 개선 사이의 형식적 연결(**Foster et al. 2026, LILO**)이 §4.1에 추가되었습니다.
> - **신규 절:** §4.4 *UPD and Convention Selection* (2×2 행렬 게임 분석)이 추가되었습니다.
> - **결과 수치 변경(중요):** v1의 일부 베이스라인 실험에 구현·설정 오류가 있었고, 이를 수정·재실행하여 **Table 1의 베이스라인 성능이 개선**되었습니다(부록 B의 *Transparency Note*). 단, **논문의 주요 결론은 동일**합니다.
> - **신규 부록:** C(관습 선택 확장), D(ROTATE 및 fine-tuned UPD 비교, Table 3), F(계산 비용 분석), G.1(LBF), J(대안 learnability) 등.
> - **사용자 연구:** 참가자 연령이 22–31세(여성 5명)로 보고됨(v1은 26–34세). 집계 신뢰도 Cronbach's α = 0.916(v1은 0.938).
> - **감사의 글:** ROTATE 비교를 제안한 익명 심사위원에게 감사. (v1의 DFG EXC 2075 지원 문구는 삭제됨.)

---

## Abstract (초록)

본 논문은 강건한 ad-hoc teamwork를 위한, population이 필요 없는(population-free) 멀티에이전트 강화학습 방법인 **Unsupervised Partner Design (UPD)** 을 제안한다. UPD는 학습 파트너를 실시간(on-the-fly)으로 생성하고 이를 learnability 기준으로 적응적으로 선택함으로써, 사전학습된 partner population이나 수동 파라미터 튜닝의 필요를 제거한다. 우리는 이 단순한 메커니즘이 효과적인 partner 다양성을 가능하게 하며, 절차적 level generator가 있을 때 **partner-environment 결합 선택(joint partner-environment selection)** 으로 직접 확장됨을 보인다. **Level-Based Foraging, Overcooked-AI, Overcooked Generalisation Challenge** 전반에서, UPD는 population 기반·population-free 베이스라인 모두에 대해 일관되게 강력한 성능을 달성한다. 인간-AI 사용자 연구에서, UPD로 학습한 에이전트는 더 높은 return을 얻으며 모든 평가 베이스라인보다 **더 적응적이고(more adaptive), 더 인간 같으며(more human-like), 덜 답답하다(less frustrating)** 고 평가받았다.

---

## 1 Introduction (서론)

알려지지 않은 파트너와의 강건한 협력, 즉 **ad-hoc teamwork**(Stone et al. 2010, AHT)는 협력적 인공지능(AI)의 핵심 요구사항이다. AHT 설정에서 에이전트는 파트너의 정책에 대한 가정 없이 협응해야 하며, 따라서 배치(deployment) 파트너가 학습 시 만난 파트너와 다를 때 학습된 협응 전략이 취약해지기 쉽다. AHT를 위한 학습은 비용이 큰데, 기존 방법들이 대개 다양한 partner 정책의 대규모 population을 필요로 하거나(Strouse et al. 2021; Zhao et al. 2023; Yu et al. 2023; Li et al. 2023b; Wang et al. 2025), 전문가 지식과 수작업 모델(Albrecht & Ramamoorthy 2013; Barrett et al. 2014, 2017; Albrecht et al. 2016)을 포함하기 때문이다. 이러한 partner population을 유지·튜닝하는 비용은 과제와 partner 다양성이 커질수록 점점 더 비싸진다. Efficient end-to-end training(E3T, Yan et al. 2023)은 partner를 ego 정책과 무작위 정책의 확률적 혼합(stochastic mixture)으로 생성하여 명시적 population을 피함으로써 이 문제를 부분적으로 해결하지만, 여전히 과제와 평가 설정마다 혼합 파라미터를 세심하게 튜닝해야 하여 확장성이 제한된다.

이와 병행하여, **unsupervised environment design**(Dennis et al. 2020, UED)은 환경 파라미터에 대한 적응적 curriculum이 일반화를 크게 향상시킬 수 있음을 보였다. 최근 연구는 partner와 environment에 걸친 결합 일반화(joint generalisation)가 강건한 협력에 핵심임을 강조하는 동시에, 기존 방법들이 이 설정으로 확장하지 못함을 지적하였다(Ruhdorfer et al. 2025b).

우리의 연구는 두 가지 상위 질문에 의해 이끌린다. **첫째**, 환경 설계와 유사하게, 협력 파트너를 명시적 population 유지 없이 *저렴하고 적응적으로* 생성하여 강건한 AHT 에이전트를 학습시킬 수 있는가? **둘째**, 그러한 partner-design 메커니즘이, population 기반 AHT 방법이 확장되기 어려운 절차적 생성 설정에서 **partner-environment 결합 curriculum** 으로 자연스럽게 확장되는가?

두 질문에 대한 답으로, 우리는 partner 행동에 대한 적응적 학습 분포를 온라인 생성·선택을 통해 구축하는 population-free AHT 방법인 **Unsupervised Partner Design (UPD)** 을 제안한다. UPD는 사전학습된 partner population과 수동 튜닝된 혼합 계수의 필요를 제거하면서도 zero-shot 협응에서 강력한 성능을 유지한다. 절차적 level generator가 있을 때, 동일한 메커니즘이 partner-environment 결합 선택으로 직접 확장되어 단순한 결합 curriculum 학습 접근법을 산출한다. 우리의 핵심 기여는, **단순한 생성·선택 메커니즘이 비자명한 창발적 협응 행동(emergent coordination behaviour)을 만들어 낼 수 있음**을 보이는 것이다. 구체적으로:

1. UED 아이디어가 partner 공간으로 확장될 수 있음을 보이고, 적응적 partner 생성·선택을 통해 AHT 에이전트를 학습시키는 population-free 방법인 **Unsupervised Partner Design** 을 도입한다. 이는 명시적 partner population의 필요를 제거한다.

2. **UPD가 강력한 AHT 성능을 달성함**을 Level-Based Foraging(Albrecht & Ramamoorthy 2013)과 Overcooked-AI(Carroll et al. 2019)에서, 다양한 인공 파트너 및 인간과의 평가를 통해(§5), population 기반·population-free 베이스라인 모두와 비교하여 보인다.

3. **동일한 메커니즘이 partner-environment 결합 curriculum으로 확장됨**을 절차적 생성 설정에서 보이며, Overcooked Generalisation Challenge(OGC, Ruhdorfer et al. 2025b)에서 강건한 zero-shot 협력을 달성한다(§6).

> **Figure 1.** *Unsupervised partner design* 은 ad-hoc teamwork를 위한 새로운 population-free 멀티에이전트 학습 프레임워크로, learnability를 이용해 ego agent에게 적합한 학습 파트너를 찾아 open-ended curriculum을 생성한다. (**Partner Generator** 가 후보를 보내고 → **Learner** 가 learnability에 기반해 채택하며 → ego agent를 학습시키고 → 그 결과가 generator를 갱신·시드(seed)한다.)

---

## 2 Related Work (관련 연구)

### 2.1 Ad-hoc Teamwork

AHT는 다양한 멀티에이전트 강화학습(RL) 환경에서 연구되어 왔다(Carroll et al. 2019; Bard et al. 2020; Kurach et al. 2020; Ruhdorfer et al. 2025a). 널리 쓰이는 AHT 방법들, 예컨대 fictitious co-play(FCP, Strouse et al. 2021)나 maximum entropy population-based training(MEP, Zhao et al. 2023)은 다양한 partner population을 사전학습하고 이들에 대한 best-response 정책을 최적화하는 데 의존한다(Yu et al. 2023; Lou et al. 2023; Rahman et al. 2023; Erlebach & Cook 2024; You et al. 2025). 최근 연구들은 open-ended 학습 목표를 도입하여 partner 다양성을 동적으로 확장하였으나(Li et al. 2023b; Wang et al. 2025), 여전히 시간이 지남에 따라 partner population을 키우거나, 사전학습 population에 대한 curriculum(Erlebach & Cook 2024) 또는 오프라인 데이터로부터 학습한 partner model에 대한 curriculum(Chaudhary et al. 2025)을 사용하였다. 주목할 예외는 E3T(Yan et al. 2023)로, 이는 ego 정책과 무작위 정책의 혼합으로 파트너를 실시간 생성한다. E3T는 human-AI 협응 설정에서 FCP·MEP 같은 선행 population 기반 접근법을 능가하는 강력한 성능을 보였다. 이 접근법은 어떤 partner population도 필요로 하지 않아 계산 부담을 크게 줄이지만, 여전히 과제와 평가 시나리오마다 ego 정책과 무작위 정책 사이의 혼합 계수를 세심하게 튜닝해야 하여 설정 전반의 강건성이 제한된다. 이와 대조적으로, 우리는 고정 파라미터나 사전학습 population 없이 다양한 partner 행동을 적응적으로 생성하며 기존 curriculum learning 프레임워크와 호환되는, 경량의 population-free 접근법을 제안한다.

### 2.2 Unsupervised Environment Design

UED(Dennis et al. 2020)는 에이전트의 능력에 맞춘 학습 환경을 적응적으로 생성하며, 일반화 향상에 효과적임이 입증되었다. domain randomisation(Tobin et al. 2017, DR)과 달리, UED는 에이전트의 학습 경계(learning frontier)를 겨냥하여 환경을 생성한다. 기존 UED 방법들은 주로 단일 에이전트 설정에 초점을 두며, 환경 생성을 유도하기 위해 regret 기반 목적함수에 의존한다(Wang et al. 2019, 2020; Dennis et al. 2020; Jiang et al. 2021a,b; Parker-Holder et al. 2022; Li et al. 2023a; Beukman et al. 2024). 멀티에이전트 설정으로의 확장은 제한적이다: Samvelyan et al. (2023)은 경쟁적(competitive) 설정에 초점을 두었고, Ruhdorfer et al. (2025b)은 협력적 멀티에이전트 UED 벤치마크를 제안했으나 방법(method)은 제시하지 않았으며, You et al. (2025)은 과거 self-play 체크포인트만으로 학습한다. 최근 연구들은 UED를 learnability 주도 문제로 재정식화하여, regret 근사를 환경의 학습 잠재력(learning potential)을 직접 측정하는 채점 함수로 대체하였다(Rutherford et al. 2024; Monette et al. 2025). 그러나 선행 연구는 환경 파라미터화에만 초점을 두었고 partner 정책을 curriculum 공간의 일부로 고려하지 않았다. 우리는 비지도 설계(unsupervised design)를 partner 정책으로 확장하여, population-free curriculum 메커니즘으로서의 적응적 partner 생성을 도입한다. 기존 UED 접근법과 결합되면, 이는 절차적 생성 설정에서 zero-shot 협력을 위한 partner-environment 결합 선택을 가능하게 한다.

---

## 3 Preliminaries (사전 지식)

### 3.1 Reinforcement Learning

멀티에이전트 unsupervised environment design에 관한 선행 연구(Samvelyan et al. 2023; Ruhdorfer et al. 2025b)에 착안하여, 우리는 본 설정을 **under-specified cooperative two-agent stochastic game(과소명세 협력적 2-에이전트 확률 게임)** 으로 모델링한다. 각 환경 인스턴스는 환경 파라미터 $\theta \in \Theta$ 로 인덱싱되는 Markov game $G_\theta = \langle S, A, T_\theta, R_\theta, \gamma, \rho_0 \rangle$ 로 정의된다. "under-specified(과소명세)" 설정은 level들의 *집합(family)* $\{G_\theta\}_{\theta \in \Theta}$ 에 대응하며, 여기서 각 $\theta$ 는 게임의 특정 인스턴스(하나의 **level**)를 정의한다 — 예컨대 벽이나 물체의 위치를 명세한다.

각 시점에서 두 에이전트는 행동 $a^{(1)}_t, a^{(2)}_t \in A$ 를 선택하고, 환경은 $s_{t+1} \sim T_\theta(\cdot \mid s_t, a^{(1)}_t, a^{(2)}_t)$ 에 따라 전이하며, 두 에이전트는 공유 보상 $R_\theta(s_t, a^{(1)}_t, a^{(2)}_t)$ 를 받는다. $\gamma \in [0,1]$ 은 할인계수이다. 궤적(trajectory) $\tau$ 를 $(s_0, a_0, \dots, s_T, a_T)$ 로 표기한다. 선행 연구(Carroll et al. 2019; Yan et al. 2023)와 마찬가지로, 우리는 ego agent를 고정 partner 정책 $\pi_p$ 와 짝지어 self-play로 협응 문제를 풀며 다음을 최적화한다:

$$J(\pi_{\text{ego}}, \pi_p) = \mathbb{E}\left[ \sum_{t=0}^{\infty} \gamma^t R_\theta(s_t, a^{(1)}_t, a^{(2)}_t) \right].$$

본 연구에서는 완전 명세된(fully specified; LBF, Overcooked-AI) 환경과 과소명세된(under-specified; 절차적 level 생성, 즉 OGC) 환경을 모두 다룬다. 완전 명세된 경우, 이 형식은 표준 2-에이전트 확률 게임으로 축소된다. 결정적으로, 이 관점은 과제($\theta$ 를 통해)와 파트너($\pi_p$ 를 통해)의 **결합 분포(joint distribution)** 에 대한 학습을 추론할 수 있게 한다.

### 3.2 Ad-hoc Teamwork

위의 과소명세 확률 게임 형식 안에서, AHT는 인간을 포함한(Stone et al. 2010) 미지의 partner 정책 집합 $\Pi_{\text{eval}}$ 에 대하여 $\pi_{\text{ego}}$ 를 최적화하는 문제에 해당한다. 목표는 다음을 찾는 것이다:

$$\pi^*_{\text{ego}} = \arg\max_{\pi_{\text{ego}}} \; \mathbb{E}_{\pi_p \sim \Pi_{\text{eval}}}[J(\pi_{\text{ego}}, \pi_p)].$$

$\Pi_{\text{eval}}$ 은 일반적으로 알려지지 않으므로 이 문제를 직접 풀 수 없다. 많은 접근법은 대신 전문가 도메인 지식을 사용하거나, 다양한 파트너 집합 $\Pi_{\text{train}}$ 을 학습하여 $\pi_{\text{ego}}$ 를 $\Pi_{\text{train}}$ 에 대한 best response로 학습한다: $\pi_{\text{ego}} = \arg\max_{\pi_{\text{ego}}} \mathbb{E}_{\pi_p \sim \Pi_{\text{train}}}[J(\pi_{\text{ego}}, \pi_p)]$. $\Pi_{\text{eval}}$ 과 $\Pi_{\text{train}}$ 은 보통 다르므로, 분포 이동(distribution shift) 때문에 $\pi_{\text{ego}}$ 가 $\Pi_{\text{eval}}$ 에 최적이 아닐 수 있다. 그러나 $\Pi_{\text{train}}$ 이 $\Pi_{\text{eval}}$ 을 충분히 근사하면 $\pi_{\text{ego}}$ 는 여전히 잘 동작할 수 있다. 이는 자연스럽게 2단계 과정을 구성한다: $\Pi_{\text{train}}$ 을 학습한 뒤 $\pi_{\text{ego}}$ 를 best response로 학습. $\Pi_{\text{train}}$ 을 얻는 비용은 상당한데, 전문가에게 질의하거나 partner를 학습시키는 것이 비싸기 때문이다. 예컨대 FCP는 서로 다른 무작위 초기화로 $N$ 개 정책을 RL로 학습해 $\Pi_{\text{train}}$ 을 구성하고, MEP는 추가로 이 $N$ 개 정책 간 상호작용을 고려한다. 하나의 정책을 끝까지 학습하는 비용이 $C$ 라면, population 기반 방법은 $O(NC)$ 차수(또는 정책 간 상호작용으로 인해 그 이상, 예: MEP)의 학습 비용을 치른다.

이에 반해 E3T는 population-free 접근법으로, 사전 구성된 $\Pi_{\text{train}}$ 없이 self-play로 $\pi_{\text{ego}}$ 를 효율적으로 학습한다. 이때 $\pi_p$ 를 $\pi_{\text{ego}}$ 와 무작위 정책 $\pi_r$ 의 혼합으로 계산한다:

$$\pi_p = \epsilon \pi_r + (1 - \epsilon)\pi_{\text{ego}}. \tag{1}$$

여기서 $\epsilon \in [0,1]$ 은 학습 기간 동안 고정된다. Yan et al. (2023)은 과제와 평가 설정에 따라 $\epsilon$ 을 수동으로 선택하는데, 이는 하이퍼파라미터 선택에 대한 민감성을 도입하고 서로 다른 학습 단계 전반의 적응성을 제한한다.

본 연구에서 우리는 (i) $\Pi_{\text{eval}}$ 은 모르지만 평가 협력 과제는 알려진 표준 AHT 문제($\theta_{\text{train}} = \theta_{\text{eval}}$)와, (ii) $\pi_{\text{ego}}$ 가 여러 평가 level에서 평가되고 각 level $\theta_{\text{eval}}$ 이 자신의 $\Pi_{\text{eval}}$ 을 갖는, 더 까다로운 *미지 level에서의 AHT*(Ruhdorfer et al. 2025b; Jha et al. 2025) 를 모두 고려한다.

> **Figure 2.** *개념적 도해:* 학습이 진행되며 ego 능력(competence)이 향상될 때(검정), E3T는 고정 혼합 계수(여기서 $\epsilon = 0.5$)로 파트너를 생성하여 ego 능력의 *고정된 비율*에 해당하는 파트너를 만든다(파랑). 이와 대조적으로 UPD는 $\epsilon \sim U(0,1)$ 로 샘플링하고(초록) learnability 기준으로 파트너를 필터링하여, 동적인 범위의 partner 능력을 만들어 낸다(주황).

### 3.3 Unsupervised Environment Design

단일 에이전트 RL에서, UED 알고리즘은 환경의 자유 파라미터 $\theta \in \Theta$ 와 효용 함수 $U$ 를 이용해 curriculum을 만든다. 많은 알고리즘이 regret을 UED 목적으로 사용하며(Dennis et al. 2020; Samvelyan et al. 2023), 현재 정책과 최적 정책의 성능 차이에 기반해 $\theta$ 를 선택한다: $U(\pi, \theta) = \text{REGRET}_\theta(\pi, \pi^*_\theta)$. 그러나 이는 최적 정책 $\pi^*_\theta$ 에 대한 접근을 가정한다. 따라서 최근 연구는 효용으로서의 regret에서 벗어났다. Sampling for learnability(SFL, Rutherford et al. 2024)는 에이전트의 학습 경계에 가까운 인스턴스를 우선시하는 learnability 함수로 level을 채점한다. $R(\tau, \theta) \in \{0,1\}$ 의 이진 결과에 대해, learnability는 $\ell_{sr}(\pi, \theta) = p(1-p)$ 로 정의되며 $p = \mathbb{E}_{\tau \sim p(\tau \mid \pi, \theta)}[R(\tau, \theta)]$ 는 해당 level에서의 성공률이다. Monette et al. (2025)은 이 아이디어를 평균 성능 주변의 return 분산에 가중을 둠으로써 연속 보상으로 확장하였다. 본 연구에서 우리는 이 아이디어를 level이 아니라 **partner** 에 적용하여, partner를 learnability에 기반해 생성·선택할 수 있는 학습 인스턴스로 취급한다.

---

## 4 Unsupervised Partner Design

**Unsupervised Partner Design** 은 협력 파트너를 learnability에 기반해 적응적으로 생성·선택하는 population-free 학습 접근법이다. 핵심 아이디어는, unsupervised environment design을 환경 파라미터가 아니라 **partner 정책으로 정의되는 유도 학습 환경(induced training environment)** 에 대해 적용하는 것이다. 선행 연구와 달리, UPD는 온라인 생성과 learnability 주도 선택을 통해 partner 행동에 대한 적응적 학습 분포를 구축하며(Figure 2), self-play로 동작하고 단일 학습 단계(single training stage)만을 요구한다.

> **Algorithm 1: Unsupervised Partner Design**
> **요구사항:** 환경 $G$, ego 정책 $\pi_{\text{ego}}$, partner generator $S_p$, 채점 rollout 수 $N$, rollout 길이 $L$, 갱신 빈도 $R$, buffer 크기 $|B|$, SFL 비율 $\rho$
> 1: 빈 buffer $B, B_{\text{temp}}$ 초기화; $t \leftarrow 0$
> 2: **while** 수렴하지 않음 **do**
> 3: &nbsp;&nbsp;**if** $t \bmod R = 0$ **then**
> 4: &nbsp;&nbsp;&nbsp;&nbsp;$B_{\text{temp}} \leftarrow \emptyset$ 으로 리셋 {여기서부터 병렬화}
> 5: &nbsp;&nbsp;&nbsp;&nbsp;**for** 원하는 각 partner **do**
> 6: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$(\pi_p, \theta) \sim S_p(\pi_{\text{ego}}) \times \Theta$ 샘플링 (Alg. 2)
> 7: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;쌍 $(\pi_{\text{ego}}, \pi_p)$ 에 대해 $G_\theta$ 에서 길이 $L$ 의 rollout $N$ 개를 얻고 return $R_1, \dots, R_N$ 수집
> 8: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\ell_{(\pi_{\text{ego}}, \pi_p, \theta)} = \ell(\pi_{\text{ego}}, \pi_p, \theta)$ 계산
> 9: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$(\ell_{(\pi_{\text{ego}}, \pi_p, \theta)}, \pi_p, \theta)$ 를 $B_{\text{temp}}$ 에 저장
> 10: &nbsp;&nbsp;&nbsp;&nbsp;**end for**
> 11: &nbsp;&nbsp;&nbsp;&nbsp;$B \leftarrow B_{\text{temp}}$ 에서 $\ell$ 이 가장 높은 상위 $|B|$ 개 항목
> 12: &nbsp;&nbsp;**end if**
> 13: &nbsp;&nbsp;$\ell$ 과 $\rho$ 를 사용해 $B$ 와 $S_p$ 로부터 $(\pi_p, \theta)$ 배치 샘플링
> 14: &nbsp;&nbsp;$G_{\pi_p, \theta}$ 에서 PPO로 $\pi_{\text{ego}}$ 갱신
> 15: &nbsp;&nbsp;$t \leftarrow t + 1$
> 16: **end while**

### 4.1 Curriculum over Partners

우리는 $\Pi_{\text{train}}$ 의 파트너가 — 그 비용 때문에 보통 소수의 파트너로 고정되므로 — 특정 시점의 학습에 최적이 아닐 수 있다는 관찰에서 출발한다. 대신 학습 전반에 걸쳐 $\pi_{\text{ego}}$ 에게 유용한 학습 인스턴스가 되는 파트너를 *생성*할 수 있다면 어떨까?

당분간 다양한 partner generator $\pi_p \sim S_p$ 와 고정 level $\theta$ 에 대한 접근을 가정하자. 위에서 도입한 확률 게임 형식 안에서, partner 정책 $\pi_p$ 를 고정하면 기저 게임의 co-player를 조건화함으로써 ego agent를 위한 학습 환경이 유도된다:

$$G_{\pi_p, \theta} := \langle S, A, T_{\pi_p, \theta}, R_{\pi_p, \theta}, \gamma, \rho_\theta \rangle, \tag{2}$$

$$T_{\pi_p, \theta}(s' \mid s, a^{(1)}) = \sum_{a^{(2)}} \pi_p(a^{(2)} \mid s)\, T_\theta(s' \mid s, a^{(1)}, a^{(2)}), \tag{3}$$

$$R_{\pi_p, \theta}(s, a^{(1)}) = \sum_{a^{(2)}} \pi_p(a^{(2)} \mid s)\, R_\theta(s, a^{(1)}, a^{(2)}). \tag{4}$$

따라서 $S_p$ 에서 partner 정책을 샘플링하는 것은 *유도 학습 환경에 대한 분포*를 정의한다. curriculum learning 관점에서 목표는 ego agent의 학습 진척을 유도할 가능성이 높은 파트너를 우선시하는 것이다. 우리는 각 파트너 $\pi_p$ 를 학습 인스턴스로 취급하고, rollout return에서 추정한 learnability 함수 $\ell(\pi_{\text{ego}}, \pi_p)$ 로 채점함으로써 이를 구현한다. 구체적으로, $S_p$ 를 사용해 여러 파트너를 샘플링하고 $G_{\pi_p, \theta}$ 에서 채점한다:

$$\ell_{\text{var}}(\pi_{\text{ego}}, \pi_p, \theta) = \mathrm{Var}_{\tau \sim G_{\pi_p, \theta}}[R(\tau)]. \tag{5}$$

낮은 return 분산은 일관된 실패 또는 성공을 가리키는 반면, 높은 분산은 협력이 때때로 성공하는 중간 난이도(intermediate difficulty)의 파트너에 대응한다. 중간 난이도 표본이 학습을 촉진함은 알려져 있다(Florensa et al. 2018; Tzannetos et al. 2023).

최근 연구는 learnability와 기대 정책 개선(expected policy improvement) 사이의 형식적 연결을 제공한다. 특히 Foster et al. (2026)은 PPO를 포함한 폭넓은 advantage 기반 정책 경사(policy gradient) 방법들에 대해, 정책의 기대 개선이 advantage를 형성하는 데 쓰이는 스칼라 학습 신호의 분산에 비례함을 보였다. 우리의 설정에서, partner 정책 $\pi_p$ 와 level $\theta$ 를 고정하면 ego agent를 위한 단일 에이전트 게임 $G_{\pi_p, \theta}$ 가 유도된다. Foster et al.의 결과를 이 유도 게임에 적용하면, $G_{\pi_p, \theta}$ 에서 학습할 때 $\pi_{\text{ego}}$ 의 기대 1-스텝 개선이 $\mathrm{Var}_{\tau \sim G_{\pi_p, \theta}}[R(\tau)]$ 에 따라 증가함을 함의한다. 따라서 return 분산은 ego agent에 정책 개선을 유도할 것으로 기대되는 파트너를 우선시하는 데 원리적(principled) 신호를 제공한다. 이러한 의미에서 UPD는 SFL의 curriculum 메커니즘을 level에서 partner 정책으로 확장한다.

### 4.2 Curriculum over Partners and Levels

과소명세 환경에서 강건한 협력을 위해서는 새로운 partner와 level 양쪽에 일반화하는 것이 핵심이다(Ruhdorfer et al. 2025b). 특히 UPD는 과소명세 게임에서 partner와 level 양쪽에 대한 결합 curriculum으로 쉽게 확장된다. 이 경우 $\ell$ 은 단순히 튜플 $(\pi_p, \theta) \sim S_p \times \Theta$ 를 샘플링하고 $\tau \sim p(\tau \mid \pi_{\text{ego}}, \pi_p, \theta)$ 를 얻어 계산한다. 이 형식의 한 가지 문제는 서로 다른 level $\theta$ 가 서로 다른 보상 범위를 유도할 수 있다는 점이다. 이를 해결하기 위해, 우리는 $\ell$ 을 보정하는 **coefficient-of-variation squared (CV²)** 점수를 사용한다:

$$\ell_{CV^2}(\pi_{\text{ego}}, \pi_p, \theta) = \frac{\mathrm{Var}_{\tau \sim (\pi_{\text{ego}}, \pi_p, \theta)}[R(\tau, \theta)]}{\left(\mathbb{E}_{\tau \sim (\pi_{\text{ego}}, \pi_p, \theta)}[R(\tau, \theta)] + \delta\right)^2}, \tag{6}$$

여기서 $\delta$ 는 작은 상수이다. 우리는 이 확장을 **Joint UPD (JUPD)** 라 부르며, 방법 개요(method sketch)는 Alg. 1에 제시한다.

### 4.3 Online Partner Generation

partner에 대한 curriculum을 구현하려면 후보 파트너를 샘플링할 partner generator $S_p$ 가 필요하다(Alg. 2 참조). 본 연구에서 우리는 E3T의 partner 생성 전략을 확장하여 능력(competence)과 행동(behavioural) 양 차원에 걸친 확률성을 도입한다. 다만 UPD는 *어떤* 형태의 partner generator와도 함께 쓸 수 있다.

E3T는 고정 혼합 계수 $\epsilon$ 에 의존하는데, 이는 학습 단계마다 차선(suboptimal)일 수 있다. 학습 초기에는 더 유능한 파트너가 과제 학습을 도울 수 있고, 후반에는 덜 예측 가능한 파트너가 도전과제를 제공한다. $\epsilon$ 을 고정하는 대신, UPD는 학습 전반에 걸쳐 $\epsilon \sim U(0,1)$ 을 샘플링하여 넓은 능력 범위를 아우르는 파트너를 생성한다.

능력 변동을 넘어, 협력 파트너는 흔히 체계적 행동 경향을 보인다. 예컨대 Overcooked의 인간 플레이어는 특정 이동 방향을 선호하거나 stay(머무름) 행동을 과용하는 등 지속적인 행동 선호를 드러낸다(Carroll et al. 2019; Yu et al. 2023). 이러한 저수준 편향을 포착하기 위해, UPD는 **bias masking** 메커니즘을 도입한다. 파트너를 생성할 때 우리는 이산 행동 공간에 대한 편향된 무작위 정책을 정의하는 지속적 편향 마스크 $m \sim \mathrm{Dir}(\alpha \cdot \mathbf{1}_A)$ 를 샘플링한다. Dirichlet 분포는 단일 파라미터 $\alpha$ 를 통해 그러한 편향의 강도를 제어하는 간단한 수단을 제공한다.

> **Algorithm 2: Partner Policy Generator $S_p$**
> **요구사항:** ego 정책 $\pi_{\text{ego}}$, 편향 확률 $p_{\text{bias}} = 0.5$
> 1: 혼합 파라미터 샘플링 $\epsilon \sim U(0,1)$
> 2: 확률 $p_{\text{bias}}$ 로:
> 3: &nbsp;&nbsp;지속적 편향 마스크 샘플링 $m \sim \mathrm{Dirichlet}(\alpha \cdot \mathbf{1}_A)$
> 4: 그렇지 않으면:
> 5: &nbsp;&nbsp;$m = \mathbf{1}_A / A$ 로 설정
> 6: 편향 마스크 $m$ 으로 편향된 무작위 정책 $\pi_{r,m}$ 구성
> 7: partner 정책 $\pi_p = \epsilon \pi_{r,m} + (1 - \epsilon)\pi_{\text{ego}}$ 반환

### 4.4 UPD and Convention Selection

협응 게임에서 self-play는 여러 동등한 균형(equilibria) 중 하나로 수렴하는데, 이는 높은 self-play 성능을 내지만 partner 일반화는 나쁘다(Carroll et al. 2019; Hu et al. 2020). 다음 $2 \times 2$ 행렬 게임을 보자:

$$\begin{array}{c|cc} & \text{up} & \text{down} \\ \hline \text{up} & 1 & 0 \\ \text{down} & 0 & 1 \end{array} \tag{7}$$

self-play는 하나의 균형만 선택한다(예: $\pi^{\text{up}}_{\text{SP}}$), 그러면

$$\mathbb{E}_{\tau \sim (\pi^{\text{up}}_{\text{SP}}, \pi^{\text{up}}_{\text{SP}})}[R(\tau)] = 1, \quad \mathrm{Var}_{\tau \sim (\pi^{\text{up}}_{\text{SP}}, \pi^{\text{up}}_{\text{SP}})}[R(\tau)] = 0. \tag{8}$$

이와 대조적으로, UPD의 partner generator $S_p$ 가 $0 < p(\text{up}) < 1$ 인 파트너 $\pi_p$ 를 적어도 하나 샘플링한다고 하자. 그러면

$$\mathbb{E}_{\tau \sim (\pi^{\text{up}}_{\text{SP}}, \pi_p)}[R(\tau)] < 1, \quad \mathrm{Var}_{\tau \sim (\pi^{\text{up}}_{\text{SP}}, \pi_p)}[R(\tau)] > 0, \tag{9}$$

이며 $\ell(\pi^{\text{up}}_{\text{SP}}, \pi_p) > \ell(\pi^{\text{up}}_{\text{SP}}, \pi^{\text{up}}_{\text{SP}})$ 이다. 이는 UPD가 이 관습을 완전히 공유하지 *않는* 파트너로 학습하도록 편향시킨다. 더 나아가, 이 게임에서 정책 $\pi^{\text{up}}_{\text{SP}}$ 와 $p = \pi_p(\text{up})$ 인 파트너 $\pi_p$ 에 대해 $\ell_{\text{var}}$ 은 $p = 0.5$ 에서 최대가 된다. 따라서 partner generator $S_p$ 가 $p = 0.5$ 근처에 지지(support)를 가지면, UPD는 최대로 모호한(maximally ambiguous) 파트너를 선택하여 현재 균형에서 벗어나는 경사 압력(gradient pressure)을 도입할 수 있으며, 이는 관습에 대한 과적합을 억제하면서 동시에 기대 정책 개선(§4.1)을 최대화한다.

> **각주:** 본 연구의 partner generator(Alg. 2)와 고정 $\pi^{\text{up}}_{\text{SP}}$ 에 대해, $\ell$ 은 $\epsilon = 1.0$ 과 $p_{\text{bias}} = \text{False}$ 에서 최대가 되며, 이는 $p(\text{up}) = 0.5$ 인 균등 무작위 파트너를 산출한다. 이 설정은 $S_p$ 에서 충분히 샘플링하면 기댓값적으로 발견된다. 반면 E3T는 $\epsilon$ 을 최적값으로 수동 설정해야 하는데, 이 예제에서는 가능하지만 일반적으로는 불가능하다. 게다가 최적 $\epsilon$ 이 과제 $\theta$ 에 의존하므로(이 예제처럼), E3T의 고정 전략은 level 전반에 최적 파트너를 제공하지 못한다.

---

## 5 Experiments in Fixed Environments (고정 환경 실험)

우리는 실험을 두 절로 나눈다. **이 절에서는** 절차적 생성이 없는, 즉 $\theta$ 가 고정된 환경(Level-Based Foraging과 Overcooked-AI)에서 UPD를 평가하여 AHT 방법으로서의 효과를 검증한다. §6에서는 절차적 생성 환경(OGC)으로 넘어가, 동일한 메커니즘이 partner와 level 전반의 결합 일반화로 확장되는지 평가한다. 본 논문은 총 **282개의 학습된 정책**을 평가한다.

### 5.1 Comparing UPD and E3T in LBF

먼저 현행 population-free 베이스라인인 E3T의 동작을 LBF(Albrecht & Ramamoorthy 2013)에서 UPD와 비교하여 분석한다. LBF는 AHT 연구에서 인기 있는 환경이다(Rahman et al. 2021; Papoudakis et al. 2021; Mirsky et al. 2022; Rahman et al. 2023; Wang et al. 2025). 우리는 (Bonnet et al. 2024; Wang et al. 2025)에서 사용된 버전에 기반하며, 여기서 두 에이전트가 협력하여 $7 \times 7$ 격자에서 세 개의 음식을 수집해야 하고, 각 음식은 두 에이전트가 모두 적재(loaded)되어야 한다. 게임은 100 스텝 후, 에이전트가 충돌할 때, 또는 모든 음식이 수집되었을 때 종료되며, 최대 return은 0.5이다. E3T는 LBF에 사용된 적이 없으므로, $\epsilon \in \{0.2, 0.3, \dots, 0.8\}$ 을 휩쓸어(sweep) 경험적으로 결정한다.

다양한 평가 population $\Pi_{\text{eval}}$ 을 구성하기 위해, Wang et al. (2025)과 유사하게 하드코딩 에이전트, 계획(planning) 모델, BRDiv(Rahman et al. 2023)로 학습된 에이전트를 결합한다. 우리는 BRDiv 에이전트 3개, 무작위 1개, 계획 에이전트 6개((역)열 우선, (역)사전식, (역)가까운 것부터 먼 것 순으로 음식 수집)로 총 10개의 평가 파트너를 사용한다. BRDiv population은 self-play를 최적화하면서 cross-play return을 최소화하여 유능하지만 호환되지 않는 전략을 채택한다. 이 파트너들이 모두 최적으로 행동하는 것은 아니며, 그들과의 최적 협력조차 최대 미만의 return을 낼 수 있음에 유의하라.

> **Figure 3.** 10개의 평가 파트너를 사용한 협력적 LBF에서의 평균 return. 막대는 mean ± 표준편차. UPD는 테스트된 모든 $\epsilon$ 에 걸쳐 E3T보다 높은 평균 return을 달성한다. (최대 0.5)

6개 시드, $10^7$ 타임스텝, 학습률 $1e{-}3$, 512개 환경(환경당 400 스텝)으로 학습하며, UPD는 $|B| = 64$, $R = 4$, $N = 5$, $p_{\text{bias}} = 0.5$, $\alpha = 1.0$ 을 사용한다. Figure 3은 테스트된 어떤 $\epsilon$ 도 최고 협력 성능을 내지 못하는 반면, UPD는 $\Pi_{\text{eval}}$ 에서 일관되게 더 높은 return을 낸다. 이는 E3T의 실질적 한계를 보여준다: $\epsilon$ 을 고정하면 학습 중 마주치는 partner 행동의 범위가 제한된다(cf. Figure 2). UPD는 이 한계를 해결하며, 그 적응적 partner 행동 범위가 가장 강력한 return을 제공한다. 실제로 더 큰 설정에서는 $\epsilon$ 휩쓸기가 실행 불가능할 수 있는 반면, UPD는 단지 미미한 추가 런타임(<10%; 부록 F에서 효율성을 특성화)만으로 이 튜닝을 완전히 회피한다.

### 5.2 Overcooked-AI with Artificial Partners

#### 5.2.1 실험 설정

더 풍부한 협응 동역학 하에서 UPD를 평가하기 위해, 우리는 다섯 개의 표준 레이아웃 — Cramped Room (CRoom), Asymmetric Advantages (AA), Coordination Ring (CR), Counter Circuit (CC), Forced Coordination (FC) — 을 갖는 Overcooked-AI(Carroll et al. 2019)를 고려한다. $\Pi_{\text{eval}}$ 생성은 LBF와 동일한 프로토콜을 따른다. 구체적으로 $\Pi_{\text{eval}}$ 은 (1) BRDiv 에이전트 3–4개, (2) 확률적 계획 에이전트(Wang et al. 2025), (3) 과제의 부분집합을 수행하는 하드코딩 에이전트(예: 양파만 담당하는 작업자)(Yu et al. 2023), (4) 저능력 random·stay 에이전트를 포함한다. CRoom은 9개, AA·CR·CC는 8개, FC는 제약적 동역학으로 실행 가능한 하드코딩 전략이 제한되어 4개를 사용한다.

우리는 Overcooked-AI에서 가장 널리·성공적으로 쓰이는 AHT 알고리즘들을 함께 아우르는 네 개의 베이스라인과 UPD를 비교한다: (1) IPPO(de Witt et al. 2020)로 학습된 self-play 베이스라인, (2) 보고된 하이퍼파라미터를 사용하는 E3T(CRoom·AA·CR·CC에 $\epsilon = 0.5$, FC에 $\epsilon = 0.0$), (3) population 크기 48의 MEP, (4) population 크기 48로 학습된 FCP. 경쟁력 있는 베이스라인을 위해 MEP는 원논문보다 큰 population을 사용하고(성능 향상; Yu et al. 2023), 필요 시 레이아웃별로 튜닝한다. 이와 대조적으로 UPD는 모든 레이아웃에 하나의 하이퍼파라미터 세트로 동작함을 확인하였다. 모든 방법에 동일한 recurrent 정책 네트워크를 사용했으며, E3T와 UPD에는 E3T가 제안한 partner model을 추가하였다. 학습은 512개 환경(환경당 400 스텝), 총 $5 \times 10^7$ 타임스텝, 첫 $3 \times 10^7$ 동안 reward shaping을 사용한다. UPD는 partner buffer 크기 $|B| = 512$, 파트너당 평가 rollout $N = 10$, Dirichlet 파라미터 $\alpha = 1.0$ 을 사용하고 네 번째 학습 루프마다 buffer를 갱신($R = 4$)한다. 6개 시드로 학습·보고하며, 모든 방법은 안정적으로 수렴한다(Figure 10–14 참조).

#### 5.2.2 결과

> **Table 1.** Overcooked-AI 평가 population에 대한 평균 return (mean ± std.). 두 시작 위치 모두에 대해 평균. 최고는 **굵게**, 두 번째는 밑줄. UPD가 종합적으로 베이스라인을 능가한다.

| Method | CRoom | AA | CR | CC | FC | Average | % Gain rel. to E3T |
|---|---|---|---|---|---|---|---|
| SP | 68.9 ± 5.6 | 64.4 ± 2.3 | 25.3 ± 4.5 | 14.5 ± 10.2 | 33.9 ± 8.3 | 41.4 ± 4.7 | −62.2% |
| FCP | 109.9 ± 5.8 | 117.4 ± 7.0 | 64.7 ± 3.0 | 30.8 ± 9.7 | 27.2 ± 4.8 | 70.0 ± 2.2 | −11.8% |
| MEP | 109.5 ± 5.3 | 142.8 ± 36.1 | 64.0 ± 2.2 | 13.7 ± 5.3 | 46.2 ± 6.1 | 75.3 ± 7.6 | −4.5% |
| E3T | _111.3 ± 7.8_ | 127.9 ± 22.1 | 58.4 ± 4.5 | 55.0 ± 2.4 | 41.3 ± 8.3 | 78.8 ± 12.5 | – |
| UPD w/o bias | **111.5 ± 2.3** | 159.9 ± 20.7 | 66.2 ± 3.8 | _57.1 ± 2.8_ | 44.4 ± 5.5 | 87.9 ± 3.6 | +10.9% |
| UPD w/o ℓ | 110.8 ± 3.1 | _164.0 ± 18.7_ | _68.9 ± 3.4_ | **64.5 ± 3.9** | _45.7 ± 3.8_ | _90.8 ± 6.0_ | +14.1% |
| **UPD (Ours)** | 107.5 ± 2.8 | **181.4 ± 6.4** | **69.2 ± 5.6** | **64.5 ± 1.3** | **48.7 ± 2.8** | **94.4 ± 2.3** | **+18.0%** |

Table 1은 UPD가 다양하고 미관측인 파트너와 짝지어졌을 때 가장 높은 평균 return을 달성하여, 이 평가에서 population 기반·population-free 베이스라인을 능가함을 보인다. 다른 방법들은 평균 return이 낮지만 CRoom에서는 UPD를 능가한다. 우리의 결론은 UPD가 모든 레이아웃을 지배한다는 것이 아니라, **하나의 population-free 구성**이 명시적 partner 학습을 피하면서도 강력한 population 기반 방법과 경쟁하거나 더 나음을 유지한다는 것이다.

또한 Table 1(하단)에서 두 ablation과 비교한다: **UPD w/o bias** 는 partner 혼합에 균등 무작위 정책 $\pi_r$ 을 사용하고, **UPD w/o ℓ** 은 learnability 채점을 제거하고 rollout마다 무작위 파트너를 샘플링한다. 각 변형은 여전히 경쟁력 있는 AHT 에이전트를 만들지만, 전체 UPD에서 결합했을 때 최고의 결과를 낸다. 흥미롭게도 UPD w/o ℓ 이 꽤 잘 동작하는데, 이는 online partner generator만으로도 넓은 스펙트럼의 파트너를 샘플링하여 자연스러운 curriculum을 유도하기 때문일 가능성이 크다. 그러나 UPD는 평균적으로, 특히 learnability의 영향이 가장 큰 AA에서 UPD w/o ℓ 을 능가한다. AHT 향상의 큰 몫은 대규모 partner 생성과 biasing에서 오며, learnability가 추가 이득을 제공하는 것으로 보인다. 종합하면 UPD는 평균적으로 모든 ablation을 능가하여, 동적이고 learnability 주도적인 curriculum의 이점을 부각한다.

#### 5.2.3 Partner 선택 동역학 분석

> **Figure 4.** 학습 동안 레이아웃별로 $\ell_{\text{var}}$ 이 어떻게 서로 다른 평균 $\epsilon$ 값을 선택하는지 비교(buffer 내 $\epsilon$ 값 평균). 더 좁은 협응 도전과제로 알려진 레이아웃(CC, CR, FC)에서 UPD는 더 작은 $\epsilon$ 을, CRoom과 AA에서는 더 높은 $\epsilon$ 을 선호한다.

UPD가 학습 중 파트너를 어떻게 선택하는지 분석한다. Figure 4는 UPD가 레이아웃과 시점에 따라 서로 다른 평균 $\epsilon$ 값을 선호함을 보인다. 더 좁은 협력 관습을 갖는 레이아웃(CR, CC, FC)에서 UPD는 낮은 $\epsilon$ 으로, 더 유연한 레이아웃(CRoom, AA)에서는 더 확률적인 파트너로 끌린다. **이는 learnability가 레이아웃과 시간에 따라 서로 다른 범위의 partner 능력을 선호하며, 어떤 단일 $\epsilon$ 도 최적이 아님을 시사한다.**

**UPD 하의 학습 동역학.** learnability의 역할을 더 조사하기 위해, 선행 연구(Rutherford et al. 2024; Monette et al. 2025)와 유사하게 모든 생성 파트너에 대해 learnability 대 return을 Figure 5에 도시한다. 대표 레이아웃 3개(CR, AA, FC)를 사용하고, 학습 40% 시점의 ego agent를 8,192개의 무작위 생성 파트너와 rollout하여 평가한다(CRoom·CC는 CR 사례와 매우 유사하게 행동하므로 대표성을 가짐). 세 가지 주요 현상을 관찰한다: (1) ego가 잘하는 파트너와 비교적 못하는 파트너 모두에서 learnability가 0이고, (2) learnability는 중간 난이도 파트너에서 가장 높으며, (3) learnability가 높은 파트너는 극소수다 — 대부분의 파트너는 0이거나 0에 가까운 learnability 점수를 받는다. 이는 §4.1의 동기와 부합한다: 대부분의 파트너가 똑같이 유익하지는 않으며, **UPD는 드문 중간 난이도 파트너를 일관되게 식별한다.**

> **Figure 5.** 세 대표 레이아웃(CR, AA, FC)에서 learnability 대 return. 각 점은 하나의 잠재적 파트너. 축의 막대 그래프는 각 return/learnability 구간의 파트너 수를 센다. UPD는 중간 난이도의 파트너를 식별한다.

> **Figure 6.** 학습에 걸쳐 partner 생성에 더해진 행동 편향. UPD가 **창발적 관습 깨기(emergent convention breaking)** 를 유도함을 발견한다 — 파트너가 처음에는 한 행동 쪽으로 편향되었다가 전환한다(예: Left↔Right, Up↔Down).

**창발적 관습 깨기.** Figure 6은 학습 동안 생성 파트너의 평균 방향성 행동 편향에서 체계적 전환을 보인다: 초기에는 파트너가 일관된 방향 선호(예: 오른쪽 선호)를 보이지만, 이후 반대 편향으로 전환한다. 이 행동은 레이아웃과 시드 전반에서 일관된다. 이러한 편향 전환은, ego agent의 현재 협응 관습을 위반하는 파트너가 더 높은 return 변동성을 유도하므로 learnability가 그러한 파트너를 선호한다는 점과 부합한다(§4.4 참조). 수작업 설계된 관습 깨기 메커니즘에 의존하는 선행 zero-shot 협응 접근법(Hu et al. 2020)과 달리, UPD는 어떤 명시적 관습 깨기 구성요소 없이도 유사한 동역학을 보인다.

### 5.3 Human-AI Experiments in Overcooked-AI

Overcooked-AI에 대한 마지막 실험으로, 이중 맹검(double-blind) 사용자 연구에서 UPD가 인간과 어떻게 동작하는지 평가하였다. 우리는 도전적인 협응 동역학을 갖는 세 레이아웃(AA, CR, CC)에서 네 개의 대표 에이전트(SP, MEP, E3T, UPD)를 평가하였다. 12명의 참가자(연령 22–31세, 여성 5명)를 모집했으며, 각자 모든 에이전트-레이아웃 조합을 무작위 순서로 플레이하였다. 총 144개 게임을 수집했고 방법당 36개 게임이 플레이되었다(게임당 200 스텝, (Jha et al. 2025) 준수). 시험에 앞서 튜토리얼 세션이 진행되었고, 기관 윤리 지침에 따라 보상이 제공되었다. 본 연구는 기관 윤리심의위원회의 승인을 받았다.

> **Figure 7.** 파트너에 대한 인간 평가: Frust = '답답한가?'(↓), Adapt = '잘 적응했는가?'(↑), Human = '인간 같은가?'(↑), Coord = '잘 협응했는가?'(↑). 개별 설문 질문에 단측 Wilcoxon signed-rank 검정을, return에 단측 대응표본 t-검정을 수행하여 UPD를 각 베이스라인과 비교. 유의수준은 세 비교를 고려해 Holm-Bonferroni 보정($* = p < 0.05$, $** = p < 0.01$, $*** = p < 0.001$).

Figure 7은 UPD가 인간-AI 상호작용에서 베이스라인보다 높은 평균 return을 달성하며(오른쪽), 대부분의 주관적 설문 항목에서도 선호됨(왼쪽)을 보인다: 참가자들은 UPD가 **유의미하게 더 적응적이고, 더 인간 같으며, 더 나은 협력자이고, 함께 일하기에 덜 답답하다**고 느꼈다. 전반적 주관적 선호를 평가하기 위해 설문 항목에 걸쳐 개별 응답을 집계하였다. 평정은 높은 내적 일관성(Cronbach's $\alpha = 0.916$)을 보여 합성 점수 사용을 정당화하였다. UPD는 다른 방법들보다 높은 집계 선호 점수를 받았으며 그 차이는 통계적으로 유의했다(Wilcoxon signed-rank, Holm-Bonferroni 보정, $p < 0.05$). 표본 크기를 고려할 때, 이 결과는 UPD를 전체 최고 성능 방법으로 확립하기보다는, **UPD가 인간과의 협력에 효과적**임을 입증한다.

---

## 6 Experiments with Procedural Generation (절차적 생성 실험)

UPD의 핵심 동기 중 하나는 UED와의 호환성으로, partner와 level 양쪽에 대한 결합 curriculum을 가능하게 한다는 점이다. 이를 시험하기 위해, 우리는 OGC(Ruhdorfer et al. 2025b) 내에서 UPD를 평가하였다. OGC는 level이 무작위로 생성되고 풀 수 있다는 보장이 없어 UED 메커니즘에 큰 부담을 주는, 특히 어려운 환경이다. OGC에서 $\Theta$ 는 벽, 물체, 에이전트의 위치를 변화시킨다. 처리 가능성을 위해 OGC의 5×5 버전을 사용하고, 1,024개의 병렬 환경으로 모든 방법을 $10^9$ 스텝 동안 학습하였다(절차적 level 생성이 유발하는 높은 분산으로 인해 OGC 수렴에 필요). 평가는 보류된 level-partner 조합과 보류된 레이아웃에서, Overcooked-AI 실험에 맞춘 조정된 평가 population으로 수행하였다.

고전적 AHT 방법은 절차적 생성 환경으로 확장되도록 설계되지 않았다(Ruhdorfer et al. 2025b; Jha et al. 2025). 이 때문에 우리는 level과 partner 생성을 결합하는 여러 베이스라인과 JUPD를 비교한다: (1) **DR-DR** 은 파트너와 level을 무작위로 샘플링한다. (2) **Cross-Environment-Cooperation (CEC)** 변용은 level을 무작위로 선택하고 self-play로 플레이한다. CEC는 학습 level generator가 평가 레이아웃과 밀접하게 일치할 때 잘 동작한다(Jha et al. 2025). (3) **SFLE3T** 는 문헌의 장점을 결합하여 SFL로 level을 선택하고 $\epsilon = 0.5$ 로 E3T 파트너를 만든다.

> **Table 2.** 5×5 Overcooked Generalisation Challenge에서 환경과 partner의 결합 비지도 일반화 결과 (mean ± std.). 최고는 **굵게**, 두 번째는 밑줄.

| Method | CRoom | CR | FC | Avg. |
|---|---|---|---|---|
| DR-DR | _86.4 ± 7.2_ | _53.6 ± 8.3_ | _9.7 ± 2.4_ | _49.9 ± 3.2_ |
| CEC | 41.4 ± 9.2 | 20.3 ± 7.0 | 12.7 ± 3.5 | 23.9 ± 9.7 |
| SFLE3T | 81.7 ± 7.8 | 41.1 ± 10.1 | 7.4 ± 2.5 | 44.0 ± 3.0 |
| **JUPD** | **97.0 ± 7.4** | **60.0 ± 6.1** | **17.1 ± 4.2** | **58.9 ± 4.4** |

Table 2에 보이듯, JUPD는 모든 평가 레이아웃에서 가장 높은 평균 return을 달성한다. partner 난이도를 적응시키지 않는 DR-DR·CEC나 정적 partner 생성에 의존하는 SFLE3T와 달리, JUPD는 학습 동안 적합한 partner와 level을 결합으로 선택한다. 시간에 따라 샘플링된 level은 Figure 8에 시각화되어 있다.

> **Figure 8.** 학습의 서로 다른 단계(Early / Middle / Final)에서 JUPD 학습 buffer로 샘플링된 level의 예.

---

## 7 Discussion and Limitations (논의 및 한계)

본 논문의 실험은 두 질문에서 출발하였다: 협력 파트너를 level 설계와 유사하게 저렴하고 적응적으로 설계할 수 있는가, 그리고 그러한 메커니즘이 partner-environment 결합 curriculum으로 자연스럽게 확장되는가.

우리의 목표가 결합 curriculum 설정을 위한 AHT 방법 설계이므로, 우리는 learnability를 통해 그리고 ego 정책에만 의존함으로써 절차적 생성 환경으로 깔끔하게 확장되는 partner generator에 초점을 둔다. 이 설정에서는 사전학습된 partner population이 미관측 환경 인스턴스로 전이되지 않을 수 있어, online partner 생성이 자연스러운 선택이 된다.

UPD w/o ℓ 이 이미 강력하게 동작하므로, 우리 연구의 한 가지 실질적 함의는 **E3T를 무작위화된 혼합 계수와 대규모 병렬 partner 생성으로 확장하면 표준 E3T보다 더 강한 population-free 베이스라인이 될 수 있다**는 점이다.

더 일반적으로, UPD는 partner 공간에 대해 동작하는 프레임워크로 이해될 수 있다. 우리의 실험에서는 SFL로 구체화하고 공간을 확률적 generator로 정의했지만, 원리적으로 다른 UED 방법을 쓸 수 있고 partner 공간을 partner population이나 잠재(latent) partner 공간(Liang et al. 2024)으로 정의할 수도 있다(이들이 가용할 때). 우리 프레임워크 안에서 더 풍부한 partner 공간을 탐구하는 것은 흥미로운 향후 연구이다. UPD는 명시적 population 사전학습을 피하지만, 계산을 대규모 online partner 평가 쪽으로 옮긴다. 실제로 이 트레이드오프는 JAX 환경처럼 고도로 벡터화된 시뮬레이터에서는 유리하나, 환경 상호작용이 비싼 설정에서는 덜 유리할 수 있다.

---

## 8 Conclusion (결론)

우리는 learnability 기반 partner 선택을 통해 ad-hoc teamwork 에이전트를 학습시키는 population-free 방법인 **Unsupervised Partner Design (UPD)** 을 도입하였다. 여러 벤치마크와 인간 연구에 걸쳐, UPD는 사전학습된 partner population이나 수동 파라미터 튜닝 없이 강력한 AHT 성능을 달성한다. 이 결과는 적응적 partner curriculum이 협력적 멀티에이전트 시스템의 일반화를 향상시키는 실용적이고 확장 가능한 접근법을 제공함을 시사한다.

---

## Impact Statement (영향 진술)

본 연구는 이전에 보지 못한 파트너, 특히 인간과 협력할 수 있는 에이전트의 학습 방법을 향상시키고자 한다. 우리의 기여는 통제된 연구 환경에서 평가되었지만, 향상된 human–AI 협응은 결국 실세계 상호작용 시스템에 적용될 수 있다. 여기서 human-AI 협응 실패나 정합 어긋남(misalignment)에서 비롯될 수 있는 배치 문제는 신중한 고려를 요한다. 이러한 우려는 우리의 접근법에 고유한 것이 아니며 상호작용·협력 AI 시스템 연구 전반에서 널리 공유된다. 우리는 본 연구가 human–AI 상호작용에 일반적으로 결부된 것 이상으로 특정한 새로운 윤리적 위험을 도입한다고 보지 않는다.

---

## Acknowledgements (감사의 글)

저자들은 익명 심사위원들의 피드백, 특히 ROTATE와의 추가 비교를 제안하고 UPD와 관습 선택에 관한 논의를 확장하도록 제안해 준 점에 감사한다. 또한 C. Ruhdorfer와 V. Oei를 지원해 준 International Max Planck Research School for Intelligent Systems (IMPRS-IS)에 감사한다.

---

## References (참고문헌)

> 참고문헌 목록은 원문(저자·제목·게재처)을 그대로 유지합니다. 원본 PDF의 References 절을 참조하세요. 본문에서 인용된 주요 문헌(약식): Stone et al. (2010, AHT); Albrecht & Ramamoorthy (2013, LBF/best-response); Dennis et al. (2020, UED/PAIRED); Jiang et al. (2021, DCD/REPAIRED); Jiang, Grefenstette & Rocktäschel (2021, PLR); Yan et al. (2023, E3T); Strouse et al. (2021, FCP); Zhao et al. (2023, MEP); Rahman et al. (2023, BRDiv); Carroll et al. (2019, Overcooked-AI); Ruhdorfer et al. (2025b, OGC); Rutherford et al. (2024, SFL); Monette et al. (2025); Foster et al. (2026, LILO); Wang et al. (2025, ROTATE); Jha et al. (2025, CEC); Chaudhary et al. (2025); Liang et al. (2024, generative partners); Hu et al. (2020, Other-Play); de Witt et al. (2020, IPPO); Schulman et al. (2017, PPO); Cho et al. (2014, GRU); Carvalho et al. (2025, NiceWebRL). 전체 서지정보는 원문을 따른다.

---

# Appendix (부록)

## A Infrastructure & Tools (인프라 및 도구)

우리는 94GB 메모리의 NVIDIA H100-NVL GPU와 AMD EPYC 9454 CPU를 갖춘 서버 시스템에서 실험을 수행하였다. 모든 학습 실행은 단일 GPU에서 이루어진다. 모델 학습에는 JAX(Bradbury et al. 2018), Flax(Heek et al. 2023), Optax(DeepMind et al. 2020)를 사용했고, 분석에는 NumPy(Harris et al. 2020), Pandas(Team 2020), SciPy(Virtanen et al. 2020), Matplotlib(Hunter 2007)을 사용하였다. 단일 파일 IPPO 구현은 JaxMARL(Rutherford et al. 2023)에 기반한다. FCP·E3T(및 확장으로 MEP·UPD) 구현의 기반은 Jha et al. (2025)의 공개 코드이며, ROTATE와 평가 파트너는 Wang et al. (2025)의 코드로 생성하였다. 우리의 curriculum learning 알고리즘은 SFL이며 Rutherford et al. (2024)의 오픈소스 릴리스에 기반한다.

우리의 실험은 위 시스템이 제공하는 계산 능력의 일부만 필요로 한다. 최소한 16GB 메모리 GPU(아마 그보다 적어도)로 재현 가능하다. 개별 Overcooked-AI 학습 실행은 방법에 따라 분~시간 단위로 걸리며, population 기반 베이스라인과 ROTATE는 사전학습/open-ended 반복으로 더 비싸다. OGC 실험은 보통 몇 시간이 걸리지만 하루 이내에 끝난다. human-AI 실험은 NiceWebRL(Carvalho et al. 2025)로 수행하였다.

## B Reproducibility Statement (재현성 진술)

연구 재현을 위해 본 문서에 모든 핵심 하이퍼파라미터를 제공한다. 프로젝트 페이지: https://git.hcics.simtech.uni-stuttgart.de/public-projects/UPD

> **⚠️ Transparency Note (투명성 노트):** 본 논문의 **이전 버전(v1)** 에는 베이스라인 실험에 영향을 주는 구현·설정 문제가 있었다. 영향을 받은 실험을 수정·재실행한 결과 **베이스라인 성능이 향상**되었으나, **논문의 주요 결론은 변하지 않았다.** (이것이 v2에서 Table 1의 베이스라인 수치가 바뀐 이유이다.)

## C UPD and Convention Selection Extended (관습 선택 확장)

여기서는 §4.4의 논의를 확장한다. §4.4는 learnability 정식화와 관습 선택이 어떻게 상호작용하는지 개념적으로 설명한다. 다음과 같은 행렬 게임을 가정하자:

$$\begin{array}{c|cc} & \text{up} & \text{down} \\ \hline \text{up} & 1 & 0 \\ \text{down} & 0 & 1 \end{array} \tag{10}$$

여기서 단순 self-play는 self-play 목적 $J_{SP}(\pi)$ 를 최대화하므로 하나의 균형으로 수렴하는데, 이 목적은 동등하게 좋지만 상호 배타적인 두 해 $\pi^{\text{up}}_{\text{SP}}$ 와 $\pi^{\text{down}}_{\text{SP}}$ (두 에이전트가 각각 항상 up 또는 down을 플레이)를 허용한다. 이들은 다음 때문에 상호 비호환적이다:

$$\mathbb{E}_{\tau \sim (\pi^{\text{up}}_{\text{SP}}, \pi^{\text{up}}_{\text{SP}})}[R(\tau)] = \mathbb{E}_{\tau \sim (\pi^{\text{down}}_{\text{SP}}, \pi^{\text{down}}_{\text{SP}})}[R(\tau)] = 1 \tag{11}$$

이지만

$$\mathbb{E}_{\tau \sim (\pi^{\text{up}}_{\text{SP}}, \pi^{\text{down}}_{\text{SP}})}[R(\tau)] = 0. \tag{12}$$

zero-shot 협응과 ad-hoc teamwork의 목표는 서로 다른 파트너에 일반화하고 특히 단일 균형에 과적합하지 않는 정책을 학습하는 것이다. UPD는 이를 두 상호작용 메커니즘으로 달성한다: 현재 $\pi_{\text{ego}}$ 에 기반해 파트너를 생성하고, 결과의 분산에 기반한 learnability 점수 $\ell$ 로 파트너를 선택한다. 논의를 위해, 이제 $\pi^{\text{up}}_{\text{SP}}$ 에 UPD 갱신 연산자를 적용한다고(즉 $\pi_{\text{ego}} = \pi^{\text{up}}_{\text{SP}}$) 하고, UPD가 $S_p$ 를 통해 세 개의 새 파트너를 만든다고 하자:

$$\pi^{\text{up}}_p = \{a \mid p(a) = 1 \text{ if } a = \text{up else } 0\} \tag{13}$$
$$\pi^{\text{down}}_p = \{a \mid p(a) = 1 \text{ if } a = \text{down else } 0\} \tag{14}$$
$$\pi^{\text{mix}}_p = \{a \mid p(a) = 0.5 \text{ if } a = \text{up else } 0.5\} \tag{15}$$

$\mathrm{Var}_{\tau \sim (\pi^{\text{up}}_{\text{SP}}, \pi^{\text{up}}_p)}[R(\tau)] = \mathrm{Var}_{\tau \sim (\pi^{\text{up}}_{\text{SP}}, \pi^{\text{down}}_p)}[R(\tau)] = 0$ 임은 명백하다. 이는 §4.1의 직관과 부합한다: 파트너로서 $\pi^{\text{up}}_p$ 는 너무 쉬운데, ego가 이미 그와 최적으로 동작하기 때문이다. 마찬가지로 $\mathbb{E}_{\tau \sim (\pi^{\text{up}}_{\text{SP}}, \pi^{\text{down}}_p)}[R(\tau)] = 0$ 이 모든 가능한 테스트 게임에서 성립하므로, $\pi^{\text{down}}_p$ 는 현재 비호환적이다. 여기서 UPD는 $\mathrm{Var}_{\tau \sim (\pi^{\text{up}}_{\text{SP}}, \pi^{\text{mix}}_p)}[R(\tau)] = 0.25 > 0$ 이므로 $\pi^{\text{mix}}_p$ 를 선택한다.

이제 $\pi^{\text{mix}}_p$ 로 학습한 후 ego 정책이 더 이상 결정론적으로 up을 선택하지 않고 0이 아닌 확률로 down도 플레이한다고 하자. 이 경우 $\pi^{\text{down}}_p$ 와의 상호작용은 더 이상 결정론적 실패를 내지 않고 0이 아닌 return 분산을 유도하기 시작한다. 결과적으로, $\ell_{\text{var}}$ 하에서 이전에는 정보가 없던(uninformative) 파트너가 나중에는 learnable한 학습 파트너가 될 수 있다.

이는 UPD의 중요한 성질을 보여준다: **파트너의 유용성은 정적이지 않으며 현재 ego 정책에 의존한다.** ego 정책이 학습에 따라 변하면 high-learnability 파트너의 집합도 변하여, 협응 관습에 대한 적응적 curriculum을 자연스럽게 유도한다.

**learnability 채점 없는 대규모 생성.** 흥미롭게도 이 예제는, 명시적 learnability 기반 필터링 없이도 대규모 online partner 생성만으로 이미 유용한 curriculum이 유도될 수 있는 이유도 시사한다. ego 정책이 진화하면서 무작위 생성 파트너들의 호환성과 협응 난이도가 자연스럽게 달라진다. 결과적으로 효과적 학습 상호작용의 분포가 시간에 따라 변하여, ego agent를 점진적으로 다른 협응 관습과 partner 행동에 노출시킨다. 따라서 learnability 기반 선택은 이미 적응적인 partner 생성 과정 안에서 특히 유익한 파트너를 우선시하는 메커니즘으로 해석될 수 있다.

## D Additional Comparison with ROTATE and fine-tuned UPD (ROTATE 및 fine-tuned UPD 추가 비교)

Table 3에서 우리는 ROTATE(Wang et al. 2025) 및 fine-tuned UPD와 추가 비교한다. ROTATE는 원논문에서 강력한 성능을 보이고 독립적으로 튜닝되었으므로 Wang et al. (2025)의 구현·하이퍼파라미터·네트워크 아키텍처를 그대로 사용한다(recurrent 정책 사용). 다른 방법은 6개 시드로 비교하나, ROTATE는 런타임 때문에 3개 시드로 보고한다. 또한 UPD가 레이아웃별 하이퍼파라미터로 이득을 얻을 수 있는지 보이기 위해, curriculum 파라미터만 레이아웃별로 fine-tune한 UPD 버전도 포함한다(Table 5; 3개 시드).

> **Table 3.** ROTATE 및 fine-tuned UPD와의 추가 비교 (Overcooked-AI, mean ± std., 두 시작 위치 평균). 최고는 **굵게**, 두 번째는 밑줄.

| Method | CRoom | AA | CR | CC | FC | Average | % Gain rel. to E3T |
|---|---|---|---|---|---|---|---|
| ROTATE | **119.9 ± 7.9** | 147.7 ± 22.6 | **80.5 ± 7.2** | _58.1 ± 0.1_ | **52.1 ± 5.1** | 91.7 ± 6.8 | +15.1% |
| UPD (Ours) | 107.5 ± 2.8 | **181.4 ± 6.4** | 69.2 ± 5.6 | **64.5 ± 1.3** | 48.7 ± 2.8 | _94.4 ± 2.3_ | _+18.0%_ |
| **UPD (fine-tuned)** | _111.6 ± 7.7_ | _201.8 ± 3.0_ | _79.6 ± 5.1_ | _66.7 ± 1.1_ | _51.3 ± 1.3_ | **102.3 ± 0.6** | **+25.9%** |

UPD(fine-tuned)가 가장 높은 평균 return을, 그다음 UPD, 그다음 ROTATE가 뒤따른다. ROTATE는 세 레이아웃(CRoom, CR, FC)에서 가장 좋지만 일부 오차막대는 UPD(FC)나 fine-tuned UPD(CR·FC)와 겹친다. 결론은, UPD가 단순함에도 강력한 현대 AHT 방법과 경쟁할 수 있으며, 공유 하이퍼파라미터 구성으로도 잘 동작하지만 레이아웃별 튜닝으로도 이득을 볼 수 있다는 점이다.

## E Additional Training Details (추가 학습 세부사항)

### E.1 Reinforcement Learning Details

본문의 모든 방법에 independent PPO(de Witt et al. 2020; Schulman et al. 2017)를 사용한다. 모든 하이퍼파라미터 개요는 Table 4·Table 6에, 탐색 범위는 Table 7에 제시한다. UPD는 SFL을 partner 공간으로 확장하므로 Rutherford et al. (2024)이 도입한 여러 하이퍼파라미터를 상속한다: (1) **# generated agents** 는 learnability 채점을 위해 생성되는 파트너 수, (2) **buffer size** 는 buffer에 채택되는 에이전트 수, (3) **$N$** 은 채점 게임 수, (4) **$R$** 은 buffer 갱신 빈도, (5) **$\rho$** 는 각 학습 반복에서 buffer로부터 뽑는 파트너 대 새로 생성하는 파트너의 비율을 결정한다. $\rho$ 에 대해 Rutherford et al. (2024)은 기본값 0.5를 권장한다.

베이스라인 SP·FCP·MEP·E3T·ROTATE는 여러 오픈소스 구현에 기반한다. FCP·E3T는 (Jha et al. 2025), ROTATE는 (Wang et al. 2025)를, MEP는 FCP와 Zhao et al. (2023)을 채택한다. SP-IPPO는 공유 PPO 하이퍼파라미터를 쓰되 value 함수 계수 0.5, 학습률 $2.5 \times 10^{-4}$, PPO epoch 4를 사용한다. 기본 구성은 표준 self-play에서 불안정하고 AHT 평가에서 상당히 나빠(인공 파트너 평균 성능 32.4 대 41.4), 더 강한 구성을 보고한다. FCP·MEP의 학습 population은 MLP 에이전트를 쓰며, 16개 시드를 학습하고 시드당 3개 체크포인트(초기화·최종 보상의 50%·최종 성능)를 추출해 population 크기 48을 만든다. MEP는 레이아웃별로 population 생성 하이퍼파라미터 조정이 필요할 때가 있어 레이아웃별 튜닝하였다.

OGC 실험은 두 가지 차이를 제외하면 동일한 하이퍼파라미터를 쓴다: actor·critic 레이어를 각각 1개씩 추가(각 5개)하고 1,024개 환경에서 $1 \times 10^9$ 타임스텝 학습. SFL 기반 OGC 방법(SFLE3T, JUPD)은 8192개 생성 level, $N = 5$, $\rho = 1.0$, 더 큰 buffer(4096, 16,384)를 쓴다. JUPD는 생성 level당 12개 파트너를 만들어 쌍을 채점하므로 더 큰 buffer가 필요하고 $R = 2$ 를 쓴다.

> **Table 4.** Overcooked-AI 실험 기본 하이퍼파라미터.

| 범주 | 값 |
|---|---|
| 환경 수 | 512 |
| 총 타임스텝 | $5 \times 10^7$ |
| Reward shaping horizon | $3 \times 10^7$ |
| 학습률 | $1 \times 10^{-3}$ |
| 학습률 annealing | Linear |
| 사용 시드 | 0 – 5 |
| **PPO** | |
| PPO rollout 길이 | 400 스텝 |
| PPO epochs | 6 |
| 갱신당 minibatch 수 | 8 |
| 할인계수 ($\gamma$) | 0.99 |
| GAE 파라미터 ($\lambda$) | 0.95 |
| Clipping 임계값 | 0.2 |
| 엔트로피 계수 | 0.01 |
| Value loss 계수 | 1.0 |
| Gradient norm clipping | 0.5 |
| **아키텍처** (actor·critic 공유) | |
| Embedding 레이어 | 2 |
| Actor 레이어 | 4 |
| Critic 레이어 | 4 |
| Fully connected 레이어 크기 | 256 |
| GRU hidden 크기 | 256 |
| 활성화 | Tanh |
| Layer normalisation | Enabled |
| **Partner modelling** (E3T/UPD only) | |
| 보조 모델 깊이 | 4 레이어 (크기 64) |
| MOA loss 계수 | 1.0 |
| Trajectory history 길이 | 5 스텝 |
| Action embedding 크기 | 256 |
| Prediction normalisation | L2 norm |
| **UPD 전용** | |
| # generated agents (SFL batch size) | 4000 |
| Buffer 크기 ($\|B\|$) | 512 |
| $N$ | 10 |
| $R$ | 4 |
| Sample ratio $\rho$ | 0.5 |

> **Table 5.** UPD fine-tuned 하이퍼파라미터(레이아웃별 curriculum 파라미터로 얼마나 이득을 보는지 평가).

| 범주 | CRoom FT | AA FT | CR FT | CC FT | FC FT |
|---|---|---|---|---|---|
| Buffer 크기 ($\|B\|$) | 1024 | 1024 | 128 | 1024 | 128 |
| $N$ | 5 | 10 | 5 | 10 | 5 |
| $R$ | 4 | 8 | 1 | 1 | 1 |
| # sampled partners | 128 | 128 | 512 | 128 | 512 |

> **Table 6.** ROTATE 하이퍼파라미터는 (Wang et al. 2025)의 보고값과 코드 릴리스를 그대로 사용(원논문에서 강력한 성능을 보였기에 재사용). 주요 항목: 환경 16개, PPO rollout 400, reward shaping Full, $\gamma=0.99$, $\lambda=0.95$, 시드 20374–20376. Open-Ended 반복(CRoom/AA 30, CR/CC/FC 20·20·20), Regret SP Weight(CRoom 3.0, 그 외 2.0), 그리고 teammate·ego별 PPO 하이퍼파라미터(레이아웃별 상이). 전체 표는 원문 참조.

> **Table 7.** AHT 방법의 학습 하이퍼파라미터 탐색 공간. 선택값은 **굵게**(AHT 방법 간 공유). 어떤 파라미터를 유지할지는 저렴한 대리지표로서 BRDiv population 평가 결과를 보고 결정. OGC 실험은 위 굵은 값을 재사용하고 환경(1024)·스텝($1\times10^9$)·actor·critic 레이어(각 1개 추가)만 늘림. CV² 사용 시 안정 상수 $\delta = 10^{-8}$. LBF는 본문에 언급된 값을 제외하면 Overcooked-AI와 동일.

| 범주 | 하이퍼파라미터 | 범위 |
|---|---|---|
| | 총 타임스텝 | $5 \times 10^7$ |
| | Reward shaping | $3 \times 10^7$ |
| | 학습률 | $1 \times 10^{-3}$, $3 \times 10^{-4}$, $1 \times 10^{-4}$ |
| | PPO epochs | 4, **6** |
| | PPO minibatches | 4, 6, **8** |
| | Layernorm? | **True**, False |
| **UPD 전용** | Buffer 크기 ($\|B\|$) | 128, 256, **512** |
| | $N$ | 3, **10** |
| | Buffer refresh freq. | 1, 2, **4** |

**Partner Populations.** 인공 평가 파트너 생성에는 Wang et al. (2025)의 구현을 재사용한다. BRDiv population은 레이아웃별로 강하지만 비호환적이도록 튜닝되었다(높은 self-play·낮은 cross-play return). 하드코딩 파트너는 독립적으로 행동하거나 전체 과제의 부분 과제만 기계적으로 수행한다. FC(제약적 동역학)에서는 BRDiv와 독립 파트너만 사용한다. 독립 파트너는 FC 과제를 혼자 풀 수 없으므로 60% 확률로 근처 카운터에 양파나 접시를 떨어뜨리도록 프로그램되어 있다(선행 연구 Wang et al. 2025와 일치).

### E.2 Neural Network Architecture

Table 1의 모든 방법에 recurrent actor-critic 아키텍처를 사용한다. 모델은 공유 encoder, recurrent 처리 모듈, 정책·가치 추정을 위한 분리된 head로 구성된다. 각 에이전트의 관측은 선형 레이어와 설정 가능한 수의 fully connected 레이어(기본 2개)로 이루어진 feedforward encoder를 통과한다(각 256 unit, ReLU 또는 Tanh, 선택적 layer norm). 시간적 의존성은 hidden state 256의 GRU(Cho et al. 2014)로 포착하며, hidden state는 종단 상태에서 리셋되고 GRU 출력은 actor·critic head에 공유된다. policy head는 각 256 unit의 fully connected 4레이어와 비선형 활성화를 거쳐 마지막 선형 레이어가 이산 행동 logit을 출력한다. value head도 동일한 recurrent embedding에 각 256 unit의 4레이어를 적용해 스칼라 가치를 낸다.

E3T가 제안한 **other agent modelling** 네트워크를 사용한다. 이 보조 모듈은 팀메이트 행동을 모델링하며, 각 에이전트는 상대의 지난 5개 상태-행동 쌍을 받는다. 관측·행동은 embedding되어 4레이어 MLP(레이어당 64 unit)를 거쳐 팀메이트의 다음 행동 분포를 예측하고, 예측은 L2 정규화되어 자신의 embedding과 연결된 뒤 policy head에 입력된다. 이 보조 손실은 cross-entropy로 최적화되고 튜닝 가능한 계수로 가중된다(Yan et al. 2023과 일치). ROTATE는 보고된 하이퍼파라미터에 맞추기 위해 (Wang et al. 2025)의 아키텍처를 사용한다.

## F Observations on Computational Requirements of UPD (UPD 계산 요구량에 대한 관찰)

우리는 population 기반 접근법과 UPD의 계산 요구량을 비교한다. 아래의 보수적 가정 하에서, 우리의 설정과 합리적으로 큰 population 크기($n \geq 3$)에 대해 UPD가 더 낮은 계산 요구량을 가질 수 있음을 보인다. 실제로 JAX 기반 구현에서는 UPD·E3T·best response(MEP·FCP) 학습 실행이 비슷하게 빠르지만, population 사전학습이 MEP·FCP의 런타임을 지배하며 population 크기에 따라 증가한다. 우리 실험에서 UPD는 self-play에 가까운 속도로 실행되는데, 정책 경사 갱신이 환경 스텝보다 비용을 지배하는 현대 JAX 기반 시뮬레이터에서는 그렇다.

PPO 종단간 학습에는 두 주요 비용이 있다: 환경 전이 비용 $C_{\text{env}}$ 와 back-propagation을 통한 PPO 정책 갱신 비용 $C_{\text{PPO}}$. $E$ 를 병렬 환경 수, $H$ 를 루프당 환경당 *학습* 스텝 수라 하면, 루프당 총 학습 배치는 $N_{\text{steps}} := EH$ 이다. UPD에서 partner 채점은 후보당 길이 $L$ 의 평가 rollout $N$ 개를 쓴다($L$ 은 $H$ 와 같을 필요 없음).

크기 $n = |P|$ 의 population에 대해, population을 학습한 뒤 best response를 학습하면:

$$\text{Cost}_{\text{2 Stage}} \approx (n+1) N_{\text{loops}} (N_{\text{steps}} C_{\text{env}} + C_{\text{PPO}}). \tag{16}$$

이에 반해 UPD는 population 학습을 제거하고 partner 채점 항을 더한다. $K$ 를 생성 파트너 수라 하면, 파트너 생성은 사실상 무료이므로($C_{\text{env}}$ 대비 단지 몇 파라미터 샘플링), $R$ PPO 루프마다 top-$|B|$ buffer를 갱신하는 비용은 $\text{EnvSteps}_{\text{UPD}} \approx N_{\text{loops}} KNL / R$ (식 17)이다. 그러나 후보 파트너가 logit의 단순 섭동(perturbation)으로만 다르므로 $K$ 개 후보를 병렬 평가하여($K$ 개 시뮬레이터), PPO 루프당 오버헤드는 $KNL/R$ 이 아니라

$$\text{Overhead}_{\text{UPD}} \approx \frac{NL}{R} \tag{18}$$

으로 스케일한다. 따라서 UPD의 루프당 wall-clock 비용은 표준 PPO 루프에 튜닝 가능한 채점 항을 더한 것이다:

$$\text{Cost}_{\text{UPD}} \approx N_{\text{loops}}\left( H C_{\text{env}} + C_{\text{PPO}} + \frac{NL}{R} C_{\text{env}} \right). \tag{19}$$

population 방법이 UPD보다 비싼 손익분기 population 크기는 식 (20)을 $n$ 에 대해 풀어

$$n^\star = \frac{NL/R}{H + C_{\text{PPO}}/C_{\text{env}}} \tag{21}$$

으로 주어진다. 환경 전이 비용이 PPO 갱신 비용을 지배하는 극한($C_{\text{PPO}}/C_{\text{env}} \to 0$)에서는 $n^\star = (NL/R)/H$ (식 22). 어떤 비자명한 population($n \geq 1$)보다 싸려면 $NL/R \leq H$ 이면 충분하고, 일반적으로 population 크기 $n$ 을 밑돌려면 $NL/R \leq nH$ 이면 충분하다.

우리 Overcooked-AI 실행에서는 $E = 512$, $H = 400$($N_{\text{steps}} = 204{,}800$), $N = 10$, $L = 400$, $R = 4$, $K = 4000$ 이며, 분할상환 채점 예산 $NL/R = 1000$ 이다. 따라서 $n^\star = 1000/(400 + C_{\text{PPO}}/C_{\text{env}})$ (식 23). 두 함의: 첫째, $C_{\text{PPO}}/C_{\text{env}} > 600$ 이면 $n^\star < 1$ 로, 단일 파트너 population조차 UPD의 wall-clock 오버헤드를 초과한다. 둘째, 시뮬레이션 지배 극한($C_{\text{PPO}}/C_{\text{env}} \to 0$)에서 $n^\star = 1000/400 = 2.5$ 로, 실용적인 $n \geq 3$ population은 UPD보다 비싸다. 우리는 >100000 steps/sec로 실행되는 JAX 시뮬레이터를 쓰므로 PPO 지배 영역에 있으며, population 구성원 간 상호작용을 계산하는 방법은 비용이 더 높다.

위는 또한 population 구성원이 병렬 학습되지 않음을 가정한다. 만약 $n$ 개 구성원이 완전 병렬 학습될 수 있다면(대부분 방법에서는 불가), population 사전학습의 wall-clock 비용은 더 이상 $n$ 에 비례하지 않는다. 이 최선의 병렬 설정에서도 2단계 방법은 사전학습→best-response의 두 순차 단계를 요하므로 $\text{Cost}_{\text{2 Stage, parallel}} \approx 2 N_{\text{loops}}(H C_{\text{env}} + C_{\text{PPO}})$ (식 24). UPD는 $C_{\text{PPO}}/C_{\text{env}} > NL/R - H$ (식 26)일 때 더 싸며, 우리 값($N=10, L=400, R=4, H=400$)에서 임계값은 $C_{\text{PPO}}/C_{\text{env}} > 600$ (식 27)이다. 이 조건은 우리 설정에서 현실적이고 보수적일 가능성이 크다(경량 JAX 시뮬레이터 + 전체 PPO 최적화 비용). 경험적으로 UPD는 self-play/E3T에 가까운 속도로, population 기반 방법은 사전학습(및 MEP의 population 상호작용 비용)으로 더 느리게 실행된다.

## G Additional Environment Details (추가 환경 세부사항)

### G.1 Level-Based Foraging

본문대로, 우리는 먼저 E3T와 UPD를 비교하기 위해 Level-Based Foraging(LBF; Albrecht & Ramamoorthy 2013)을 사용한다. 우리 설정의 LBF는 $7 \times 7$ 격자, 음식 3개, 에이전트 2개를 쓰며 두 에이전트 모두 전체 환경을 관측한다. 음식을 수집하려면 두 에이전트가 동시에 그 음식에 load 행동을 해야 한다. 행동은 up, down, left, right, no-op, load이다. 리셋 시 음식과 에이전트 위치가 무작위화된다. Bonnet et al. (2024)의 JAX 버전(Wang et al. 2025의 개선 포함)과, 평가 파트너에 대한 Wang et al. (2025)의 partner 생성 파이프라인을 사용한다.

### G.2 Overcooked-AI

Overcooked-AI는 Overcooked 비디오 게임에 기반한 멀티에이전트 협응 벤치마크로, 원래 Carroll et al. (2019)이 제안하였다. 시간적 협응과 공간적 추론을 요구하는 2인용 요리 과제이며, 에이전트는 재료를 집어 냄비에 조리하고 요리를 서빙해야 한다. 행동 공간은 이산적으로 6개 행동(상·하·좌·우 이동, interact, stay)이다. JaxMARL 버전(Rutherford et al. 2023, Wang et al. 2025의 개선 포함)을 사용한다.

AHT 문헌에서 흔한 다섯 표준 레이아웃을 사용한다(Figure 9): CRoom, AA, CR, CC, FC. JaxMARL 구현을 쓰고 모든 보상을 기본값으로 유지한다(서빙에 sparse, 표시된 중간 행동에 shaped). shaped reward 단계 동안 냄비에 재료를 넣으면 3, 수프 조리 중 접시를 집으면 3, 준비된 수프를 집으면 5의 보상을 받는다.

> **Figure 9.** Carroll et al. (2019)의 다섯 평가 레이아웃을 JaxMARL 시각화 파이프라인으로 재작도. 왼쪽부터 CRoom, AA, CR, CC, FC.

### G.3 The Overcooked Generalisation Challenge

OGC(Ruhdorfer et al. 2025b)는 Overcooked-AI를 확장하여 partner와 level 분포 양쪽에 걸친 zero-shot 일반화를 평가한다. 고정 레이아웃 대신, 다양한 위상·난이도의 무작위 주방을 만드는 절차적 level generator를 포함한다. 많은 생성 레이아웃이 풀 수 없거나 효율적 협응에 비자명한 관습을 요구하여 특히 까다롭다. 우리는 계산 비용을 줄이면서 과제 다양성을 보존하기 위해 5×5 버전을 사용한다. generator는 주방 구조(벽·카운터·아이템 배치)와 목표 설정(예: 수프당 양파 수)을 무작위 샘플링한다: 먼저 경계 벽 레이아웃을 만들고 1–10의 벽 예산을 샘플링해 벽을 무작위 배치하며, 레이아웃을 좁히기 위해 분할 벽이나 측면 벽을 추가하고, 벽 위에 아이템을 놓은 뒤 두 에이전트를 빈 타일에 배치한다. 우리는 OGC 벤치마크의 작은 레이아웃 고정 부분집합에서 평가하고, 이 생성 환경 전반에서 보류된 에이전트와의 평균 return으로 비교한다.

## H Training Curves (학습 곡선)

모든 Overcooked-AI 방법의 학습 return을 Figure 10(SP), 11(FCP), 12(MEP), 13(E3T), 14(UPD)에 표시한다. 이 return들은 주의해서 해석해야 한다: FCP·MEP는 각자의 population과의 return을, UPD·E3T는 생성 파트너에 기반한 학습 return을 보인다. 모든 보고 구성은 안정적으로 수렴한다. 많은 방법이 UPD보다 높은 학습 return을 내지만, 문헌에서 확립되었듯(Carroll et al. 2019) 이것이 미지 파트너와의 테스트 return을 반드시 예측하지는 않는다.

Figure 15에 네 OGC 방법의 학습 return을 표시한다. 모든 방법이 수렴하며, CEC만 끝에서 성능 하락을 보인다. 다시, 학습 성능은 테스트 시점 성능을 예측하지 않는다: learnability로 level을 샘플링하는 방법(SFLE3T, JUPD)은 그렇지 않은 방법(DR, CEC)보다 더 어려운 level을 샘플링할 수 있고, SFLE3T·JUPD 모두 self-play로 플레이하지 않는다.

> **Figure 10–14.** 각각 SP·FCP·MEP·E3T·UPD의 학습 곡선(6개 시드 평균 ± 표준편차, 레이아웃별).
> **Figure 15.** OGC 학습 곡선(DR, CEC, SFLE3T, JUPD; 6개 시드 평균 ± 표준편차).
> **Figure 16–18.** 각각 Coordination Ring·Asymmetric Advantages·Forced Coordination에서 학습에 걸친 learnability 대 return(Early / Middle / Final).
> **Figure 19.** 여러 learnability 함수의 평가 population 대비 성능(레이아웃 평균). $\ell_{\text{mean}}$(평균 return 기반), $\ell_{\text{gauss}}$(전역 평균에서 먼 파트너 하향 가중; Monette et al. 2025), adaptive-SR(중앙값 초과 여부로 순위), $\ell_{\text{var}}$. 단순한 $\ell_{\text{var}}$ 이 더 복잡한 측도와 동등하거나 더 낫다.

## I Partner Curriculum Dynamics (Partner Curriculum 동역학)

UPD가 Overcooked-AI 학습 전반에서 효과적 학습 파트너를 어떻게 식별하는지 더 잘 이해하기 위해, 실험 절의 분석을 반복하여 학습의 여러 시점에서 learnability 점수를 시각화한다. 세 레이아웃(CR, AA, FC)에 대해 학습된 ego agent를 세 시점(2/5, 4/5, 끝)에서 8,192개의 무작위 생성 파트너와 rollout하여 평가한다(Figure 16, 17, 18). 이 세 레이아웃은 대표적이다(CRoom·CC는 CR과 매우 유사). 학습 때와 동일한 분산 기반 지표로 learnability를 계산해 대응 평균 return에 대해 도시한다. 모든 설정에서 learnability는 최고·최저 return 파트너에 집중되지 않고, 대부분의 파트너가 위치한 곳에서 정점을 이루지도 않으며, 중간 난이도 파트너에서 높다. 대부분의 level에서 생성 파트너는 낮은 return·낮은 learnability 버킷에 속한다. 이는 *모든 파트너가 학습에 최적이 아니다* 라는 가설이 옳음을 시사한다: UPD는 소수의 파트너만 분포하는 중간 return 버킷을 가장 높게 순위 매긴다.

마지막으로, 이진 결과의 level-space curriculum에 관한 선행 연구(Rutherford et al. 2024)와 달리, 우리 설정에서는 에이전트가 어떤 파트너와도 보상 0을 받는 경우가 드물다. 이 점과 연속 보상 구조로 인해, learnability 분석 도표는 그 *형태*뿐 아니라 시간에 따른 *오른쪽 이동*을 통해서도 정보를 전달한다. 예컨대 Figure 17에서 최저 점수 파트너는 평균 return이 약 20에서 120으로 향상된다. 이 이동은 ego agent의 능력이 시간에 따라 확장되며 UPD가 점점 더 유능한 파트너를 샘플링함으로써 적응함을 시사한다. UPD는 정보가 풍부한 파트너를 식별할 뿐 아니라 에이전트의 학습 진척을 추적하여 curriculum을 조정한다. 주목할 예외는 FC처럼 강한 상호의존성을 갖는 과제에서 발생한다(Figure 18).

## J Additional Results in Overcooked (Overcooked 추가 결과)

### J.1 대안적 learnability 함수가 결과를 개선하는가?

우리는 서로 다른 learnability 함수가 partner 선택과 학습 성능에 어떻게 영향을 미치는지 조사하였다: (1) 우리의 분산 기반 점수 $\ell_{\text{var}}$ 이 효과적 curriculum을 유도하는가? (2) UPD는 learnability 함수 선택에 얼마나 민감한가? RQ2의 다섯 Overcooked-AI 레이아웃에 대해 네 함수를 비교하였다: (1) $\ell_{\text{mean}} = \mathbb{E}_{\tau \sim (\pi, \pi_p)}[R(\tau)]$ — 더 높은 기대 return 파트너 선택. (2) Rutherford et al. (2024)의 성공률 기반 learnability를 연속 보상으로 변용한 것으로, return을 중앙값에서 임계화한 추정 의사-성공률:

$$\ell_{\text{adaptive-SR}} = p \cdot (1 - p), \tag{28}$$
$$\text{여기서 } p \text{ 는 전역 중앙값 return을 초과하는 rollout return의 비율.} \tag{29}$$

(3) (Monette et al. 2025)의 Gauss 가중 정식화 $\ell_{\text{gauss}}$. (4) 우리의 $\ell_{\text{var}}$.

Figure 19에 보이듯 $\ell_{\text{var}}$ 이 가장 높은 성능을 달성한다. 특히 UPD는 병리적 경우(예: self-play)를 피하는 한 대안 함수들 전반에서 강건하다. 선행 결과(Monette et al. 2025)와 달리, 우리는 더 복잡한 learnability 함수가 UPD에 명확한 이점을 제공하지 않음을 관찰한다.

우리 실험은 UPD가 AHT 에이전트 학습에 효과적이며, self-play로 붕괴하지 않는 한 서로 다른 learnability 정식화에 강건함을 보였다. 또한 UPD가 표준 UED에 통합되어 환경·partner 분포 양쪽에 대한 완전 비지도 curriculum을 가능하게 함을 입증하였다. 자연스러운 베이스라인 하나는 *가장 어려운 파트너*(평균 return 최저)를 우선시하는 것으로 선행 연구에서 탐구되었다(Zhao et al. 2023; Li et al. 2023b; You et al. 2025). 그러나 이론적으로도 예비 실험에서도 이 접근법은 적대적 동역학으로 이어진다: Overcooked-AI에서는 거의 무작위 파트너($\epsilon \to 1$)를, OGC에서는 풀 수 없는 level을 크게 선호하여 두 경우 모두 학습이 정체된다. 이는 우리 설정이, 적대적 행동을 암묵적으로 제한하는 사전학습·유계 population에 의존하는 선행 연구와 달리 open-ended partner generator를 수반하기 때문이다. 그러한 제약된 설정에서는 "어려운" 예제를 우선시해도 실현 가능한 협응 공간에 머물지만, (J)UPD에서는 제약 없는 난이도 선택이 학습 신호를 완전히 붕괴시킬 수 있다.

#### J.1.1 파트너별 결과

Figure 20에서 UPD가 서로 다른 파트너와 어떻게 동작하는지에 대한 추가 결과를 표시한다. 평균적으로, 그리고 대부분의 파트너와 함께, UPD가 가장 잘 동작한다.

> **Figure 20.** Overcooked-AI에서 레이아웃 평균 파트너별 결과. UPD는 대부분의 파트너와 함께 다른 방법들을 능가한다. 일부 방법은 Onion·Plate 에이전트와 함께 유사하거나 약간 더 나은 성능을 보이지만 UPD보다 일관되게 우수하지는 않다.

#### J.1.2 UPD ablation의 curriculum 동역학

bias 메커니즘의 역할을 더 이해하기 위해, UPD w/o bias의 partner-선택 동역학을 분석한다. Figure 21은 학습 전반의 레이아웃별 평균 선택 혼합 계수 $\epsilon$ 을 보인다. 전체적으로 전체 UPD(Figure 4)와 유사한 동역학을 관찰하며, 주된 차이는 CRoom에서 UPD w/o bias가 전체 UPD보다 낮은 평균 $\epsilon$ 으로 수렴한다는 점이다. 반면 UPD w/o ℓ 은 모든 레이아웃에서 평균 $\epsilon$ 0.5 부근에서 변동한다(예상대로). 또한 UPD w/o ℓ 의 행동 편향은 학습 내내 거의 일정하게 무작위 정책 사전(prior) 부근에서 변동한다. 이는 Figure 6에서 관찰된 관습 깨기 동역학이 확률적 partner generator 단독이 아니라 **partner 생성과 learnability 기반 선택의 상호작용**에서 창발함을 시사한다.

> **Figure 21.** Figure 4와 동일한 분석을 UPD w/o bias에 대해 수행. 전반적으로 유사한 curriculum 동역학을 보이며, CRoom에서만 전체 UPD보다 눈에 띄게 낮은 평균 $\epsilon$ 으로 수렴한다.

## K Additional User Study Details (사용자 연구 추가 세부사항)

피험자내 설계(within-subjects)를 사용하였다: 각 참가자는 3개 Overcooked 레이아웃(AA, CR, CC)에 걸쳐 4개 에이전트(SP, MEP, E3T, UPD) 모두와 상호작용하였다. 각 에이전트-레이아웃 쌍을 한 번씩 플레이하여 참가자당 36개 에피소드가 되었고, 에이전트·레이아웃 순서는 사용자별로 무작위화되었다. 각 전체 연구는 18–35분 소요되었으며 본 연구 전 짧은 튜토리얼을 완료하였다. 게임은 200 환경 스텝(배달당 20점)으로, 선행 인간 평가 프로토콜(Jha et al. 2025)과 일치한다.

연구는 NiceWebRL 프레임워크를 사용한 웹 기반 인터페이스로 진행되었다. 12명의 참가자(여성 5명, 연령 22–31세)를 모집했으며, 기관 윤리심의위원회 승인을 받고 모든 참가자가 사전 동의를 제공하였다. 보상은 기관 지침에 따라 제공되었다. 참가자는 키보드(화살표 키, 스페이스, S 키)로 캐릭터를 조작하였다. 편향을 피하기 위해 참가자는 에이전트 정체를 알지 못했으며 실험자도 실험 중 알지 못했다(이중 맹검). 각 에이전트 상호작용 후, 7개의 5점 리커트 척도 질문으로 에이전트를 평정하였다:

1. 나는 이 에이전트와 플레이하는 것이 즐거웠다.
2. 에이전트가 나와 협응하는 능력은 다음과 같았다고 느꼈다: (very poor, poor, neutral, good, very good)
3. 에이전트는 의사결정을 할 때 나에게 적응했다.
4. 에이전트가 자주 내 길을 막았다. (부정 문항)
5. 에이전트는 자신의 행동에서 일관적이었다.
6. 에이전트의 행동은 인간 같았다.
7. 에이전트의 행동은 답답했다. (부정 문항)

연구 마지막에 자유 형식 피드백을 허용하였다. 응답은 1–5로 사상되었고, 전반적 주관적 선호 분석을 위해 부정 가치 문항은 집계 전 반전하였다. 내적 일관성은 Cronbach's $\alpha = 0.916$ 으로 강한 신뢰도를 시사하여, 질문 전반의 점수를 집계해 에이전트당 단일 선호 점수를 산출하는 것을 정당화한다. UPD를 각 베이스라인과 비교하는 단측 Wilcoxon signed-rank 검정(UPD > baselines 가설)을 수행했고, 다중 비교 통제를 위해 각 질문 내 Holm-Bonferroni 보정을 적용했다. 집계 선호 점수도 동일 절차로 검정했다. 성능(보상)은 Holm 보정과 함께 단측 대응표본 t-검정으로 분석했다. 각 설문 질문의 에이전트별 리커트 응답 분포는 Figure 22에 제공한다.

> **Figure 22.** 모든 에이전트에 걸친 각 설문 질문의 인간 평정 분포. 각 막대는 각 리커트 항목(x축)에 주어진 응답 수이며 색상은 에이전트를 나타낸다.

---

*— 번역 끝 (arXiv v2, 2026-06-07 기준) —*
