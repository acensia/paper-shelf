# Unsupervised Partner Design Enables Robust Ad-hoc Teamwork
## (비지도 파트너 설계를 통한 강건한 Ad-hoc Teamwork)

**저자:** Constantin Ruhdorfer¹, Matteo Bortoletto¹, Victor Oei¹, Anna Penzkofer¹, Andreas Bulling¹
¹Collaborative Artificial Intelligence, University of Stuttgart, Germany
firstname.lastname@vis.uni-stuttgart.de

**게재:** ICML 2026 Oral · arXiv: [2508.06336](https://arxiv.org/abs/2508.06336)

> **번역 노트:** 주요 기술 용어(ad-hoc teamwork, learnability, ego agent, curriculum, partner generator 등)는 영어 원문 그대로 유지하고 설명만 한국어로 번역했습니다. 수식과 표는 원문 구조를 보존했으며, 참고문헌 목록(저자·제목)은 원문 그대로 두었습니다.

---

## Abstract (초록)

본 논문은 **Unsupervised Partner Design (UPD)** — 강건한 ad-hoc teamwork를 위한, population이 필요 없는(population-free) 멀티에이전트 강화학습 프레임워크 — 를 제안한다. UPD는 사전학습된 파트너나 수동 파라미터 튜닝 없이도 학습 파트너를 적응적으로 생성한다. UPD는 ego agent의 정책을 편향된(biased) 무작위 행동과 확률적으로 혼합(stochastic mixing)하여 다양한 파트너를 구성하고, 이들을 분산(variance) 기반 learnability 지표로 채점하여 ego agent의 현재 학습 경계(learning frontier)에 가까운 파트너를 우선시한다. 우리는 UPD가 unsupervised environment design(UED)과 통합될 수 있음을 보이며, 이를 통해 협력 환경에서 **level 분포와 partner 분포 양쪽에 대해 완전히 비지도(fully unsupervised) curriculum을 가능하게 하는 최초의 방법**을 만들어 낸다. Overcooked-AI와 Overcooked Generalisation Challenge에 대한 광범위한 평가를 통해, 이 동적 partner curriculum이 매우 효과적임을 보인다: UPD는 population 기반·population-free 베이스라인과 ablation 모두를 일관되게 능가한다. 추가로 사용자 연구(user study)에서, UPD가 모든 베이스라인보다 높은 return을 달성하며 동시에 유의미하게 더 적응적이고(more adaptive), 더 인간 같으며(more human-like), 더 나은 협력자이고(better collaborator), 덜 답답하다(less frustrating)고 인식됨을 보인다.

---

## 1 Introduction (서론)

알려지지 않은 파트너와의 강건한 협력, 즉 **ad-hoc teamwork**(Stone et al. 2010, AHT)는 범용 멀티에이전트 시스템을 구축하는 데 핵심이다. AHT를 위한 에이전트 학습은 비용이 큰데, 기존 방법들이 대개 다양한 파트너로 구성된 대규모 population을 필요로 하기 때문이다(Strouse et al. 2021; Zhao et al. 2023; Yu et al. 2023; Li et al. 2023; Wang et al. 2025). 효율적인 종단간(end-to-end) 학습 방법인 E3T(Yan et al. 2023)는 파트너 정책을 ego 정책의 확률적 혼합(stochastic mixture)으로 생성하여 이 문제를 완화하지만, 여전히 평가 설정에 맞춰 혼합 파라미터를 세심하게 튜닝해야 한다. 한편, **unsupervised environment design**(Dennis et al. 2020, UED)과 **dual curriculum design**(Jiang et al. 2021, DCD) 방법들은 환경 파라미터에 대한 적응적 curriculum이 효율적 학습과 강건한 일반화를 가져올 수 있음을 보였다. 최근 연구는 partner와 environment의 결합 일반화(joint generalisation)의 중요성을 부각하는 동시에, 현재 모델들이 이를 달성하지 못함도 지적하였다(Ruhdorfer et al. 2025).

본 연구에서 우리는 partner–environment 결합 공간에 대한 curriculum을 구성함으로써 **AHT와 UED를 통합하는 최초의 접근법**을 제시한다. 이를 위해 우리는 경량(lightweight)이고 population-free한 AHT 알고리즘인 **Unsupervised Partner Design (UPD)** 를 제안한다. UPD는 ego agent의 현재 학습 경계(learning frontier)에 기반하여 partner 파라미터를 적응적으로 선택함으로써, 효율적이고 강건한 zero-shot 협응(coordination)을 가능하게 한다. 우리는 UPD를 단지 새로운 알고리즘이 아니라 **partner 생성을 위한 일반적 프레임워크**로 본다. 이 관점에서 기존 접근법들 — 예컨대 다양한 population에 대한 best response 학습이나 E3T — 은 고정되고 비적응적인 partner 분포를 갖는 특수 사례(special case)로 나타난다. 기존 UED 알고리즘과 결합되면, UPD는 AHT를 위한 **최초의 결합 open-ended partner–environment curriculum 학습 방법**을 가능하게 한다. 우리의 기여는 다음과 같다:

1. **Unsupervised Partner Design (UPD)** 를 도입한다. 이는 partner-space curriculum learning을 통한 ad-hoc teamwork의 새로운 프레임워크로, 사전학습된 population 없이 다양한 파트너를 실시간(on-the-fly)으로 생성한다.

2. **유연한 partner 생성 프레임워크**를 설계한다. 이는 확률적 혼합 샘플링(stochastic mixture sampling)과 Dirichlet 편향 무작위화(Dirichlet-biased randomisation)를 결합하여, 능력(competence) 변동성과 체계적 행동 편향(systematic behavioural bias)을 모두 포착한다. 우리는 이를 return-분산 기반 선택 메커니즘과 통합하여 partner curriculum을 적응적으로 구성한다.

3. Overcooked-AI(Carroll et al. 2019)와 **Overcooked Generalisation Challenge**(Ruhdorfer et al. 2025, OGC)에 대해 강력한 population 기반·population-free 베이스라인, 다양한 평가 population, 그리고 인간을 대상으로 포괄적인 실증 평가를 제공한다. UPD는 주요 지표 전반에서 베이스라인을 일관되게 능가한다.

> **Figure 1.** *Unsupervised partner design*은 ad-hoc teamwork를 위한 새로운 population-free 멀티에이전트 학습 프레임워크로, learnability를 이용해 ego agent에게 최적인 학습 파트너를 찾아 open-ended curriculum을 생성한다. (구성: **Partner Generator**가 후보를 보내고 → **Learner**가 learnability에 기반해 채택하며 → ego agent를 학습시키고 → 그 결과가 generator를 갱신·시드(seed)한다.)

---

## 2 Related Work (관련 연구)

### 2.1 Ad-hoc Teamwork

AHT는 다양한 멀티에이전트 강화학습(RL) 환경에서 연구되어 왔다(Carroll et al. 2019; Bard et al. 2020; Kurach et al. 2020). 널리 쓰이는 AHT 방법들, 예컨대 fictitious co-play(Strouse et al. 2021, FCP)나 maximum entropy population-based training(Zhao et al. 2023, MEP)은 다양한 partner population을 사전학습하고 이들에 대한 best-response 정책을 최적화하는 데 의존한다(Yu et al. 2023; Lou et al. 2023; Rahman et al. 2023). 최근 연구들은 채점 함수(scoring function)에 기반하여 population에서 학습 파트너를 적응적으로 선택하는 curriculum learning을 도입하였다(Erlebach and Cook 2024; You et al. 2025). 또 다른 흐름은 open-ended 학습 목표를 도입하여 partner 다양성을 동적으로 확장하였으나(Li et al. 2023; Wang et al. 2025), 여전히 시간이 지남에 따라 partner population을 키우는 방식이었다. 주목할 예외는 E3T(Yan et al. 2023)로, 이는 ego 정책과 무작위 정책의 혼합으로 파트너를 실시간 생성한다. 이 접근법은 어떤 partner population도 필요로 하지 않아 계산 부담을 크게 줄이지만, 여전히 각 과제와 평가 시나리오마다 ego 정책과 무작위 정책 사이의 혼합 계수(mixture coefficient)를 세심하게 튜닝해야 한다. 이와 대조적으로, 우리는 고정 파라미터나 사전학습 population 없이 다양한 partner 행동을 적응적으로 생성하며 현행 curriculum learning 프레임워크에 매끄럽게 통합되는, 경량의 population-free 접근법을 제안한다.

### 2.2 Unsupervised Environment Design

UED(Dennis et al. 2020)는 에이전트의 능력에 맞춘 학습 환경을 적응적으로 생성하며, 일반화 향상에 효과적임이 입증되었다. domain randomisation(Tobin et al. 2017, DR)과 달리, UED는 에이전트의 학습 경계(learning frontier)를 겨냥하여 환경을 생성한다. 기존 UED 방법들은 주로 단일 에이전트 설정에 초점을 두며, 환경 생성을 유도하기 위해 regret 기반 목적함수에 의존한다(Wang et al. 2019, 2020; Dennis et al. 2020; Jiang, Grefenstette, and Rocktäschel 2021; Jiang et al. 2021; Parker-Holder et al. 2022; Li, Varakantham, and Li 2023; Beukman et al. 2024). 멀티에이전트 설정으로의 확장은 제한적이다: Samvelyan et al. (2023)은 경쟁적(competitive) 설정에 초점을 두었고, Ruhdorfer et al. (2025)은 협력적(cooperative) 멀티에이전트 UED 벤치마크를 제안했으나 방법(method)은 제시하지 않았다. 최근 연구들은 UED를 learnability 주도 문제로 재정식화하여, regret 근사를 환경의 학습 잠재력(learning potential)을 직접 측정하는 채점 함수로 대체하였다(Rutherford et al. 2024; Monette et al. 2025). 그러나 어떤 선행 연구도 partner 정책의 파라미터화는 탐구하지 않았다. 우리는 비지도 설계(unsupervised design)를 partner 정책으로 확장하여, population-free curriculum 메커니즘으로서의 적응적 partner 생성을 도입한다. 나아가 우리의 방법은 표준 UED 접근법과 자연스럽게 통합되어, 강건한 zero-shot 협력을 위한 partner와 environment의 결합 비지도 설계(joint unsupervised design)를 가능하게 한다.

---

## 3 Preliminaries (사전 지식)

### 3.1 Reinforcement Learning

본 연구에서 우리는 Ruhdorfer et al. (2025)의 **decentralised under-specified partially observable Markov decision process (Dec-UPOMDP)** 형식을 채택하며, 이는 튜플 $\mathcal{M} = \langle \mathcal{N}, A, \Omega, \Theta, S_\mathcal{M}, T_\mathcal{M}, O_\mathcal{M}, R_\mathcal{M}, \gamma \rangle$ 로 주어진다. "under-specified(과소명세)"라는 용어는 Dec-UPOMDP가 환경들의 *집합(family)* 을 정의함을 반영한다. 여기서 각각의 완전 명세된(fully-specified) 인스턴스 $\mathcal{M}_\theta$ (하나의 **level**)는 특정 환경 파라미터 $\theta \in \Theta$ 의 선택에 대응한다. UPOMDP(Dennis et al. 2020)는 $\Theta$ 로 주어지는 level 분포에 대해 에이전트를 학습시키는 형식을 제공한다는 점에서 유용하다.

Dec-UPOMDP 내에서 $\mathcal{N}$ 은 에이전트 집합, $A$ 는 행동 집합, $\Omega$ 는 관측 집합이며, 환경의 참 상태(true state) 집합은 $S_\mathcal{M}$ 으로 주어진다. 주어진 level $\mathcal{M}_\theta$ 의 각 시점 $t$ 에서, 각 에이전트 $i \in \mathcal{N}$ 은 관측 함수 $O: S \times \mathcal{N} \to \Omega$ 를 통해 관측 $o^i_t \in \Omega$ 를 받는다. 에이전트들은 결합 행동(joint action) $\boldsymbol{a}_t = (a^{(1)}_t, \dots, a^{(n)}_t)$ 를 산출하며, 여기서 $a^{(i)} \in A$ 이다. 그런 다음 공유 보상(shared reward) $R(s, \boldsymbol{a})$ 를 관측하고, 환경은 전이 함수 $T: S \times A_1 \times \cdots \times A_n \times \Theta \to \Delta(S)$ 에 따라 전이하는데, 이는 다음 상태에 대한 확률 분포로 사상된다.

궤적(trajectory) $\tau$ 는 상태와 결합 행동의 수열이다. 표기를 다소 남용하여, $R(\tau)$ 를 $\tau$ 를 따라 얻은 (할인되지 않은) 보상의 합으로 표기한다. 에이전트들은 기대 할인 보상합을 최대화한다:

$$J(\pi^{(1)}, \dots, \pi^{(n)}) = \mathbb{E}\left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a^{(1)}_t, \dots, a^{(n)}_t) \right]. \tag{1}$$

우리는 실험에서 완전 명세된(Overcooked-AI) 환경과 과소명세된(OGC) 환경을 모두 탐구한다. 완전 명세된 환경의 경우, 이 형식은 Dec-POMDP(Oliehoek and Amato 2016)로 축소된다.

### 3.2 Dual Curriculum Design and Learnability

(Dec-)UPOMDP를 푸는 알고리즘들은 **dual curriculum design**(Jiang et al. 2021, DCD) 프레임워크 아래에서 이해될 수 있다. 여기서 학습 에이전트는 **level generator**가 생성하고 **curator**가 선택한 level들에서 학습된다. 방법들은 주로 generator와 curator를 어떻게 구현하느냐에서 차이가 난다. Prioritised level replay(Jiang, Grefenstette, and Rocktäschel 2021, PLR)는 level을 무작위로 만들고 채점 함수에 기반해 샘플링한다. PAIRED(Dennis et al. 2020)는 teacher-student 구성을 사용한다. REPAIRED(Jiang et al. 2021)는 두 아이디어를 결합한다. 이와 대조적으로 Rutherford et al. (2024)은 **sampling for learnability (SFL)** 를 도입했는데, 여기서는 무작위로 생성된 level들을 learnability 함수로 채점하여 curriculum 구성을 더 효과적으로 유도한다. 구체적으로, 이들은 점수를 $\ell_{sr} = p(1 - p)$ 로 정의하며, $p$ 는 이진(binary) 결과를 갖는 level $\theta$ 에 대한 성공률(success rate)이다. 직관적으로, 에이전트가 어떤 level에서 항상 실패하거나 항상 성공한다면 $\ell_{sr}$ 은 0이 된다. 즉 learnability는 에이전트의 학습 경계에 있는 level에서만 높아진다. Monette et al. (2025)은 이를 평균 주변의 return 분산에 가중을 둠으로써 연속 보상(continuous reward)으로 확장하였다. 경험적 분산 $\sigma$ 와 평균 $\mu$, 그리고 가우시안 확률밀도함수 $\mathcal{N}$ 에 대해:

$$\ell_{\text{gauss}} = \sigma_\theta \cdot \mathcal{N}(\mu_\theta \mid \mu, \sigma^2). \tag{2}$$

저자들은 이 가중이 평균에서 멀리 떨어진 level을 하향 가중(downweight)하여 학습을 안정화한다고 주장하였다.

> **Figure 2.** 학습이 진행되며 ego 능력(competence)이 향상될 때(검정), E3T는 고정 혼합 계수(여기서 $\epsilon = 0.5$)로 파트너를 생성하여 ego 능력의 *고정된 비율*에 해당하는 파트너를 만든다(파랑). 이와 대조적으로 UPD는 $\epsilon \sim U(0,1)$ 로 샘플링하고(초록) learnability 기준으로 파트너를 필터링하여, 동적인 범위의 partner 능력을 만들어 낸다(주황).

### 3.3 Efficient End-To-End Training for Human-AI Cooperation

우리는 E3T(Yan et al. 2023)에 기반한 새로운 AHT 알고리즘을 제안한다. E3T는 새로운 파트너에 강건한 ego 정책 $\pi_{\text{ego}}$ 를 학습한다. 이를 위해 에피소드마다 partner 정책 $\pi_p$ 를 구성하는데, 이는 현재 정책 $\pi_{\text{ego}}$ 와 균등 무작위(uniformly random) 정책 $\pi_r$ 을 혼합 계수 $\epsilon$ 으로 혼합한 것이다:

$$\pi_p = \epsilon \pi_r + (1 - \epsilon)\pi_{\text{ego}}, \quad \text{단 } \epsilon \in [0, 1]. \tag{3}$$

중요하게도, Yan et al. (2023)은 $\epsilon$ 값을 레이아웃과 협력 파트너에 따라 선택하였다. 우리 연구의 핵심 신규성 중 하나는, 어떤 과제 지식(task knowledge)도 요구하지 않고 $\epsilon$ 을 실시간으로 적응시킨다는 점이다.

---

## 4 Unsupervised Partner Design

**Unsupervised partner design (UPD)** 는 learnability 잠재력에 기반하여 협력 파트너를 적응적으로 생성·선택하는 프레임워크이자 종단간(end-to-end) 학습 알고리즘이다. 우리의 핵심 통찰은, DCD를 환경 수준이 아니라 **partner 정책 공간에 직접** 적용하는 것이다. UPD는 두 가지 구성요소를 필요로 한다: (1) **partner generator** 와 (2) open-ended curriculum을 구축하기 위한 **적응적 선택 기준(adaptive selection criterion)** (Figure 1 참조). 고정 혼합 계수로 파트너를 생성하는 E3T와 같은 기존 접근법과 달리, 우리의 방법은 파트너를 샘플링하고 채점하여 다양한 파트너들에 대한 적응적 curriculum을 구축한다(Figure 2 참조). 이러한 점에서 UPD는 선행 AHT 방법들의 일반화로 이해될 수 있다: FCP나 MEP 같은 인기 있는 population 기반 접근법은 partner generator 대신 고정되고 정적인 partner 집합에서 샘플링하는 것에 해당하며, E3T는 고정 혼합 계수($\epsilon = c$)를 사용하고 ego agent의 학습 진척에 대한 적응이 없는 특수 사례에 해당한다.

### 4.1 Online Partner Generation

우리는 E3T의 partner 생성 전략을 확장하여 추가적인 확률성(stochasticity)과 행동 편향(behavioural bias)을 도입한다. partner 확률성을 도입하기 위해, 우리는 *고정된 $\epsilon$ 값이 학습의 서로 다른 단계에 걸쳐 최적이 아닐 수 있다*는 통찰을 활용한다.

> **Algorithm 1: Unsupervised Partner Design Training**
> **요구사항:** 환경 $E$, ego agent 정책 $\pi_{\text{ego}}$, partner 정책 생성기 $G_p$, 평가 rollout 수 $N$, rollout 길이 $L$, learnability buffer 크기 $K$
> 1: 빈 partner buffer $B$ 를 초기화
> 2: **while** 수렴하지 않음 **do**
> 3: &nbsp;&nbsp;**for** 각 partner 샘플링 iteration **do**
> 4: &nbsp;&nbsp;&nbsp;&nbsp;파트너 샘플링: $\pi_p \sim G_p(\pi_{\text{ego}})$ (Alg. 2)
> 5: &nbsp;&nbsp;&nbsp;&nbsp;쌍 $(\pi_{\text{ego}}, \pi_p)$ 에 대해 길이 $L$ 의 rollout $N$ 개 수집
> 6: &nbsp;&nbsp;&nbsp;&nbsp;해당 쌍의 return $R_1, \dots, R_N$ 수집
> 7: &nbsp;&nbsp;&nbsp;&nbsp;learnability 점수 계산: $\ell = \mathrm{Var}(R_1, \dots, R_N)$
> 8: &nbsp;&nbsp;&nbsp;&nbsp;$(\ell, \pi_p)$ 를 buffer $B$ 에 저장
> 9: &nbsp;&nbsp;**end for**
> 10: &nbsp;learnability가 가장 높은 상위 $K$ 개 정책을 $B$ 에서 가져옴
> 11: &nbsp;**for** 각 PPO 갱신 스텝 **do**
> 12: &nbsp;&nbsp;선택된 $K$ 개 중에서 파트너 $\pi_p$ 샘플링
> 13: &nbsp;&nbsp;궤적 $(\pi_{\text{ego}}, \pi_p)$ 수집
> 14: &nbsp;&nbsp;PPO를 사용해 ego 정책 $\pi_{\text{ego}}$ 갱신
> 15: &nbsp;**end for**
> 16: **end while**

> **Algorithm 2: Partner Policy Generator $G_p$**
> **요구사항:** ego 정책 $\pi_{\text{ego}}$
> 1: 혼합 파라미터 샘플링 $\epsilon \sim U(0, 1)$
> 2: 확률 $p_{\text{bias}}$ 로:
> 3: &nbsp;&nbsp;편향 마스크(bias mask) 샘플링 $m \sim \mathrm{Dirichlet}(\alpha \cdot \mathbf{1}_A)$
> 4: 그렇지 않으면:
> 5: &nbsp;&nbsp;$m = \mathbf{1}_A / A$ 로 설정
> 6: 편향 마스크 $m$ 을 사용해 무작위 정책 $\pi_r$ 구성
> 7: 혼합 partner 정책 $\pi_p = \epsilon \pi_r + (1 - \epsilon)\pi_{\text{ego}}$ 반환

학습 초기에는 더 유능한 파트너가 과제 학습을 돕고, 후반에는 더 무작위적인 파트너가 협응 도전과제를 제공한다. 따라서 $\epsilon$ 을 고정하는 대신, 우리는 먼저 학습 전반에 걸쳐 $\epsilon$ 을 균등 분포에서 샘플링한다. 그런 다음 우리의 curator가 learnability 점수에 기반하여 파트너를 선택한다. 이로써 ego agent는 자신의 현재 학습 단계에 부합하는 다양한 능력 범위(range of competencies)에 노출된다.

게다가 Overcooked-AI와 같은 많은 협력 도메인에서 partner 변동성은 능력에만 국한되지 않는다: 파트너는 체계적 행동 편향을 보일 수 있다 — 예컨대 인간 파트너는 특정 행동에 대한 선호를 보이거나(Yu et al. 2023), stay(머무름) 행동에 대한 성향이 높아지는 경향이 있다(Carroll et al. 2019). 이러한 편향을 포착하는 것은 강건한 AHT 정책을 학습하는 데 중요하다. 능력과 행동 다양성을 모두 수용하기 위해, 우리는 두 가지 구성요소로 된 partner generator를 설계한다:

1. **Variable Mixing(가변 혼합):** 각 파트너에 대해 혼합 계수 $\epsilon \sim U(0, 1)$ 을 샘플링하여, 파트너가 완전 무작위($\epsilon = 1$)부터 완전 ego($\epsilon = 0$)까지의 범위를 가질 수 있게 한다.

2. **Bias Masking(편향 마스킹):** $A$ 에 대한 Dirichlet 분포에서 편향 마스크 $m \sim \mathrm{Dir}(\alpha \cdot \mathbf{1}_A)$ 를 샘플링하여, 지속적인 행동 선호(persistent action preference)를 갖는 편향된 무작위 정책을 만든다.

이 partner 생성 메커니즘은 다양한 능력과 행동 특성을 아우르는 잠재적 파트너 집합을 온라인으로 생성할 수 있게 한다. 그러나 이는 또한 다음 질문을 제기한다: 학습 스텝 $t$ 에서, 최적의 학습을 촉진하려면 partner 파라미터를 어떻게 골라야 하는가?

### 4.2 Sampling Partners for Learnability

우리가 다양한 파트너 집합을 생성하더라도, 모든 파트너가 학습에 똑같이 유용한 것은 아니다. learnability 기반 curriculum에 대한 선행 통찰을 따라, 우리는 파트너를 learnability 잠재력에 대해 채점할 수 있다고 가정한다. 우리는 **sampling-for-learnability (SFL)** 의 개념을 다음과 같이 변용한다: 학습 과정의 매 스텝마다 Alg. 2를 사용해 다수의 파트너를 샘플링하고, 환경에서의 rollout을 통해 이들의 learnability 잠재력을 채점한 뒤, partner buffer $B$ 를 사용해 learnability 점수가 높은 파트너를 선택한다(Alg. 1 참조).

협력 설정에는 흔히 이진 성공(binary success)이 존재하지 않으므로, 우리는 후보 파트너 $\pi_p$ 에 대한 rollout들의 return 분산에 기반한 일반화된 learnability 점수 함수를 사용한다:

$$\ell_{\text{var}} = \mathrm{Var}_{\tau \sim (\pi, \pi_p)}[R(\tau)], \quad \text{단 } R \in [0, R_{\max}]. \tag{4}$$

낮은 분산은 일관된 실패(consistent failure) 또는 숙달(mastery) 중 하나를 가리킨다. 높은 분산은 ego agent가 때때로 성공함을 시사한다: $\ell_{\text{var}}$ 은 수집된 return의 정확히 절반이 0이고 절반이 $R_{\max}$ 일 때, 즉 에이전트가 때로는 성공하고 때로는 실패할 때 가장 높다. 이는 curriculum learning 원리와 부합하는데, 그 원리에 따르면 중간 난이도(높은 분산)의 과제가 최대의 학습 진척을 이끈다(Florensa et al. 2018; Tzannetos et al. 2023). 이 분산 기반 채점은 선행 연구(Rutherford et al. 2024)에서 이진 결과 과제에 사용된 성공률 기반 정의를 자연스럽게 확장하지만, Monette et al. (2025)과 달리 성능의 극단(performance extremes)을 하향 가중하지 *않는다.* 따라서 이는 에이전트를 최선 및 최악의 파트너 모두에 대비시킨다는 AHT의 목표를 반영한다.

### 4.3 Training Loop

UPD의 전체 학습 과정은 partner 생성, learnability 채점, 학습 사이를 번갈아 수행한다(Alg. 1 참조):

1. **Partner Sampling:** 편향된 확률적 생성기(Alg. 2)를 사용해 여러 partner 정책을 생성한다.
2. **Learnability Scoring:** 샘플링된 각 파트너에 대해 여러 rollout을 수집하고 learnability 점수를 계산한다.
3. **Partner Selection:** learnability 점수가 가장 높은 상위 $K$ 개 파트너를 선택해 학습 buffer를 구성한다.
4. **Policy Update:** buffer에서 샘플링한 파트너와 함께 임의의 RL 방법으로 ego agent를 학습한다.

이 절차는 partner 정책에 대한 진화하는(evolving) curriculum을 형성하여, 학습 잠재력이 가장 높게 남아 있는 파트너에 학습을 적응적으로 집중시킨다. 결정적으로, 그리고 선행 연구와 대조적으로, 이는 *어떤 고정된 partner 다양성을 미리 정의하거나 $\epsilon$ 을 수동 튜닝하지 않고도* 달성된다.

### 4.4 Joint Unsupervised Design Learning

level과 partner 양쪽에 걸친 일반화를 요구하는 도전과제(예: OGC)의 경우, 우리의 방법은 기성(off-the-shelf) UED 학습 알고리즘 — 구체적으로 SFL — 에 쉽게 통합되어, level-policy 쌍을 샘플링하고 채점하는 결합 학습 알고리즘을 구성할 수 있다. 우리는 이 알고리즘을 **Joint UPD (JUPD)** 라 부른다. 이를 위해 우리는 위의 learnability 개념을 재사용하고, Alg. 1의 learnability 채점 루프를 Alg. 3에 보인 것으로 대체하여 조정한다: level $\theta \sim \Theta$ 와 partner 정책 $\pi_p \sim G_{\text{partner}}$ 를 샘플링한 뒤, 결합 튜플 $(\pi_{\text{ego}}, \pi_p, \theta)$ 를 채점하여 buffer $B$ 에 저장한다.

> **Algorithm 3: Adjusted Sampling for JUPD**
> **요구사항:** 과소명세 환경 $E$, 환경 파라미터 분포 $\Theta$
> 1: 환경의 자유 파라미터 샘플링 $\theta \sim \Theta$
> 2: partner 정책 샘플링: $\pi_p \sim G_p$ (Alg. 2)
> 3: 쌍 $(\pi_\theta, \pi_p)$ 에 대해 $E(\theta)$ 에서 $N$ 개 에피소드 rollout
> 4: 해당 쌍의 return $R_1, \dots, R_N$ 수집
> 5: $\ell_{CV^2} = \mathrm{Var}(R_1, \dots, R_N) / \mathrm{Mean}(R_1, \dots, R_N)^2$
> 6: $(\ell_{CV^2}, \pi_p, \theta)$ 를 buffer $B$ 에 저장
> 7: {Alg. 1과 동일하게 진행}

서로 다른 보상 스케일을 갖는 level들 사이에서 learnability 점수가 비교 가능하게 유지되도록, 우리는 **coefficient-of-variation squared (CV²)** 정식화를 채택한다. 이는 return 분산을 $\theta$ 별 평균의 제곱으로 정규화한다:

$$\ell_{CV^2} = \frac{\mathrm{Var}_{\tau \sim (\pi, \pi_p, \theta)}[R(\tau, \theta)]}{\left(\mathbb{E}_{\tau \sim (\pi, \pi_p, \theta)}[R(\tau, \theta)]\right)^2}. \tag{5}$$

이는 스케일 불변(scale-invariant) 지표를 산출하는데, CV²는 서로 다른 데이터셋 — 우리의 경우 서로 다른 level — 사이의 변동성을 비교하는 데 사용될 수 있기 때문이다. 종합하면, 이는 Ruhdorfer et al. (2025)이 구상한 대로 결합 partner-level curriculum을 확립한다. 우리 접근법의 핵심 장점은, UPD를 SFL에 추가해도 단순한 파라미터 샘플링 외에는 추가적인 계산 비용이 들지 않는다는 점이다.

---

## 5 Experiments (실험)

우리의 실험은 다음 여러 질문에 답하고자 한다:
- **RQ1** UPD는 강건한 AHT 정책을 학습하는가?
- **RQ2** UPD의 모든 구성요소가 중요한가?
- **RQ3** 인간은 UPD를 어떻게 평가하는가?
- **RQ4** UPD를 UED 알고리즘에 통합할 수 있는가?

### 5.1 RQ1: Ad-Hoc Teamwork in Overcooked-AI

우리는 먼저 AHT의 잘 확립된 벤치마크인 Overcooked-AI(Carroll et al. 2019)에서 UPD를 평가하였다. 다섯 개의 표준 레이아웃을 고려하였다: Cramped Room (CRoom), Asymmetric Advantages (AA), Coordination Ring (CR), Counter Circuit (CC), Forced Coordination (FC). 각 방법에 대해 독립적인 6개 시드(seed)를 학습하고 새로운 파트너와의 협응을 평가하였다. 협응은 AHT 설정에서 에이전트를 다양한 partner 정책을 포함하는 평가 population과 짝지어 평가하였다.

도전적이고 다양한 평가 population을 구성하기 위해, 우리는 관련 연구(Wang et al. 2025)에서 영감을 받아 하드코딩 에이전트, 인간 영감 계획(human-inspired planning) 모델, 그리고 BRDiv(Rahman et al. 2023)로 학습된 에이전트를 결합하였다. BRDiv population은 self-play return을 최적화하면서 cross-play 호환성을 최소화한다. 이러한 에이전트들은 유능하지만 호환되지 않는(competent but incompatible) 전략을 채택하는 경향이 있어 협력하기 어려울 수 있다. 구체적으로, 평가 파트너는 네 가지 범주에서 추출되었다: (1) self-play용으로 학습되었으나 cross-play에서 호환되지 않는 BRDiv 에이전트 3–4개(Rahman et al. 2023), (2) 인간 영감 휴리스틱을 따르는 확률적 계획(probabilistic planning) 에이전트(Wang et al. 2025), (3) 과제의 부분집합을 수행하는 하드코딩 에이전트(예: 양파만 담당하는 작업자)(Yu et al. 2023), (4) 최악의 경우 파트너로서 무작위(random) 또는 stay 에이전트. 각 레이아웃은 이러한 파트너를 4–9개 사용하였다(CRoom 9개, AA·CR·CC 8개, FC 4개). FC는 제약적인 동역학으로 인해 실행 가능한 하드코딩 파트너 전략이 제한되어 더 적게 사용하였다.

우리는 Overcooked-AI에서 가장 널리·성공적으로 쓰이는 AHT 알고리즘들을 함께 아우르는 네 개의 베이스라인과 UPD를 비교하였다: (1) IPPO(de Witt et al. 2020)로 학습된 self-play 베이스라인. (2) 보고된 하이퍼파라미터를 사용한 E3T: CRoom·AA·CR·CC에는 $\epsilon = 0.5$, FC에는 $\epsilon = 0.0$. (3) 엔트로피 계수 $\alpha = 0.01$ 과 population 크기 48을 사용한 MEP 에이전트. (4) 크기 48의 population에 대한 best response로 학습된 FCP 에이전트. 공정한 비교를 위해 MEP와 FCP를 다음과 같이 튜닝하였다: 첫째, 우리의 population은 원논문보다 컸으며 이는 성능을 향상시킨다(Yu et al. 2023). 둘째, best-response 학습 동안 수렴을 보장하기 위해 최대 $1 \times 10^8$ 환경 스텝(UPD의 2배)까지 학습을 연장하였다. 셋째, best-response 학습에서 레이아웃별로 학습률을 미세조정하였다($1e{-}3$, $3e{-}4$, $1e{-}4$ 중에서). 이와 대조적으로, UPD는 모든 레이아웃에 대해 하나의 하이퍼파라미터 세트로 동작함을 확인하였다.

모든 방법에 대해 동일한 recurrent 정책 기반(base)을 사용하였다. E3T와 UPD의 경우, E3T가 제안한 partner model을 추가하였다. 학습은 512개 환경(환경당 400 스텝), 총 $5 \times 10^7$ 타임스텝을 사용했으며, 첫 $3 \times 10^7$ 동안 reward shaping을 적용하였다. UPD의 경우 partner buffer 크기 $|B| = 512$, 파트너당 평가 rollout $N = 10$, Dirichlet 파라미터 $\alpha = 1.0$ 을 사용했으며, 네 번째 학습 루프마다 buffer를 완전히 갱신(refresh)하였다. 하이퍼파라미터, 계산 인프라, 추가 결과 및 학습 곡선에 대한 세부사항은 부록에 있다.

> **Table 1.** 평가 population에 대한 평균 return (mean ± std.). 환경의 두 시작 위치 모두에 대해 평균을 냈다. 최고 결과는 **굵게**, 두 번째는 밑줄로 표시. 평균적으로 UPD가 모든 베이스라인을 능가한다. ablation인 *UPD w/o ℓ* 가 두 번째로 우수하다. 이는 online partner 생성과 learnability 채점의 이점을 부각한다.

| Method | CRoom | AA | CR | CC | FC | Average |
|---|---|---|---|---|---|---|
| SP | 64.0 ± 11.2 | 64.4 ± 21.6 | 25.2 ± 6.4 | 14.5 ± 4.7 | 33.4 ± 8.7 | 40.4 ± 10.5 |
| FCP | 44.2 ± 10.3 | 68.6 ± 26.7 | 48.2 ± 12.9 | 14.4 ± 4.1 | 50.6 ± 6.7 | 45.2 ± 11.2 |
| MEP | 100.5 ± 12.1 | 136.2 ± 19.7 | 40.3 ± 3.8 | 12.7 ± 8.7 | 48.8 ± 2.5 | 67.8 ± 9.6 |
| E3T | 101.3 ± 8.0 | 127.9 ± 21.0 | 58.4 ± 5.2 | 55.0 ± 5.8 | 41.3 ± 6.7 | 76.8 ± 9.7 |
| UPD w/o bias | 102.1 ± 6.7 | 159.9 ± 23.0 | 66.2 ± 6.0 | 57.1 ± 6.7 | 44.4 ± 5.7 | 85.6 ± 11.1 |
| UPD w/o ℓ | 106.9 ± 6.4 | _164.0 ± 28.4_ | _68.9 ± 7.4_ | **64.5 ± 8.5** | 45.7 ± 4.4 | _90.0 ± 11.0_ |
| **UPD (Ours)** | **108.1 ± 5.5** | **181.4 ± 15.1** | **69.2 ± 7.2** | **64.5 ± 8.7** | **48.7 ± 3.7** | **94.4 ± 8.1** |

Table 1은 UPD가 모든 베이스라인을 능가하며, 다양하고 미관측(unseen)인 파트너와 짝지어졌을 때 가장 높은 평균 return을 달성함을 보인다. 다른 방법들은 일반화에 어려움을 겪는데, 이는 평가 population의 높은 행동 다양성과 능력 범위 때문일 가능성이 크다. 이 결과는 뒤에 제시되는 RQ3의 인간 평가 결과로 추가 뒷받침된다.

Figure 3은 UPD가 레이아웃마다 서로 다른 $\epsilon$ 값을 선호하며, 흔히 E3T가 사용하는 고정값에서 벗어남을 보인다. 더 경직된 협력 관습(rigid cooperation convention)을 갖는 레이아웃(예: CR, CC, FC)에서 UPD는 낮은 $\epsilon$ 쪽으로 끌리고, 더 유연한 레이아웃(예: CRoom, AA)에서는 더 확률적인 파트너를 선호한다. **이는 UPD가 과제 요구에 따라 partner 능력을 동적으로 조정함을 보여준다.**

> **Figure 3.** 학습 동안 레이아웃별로 $\ell_{\text{var}}$ 이 서로 다른 $\epsilon$ 값을 어떻게 선호하는지 비교. 더 좁은 협응 도전과제를 갖는 것으로 알려진 레이아웃(CC, CR, FC)에서 UPD는 self-play에 가까운 $\epsilon$ 을 선호하는 반면, CRoom과 AA에서는 더 무작위적인 파트너를 선호한다.

Figure 4는 UPD가 **창발적 관습 깨기(emergent convention-breaking)** 를 유도함을 드러낸다: 학습 초기에 파트너는 방향성 편향(예: "오른쪽" 선호)을 보이지만, 이후 반대 선호로 전환한다. 우리는 이것이 ego agent의 파트너 행동에 대한 기대를 위반함으로써 return 분산을 증가시킨다고 가정한다. 이 효과는 레이아웃과 시드 전반에서 일관된다. 선행 zero-shot 협응 접근법들은 흔히 관습 깨기를 위해 수작업으로 설계된 메커니즘을 요구하지만(Hu et al. 2020), 우리는 관습 깨기와 유사한 현상이 UPD에서 자연스럽게(organically) 창발함을 발견하였다.

### 5.2 RQ2: Ablations

우리는 Table 1(하단)에서 두 가지 ablation과도 비교하였다: *UPD w/o bias* 는 partner 혼합에 균등 무작위 정책 $\pi_r$ 을 사용하고, *UPD w/o ℓ* 은 learnability 채점을 제거하고 대신 rollout마다 무작위 파트너를 샘플링한다. 각 변형은 여전히 경쟁력 있는 AHT 에이전트를 만들지만, 이들을 전체 UPD에서 결합했을 때 최고의 결과를 낸다 — 이는 확률적 bias와 적응적 선택이 *둘 다* 유의미하게 기여함을 확인해 준다. 흥미롭게도 *UPD w/o ℓ* 이 꽤 잘 동작하는데, 이는 online partner generator만으로도 넓은 범위의 partner 능력을 샘플링하여 자연스러운 curriculum을 유도하기 때문일 가능성이 크다. 특히 E3T는 추가적인 ablation으로 볼 수 있다: 이는 고정 partner generator를 사용하고 learnability 기준이 없다. UPD는 이 세 ablation 모두를 능가하여, 동적이고 learnability 주도적인 curriculum의 이점을 강조한다.

> **Figure 4.** 학습에 걸쳐 partner 생성에 더해진 방향성 행동 편향. 우리는 UPD가 **창발적 대칭 깨기(emergent symmetry breaking)** 를 유도함을 발견한다 — 파트너가 처음에는 한 관습(예: 왼쪽) 쪽으로 편향되었다가 전환한다. 이러한 전환은 적응적 curriculum 동역학을 반영하며, learnability를 최대화하는 것이 어떻게 관습 탐색을 장려하는지 부각한다.

다음으로, 우리는 서로 다른 learnability 함수가 partner 선택과 학습 성능에 어떻게 영향을 미치는지 검토하였다. 구체적으로 우리는 다음을 물었다: 우리가 제안한 분산 기반 점수 $\ell_{\text{var}}$ 은 효과적인 curriculum을 유도하는가? 그리고 UPD는 learnability 함수의 선택에 얼마나 민감한가? 우리는 RQ1의 다섯 개 Overcooked-AI 레이아웃에 대해 다음 비교를 사용하여 네 가지 함수를 평가하였다: (1) $\ell_{\text{mean}} = \mathbb{E}_{\tau \sim (\pi, \pi_p)}[R(\tau)]$ — 더 높은 기대 return을 갖는 파트너를 선택. (2) Rutherford et al. (2024)의 성공률 기반 learnability를 연속 보상으로 변용한 것으로, return을 중앙값(median)에서 임계화하여 추정 의사-성공률(pseudo-success rate)을 사용:

$$\ell_{\text{adaptive-SR}} = p \cdot (1 - p), \tag{6}$$
$$\text{여기서 } p = 1 \text{ if } R(\tau) > \text{Median, else } 0. \tag{7}$$

(3) (Monette et al. 2025)의 Gauss 가중 정식화. (4) 우리의 $\ell_{\text{var}}$. 종합하면 이들은 단순한 보상 주도(naive reward-driven) 전략과 관련 연구에서 식별된 다른 선택 전략들을 아우른다.

Figure 5에 보이듯, 우리의 $\ell_{\text{var}}$ 이 가장 높은 성능을 달성한다. 특히 UPD는 대안 함수들 전반에서 강건하게 유지된다 — 그 함수들이 병리적 경우(pathological case, 예: self-play)를 피하는 한에서. 선행 연구 결과(Monette et al. 2025)와 달리, 우리는 적어도 partner 선택에 관해서는 UED에서 쓰이는 더 복잡한 learnability 함수가 명확한 이점을 제공하지 않음을 관찰한다.

> **Figure 5.** 레이아웃에 걸쳐 평균낸, 평가 population에 대한 모든 learnability 함수의 성능. 우리의 단순한 $\ell_{\text{var}}$ 이 더 복잡한 측도들과 동등하거나 더 나은 성능을 낸다. 또한 UPD는 함수가 합리적인 한 learnability 함수에 강건하다.

### 5.3 RQ3: Human-AI Teamwork Study

Overcooked-AI에 대한 마지막 실험으로, 우리는 이중 맹검(double-blind) 사용자 연구에서 UPD가 인간과 함께 어떻게 동작하는지 평가하였다. 우리는 네 개의 대표 에이전트 — SP, MEP, E3T, UPD — 를 도전적인 협응 동역학을 갖는 세 레이아웃(AA, CR, CC)에서 평가하였다. 12명의 참가자(연령 26–34세, 여성 5명)를 모집했으며, 각 참가자는 모든 에이전트-레이아웃 조합을 무작위 순서로 플레이하였다. 총 144개 게임을 수집했으며, 방법당 36개 게임이 플레이되었다. 시험에 앞서 튜토리얼 세션이 진행되었고, 기관 윤리 지침에 따라 보상이 제공되었다. 본 연구는 기관 윤리심의위원회의 승인을 받았다.

Figure 6은 UPD가 인간과 상호작용할 때 모든 베이스라인보다 유의미하게 높은 return을 달성하며(오른쪽), 대부분의 주관적 설문 항목에서도 선호됨(왼쪽)을 보인다: 참가자들은 UPD가 **유의미하게 더 적응적이고, 더 인간 같으며, 더 나은 협력자이고, 함께 일하기에 덜 답답하다**고 느꼈다. 전반적 주관적 선호를 평가하기 위해, 우리는 모든 설문 항목에 걸쳐 개별 응답을 집계하였다. 평정은 높은 내적 일관성(Cronbach's $\alpha = 0.938$)을 보여, 합성 점수(composite score) 사용을 정당화하였다. 이 집계에 기반하여, UPD는 다른 모든 에이전트보다 유의미하게 선호되었다(Wilcoxon signed-rank 검정, Holm-Bonferroni 보정, 모든 경우 $p < 0.05$). **UPD는 인공 파트너와 인간 파트너 모두에게 선호된다.**

> **Figure 6.** 파트너에 대한 인간 평가: Frust = '답답한가?'(↓), Adapt = '잘 적응했는가?'(↑), Human = '인간 같은가?'(↑), Coord = '잘 협응했는가?'(↑). 개별 설문 질문에 대해 단측 Wilcoxon signed-rank 검정을, return에 대해 단측 대응표본 t-검정을 수행하여 UPD를 각 베이스라인과 비교하였다. 유의수준은 세 번의 비교를 고려해 질문별로 Holm-Bonferroni 보정하였다($* = p < 0.05$, $** = p < 0.01$, $*** = p < 0.001$). 결과는 UPD가 대부분의 설문 항목에서 유의미하게 높게 평가되며 가장 높은 평균 return을 산출함을 보인다.

### 5.4 RQ4: Joint Environment and Partner Design

우리의 중심 목표 중 하나는 기존 UED 방법과 매끄럽게 통합될 수 있는 경량 알고리즘을 설계하는 것이었다. 이를 시험하기 위해, 우리는 OGC(Ruhdorfer et al. 2025) 내에서 UPD를 평가하였다. OGC는 level이 무작위로 생성되고 풀 수 있다는 보장이 없어 UED 메커니즘에 큰 부담을 주는, 특히 어려운 환경이다.

우리는 UPD를 SFL과 결합하여 결합 방법인 JUPD를 구현하였다: Alg. 1의 표준 partner/environment 샘플링을 Alg. 3에 기술된 결합 샘플링·채점 절차로 대체하였다. 처리 가능성(tractability)을 위해 OGC의 5×5 버전을 사용했으며, 1,024개의 병렬 환경으로 모든 방법을 $10^9$ 스텝 동안 학습하였다. 평가는 보류된(held-out) level-partner 조합과 보류된 레이아웃에서, RQ1의 조정된 평가 population을 사용해 수행하였다.

우리는 level과 partner 생성을 결합하는 여러 베이스라인과 JUPD를 비교하였다:
- **DR-DR** 은 파트너와 level을 무작위로 샘플링한다.
- **Cross-Environment-Cooperation (CEC)** 는 level을 무작위로 선택하고 self-play로 플레이한다. CEC는 level generator가 평가 레이아웃에 가까울 때 큰 성공을 보였다(Jha et al. 2025).
- **SFL-E3T** 는 문헌의 장점을 결합하여, SFL로 level을 선택하고 $\epsilon = 0.5$¹ 로 E3T 파트너를 생성한다.

> **Table 2.** 5×5 OGC에서 환경과 partner의 결합 비지도 일반화에 대한 결과 (mean ± std.). JUPD는 이 도전적인 벤치마크에서 경쟁력 있는 베이스라인을 능가한다. 최고 결과는 **굵게**, 두 번째는 밑줄로 표시. CRoom에서는 에이전트가 특정 단일 레이아웃에 맞춰 학습된 베이스라인에 근접한 성능을 보이지만, 일반적으로 성능은 그에 뒤처진다.

| Method | CRoom | CR | FC | Average |
|---|---|---|---|---|
| DR-DR | _86.4 ± 7.2_ | _53.6 ± 8.3_ | _9.7 ± 2.4_ | _49.9 ± 5.1_ |
| CEC | 41.4 ± 9.2 | 20.3 ± 7.0 | 12.7 ± 3.5 | 23.9 ± 6.4 |
| SFL-E3T | 81.7 ± 7.8 | 41.1 ± 10.1 | 7.4 ± 2.5 | 44.0 ± 5.1 |
| **JUPD (Ours)** | **97.0 ± 7.4** | **60.0 ± 6.1** | **17.1 ± 4.2** | **58.9 ± 5.9** |

Table 2에 보이듯, JUPD는 모든 레이아웃에서 모든 베이스라인을 능가하며, 이 도전적 설정에서 가장 높은 평균 return을 달성한다. 구조화된 partner 적응이 결여된 DR-DR·CEC나, 정적 partner 생성을 사용하는 SFL-E3T와 달리, JUPD는 학습 동안 level과 partner 난이도를 *모두* 동적으로 적응시킨다. 시간에 따라 샘플링된 level은 Figure 7에 시각화되어 있다.

Overcooked-AI와 OGC 사이의 관측(observation)에 약간의 차이가 있으므로, Table 1과 Table 2의 결과는 주의해서 비교해야 함에 유의하라.

> **Figure 7.** 시간에 따른 JUPD buffer 내 level들 (Early → Middle → Final).

> ¹ 관련 연구에서 흔히 쓰이는 선택이므로 $\epsilon = 0.5$ 를 사용한다.

---

## 6 Discussion & Limitations (논의 및 한계)

UPD는 DCD의 철학을 반영하는, AHT를 위한 일반적 curriculum 프레임워크를 도입한다: DCD가 level 난이도를 적응시키듯, UPD는 online 생성과 선택을 통해 partner 난이도를 적응시킨다. 이러한 점에서 UPD는 선행 AHT 접근법들 — 예컨대 E3T, FCP, MEP — 을 고정 generator나 정적 population 아래의 특수 사례로 통합하면서도, 더 강한 성능을 달성한다. 향상된 성능을 넘어, UPD는 명시적 설계 없이도 대칭 깨기(symmetry breaking)나 과제 특화 적응(task-specific adaptation) 같은 창발적 행동을 만들어 내며, 이는 비지도 partner 형성(unsupervised partner shaping)을 일반화 가능한 멀티에이전트 협력을 위한 유망한 도구로 자리매김한다.

효과적이긴 하지만, UPD는 현재 이산 행동 공간(discrete action space)을 대상으로 하며 가능한 행동 다양성의 일부 — 주로 능력(competence)과 저수준 행동 편향(low-level action bias) — 만을 포착한다. UPD를 더 풍부한 generator(예: latent-conditioned 정책(Eysenbach et al. 2019) 사용)로 확장하면 더 다양한 선호·의도·목표를 갖는 파트너를 가능하게 할 수 있다. 우리는 이를 협력적 AI를 위한 open-ended 학습의 유망한 향후 연구 방향으로 본다.

마지막으로, 우리의 결과는 인간 평가(human evaluation)의 중요성을 강조한다: UPD가 베이스라인을 일관되게 능가하지만, 일부 방법들(예: MEP vs. E3T)은 인간과 짝지어졌을 때와 인공 에이전트와 짝지어졌을 때의 순위가 다르다. 이는 지속적인 간극을 부각한다: 에이전트-에이전트 평가만으로는 인간-에이전트 정합(human-agent alignment)을 신뢰성 있게 예측하지 못할 수 있으며, 따라서 협력 능력을 측정하는 데 사용자 연구가 필수적이다.

---

## 7 Conclusion (결론)

우리는 **Unsupervised Partner Design (UPD)** — partner 정책 공간에서 learnability 주도 curriculum을 사용하는, 새롭고 population-free한 ad-hoc teamwork 프레임워크 — 를 도입하였다. unsupervised environment design과 통합함으로써, UPD는 과제와 파트너 양쪽에 대한 결합 적응(joint adaptation)을 지원하여, 멀티에이전트 일반화의 핵심 과제를 다룬다. 우리는 UPD가 사전학습된 파트너나 수동 튜닝 없이도 강건한 zero-shot 협응을 가능하게 하며, Overcooked-AI에서 강력한 베이스라인을 능가함을 보이는 광범위한 평가를 보고하였다. 또한 우리는 UPD를 인간과 함께 평가했으며, UPD가 모든 베이스라인보다 유의미하게 높은 return을 달성할 뿐 아니라 유의미하게 더 적응적이고, 더 인간 같으며, 더 나은 협력자이고, 덜 답답하다고 인식됨을 발견하였다. 종합하면, 우리의 연구는 비지도·분산 기반 partner 생성 및 선택이 더 적응적인 멀티에이전트 시스템을 향한 확장 가능한(scalable) 경로를 제공함을 보여준다.

---

## Acknowledgments (감사의 글)

저자들은 C. Ruhdorfer와 V. Oei를 지원해 준 International Max Planck Research School for Intelligent Systems (IMPRS-IS)에 감사한다. A. Penzkofer는 독일 연구재단(Deutsche Forschungsgemeinschaft, DFG)의 독일 우수전략(Germany's Excellence Strategy) — EXC 2075 – 390740016 — 의 지원을 받았다.

---

## References (참고문헌)

> 참고문헌 목록은 원문(저자·제목·게재처)을 그대로 유지합니다. 원본 PDF의 References 절을 참조하세요. 본문에서 인용된 주요 문헌은 다음과 같습니다(약식): Stone et al. (2010, AHT); Dennis et al. (2020, UED/PAIRED); Jiang et al. (2021, DCD/REPAIRED); Jiang, Grefenstette, and Rocktäschel (2021, PLR); Yan et al. (2023, E3T); Strouse et al. (2021, FCP); Zhao et al. (2023, MEP); Rahman et al. (2023, BRDiv); Carroll et al. (2019, Overcooked-AI); Ruhdorfer et al. (2025, OGC); Rutherford et al. (2024, SFL); Monette et al. (2025); Wang et al. (2025, ROTATE); Jha et al. (2025, CEC); Hu et al. (2020, Other-Play); Eysenbach et al. (2019, DIAYN); de Witt et al. (2020, IPPO); Schulman et al. (2017, PPO); Cho et al. (2014, GRU). 전체 서지정보는 원문을 따른다.

---

# Appendix (부록)

## A Infrastructure & Tools (인프라 및 도구)

우리는 94GB 메모리의 NVIDIA H100-NVL GPU와 AMD EPYC 9454 CPU를 갖춘 서버 시스템에서 실험을 수행하였다. 모든 학습 실행은 단일 GPU에서만 이루어진다. 우리는 JAX(Bradbury et al. 2018), Flax(Heek et al. 2023), Optax(DeepMind et al. 2020)를 사용해 모델을 학습하였다. 분석 도구로는 NumPy(Harris et al. 2020), Pandas(Team 2020), SciPy(Virtanen et al. 2020), Matplotlib(Hunter 2007)을 사용하였다. 우리의 단일 파일(single-file) IPPO 구현은 JaxMARL(Rutherford et al. 2023)이 제공하는 것에 기반한다.

우리의 실험은 이 시스템이 제공하는 계산 능력의 일부만을 필요로 한다. 최소한, 우리의 실험은 16GB 메모리의 GPU(아마 그보다 적어도 가능)로 재현 가능하다. Overcooked-AI 실험은 분(minutes) 단위로 실행되며, OGC 실험은 보통 몇 시간이 걸리지만 하루 이내에 충분히 끝난다.

## B Reproducibility Statement (재현성 진술)

우리의 연구를 재현하기 위해, 본 문서에 모든 핵심 하이퍼파라미터와 하이퍼파라미터 범위를 제공한다. 코드는 다음에서 이용 가능하다: https://git.hcics.simtech.uni-stuttgart.de/public-projects/UPD

> **Table 3.** Overcooked-AI 실험(SP, FCP, MEP, E3T, UPD)에 사용된 하이퍼파라미터.

| 범주 | 하이퍼파라미터 | 값 |
|---|---|---|
| | 환경 수 | 512 |
| | 총 타임스텝 | $5 \times 10^7$ |
| | Reward shaping horizon | $3 \times 10^7$ |
| | 학습률 | $1 \times 10^{-3}$ |
| | 학습률 annealing | Linear |
| | 사용 시드 | 0 – 5 |
| **PPO** | PPO rollout 길이 | 400 스텝 |
| | PPO epochs | 6 |
| | 갱신당 minibatch 수 | 8 |
| | 할인계수 ($\gamma$) | 0.99 |
| | GAE 파라미터 ($\lambda$) | 0.95 |
| | Clipping 임계값 ($\epsilon$) | 0.2 |
| | 엔트로피 계수 | 0.01 |
| | Value loss 계수 | 1.0 |
| | Gradient norm clipping | 0.5 |
| **아키텍처** (actor·critic 공유) | Embedding 레이어 | 2 |
| | Actor 레이어 | 4 |
| | Critic 레이어 | 4 |
| | Fully connected 레이어 크기 | 256 |
| | GRU hidden 크기 | 256 |
| | 활성화 함수 | Tanh |
| | Layer normalization | Enabled |
| **Partner modeling** (E3T/UPD only) | 보조 모델 깊이 | 4 레이어 (크기 64) |
| | MOA loss 계수 | 1.0 |
| | Trajectory history 길이 | 5 스텝 |
| | Action embedding 크기 | 256 |
| | Prediction normalization | L2 norm |
| **UPD 전용** | 생성 에이전트 수 | 8192 |
| | Buffer 크기 ($\|B\|$) | 512 |
| | $N$ | 10 |
| | Buffer refresh 빈도 | 4 학습 루프 |

> **Table 4.** Overcooked-AI 실험(SP, E3T, UPD)에 사용된 학습·아키텍처 하이퍼파라미터 탐색 공간. 선택값은 **굵게** 표시. FCP와 MEP의 경우 때때로 약간 다른 하이퍼파라미터를 찾았으며 그 경우 밑줄로 표시. 어떤 파라미터를 유지할지 결정하기 위해, 저렴한 대리지표(cheap proxy)로서 BRDiv population에 대한 평가 결과를 살펴보았다. 일부 하이퍼파라미터(예: 총 타임스텝, reward shaping horizon)는 FCP와 MEP에 대해서만 튜닝되었다. OGC 실험에서는 위에서 찾은 굵은 값들을 재사용하고, 환경 수(1024), 스텝($1 \times 10^9$), actor·critic 레이어(각각 1개 추가)만 늘렸다(본문 참조).

| 범주 | 하이퍼파라미터 | 범위 |
|---|---|---|
| | 총 타임스텝 | **$5 \times 10^7$**, $1 \times 10^8$ |
| | Reward shaping | **$3 \times 10^7$**, $5 \times 10^7$ |
| | 학습률 | **$1 \times 10^{-3}$**, $3 \times 10^{-4}$, $1 \times 10^{-4}$ |
| | PPO epochs | 4, **6** |
| | PPO minibatches | 4, 6, **8** |
| | Layernorm? | **True**, False |
| **UPD 전용** | Buffer 크기 ($\|B\|$) | 128, 256, **512** |
| | $N$ | 3, **10** |
| | Buffer refresh 빈도 | 1, 2, **4** |

## C Additional Training Details (추가 학습 세부사항)

### C.1 Reinforcement Learning Details

본문의 모든 방법에 대해 우리는 independent PPO(de Witt et al. 2020; Schulman et al. 2017)를 사용한다. 모든 하이퍼파라미터의 개요는 Table 3에 제시한다. 모든 방법에 대해, 해당 하이퍼파라미터에 도달하기 위해 소규모 예비 하이퍼파라미터 탐색을 수행하였다. 그 범위는 Table 4에 제시한다. OGC 실험은 한 가지 작은 차이를 제외하면 정확히 동일한 하이퍼파라미터를 사용하였다: actor와 critic 레이어를 각각 1개씩 추가(각 5개)하고, 1,024개 환경에서 총 $1 \times 10^9$ 타임스텝을 학습하였다.

### C.2 Neural Network Architecture

우리는 recurrent actor-critic 아키텍처를 사용한다. 모델은 공유 encoder, recurrent 처리 모듈, 그리고 정책(policy)·가치(value) 추정을 위한 분리된 head들로 구성된다.

각 에이전트의 관측은 선형 레이어 다음에 설정 가능한 수의 fully connected 레이어(기본값: 2개)로 이루어진 feedforward encoder를 통과한다. 각 레이어는 256개의 hidden unit을 가지며 ReLU 또는 Tanh 활성화를 사용하고, 선택적으로 layer normalisation이 뒤따른다. 그 결과 표현(representation)은 recurrent 모듈의 입력으로 사용된다.

시간적 의존성(temporal dependency)은 hidden state 크기 256의 Gated Recurrent Unit (GRU)(Cho et al. 2014)으로 포착된다. recurrent hidden state는 환경의 종단 상태(terminal state)에서 리셋된다. GRU 출력은 시간적 embedding으로서 actor와 critic head 모두에 공유된다.

policy head는 recurrent embedding을 각 256 unit의 fully connected 레이어 4개와 비선형 활성화를 통해 처리한다. 마지막 선형 레이어가 이산 행동에 대한 비정규화 logit을 출력하여 categorical 행동 분포를 정의한다. value head 또한 recurrent embedding을 입력으로 사용하며, 각 256 unit의 fully connected 레이어 4개를 적용한 뒤 상태 가치 추정치를 나타내는 스칼라 출력을 낸다.

우리는 E3T가 제안한 other agent modelling 네트워크를 사용한다. 이 보조 모듈은 팀메이트(teammate)의 행동을 모델링한다. 각 에이전트는 상대 에이전트의 지난 5개 상태-행동 쌍을 받는다. 관측과 행동은 embedding되어 4레이어 다층 퍼셉트론(레이어당 64 unit)을 통과해 팀메이트의 다음 행동 분포를 예측한다. 예측값은 L2 정규화되어 에이전트 자신의 embedding과 연결(concatenate)된 후 policy head에 입력된다. 이 보조 손실(auxiliary loss)은 cross-entropy 목적함수로 최적화되며 튜닝 가능한 계수로 가중된다. 이는 원래 정식화(Yan et al. 2023)와 일관된다.

## D Additional Environment Details (추가 환경 세부사항)

### D.1 Overcooked-AI

Overcooked-AI는 Overcooked 비디오 게임에 기반한 멀티에이전트 협응 벤치마크로, 원래 Carroll et al. (2019)이 제안하였다. 시간적 협응과 공간적 추론을 요구하는 2인용 요리 과제를 특징으로 한다. 에이전트는 재료를 집고, 냄비에 조리하고, 요리를 서빙해야 하며, 부분 관측성(partial observability)과 희소 보상(sparse reward)을 갖는다. 행동 공간은 이산적이며 여섯 가지 행동으로 구성된다: 위·아래·왼쪽·오른쪽 이동, interact(상호작용), stay(머무름).

우리는 ad-hoc teamwork 문헌에서 흔히 쓰이는 다섯 개의 표준 레이아웃을 사용한다(Figure 8 참조): Cramped Room (CRoom), Asymmetric Advantages (AA), Coordination Ring (CR), Counter Circuit (CC), Forced Coordination (FC). 우리는 JaxMARL이 제공하는 구현을 사용하고 모든 보상 설정을 기본값으로 유지한다(서빙에 대한 sparse reward, 표시된 중간 행동에 대한 shaped reward). 학습 시 기본적으로 sparse reward 설정을 사용하되, 첫 $3 \times 10^7$ 타임스텝 동안 reward shaping을 활성화한다. shaped reward 단계 동안, 에이전트는 냄비에 재료를 넣으면 보상 3, 수프가 조리 중일 때 접시를 집으면 보상 3, 준비된 수프를 집으면 보상 5를 받는다.

### D.2 The Overcooked Generalisation Challenge

Overcooked Generalisation Challenge (OGC)(Ruhdorfer et al. 2025)는 Overcooked-AI를 확장하여 partner 분포와 level 분포 양쪽에 걸친 zero-shot 일반화를 평가한다. 고정 레이아웃에서 학습하는 대신, 환경은 다양한 위상(topology)과 난이도를 갖는 무작위 주방을 생성하는 절차적(procedural) level generator를 포함한다. 이 도전과제는 특히 까다로운데, 생성된 많은 레이아웃이 풀 수 없거나 효율적으로 협응하려면 비자명한 관습을 요구하기 때문이다.

우리 연구에서는 과제 다양성을 보존하면서 계산 비용을 줄이기 위해 OGC의 5×5 버전을 사용한다. 이 generator는 주방 구조(벽, 카운터, 아이템 배치)와 목표 설정(예: 수프당 양파 개수)을 무작위로 샘플링한다. generator는 먼저 경계에 벽이 있는 레이아웃을 생성한 다음, 1에서 10 사이의 벽 예산(wall budget)을 무작위로 샘플링한다. 이 예산으로 시스템은 벽을 무작위로 배치한다. generator는 또한 레이아웃을 좁히기 위해 분할 벽(dividing wall)이나 측면 추가 벽을 무작위로 더한다. 그 후 시스템은 벽 위에 아이템을 배치한다. 마지막으로, 두 에이전트가 빈 타일에 배치된다.

우리는 OGC 벤치마크에서 사용되는 작은 레이아웃의 고정 부분집합에서 성능을 평가하고, 이러한 생성된 환경 전반에서 보류된(held-out) 에이전트와 짝지어졌을 때의 평균 return을 기준으로 에이전트를 비교한다.

> **Figure 8.** 비-Jax 버전 Overcooked의 다섯 개 평가 레이아웃을 Carroll et al. (2019)의 그림을 사용해 재수록. 왼쪽부터: CR, AA, CR, CC, FC.

> **Figure 9–13.** 각각 SP, FCP, MEP, E3T, UPD의 학습 곡선(Mean ± Std, 6개 시드 평균). 레이아웃별 평균 에피소드 return을 환경 스텝에 대해 표시.

## E Training Curves (학습 곡선)

우리는 모든 Overcooked-AI 방법의 수신 학습 return을 Figure 9(SP), 10(FCP), 11(MEP), 12(E3T), 13(UPD)에 표시한다. 이 return들은 주의해서 해석해야 함에 유의하라. FCP와 MEP의 경우 각자의 population과의 return을 표시한다. UPD와 E3T의 경우 생성된 파트너에 기반한 수신 학습 return을 표시한다. 어느 경우든 모든 그림은 방법들이 안정적으로 수렴함을 보인다. 또한 많은 방법이 UPD보다 높은 학습 return을 달성하지만, 문헌에서 확립되었듯이(Carroll et al. 2019) 이것이 알려지지 않은 파트너와의 테스트 return을 반드시 예측하지는 않음에 유의하라.

Figure 14에서 우리는 네 개 OGC 방법 모두의 학습 return을 표시한다. 모든 방법이 수렴함을 본다. CEC만이 끝에서 성능 하락을 겪는 것으로 보인다. 다시 한번, 학습 성능이 테스트 시점 성능을 예측하지 않음에 유의하라: learnability로 level을 샘플링하는 방법(SFL-E3T, JUPD)은 그렇지 않은 방법(DR, CEC)보다 더 어려운 level을 샘플링할 수 있으며, SFL-E3T와 JUPD 모두 self-play 설정에서 플레이하지 않는다.

> **Figure 14.** OGC 학습 곡선. 모든 방법(DR, CEC, SFL-E3T, JUPD)에 대해 6개 시드 평균과 표준편차를 표시.

## F Partner Curriculum Dynamics (Partner Curriculum 동역학)

UPD가 Overcooked-AI 학습 전반에서 효과적인 학습 파트너를 어떻게 식별하는지 더 잘 이해하기 위해, 우리는 선행 연구(Rutherford et al. 2024; Monette et al. 2025)와 유사하게 학습에 걸친 learnability 점수를 시각화한다. 세 레이아웃(CR, AA, FC)에 대해, 학습된 ego agent를 학습 중 세 시점(2/5, 4/5, 끝)에서 8,192개의 무작위 생성 파트너와 rollout하여 평가한다(Figure 15, 16, 17 참조). 이 세 레이아웃을 선택한 이유는 대표적이기 때문이다: CRoom과 CC는 모두 CR 사례와 매우 유사하게 행동한다. 우리는 학습 때와 동일한 분산 기반 지표로 learnability를 계산하고 이를 대응하는 평균 return에 대해 도시한다. 모든 설정에서, learnability가 최고 또는 최저 return 파트너에 집중되지 않으며, 대부분의 파트너가 위치한 곳에서 정점을 이루지도 않음을 관찰한다. 대신 learnability는 중간 난이도(intermediate difficulty)의 파트너에서 높다. 대부분의 level에서 생성된 파트너는 낮은 return·낮은 learnability 버킷에 속함을 관찰한다. 이는 우리의 가설 — *모든 파트너가 학습에 최적인 것은 아니다* — 이 옳음을 시사한다: 대부분의 파트너는 learnability 지표에서 낮은 점수를 받는다. 대신 UPD는 중간 return 버킷의 파트너를 가장 높게 순위 매기며, 이 버킷에는 소수의 파트너만 분포한다.

마지막으로, 이진 결과를 갖는 level-space curriculum에 관한 선행 연구(Rutherford et al. 2024; Monette et al. 2025)와 달리, 우리의 설정에서는 에이전트가 어떤 파트너와도 보상 0을 받는 경우가 드물다. 이 점과 연속 보상 구조로 인해, 우리의 learnability 분석 도표는 그 *형태*뿐 아니라 시간에 따른 *오른쪽 이동(rightward shift)*을 통해서도 정보를 전달함을 발견한다. 예를 들어 Figure 16에서, 최저 점수 파트너는 학습 초기와 중기 사이에 평균 return이 약 20에서 120으로 향상된다. 이 이동은 ego agent의 능력이 시간에 따라 확장되며 UPD가 점점 더 유능한 파트너를 샘플링함으로써 적응함을 시사한다. 그렇게 함으로써 UPD는 정보가 풍부한 파트너를 식별할 뿐 아니라 에이전트의 학습 진척을 추적하여 curriculum을 그에 맞게 조정한다. 주목할 예외는 FC와 같이 강한 상호의존성(strong interdependence)을 갖는 과제에서 발생한다(Figure 17 참조).

> **Figure 15–17.** 각각 Coordination Ring, Asymmetric Advantages, Forced Coordination에서 학습에 걸친 learnability 대 return (Early / Middle / Final).

## G Additional Results (추가 결과)

Figure 18에서 우리는 UPD가 서로 다른 파트너와 어떻게 동작하는지에 대한 추가 결과를 표시한다. 평균적으로, 그리고 대부분의 파트너와 함께, UPD가 가장 잘 동작한다. UPD는 onion 에이전트와 함께일 때만 MEP에 근소하게 뒤지지만, 본문에서도 논의했듯 다른 모든 면에서 UPD가 MEP를 능가한다.

> **Figure 18.** Overcooked-AI에서 레이아웃에 걸쳐 평균낸 파트너별(Per-Partner) 결과. UPD는 대부분의 파트너와 함께 다른 방법들을 능가한다. 일부 방법은 Onion 및 독립 계획(independent planning) 에이전트와 함께 유사하거나 약간 더 나은 성능을 보이지만, 이들은 UPD보다 일관되게 우수하지는 않다.

## H Additional Notes on Alternative Learnability Formulations (대안적 Learnability 정식화에 대한 추가 노트)

우리의 실험은 UPD가 AHT 에이전트 학습에 매우 효과적이며, learnability의 서로 다른 정식화가 self-play 설정으로 붕괴하지 않는 한 그것들에 강건함을 보였다. 또한 우리는 UPD가 표준 UED에 통합되어 환경과 partner 분포 양쪽에 대한 완전 비지도 curriculum을 가능하게 함을 입증하였다. 우리는 여러 learnability 함수를 탐구했으나 많은 대안이 남아 있다. 자연스러운 베이스라인 하나는 *가장 어려운 파트너* — 평균 return이 가장 낮은 파트너 — 를 우선시하는 것으로, 선행 연구에서 탐구되었다(Zhao et al. 2023; Li et al. 2023; You et al. 2025). 그러나 이론적으로도 예비 실험에서도, 우리는 이 접근법이 적대적 동역학(adversarial dynamics)으로 이어짐을 발견한다. 구체적으로, Overcooked-AI에서는 거의 무작위인 파트너($\epsilon \to 1$)를 크게 선호하고, OGC에서는 풀 수 없는 level을 선호한다. 두 경우 모두 학습이 정체된다. 이는 우리의 설정이, 적대적 행동을 암묵적으로 제한하는 사전학습 또는 유계(bounded) population에 의존하는 선행 연구와 달리, open-ended partner generator를 수반하기 때문이다. 그러한 제약된 설정에서는 "어려운" 예제를 우선시해도 실현 가능한 협응의 공간 안에 머물지만, (J)UPD에서는 제약 없는 난이도 선택이 학습 신호를 완전히 붕괴시킬 수 있다.

## I Additional User Study Details (사용자 연구 추가 세부사항)

우리는 피험자내 설계(within-subjects design)를 사용하였다: 각 참가자는 3개 Overcooked 레이아웃(Asymmetric Advantages, Cramped Room, Counter Circuit)에 걸쳐 4개 에이전트(SP, MEP, E3T, UPD) 모두와 상호작용하였다. 각 에이전트-레이아웃 쌍은 한 번씩 플레이되어, 참가자당 총 36개 에피소드가 되었다. 에이전트와 레이아웃 순서는 사용자별로 무작위화되었다. 각 전체 연구는 18–35분이 소요되었으며, 참가자는 본 연구에 앞서 짧은 튜토리얼을 완료하였다.

연구는 NiceWebRL 프레임워크²를 사용한 웹 기반 인터페이스로 진행되었다. 우리는 사용자 연구를 위해 12명의 참가자(여성 5명, 연령 26–34세)를 모집하였다. 본 연구는 기관 윤리심의위원회의 승인을 받았으며, 모든 참가자가 사전 동의(informed consent)를 제공하였다. 보상은 기관 지침에 따라 제공되었다. 참가자는 키보드 배치(화살표 키, 스페이스, s 키)를 사용해 캐릭터를 조작하였다. 편향을 피하기 위해 참가자는 에이전트의 정체를 알지 못했다(이중 맹검). 각 에이전트 상호작용 후, 참가자는 7개의 5점 리커트 척도(Likert-scale) 질문(strongly disagree, disagree, neutral, agree, strongly agree)으로 에이전트를 평정하였다. 질문은 다음과 같다:

1. 나는 이 에이전트와 플레이하는 것이 즐거웠다.
2. 에이전트가 나와 협응하는 능력은 다음과 같았다고 느꼈다: (very poor, poor, neutral, good, very good)
3. 에이전트는 의사결정을 할 때 나에게 적응했다.
4. 에이전트가 자주 내 길을 막았다. (부정 문항)
5. 에이전트는 자신의 행동에서 일관적이었다.
6. 에이전트의 행동은 인간 같았다.
7. 에이전트의 행동은 답답했다. (부정 문항)

우리는 추가로 연구 마지막에 사용자가 자유 형식 피드백을 줄 수 있게 하였다. 응답은 1부터 5까지 수치로 사상되었다. 전반적 주관적 선호에 대한 분석을 위해, 부정 가치(negative-valence) 문항은 집계 전에 반전(invert)하였다. 질문 응답의 내적 일관성을 평가하기 위해 Cronbach's $\alpha = 0.938$ 을 계산했으며, 이는 강한 신뢰도를 시사한다. 이는 질문 전반에 걸쳐 점수를 집계하여 에이전트당 단일 주관적 선호 점수를 산출하는 것을 정당화한다.

우리는 UPD를 각 베이스라인과 비교하는 단측 Wilcoxon signed-rank 검정을 수행하였다. 다중 비교를 통제하기 위해 각 질문 내에서 Holm–Bonferroni 보정을 적용하였다. 또한 동일한 절차로 집계된 선호 점수를 검정하였다. 성능(보상)은 Holm 보정과 함께 단측 대응표본 t-검정으로 분석하였다. 우리는 각 설문 질문에 대해 에이전트별 리커트 응답 분포를 보여주는 막대 그래프를 Figure 19에 제공한다.

> ² https://github.com/KempnerInstitute/nicewebrl

> **Figure 19.** 모든 에이전트에 걸친 각 설문 질문에 대한 인간 평정 분포. 각 막대는 각 리커트 항목(x축)에 주어진 응답 수를 나타내며, 색상은 에이전트를 나타낸다. 질문들은 주관적 인상(예: 즐거움, 일관성)과 협력 품질(예: 협응, 답답함)을 모두 포함한다.

---

*— 번역 끝 —*
