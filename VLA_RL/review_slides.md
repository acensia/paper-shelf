---
marp: true
theme: default
paginate: true
math: katex
title: From AWR to VLAs that Learn from Experience
description: Paper review — AWR (2019), π*₀.₆ / RECAP (2025), Learning to Fold (2026)
---

<style>
section {
  font-size: 18.5px;
  line-height: 1.42;
  padding: 34px 44px;
  justify-content: start;
}
section.lead {
  text-align: center;
  justify-content: center;
}
h1 { font-size: 1.7em; color: #1a3d5c; margin: 0 0 .4em; }
h2 {
  font-size: 1.28em; color: #1a3d5c;
  border-bottom: 2px solid #dde5ec;
  padding-bottom: .18em; margin: 0 0 .55em;
}
h3 { font-size: 1.02em; color: #2c5f8a; margin: .1em 0 .3em; }
p { margin: .45em 0; }
ul, ol { margin: .35em 0; padding-left: 1.15em; }
li { margin: .18em 0; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 1.3em; }
.cols ul { padding-left: 1em; }
.small { font-size: .86em; }
.tiny { font-size: .76em; color: #667; }
.box {
  background: #f4f7fa; border-left: 4px solid #2c5f8a;
  padding: .5em .8em; margin: .45em 0; border-radius: 3px;
}
.box p:first-child { margin-top: 0; }
.box p:last-child { margin-bottom: 0; }
.box ul { margin: .2em 0; }
.warn { background: #fdf5ee; border-left-color: #c47b30; }
.good { background: #f1f8f2; border-left-color: #3f8a4e; }
table { font-size: .84em; border-collapse: collapse; margin: .4em 0; }
th, td { padding: .22em .5em; }
.small table { font-size: 1em; }
.katex-display { margin: .5em 0 !important; }
footer { font-size: .6em; color: #889; }
section::after { font-size: .62em; color: #99a; }
</style>

<!-- _class: lead -->

# From AWR to VLAs that Learn from Experience

### Advantage-weighted policy improvement, from MuJoCo to real robots

<br>

**Advantage-Weighted Regression** · Peng, Kumar, Zhang, Levine — arXiv 1910.00177 (2019)
**π\*₀.₆: a VLA That Learns From Experience** · Physical Intelligence — arXiv 2511.14759 (2025)
**Learning to Fold** · Ilia Larchenko — arXiv 2606.27163 (ICRA 2026)

<span class="tiny">Paper review · VLA_RL study set</span>

---

## Why these three, in this order

<div class="box">
One idea, three scales: <b>make a policy better by regressing onto its own good actions, weighted by advantage.</b>
</div>

| | Paper | Contribution |
|---|---|---|
| **1** | AWR (2019) | The algorithm. Off-policy RL = two supervised regressions. |
| **2** | π\*₀.₆ / RECAP (2025) | The adaptation. Advantage as a *conditioning input*, so it works on flow-matching VLAs. |
| **3** | Learning to Fold (2026) | The recipe. AWR **and** RECAP together, plus everything engineering it takes to win. |

**The plot:** AWR's weighting scheme is elegant but throws data away. RECAP fixes that by conditioning instead of weighting. Learning to Fold shows they are complementary, not alternatives.

---

<!-- _class: lead -->

# Part 1 — Background

### RL, policy gradients, and reward-weighted regression

---

## The RL objective

An agent sees state $s_t$, samples $a_t \sim \pi(a_t|s_t)$, gets reward $r_t = r(s_t,a_t)$.

$$J(\pi) = \mathbb{E}_{\tau \sim p_\pi(\tau)}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right] = \mathbb{E}_{s \sim d_\pi(s)}\,\mathbb{E}_{a\sim\pi(a|s)}\left[r(s,a)\right]$$

where $d_\pi(s) = \sum_t \gamma^t p(s_t = s|\pi)$ is the (unnormalized) discounted state distribution.

Two quantities we will use constantly:

$$V^\pi(s) = \mathbb{E}\!\left[R_{s}\,\middle|\,\pi\right], \qquad A^\pi(s,a) = R^\pi_{s,a} - V^\pi(s)$$

<div class="box">

<b>Advantage</b> $A^\pi(s,a)$: how much better is taking $a$ in $s$ than what $\pi$ would have done on average? Positive = do more of this. This is the only signal any of the three papers ultimately uses.

</div>

---

## On-policy vs. off-policy — the axis the routes differ on

Before comparing algorithms, one distinction: **whose data are you learning from?**

<div class="cols">
<div>

### On-policy
The update assumes the data was collected by the **current** policy $\pi_\theta$.

- Collect fresh rollouts → take a gradient step → **throw the data away.**
- Statistically clean: expectations under $\pi_\theta$ are estimated with samples *from* $\pi_\theta$.
- Cost: every update requires new environment interaction.

</div>
<div>

### Off-policy
The update can consume data from a **different** policy $\mu$: past iterations of the agent, scripted controllers, human demonstrations.

- Store everything in a replay buffer, reuse it many times.
- Far more sample-efficient.
- Cost: a **distribution mismatch** — the data no longer reflects what the current policy would do, and something must correct for that (importance weights, trust regions, …).

</div>
</div>

<div class="box">

**Why this axis matters here:** on a real robot one sample is minutes of wall-clock, and the best data (demos, corrections) comes from humans — who are by definition not the current policy. So robot learning *must* be off-policy. The three routes that follow are three different prices paid for that: Route A refuses to pay (stays on-policy), Route B pays in stability, Route C pays with a trust region — and that is the route AWR takes.

</div>

---

## Route A: policy gradients

Differentiate $J$ directly and ascend:

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\pi_\theta}\!\left[\nabla_\theta \log \pi_\theta(a|s)\, A^{\pi_\theta}(s,a)\right]$$

**The good:** unbiased, conceptually simple, works with any differentiable policy.

**The bad:**
- **On-policy.** The expectation is under $\pi_\theta$ — data from an older policy is invalid without importance weights, and importance weights explode in variance.
- **Unstable.** Requires trust regions (TRPO), clipping (PPO), careful tuning.
- **Sample-hungry.** Every gradient step needs fresh environment interaction — fatal for real robots.

<div class="box warn">

Requires a tractable $\log \pi_\theta(a|s)$. Remember this — it is exactly what breaks in Part 3.

</div>

---

## Route B: Q-learning / off-policy actor-critic

DDPG, TD3, SAC: learn $Q(s,a)$ by bootstrapping, push the actor up $\nabla_a Q$.

**The good:** genuinely off-policy — reuse a replay buffer, far better sample efficiency.

**The bad:**
- Bootstrapped TD targets are a *contraction that can diverge* under function approximation — needs double-Q, target networks, delayed updates, entropy tuning.
- **Out-of-distribution actions.** $Q$ gets queried at actions never seen in the data, over-estimates them, and the actor chases the error. This is why BCQ / BEAR exist.
- With a purely static dataset, plain behavior cloning often beats TD3/SAC.

<div class="box">
<b>The gap AWR aims at:</b> something as stable as supervised learning, but that can still reuse off-policy data.
</div>

---

## Route C: policy search as supervised learning

Reframe policy improvement as **inference**: treat "the trajectory was optimal" as an observed event, and fit the policy by maximum likelihood weighted by how optimal each sample was.

**Reward-Weighted Regression** (Peters & Schaal, 2007; Peters et al. 2010) — an EM algorithm:

$$\pi_{k+1} = \arg\max_\pi\ \mathbb{E}_{s\sim d_{\pi_k}(s)}\ \mathbb{E}_{a \sim \pi_k(a|s)}\!\left[\log \pi(a|s)\, \exp\!\left(\tfrac{1}{\beta} R_{s,a}\right)\right]$$

Read it literally: **fit a new policy to the actions you just took, where each action's likelihood is weighted by $e^{R/\beta}$.**

- The loss is a plain weighted maximum likelihood — convergent, stable, no bootstrapping.
- $\beta > 0$ is a temperature: $\beta \to \infty$ recovers behavior cloning, $\beta \to 0$ keeps only the single best action.

---

## What is wrong with RWR

Three problems, all of which AWR will fix:

<div class="cols">
<div>

**1. It weights by return, not advantage.**
$e^{R_{s,a}/\beta}$ has no baseline. In a state where every action returns ≈ 800, *all* actions get a huge weight. The weights carry state-value information, not action-quality information.

**2. It is on-policy.**
$s \sim d_{\pi_k}$, $a \sim \pi_k$ — data is discarded after one update. Importance sampling to reuse it re-introduces the variance problem.

</div>
<div>

**3. Monte Carlo returns are high-variance.**
$R_{s,a} = \sum_t \gamma^t r_t$ over a full rollout, then exponentiated — noise gets amplified multiplicatively.

<div class="box warn">
<b>Empirically:</b> RWR with neural networks simply does not work. On Ant-v2 it scores <b>181</b> where AWR scores <b>5067</b>. It was known to under-perform (Schulman 2015, Duan 2016) and was largely written off.
</div>

</div>
</div>

---

<!-- _class: lead -->

# Part 2 — Advantage-Weighted Regression

### Peng, Kumar, Zhang & Levine, 2019

---

## AWR in one slide

> "Two standard supervised learning steps: one to regress onto target values for a value function, and another to regress onto weighted target actions for the policy."

<div class="box good">

**Algorithm 1 — Advantage-Weighted Regression**
1. $\pi_1 \leftarrow$ random policy, $\mathcal{D} \leftarrow \emptyset$
2. **for** iteration $k = 1 \dots k_{\max}$:
3. &nbsp;&nbsp;&nbsp;&nbsp; add trajectories $\{\tau_i\}$ sampled with $\pi_k$ to the FIFO replay buffer $\mathcal{D}$
4. &nbsp;&nbsp;&nbsp;&nbsp; $V_k^{\mathcal{D}} \leftarrow \arg\min_V\ \mathbb{E}_{s,a\sim\mathcal{D}}\big[\|R^{\mathcal{D}}_{s,a} - V(s)\|^2\big]$
5. &nbsp;&nbsp;&nbsp;&nbsp; $\pi_{k+1} \leftarrow \arg\max_\pi\ \mathbb{E}_{s,a\sim\mathcal{D}}\big[\log\pi(a|s)\exp\big(\tfrac{1}{\beta}(R^{\mathcal{D}}_{s,a} - V^{\mathcal{D}}_k(s))\big)\big]$

</div>

That is the entire algorithm. No target networks, no bootstrapped critic, no trust region, no importance weights. Both lines are `model.fit(X, y, sample_weight=w)`.

---

## Derivation 1/4 — maximize *improvement*, not return

Goal: find $\pi$ maximizing the expected improvement over a **sampling policy** $\mu$:

$$\eta(\pi) = J(\pi) - J(\mu) = \mathbb{E}_{s\sim d_\pi}\mathbb{E}_{a\sim\pi}\!\left[A^\mu(s,a)\right] = \mathbb{E}_{s\sim d_\pi}\mathbb{E}_{a\sim\pi}\!\left[R^\mu_{s,a} - V^\mu(s)\right]$$

<div class="box">

<b>This is the pivotal choice.</b> RWR and REPS maximize $J(\pi)$; AWR maximizes $J(\pi) - J(\mu)$. Since the two differ by a constant they have the same argmax — but they lead to <i>different surrogate objectives</i>, and the improvement form is the one whose integrand is the <b>advantage</b>. That baseline is what survives into the final weights.

</div>

$d_\pi$ still depends on $\pi$, so approximate it with $\mu$'s state distribution (Kakade & Langford 2002; Schulman 2015):

$$\hat\eta(\pi) = \mathbb{E}_{s\sim d_\mu(s)}\,\mathbb{E}_{a\sim\pi(a|s)}\!\left[R^\mu_{s,a} - V^\mu(s)\right]$$

$\hat\eta$ matches $\eta$ to first order, and stays a good estimate while $\pi$ is close to $\mu$ in KL.

---

## Derivation 2/4 — constrained policy search

That "close in KL" caveat becomes an explicit constraint:

$$\arg\max_\pi \int_s d_\mu(s)\int_a \pi(a|s)\left[R^\mu_{s,a} - V^\mu(s)\right] da\, ds$$
$$\text{s.t.}\quad \int_s d_\mu(s)\, D_{\mathrm{KL}}\!\left(\pi(\cdot|s)\,\|\,\mu(\cdot|s)\right) ds \le \epsilon$$

Form the Lagrangian with multiplier $\beta$:

$$\mathcal{L}(\pi,\beta) = \int_s d_\mu \int_a \pi\left[R^\mu_{s,a} - V^\mu(s)\right] + \beta\left(\epsilon - \int_s d_\mu D_{\mathrm{KL}}(\pi\|\mu)\right)$$

Differentiate w.r.t. $\pi(a|s)$, set to zero, solve:

$$\boxed{\ \pi^*(a|s) = \frac{1}{Z(s)}\,\mu(a|s)\,\exp\!\left(\frac{1}{\beta}\left[R^\mu_{s,a} - V^\mu(s)\right]\right)\ }$$

---

## Derivation 3/4 — project onto the policy class

$\pi^*$ is a non-parametric density we can evaluate but not sample from. Project it onto the parametric family by minimizing reverse KL:

$$\arg\min_\pi\ \mathbb{E}_{s\sim\mathcal{D}}\left[D_{\mathrm{KL}}\!\left(\pi^*(\cdot|s)\,\|\,\pi(\cdot|s)\right)\right]$$

$$= \arg\max_\pi\ \mathbb{E}_{s\sim d_\mu(s)}\,\mathbb{E}_{a\sim\mu(a|s)}\!\left[\log\pi(a|s)\,\exp\!\left(\frac{1}{\beta}\left[R^\mu_{s,a}-V^\mu(s)\right]\right)\right]$$

<div class="cols">
<div>

**Compare to RWR:**

$$\text{RWR: } \exp\!\left(\tfrac{1}{\beta}R_{s,a}\right)$$
$$\text{AWR: } \exp\!\left(\tfrac{1}{\beta}\big[R_{s,a}-V(s)\big]\right)$$

</div>
<div>

One subtracted baseline. That is the whole difference in the on-policy case — and it is worth an order of magnitude in final return.

</div>
</div>

---

## Derivation 4/4 — experience replay

With a replay buffer, $\mu$ is not one policy but a **mixture** $\mu_k(\tau) = \sum_i w_i \pi_i(\tau)$ over past iterations. Redo the derivation with $\mu(s,a) = \sum_i w_i d_{\pi_i}(s)\pi_i(a|s)$ and the same Lagrangian machinery yields:

$$\arg\max_\pi \sum_{i=1}^{k} w_i\, \mathbb{E}_{s\sim d_{\pi_i}}\mathbb{E}_{a\sim\pi_i}\!\left[\log\pi(a|s)\exp\!\left(\frac{1}{\beta}\left[R^{\pi_i}_{s,a} - \frac{\sum_j w_j d_{\pi_j}(s)V^{\pi_j}(s)}{\sum_j w_j d_{\pi_j}(s)}\right]\right)\right]$$

The baseline is now a **$d$-weighted average of the past policies' value functions**. Fitting each $V^{\pi_i}$ separately would be hopeless (too little data each) — but note:

$$\bar V = \arg\min_V \sum_i w_i\, \mathbb{E}_{s,a\sim\pi_i}\!\left[\|R^{\pi_i}_{s,a} - V(s)\|^2\right] \quad\Longrightarrow\quad \bar V(s) = \frac{\sum_i w_i d_{\pi_i}(s) V^{\pi_i}(s)}{\sum_j w_j d_{\pi_j}(s)}$$

<div class="box good">
Regressing <b>one</b> value function on the <b>whole buffer</b> lands exactly on the baseline the theory asks for. Line 5 of Algorithm 1 is not an approximation — it is the right thing.
</div>

---

## Implementation details that matter

| Detail | Choice | Why |
|---|---|---|
| State sampling | uniform from $\mathcal{D}$, not $d_\mu$ | simpler, works fine in practice |
| Return estimate | **TD($\lambda$)**, bootstrapping off $V^{\mathcal{D}}_{k-1}$ | Monte Carlo returns are too high-variance |
| Temperature $\beta$ | **fixed constant** | REPS/MPO adapt the multiplier; a constant is enough |
| Weight explosion | clip $\hat\omega = \min(\omega, \omega_{\max})$ | $\exp(\cdot)$ occasionally blows up gradients |
| Buffer | 50k FIFO, 2k samples/iter | 200 value steps + 1000 policy steps per iteration |

<div class="box">

<b>The replay buffer is a trust region.</b> $\mu$ is <i>defined</i> by the buffer, so a bigger buffer changes $\mu$ more slowly, which — through the KL penalty — keeps $\pi$ from moving fast. Buffer size trades stability against learning speed. This is a genuinely nice consequence of the derivation.

</div>

---

## Results — OpenAI Gym

<div class="small">

| Task | TRPO | PPO | DDPG | TD3 | SAC | **RWR** | **AWR** |
|---|---|---|---|---|---|---|---|
| Ant-v2 | 2901 | 1161 | 72 | 4285 | **5909** | 181 | 5067 |
| HalfCheetah-v2 | 3302 | 4920 | **10563** | 4309 | 9297 | 1400 | 9136 |
| Hopper-v2 | 1880 | 1391 | 855 | 935 | 2769 | 605 | **3405** |
| Humanoid-v2 | 552 | 695 | 4382 | 81 | **8048** | 509 | 4996 |
| LunarLander-v2 | 104 | 121 | – | – | – | 185 | **229** |
| Walker2d-v2 | 2765 | 2617 | 401 | 4212 | 5805 | 406 | **5813** |

</div>

- Beats TRPO/PPO on both sample efficiency and asymptote.
- Comparable asymptote to SAC/TD3, though **less sample-efficient** than them. Humanoid is the weak spot.
- Handles discrete actions natively (LunarLander) — DDPG/TD3/SAC cannot without modification.
- **RWR loses everywhere.** The modifications are the algorithm.

---

## Ablations and the offline result

<div class="cols">
<div>

### What each piece is worth
Removing components from AWR:
- **No experience replay** (on-policy only) → large degradation, instability
- **No baseline** (= RWR weights) → clear degradation
- **No TD($\lambda$)** (Monte Carlo) → mildly worse, still viable

Ranked: replay ≈ baseline > TD($\lambda$).

Also: 34-DoF humanoid and 82-DoF dog motion imitation — AWR ≥ a *highly tuned* PPO (e.g. cartwheel 0.78 vs 0.76, canter 0.86 vs 0.76).

</div>
<div>

### Fully offline, no interaction
Static 1M-step datasets from SAC demo policies. AWR is applied **unchanged** — just treat the dataset as $\mathcal{D}$.

Result: matches or beats the demo policy, and is competitive with **BCQ** and **BEAR**, which are purpose-built for this setting.

<div class="box good">
In the offline limit AWR <i>is</i> <b>advantage-weighted behavior cloning</b>. It never evaluates out-of-distribution actions — the policy is only ever trained on actions that appear in the data. That is why it dodges the failure mode that sinks TD3/SAC offline.
</div>

</div>
</div>

---

## AWR — summary

<div class="box good">

**Take-aways**
1. Constrained policy search + a KL trust region gives $\pi^* \propto \mu \exp(A/\beta)$ in closed form.
2. Projecting that onto a parametric policy is a *weighted maximum-likelihood* problem — i.e. supervised learning.
3. Modelling $\mu$ as a replay buffer makes it off-policy for free, and a single value regression gives exactly the right baseline.
4. Advantage (not return) in the exponent is what makes it work with neural networks.

</div>

**Limitations the authors name:** less sample-efficient than SAC/TD3; convergence properties under replay not well understood.

**A limitation they do not name — and the one Part 3 is about:** the weight $e^{A/\beta}$ multiplies a *log-likelihood*. If your policy has no tractable log-likelihood, this objective does not exist.

---

<!-- _class: lead -->

# Part 3 — The VLA problem

### Why you cannot just run AWR on a modern robot policy

---

## Modern VLAs are flow-matching models

A VLA maps (images, language, proprioception) → action chunks. The state of the art (π₀, π₀.₅, π\*₀.₆) generates the continuous action chunk with **flow matching**: a learned velocity field $f_\theta$ integrated from noise to a clean action.

$$a^{\eta,\omega}_{t:t+H} = \eta\, a_{t:t+H} + (1-\eta)\,\omega,\qquad \omega\sim\mathcal{N}(0,I)$$
$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{\eta,\omega}\left\|\omega - a_{t:t+H} - f_\theta(a^{\eta,\omega}_{t:t+H}, o_t, \ell)\right\|^2$$

<div class="box warn">

**The consequence:** flow matching gives you a *sampler*, not a *density*. There is no tractable $\log \pi_\theta(a|o)$.

- **PPO / GRPO** need $\log \pi_\theta(a|o)$ for the ratio $\pi_\theta/\pi_{\text{old}}$. ✗
- **AWR** needs $\log \pi_\theta(a|o)$ for the weighted MLE. ✗ (though the FM loss is a usable surrogate)

</div>

Both papers in Parts 4 and 5 are answers to this one problem.

---

## Two more problems, specific to real robots

**1. The manifold problem.** Valid robot actions occupy a thin manifold in a 12–32-dimensional chunk space. Any objective that *pushes probability away* from bad actions mostly pushes predictions **off the manifold** — producing not "different" actions but invalid ones.

<div class="box">
Conditioning/reweighting methods never leave the manifold: they only <b>redistribute mass toward good actions the policy already produces</b>. The price is weak exploration. For "reliably do the task well", that trade is favourable. <span class="tiny">(argued explicitly in Learning to Fold §2.2)</span>
</div>

**2. AWR discards data.** With $\omega = e^{A/\beta}$, low-advantage samples get weight ≈ 0. π\*₀.₆'s critique:

> "these methods discard or significantly downweight a significant portion of the data, effectively implementing a kind of filtered imitation technique."

On a real robot, where every episode costs minutes of wall-clock and human supervision, throwing away the failures is expensive — and the failures are exactly where the information about what *not* to do lives.

---

<!-- _class: lead -->

# Part 4 — π\*₀.₆ and RECAP

### Physical Intelligence, 2025

---

## RECAP — the idea

**R**L with **E**xperience and **C**orrections via **A**dvantage-conditioned **P**olicies.

<div class="box good">
Instead of <b>weighting</b> the loss by the advantage, feed the advantage in as an <b>input</b>. Train on all the data — good and bad — and at inference time simply ask for the good behavior.
</div>

Three subroutines, repeated:

1. **Data collection.** Run the VLA on the task; label each episode with a success outcome; optionally let a human teleoperator intervene to correct mistakes mid-rollout.
2. **Value function training.** Fit a large multi-task value function $V^{\pi_{\text{ref}}}$ on all data collected so far — it learns to detect failures and judge time-to-completion.
3. **Advantage-conditioned training.** Compute advantages from $V$, binarize them into an "improvement indicator", and put that indicator **in the VLA's prompt**.

Pre-training = steps 2–3 over tens of thousands of hours of demos. Post-training = steps 1–3 per task.

---

## The reward and the value function

A general sparse reward that applies to essentially any task — derived from a single episode-level success label:

$$r_t = \begin{cases} 0 & t = T \text{ and success} \\ -C_{\text{fail}} & t = T \text{ and failure} \\ -1 & \text{otherwise}\end{cases}$$

<div class="box">

So $V$ learns <b>the negative number of steps remaining until success</b> — a large negative constant for episodes that fail. Normalized per task to $(-1, 0)$ so tasks of different lengths are comparable. One reward definition, every task.

</div>

**Distributional** value function: discretize the empirical return $R_t(\tau)$ into $B=201$ bins, minimize cross-entropy:

$$\min_\phi\ \mathbb{E}_{\tau\in\mathcal{D}}\left[\sum_{o_t\in\tau} H\!\left(R^B_t(\tau),\, p_\phi(V|o_t,\ell)\right)\right], \qquad V^{\pi_{\text{ref}}}(o_t,\ell) = \sum_b p_\phi(V{=}b|o_t)\,v(b)$$

This is a **Monte Carlo, on-policy estimator** for the behavior policy of $\mathcal{D}$ — deliberately simpler and more reliable than an off-policy Q-function, at the cost of some optimality.

---

## Derivation — from weighting to conditioning

Start from the same regularized RL objective as AWR, with $D = D_{\mathrm{KL}}$:

$$\hat\pi(a|o) \propto \pi_{\text{ref}}(a|o)\exp\!\big(A^{\pi_{\text{ref}}}(o,a)/\beta\big) \qquad \text{— the AWR solution}$$

Now the less well-known variant. Define the **probability that action $a$ improves over $\pi_{\text{ref}}$**:

$$p(I|A^{\pi_{\text{ref}}}(o,a)) = \frac{g(A^{\pi_{\text{ref}}}(o,a))}{\int g(A^{\pi_{\text{ref}}}(o,a'))\,da'},\qquad g \text{ monotonically increasing}$$

Then $\hat\pi(a|o) \propto \pi_{\text{ref}}(a|o)\, p(I|A^{\pi_{\text{ref}}}(o,a))^\beta$ is **guaranteed to improve**: $J(\hat\pi) \ge J(\pi_{\text{ref}})$.

Apply Bayes' rule — $p(I|A^{\pi_{\text{ref}}}(o,a)) = \pi_{\text{ref}}(a|I,o)/\pi_{\text{ref}}(a|o)$ — and substitute:

$$\boxed{\ \hat\pi(a|o,\ell) \propto \pi_{\text{ref}}(a|o,\ell)\left(\frac{\pi_{\text{ref}}(a|I,o,\ell)}{\pi_{\text{ref}}(a|o,\ell)}\right)^{\beta}\ }$$

---

## Why that box is the whole trick

$$\hat\pi(a|o,\ell) \propto \pi_{\text{ref}}(a|o,\ell)\left(\frac{\pi_{\text{ref}}(a|I,o,\ell)}{\pi_{\text{ref}}(a|o,\ell)}\right)^{\beta} \qquad\xrightarrow{\ \beta=1\ }\qquad \hat\pi(a|o,\ell) = \pi_{\text{ref}}(a|I,o,\ell)$$

<div class="box good">

At $\beta = 1$ the improved policy <b>is just the behavior policy conditioned on "this action was an improvement."</b> No explicit $p(I|A)$, no importance weights, no log-likelihood ratio. Train one model that can represent both $\pi_{\text{ref}}(a|o)$ and $\pi_{\text{ref}}(a|I,o)$ — exactly the classifier-free guidance construction — and $\beta>1$ comes free as CFG at inference.

</div>

The training objective is then plain conditional maximum likelihood over **all** the data:

$$\min_\theta\ \mathbb{E}_{\mathcal{D}^{\pi_{\text{ref}}}}\!\left[-\log\pi_\theta(a_t|o_t,\ell) - \alpha\log\pi_\theta(a_t|I_t,o_t,\ell)\right],\qquad I_t = \mathbf{1}\!\left[A^{\pi_{\text{ref}}}(o_t,a_t,\ell) > \epsilon_\ell\right]$$

For flow matching, $\log\pi_\theta$ is replaced by its lower bound: discrete-token log-likelihood + the flow-matching loss. **The objective survives the transition to flow matching; AWR's does not.**

---

## How the indicator is actually implemented

<div class="cols">
<div>

### It is text
The advantage indicator is literally injected as a text token in the prompt:

<div class="box">

<code>"Advantage: positive"</code> when $I_t = \text{True}$<br>
<code>"Advantage: negative"</code> otherwise

</div>

Placed **after** the predicted subtask $\hat\ell$ but **before** the actions, so only the action log-likelihoods are affected.

**Threshold.** $\epsilon_\ell$ = the 30th percentile of the value function's predictions for task $\ell$ — per-task, so it adapts to task difficulty. Tuning $\epsilon_\ell$ replaces tuning $\beta$.

</div>
<div>

### Three data sources, one mechanism
- **Demonstrations** → indicator from $V$; during task SFT it is pinned to True.
- **Autonomous rollouts** → indicator from $V$; the *failures are kept* and labeled negative.
- **Human interventions** → $I_t$ **forced True**, on the assumption that an expert correction is always a good action.

<div class="box good">
That last line is the elegant part: DAgger-style corrections and RL rollouts enter through <b>the same channel</b>. No separate imitation loss, no separate loss weight.
</div>

$I_t$ is randomly dropped during training, so the model learns the unconditional branch too — enabling CFG at $\beta>1$.

</div>
</div>

---

## The model and the loop

<div class="cols">
<div>

### π\*₀.₆
- Base: **π₀.₆**, an evolution of π₀.₅ — Gemma 3 4B VLM backbone, **860M** flow-matching action expert, Knowledge Insulation training (stop-gradient so the action expert does not corrupt the VLM).
- Outputs joint angles + gripper at **50 Hz** in chunks; also emits a text subtask ("pick up the coffee cup") *before* the actions, so action generation is conditioned on its own high-level plan.
- Value function: **same architecture, smaller 670M** Gemma 3 backbone, co-trained on web data to avoid overfitting. Runs on-the-fly during VLA training — negligible extra cost.

</div>
<div>

### Algorithm 1 (RECAP)
<div class="box small">

1. Train $V_{\text{pre}}$ on $\mathcal{D}_{\text{demo}}$
2. Train $\pi_{\text{pre}}$ on $\mathcal{D}_{\text{demo}}$ using $V_{\text{pre}}$
3. Initialize $\mathcal{D}_\ell$ with demos for task $\ell$
4. Train $V_\ell^0$ from $V_{\text{pre}}$; train $\pi_\ell^0$ from $\pi_{\text{pre}}$
5. **for** $k = 1 \dots K$:
6. &nbsp;&nbsp; collect data with $\pi_\ell^{k-1}$, add to $\mathcal{D}_\ell$
7. &nbsp;&nbsp; train $V_\ell^k$ **from $V_{\text{pre}}$** on $\mathcal{D}_\ell$
8. &nbsp;&nbsp; train $\pi_\ell^k$ **from $\pi_{\text{pre}}$** on $\mathcal{D}_\ell$

</div>

**Note lines 7–8:** each iteration restarts from the *pre-trained* checkpoint, not the previous iteration's. Deliberate — it prevents drift across iterations.

</div>
</div>

---

## Results

Tasks: laundry folding (t-shirts/shorts; 11 diverse item types; a strict-criteria variant), **double-shot espresso** on a commercial machine, **cardboard box assembly** in a factory. Each 5–15 minutes long, on a bimanual 6-DoF setup.

Metrics: **throughput** (successful completions per hour — captures speed *and* success) and **success rate**.

<div class="box good">

- Throughput **more than doubles** on diverse laundry and espresso, going from offline-RL+SFT to the final model.
- Failure rate **roughly halves** across the board.
- Success **90%+** on all tasks except diverse laundry.
- Ran 13 hours straight making espresso; folded novel laundry in a new home for 2+ hours uninterrupted.

</div>

**Over iterations:** t-shirt folding (RL only, no interventions, 300 trajectories × 4 robots per iteration) → +50% throughput over two iterations, success >90% after the first. Box assembly (600 autonomous + 360 intervention trials per iteration) → 2× throughput by iteration 2.

**Targeted failure removal:** adversarial initial states, strict "collar facing up" criterion, 2 iterations × 600 trajectories, **no interventions or extra demos** → 97% success.

---

## The comparison that matters for us

Same data, same base model, three different **policy extraction** methods, on t-shirts and shorts:

| Method | Result |
|---|---|
| **PPO** (DPPO/FPO variant, SPO-style constraint) | Needed a tiny trust region ($\eta = 0.01$) to be stable at all; stable but does not reach good performance. |
| **AWR** (advantages from the same value function) | Reasonable success rate, but **much slower policies → far lower throughput**. |
| **Advantage conditioning (RECAP)** | **By far the highest throughput.** |

<div class="box warn">
Note the baselines were given a slight <b>advantage</b>: they trained on the data collected <i>by</i> RECAP, which is better data than they would have generated themselves.
</div>

**Why AWR under-performs here:** the exponential weighting concentrates on a filtered slice of the data. Speed improvements need the model to see the whole distribution and learn *what makes an episode fast*, which is precisely what the discarded low-weight data encodes.

---

## RECAP — summary and limitations

<div class="cols">
<div>

### What it buys
- Works with expressive **flow-matching** VLAs — no log-likelihood ratio anywhere.
- Trains on **all** data: demos, successes, failures, corrections — one objective.
- The value function is a plain MC regression on a sparse, universal reward.
- CFG at inference gives a knob to trade regularization against optimality after training.

</div>
<div>

### What the authors flag
- **Not autonomous** — humans still label rewards, intervene, and reset scenes.
- **Exploration is greedy** — relies on policy stochasticity and human interventions to find anything new.
- **Iterated offline**, not truly online — collect a batch, retrain, repeat.
- The value estimator is on-policy MC, not a proper off-policy Q.

</div>
</div>

<div class="box">
<b>The bet:</b> give up exploration, keep everything else. That is defensible when the imitation-pretrained policy already does something reasonable — which is exactly the VLA setting.
</div>

---

<!-- _class: lead -->

# Part 5 — Learning to Fold

### Ilia Larchenko · LeHome Challenge @ ICRA 2026 · 1st online / 2nd real

---

## The competition

**Task:** bimanual garment folding on a SO-ARM101 setup — two 6-DoF arms, 12-dim joint actions at 30 Hz (sim) / 20 Hz (real), three RGB cameras. Four garment types: long/short tops, long/short pants.

**Success is binary and geometric.** Each garment has keypoints; success = a conjunction of pairwise distance conditions (pairs that should meet must be closer than a threshold; pairs that should stay apart must not). **No partial credit.**

<div class="cols">
<div>

**The four hard parts**
1. Cloth is deformable — small trajectory differences → very different states. The provided BC data is scripted, clean, and inflexible.
2. Reward is **sparse and binary** — an easy success looks identical to a hard one.
3. Half the eval set is **unseen garments**, some never exposed at all.
4. **No access to the evaluation robot** — really sim → my robot → their robot.

</div>
<div>

**Two rounds**
- **Online (sim), Feb–Apr 2026.** Isaac Sim, public leaderboard, 62 teams. ~3 months of work.
- **Real-world final, June 2026, ICRA Vienna.** Top 8 teams on physical hardware, jury-scored with partial credit. **~1 week** of work.

<div class="box warn">
The author is explicit: this is an <b>engineering case study, not a controlled experiment</b>. Almost nothing is ablated. Read it for the recipe, not for causal claims.
</div>

</div>
</div>

---

## The central claim: use both AWR and RECAP

Given a BC-pretrained policy with non-zero success rate, the advantage signal is consumed **two ways at once**:

<div class="cols">
<div>

**AWR — through the sampler**
High-advantage frames are *trained on more often*.
→ the model ends up better than its average rollout.

</div>
<div>

**RECAP — through conditioning**
Advantage is an *input token*.
→ "predict good actions only", and CFG at inference.

</div>
</div>

<div class="box good">
<b>Why they compose.</b> Think of the behavior data as a mixture of a large "bad actions" mode and a smaller "good actions" mode. <b>AWR reweights the mixture toward the good mode. RECAP conditioning selects the positive-advantage slice</b> (mostly good, with some spillover). Doing both leaves the policy target almost entirely on the good mode.
</div>

They share the same primitives — one advantage estimate feeds both — so the marginal cost of adding the second is near zero. The author's argument against PPO/GRPO is the manifold argument from Part 3.

---

## AWR through the sampler, not the loss

<div class="box">

$$P(\text{sample frame } i) \ \propto\ e^{\,\mathrm{clip}(A_i,\,-2,\,2)}$$

The flow-matching loss over the batch is then **plain unweighted MSE**.

</div>

Equivalent in expectation to loss weighting, but strictly more **data-efficient**: a frame with weight $e^{-2}$ is not down-weighted after being loaded — it is simply almost never loaded. Its images are never decoded and never occupy a batch slot. Batch utilization stays at 100% of the weight mass.

<div class="box warn">

**The correction this requires.** The auxiliary heads (next slide) predict *statistics of the data distribution*, not of the advantage-tilted distribution. So every sampled frame carries an inverse-sampling importance weight

$$w_i = \frac{1}{N\, p_i\, T_{\text{ep}(i)}}$$

and **all auxiliary losses are weighted by $w_i$ while the action loss ignores it.** Same machinery for BC/DAgger sources, where per-frame priority is $P \propto e^{3(1 - SR_{\text{garment}})}$ so struggling garments get over-sampled.

</div>

---

## The policy is its own value function

A single learned **query token**, placed in the *image* attention group, feeds a set of cheap linear heads. Critically, that token sees **images only** — not state, not garment type, not advantage — so heads cannot overfit to proprioception or trivially copy the garment-type input.

<div class="cols">
<div>

**Current-frame heads**

| Head | Target |
|---|---|
| **success** | P(episode succeeds) — *the value function* |
| **completion** | $t/T$, successful episodes only |
| garment type | 4-class, drives inference bootstrap |
| keypoint distances | 21 outputs, $d^{(i)} = \text{dist}_i/\text{thresh}_i$ |
| checkpoint, TTC | legacy, not used downstream |

</div>
<div>

**Future heads (+30 frames)** — a cheap world-model substitute

- **FAST query** (training-only): pushes future-awareness into the VLM representation.
- **FM query**: sits at the tail of the action-expert suffix, *bidirectional with the action tokens* — so it reads what the policy is about to do. Predicts future keypoint distances and a **success residual**
$$\Delta_{\text{success}} = y - \mathrm{sg}\big[\hat P_{\text{success}}\big]$$
i.e. **a Q-function**: "given these specific actions, how much better or worse than the image-only baseline will this end?"

</div>
</div>

<div class="box good">
One model to train, serve and version — no separate critic. And predicting success shares most of its primitives with choosing an action chunk, so the heads act as auxiliary signal on a shared representation.
</div>

---

## Reward design — densifying a binary signal

**1. Checkpoints from the success checker itself.** Reuse the challenge's own keypoint conditions to define an intermediate fold checkpoint (worth 0.5) and full success (bringing cumulative reward to 1.0). No new keypoints, no new success definition.

**2. Gradual first checkpoint** (tops only) — allocate the first 0.5 in proportion to how much of the primary distance gap has closed:
$$R^{\text{cp1}}_t = 0.5\,\mathrm{clip}\!\left(\frac{d_0 - m_t}{d_0 - d_{t_1}},\,0,\,1\right),\qquad m_t = \min_{\tau\le t} d_\tau$$

**3. Failure withdrawal.** If the episode fails, **all** accumulated reward is withdrawn so the return stays exactly $\sum_t r_t = \mathbf{1}[\text{success}]$ — spread uniformly from the last frame $t_p$ where cumulative reward peaked, to avoid one sharp negative spike.

<div class="box">
<b>The point:</b> intermediate checkpoints provide <i>temporal credit assignment within</i> an episode, while the episode-level return stays perfectly aligned with the true competition objective. Reaching a checkpoint can be misleading — the conditions may hold while the rest of the garment is wrecked.
</div>

---

## The value baseline — a CUPED-style correction

Since the return is the success indicator, the value function is just $V(s_t) = P(\text{success}|s_t) - R^{\text{cum}}_t$.

**Two problems with subtracting $\hat V$ literally:**
1. **Checkpoint rewards cancel.** Reward at step $t$ enters both the remaining return (through $P$) and $R^{\text{cum}}_t$ — the two contributions offset, so the reward vanishes from the advantage.
2. The variance-minimization argument assumes a *perfect* predictor. With imperfect $\hat V$, full subtraction is not optimal.

Borrowing **CUPED** from A/B testing (a control variate with an estimated optimal coefficient), the variance-minimizing correction is not $\hat V$ but

$$\theta^* \hat V, \qquad \theta^* = \rho(\hat V, V)\frac{\sigma(V)}{\sigma(\hat V)} = \rho(\hat V, G)\frac{\sigma(G)}{\sigma(\hat V)}, \qquad G_t = \mathbf{1}[\text{success}] - R^{\text{cum}}_t$$

→ collapses to full subtraction for a perfect predictor, to zero for a random one. The second form is **directly estimable from rollouts**.

<div class="box warn">

<b>Honest footnote from the author:</b> during the competition he believed $\theta^*$ was not computable and used a fixed $\alpha_s = 0.5$ instead. Post-hoc, the true per-garment coefficient ranged 0.4–0.8 with 0.5 near the median — so the mistake was not critical. Damping also <i>partially</i> un-cancels the checkpoint rewards (problem 1), which is why checkpoint moments retain positive advantage.

</div>

---

## Advantage — GAE over two heads, degrading gracefully

**Success GAE** — the exact TD residual of the damped baseline $\hat V_t = \alpha_s\big(\bar S_t\big) - R^{\text{cum}}_t$, terminal pinned to the true outcome:
$$\delta^s_t \approx \big(1 - \alpha_s\gamma\big) r_t + \alpha_s\gamma\big(\bar S_{t+1} - \bar S_t\big), \qquad A^s_t = \delta^s_t + \gamma\lambda A^s_{t+1}$$

**Completion shaping** — potential-based, $\Phi_t = \alpha_c \bar C_t$. The completion head is **policy-stable** (task progress looks the same regardless of which policy produced it), so it stays valid on old data and keeps giving signal even when $P(\text{success}) \approx 1$.

<div class="box good">

**The stale-rollout problem.** $P(\text{success})$ is policy-dependent, and the predicting model both evolves *and* overfits to data it has already trained on. Solution: predict at **collection time** and never re-predict; decay each rollout dataset's share by 0.98/iteration; and **blend toward an outcome-only baseline** as data ages:

$$\tilde A_t = w\,A^s_t + (1-w)\,A^{\text{seg}}_t + \Phi^c_t, \qquad w = \min(\text{sampling share}, 1)$$

$A^{\text{seg}}$ is a GRPO-style relative-success signal computed per episode segment, depending only on outcomes. **The staler a rollout, the more its advantage degrades toward a sparse, objective signal.**

</div>

Plus a **precision boost**: top 20% of successes per garment (by tightest satisfaction margin) get $\Delta A = +0.3$ on every frame — biasing toward high-quality rather than marginal folds.

---

## The asynchronous flywheel

Three independent components that communicate **only through HuggingFace Hub** — no synchronization barriers anywhere.

<div class="cols">
<div>

**Training worker** (1× H200)
recompute advantages over all rollout datasets → train ~1000 steps → upload checkpoint every ~500 steps.

**Rollout workers** (any number)
pull latest checkpoint, run 3–5 parallel Isaac Sim instances, upload episodes *with the values predicted at collection time*.

**Manual DAgger station**
a human fixes saved failure states via teleop; episodes ship through the same channel.

</div>
<div>

<div class="box">
Scaling data collection = starting another machine. A background HF-sync daemon means neither training nor collection ever blocks on the network.
</div>

**Datasets are never merged** — the loader holds every source (BC, every DAgger session, every RL batch) and samples by per-source shares at runtime. The data mix is a *config parameter*, not a preprocessing step.

**Checkpoint rollbacks** — periodically roll back a few days and retrain on everything collected since, including by newer checkpoints. Reliably kicks the policy out of local optima. *(π\*₀.₆ does this systematically — Algorithm 1, lines 7–8.)*

</div>
</div>

---

## Inference-time optimization

The same checkpoint behaves very differently depending on how it is run. **None of these touch the weights.**

<div class="cols">
<div>

- **Execution length $n_e$** — how many of the $H{=}30$ actions actually get sent before re-planning.
- **Playback stretch $k$** — time-rescale the executed slice; $k>1$ moves slower.
- **Soft inpainting onset $t_{ip}$** — the tail of the previous chunk anchors the next one, but only in the *high-noise* part of the denoising loop, leaving the final steps free to self-correct.
- **Noise temperature $\tau$** — the initial noise is drawn from a fitted action covariance (not i.i.d.), scaled by $\sqrt\tau$.

</div>
<div>

- **CFG scale $\alpha$** on the advantage conditioning:
$$\hat v = v_{\text{uncond}} + \alpha\big(v_{\text{cond}} - v_{\text{uncond}}\big)$$
Both passes share the prefix KV cache, so guidance only doubles the *cheap* action-expert cost.
- **Best-of-N** — sample $N$ chunks from the same prefix, score each by the FM head's predicted $\Delta_{\text{success}}$, execute the best:
$$\text{score} = \tfrac12\big(\Delta^{\text{cond}}_{\text{success}} + \Delta^{\text{uncond}}_{\text{success}}\big)$$
If *all* candidates predict $\Delta < 0$, draw a second larger batch.

</div>
</div>

<div class="box warn">
<b>A genuinely surprising admission:</b> the correlation between the FM head's prediction and the actual outcome was <b>effectively zero</b> in all experiments — yet 2–3 candidates consistently beat 1. The author's read: for most chunks best-of-N does nothing, but at rare genuinely-multimodal bottleneck states, avoiding the worst candidate matters.
</div>

---

## Tuning those knobs with a Thompson-sampling bandit

Grid-searching seven parameters × four garment types is prohibitive. Instead, tune them **online during rollout collection**.

- Each candidate value of each parameter is an arm with a $\text{Beta}(\alpha,\beta)$ posterior. Parameters are optimized **independently** (factorized bandit, not a joint product-space search) — keeps arm counts small and posteriors well-fed.
- Reward = binary outcome with the per-type baseline subtracted, $r = \text{success} - SR(\text{type})$; update $\alpha \mathrel{+}= \max(r,0)$, $\beta \mathrel{+}= \max(-r,0)$.
- Only unbiased (full/random) rollouts update the bandit. Posteriors **decay toward uniform** each iteration so the bandit tracks the moving policy.
- Freeze the posterior-mean config per garment type for the final submission.

<div class="box good">
Three wins at once: it is a <b>cheap</b> search; the hyperparameters <b>evolve with the policy</b>, so you stay near-optimal at every stage rather than only at the end; and the exploration comes back as <b>useful training variance</b> — e.g. varying execution speed teaches the policy to complete the task faster.
</div>

<div class="small">

| Parameter | top_long | top_short | pant_long | pant_short |
|---|---|---|---|---|
| Executed actions $n_e$ / playback | 5 / 5 | 5 / 5 | 3 / 3 | 3 / 3 |
| Anchor $n_a$ · inpaint onset $t_{ip}$ | 6 · 0.4 | 3 · 0.4 | 3 · 0.5 | 3 · 0.5 |
| **Guidance $\alpha$** · noise $\tau$ · candidates $N$ | **7** · 0.9 · 2 | **7** · 0.7 · 3 | **9** · 0.7 · 3 | **7** · 0.7 · 3 |

</div>

---

## What the bandit converged to — the de-facto ablation

With no formal ablations in the paper, **where the posteriors settled is the strongest evidence available** about which design choices carry signal.

<div class="box good">

- **Guidance scale → very high.** The initial search range was 0–2; the top arm always pinned, so the range was shifted upward repeatedly until it settled around **7–9**. Strong evidence the advantage conditioning is carrying real signal — a policy conditioned on "predict good actions" behaves *substantially* differently from its unconditional self, and pushing hard along that direction keeps helping.
- **Candidates $N > 1$ helps, $N > 3$ does not** — consistent with best-of-N mostly *avoiding very bad chunks* rather than finding an optimal one.
- **Execution length → small.** The model predicts the nearest steps far more reliably than the far ones, so frequent re-planning against a fresh observation pays off.
- **Playback stretch → eventually matched execution length.** Early in training the model does not predict the optimal velocity, so exploring around the predicted speed helps; by the end it predicts the right velocity directly.
- **Inpainting → light anchoring.** A little beats none, but strong forced inpainting limits the model's ability to self-correct.

</div>

<div class="tiny">Caveat on reading the posteriors: rewards are baseline-subtracted, so all means sit near 0.5 and only their relative order matters; and posteriors decay each iteration, so they reflect the recent on-policy window rather than the whole run.</div>

---

## Online-round results

**1st of 62 teams at 79.63% overall — 6.1 points ahead of 2nd place.**

<div class="small">

| Rank | Team | Long top | Short top | Long pants | Short pants | **Overall** |
|---|---|---|---|---|---|---|
| **1** | **ilya (this work)** | 74.5% | **70.0%** | **80.5%** | **93.5%** | **79.63%** |
| 2 | Shubham @ Vorwerk | 73.0% | 62.5% | 71.5% | 87.0% | 73.50% |
| 3 | Dum-E | 76.5% | 62.0% | 75.5% | 79.5% | 73.38% |
| 4 | SCUT-Unlimited | 65.5% | 66.0% | 70.0% | 91.0% | 73.13% |

</div>

The win was **broad** — top score outright on three of four garment types, third on long tops.

**Scale:** ~12,500 rollout episodes (~4.3M frames) across ~140 sessions, single H200, batch 192, ~300k steps. Rollouts on RTX PRO 6000.

**Remaining failures:** (a) *dexterity* — the fold looks almost right but a keypoint lands just the wrong side of its threshold; (b) *simulator physics* — occasional unfair grasp failures; (c) **no recovery** — once in an OOD state the policy fails outright rather than working back.

<div class="box warn">
The author's own verdict: <b>"I would not call this approach sample-efficient."</b> Prime suspect: no recovery logic, and a scripted BC dataset with almost no exploration or diversity to bootstrap from.
</div>

---

## Sim-to-real — one week, and most of the RL comes off

**Zero-shot failed.** The clearest diagnosis: resizing 640×480 → 224×224 *directly* vs. via 320×240 — invisible to a human — dropped sim success significantly, and the auxiliary heads could **perfectly distinguish** the two. The policy had overfit to rendering artifacts.

<div class="cols">
<div>

**What was kept:** action objective, garment-type head, completion head.
**Frozen:** success, checkpoint, TTC heads.
**Removed entirely:** keypoint-distance head, both world-model heads — all need privileged sim targets.
**Removed from the pipeline:** advantage conditioning, the AdaRMS advantage channel, and therefore **CFG and best-of-N**.

<div class="box warn">
So the real-robot policy ran as <b>essentially plain BC</b> — a denoised chunk, soft inpainting, garment-type bootstrap. Nothing else.
</div>

</div>
<div>

**Two levers:** make the environments more similar, or make the training distribution diverse enough that the gap falls inside it. He used both.

- **Alignment:** a camera-overlay tool — drive the robot to a recorded joint state, overlay live cameras on the dataset frame. Shared with organizers and other teams.
- **Diversity:** very aggressive augmentation (per-camera color jitter, gain/gamma, blur, noise, independent crop/rotate/zoom, cutout, **camera dropout**, state noise/dropout) — *plus deliberately randomizing his own rig* over the week.
- **Motion-intensity alignment:** each source is time-resampled to a common speed (organizer BC ×1.0, home teleop ×1.5, sim ×0.65, DAgger ×2.0) so "how far to move in one step" stays consistent.

</div>
</div>

---

## Real DAgger, and a crude stand-in for advantage

Recovery data is the one thing the organizer BC set cannot provide — it only ever shows clean successful folds. So: policy folds autonomously, human takes over the moment it starts to fail, corrects, hands back. Every frame labeled by who was driving.

<div class="box">

With no reward or value function on the real robot, advantages are unavailable. But **the intervention signal is itself informative** — a takeover marks a recent policy failure, and the human's correction is by construction a high-advantage action. Turned into per-frame sampling weights:

- **Human-correction frames** → highest weight.
- **Autonomous frames far from any intervention** → low weight (the policy was doing fine; little new signal).
- **Autonomous frames in the 5 s window before a takeover** → **ramped to zero.** Do not reinforce the moves that led into the bad state.

</div>

**Result: 2nd place, 865 vs 895 points** (composite jury metric; unseen garments carried a 50% bonus; max 1080).

**Training mix:** organizer BC 60% (500 eps — closest to the eval distribution), own teleop+DAgger 30% (792 eps — mismatched rig, but the only source of recovery), sim success-replays 10% (1,723 eps — regularizer).

<div class="box warn">
<b>The units bug.</b> Between two LeRobot 0.4.x releases the SO-101 state representation changed from normalized −100…100 to degrees. Visually near-identical, but skews every joint ~10%. Most of the real data and training ran on the wrong convention until <b>two days before the deadline</b>.
</div>

---

## What the author would keep — and the open problem

<div class="cols">
<div>

**Keep**
- **Policy as its own value function** — no separate critic to train, sync, or serve; one forward pass yields everything.
- **Reward engineering** — the building blocks (ground-truth checkpoints, success probability, completion, relative success) are right even if his mix is not optimal.
- **Throughput engineering** — data collection is a numbers game.
- **Inference-time tuning** — a big win at zero training cost.
- **DAgger over full teleop** — keep interventions short.

</div>
<div>

**Didn't work / was hard**
- Recovery in **sim** — teleoperating cloth through a sim interface is hard, and the policy soon folded better than he could.
- Overfitting to rendering artifacts.
- The model is **bigger than the task needs**.

<div class="box warn">
<b>Biggest regret:</b> two rounds solved with two separate toolkits. Sim had the full RL machinery but no recovery; real had human interventions that <i>did</i> produce recovery but ran as plain BC. <b>Fusing them</b> — a real-side value function driving advantage conditioning, best-of-N, and DAgger weighting on the real robot — is where he'd put the next effort. His guess: 90%+ on this task.
</div>

</div>
</div>

---

## The open problem: exploration and recovery

<div class="box warn">

> "The proposed RL approach is great at reproducing and sharpening behavior it has seen, but it doesn't naturally explore its way out of states off the training distribution."

</div>

Notice that **every inference-time trick pushes toward known-good modes, not new ones:**

| Mechanism | What it does |
|---|---|
| Advantage conditioning + CFG | amplifies the *average good* behavior already in the data |
| Best-of-N | picks the least-bad of several samples **from the same prefix** |
| Soft inpainting | keeps the policy in the mode it already committed to |

**None of them invent a recovery move that isn't already in the data.** Attempts at explicit exploration mostly pushed the chunk off the action manifold and degraded performance — exactly the failure mode predicted in Part 3.

The only reliable source of recovery was **human intervention** — which worked on the real robot but not in sim. This is the same limitation π\*₀.₆ names ("exploration is largely greedy"). It is the shared blind spot of the whole conditioning/reweighting family.

---

<!-- _class: lead -->

# Synthesis

---

## The three papers side by side

<div class="small">

| | **AWR** (2019) | **RECAP / π\*₀.₆** (2025) | **Learning to Fold** (2026) |
|---|---|---|---|
| **Policy class** | Gaussian MLP | flow-matching VLA (4B + 860M) | flow-matching VLA (π₀.₅-based) |
| **How advantage enters** | loss weight $e^{A/\beta}$ | **conditioning input** $I = \mathbf{1}[A > \epsilon_\ell]$ | **both** — sampler *and* conditioning |
| **Value function** | separate net, TD($\lambda$) regression | separate 670M distributional VLM, MC | **auxiliary head of the policy itself** |
| **Reward** | environment reward | sparse success → $-$steps-to-success | dense keypoint checkpoints, binary return |
| **Data reuse** | FIFO replay = $\mu$ | all data, incl. failures + corrections | multi-source runtime mixing + decay |
| **Trust region** | KL, via buffer size | KL, via $\epsilon_\ell$ (and CFG $\beta$) | advantage clip $[-2,2]$ + conditioning |
| **Scale** | 2k samples/iter, 50k buffer | 300–960 real trials/iter | 12.5k sim episodes, 4.3M frames |
| **Validation** | benchmarks + ablations | real tasks, controlled comparisons | **competition result, no ablations** |

</div>

<div class="box good">
<b>The through-line:</b> stay on the data manifold, use a KL-style trust region to stay near the behavior policy, and turn RL into supervised learning. What changes across six years is <i>how</i> the advantage gets into the objective — and each change is forced by the policy class getting more expressive.
</div>

---

## Discussion points

**1. Weighting vs. conditioning.** RECAP's own comparison says conditioning ≫ AWR for *throughput*, because AWR filters away the data that teaches speed. Learning to Fold says use both. Are these in tension, or does moving AWR into the *sampler* (with importance-weight debiasing for the aux heads) change the calculus?

**2. Is a separate critic worth it?** π\*₀.₆ trains a 670M value model; Learning to Fold makes it a linear probe on the policy. The latter is cheaper and shares representation — but the author lists "the value head drifts because the same network predicts actions" as a known weakness, and works around it with EMA smoothing, tail correction, and stale-data blending.

**3. Sparse-and-universal vs. dense-and-engineered rewards.** π\*₀.₆ deliberately uses one sparse reward that applies to *any* task. Learning to Fold hand-engineers dense checkpoints from the success checker — and calls reward design "still an art". Which generalizes?

**4. The shared blind spot.** Both papers admit exploration is greedy and recovery comes from humans. Everything in this lineage sharpens known-good behavior. **What would it take to get a flow-matching policy to invent a recovery move that is not in its data?**

---

<!-- _class: lead -->

# Appendix

---

## Notation cheat-sheet

| Symbol | Meaning |
|---|---|
| $\mu(a\|s)$ | sampling / behavior policy — for AWR, the replay buffer's conditional action distribution |
| $\pi_{\text{ref}}$ | RECAP's name for the same thing: the mixture of humans + past policies in $\mathcal{D}$ |
| $A^\mu(s,a) = R^\mu_{s,a} - V^\mu(s)$ | advantage w.r.t. the sampling policy |
| $\beta$ | Lagrange multiplier on the KL constraint = temperature. Big $\beta$ → closer to BC |
| $\omega_{s,a} = \exp(A/\beta)$ | AWR's regression weight (clipped at $\omega_{\max}$) |
| $I_t = \mathbf{1}[A > \epsilon_\ell]$ | RECAP's binarized improvement indicator, injected as text |
| $\epsilon_\ell$ | per-task improvement threshold — 30th percentile of $V$'s predictions |
| $\alpha$ (CFG) | guidance scale at inference; corresponds to $\beta > 1$ |
| $\Delta_{\text{success}}$ | Learning to Fold's action-conditional success residual — a Q-function |
| $H, n_e, n_a, k$ | chunk horizon, executed actions, anchor actions, playback stretch |

---

## Links

**Advantage-Weighted Regression** (Peng, Kumar, Zhang, Levine — 2019)
arXiv [1910.00177](https://arxiv.org/abs/1910.00177) · project [xbpeng.github.io/projects/AWR](https://xbpeng.github.io/projects/AWR/index.html) · code [github.com/xbpeng/awr](https://github.com/xbpeng/awr)

**π\*₀.₆: a VLA That Learns From Experience** (Physical Intelligence — 2025)
arXiv [2511.14759](https://arxiv.org/abs/2511.14759) · blog [pi.website/blog/pistar06](https://www.pi.website/blog/pistar06)

**Learning to Fold** (Ilia Larchenko — ICRA 2026)
arXiv [2606.27163](https://arxiv.org/abs/2606.27163) · checkpoints: `IliaLarchenko/lehome_sim`, `IliaLarchenko/lehome_real`

<br>

**Useful background**
- Peters & Schaal 2007 — Reward-Weighted Regression · Peters et al. 2010 — REPS
- Abdolmaleki et al. 2018 — MPO · Nair et al. 2020 — AWAC
- Frans et al. 2025 — CFGRL, "Diffusion guidance is a controllable policy improvement operator" (the direct basis for RECAP's advantage conditioning)
- Black et al. 2025 — π₀.₅ · Driess et al. — Knowledge Insulation
