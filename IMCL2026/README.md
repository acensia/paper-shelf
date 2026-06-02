# ICML 2026 — Cherry-Picked Papers

**43rd International Conference on Machine Learning** · Seoul, South Korea · July 6–11, 2026.
6,352 accepted / 23,918 submitted (26.6%). 168 Orals · 536 Spotlights.

Curated to three areas: **LLMs/Reasoning**, **Multimodal/Vision**, **Robotics** (plus an Agents/Benchmarks bucket).
PDFs are author **arXiv preprints** (official PMLR proceedings publish after the July conference; OpenReview camera-ready PDFs are not yet public). A few papers have no preprint yet — marked below.

_Generated 2026-06-02._

---

## Orals (top 0.7%)

### 🧠 LLMs / Reasoning

#### Do We Need Adam? Surprisingly Strong and Sparse RL with SGD in LLMs
*Sagnik Mukherjee, Lifan Yuan, Pavan Jayasinha, Dilek Hakkani-Tür, +1 more*

Reinforcement learning (RL), particularly RL from verifiable reward (RLVR), has become a crucial phase of training large language models (LLMs) and a key focus of current scaling efforts. However, optimization practices in RL largely follow those of next-token prediction stages (e.g., pretraining and supervised fine-tuning), despite fundamental differences between RL and these stages highlighted by recent work. One such practice is the use of the AdamW optimizer, which is widely adopted for training large-scale transformers despite its high memory overhead. Our analysis shows that both momentum and adaptive learning rates in AdamW are less influential in RL than in SFT, leading us to hypothesize that RL benefits less from Adam-style per-parameter adaptive learning rates and momentum. Confirming this hypothesis, our experiments demonstrate that the substantially more memory-efficient SGD, which is known to perform poorly in supervised learning of large-scale transformers, matches or even outperforms AdamW in RL for LLMs. Remarkably, full fine-tuning with SGD updates fewer than 0.02% of model parameters without any sparsity-promoting regularization, more than 1000 times fewer than AdamW. Our analysis offers potential reasons for this update sparsity. These findings provide new insights into the optimization dynamics of RL in LLMs and show that RL can be substantially more parameter-efficient than previously recognized.

📄 PDF downloaded · [ICML](https://icml.cc/virtual/2026/oral/71027) · [arXiv](https://arxiv.org/abs/2602.07729)

#### daVinci-Dev: Agent-native Mid-training for Software Engineering
*Ji Zeng, Dayuan Fu, Tiantian Mi, Yumin Zhuang, +13 more*

Recently, the frontier of Large Language Model (LLM) capabilities has shifted from single-turn code generation to agentic software engineering-a paradigm where models autonomously navigate, edit, and test complex repositories. While post-training methods have become the de facto approach for code agents, **agentic mid-training**-mid-training (MT) on large-scale data that mirrors authentic agentic workflows-remains critically underexplored due to substantial resource requirements, despite offering a more scalable path to instilling foundational agentic behaviors than relying solely on expensive reinforcement learning. A central challenge in realizing effective agentic mid-training is the distribution mismatch between static training data and the dynamic, feedback-rich environment of real development. To address this, we present a systematic study of agentic mid-training, establishing both the data synthesis principles and training methodology for effective agent development at scale. Central to our approach is **agent-native data**-supervision comprising two complementary types of trajectories: **contextually-native trajectories** that preserve the complete information flow an agent experiences, offering broad coverage and diversity; and **environmentally-native trajectories** collected from executable repositories where observations stem from actual tool invocations and test executions, providing depth and interaction authenticity. We verify the model's agentic capabilities on `SWE-Bench Verified`. We demonstrate our superiority over the previous open software engineering mid-training recipe `Kimi-Dev` under two post-training settings with an aligned base model and agentic scaffold, while using less than half mid-training tokens (73.1B). Besides relative advantage, our best performing 32B and 72B models achieve **56.1%** and **58.5%** resolution rates, respectively, which are ...

📄 PDF downloaded · [ICML](https://icml.cc/virtual/2026/oral/71032) · [arXiv](https://arxiv.org/abs/2601.18418)

#### Learning Unmasking Policies for Diffusion Language Models
*Metod Jazbec, Theo X. Olausson, Louis Béthune, Pierre Ablin, +5 more*

Diffusion (Large) Language Models (dLLMs) now match the downstream performance of their autoregressive counterparts on many tasks, while holding the promise of being more efficient during inference. One critical design aspect of dLLMs is the sampling procedure that selects which tokens to unmask at each diffusion step. Indeed, recent work has found that heuristic strategies such as confidence thresholding improve both sample quality and token throughput compared to random unmasking. However, such heuristics have downsides: they require manual tuning, and we observe that their performance degrades with larger block sizes. In this work, we instead propose to train sampling procedures using reinforcement learning. Specifically, we formalize masked diffusion sampling as a Markov decision process in which the dLLM serves as the environment, and propose a lightweight policy based on a single-layer transformer that maps dLLM token confidences to unmasking decisions. Our experiments show that these trained policies match the performance of state-of-the-art heuristics when combined with semi-autoregressive (block) generation, while outperforming them in the full-diffusion setting.

📄 PDF downloaded · [ICML](https://icml.cc/virtual/2026/oral/71028) · [arXiv](https://arxiv.org/abs/2512.09106)

#### Less is Enough: Synthesizing Diverse Data in LLM Feature Space
*Zhongzhi Li, Xuansheng Wu, Yijiang Li, Lijie Hu, +1 more*

The diversity of post-training data is critical for effective downstream performance in large language models (LLMs). Many existing approaches to constructing post-training data quantify diversity using text-based metrics that capture linguistic variation, but such metrics provide only weak signals for the task-relevant features that determine downstream performance. In this work, we introduce Feature Activation Coverage (FAC) which measures data diversity in an interpretable feature space. Building upon this metric, we further propose a diversity-driven data synthesis framework, named FAC Synthesis, that first uses a sparse autoencoder to identify missing features from a seed dataset, and then generates synthetic samples that explicitly reflect these features. Experiments show that our approach consistently improves both data diversity and downstream performance on various tasks, including instruction following, toxicity detection, reward modeling, and behavior steering. Interestingly, we identify a shared, interpretable feature space across model families (i.e., LLaMA, Mistral, and Qwen), enabling cross-model knowledge transfer. Our work provides a solid and practical methodology for exploring data-centric optimization of LLMs.

📄 PDF downloaded · [ICML](https://icml.cc/virtual/2026/oral/71029) · [arXiv](https://arxiv.org/abs/2602.10388)

#### OPUS: Efficient and Principled Data Selection in LLM Pre-training
*Shaobo Wang, Xuan Ouyang, Tianyi Xu, Yuzheng Hu, +8 more*

As high-quality public text approaches exhaustion, a phenomenon known as the Data Wall, pre-training is shifting from more tokens to better tokens. However, existing methods either rely on heuristic static filters that ignore training dynamics, or use dynamic yet optimizer-agnostic criteria based on raw gradients. We propose OPUS (Optimizer-induced Projected Utility Selection), a dynamic data selection framework that defines utility in the optimizer-induced update space. OPUS scores candidates by projecting their effective updates, shaped by modern optimizers, onto a target direction derived from a stable, in-distribution proxy. To ensure scalability, we employ Ghost technique with CountSketch for computational efficiency, and Boltzmann sampling for data diversity, incurring only 4.7\% additional compute overhead. OPUS achieves remarkable results across diverse corpora, quality tiers, optimizers, and model scales. In pre-training of GPT-2 Large/XL on FineWeb and FineWeb-Edu with 30B tokens, OPUS outperforms industrial-level baselines and even full 200B-token training. Moreover, when combined with industrial-level static filters, OPUS further improves pre-training efficiency, even with lower-quality data. Furthermore, in continued pre-training of Qwen3-8B-Base on SciencePedia, OPUS achieves superior performance using only 0.5B tokens compared to full training with 3B tokens, demonstrating significant data efficiency gains in specialized domains.

📄 PDF downloaded · [ICML](https://icml.cc/virtual/2026/oral/71030) · [arXiv](https://arxiv.org/abs/2602.05400)

#### Controlled LLM Training on Spectral Sphere
*Tian Xie, Haoming Luo, Haoyu Tang, Yiwen Hu, +8 more*

Scaling large models requires optimization strategies that ensure rapid convergence grounded in stability. Maximal Update Parametrization ($\boldsymbolμ$P) provides a theoretical safeguard for width-invariant $Θ(1)$ activation control, whereas emerging optimizers like Muon are only ``half-aligned'' with these constraints: they control updates but allow weights to drift. To address this limitation, we introduce the \textbf{Spectral Sphere Optimizer (SSO)}, which enforces strict module-wise spectral constraints on both weights and their updates. By deriving the steepest descent direction on the spectral sphere, SSO realizes a fully $\boldsymbolμ$P-aligned optimization process. To enable large-scale training, we implement SSO as an efficient parallel algorithm within Megatron. Through extensive pretraining on diverse architectures, including Dense 1.7B, MoE 8B-A1B, and 200-layer DeepNet models, SSO consistently outperforms AdamW and Muon. Furthermore, we observe significant practical stability benefits, including improved MoE router load balancing, suppressed outliers, and strictly bounded activations.

📄 PDF downloaded · [ICML](https://icml.cc/virtual/2026/oral/71055) · [arXiv](https://arxiv.org/abs/2601.08393)

#### The Obfuscation Atlas: Where Honesty Emerges in RLVR with Deception Probes
*Mohammad Taufeeque, Stefan Heimersheim, Adam Gleave, Chris Cundy*

Training against white-box deception detectors has been proposed as a way to make AI systems honest. However, such training risks models learning to obfuscate their deception to evade the detector. Prior work has studied obfuscation only in artificial settings where models were directly rewarded for harmful output. We construct a realistic coding environment where reward hacking via hardcoding test cases naturally occurs, and show that obfuscation emerges in this setting. We introduce a taxonomy of possible outcomes when training against a deception detector. The model either remains honest, or becomes deceptive via two possible obfuscation strategies. (i) Obfuscated activations: the model outputs deceptive text while modifying its internal representations to no longer trigger the detector. (ii) Obfuscated policy: the model outputs deceptive text that evades the detector, typically by including a justification for the reward hack. Empirically, obfuscated activations arise from representation drift during RL, with or without a detector penalty. The detector penalty only incentivizes obfuscated policies; we theoretically show this is expected for policy gradient methods. Sufficiently high KL regularization and detector penalty can yield honest policies, establishing white-box deception detectors as viable training signals for tasks prone to reward hacking.

📄 PDF downloaded · [ICML](https://icml.cc/virtual/2026/oral/71065) · [arXiv](https://arxiv.org/abs/2602.15515)

#### TG-RAG: Retrieval-Augmented Reasoning Guidance in Specialized Domains

⏳ no preprint yet (PMLR after conf.) · [ICML](https://icml.cc/virtual/2026/oral/71061)

#### Understanding Reasoning Collapse in LLM Agent Reinforcement Learning

⏳ no preprint yet (PMLR after conf.) · [ICML](https://icml.cc/virtual/2026/oral/71062)


### 👁️ Multimodal / Vision

#### Are VLMs Seeing or Just Saying? Uncovering the Illusion of Visual Re-examination
*Chufan Shi, Cheng Yang, Yaokang Wu, Linghao Jin, +3 more*

Vision-Language Models (VLMs) often produce self-reflective statements like "let me check the figure again" during reasoning. Do such statements trigger genuine visual re-examination, or are they merely learned textual patterns? We investigate this via VisualSwap, an image-swap probing framework: after a model reasons over an image, we replace it with a visually similar but semantically different one and test whether the model notices. We introduce VS-Bench, 800 image pairs curated from MathVista, MathVerse, MathVision, and MMMU-Pro. Experiments on Qwen3-VL, Kimi-VL, and ERNIE-VL reveal a striking failure: models overwhelmingly miss the swap, with accuracy dropping by up to 60%. Counterintuitively, thinking models are nearly 3x more vulnerable than their instructed counterparts, and scaling offers no mitigation. Multi-turn user instructions restore visual grounding, but self-generated reflective statements during continuous generation do not. Attention analysis explains why: user instructions substantially elevate attention to visual tokens, whereas self-reflection does not. Current VLMs tend to say rather than actually see when claiming to perform visual re-examination. Our code and dataset are available at the project page: https://visualswap.github.io

📄 PDF downloaded · [ICML](https://icml.cc/virtual/2026/oral/71047) · [arXiv](https://arxiv.org/abs/2605.15864)

#### Bad Seeing or Bad Thinking? Rewarding Perception for Multimodal Reasoning
*Haozhe Wang, Qixin Xu, Changpeng Wang, Taofeng Xue, +3 more*

Achieving robust perception-reasoning synergy is a central goal for advanced Vision-Language Models (VLMs). Recent advancements have pursued this goal via architectural designs or agentic workflows. However, these approaches are often limited by static textual reasoning or complicated by the significant compute and engineering burden of external agentic complexity. Worse, this heavy investment does not yield proportional gains, often witnessing a "seesaw effect" on perception and reasoning. This motivates a fundamental rethinking of the true bottleneck. In this paper, we argue that the root cause of this trade-off is an ambiguity in modality credit assignment: when a VLM fails, is it due to flawed perception ("bad seeing") or flawed logic ("bad thinking")? To resolve this, we introduce a reinforcement learning framework that improves perception-reasoning synergy by reliably rewarding the perception fidelity. We explicitly decompose the generation process into interleaved perception and reasoning steps. This decoupling enables targeted supervision on perception. Crucially, we introduce Perception Verification (PV), leveraging a "blindfolded reasoning" proxy to reward perceptual fidelity independently of reasoning outcomes. Furthermore, to scale training across free-form VL tasks, we propose Structured Verbal Verification, which replaces high-variance LLM judging with structured algorithmic execution. These techniques are integrated into a Modality-Aware Credit Assignment (MoCA) mechanism, which routes rewards to the specific source of error -- either bad seeing or bad thinking -- enabling a single VLM to achieve simultaneous performance gains across a wide task spectrum.

📄 PDF downloaded · [ICML](https://icml.cc/virtual/2026/oral/71095) · [arXiv](https://arxiv.org/abs/2605.14054)

#### Motion Attribution for Video Generation
*Xindi Wu, Despoina Paschalidou, Jun Gao, Antonio Torralba, +4 more*

Despite the rapid progress of video generation models, the role of data in influencing motion is poorly understood. We present Motive (MOTIon attribution for Video gEneration), a motion-centric, gradient-based data attribution framework that scales to modern, large, high-quality video datasets and models. We use this to study which fine-tuning clips improve or degrade temporal dynamics. Motive isolates temporal dynamics from static appearance via motion-weighted loss masks, yielding efficient and scalable motion-specific influence computation. On text-to-video models, Motive identifies clips that strongly affect motion and guides data curation that improves temporal consistency and physical plausibility. With Motive-selected high-influence data, our method improves both motion smoothness and dynamic degree on VBench, achieving a 74.1% human preference win rate compared with the pretrained base model. To our knowledge, this is the first framework to attribute motion rather than visual appearance in video generative models and to use it to curate fine-tuning data.

📄 PDF downloaded · [ICML](https://icml.cc/virtual/2026/oral/71049) · [arXiv](https://arxiv.org/abs/2601.08828)

#### CLEAR: Mask-Free Adaptive Subtitle Removal
*Qingdong He, Chaoyi Wang, Peng Tang, Yifan Yang, +1 more*

Video subtitle removal aims to distinguish text overlays from background content while preserving temporal coherence. Existing diffusion-based methods necessitate explicit mask sequences during both training and inference phases, which restricts their practical deployment. In this paper, we present CLEAR (Context-aware Learning for End-to-end Adaptive Video Subtitle Removal), a mask-free framework that achieves truly end-to-end inference through context-aware adaptive learning. Our two-stage design decouples prior extraction from generative refinement: Stage I learns disentangled subtitle representations via self-supervised orthogonality constraints on dual encoders, while Stage II employs LoRA-based adaptation with generation feedback for dynamic context adjustment. Notably, our method only requires 0.77% of the parameters of the base diffusion model for training. On Chinese subtitle benchmarks, CLEAR outperforms mask-dependent baselines by + 6.77dB PSNR and -74.7% VFID, while demonstrating superior zero-shot generalization across six languages (English, Korean, French, Japanese, Russian, German), a performance enabled by our generation-driven feedback mechanism that ensures robust subtitle removal without ground-truth masks during inference.

📄 PDF downloaded · [ICML](https://icml.cc/virtual/2026/oral/71048) · [arXiv](https://arxiv.org/abs/2603.21901)

#### PhotoAgent: Exploratory Visual Aesthetic Planning with Large Vision Models
*Mingde Yao, Zhiyuan You, King-Man Tam, Menglu Wang, +1 more*

With the recent fast development of generative models, instruction-based image editing has shown great potential in generating high-quality images. However, the quality of editing highly depends on carefully designed instructions, placing the burden of task decomposition and sequencing entirely on the user. To achieve autonomous image editing, we present PhotoAgent, a system that advances image editing through explicit aesthetic planning. Specifically, PhotoAgent formulates autonomous image editing as a long-horizon decision-making problem. It reasons over user aesthetic intent, plans multi-step editing actions via tree search, and iteratively refines results through closed-loop execution with memory and visual feedback, without requiring step-by-step user prompts. To support reliable evaluation in real-world scenarios, we introduce UGC-Edit, an aesthetic evaluation benchmark consisting of 7,000 photos and a learned aesthetic reward model. We also construct a test set containing 1,017 photos to systematically assess autonomous photo editing performance. Extensive experiments demonstrate that PhotoAgent consistently improves both instruction adherence and visual quality compared with baseline methods. The project page is https://mdyao.github.io/PhotoAgent/.

📄 PDF downloaded · [ICML](https://icml.cc/virtual/2026/oral/71050) · [arXiv](https://arxiv.org/abs/2602.22809)


### 🤖 Robotics

#### RoboMME: Benchmarking Memory for Robotic Generalist Policies
*Yinpei Dai, Hongze Fu, Jayjun Lee, Yuejiang Liu, +5 more*

Memory is critical for long-horizon and history-dependent robotic manipulation. Such tasks often involve counting repeated actions or manipulating objects that become temporarily occluded. Recent vision-language-action (VLA) models have begun to incorporate memory mechanisms; however, their evaluations remain confined to narrow, non-standardized settings. This limits systematic understanding, comparison, and progress measurement. To address these challenges, we introduce RoboMME: a large-scale standardized benchmark for evaluating and advancing VLA models in long-horizon, history-dependent scenarios. Our benchmark comprises 16 manipulation tasks constructed under a carefully designed taxonomy that evaluates temporal, spatial, object, and procedural memory. We further develop a suite of 14 memory-augmented VLA variants built on the π0.5 backbone to systematically explore different memory representations across multiple integration strategies. Experimental results show that the effectiveness of memory representations is highly task-dependent, with each design offering distinct advantages and limitations across different tasks. Videos and code can be found at our website https://robomme.github.io.

📄 PDF downloaded · [ICML](https://icml.cc/virtual/2026/oral/71077) · [arXiv](https://arxiv.org/abs/2603.04639)

#### HALO: Human-Robot Collaboration via Heterogeneous-Agent Lyapunov Policy Optimization
*Hao Zhang, Yaru Niu, Yikai Wang, Ding Zhao, +1 more*

To improve generalization and resilience in human-robot collaboration (HRC), robots must contend with diverse combinations of human behaviors and contexts, motivating multi-agent reinforcement learning (MARL). However, inherent heterogeneity between robots and humans creates a rationality gap (RG), where decentralized policy updates deviate from cooperative joint optimization. The resulting learning problem is a general-sum differentiable game, so independent policy-gradient updates can oscillate or diverge without added structure. We propose heterogeneous-agent Lyapunov policy optimization (HALO), a framework that stabilizes decentralized MARL by enforcing Lyapunov-based contraction in policy-parameter space. Unlike Lyapunov-based safe RL, which targets state/trajectory constraints in constrained Markov decision processes, HALO uses Lyapunov certification to stabilize decentralized policy learning. HALO rectifies decentralized gradients via optimal quadratic projections, ensuring monotonic contraction of RG and enabling effective exploration of open-ended interaction spaces. Extensive simulations and real-world humanoid-robot experiments show that this certified stability improves generalization and robustness in collaborative corner cases. Our project website is available at https://HaoZhang-THU.github.io/HALO/.

📄 PDF downloaded · [ICML](https://icml.cc/virtual/2026/oral/71075) · [arXiv](https://arxiv.org/abs/2603.03741)

#### Unsupervised Partner Design Enables Robust Ad-hoc Teamwork
*Constantin Ruhdorfer, Matteo Bortoletto, Victor Oei, Anna Penzkofer, +1 more*

We introduce Unsupervised Partner Design (UPD) - a population-free, multi-agent reinforcement learning framework for robust ad-hoc teamwork that adaptively generates training partners without requiring pretrained partners or manual parameter tuning. UPD constructs diverse partners by stochastically mixing an ego agent's policy with biased random behaviours and scores them using a variance-based learnability metric that prioritises partners near the ego agent's current learning frontier. We show that UPD can be integrated with unsupervised environment design, resulting in the first method enabling fully unsupervised curricula over both level and partner distributions in a cooperative setting. Through extensive evaluations on Overcooked-AI and the Overcooked Generalisation Challenge, we demonstrate that this dynamic partner curriculum is highly effective: UPD consistently outperforms both population-based and population-free baselines as well as ablations. In a user study, we further show that UPD achieves higher returns than all baselines and was perceived as significantly more adaptive, more human-like, a better collaborator, and less frustrating.

📄 PDF downloaded · [ICML](https://icml.cc/virtual/2026/oral/71078) · [arXiv](https://arxiv.org/abs/2508.06336)


### 🧭 Agents / Benchmarks

#### VenusBench-Mobile: User-Centric Benchmark for Mobile GUI Agents
*Yichen Gong, Zhuohan Cai, Sunhao Dai, Yuqi Zhou, +3 more*

Existing online benchmarks for mobile GUI agents remain largely app-centric and task-homogeneous, failing to reflect the diversity and instability of real-world mobile usage. To this end, we introduce VenusBench-Mobile, a challenging online benchmark for evaluating general-purpose mobile GUI agents under realistic, user-centric conditions. VenusBench-Mobile builds two core evaluation pillars: defining what to evaluate via user-intent-driven task design that reflects real mobile usage, and how to evaluate through a capability-oriented annotation scheme for fine-grained agent behavior analysis. Extensive evaluation of state-of-the-art mobile GUI agents reveals large performance gaps relative to prior benchmarks, indicating that VenusBench-Mobile poses substantially more challenging and realistic tasks and that current agents remain far from reliable real-world deployment. Diagnostic analysis further shows that failures are dominated by deficiencies in perception and memory, which are largely obscured by coarse-grained evaluations. Moreover, even the strongest agents exhibit near-zero success under environment variations, highlighting their brittleness in realistic settings. Based on these insights, we believe VenusBench-Mobile provides an important stepping stone toward robust real-world deployment of mobile GUI agents. Code and data are available at https://github.com/inclusionAI/UI-Venus/tree/VenusBench-Mobile.

📄 PDF downloaded · [ICML](https://icml.cc/virtual/2026/oral/71034) · [arXiv](https://arxiv.org/abs/2604.06182)

#### Strategic Navigation or Stochastic Search? Agents and Humans over Documents
*Łukasz Borchmann, Jordy Van Landeghem, Michał Turski, Shreyansh Padarha, +11 more*

Multimodal agents offer a promising path to automating complex document-intensive workflows. Yet, a critical question remains: do these agents demonstrate genuine strategic reasoning, or merely stochastic trial-and-error search? To address this, we introduce MADQA, a benchmark of 2,250 human-authored questions grounded in 800 heterogeneous PDF documents. Guided by Classical Test Theory, we design it to maximize discriminative power across varying levels of agentic abilities. To evaluate agentic behaviour, we introduce a novel evaluation protocol measuring the accuracy-effort trade-off. Using this framework, we show that while the best agents can match human searchers in raw accuracy, they succeed on largely different questions and rely on brute-force search to compensate for weak strategic planning. They fail to close the nearly 20% gap to oracle performance, persisting in unproductive loops. We release the dataset and evaluation harness to help facilitate the transition from brute-force retrieval to calibrated, efficient reasoning.

📄 PDF downloaded · [ICML](https://icml.cc/virtual/2026/oral/71033) · [arXiv](https://arxiv.org/abs/2603.12180)

#### Benchmarking at the Edge of Comprehension

⏳ no preprint yet (PMLR after conf.) · [ICML](https://icml.cc/virtual/2026/oral/71031)


---

## Additional Orals — from `report.md` cross-reference

These 7 are ICML 2026 **Orals** outside the original 3-area shortlist (theory, trust, spatial/3D, causality, RL, interpretability). Added with new category folders. Only those with a public arXiv preprint have a PDF; the rest carry a placeholder until PMLR publishes.

### 🌐 Theory / World Modeling

#### Learning to Theorize the World from Observation
⏳ no preprint yet (PMLR after conf.) · [ICML](https://icml.cc/virtual/2026/oral/71176) · [OpenReview](https://openreview.net/forum?id=wsA8LgHU5U)


### 🛡️ Trustworthiness / Hallucination

#### DOUBT: Decoupled Object-level Understanding and Bridging via vMF-based Trustworthiness for Hallucination Detection in MLLMs
⏳ no preprint yet (PMLR after conf.) · [ICML](https://icml.cc/virtual/2026/oral/71175) · [OpenReview](https://openreview.net/forum?id=QMf9CBpEIf)


### 🧊 Spatial / 3D Vision

#### Holi-Spatial: Evolving Video Streams into Holistic 3D Spatial Intelligence
*Yuanyuan Gao, Hao Li, Yifei Liu, Xinhao Ji, +13 more*

The pursuit of spatial intelligence fundamentally relies on access to large-scale, fine-grained 3D data. However, existing approaches predominantly construct spatial understanding benchmarks by generating question-answer (QA) pairs from a limited number of manually annotated datasets, rather than systematically annotating new large-scale 3D scenes from raw web data. As a result, their scalability is severely constrained, and model performance is further hindered by domain gaps inherent in these narrowly curated datasets. In this work, we propose Holi-Spatial, the first fully automated, large-scale, spatially-aware multimodal dataset, constructed from raw video inputs without human intervention, using the proposed data curation pipeline. Holi-Spatial supports multi-level spatial supervision, ranging from geometrically accurate 3D Gaussian Splatting (3DGS) reconstructions with rendered depth maps to object-level and relational semantic annotations, together with corresponding spatial Question-Answer (QA) pairs. Following a principled and systematic pipeline, we further construct Holi-Spatial-4M, the first large-scale, high-quality 3D semantic dataset, containing 12K optimized 3DGS scenes, 1.3M 2D masks, 320K 3D bounding boxes, 320K instance captions, 1.2M 3D grounding instances, and 1.2M spatial QA pairs spanning diverse geometric, relational, and semantic reasoning tasks. Holi-Spatial demonstrates exceptional performance in data curation quality, significantly outperforming existing feed-forward and per-scene optimized methods on datasets such as ScanNet, ScanNet++, and DL3DV. Furthermore, fine-tuning Vision-Language Models (VLMs) on spatial reasoning tasks using this dataset has also led to substantial improvements in model performance.

📄 PDF downloaded · [ICML](https://icml.cc/virtual/2026/oral/71165) · [OpenReview](https://openreview.net/forum?id=UGAP2F6FfV) · [arXiv](https://arxiv.org/abs/2603.07660)


### 🔗 Causality

#### A Recursive Decomposition Framework for Causal Structure Learning in the Presence of Latent Variables
⏳ no preprint yet (PMLR after conf.) · [ICML](https://icml.cc/virtual/2026/oral/71123) · [OpenReview](https://openreview.net/forum?id=jdbQzlnYll)


### 🎮 Reinforcement Learning

#### Video-Based Optimal Transport for Feedback-Efficient Offline Preference-Based Reinforcement Learning
⏳ no preprint yet (PMLR after conf.) · [ICML](https://icml.cc/virtual/2026/oral/71090) · [OpenReview](https://openreview.net/forum?id=G8LVO5easu)


### 🧬 Interpretability / Memory

#### AI Engram: In Search of Memory Traces in Artificial Intelligence
⏳ no preprint yet (PMLR after conf.) · [ICML](https://icml.cc/virtual/2026/oral/71045) · [OpenReview](https://openreview.net/forum?id=QZO3oby12w)


### 🧭 Agents / Benchmarks

#### $\tau^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment
⏳ no preprint yet (PMLR after conf.) · [ICML](https://icml.cc/virtual/2026/oral/71171) · [OpenReview](https://openreview.net/forum?id=OC2z7iSQKa)


---

## Spotlights (2.2%) — in-scope: 188 · PDFs downloaded: 84

Listed by area. ✅ = arXiv PDF in repo; ⏳ = no public preprint matched yet.

### 🧠 LLMs / Reasoning (127)

- ✅ ADEPT: RL-Aligned Agentic Decoding of Emotion via Evidence Probing Tools — From Consensus Learning to Ambiguity-Driven Emotion Reasoning · [arXiv](https://arxiv.org/abs/2602.12714) · [OpenReview](https://openreview.net/forum?id=maA1s1S0db)
- ✅ Alignment Pretraining: AI Discourse Causes Self-Fulfilling (Mis)alignment · [arXiv](https://arxiv.org/abs/2601.10160) · [OpenReview](https://openreview.net/forum?id=951OAanYyQ)
- ✅ Alignment-Sensitive Minimax Rates for Spectral Algorithms with Learned Kernels · [arXiv](https://arxiv.org/abs/2509.20294) · [OpenReview](https://openreview.net/forum?id=4HrWo5x7YF)
- ✅ Beyond Global Alignment: Fine-Grained Motion-Language Retrieval via Pyramidal Shapley-Taylor Learning · [arXiv](https://arxiv.org/abs/2601.21904) · [OpenReview](https://openreview.net/forum?id=RuTNWE1wr7)
- ✅ Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum · [arXiv](https://arxiv.org/abs/2510.00526) · [OpenReview](https://openreview.net/forum?id=2hQBG2ZlFb)
- ✅ Controlled LLM Training on Spectral Sphere · [arXiv](https://arxiv.org/abs/2601.08393) · [OpenReview](https://openreview.net/forum?id=5kTn1c3vtt)
- ✅ daVinci-Dev: Agent-native Mid-training for Software Engineering · [arXiv](https://arxiv.org/abs/2601.18418) · [OpenReview](https://openreview.net/forum?id=a86luANykT)
- ✅ DLM: Unified Decision Language Models for Offline Multi-Agent Sequential Decision Making · [arXiv](https://arxiv.org/abs/2604.23557) · [OpenReview](https://openreview.net/forum?id=qxG09uid4p)
- ✅ Effective Reasoning Chains Reduce Intrinsic Dimensionality · [arXiv](https://arxiv.org/abs/2602.09276) · [OpenReview](https://openreview.net/forum?id=vCjGjvrIR9)
- ✅ Efficient numeracy in language models through single-token number embeddings · [arXiv](https://arxiv.org/abs/2510.06824) · [OpenReview](https://openreview.net/forum?id=Bh4Ubk80M8)
- ✅ Efficient Parallel Samplers for Recurrent-Depth Models · [arXiv](https://arxiv.org/abs/2510.14961) · [OpenReview](https://openreview.net/forum?id=h7WBYYJF1Q)
- ✅ Emergent Analogical Reasoning in Transformers · [arXiv](https://arxiv.org/abs/2602.01992) · [OpenReview](https://openreview.net/forum?id=GxTkgMBiz8)
- ✅ End-to-End Compression for Tabular Foundation Models · [arXiv](https://arxiv.org/abs/2602.05649) · [OpenReview](https://openreview.net/forum?id=84mfkGDxYh)
- ✅ Estimating Tail Risks in Language Model Output Distributions · [arXiv](https://arxiv.org/abs/2604.22167) · [OpenReview](https://openreview.net/forum?id=Joka19sTny)
- ✅ FIRE: Multi-fidelity Regression with Distribution-conditioned In-context Learning using Tabular Foundation Models · [arXiv](https://arxiv.org/abs/2601.22371) · [OpenReview](https://openreview.net/forum?id=JxbxHB5d9v)
- ✅ How RL Unlocks the Aha Moment in Geometric Interleaved Reasoning · [arXiv](https://arxiv.org/abs/2603.01070) · [OpenReview](https://openreview.net/forum?id=JT4zAf3zDs)
- ✅ Incremental BPE Tokenization · [arXiv](https://arxiv.org/abs/2605.30813) · [OpenReview](https://openreview.net/forum?id=ZbWgrDzCQo)
- ✅ Jailbreak to Protect: Buffering Harmful Fine-Tuning via Temporary Jailbreaking LoRA in Large Language Models · [arXiv](https://arxiv.org/abs/2605.24550) · [OpenReview](https://openreview.net/forum?id=O0JXgBBo3k)
- ✅ Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates · [arXiv](https://arxiv.org/abs/2601.18510) · [OpenReview](https://openreview.net/forum?id=pLvye0zHUC)
- ✅ Language Model Circuits Are Sparse in the Neuron Basis · [arXiv](https://arxiv.org/abs/2601.22594) · [OpenReview](https://openreview.net/forum?id=OrviwFWcN4)
- ✅ Learning Structured Reasoning via Tractable Trajectory Control · [arXiv](https://arxiv.org/abs/2603.01641) · [OpenReview](https://openreview.net/forum?id=x6J7va4BYz)
- ✅ Learning Unmasking Policies for Diffusion Language Models · [arXiv](https://arxiv.org/abs/2512.09106) · [OpenReview](https://openreview.net/forum?id=F9NDKf5oPy)
- ✅ Long-Context Modeling with Dynamic Hierarchical Sparse Attention for Memory-Constrained LLM Inference · [arXiv](https://arxiv.org/abs/2510.24606) · [OpenReview](https://openreview.net/forum?id=o3gN27ITWV)
- ✅ MASPOB: Bandit-Based Prompt Optimization for Multi-Agent Systems with Graph Neural Networks · [arXiv](https://arxiv.org/abs/2603.02630) · [OpenReview](https://openreview.net/forum?id=A5venxSvpw)
- ✅ Mechanistic Data Attribution: Tracing the Training Origins of Interpretable LLM Units · [arXiv](https://arxiv.org/abs/2601.21996) · [OpenReview](https://openreview.net/forum?id=PQaxfoEcRc)
- ✅ MetaphorVU: Towards Metaphorical Video Understanding · [arXiv](https://arxiv.org/abs/2605.25461) · [OpenReview](https://openreview.net/forum?id=yKcBAJMPXZ)
- ✅ MiniAppBench: Evaluating the Shift from Text to Interactive HTML Responses in LLM-Powered Assistants · [arXiv](https://arxiv.org/abs/2603.09652) · [OpenReview](https://openreview.net/forum?id=pwbLmew1aq)
- ✅ Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling · [arXiv](https://arxiv.org/abs/2602.10623) · [OpenReview](https://openreview.net/forum?id=DfhMMHXDuu)
- ✅ NonZero: Interaction-Guided Exploration for Multi-Agent Monte Carlo Tree Search · [arXiv](https://arxiv.org/abs/2605.00751) · [OpenReview](https://openreview.net/forum?id=Jh6gq9QsFa)
- ✅ Offline Reinforcement Learning of High-Quality Behaviors Under Robust Style Alignment · [arXiv](https://arxiv.org/abs/2601.22823) · [OpenReview](https://openreview.net/forum?id=jxdjDmWpNX)
- ✅ PLANTAIN: Plan-Answer Interleaved Reasoning · [arXiv](https://arxiv.org/abs/2512.03176) · [OpenReview](https://openreview.net/forum?id=lK2o9OjoXf)
- ✅ POET-X: Memory-efficient LLM Training by Scaling Orthogonal Transformation · [arXiv](https://arxiv.org/abs/2603.05500) · [OpenReview](https://openreview.net/forum?id=et8jpWLUuD)
- ✅ PonderLM-2: Pretraining LLM with Latent Thoughts in Continuous Space · [arXiv](https://arxiv.org/abs/2509.23184) · [OpenReview](https://openreview.net/forum?id=yVFxjNzCQm)
- ✅ Position: Assistive Agents Need Accessibility Alignment · [arXiv](https://arxiv.org/abs/2605.13579) · [OpenReview](https://openreview.net/forum?id=aAQNmZxEJa)
- ✅ Position: Modular Memory is the Key to Continual Learning Agents · [arXiv](https://arxiv.org/abs/2603.01761) · [OpenReview](https://openreview.net/forum?id=iBXcqA5N6j)
- ✅ Position: Universal Aesthetic Alignment Narrows Artistic Expression · [arXiv](https://arxiv.org/abs/2512.11883) · [OpenReview](https://openreview.net/forum?id=1gQ4zc1Q8I)
- ✅ Prescriptive Scaling Reveals the Evolution of Language Model Capabilities · [arXiv](https://arxiv.org/abs/2602.15327) · [OpenReview](https://openreview.net/forum?id=IkjsHRpuYY)
- ✅ Pressure Reveals Character: Behavioural Alignment Evaluation at Depth · [arXiv](https://arxiv.org/abs/2602.20813) · [OpenReview](https://openreview.net/forum?id=3U1l24EDvL)
- ✅ Procedural Pretraining: Warming Up Language Models with Abstract Data · [arXiv](https://arxiv.org/abs/2601.21725) · [OpenReview](https://openreview.net/forum?id=XFTTezxLdU)
- ✅ Quantifying Frontier LLM Capabilities for Container Sandbox Escape · [arXiv](https://arxiv.org/abs/2603.02277) · [OpenReview](https://openreview.net/forum?id=19AbP986bv)
- ✅ Reinforcement Learning with Evolving Rubrics for Deep Research · [arXiv](https://arxiv.org/abs/2511.19399) · [OpenReview](https://openreview.net/forum?id=97NEP1pyS3)
- ✅ Rethinking LLM Ensembling from the Perspective of Mixture Models · [arXiv](https://arxiv.org/abs/2605.00419) · [OpenReview](https://openreview.net/forum?id=YVk8EDxBWx)
- ✅ Reward-free Alignment for Conflicting Objectives · [arXiv](https://arxiv.org/abs/2602.02495) · [OpenReview](https://openreview.net/forum?id=vSzRJyg6k0)
- ✅ Safety Alignment of LMs via Non-cooperative Games · [arXiv](https://arxiv.org/abs/2512.20806) · [OpenReview](https://openreview.net/forum?id=Bve790HQrA)
- ✅ Scaling Law for Quantization-Aware Training · [arXiv](https://arxiv.org/abs/2505.14302) · [OpenReview](https://openreview.net/forum?id=fXr3uPr1G5)
- ✅ Skill Neologisms: Towards Skill-based Continual Learning · [arXiv](https://arxiv.org/abs/2605.04970) · [OpenReview](https://openreview.net/forum?id=5VgZUEpK6W)
- ✅ Steer Like the LLM: Activation Steering that Mimics Prompting · [arXiv](https://arxiv.org/abs/2605.03907) · [OpenReview](https://openreview.net/forum?id=06Nk3dJDMq)
- ✅ Surgery: Mitigating Harmful  Fine-Tuning for Large Language Models via Attention Sink · [arXiv](https://arxiv.org/abs/2602.05228) · [OpenReview](https://openreview.net/forum?id=6ojsIliNF0)
- ✅ T$^2$PO: Uncertainty-Guided Exploration Control for Stable Multi-Turn Agentic Reinforcement Learning · [arXiv](https://arxiv.org/abs/2605.02178) · [OpenReview](https://openreview.net/forum?id=aD1zjvdJN4)
- ✅ The Obfuscation Atlas: Mapping Where Honesty Emerges in RLVR with Deception Probes · [arXiv](https://arxiv.org/abs/2602.15515) · [OpenReview](https://openreview.net/forum?id=wrGSN9kAVD)
- ✅ The Power of Power Law: Asymmetry Enables Compositional Reasoning · [arXiv](https://arxiv.org/abs/2604.22951) · [OpenReview](https://openreview.net/forum?id=K83orsg2X9)
- ✅ The Signal is in the Steps: Local Scoring for Reasoning Data Selection · [arXiv](https://arxiv.org/abs/2510.03988) · [OpenReview](https://openreview.net/forum?id=GcB3a6IonG)
- ✅ The Value of Variance: Mitigating Debate Collapse in Multi-Agent Systems via Uncertainty-Driven Policy Optimization · [arXiv](https://arxiv.org/abs/2602.07186) · [OpenReview](https://openreview.net/forum?id=Bc6c2OVWRh)
- ✅ ThreadWeaver: Adaptive Threading for Efficient Parallel Reasoning in Language Models · [arXiv](https://arxiv.org/abs/2512.07843) · [OpenReview](https://openreview.net/forum?id=Efq2VvYk1o)
- ✅ TokSuite: Measuring the Impact of Tokenizer Choice on Language Model Behavior · [arXiv](https://arxiv.org/abs/2512.20757) · [OpenReview](https://openreview.net/forum?id=vIZz7LvObC)
- ✅ Toward Stable Value Alignment: Introducing Independent Modules for Consistent Value Guidance · [arXiv](https://arxiv.org/abs/2605.11712) · [OpenReview](https://openreview.net/forum?id=R80JLmXXPJ)
- ✅ Towards Efficient LLMs Annealing with Principled Sample Selection · [arXiv](https://arxiv.org/abs/2605.31175) · [OpenReview](https://openreview.net/forum?id=2UH01A9Za0)
- ✅ WaterSIC: information-theoretically (near) optimal linear layer quantization · [arXiv](https://arxiv.org/abs/2603.04956) · [OpenReview](https://openreview.net/forum?id=fCPgAHIciE)
- ✅ WeDLM: Reconciling Diffusion Language Models with Standard Causal Attention for Fast Inference · [arXiv](https://arxiv.org/abs/2512.22737) · [OpenReview](https://openreview.net/forum?id=QwtmbKAOZU)
- ⏳ A Regret Minimization Framework on Preference Learning  in Large Language Models · [OpenReview](https://openreview.net/forum?id=genVnYBAV7)
- ⏳ Adaptive Testing for LLM Evaluation: A Psychometric Alternative to Static Benchmarks · [OpenReview](https://openreview.net/forum?id=ItJEC1Mk0V)
- ⏳ Advancing LLM Reasoning with Natural Language and Numerical Feedback · [OpenReview](https://openreview.net/forum?id=gz7hVnrRWq)
- ⏳ Balancing Understanding and Generation in Discrete Diffusion Models · [OpenReview](https://openreview.net/forum?id=pZNo1YWT5x)
- ⏳ CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs · [OpenReview](https://openreview.net/forum?id=9uZJLXt7fq)
- ⏳ Catch-22: On the Fundamental Tradeoff Between Detectability and Robustness in LLM Watermarking · [OpenReview](https://openreview.net/forum?id=097IlaUfQY)
- ⏳ CausalGame: Benchmarking Causal Thinking of LLM Agents in Games · [OpenReview](https://openreview.net/forum?id=WNqIX3IFZU)
- ⏳ Characterizing, Evaluating, and Optimizing Complex Reasoning · [OpenReview](https://openreview.net/forum?id=IMFgiWw4jd)
- ⏳ ClinTutor-R1: Advancing Scalable and Robust One-to-Many Alignment in Clinical Socratic Education · [OpenReview](https://openreview.net/forum?id=15Gs65kvHS)
- ⏳ Conditional Equivalence of DPO and RLHF: Assumptions, Failure Modes, and Provable Alignment · [OpenReview](https://openreview.net/forum?id=7UEBX1KU1y)
- ⏳ Decision Transformers As Zero-Shot Learners via Text-Behavior Alignment · [OpenReview](https://openreview.net/forum?id=jqLPUccarO)
- ⏳ DecodeShare: Tracing the Shared Pathways of LLM Decode-Time Decisions · [OpenReview](https://openreview.net/forum?id=txdOohIkYJ)
- ⏳ Demystifying Entropy Control in LLM RL Training: Theoretical Analysis and Dynamic Scheduling · [OpenReview](https://openreview.net/forum?id=hq2MPYXAko)
- ⏳ Diffract: Spectral View of LLM Domain Adaptation · [OpenReview](https://openreview.net/forum?id=XBUHoiAGDE)
- ⏳ Disentangling Geometry, Performance, and Training in Language Models · [OpenReview](https://openreview.net/forum?id=kNb6NNhKgE)
- ⏳ Don't Force the Fit: Bounded Log-Likelihood Loss for Enhanced Reasoning in Large Language Models · [OpenReview](https://openreview.net/forum?id=mTMA9yvsnx)
- ⏳ Don't Reinvent the Wheel, Just Realign the Spokes: Resource-Efficient Federated Fine-Tuning via Rank-Wise Expert Assembly · [OpenReview](https://openreview.net/forum?id=HdsEUNo4C4)
- ⏳ Evaluating Robustness of Reasoning Models on Parameterized Logical Problems · [OpenReview](https://openreview.net/forum?id=gnLZWOubWa)
- ⏳ FedPissa: Towards Federated Personalized Adaptation of Foundation Models via LoRA Subspace Mapping · [OpenReview](https://openreview.net/forum?id=ZEWN40uNEh)
- ⏳ FOCUS & RePAIR: Mitigating Text Degeneration via Token-Level Guidance For Pruned Large Language Models · [OpenReview](https://openreview.net/forum?id=3SgRoBn3XM)
- ⏳ Geometry-Aware Decoding with Wasserstein-Regularized Truncation and Mass Penalties for Large Language Models · [OpenReview](https://openreview.net/forum?id=HSuU4xBmAv)
- ⏳ Hista and Numca: Estimate State Value Effectively for Large Language Model Reinforcement Learning · [OpenReview](https://openreview.net/forum?id=ff4b0ELdU0)
- ⏳ HypoSpace: A Diagnostic Benchmark for Set-Valued Hypothesis Generation under Underdetermination and Sublinear Coverage Bounds · [OpenReview](https://openreview.net/forum?id=QpjtK65JHO)
- ⏳ Information Flow Reveals When to Trust Language Models · [OpenReview](https://openreview.net/forum?id=vd8HzoFZ7v)
- ⏳ Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives · [OpenReview](https://openreview.net/forum?id=aGaBFE9pta)
- ⏳ Latent Collaboration in Multi-Agent Systems · [OpenReview](https://openreview.net/forum?id=syG9I9ofd8)
- ⏳ Learning Randomized Reductions · [OpenReview](https://openreview.net/forum?id=hCAEcqig2C)
- ⏳ LiftQuant: Continuous Bit-Width Control for Pareto-Optimal LLM Deployment · [OpenReview](https://openreview.net/forum?id=1GvXUhLIMP)
- ⏳ Maximum Likelihood Reinforcement Learning · [OpenReview](https://openreview.net/forum?id=EeuLO2BjFN)
- ⏳ MemoryBench: A Benchmark for Memory and Continual Learning in LLM Systems · [OpenReview](https://openreview.net/forum?id=If4X4W2HWx)
- ⏳ Modeling Hierarchical Thinking in Large Reasoning Models · [OpenReview](https://openreview.net/forum?id=N44P3zrrgO)
- ⏳ Modular Pretraining Enables Access Control · [OpenReview](https://openreview.net/forum?id=yIubI9l3IT)
- ⏳ OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration · [OpenReview](https://openreview.net/forum?id=6C4YQcq8YX)
- ⏳ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models · [OpenReview](https://openreview.net/forum?id=TBaUfO9znF)
- ⏳ On the Limits of LLM Adaptability: Impact of LLM Pre-Training on Annotation Task Performance · [OpenReview](https://openreview.net/forum?id=oTv2bKG5Qg)
- ⏳ OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration · [OpenReview](https://openreview.net/forum?id=lcLxGrk5N9)
- ⏳ Position: Measuring Human Preferences in RLHF is a Social Science Problem · [OpenReview](https://openreview.net/forum?id=5l1pca4KhM)
- ⏳ Position: The Alignment Community is Unintentionally Building a Censor’s Toolkit · [OpenReview](https://openreview.net/forum?id=dy2HwmOvFX)
- ⏳ PRISM: Demystifying Retention and Interaction in Mid-Training · [OpenReview](https://openreview.net/forum?id=giNBsVVFGt)
- ⏳ ProcMEM: Learning Reusable Procedural Memory from Experience via Non-Parametric PPO for LLM Agents · [OpenReview](https://openreview.net/forum?id=9kJQjx2B80)
- ⏳ Rare Event Analysis of Large Language Models · [OpenReview](https://openreview.net/forum?id=2RJN5vDHG0)
- ⏳ ReQAT: Achieving Full-Precision Reasoning Accuracy with 4-bit Floating-Point Quantization-Aware Training · [OpenReview](https://openreview.net/forum?id=aKY52fnzgc)
- ⏳ Reward and Guidance through Rubrics: Promoting Exploration to Improve Multi-Domain Reasoning · [OpenReview](https://openreview.net/forum?id=AfqsNFzJcs)
- ⏳ Security–Fidelity Tradeoffs: No Universal Defense Against Prompt Injection · [OpenReview](https://openreview.net/forum?id=vTJmIhNvC3)
- ⏳ Shared Semantics, Divergent Mechanisms: Unsupervised Feature Discovery by Aligning Semantics and Mechanisms · [OpenReview](https://openreview.net/forum?id=C9AhjL8aUZ)
- ⏳ Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance · [OpenReview](https://openreview.net/forum?id=OyPE1ganBR)
- ⏳ Stop When Further Reasoning Won’t Help: Attention-State Adaptive Generation in Reasoning Models · [OpenReview](https://openreview.net/forum?id=zRMK32N6t6)
- ⏳ SVL: Empowering Spiking Neural Networks for Efficient 3D Open-World Understanding · [OpenReview](https://openreview.net/forum?id=Ai3a79cTvr)
- ⏳ Sycophancy Towards Researchers Drives Performative Misalignment · [OpenReview](https://openreview.net/forum?id=rLFIOikFR2)
- ⏳ Symmetry Reveals the In-Context Classifier: Transformers Implement Mean-Shift Dynamics · [OpenReview](https://openreview.net/forum?id=fDMizWsNoG)
- ⏳ Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability · [OpenReview](https://openreview.net/forum?id=GnqHK8Ww98)
- ⏳ TG-RAG: A Retrieval-Augmented Framework for Reasoning Guidance in Specialized Domains · [OpenReview](https://openreview.net/forum?id=W34UyCRQel)
- ⏳ The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models · [OpenReview](https://openreview.net/forum?id=kpgURPRMGf)
- ⏳ The Tell-Tale Norm: $\ell_2$ Magnitude as a Signal for Reasoning Dynamics in Large Language Models · [OpenReview](https://openreview.net/forum?id=03ZTlJuX0y)
- ⏳ Thinking in Flow: A Dissipative Stabilization Operator for Robust Autoregressive Reasoning · [OpenReview](https://openreview.net/forum?id=9IQpUEKOGM)
- ⏳ Tilt Matching for Scalable Sampling and Fine-Tuning · [OpenReview](https://openreview.net/forum?id=dQA4Gjt4KU)
- ⏳ Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs · [OpenReview](https://openreview.net/forum?id=KY5Q9V9F5C)
- ⏳ Training Diffusion Language Models for Black-Box Optimization · [OpenReview](https://openreview.net/forum?id=Z7wI0sor6i)
- ⏳ Uncovering the Latent Potential of Deep Intermediate Representations · [OpenReview](https://openreview.net/forum?id=6up1qGJwYZ)
- ⏳ Understanding Reasoning Collapse in LLM Agent Reinforcement Learning · [OpenReview](https://openreview.net/forum?id=01caH9oj7C)
- ⏳ Universal Redundancies in Time Series Foundation Models · [OpenReview](https://openreview.net/forum?id=DyA4KHj1wy)
- ⏳ VALUEFLOW: Toward Pluralistic and Steerable Value-based Alignment in Large Language Models · [OpenReview](https://openreview.net/forum?id=6zVV84vnCJ)
- ⏳ VenusBench-Mobile: A Challenging and User-Centric Benchmark for Mobile GUI Agents with Capability Diagnostics · [OpenReview](https://openreview.net/forum?id=coHiGZOFtS)
- ⏳ VisionWebDev: A Hierarchical Benchmark for Visual Website Development with Agent Verification · [OpenReview](https://openreview.net/forum?id=lJpXXwhRRF)
- ⏳ Wait, Wait, Wait... Why Do Reasoning Models Loop? · [OpenReview](https://openreview.net/forum?id=oZWE7mSqlk)
- ⏳ What Preferences Can—and Cannot—Predict in Multi-Agent Online Learning · [OpenReview](https://openreview.net/forum?id=5W30WwL8wt)
- ⏳ When to Trust the Cheap Check: Weak and Strong Verification for Reasoning · [OpenReview](https://openreview.net/forum?id=dfN4HMZbc9)
- ⏳ Why Deep Jacobian Spectra Separate: Depth-Induced Scaling and Singular-Vector Alignment · [OpenReview](https://openreview.net/forum?id=2kSBDoP1rE)

### 👁️ Multimodal / Vision (41)

- ✅ Agent0-VL: Exploring Self-Evolving Agent for Tool-Integrated Vision-Language Reasoning · [arXiv](https://arxiv.org/abs/2511.19900) · [OpenReview](https://openreview.net/forum?id=aA02f0y0LY)
- ✅ Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models · [arXiv](https://arxiv.org/abs/2605.30038) · [OpenReview](https://openreview.net/forum?id=vKWxArobP3)
- ✅ Bad Seeing or Bad Thinking? Rewarding Perception for Multimodal Reasoning · [arXiv](https://arxiv.org/abs/2605.14054) · [OpenReview](https://openreview.net/forum?id=duzdftKtUA)
- ✅ Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding · [arXiv](https://arxiv.org/abs/2606.00564) · [OpenReview](https://openreview.net/forum?id=Jb1YBPqyb7)
- ✅ Diffuse to Detect: Bi-Level Sample Rebalancing with Pseudo-Label Diffusion for Point-Supervised Infrared Small-Target Detection · [arXiv](https://arxiv.org/abs/2605.20766) · [OpenReview](https://openreview.net/forum?id=RNn2wtNeoK)
- ✅ End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer · [arXiv](https://arxiv.org/abs/2605.00503) · [OpenReview](https://openreview.net/forum?id=fSnxuRhWoC)
- ✅ Enhancing Reasoning for Diffusion LLMs via Distribution Matching Policy Optimization · [arXiv](https://arxiv.org/abs/2510.08233) · [OpenReview](https://openreview.net/forum?id=09CSjVeDug)
- ✅ LiME: Lightweight Mixture of Experts for Efficient Multimodal Multi-task Learning · [arXiv](https://arxiv.org/abs/2604.02338) · [OpenReview](https://openreview.net/forum?id=KRSZj8z5Lr)
- ✅ LIMSSR: LLM-Driven Sequence-to-Score Reasoning under Training-Time Incomplete Multimodal Observations · [arXiv](https://arxiv.org/abs/2605.00434) · [OpenReview](https://openreview.net/forum?id=0RLF7Sj2Fg)
- ✅ Mitigating Hallucinations in Large Vision-Language Models via Causal Route Gating · [arXiv](https://arxiv.org/abs/2605.24024) · [OpenReview](https://openreview.net/forum?id=LIcj73RLX6)
- ✅ Multimodal Latent Language Modeling with Next-Token Diffusion · [arXiv](https://arxiv.org/abs/2412.08635) · [OpenReview](https://openreview.net/forum?id=PnTXyTR2VG)
- ✅ PanoWorld-X: Generating Explorable Panoramic Worlds via Sphere-Aware Video Diffusion · [arXiv](https://arxiv.org/abs/2509.24997) · [OpenReview](https://openreview.net/forum?id=xEgoeNrp8B)
- ✅ RelaxFlow: Text-Driven Amodal 3D Generation · [arXiv](https://arxiv.org/abs/2603.05425) · [OpenReview](https://openreview.net/forum?id=UamxHbDR3p)
- ✅ VectorWorld: Efficient Streaming World Model via Diffusion Flow on Vector Graphs · [arXiv](https://arxiv.org/abs/2603.17652) · [OpenReview](https://openreview.net/forum?id=yTEEiE3YtD)
- ✅ VideoFlexTok: Flexible-Length Coarse-to-Fine Video Tokenization · [arXiv](https://arxiv.org/abs/2604.12887) · [OpenReview](https://openreview.net/forum?id=KyVlaw4BxE)
- ✅ What You Think is What You See: Driving Exploration in VLM Agents via Visual-Linguistic Curiosity · [arXiv](https://arxiv.org/abs/2605.03782) · [OpenReview](https://openreview.net/forum?id=zWStW3hrfW)
- ✅ When the Prompt Becomes Visual: Vision-Centric Jailbreak Attacks for Large Image Editing Models · [arXiv](https://arxiv.org/abs/2602.10179) · [OpenReview](https://openreview.net/forum?id=wQxRphkfxn)
- ⏳ 3ViewSense: Spatial and Mental Perspective Reasoning from Orthographic Views in Vision-Language Models · [OpenReview](https://openreview.net/forum?id=Hm8OEDKpiO)
- ⏳ Any-Order GPT as Masked Diffusion Model: Decoupling Formulation and Architecture · [OpenReview](https://openreview.net/forum?id=sEYoG3tAXN)
- ⏳ Decoupling Skeleton and Flesh: Efficient Multimodal Table Reasoning with  Disentangled Alignment and Structure-aware Guidance · [OpenReview](https://openreview.net/forum?id=PN7l7mBPDO)
- ⏳ Dissecting Multimodal In-Context Learning: Modality Asymmetries and Circuit Dynamics in modern Transformers · [OpenReview](https://openreview.net/forum?id=fhPu6dCiwt)
- ⏳ DroneDINO: Towards Heterogeneous Routed Mixture of Experts for Drone-based Unified Object Detection · [OpenReview](https://openreview.net/forum?id=vPGmYeL1rG)
- ⏳ Dual-Latent Memory Routing for Vision-Language Reasoning · [OpenReview](https://openreview.net/forum?id=SFWWUr9V7c)
- ⏳ Efficient Diffusion Models under Nonconvex Equality and Inequality constraints via Landing · [OpenReview](https://openreview.net/forum?id=6VEqdKmfKh)
- ⏳ Flex-Forcing: Towards a Unified Autoregressive and Bidirectional Video Diffusion Model · [OpenReview](https://openreview.net/forum?id=COGc3MSR7n)
- ⏳ Inference Time Concept Removal Guidance for Text-to-Image Diffusion Models · [OpenReview](https://openreview.net/forum?id=BSDvEh7oqy)
- ⏳ Information-Theoretic Disentangled Latent Modeling with Conditional Diffusion for Incomplete Multi-View Clustering · [OpenReview](https://openreview.net/forum?id=Wm3XgP6xQ8)
- ⏳ Mind-Omni: A Unified Multi-Task Framework for Brain-Vision-Language Modeling via Discrete Diffusion · [OpenReview](https://openreview.net/forum?id=3gCdh3u2GK)
- ⏳ Motion Attribution for Video Generation · [OpenReview](https://openreview.net/forum?id=zAl9heLw4q)
- ⏳ Multimodal Nested Learning for Decoupled and Coordinated Optimization · [OpenReview](https://openreview.net/forum?id=8XOncDWc5j)
- ⏳ PhotoAgent: Exploratory Visual Aesthetic Planning with Large Vision Models · [OpenReview](https://openreview.net/forum?id=Ws8swqL5ob)
- ⏳ Position: VLM Causal Reasoning Benchmarks Should Probe Temporal Understanding, Not Presume It · [OpenReview](https://openreview.net/forum?id=NmiiK9yQMM)
- ⏳ Prototype-guided Bilateral Alignment Multimodal Federated Learning · [OpenReview](https://openreview.net/forum?id=aNgkaEagBG)
- ⏳ Securing Multimodal AI through Internal Information Decomposition · [OpenReview](https://openreview.net/forum?id=GEzZIUmEqE)
- ⏳ Seizure-Semiology-Suite($S^3$): A Clinically Multimodal Dataset, Benchmark, and Models for Seizure Semiology Understanding · [OpenReview](https://openreview.net/forum?id=MyorUlHKVc)
- ⏳ SpatioLM: Towards General Physical Spatial Intelligence in Vision-Language Models · [OpenReview](https://openreview.net/forum?id=CHavqrN1X9)
- ⏳ SplAttN: Bridging 2D and 3D with Gaussian Soft Splatting and Attention for Point Cloud Completion · [OpenReview](https://openreview.net/forum?id=vTp9JToZl9)
- ⏳ Towards High-Fidelity CAD Generation via LLM-Driven Program Generation and Text-Based B-Rep Primitive Grounding · [OpenReview](https://openreview.net/forum?id=jfOhTGs5G5)
- ⏳ Towards Unified Multimodal Pretraining · [OpenReview](https://openreview.net/forum?id=xAKrRYm6pc)
- ⏳ When Attributes Disagree:  Gradient Conflict in Image Aesthetic Assessment · [OpenReview](https://openreview.net/forum?id=GrIs035ec3)
- ⏳ World-Model Inspired Emotion-aware Token Refinement for Training-Free Multimodal Emotion Recognition · [OpenReview](https://openreview.net/forum?id=ViQO8FlRFR)

### 🤖 Robotics (20)

- ✅ EcoVLA: Environment-Aware Adaptive Pruning with Interleaved Inference Orchestration for Vision-Language-Action Models · [arXiv](https://arxiv.org/abs/2602.00780) · [OpenReview](https://openreview.net/forum?id=7kH5uphAes)
- ✅ From Abstraction to Instantiation: Learning Behavioral Representation for Vision-Language-Action Model · [arXiv](https://arxiv.org/abs/2605.22671) · [OpenReview](https://openreview.net/forum?id=2NRrsz4nUB)
- ✅ LaST$_{0}$: Latent Spatio-Temporal Chain-of-Thought for Robotic Vision-Language-Action Model · [arXiv](https://arxiv.org/abs/2601.05248) · [OpenReview](https://openreview.net/forum?id=lwOoBzJykL)
- ✅ Learning Human-Robot Collaboration via Heterogeneous-Agent Lyapunov Policy Optimization · [arXiv](https://arxiv.org/abs/2603.03741) · [OpenReview](https://openreview.net/forum?id=uBvlUE2wrP)
- ✅ Position: Good Embodied Reward Models Need Bad Behavior Data · [arXiv](https://arxiv.org/abs/2606.01036) · [OpenReview](https://openreview.net/forum?id=0zeOMzrV2N)
- ✅ Pretrained Vision-Language-Action Models are Surprisingly Resistant to Forgetting in Continual Learning · [arXiv](https://arxiv.org/abs/2603.03818) · [OpenReview](https://openreview.net/forum?id=VzdSHEab4G)
- ✅ SceneSmith: Agentic Generation of Simulation-Ready Indoor Scenes · [arXiv](https://arxiv.org/abs/2602.09153) · [OpenReview](https://openreview.net/forum?id=WwS8CTpUA6)
- ✅ XR-1: Towards Versatile Vision-Language-Action Models via Learning Unified Vision-Motion Representations · [arXiv](https://arxiv.org/abs/2511.02776) · [OpenReview](https://openreview.net/forum?id=JO0IsGJg16)
- ⏳ DreamDojo: A Real-Time Robot World Model from Large-Scale Human Videos · [OpenReview](https://openreview.net/forum?id=FuvU7PTyED)
- ⏳ EgoTactile: Learning Grasp Pressure for Everyday Objects from Egocentric Video · [OpenReview](https://openreview.net/forum?id=xBkLTpOu2V)
- ⏳ From Pixels to Tokens: A Systematic Study of Latent Action Supervision for Vision-Language-Action Models · [OpenReview](https://openreview.net/forum?id=VMsumctGvg)
- ⏳ Joint-Space Empowerment as a Theory of Dexterous Motor Coordination · [OpenReview](https://openreview.net/forum?id=qI2eHwfNfh)
- ⏳ MSP: Probabilistically Consistent Multi-Scale Action Generation · [OpenReview](https://openreview.net/forum?id=zqjKnzyRo4)
- ⏳ OXE-AugE: A Large-Scale Robot Augmentation of OXE for Scaling Cross-Embodiment Policy Learning · [OpenReview](https://openreview.net/forum?id=LcswwEzzX7)
- ⏳ PACT: Self-Evolving Physical Safety Alignment for Diffusion Policies in Embodied Manipulation · [OpenReview](https://openreview.net/forum?id=ePFvXPdvhM)
- ⏳ RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies · [OpenReview](https://openreview.net/forum?id=8m30ogkPk2)
- ⏳ SCALE: Self-uncertainty Conditioned Adaptive Looking and Execution for Vision-Language-Action Models · [OpenReview](https://openreview.net/forum?id=7MlfE2Da2W)
- ⏳ Scaling Real-World Robot Policy Evaluation via Discrete Diffusion World Model · [OpenReview](https://openreview.net/forum?id=93uNlQ1Qp0)
- ⏳ UniMapping: Unified SLAM Framework for Map-Centric Embodied Perception · [OpenReview](https://openreview.net/forum?id=bQkKVGuHZA)
- ⏳ WestWorld: A Knowledge-Encoded Scalable Trajectory World Model for Diverse Robotic Systems · [OpenReview](https://openreview.net/forum?id=ncRRCG4BfP)
