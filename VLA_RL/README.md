# VLA_RL — Reinforcement Learning for VLA Models

Study set on RL-based self-improvement of vision-language-action (VLA) policies.
The three papers form one lineage: **AWR** is the base offline-RL algorithm,
**π*₀.₆ / RECAP** builds advantage-conditioned RL for VLAs on top of it, and
**Learning to Fold** combines AWR + RECAP into a prizewinning competition recipe.

| Paper | Year | arXiv | Why it's here |
|-------|------|-------|---------------|
| [Advantage-Weighted Regression](Advantage-Weighted_Regression_-_Simple_and_Scalable_Off-Policy_Reinforcement_Learning/) | 2019 | [1910.00177](https://arxiv.org/abs/1910.00177) | Foundation: off-policy RL via two supervised steps — value regression + advantage-weighted policy regression. |
| [π*₀.₆: a VLA That Learns From Experience](Pi-star-0.6_-_a_VLA_That_Learns_From_Experience/) | 2025 | [2511.14759](https://arxiv.org/abs/2511.14759) | RECAP: advantage-conditioned RL for VLAs mixing demos, on-policy rollouts, and teleop corrections; real-world laundry / box assembly / espresso tasks. |
| [Learning to Fold (LeHome Challenge 2026)](Learning_to_Fold_-_Prizewinning_Solution_at_LeHome_Challenge_2026/) | 2026 | [2606.27163](https://arxiv.org/abs/2606.27163) | Applied recipe: AWR + RECAP for flow-matching VLA, policy-as-value-function, async training pipeline, sim-to-real; 1st online / 2nd offline at ICRA 2026. |

Per-paper layout follows the repo convention: `<Title>__arXiv-<id>.pdf` (untracked) + `source.txt`.
