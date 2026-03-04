# We implemented 51 post-training algorithms in one framework -- here's what we learned about which ones actually work

We spent weeks implementing every DPO variant we could find in the literature. 51 algorithms total, from vanilla DPO to things like "Discovering Preference Optimization Algorithms with LLMs" (yes, that is a real paper). After building, testing, and running them across 38 verified models from SmolLM2-135M to Qwen3.5-35B, here is our honest take on what actually matters.

## The framework

[oxRL](https://github.com/warlockee/oxRL) is a lightweight post-training framework for LLMs and VLMs. It handles SFT, DPO and its many variants, GRPO/PPO for RL, reward modeling, knowledge distillation, and continued pre-training. It scales with DeepSpeed and vLLM. The codebase is intentionally kept simple -- each algorithm is a single self-contained file, no 14-level inheritance chains.

```python
from oxrl import Trainer
trainer = Trainer(model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
trainer.train(task="reasoning")
```

38 models verified end-to-end (we actually run training, not just "it loads"). LoRA by default for 7B+, full fine-tuning for smaller models.

But you did not click on this for a framework pitch. Here is the interesting part.

## Tier list: which algorithms actually matter

### Tier 1: Use these

**SimPO** -- Reference-free, length-normalized. No ref model means half the GPU memory. The length normalization is simple but addresses a real problem: vanilla DPO rewards longer outputs. SimPO's loss is just `-log(sigmoid(beta * (avg_logp_chosen - avg_logp_rejected) - gamma))`. Clean and effective.

**CPO** -- Also reference-free, but adds an NLL regularizer on chosen responses. This is the "just keep the model good at generating the stuff you want" term. Sounds obvious, but it fixes DPO's biggest practical problem: chosen probability collapse (where the model learns by suppressing rejected responses instead of reinforcing chosen ones).

**DPNLL (DPO+NLL)** -- Same idea as CPO's regularizer but applied to standard DPO. Meta uses this in production. Axolotl and LLaMA-Factory support it. If you are running vanilla DPO, adding `alpha * NLL_on_chosen` is probably the single highest-value change you can make. It directly addresses likelihood displacement.

**R-DPO** -- Adds explicit length regularization. If you have noticed your DPO-trained model suddenly producing 3x longer outputs, this is the fix. Simple penalty term, meaningful impact.

**GRPO/SGRPO** -- For reasoning and math, RL still wins. Token-level clipped surrogate, no critic needed. We also have GSPO for MoE models (sequence-level ratios absorb routing noise between vLLM and DeepSpeed) and CISPO for when SGRPO shows reward hacking.

### Tier 2: Situationally useful

**IPO** -- Squared loss instead of log-sigmoid. More stable when beta tuning is hard. Good default if vanilla DPO is unstable but you do not want to think too hard.

**cDPO / CalDPO** -- Label smoothing variants. If your preference data is noisy (crowdsourced annotations, LLM-as-judge with known biases), these help. If your preference data is clean (verifiable rewards, math correctness), skip them. cDPO is the simpler one. CalDPO is more principled (NeurIPS 2024) but harder to tune.

**KTO** -- Does not need paired preferences, only binary "good/bad" labels. Useful when you have thumbs-up/thumbs-down data but not side-by-side comparisons. The Kahneman-Tversky framing is elegant but the practical advantage is about data format, not loss shape.

**ORPO** -- Reference-free via log-odds ratio. Works well in practice, good alternative to SimPO if you want a different inductive bias.

**DPOP** -- Prevents chosen probability from *decreasing*. Similar motivation to DPNLL but achieved through a different mechanism (max penalty). Useful safety net.

### Tier 3: Interesting research, unclear practical gains

**DiscoPOP** -- An LLM-discovered loss function that blends logistic and exponential losses. The paper is creative (NeurIPS 2024), and the adaptive blending is theoretically motivated. In practice, the improvement over DPO is marginal and you have an extra temperature hyperparameter to tune.

**EXO** -- Closed-form preference optimization. Mathematically elegant. We have not seen it clearly outperform SimPO or CPO in our runs.

**AOT** -- Alignment via Optimal Transport. Uses Wasserstein distance instead of KL. Cool math, unclear practical advantage for standard preference datasets.

**AlphaDPO, ChiPO, GPO** -- These generalize the divergence measure (alpha-divergence, chi-squared, parameterized family). They subsume DPO as a special case. In theory you could find a better divergence for your problem. In practice, the defaults usually land close to DPO.

**The remaining ~15 variants** -- BetaDPO, BPO, C2DPO, DPOShift, DrDPO, FDPO, FocalPO, HDPO, Hinge, MinorDPO, NCA, ODPO, RobustDPO, SamPO, SPO, SPPO, WPO. Each has a paper with benchmarks showing improvement. Most are marginal modifications to the loss function. We implemented them for completeness, not because we think you should use them over the Tier 1 options.

## The meta-observation

After implementing all 51, the pattern is clear: **the algorithms that matter solve structural problems** (no ref model, length bias, chosen collapse), not optimization-theoretic problems (better divergence measure, tighter bound, adaptive weighting). The DPO loss function itself is fine. The problems are in what surrounds it.

If someone asks us "which algorithm should I use?":
- **Reasoning/math/code**: GRPO with verifiable rewards.
- **General preference alignment with clean data**: SimPO or CPO.
- **General preference alignment with noisy data**: DPO+NLL with cDPO label smoothing.
- **Limited GPU memory**: SimPO (no ref model).

## Try it

```bash
pip install oxrl
```

GitHub: [github.com/warlockee/oxRL](https://github.com/warlockee/oxRL)

MIT licensed. 51 algorithms, 38 verified models, works with any HuggingFace model. Contributions welcome.
