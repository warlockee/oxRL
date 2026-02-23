# oxRL: Post-Training That You Can Actually Read

**A lightweight, modular framework for reinforcement learning on large language models. ~4,300 lines of code. No magic. No mystery.**

---

## The Problem

Post-training large models with RL should not require a PhD in distributed systems. Yet the current landscape of open-source frameworks has drifted toward sprawling codebases where algorithm logic is tangled with infrastructure plumbing. You want to try a new reward function or tweak a policy loss? Good luck tracing through 50,000+ lines of tightly coupled code, hoping you don't break something upstream.

Researchers deserve better. Teams working on alignment, reasoning, and instruction following need tools that are powerful enough for production and simple enough to actually understand.

## What Is oxRL

oxRL is a post-training framework for LLMs, VLMs, and VLAs built around one idea: **you should be able to read the code that trains your model.**

It ships with PPO, SGRPO (a stability-focused GRPO variant), CISPO, and SFT. It scales to billions of parameters using DeepSpeed ZeRO-3, generates rollouts with vLLM, and orchestrates everything with Ray. The entire framework is ~4,300 lines of Python across 24 files.

```
oxRL/
├── algs/          PPO, SGRPO, CISPO, SFT    (~1,475 lines)
├── rollouts/      vLLM engine, replay buffer  (~706 lines)
├── configs/       Type-safe Pydantic configs   (~393 lines)
├── custom_datasets/  Data loading & mixing     (~514 lines)
├── rewards/       Pluggable reward functions    (~22 lines)
├── main_rl.py     RL training loop             (~582 lines)
└── main_sl.py     SL training loop             (~413 lines)
```

That's it. No hidden abstractions. No framework-within-a-framework.

## Why oxRL

### Algorithm code stays algorithmic

Every training algorithm lives in a single file under `algs/`. PPO is 460 lines. SGRPO is 436. You can read the entire policy loss computation, advantage estimation, and KL regularization in one sitting. Systems concerns (DeepSpeed init, distributed comms) are confined to explicit, well-marked boundaries.

### Stability over speed

RL for large models is brittle. Rewards are sparse, gradients are noisy, and subtle bugs silently corrupt training for hours before you notice. oxRL includes:

- **NaN detection** in advantage computation
- **Padding hole detection** to catch malformed masks
- **Terminal state validation** for correct EOS handling
- **Strict on-policy enforcement** with policy version tracking
- **Per-token log probability alignment** documented and verified throughout

These checks catch real bugs. The kind that waste GPU-days.

### Scales when you need it

oxRL is not a toy. It trains billion-parameter models using:

- **DeepSpeed ZeRO-3** with activation checkpointing and CPU offloading
- **vLLM** for high-throughput rollout generation with tensor parallelism
- **Ray** for actor-based distributed orchestration
- **Hot model refresh** — update the policy in rollout engines without restarting inference

### Configuration that catches mistakes early

The entire config system is built on Pydantic. Invalid types, missing fields, and inconsistent settings are caught at load time with clear error messages — not 45 minutes into a training run.

```yaml
train:
  alg_name: "sgrpo"
  clip_low: -0.2
  clip_high: 0.2
  kl_coeff: 0.0

rollout:
  temperature: 1.0
  max_tokens: 512
  n_samples: 8
  force_strict_on_policy: true
```

### Debuggable by design

oxRL runs rollout and training phases sequentially. This is a deliberate choice. Pipelined overlap would improve GPU utilization, but it makes debugging significantly harder. When your RL training diverges at step 4,000, you want to know exactly what happened — not untangle race conditions between concurrent rollout and gradient computation.

We plan to add pipelining as an option. But the default will always prioritize correctness and clarity.

## Algorithms

| Algorithm | Lines | What It Does |
|-----------|-------|-------------|
| **PPO** | 460 | GAE advantages, clipped policy loss, value function with clipping, entropy regularization |
| **SGRPO** | 436 | Stable GRPO variant with reference model KL, z-score advantages, deliberate deviations from the original paper for stability |
| **CISPO** | 437 | Importance-sampling weighted policy gradient with ratio clipping |
| **SFT** | 142 | Standard supervised fine-tuning with masked cross-entropy |

Each algorithm is a self-contained Ray actor with its own DeepSpeed engine. No shared mutable state. No implicit dependencies between algorithms.

## Who Is This For

**Research teams** who need to iterate on post-training methods quickly, understand what their code is doing, and reproduce results reliably.

**Production teams** who want a framework they can audit, extend, and trust — without committing to a monolithic codebase that evolves faster than their ability to review changes.

**Anyone** who has spent too long debugging a training run only to find the issue was three abstraction layers deep in framework code they never intended to touch.

## What oxRL Is Not

- It is not the fastest framework. Sequential rollout-training is a deliberate trade-off.
- It is not the most feature-complete. DPO is on the roadmap, not shipped yet.
- It does not hide complexity behind abstractions. If something is hard, you see it.

## Getting Started

```bash
git clone https://github.com/warlockee/oxRL.git
cd oxRL
pip install -r req.txt
```

Configure your training run in a YAML file, then:

```bash
# RL training (PPO, SGRPO, CISPO)
python main_rl.py --config configs/rl_args.yaml

# Supervised fine-tuning
python main_sl.py --config configs/sl_args.yaml
```

## Contributing

oxRL is open source because we want the community to stress-test it, extend it, and make it better. Contributions are welcome. The bar is simple: keep changes readable, testable, and debuggable. Follow the existing style. If your change adds complexity, it should be worth it.

## The Bet

Most RL frameworks optimize for throughput first and hope clarity follows. oxRL makes the opposite bet: **get the algorithms right first, make them understandable, and then optimize the systems around them.** RL for large models is still an open research problem. The bottleneck is not tokens per second — it is whether your training loop is doing what you think it is doing.

---

