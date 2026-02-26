<p align="center">
  <img src="assets/logo.png" alt="oxRL" width="200">
</p>

<h1 align="center">oxRL</h1>

<p align="center"><strong>Post-train any model under 10 lines of code.</strong></p>

<p align="center">A lightweight post-training framework for LLMs, VLMs, and VLAs. Maximizing developer speed. Scales to billions of parameters with DeepSpeed, vLLM, and Ray.</p>

---
## Usage

Post-train any model in under 10 lines of code. oxRL auto-detects your hardware, auto-prepares datasets, and scales to multi-GPU automatically.

```python
from oxrl import Trainer

# 1. Initialize with any HuggingFace model
trainer = Trainer(model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

# 2. Start reasoning post-training (Open-R1 recipe)
trainer.train(task="reasoning")
```

---

## Supported Models

The following models have been verified and onboarded using our automated pipeline. You can find ready-to-use scripts in the `examples/recipes/` directory.

| Model | Size | Task | Strategy | Status |
|:---|:---|:---|:---|:---|
| **DeepSeek-R1-Distill-Llama-8B** | 8.0B | Reasoning | LoRA | ✅ Verified |
| **DeepSeek-R1-Distill-Qwen-7B** | 7.0B | Reasoning | LoRA | ✅ Verified |
| **Qwen2.5-Coder-7B-Instruct** | 7.6B | Coding | LoRA | ✅ Verified |
| **Qwen2-Audio-7B-Instruct** | 7.0B | Audio | LoRA | ✅ Verified |
| **Qwen2-VL-7B-Instruct** | 7.0B | Vision | LoRA | ✅ Verified |
| **Gemma-3-1b-it** | 1.0B | Multimodal | Full-tuning | ✅ Verified |
| **Mistral-7B-Instruct-v0.3** | 7.0B | Instruct | LoRA | ✅ Verified |
| **Qwen2.5-7B-Instruct** | 7.0B | Math | LoRA | ✅ Verified |
| **SmolLM2-1.7B-Instruct** | 1.7B | Instruct | Full-tuning | ✅ Verified |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         oxRL Framework                          │
├─────────────────────┬───────────────────┬───────────────────────┤
│   Training Engines  │  Rollout Engines  │    Config + Data      │
│   (Ray + DeepSpeed) │  (Ray + vLLM)     │    (Pydantic + HF)    │
├─────────────────────┼───────────────────┼───────────────────────┤
│                     │                   │                       │
│  oxrl/algs/ppo.py   │                   │ oxrl/configs/load.py  │
│    LoRA / PEFT      │   replay_buffer.py│                       │
│  oxrl/algs/sft.py   │                   │ oxrl/datasets/        │
│                     │                   │   (Multimodal Ready)  │
├─────────────────────┴───────────────────┴───────────────────────┤
│  oxrl/swarm/        │  oxrl/utils/log.   │  oxrl/rewards/          │
│    orchestrator.py  │  oxrl/utils/setup. │  (Math / Code / etc.)   │
└──────────────────┴────────────────────┴─────────────────────────┘
```

## RL Training Workflow

1.  **Scout Agent:** Discovers model metadata and ensures `chat_template` compatibility.
2.  **Multimodal Pipeline:** Converts base64 images/audio into PIL/NumPy for vLLM rollouts.
3.  **LoRA Lifecycle:** Train with adapters, save with gathered ZeRO-3 weights, and **auto-strip PEFT prefixes** for immediate vLLM compatibility.
4.  **Verifiable Rewards:** Programmatic verification of CoT tags and mathematical correctness.

## Quick Start

### Installation

```bash
pip install oxrl
```

### Environment Diagnostics

Before starting a long training run, verify your environment (GPUs, CUDA Toolkit, DeepSpeed, Ray) with our diagnostic tool:

```bash
oxrl doctor
```

### Post-train a Reasoning Model

```yaml
# config.yaml
model:
  name: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
lora:
  enabled: true
reward:
  reward_func: "reasoning_reward_func"
data:
  dataset: "openr1_math"
```

```bash
python main_rl.py --config-file config.yaml
```

## Algorithms

| Algorithm | File | Description |
|-----------|------|-------------|
| **SFT** | `oxrl/algs/sft.py` | Supervised Fine-Tuning — Cross-entropy loss with masking and normalization. |
| **SGRPO** | `oxrl/algs/grpo.py` | Stable GRPO — Clipped surrogate loss with LoRA support and reference-free variants. |
| **SimPO** | `oxrl/algs/simpo.py` | Simple Preference Optimization — Reference-free and length-normalized alignment. |
| **CISPO** | `oxrl/algs/grpo.py` | Clipped importance-sampling policy optimization. |
| **PPO** | `oxrl/algs/ppo.py` | Proximal Policy Optimization with GAE and value clipping. |

## Project Structure

```
oxRL/
├── oxrl/                   # Core Framework Package
│   ├── trainer.py          # High-level Trainer API
│   ├── rewards/            # Verifiable reasoning and coding rewards (math, code, etc.)
│   ├── algs/               # Algorithm implementations (GRPO, PPO, SimPO, SFT)
│   ├── swarm/              # Autonomous model onboarding (Scout, Bugfixer)
│   ├── preprocessing/      # Reasoning (OpenR1), Multimodal (Vision/Audio) preprocessors
│   ├── rollouts/           # vLLM inference with structured prompt support
│   └── datasets/           # Dataset loaders and samplers
├── main_rl.py              RL training loop (Ray + DeepSpeed)
├── main_sl.py              SFT training loop (DeepSpeed)
├── examples/               Ready-to-use recipes and training scripts
└── setup.py                Packaging and Installation
```

## design-principles

**Debuggability over Pipelining.** oxRL avoids complex async pipelining to ensure that failure states are 100% reproducible and logs are clear.

**Robust Environment Handling.** oxRL is designed to work even in constrained environments. It automatically handles common CUDA/DeepSpeed mismatches by providing actionable warnings instead of fatal crashes.

**Autonomous Bug Reporting.** On framework failure, oxRL provides structured diagnostic signals for AI agents to automatically generate and submit GitHub issues (requires `GITHUB_TOKEN` environment variable).

**LoRA-first for 7B+**. We default to LoRA for larger models to enable high-quality research on consumer-grade and restricted high-end hardware.

**Verification-driven RL.** We prioritize datasets where the reward is verifiable (Math, Code, Format) to drive logical discovery.

## LLM Developer Map

This repository is optimized for LLM-assisted development (Claude/Gemini). If you are asking an AI to work on this framework, refer them to these "High-Signal" files:

- **Bug Reporting:** See `BUG_REPORTING.md` for instructions on autonomous issue submission.
- **Adding a New Algorithm:** See `oxrl/algs/base.py` (Base Class) and `oxrl/algs/grpo.py` (Implementation).
- **Adding a Reward Function:** Add to `oxrl/rewards/` using the signature in `oxrl/rewards/base.py`.
- **Changing Model Loading:** See `oxrl/utils/setup.py` -> `load_model_and_ref`.
- **Training Logic:** The main loop resides in `main_rl.py`.
- **Config Validation:** Logic is in `oxrl/configs/load.py`.

## Contributing

Contributions are welcome. Please follow the existing architectural patterns and style.

## FAQ

Check out the [FAQ](FAQ.md) for details on LoRA merging and Multimodal input formatting.
