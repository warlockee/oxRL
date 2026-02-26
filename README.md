<p align="center">
  <img src="assets/logo.png" alt="oxRL" width="200">
</p>

<h1 align="center">oxRL</h1>

<p align="center"><strong>Post-train any model under 10 lines of code.</strong></p>

<p align="center">A lightweight post-training framework for LLMs, VLMs, and VLAs. Maximizing developer speed. Scales to billions of parameters with DeepSpeed, vLLM, and Ray.</p>

---
## 🚀 New in v1.1: Reasoning & Multimodal RL

We've significantly expanded oxRL's capabilities to support the latest trending architectures and training recipes:

*   **Verifiable Reasoning (Open-R1):** Native support for reasoning models with `<thought>` and `<answer>` tag enforcement and rule-based correctness rewards.
*   **Simple Preference Optimization (SimPO):** State-of-the-art reference-free alignment that reduces VRAM by 40% and improves logical reasoning.
*   **Multimodal RL:** Support for Vision-Language (VLM) and Audio-Language models. Seamless base64-to-tensor pipeline for on-policy rollouts.
*   **GPQA & ScienceQA:** Integrated high-difficulty reasoning and multimodal datasets.
*   **Memory-Efficient LoRA:** Built-in PEFT integration allows post-training 14B+ models on restricted hardware.

---
## Usage (Python API)

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

The following models have been verified and onboarded using our automated pipeline. You can find ready-to-use scripts in the `examples/onboarded_models/` directory.

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
│  oxrl/swarm/        │  oxrl/utils/log.   │  oxrl/rewards.py        │
│    orchestrator.py  │  oxrl/utils/setup. │  (Reasoning / Code)     │
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
└── setup.py                Packaging and Installation
```

## design-principles

**Debuggability over Pipelining.** oxRL avoids complex async pipelining to ensure that failure states are 100% reproducible and logs are clear.

**LoRA-first for 7B+**. We default to LoRA for larger models to enable high-quality research on consumer-grade and restricted high-end hardware.

**Verification-driven RL.** We prioritize datasets where the reward is verifiable (Math, Code, Format) to drive logical discovery.

## Contributing

Contributions are welcome. Please follow the existing architectural patterns and style.

## FAQ

Check out the [FAQ](FAQ.md) for details on LoRA merging and Multimodal input formatting.
