<p align="center">
  <img src="assets/logo.png" alt="oxRL" width="200">
</p>

<h1 align="center">An agent-friendly framework for any post-training</h1>

<p align="center">A lightweight post-training framework for LLMs and VLMs. Maximizing developer speed. Scales to billions of parameters with DeepSpeed, vLLM, and Ray.</p>

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

oxRL works with **any HuggingFace model** that supports `AutoModelForCausalLM`, including multimodal models via `AutoModelForImageTextToText`. No special integration needed — just pass the model name.

### Verified Models

These models have been explicitly verified through our automated onboarding pipeline:

| Model | Size | Task | Strategy |
|:---|:---|:---|:---|
| **Qwen3-0.6B** | 0.6B | Instruct | Full-tuning |
| **Qwen2.5-0.5B-Instruct** | 0.5B | Math | Full-tuning |
| **Gemma-3-1b-it** | 1.0B | Instruct | Full-tuning |
| **Qwen2.5-1.5B-Instruct** | 1.5B | Math | Full-tuning |
| **Qwen2.5-Coder-1.5B-Instruct** | 1.5B | Coding | Full-tuning |
| **SmolLM2-1.7B-Instruct** | 1.7B | Instruct | Full-tuning |
| **Qwen2.5-3B-Instruct** | 3.0B | Math | Full-tuning |
| **DeepSeek-R1-Distill-Qwen-7B** | 7.6B | Reasoning | LoRA |
| **Qwen2.5-7B-Instruct** | 7.0B | Math | LoRA |
| **Qwen2.5-Coder-7B-Instruct** | 7.6B | Coding | LoRA |
| **Mistral-7B-Instruct-v0.3** | 7.0B | Instruct | LoRA |
| **Qwen2-Audio-7B-Instruct** | 7.0B | Audio | LoRA |
| **Qwen2-VL-7B-Instruct** | 7.0B | Vision | LoRA |
| **DeepSeek-R1-Distill-Llama-8B** | 8.0B | Reasoning | LoRA |
| **Qwen3.5-35B-A3B** | 35.0B (3B active) | Reasoning | LoRA |

### Config-Ready Models

These models have auto-generated configs in the registry and are queued for verification. They are expected to work out of the box:

| Model | Size | Task | Strategy |
|:---|:---|:---|:---|
| **Qwen2.5-Coder-0.5B-Instruct** | 0.5B | Coding | Full-tuning |
| **Llama-3.2-1B-Instruct** | 1.2B | Instruct | Full-tuning |
| **Qwen2.5-Math-1.5B-Instruct** | 1.5B | Math | Full-tuning |
| **Qwen2-VL-2B-Instruct** | 2.0B | Vision | Full-tuning |
| **Qwen3-VL-2B-Instruct** | 2.0B | Vision | Full-tuning |
| **SmolVLM-Instruct** | 2.3B | Vision | Full-tuning |
| **Gemma-2-2b-it** | 2.6B | Instruct | Full-tuning |
| **Qwen2.5-Coder-3B-Instruct** | 3.0B | Coding | Full-tuning |
| **Qwen2.5-VL-3B-Instruct** | 3.0B | Vision | Full-tuning |
| **Llama-3.2-3B-Instruct** | 3.2B | Instruct | Full-tuning |
| **Phi-3.5-mini-instruct** | 3.8B | Math | Full-tuning |
| **Qwen3-4B** | 4.0B | Math | Full-tuning |
| **Qwen3-VL-4B-Instruct** | 4.0B | Vision | Full-tuning |
| **Qwen2.5-Math-7B-Instruct** | 7.0B | Math | LoRA |
| **Qwen2.5-VL-7B-Instruct** | 7.0B | Vision | LoRA |
| **Qwen3-8B** | 8.0B | Math | LoRA |
| **Llama-3.1-8B-Instruct** | 8.0B | Reasoning | LoRA |
| **Phi-4** | 14.7B | Math | LoRA |
| **Qwen3.5-27B** | 27.0B | Instruct | LoRA |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          oxRL Framework                          │
├────────────────────────────────┬─────────────────────────────────┤
│     RL Path (main_rl.py)       │     SL Path (main_sl.py)        │
│  SGRPO / CISPO / PPO          │  SFT / DPO / ORPO / KTO         │
│  RLHF / RLAIF                 │  CPT / KD / RM / RFT            │
│  Ray actors + vLLM rollouts    │  OnlineDPO / SPIN / IPO / SimPO │
│                                │  DeepSpeed distributed training  │
├────────────────────────────────┴─────────────────────────────────┤
│  oxrl/algs/        Algorithms    │  oxrl/rollouts/   vLLM + Replay│
│  oxrl/configs/     Pydantic cfg  │  oxrl/rewards/    Verifiable   │
│  oxrl/datasets/    HF loaders    │  oxrl/utils/      Setup + Logs │
└──────────────────────────────────────────────────────────────────┘
```

## RL Training Workflow

1.  **Scout Agent:** Discovers model metadata and ensures `chat_template` compatibility.
2.  **Multimodal Pipeline:** Converts base64 images/audio into PIL/NumPy for vLLM rollouts.
3.  **LoRA Lifecycle:** Train with adapters, save with gathered ZeRO-3 weights, and **auto-strip PEFT prefixes** for immediate vLLM compatibility.
4.  **Verifiable Rewards:** Programmatic verification of CoT tags and mathematical correctness.

## Getting Started

### Installation

```bash
# From source (recommended for development)
git clone https://github.com/warlockee/oxRL.git
cd oxRL
pip install -e .

# Or from PyPI
pip install oxrl
```

### Run Tests

```bash
pip install pytest
pytest tests/test_bugs.py -v
```

### Environment Diagnostics

Before starting a long training run, verify your environment (GPUs, CUDA Toolkit, DeepSpeed, Ray) with our diagnostic tool:

```bash
oxrl doctor
```

### Configuration

oxRL uses YAML config files. See `oxrl/configs/rl_args.yaml` (RL) and `oxrl/configs/sl_args.yaml` (SL) for all available options with documentation. Example configs are in `registry/examples/`.

Key environment variables:
- `OXRL_DATA_DIR` — Override default data directory (default: `./data`)
- `OXRL_CHECKPOINT_DIR` — Override default checkpoint directory (default: `./checkpoints`)
- `HF_TOKEN` — HuggingFace token for gated models
- `GITHUB_TOKEN` — For autonomous bug reporting (optional)

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

### Reinforcement Learning (via Ray + vLLM rollouts)

| Algorithm | File | Description |
|-----------|------|-------------|
| **SGRPO** | `oxrl/algs/grpo.py` | Stable GRPO — Clipped surrogate loss with LoRA support and reference-free variants. |
| **CISPO** | `oxrl/algs/grpo.py` | Clipped importance-sampling policy optimization. |
| **PPO** | `oxrl/algs/ppo.py` | Proximal Policy Optimization with GAE, value clipping, and shared-backbone critic. |
| **RLHF** | `oxrl/algs/grpo.py` | Reinforcement Learning from Human Feedback — GRPO with a trained reward model. |
| **RLAIF** | `oxrl/algs/grpo.py` | Reinforcement Learning from AI Feedback — GRPO with AI-generated rewards. |

### Supervised Learning (via DeepSpeed)

| Algorithm | File | Description |
|-----------|------|-------------|
| **SFT** | `oxrl/algs/sft.py` | Supervised Fine-Tuning — Cross-entropy loss with masking and normalization. |
| **DPO** | `oxrl/algs/dpo.py` | Direct Preference Optimization — Pairwise preference learning with a reference model. |
| **ORPO** | `oxrl/algs/orpo.py` | Odds Ratio Preference Optimization — Reference-free preference alignment via log-odds. |
| **KTO** | `oxrl/algs/kto.py` | Kahneman-Tversky Optimization — Prospect-theory-inspired alignment with moving-average KL baseline. |
| **CPT** | `oxrl/algs/cpt.py` | Continued Pre-Training — Full-sequence language modeling on domain-specific text. |
| **KD** | `oxrl/algs/kd.py` | Knowledge Distillation — Teacher-student training with combined CE and KL divergence loss. |
| **RM** | `oxrl/algs/rm.py` | Reward Model Training — Bradley-Terry pairwise ranking with a learned scalar head. |
| **OnlineDPO** | `oxrl/algs/online_dpo.py` | Online DPO — DPO with on-the-fly rejection sampling in the data pipeline. |
| **RFT** | `oxrl/algs/rft.py` | Rejection Sampling Fine-Tuning — SFT on reward-filtered responses above a threshold. |
| **SPIN** | `oxrl/algs/spin.py` | Self-Play Improvement — DPO where rejected samples are the model's own prior outputs. |
| **IPO** | `oxrl/algs/ipo.py` | Identity Preference Optimization — Squared-loss variant of DPO for improved stability. |
| **SimPO** | `oxrl/algs/simpo.py` | Simple Preference Optimization — Reference-free, length-normalized preference alignment. |

## Project Structure

```
oxRL/
├── oxrl/                   # Core Framework Package
│   ├── trainer.py          # High-level Trainer API
│   ├── rewards/            # Verifiable reasoning and coding rewards (math, code, etc.)
│   ├── algs/               # 17 algorithm implementations (see tables above)
│   ├── swarm/              # Autonomous model onboarding (Scout, Bugfixer)
│   ├── preprocessing/      # Reasoning (OpenR1), Multimodal (Vision/Audio) preprocessors
│   ├── rollouts/           # vLLM inference with structured prompt support
│   └── datasets/           # Dataset loaders and samplers
├── main_rl.py              # RL training loop (Ray + DeepSpeed)
├── main_sl.py              # SL training loop (DeepSpeed) — 12 algorithms
├── registry/examples/      # Example configs for all 17 algorithms
├── examples/               # Ready-to-use recipes and training scripts
└── pyproject.toml          # Packaging and Installation
```

## Design Principles

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
- **Training Logic:** RL loop in `main_rl.py`, SL loop in `main_sl.py`.
- **Config Validation:** Logic is in `oxrl/configs/load.py`.

## Contributing

Contributions are welcome. Please follow the existing architectural patterns and style.

## FAQ

Check out the [FAQ](FAQ.md) for details on LoRA merging and Multimodal input formatting.
