# oxRL

**Post-train any model under 50 lines of code.**

A lightweight post-training framework for LLMs, VLMs, and VLAs. ~4,000 lines of Python. Scales to billions of parameters with DeepSpeed, vLLM, and Ray.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         oxRL Framework                          │
├─────────────────────┬───────────────────┬───────────────────────┤
│   Training Engines  │  Rollout Engines  │    Config + Data      │
│   (Ray + DeepSpeed) │  (Ray + vLLM)     │    (Pydantic + HF)   │
├─────────────────────┼───────────────────┼───────────────────────┤
│                     │                   │                       │
│  algs/grpo.py       │ rollouts/         │ configs/load.py       │
│    SGRPO loss       │   vllm_engine.py  │ configs/*.yaml        │
│    CISPO loss       │   replay_buffer.py│                       │
│  algs/PPO/ppo.py    │                   │ datasets/             │
│  algs/SFT/sft.py    │                   │   prompt_only.py      │
│                     │                   │   prompt_response.py  │
│                     │                   │   mixed_ratio_sampler  │
├─────────────────────┴───────────────────┴───────────────────────┤
│  utils/setup.py  │  utils/logging.py  │  rewards/compute_score  │
└──────────────────┴────────────────────┴─────────────────────────┘
```

## RL Training Workflow

```
┌──────────────┐     ┌───────────────────┐     ┌──────────────────┐
│ Load Config  │────▶│  Initialize Ray    │────▶│ Create Engines   │
│ (YAML file)  │     │  Cluster          │     │ N train + M roll │
└──────────────┘     └───────────────────┘     └────────┬─────────┘
                                                        │
                     ┌──────────────────────────────────┘
                     ▼
          ┌─────────────────────┐
          │   For each epoch:   │
          │                     │
          │  ┌───────────────┐  │    Rollout engines generate responses
          │  │ 1. Rollouts   │  │    using vLLM, compute rewards,
          │  │    (vLLM)     │  │    store in replay buffer
          │  └───────┬───────┘  │
          │          │          │
          │  ┌───────▼───────┐  │    Training engines run forward/backward
          │  │ 2. Train      │  │    with DeepSpeed ZeRO-3 across GPUs
          │  │    (DeepSpeed)│  │
          │  └───────┬───────┘  │
          │          │          │
          │  ┌───────▼───────┐  │    Save model, update rollout engines
          │  │ 3. Checkpoint │  │    with new policy weights
          │  │    + Refresh  │  │
          │  └───────────────┘  │
          └─────────────────────┘
```

## Quick Start

### Installation

```bash
git clone https://github.com/warlockee/oxRL.git
cd oxRL
pip install -r req.txt
```

Dependencies: PyTorch, DeepSpeed, vLLM, Ray, Transformers, Pydantic, MLflow.

### Post-train a model in 3 steps

**Step 1.** Prepare your data as a parquet or JSONL file with chat-format prompts:

```python
{"prompt": [{"role": "user", "content": "What is 2+2?"}], "answer": "4"}
```

**Step 2.** Write a minimal config (everything else uses sensible defaults):

```yaml
# config.yaml
run:
  experiment_id: "my-run"
  training_gpus: 2
  rollout_gpus: 2

train:
  alg_name: "sgrpo"           # or "cispo"
  total_number_of_epochs: 10
  train_steps_per_epoch: 20

model:
  name: "google/gemma-3-1b-it" # any HuggingFace model

data:
  train_dnames: ["my_data"]
  train_ratios: {"my_data": 1.0}
  train_files_path: "./data/train.parquet"
  val_files_path: "./data/val.parquet"
```

**Step 3.** Run:

```bash
python main_rl.py --config-file config.yaml
```

For supervised fine-tuning:

```bash
python main_sl.py --config-file configs/sl_args.yaml
```

See [`examples/quickstart.py`](examples/quickstart.py) for a complete runnable example.

### Custom Reward Functions

Write a function in `rewards/compute_score.py` and set `reward.reward_func` in your config:

```python
def my_reward(prompt_ids, response_ids, finish_reason):
    r = torch.zeros(len(response_ids), dtype=torch.float32)
    # your scoring logic here
    r[-1] = 1.0 if meets_criteria(response_ids) else 0.0
    return r, False  # (reward_tensor, is_per_token)
```

```yaml
reward:
  reward_func: "my_reward"
```

## Algorithms

| Algorithm | File | Description |
|-----------|------|-------------|
| **SGRPO** | `algs/grpo.py` | Stable GRPO — clipped surrogate loss with optional KL regularization from reference model |
| **CISPO** | `algs/grpo.py` | Clipped importance-sampling policy optimization — weighted log-probability loss |
| **PPO** | `algs/PPO/ppo.py` | Proximal Policy Optimization with GAE, value clipping, entropy bonus |
| **SFT** | `algs/SFT/sft.py` | Supervised fine-tuning with masked cross-entropy loss |

SGRPO and CISPO share the same training infrastructure (DeepSpeed + Ray actors) and differ only in the policy loss computation. Select with `train.alg_name` in your config.

## Project Structure

```
oxRL/
├── main_rl.py              RL training loop (Ray + DeepSpeed)
├── main_sl.py              SL training loop (DeepSpeed)
├── algs/
│   ├── grpo.py             SGRPO + CISPO (unified, loss_variant selects)
│   ├── PPO/ppo.py          PPO with GAE + value function
│   └── SFT/sft.py          Supervised fine-tuning
├── configs/
│   ├── load.py             Pydantic config with sensible defaults
│   ├── rl_args.yaml        Full RL config example
│   └── sl_args.yaml        Full SL config example
├── datasets/
│   ├── prompt_only.py      RL prompts (chat format → tokens)
│   ├── prompt_response.py  SL prompt-response pairs
│   └── mixed_ratio_sampler Multi-dataset weighted sampling
├── rollouts/
│   ├── vllm_engine.py      vLLM inference with hot model refresh
│   └── replay_buffer.py    On-policy sample storage
├── rewards/
│   └── compute_score.py    Pluggable reward functions
├── utils/
│   ├── setup.py            Distributed setup (seeds, rank, tokenizer)
│   ├── logging.py          Rank-aware logging + MLflow
│   └── utils.py            Tensor helpers (dtype, padding)
├── preprocessing/
│   └── gsm8k.py            GSM8K dataset preparation
└── examples/
    └── quickstart.py       End-to-end example (48 lines)
```

## Configuration

oxRL uses Pydantic for type-safe configuration. Every field has a sensible default — you only need to specify what's unique to your run.

**Required fields** (no defaults):

| Section | Field | Description |
|---------|-------|-------------|
| `run` | `experiment_id` | Name for this run |
| `run` | `training_gpus` | Number of GPUs for training |
| `run` | `rollout_gpus` | Number of GPUs for rollout generation |
| `train` | `alg_name` | Algorithm: `sgrpo`, `cispo`, `sft` |
| `train` | `total_number_of_epochs` | Training epochs |
| `train` | `train_steps_per_epoch` | Optimizer steps per epoch (RL) |
| `model` | `name` | HuggingFace model ID |
| `data` | `train_dnames` | Dataset name list |
| `data` | `train_ratios` | Dataset mixing ratios |
| `data` | `train_files_path` | Path to training data |
| `data` | `val_files_path` | Path to validation data |

Everything else (optimizer, scheduler, DeepSpeed ZeRO-3, vLLM rollouts, reward function) defaults to production-tested values. See [`configs/rl_args.yaml`](configs/rl_args.yaml) for the full reference.

## Key Design Decisions

**Sequential rollout → training.** oxRL does not pipeline rollout generation with training. This is deliberate. Pipelined overlap improves GPU utilization but makes debugging significantly harder. When training diverges at step 4,000, you want to know exactly what happened.

**One class for SGRPO and CISPO.** Both algorithms share 99% of their code — the only difference is 4 lines in the policy loss computation. A `loss_variant` parameter selects between them. No inheritance, no abstraction.

**DeepSpeed ZeRO-3 by default.** The config system auto-syncs optimizer, scheduler, dtype, and batch size settings to DeepSpeed — you configure once in the YAML and oxRL handles the rest.

**Strict on-policy enforcement.** Optional mode that validates rollouts were generated by the current policy version. Catches silent distribution shift bugs that waste GPU-days.

## Contributing

Contributions are welcome. The bar: keep changes readable, testable, and debuggable. Follow the existing style. If your change adds complexity, it should be worth it.

## FAQ

Check out the [FAQ](FAQ.md) for common questions and answers.

## Acknowledgments

Some components of this codebase are inspired by practices from open source projects. We try to cite sources wherever we directly reuse exact code. If we missed a citation, please let us know and we will credit the source.
