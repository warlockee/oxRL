# oxRL - Agent Guide

## Quick Start

```bash
# One-command training
oxrl train --model Qwen/Qwen2.5-0.5B-Instruct --task math --epochs 1 --steps 10

# With a custom dataset (JSONL or Parquet)
oxrl train --model Qwen/Qwen2.5-7B-Instruct --dataset ./data.jsonl --task reasoning
```

```python
# Programmatic API
from oxrl import Trainer
trainer = Trainer(model="Qwen/Qwen2.5-0.5B-Instruct")
trainer.train(task="math", dataset="./my_data.jsonl", epochs=1, steps_per_epoch=10)
```

## Key Files

| Path | Purpose |
|------|---------|
| `oxrl/trainer.py` | High-level Trainer API — start here |
| `oxrl/cli.py` | CLI entry point (`oxrl train`, `oxrl doctor`) |
| `oxrl/algs/` | 51 RL/post-training algorithms (GRPO, DPO, PPO, etc.) |
| `oxrl/swarm/config_generator.py` | Auto-generates training configs from (model, task) |
| `oxrl/datasets/` | Dataset loaders (prompt_only, prompt_response, prompt_preference) |
| `oxrl/rewards/` | Reward functions (math, code, LLM-judge, custom) |
| `oxrl/configs/` | YAML config templates |
| `oxrl/submission/` | NeurIPS submission compliance toolkit |
| `examples/` | Example scripts and data |

## Tasks

Supported task types: `math`, `reasoning`, `code`, `instruct`, `vision`, `audio`.
Each maps to a default dataset and reward function via `config_generator.py`.

## Dataset Format

Datasets can be `.parquet` or `.jsonl`. Each row needs:
- **Prompt-only** (RL): `{"prompt": [{"role": "user", "content": "..."}], "answer": "..."}`
- **Prompt-response** (SFT): `{"prompt": [...], "answer": "the response"}`
- **Preference** (DPO): `{"prompt": [...], "chosen": "good", "rejected": "bad"}`

## Config Generation

```python
from oxrl.swarm.config_generator import generate_config, save_config
config = generate_config(model_name="Qwen/Qwen2.5-7B-Instruct", task="math", param_count_b=7.0)
save_config(config, "config.yaml")
```

## Running Tests

```bash
python -m pytest tests/ -q
```

## NeurIPS Submission

```bash
# Full pre-submission audit (paper + README + Croissant)
oxrl submit audit --paper docs/oxrl_formal.tex --readme README.md --croissant neurips_croissant/croissant.json

# Check paper compliance only
oxrl submit check --paper docs/oxrl_formal.tex

# Generate Croissant metadata with RAI fields
oxrl submit croissant --dataset user/dataset --output croissant.json --hf-token $HF_TOKEN

# Check README against Papers With Code checklist
oxrl submit readme --readme README.md
```

The `neurips-submission` agent can also be invoked to handle the full workflow interactively.

## Common Overrides

Pass kwargs to `trainer.train()` or edit the generated config YAML:
- `lr`: learning rate
- `lora_enabled`: use LoRA (auto-enabled for models >= 3B)
- `total_number_of_epochs`: training epochs
- `train_steps_per_epoch`: optimizer steps per epoch
