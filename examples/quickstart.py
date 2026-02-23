"""
oxRL Quickstart — Post-train any model under 50 lines of code.

This script prepares data, writes a minimal config, and prints run instructions.
Usage:
    python examples/quickstart.py
    python main_rl.py --config-file examples/quickstart.yaml
"""
import json, os, yaml

# --- 1. Prepare data (chat format, one prompt per line) ---
prompts = [
    {"prompt": [{"role": "user", "content": "What is 2 + 2?"}], "answer": "4"},
    {"prompt": [{"role": "user", "content": "What is the capital of France?"}], "answer": "Paris"},
    {"prompt": [{"role": "user", "content": "Write a haiku about the ocean."}], "answer": "—"},
    {"prompt": [{"role": "user", "content": "Explain gravity in one sentence."}], "answer": "—"},
]
os.makedirs("examples/data", exist_ok=True)
with open("examples/data/train.jsonl", "w") as f:
    for p in prompts:
        f.write(json.dumps(p) + "\n")

# --- 2. Minimal config (everything else uses sensible defaults) ---
config = {
    "run": {
        "experiment_id": "quickstart",
        "training_gpus": 2,
        "rollout_gpus": 2,
    },
    "train": {
        "alg_name": "sgrpo",
        "total_number_of_epochs": 3,
        "train_steps_per_epoch": 5,
    },
    "model": {
        "name": "google/gemma-3-1b-it",
    },
    "data": {
        "train_dnames": ["quickstart"],
        "train_ratios": {"quickstart": 1.0},
        "train_files_path": "examples/data/train.jsonl",
        "val_files_path": "examples/data/train.jsonl",
    },
}
with open("examples/quickstart.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False)

print("Ready. Run: python main_rl.py --config-file examples/quickstart.yaml")
