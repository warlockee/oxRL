"""
oxRL Quickstart — Post-train any model under 10 lines of code.

Usage:
    python examples/quickstart.py
"""
from oxrl import Trainer

# 1. Initialize Trainer with a small model
# Hardware is auto-detected, but we can be explicit if needed.
trainer = Trainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    experiment_id="quickstart_gsm8k"
)

# 2. Run Training on GSM8K
# This will auto-download, preprocess, and start the RL loop.
trainer.train(
    dataset="gsm8k",
    epochs=1,
    steps_per_epoch=5
)
