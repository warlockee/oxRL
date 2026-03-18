"""
oxrl.eval -- Evaluation harness for NeurIPS 2026 experiments.

Wraps lm-evaluation-harness (EleutherAI) for standardized evaluation of
post-trained checkpoints on GSM8K, MATH, and MBPP.

Modules:
    evaluator: Core evaluation logic. Loads a checkpoint, runs evaluation
               tasks, outputs structured JSON results.
    run_eval:  CLI entry point for batch evaluation of checkpoints.
"""
from oxrl.eval.evaluator import evaluate_checkpoint, SUPPORTED_TASKS
