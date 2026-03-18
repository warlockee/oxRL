"""
CLI for evaluating post-trained checkpoints on standard benchmarks.

Usage:
    # Evaluate a single checkpoint on GSM8K and MATH:
    python -m oxrl.eval.run_eval \
        --checkpoint Qwen/Qwen2.5-1.5B-Instruct \
        --tasks gsm8k,math \
        --output-dir /home/ec2-user/fsx/oxrl_results/eval/qwen1.5b_base

    # Evaluate with multiple GPUs:
    python -m oxrl.eval.run_eval \
        --checkpoint /path/to/checkpoint \
        --tasks gsm8k,math,mbpp \
        --num-gpus 2 \
        --output-dir /home/ec2-user/fsx/oxrl_results/eval/run_001

    # Quick test with limited examples:
    python -m oxrl.eval.run_eval \
        --checkpoint Qwen/Qwen2.5-0.5B-Instruct \
        --tasks gsm8k \
        --limit 50 \
        --output-dir /tmp/test_eval

    # Batch evaluation of multiple checkpoints:
    python -m oxrl.eval.run_eval \
        --checkpoint-dir /home/ec2-user/fsx/checkpoints/exp001 \
        --tasks gsm8k,math \
        --output-dir /home/ec2-user/fsx/oxrl_results/eval/exp001
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

from oxrl.eval.evaluator import evaluate_checkpoint, SUPPORTED_TASKS


def find_checkpoints(checkpoint_dir: str) -> List[str]:
    """Find all valid model checkpoint directories under a root dir.

    A valid checkpoint directory contains either:
    - config.json (HuggingFace format)
    - model.safetensors or pytorch_model.bin
    """
    checkpoints = []
    root = Path(checkpoint_dir)

    if not root.exists():
        print(f"[run_eval] WARNING: {checkpoint_dir} does not exist")
        return checkpoints

    # Check if the root itself is a checkpoint
    if (root / "config.json").exists():
        checkpoints.append(str(root))
        return checkpoints

    # Search subdirectories (one level deep)
    for subdir in sorted(root.iterdir()):
        if subdir.is_dir() and (subdir / "config.json").exists():
            checkpoints.append(str(subdir))

    # Search two levels deep (common pattern: experiment_id/iter_tag/...)
    if not checkpoints:
        for subdir in sorted(root.iterdir()):
            if subdir.is_dir():
                for sub2 in sorted(subdir.iterdir()):
                    if sub2.is_dir() and (sub2 / "config.json").exists():
                        checkpoints.append(str(sub2))

    return checkpoints


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate post-trained checkpoints on standard benchmarks."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--checkpoint", type=str, default=None,
        help="Single checkpoint path or HuggingFace model ID.",
    )
    group.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Directory containing multiple checkpoints to evaluate.",
    )
    parser.add_argument(
        "--tasks", type=str, required=True,
        help=f"Comma-separated task names. Supported: {','.join(SUPPORTED_TASKS.keys())}",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to write evaluation results.",
    )
    parser.add_argument(
        "--batch-size", type=str, default="auto",
        help="Batch size for lm_eval ('auto' or integer).",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1,
        help="Number of GPUs for evaluation (uses vLLM backend if > 1).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max examples per task (for debugging).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true", default=True,
        help="Trust remote code when loading models.",
    )
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]

    # Validate tasks
    for t in tasks:
        if t not in SUPPORTED_TASKS:
            print(f"[run_eval] ERROR: Unknown task '{t}'. "
                  f"Supported: {list(SUPPORTED_TASKS.keys())}")
            sys.exit(1)

    # Collect checkpoints to evaluate
    if args.checkpoint:
        checkpoints = [args.checkpoint]
    else:
        checkpoints = find_checkpoints(args.checkpoint_dir)
        if not checkpoints:
            print(f"[run_eval] ERROR: No checkpoints found in {args.checkpoint_dir}")
            sys.exit(1)
        print(f"[run_eval] Found {len(checkpoints)} checkpoints:")
        for cp in checkpoints:
            print(f"  - {cp}")

    # Evaluate each checkpoint
    all_results = []
    for i, checkpoint in enumerate(checkpoints):
        print(f"\n{'='*60}")
        print(f"[run_eval] Evaluating checkpoint {i+1}/{len(checkpoints)}: {checkpoint}")
        print(f"{'='*60}")

        # Create per-checkpoint output directory
        if len(checkpoints) > 1:
            cp_name = Path(checkpoint).name
            cp_output_dir = os.path.join(args.output_dir, cp_name)
        else:
            cp_output_dir = args.output_dir

        try:
            result = evaluate_checkpoint(
                model_path=checkpoint,
                tasks=tasks,
                output_dir=cp_output_dir,
                batch_size=args.batch_size,
                trust_remote_code=args.trust_remote_code,
                num_gpus=args.num_gpus,
                limit=args.limit,
                seed=args.seed,
            )
            all_results.append(result)
        except Exception as e:
            print(f"[run_eval] ERROR evaluating {checkpoint}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "model": checkpoint,
                "error": str(e),
                "tasks": {},
            })

    # Write combined results if evaluating multiple checkpoints
    if len(checkpoints) > 1:
        combined_path = os.path.join(args.output_dir, "all_results.json")
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n[run_eval] Combined results written to {combined_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("[run_eval] Summary")
    print(f"{'='*60}")
    header = f"{'Model':<50s}" + "".join(f"  {t:>10s}" for t in tasks)
    print(header)
    print("-" * len(header))
    for result in all_results:
        model_name = Path(result["model"]).name[:48]
        row = f"{model_name:<50s}"
        for t in tasks:
            if t in result.get("tasks", {}):
                acc = result["tasks"][t].get("accuracy")
                if acc is not None:
                    row += f"  {acc:>10.4f}"
                else:
                    row += f"  {'ERROR':>10s}"
            else:
                row += f"  {'N/A':>10s}"
        print(row)


if __name__ == "__main__":
    main()
