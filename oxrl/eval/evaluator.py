"""
Core evaluation logic for NeurIPS 2026 experiments.

Wraps lm-evaluation-harness (lm_eval) to evaluate checkpoints on standard
benchmarks. Outputs structured JSON results for downstream aggregation.

Supported tasks:
    gsm8k  -- Grade-school math, 8-shot, exact match (accuracy)
    math   -- MATH competition problems, 4-shot, exact match
    mbpp   -- Mostly Basic Programming Problems, 3-shot, pass@1

Each task maps to a specific lm-evaluation-harness task name with
predetermined few-shot settings following community conventions.
"""
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────
# Task registry
# ──────────────────────────────────────────────────────────────────────

SUPPORTED_TASKS = {
    "gsm8k": {
        "lm_eval_task": "gsm8k",
        "num_fewshot": 8,
        "metric": "exact_match,strict-match",
        "description": "GSM8K grade-school math (8-shot, exact match)",
    },
    "gsm8k_cot": {
        "lm_eval_task": "gsm8k_cot",
        "num_fewshot": 8,
        "metric": "exact_match,strict-match",
        "description": "GSM8K with chain-of-thought (8-shot)",
    },
    "math": {
        "lm_eval_task": "minerva_math",
        "num_fewshot": 4,
        "metric": "exact_match",
        "description": "MATH competition problems via minerva_math (4-shot)",
    },
    "mbpp": {
        "lm_eval_task": "mbpp",
        "num_fewshot": 3,
        "metric": "pass@1",
        "description": "MBPP code generation (3-shot, pass@1)",
    },
    "humaneval": {
        "lm_eval_task": "humaneval",
        "num_fewshot": 0,
        "metric": "pass@1",
        "description": "HumanEval code generation (0-shot, pass@1)",
    },
    "arc_challenge": {
        "lm_eval_task": "arc_challenge",
        "num_fewshot": 25,
        "metric": "acc_norm",
        "description": "ARC-Challenge multiple-choice (25-shot, acc_norm)",
    },
    "hellaswag": {
        "lm_eval_task": "hellaswag",
        "num_fewshot": 10,
        "metric": "acc_norm",
        "description": "HellaSwag commonsense (10-shot, acc_norm)",
    },
    "winogrande": {
        "lm_eval_task": "winogrande",
        "num_fewshot": 5,
        "metric": "acc",
        "description": "WinoGrande pronoun resolution (5-shot, accuracy)",
    },
}


# ──────────────────────────────────────────────────────────────────────
# Core evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_checkpoint(
    model_path: str,
    tasks: List[str],
    output_dir: Optional[str] = None,
    batch_size: str = "auto",
    trust_remote_code: bool = True,
    device: str = "cuda",
    num_gpus: int = 1,
    limit: Optional[int] = None,
    seed: int = 42,
) -> Dict:
    """Evaluate a model checkpoint on one or more tasks.

    Args:
        model_path: HuggingFace model ID or local checkpoint path.
        tasks: List of task names (keys in SUPPORTED_TASKS).
        output_dir: Directory to write results JSON. If None, results are
                    returned but not written to disk.
        batch_size: Batch size for lm_eval. "auto" lets lm_eval decide.
        trust_remote_code: Trust remote code when loading model.
        device: Device string for lm_eval.
        num_gpus: Number of GPUs for tensor parallelism (vLLM backend).
        limit: Max number of examples per task (for debugging). None = all.
        seed: Random seed for reproducibility.

    Returns:
        Dict with structure:
        {
            "model": "...",
            "timestamp": "...",
            "tasks": {
                "gsm8k": {
                    "accuracy": 0.85,
                    "num_correct": 1020,
                    "num_total": 1200,
                    "metric_name": "exact_match,strict-match",
                    "num_fewshot": 8,
                    "stderr": 0.01,
                },
                ...
            },
            "elapsed_seconds": 123.4,
        }
    """
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    # Validate tasks
    unknown = [t for t in tasks if t not in SUPPORTED_TASKS]
    if unknown:
        raise ValueError(
            f"Unknown tasks: {unknown}. Supported: {list(SUPPORTED_TASKS.keys())}"
        )

    print(f"[eval] Model:  {model_path}")
    print(f"[eval] Tasks:  {tasks}")
    print(f"[eval] Device: {device}, GPUs: {num_gpus}")

    t0 = time.time()

    # Build task list and fewshot map for lm_eval
    lm_eval_tasks = []
    fewshot_map = {}
    for task_name in tasks:
        task_info = SUPPORTED_TASKS[task_name]
        lm_task = task_info["lm_eval_task"]
        lm_eval_tasks.append(lm_task)
        fewshot_map[lm_task] = task_info["num_fewshot"]

    # Use vLLM backend if multiple GPUs requested, otherwise HF
    if num_gpus > 1:
        model_args = (
            f"pretrained={model_path},"
            f"trust_remote_code={trust_remote_code},"
            f"tensor_parallel_size={num_gpus},"
            f"gpu_memory_utilization=0.85"
        )
        results = lm_eval.simple_evaluate(
            model="vllm",
            model_args=model_args,
            tasks=lm_eval_tasks,
            batch_size=batch_size,
            limit=limit,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
        )
    else:
        model_args = (
            f"pretrained={model_path},"
            f"trust_remote_code={trust_remote_code},"
            f"dtype=bfloat16"
        )
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=lm_eval_tasks,
            batch_size=batch_size,
            device=device,
            limit=limit,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
        )

    elapsed = time.time() - t0

    # Parse results into our structured format
    parsed = {
        "model": model_path,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(elapsed, 2),
        "seed": seed,
        "limit": limit,
        "tasks": {},
    }

    for task_name in tasks:
        task_info = SUPPORTED_TASKS[task_name]
        lm_task = task_info["lm_eval_task"]
        metric_key = task_info["metric"]

        if lm_task in results.get("results", {}):
            task_results = results["results"][lm_task]

            # lm_eval stores metrics with various key patterns
            accuracy = _extract_metric(task_results, metric_key)
            stderr = _extract_metric(task_results, metric_key + "_stderr", default=None)
            n_total = task_results.get("alias", task_results.get("samples", None))

            parsed["tasks"][task_name] = {
                "accuracy": accuracy,
                "metric_name": metric_key,
                "num_fewshot": task_info["num_fewshot"],
                "stderr": stderr,
                "raw": _sanitize_for_json(task_results),
            }
        else:
            print(f"[eval] WARNING: No results found for task '{lm_task}'. "
                  f"Available: {list(results.get('results', {}).keys())}")
            parsed["tasks"][task_name] = {
                "accuracy": None,
                "metric_name": metric_key,
                "num_fewshot": task_info["num_fewshot"],
                "error": f"Task '{lm_task}' not found in results",
            }

    # Write to disk if output_dir specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "eval_results.json")
        with open(output_path, "w") as f:
            json.dump(parsed, f, indent=2, default=str)
        print(f"[eval] Results written to {output_path}")

    # Print summary
    print(f"\n[eval] === Results (elapsed: {elapsed:.1f}s) ===")
    for task_name, task_res in parsed["tasks"].items():
        acc = task_res.get("accuracy")
        if acc is not None:
            stderr = task_res.get("stderr")
            stderr_str = f" +/- {stderr:.4f}" if stderr else ""
            print(f"  {task_name}: {acc:.4f}{stderr_str}")
        else:
            print(f"  {task_name}: ERROR - {task_res.get('error', 'unknown')}")

    return parsed


def _extract_metric(task_results: Dict, key: str, default=0.0):
    """Extract a metric from lm_eval results, handling various key formats."""
    # lm_eval can store metrics as "exact_match,strict-match" or just "acc"
    if key in task_results:
        val = task_results[key]
        return float(val) if val is not None else default

    # Try without commas (some versions use different separators)
    for k, v in task_results.items():
        if key.replace(",", "") in k.replace(",", ""):
            return float(v) if v is not None else default

    # Try common aliases
    aliases = {
        "exact_match": ["acc", "accuracy", "em"],
        "pass@1": ["pass_at_1", "pass_rate"],
    }
    base_key = key.split(",")[0] if "," in key else key
    for alias in aliases.get(base_key, []):
        if alias in task_results:
            return float(task_results[alias])

    return default


def _sanitize_for_json(obj):
    """Convert non-serializable values (e.g. numpy) to Python types."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj
