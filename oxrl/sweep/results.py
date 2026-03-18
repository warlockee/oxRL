"""
Results aggregation for NeurIPS 2026 experiments.

Collects per-run metrics and evaluation results from the filesystem,
aggregates them into a single CSV, and generates summary tables (LaTeX
and plain text) for the paper.

Usage:
    # Aggregate all results into CSV:
    python -m oxrl.sweep.results \
        --results-dir /home/ec2-user/fsx/oxrl_results/neurips2026 \
        --eval-dir /home/ec2-user/fsx/oxrl_results/eval \
        --output /home/ec2-user/fsx/oxrl_results/neurips2026/summary.csv

    # Generate LaTeX tables:
    python -m oxrl.sweep.results \
        --results-dir /home/ec2-user/fsx/oxrl_results/neurips2026 \
        --eval-dir /home/ec2-user/fsx/oxrl_results/eval \
        --output /home/ec2-user/fsx/oxrl_results/neurips2026/summary.csv \
        --latex /home/ec2-user/fsx/oxrl_results/neurips2026/tables.tex
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Result collection
# ──────────────────────────────────────────────────────────────────────

def collect_training_results(results_dir: str) -> List[Dict]:
    """Collect training metrics from per-run metrics.json files.

    Expected structure:
        results_dir/
            experiment_id_1/
                metrics.json
            experiment_id_2/
                metrics.json
    """
    records = []
    root = Path(results_dir)
    if not root.exists():
        print(f"[results] WARNING: results_dir does not exist: {results_dir}")
        return records

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue

        metrics_path = subdir / "metrics.json"
        if not metrics_path.exists():
            continue

        try:
            with open(metrics_path) as f:
                metrics = json.load(f)

            # Parse experiment_id to extract metadata
            exp_id = subdir.name
            parsed = parse_experiment_id(exp_id)

            record = {
                "experiment_id": exp_id,
                "status": metrics.get("status", "unknown"),
                "elapsed_seconds": metrics.get("elapsed_seconds", 0),
                **parsed,
            }

            # Include any training metrics
            for key in ("avg_loss", "avg_reward", "avg_kl_old", "avg_kl_ref",
                        "avg_clipfrac", "avg_response_len"):
                if key in metrics:
                    record[key] = metrics[key]

            records.append(record)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"[results] WARNING: Failed to parse {metrics_path}: {e}")

    return records


def collect_eval_results(eval_dir: str) -> List[Dict]:
    """Collect evaluation results from per-checkpoint eval JSON files.

    Expected structure:
        eval_dir/
            experiment_id_1/
                eval_results.json
            experiment_id_2/
                eval_results.json
    """
    records = []
    root = Path(eval_dir)
    if not root.exists():
        print(f"[results] WARNING: eval_dir does not exist: {eval_dir}")
        return records

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue

        eval_path = subdir / "eval_results.json"
        if not eval_path.exists():
            continue

        try:
            with open(eval_path) as f:
                eval_data = json.load(f)

            exp_id = subdir.name
            parsed = parse_experiment_id(exp_id)

            for task_name, task_results in eval_data.get("tasks", {}).items():
                record = {
                    "experiment_id": exp_id,
                    "eval_task": task_name,
                    "accuracy": task_results.get("accuracy"),
                    "stderr": task_results.get("stderr"),
                    "model_path": eval_data.get("model", ""),
                    **parsed,
                }
                records.append(record)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"[results] WARNING: Failed to parse {eval_path}: {e}")

    return records


def parse_experiment_id(exp_id: str) -> Dict:
    """Parse experiment metadata from the experiment_id string.

    Expected format: {experiment_name}_{algorithm}_{model_key}_{task}_{seed}[_overrides]
    Examples:
        core_math_gsm8k_sgrpo_qwen1.5b_gsm8k_s42
        dpo_variants_betadpo_qwen15b_gsm8k_s123
    """
    result = {
        "experiment_name": "",
        "algorithm": "",
        "model_key": "",
        "task": "",
        "seed": 0,
    }

    # Try to extract seed
    seed_match = re.search(r"_s(\d+)", exp_id)
    if seed_match:
        result["seed"] = int(seed_match.group(1))

    # Extract algorithm (look for known algorithm names)
    known_algs = [
        "sgrpo", "gspo", "cispo", "ppo", "dpo", "simpo", "kto", "ipo",
        "sft", "orpo", "cpo", "alphapo", "rdpo", "cdpo", "betadpo",
        "caldpo", "sppo", "apo", "hinge", "robust_dpo", "exo", "odpo",
        "dpop", "focalpo", "gpo", "wpo", "fdpo", "hdpo", "dposhift",
        "cposimpo", "sampo", "drdpo", "chipo", "spo", "dpnll",
        "minor_dpo", "c2dpo", "alpha_dpo", "bpo", "nca", "bco",
        "discopop",
    ]
    for alg in sorted(known_algs, key=len, reverse=True):
        if f"_{alg}_" in exp_id or exp_id.endswith(f"_{alg}"):
            result["algorithm"] = alg
            break

    # Extract model key
    model_patterns = [
        ("qwen05b", "qwen05b"), ("qwen15b", "qwen1.5b"),
        ("qwen1.5b", "qwen1.5b"), ("qwen3b", "qwen3b"),
        ("qwen7b", "qwen7b"), ("coder15b", "coder1.5b"),
        ("coder1.5b", "coder1.5b"), ("coder7b", "coder7b"),
    ]
    for pattern, model_key in model_patterns:
        if pattern in exp_id.lower():
            result["model_key"] = model_key
            break

    # Extract task
    for task in ["gsm8k", "math", "mbpp"]:
        if task in exp_id:
            result["task"] = task
            break

    # Extract experiment name (everything before the algorithm)
    if result["algorithm"]:
        idx = exp_id.find(f"_{result['algorithm']}_")
        if idx > 0:
            result["experiment_name"] = exp_id[:idx]

    return result


# ──────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────

def merge_results(
    training_results: List[Dict],
    eval_results: List[Dict],
) -> List[Dict]:
    """Merge training metrics with evaluation results by experiment_id."""
    # Index training results
    training_index = {r["experiment_id"]: r for r in training_results}

    # Merge eval results with training info
    merged = []
    seen_exp_ids = set()

    for eval_rec in eval_results:
        exp_id = eval_rec["experiment_id"]
        train_rec = training_index.get(exp_id, {})

        row = {**eval_rec}
        row["elapsed_seconds"] = train_rec.get("elapsed_seconds", 0)
        row["gpu_hours"] = round(row["elapsed_seconds"] / 3600, 4)
        row["status"] = train_rec.get("status", "unknown")
        merged.append(row)
        seen_exp_ids.add(exp_id)

    # Add training-only results (no eval yet)
    for exp_id, train_rec in training_index.items():
        if exp_id not in seen_exp_ids:
            merged.append({
                **train_rec,
                "eval_task": "",
                "accuracy": None,
                "stderr": None,
                "gpu_hours": round(train_rec.get("elapsed_seconds", 0) / 3600, 4),
            })

    return merged


def write_csv(records: List[Dict], output_path: str):
    """Write records to CSV."""
    if not records:
        print("[results] No records to write.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Determine columns
    columns = [
        "experiment_id", "experiment_name", "algorithm", "model_key",
        "task", "eval_task", "seed", "accuracy", "stderr",
        "gpu_hours", "elapsed_seconds", "status",
    ]
    # Add any extra columns found in the data
    all_keys = set()
    for r in records:
        all_keys.update(r.keys())
    extra = sorted(all_keys - set(columns))
    columns.extend(extra)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    print(f"[results] Wrote {len(records)} rows to {output_path}")


# ──────────────────────────────────────────────────────────────────────
# Summary tables
# ──────────────────────────────────────────────────────────────────────

def compute_summary(records: List[Dict]) -> Dict:
    """Compute mean +/- std accuracy per (algorithm, model_key, task).

    Returns nested dict: summary[task][model_key][algorithm] = (mean, std, n)
    """
    groups = defaultdict(list)
    for r in records:
        if r.get("accuracy") is None:
            continue
        key = (r.get("task", ""), r.get("model_key", ""), r.get("algorithm", ""))
        groups[key].append(float(r["accuracy"]))

    summary = defaultdict(lambda: defaultdict(dict))
    for (task, model_key, alg), accs in groups.items():
        mean = np.mean(accs)
        std = np.std(accs, ddof=1) if len(accs) > 1 else 0.0
        summary[task][model_key][alg] = (mean, std, len(accs))

    return dict(summary)


def print_summary_table(summary: Dict):
    """Print a plain-text summary table."""
    for task, models in sorted(summary.items()):
        print(f"\n{'='*80}")
        print(f"Task: {task}")
        print(f"{'='*80}")

        # Collect all algorithms
        all_algs = set()
        for model_key, algs in models.items():
            all_algs.update(algs.keys())
        all_algs = sorted(all_algs)

        # Header
        header = f"{'Model':<15s}" + "".join(f"  {a:>12s}" for a in all_algs)
        print(header)
        print("-" * len(header))

        # Rows
        model_order = ["qwen05b", "qwen1.5b", "qwen3b", "qwen7b",
                       "coder1.5b", "coder7b"]
        for model_key in model_order:
            if model_key not in models:
                continue
            algs = models[model_key]
            row = f"{model_key:<15s}"
            for alg in all_algs:
                if alg in algs:
                    mean, std, n = algs[alg]
                    row += f"  {mean:>5.1f}+/-{std:>4.1f}"
                else:
                    row += f"  {'':>12s}"
            print(row)


def generate_latex_table(
    summary: Dict,
    output_path: str,
    caption: str = "Algorithm comparison across model scales.",
    label: str = "tab:main_results",
):
    """Generate a LaTeX table from the summary.

    Produces one table per task, with models as rows and algorithms as columns.
    Bold-faces the best result per model.
    """
    lines = []
    lines.append("% Auto-generated by oxrl.sweep.results")
    lines.append("")

    for task, models in sorted(summary.items()):
        all_algs = set()
        for model_key, algs in models.items():
            all_algs.update(algs.keys())
        all_algs = sorted(all_algs)
        n_cols = len(all_algs)

        lines.append(f"\\begin{{table}}[t]")
        lines.append(f"\\centering")
        lines.append(f"\\caption{{{caption} Task: {task}.}}")
        lines.append(f"\\label{{{label}_{task}}}")
        lines.append(f"\\resizebox{{\\textwidth}}{{!}}{{")
        lines.append(f"\\begin{{tabular}}{{l{'c' * n_cols}}}")
        lines.append(f"\\toprule")

        # Header
        header = "Model & " + " & ".join(a.upper() for a in all_algs) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")

        # Rows
        model_order = ["qwen05b", "qwen1.5b", "qwen3b", "qwen7b",
                       "coder1.5b", "coder7b"]
        for model_key in model_order:
            if model_key not in models:
                continue
            algs = models[model_key]

            # Find best algorithm for this model
            best_mean = -1.0
            for alg in all_algs:
                if alg in algs:
                    mean, _, _ = algs[alg]
                    if mean > best_mean:
                        best_mean = mean

            cells = [model_key.replace("_", "\\_")]
            for alg in all_algs:
                if alg in algs:
                    mean, std, n = algs[alg]
                    val = f"{mean*100:.1f}"
                    if n > 1:
                        val += f"$_{{{std*100:.1f}}}$"
                    if abs(mean - best_mean) < 1e-4:
                        val = f"\\textbf{{{val}}}"
                    cells.append(val)
                else:
                    cells.append("--")

            lines.append(" & ".join(cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("}")
        lines.append("\\end{table}")
        lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[results] LaTeX tables written to {output_path}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate NeurIPS 2026 experiment results."
    )
    parser.add_argument(
        "--results-dir", type=str,
        default="/home/ec2-user/fsx/oxrl_results/neurips2026",
        help="Directory containing per-run metrics.json files.",
    )
    parser.add_argument(
        "--eval-dir", type=str,
        default="/home/ec2-user/fsx/oxrl_results/eval",
        help="Directory containing per-checkpoint eval_results.json files.",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--latex", type=str, default=None,
        help="Output LaTeX tables path (optional).",
    )
    args = parser.parse_args()

    # Collect results
    print("[results] Collecting training results...")
    training = collect_training_results(args.results_dir)
    print(f"  Found {len(training)} training records")

    print("[results] Collecting evaluation results...")
    evals = collect_eval_results(args.eval_dir)
    print(f"  Found {len(evals)} evaluation records")

    # Merge
    merged = merge_results(training, evals)
    print(f"[results] Merged: {len(merged)} total records")

    # Write CSV
    write_csv(merged, args.output)

    # Compute and print summary
    summary = compute_summary(merged)
    if summary:
        print_summary_table(summary)

        if args.latex:
            generate_latex_table(summary, args.latex)
    else:
        print("[results] No accuracy data available for summary table.")


if __name__ == "__main__":
    main()
