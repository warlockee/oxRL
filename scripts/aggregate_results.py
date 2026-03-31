#!/usr/bin/env python3
"""
Aggregate all evaluation results into structured tables for the NeurIPS paper.

Reads eval JSON files, computes means/stds across seeds, runs statistical tests,
and outputs:
  1. LaTeX tables (ready for copy-paste into the paper)
  2. CSV summary (for plotting)
  3. JSON dump (for figure generation)

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --eval-dir /path/to/eval --output-dir /path/to/output
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

EVAL_DIR = "/home/ec2-user/fsx/oxrl_results/eval"
OUTPUT_DIR = "/home/ec2-user/fsx/oxrl_results/aggregated"

# Canonical algorithm display names and ordering
ALG_DISPLAY = {
    "sft": "SFT",
    "dpo": "DPO",
    "simpo": "SimPO",
    "ipo": "IPO",
    "kto": "KTO",
    "hinge": "Hinge",
    "orpo": "ORPO",
    "cpo": "CPO",
    "sgrpo": "SGRPO",
    "gspo": "GSPO",
    "cispo": "CISPO",
    "ppo": "PPO",
    # DPO variants
    "rdpo": "RDPO",
    "cdpo": "CDPO",
    "betadpo": "BetaDPO",
    "caldpo": "CalDPO",
    "dpop": "DPOP",
    "odpo": "ODPO",
    "exo": "EXO",
    "alphapo": "AlphaPO",
    "apo": "APO",
    "sppo": "SPPO",
    "robust_dpo": "RobustDPO",
    "gpo": "GPO",
    "focalpo": "FocalPO",
}

ALG_ORDER = [
    "sft",
    "dpo", "simpo", "ipo", "kto", "hinge", "orpo", "cpo",
    "sgrpo", "gspo", "cispo", "ppo",
]

SCALE_DISPLAY = {
    "qwen05b": "0.5B",
    "qwen15b": "1.5B",
    "qwen3b": "3B",
    "qwen7b": "7B",
}

SCALE_ORDER = ["qwen05b", "qwen15b", "qwen3b", "qwen7b"]

TASK_METRIC = {
    "gsm8k": "exact_match,strict-match",
    "math": "exact_match",
    "mbpp": "pass@1",
}

# ──────────────────────────────────────────────────────────────────────
# Result parsing
# ──────────────────────────────────────────────────────────────────────


def parse_experiment_id(exp_id: str) -> Optional[Dict]:
    """Parse experiment ID into components.

    Handles both naming conventions:
      - Old: {alg}_05b_s{seed}
      - New: core_math_gsm8k_{alg}_{scale}_{task}_s{seed}
      - DPO variants: dpo_variants_{variant}_{scale}_{task}_s{seed}
    """
    # New naming: core_math_gsm8k_ALG_SCALE_TASK_sSEED
    m = re.match(
        r"core_math_(gsm8k|math_hard|mbpp)_(\w+?)_(qwen\d+b)_(gsm8k|math_hard|mbpp)_s(\d+)$",
        exp_id,
    )
    if m:
        return {
            "experiment": f"core_math_{m.group(1)}",
            "algorithm": m.group(2),
            "scale": m.group(3),
            "task": m.group(4),
            "seed": int(m.group(5)),
        }

    # DPO variants: dpo_variants_VARIANT_SCALE_TASK_sSEED
    m = re.match(
        r"dpo_variants_(\w+?)_(qwen\d+b)_(gsm8k|math_hard|mbpp)_s(\d+)$",
        exp_id,
    )
    if m:
        return {
            "experiment": "dpo_variants",
            "algorithm": m.group(1),
            "scale": m.group(2),
            "task": m.group(3),
            "seed": int(m.group(4)),
        }

    # Ablations: ablations_TYPE_ALG_SCALE_TASK_sSEED
    m = re.match(
        r"ablations_(\w+?)_(\w+?)_(qwen\d+b)_(gsm8k|math_hard|mbpp)_s(\d+)$",
        exp_id,
    )
    if m:
        return {
            "experiment": f"ablations_{m.group(1)}",
            "algorithm": m.group(2),
            "scale": m.group(3),
            "task": m.group(4),
            "seed": int(m.group(5)),
        }

    # Old naming: ALG_05b_sSEED
    m = re.match(r"(\w+?)_(\d+)b_s(\d+)$", exp_id)
    if m:
        alg = m.group(1)
        size = m.group(2)
        seed = int(m.group(3))
        scale_map = {"05": "qwen05b", "15": "qwen15b", "3": "qwen3b", "7": "qwen7b"}
        scale = scale_map.get(size)
        if scale:
            return {
                "experiment": "core_math_gsm8k",
                "algorithm": alg,
                "scale": scale,
                "task": "gsm8k",
                "seed": seed,
            }

    return None


def load_eval_result(eval_dir_path: str) -> Optional[Dict]:
    """Load evaluation results from an experiment eval directory.

    Returns dict with task -> accuracy mapping.
    """
    results_files = glob.glob(
        os.path.join(eval_dir_path, "**", "results_*.json"), recursive=True
    )
    if not results_files:
        # Also check for eval_results.json (our evaluator format)
        eval_file = os.path.join(eval_dir_path, "eval_results.json")
        if os.path.exists(eval_file):
            results_files = [eval_file]
        else:
            return None

    results_files.sort()
    latest = results_files[-1]

    try:
        with open(latest) as f:
            data = json.load(f)
    except Exception:
        return None

    parsed = {}

    # lm_eval raw format
    if "results" in data:
        for task_key, metrics in data["results"].items():
            # Map lm_eval task names to our short names
            task_map = {
                "gsm8k_cot_zeroshot": "gsm8k",
                "gsm8k_cot": "gsm8k",
                "gsm8k": "gsm8k",
                "minerva_math": "math",
                "math": "math",
                "mbpp": "mbpp",
            }
            task_name = task_map.get(task_key)
            if not task_name:
                continue

            metric_key = TASK_METRIC.get(task_name, "exact_match,strict-match")
            acc = None

            # Try exact key
            if metric_key in metrics:
                acc = metrics[metric_key]
            else:
                # Try partial matches
                for k, v in metrics.items():
                    if "exact_match" in k and "stderr" not in k:
                        if acc is None or "strict" in k:
                            acc = v
                    elif "pass" in k.lower() and "stderr" not in k:
                        acc = v

            if acc is not None:
                parsed[task_name] = float(acc)

    # Our evaluator format
    elif "tasks" in data:
        for task_name, task_data in data["tasks"].items():
            if "accuracy" in task_data and task_data["accuracy"] is not None:
                parsed[task_name] = float(task_data["accuracy"])

    return parsed if parsed else None


def load_base_model_results(eval_dir: str) -> Dict:
    """Load base model evaluation results."""
    base_dir = os.path.join(eval_dir, "base_models")
    results = {}

    if not os.path.isdir(base_dir):
        return results

    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        # Parse scale from directory name (e.g., "qwen05b_gsm8k")
        for scale in SCALE_ORDER:
            if scale in subdir:
                eval_result = load_eval_result(subdir_path)
                if eval_result:
                    for task, acc in eval_result.items():
                        results[(scale, task)] = acc
                break

    return results


# ──────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────


def aggregate_results(eval_dir: str) -> Tuple[Dict, Dict]:
    """
    Aggregate all results, grouping by (algorithm, scale, task) and
    collecting accuracy across seeds.

    Returns:
        results: {(alg, scale, task): [acc1, acc2, ...]}
        base_results: {(scale, task): acc}
    """
    results = defaultdict(list)
    seen_seeds = defaultdict(set)  # Track (alg, scale, task) -> set of seeds seen
    skipped = []

    for exp_id in sorted(os.listdir(eval_dir)):
        exp_path = os.path.join(eval_dir, exp_id)
        if not os.path.isdir(exp_path) or exp_id == "base_models":
            continue

        parsed_id = parse_experiment_id(exp_id)
        if parsed_id is None:
            skipped.append(exp_id)
            continue

        eval_result = load_eval_result(exp_path)
        if eval_result is None:
            continue

        alg = parsed_id["algorithm"]
        scale = parsed_id["scale"]
        seed = parsed_id["seed"]

        for task, acc in eval_result.items():
            key = (alg, scale, task)
            # Avoid duplicates from old+new naming for same seed
            # but keep identical values from different seeds
            if seed not in seen_seeds[key]:
                seen_seeds[key].add(seed)
                results[key].append(acc)

    base_results = load_base_model_results(eval_dir)

    if skipped:
        print(f"  Skipped {len(skipped)} unrecognized experiment IDs:")
        for s in skipped[:10]:
            print(f"    {s}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped)-10} more")

    return dict(results), base_results


# ──────────────────────────────────────────────────────────────────────
# LaTeX table generation
# ──────────────────────────────────────────────────────────────────────


def format_acc(accs: List[float], bold: bool = False) -> str:
    """Format accuracy for LaTeX: mean ± std if multiple seeds, else just the number."""
    if not accs:
        return r"\placeholder{--}"

    mean = np.mean(accs) * 100
    if len(accs) > 1:
        std = np.std(accs, ddof=1) * 100
        text = f"{mean:.2f}" + r"$_{\pm" + f"{std:.1f}" + r"}$"
    else:
        text = f"{mean:.2f}"

    if bold:
        return r"\textbf{" + text + "}"
    return text


def generate_main_table(
    results: Dict, base_results: Dict, task: str = "gsm8k"
) -> str:
    """Generate the main results table (Algorithm x Scale) for a given task."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{GSM8K exact-match accuracy (\%) across model scales and "
        r"post-training algorithms. Mean $\pm$ std over available seeds. "
        r"Best result per scale in \textbf{bold}.}"
    )
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}lcccc@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Algorithm} & \textbf{0.5B} & \textbf{1.5B} & \textbf{3B} & \textbf{7B} \\"
    )
    lines.append(r"\midrule")

    # Base model row
    base_row = "Base Model"
    for scale in SCALE_ORDER:
        acc = base_results.get((scale, task))
        if acc is not None:
            base_row += f" & {acc*100:.2f}"
        else:
            base_row += r" & \placeholder{--}"
    base_row += r" \\"
    lines.append(base_row)
    lines.append(r"\midrule")

    # Find best accuracy per scale (for bolding)
    best_per_scale = {}
    for scale in SCALE_ORDER:
        best = 0
        for alg in ALG_ORDER:
            accs = results.get((alg, scale, task), [])
            if accs:
                mean = np.mean(accs)
                if mean > best:
                    best = mean
        best_per_scale[scale] = best

    # Algorithm rows, grouped by type
    offline_algs = ["sft"]
    preference_algs = ["dpo", "simpo", "ipo", "kto", "hinge", "orpo"]
    online_algs = ["sgrpo", "gspo", "ppo"]

    def add_alg_rows(alg_list, add_midrule_before=False, add_midrule_after=False):
        if add_midrule_before:
            lines.append(r"\midrule")
        for alg in alg_list:
            display = ALG_DISPLAY.get(alg, alg.upper())
            row = display
            for scale in SCALE_ORDER:
                accs = results.get((alg, scale, task), [])
                if accs:
                    mean = np.mean(accs)
                    is_best = abs(mean - best_per_scale.get(scale, 0)) < 1e-6
                    row += " & " + format_acc(accs, bold=is_best)
                else:
                    row += r" & \placeholder{--}"
            row += r" \\"
            lines.append(row)
        if add_midrule_after:
            lines.append(r"\midrule")

    add_alg_rows(offline_algs, add_midrule_after=True)
    add_alg_rows(preference_algs, add_midrule_after=True)
    add_alg_rows(online_algs)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_dpo_variant_table(results: Dict, task: str = "gsm8k") -> str:
    """Generate the DPO variant comparison table with statistical tests."""
    # Collect all DPO variant algorithms
    variant_algs = set()
    for (alg, scale, t), accs in results.items():
        if t == task and scale == "qwen15b":
            variant_algs.add(alg)

    # Get vanilla DPO as baseline
    dpo_accs = results.get(("dpo", "qwen15b", task), [])

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{DPO variant comparison at 1.5B scale on GSM8K. "
        r"Mean $\pm$ std over 5 seeds. $^\dagger$ indicates statistically "
        r"significant difference from vanilla DPO ($p < 0.0026$ after "
        r"Bonferroni correction).}"
    )
    lines.append(r"\label{tab:dpo_variant_results}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}lcccc@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Variant} & \textbf{Mean} & \textbf{Std} & "
        r"\textbf{$\Delta$ vs DPO} & \textbf{$p$-value} \\"
    )
    lines.append(r"\midrule")

    # Collect and sort by mean accuracy
    variant_data = []
    for alg in sorted(variant_algs):
        accs = results.get((alg, "qwen15b", task), [])
        if not accs:
            continue
        mean = np.mean(accs) * 100
        std = np.std(accs, ddof=1) * 100 if len(accs) > 1 else 0
        delta = mean - (np.mean(dpo_accs) * 100 if dpo_accs else 0)

        # Welch's t-test against DPO
        p_val = None
        sig = False
        if len(accs) >= 2 and len(dpo_accs) >= 2:
            t_stat, p_val = scipy_stats.ttest_ind(accs, dpo_accs, equal_var=False)
            sig = p_val < 0.0026  # Bonferroni correction for 19 comparisons

        variant_data.append((alg, mean, std, delta, p_val, sig, len(accs)))

    variant_data.sort(key=lambda x: -x[1])

    for alg, mean, std, delta, p_val, sig, n in variant_data:
        display = ALG_DISPLAY.get(alg, alg.upper())
        if sig:
            display += r"$^\dagger$"

        delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
        p_str = f"{p_val:.4f}" if p_val is not None else "--"

        lines.append(
            f"{display} & {mean:.2f} & {std:.2f} & {delta_str} & {p_str} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_cross_task_table(
    results: Dict, base_results: Dict, scale: str = "qwen15b"
) -> str:
    """Generate the cross-task results table (Algorithm x Task)."""
    tasks = ["gsm8k", "math", "mbpp"]
    task_display = {"gsm8k": "GSM8K", "math": "MATH", "mbpp": "MBPP"}

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Cross-task results at "
        + SCALE_DISPLAY.get(scale, scale)
        + r" scale. Mean $\pm$ std over available seeds.}"
    )
    lines.append(r"\label{tab:cross_task}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}l" + "c" * len(tasks) + r"@{}}")
    lines.append(r"\toprule")
    header = r"\textbf{Algorithm}"
    for t in tasks:
        header += f" & \\textbf{{{task_display[t]}}}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Base model
    base_row = "Base Model"
    for t in tasks:
        acc = base_results.get((scale, t))
        if acc is not None:
            base_row += f" & {acc*100:.2f}"
        else:
            base_row += r" & --"
    base_row += r" \\"
    lines.append(base_row)
    lines.append(r"\midrule")

    for alg in ALG_ORDER:
        has_any = any(results.get((alg, scale, t)) for t in tasks)
        if not has_any:
            continue

        display = ALG_DISPLAY.get(alg, alg.upper())
        row = display
        for t in tasks:
            accs = results.get((alg, scale, t), [])
            row += " & " + format_acc(accs)
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# CSV and JSON export
# ──────────────────────────────────────────────────────────────────────


def export_csv(results: Dict, base_results: Dict, output_path: str):
    """Export all results to CSV for easy inspection and plotting."""
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "algorithm",
                "scale",
                "task",
                "n_seeds",
                "mean_accuracy",
                "std_accuracy",
                "individual_accuracies",
            ]
        )

        # Base models
        for (scale, task), acc in sorted(base_results.items()):
            writer.writerow(
                ["base", SCALE_DISPLAY.get(scale, scale), task, 1, acc, 0.0, str(acc)]
            )

        # All experiments
        for (alg, scale, task), accs in sorted(results.items()):
            mean = np.mean(accs)
            std = np.std(accs, ddof=1) if len(accs) > 1 else 0
            writer.writerow(
                [
                    alg,
                    SCALE_DISPLAY.get(scale, scale),
                    task,
                    len(accs),
                    round(mean, 6),
                    round(std, 6),
                    ";".join(f"{a:.6f}" for a in accs),
                ]
            )


def export_json(results: Dict, base_results: Dict, output_path: str):
    """Export all results to JSON for figure generation."""
    export = {
        "base_models": {
            f"{SCALE_DISPLAY.get(s, s)}_{t}": round(a, 6)
            for (s, t), a in base_results.items()
        },
        "experiments": {},
    }

    for (alg, scale, task), accs in sorted(results.items()):
        key = f"{alg}_{SCALE_DISPLAY.get(scale, scale)}_{task}"
        export["experiments"][key] = {
            "algorithm": alg,
            "scale": SCALE_DISPLAY.get(scale, scale),
            "task": task,
            "n_seeds": len(accs),
            "mean": round(float(np.mean(accs)), 6),
            "std": round(float(np.std(accs, ddof=1)), 6) if len(accs) > 1 else 0.0,
            "accuracies": [round(a, 6) for a in accs],
        }

    with open(output_path, "w") as f:
        json.dump(export, f, indent=2)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Aggregate NeurIPS eval results")
    parser.add_argument(
        "--eval-dir", type=str, default=EVAL_DIR, help="Directory with eval results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Output directory for tables/CSVs",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from {args.eval_dir}...")
    results, base_results = aggregate_results(args.eval_dir)

    print(f"\nFound {len(results)} (algorithm, scale, task) combinations")
    print(f"Found {len(base_results)} base model results")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for (alg, scale, task), accs in sorted(results.items()):
        mean = np.mean(accs) * 100
        std = np.std(accs, ddof=1) * 100 if len(accs) > 1 else 0
        display_alg = ALG_DISPLAY.get(alg, alg)
        display_scale = SCALE_DISPLAY.get(scale, scale)
        print(
            f"  {display_alg:10s} {display_scale:>4s} {task:10s}: "
            f"{mean:6.2f}% +/- {std:5.2f}% (n={len(accs)})"
        )

    # Generate tables
    print("\nGenerating LaTeX tables...")

    main_table = generate_main_table(results, base_results, task="gsm8k")
    table_path = os.path.join(args.output_dir, "table_main_results.tex")
    with open(table_path, "w") as f:
        f.write(main_table)
    print(f"  Main table: {table_path}")

    dpo_table = generate_dpo_variant_table(results, task="gsm8k")
    table_path = os.path.join(args.output_dir, "table_dpo_variants.tex")
    with open(table_path, "w") as f:
        f.write(dpo_table)
    print(f"  DPO variant table: {table_path}")

    cross_table = generate_cross_task_table(results, base_results, scale="qwen15b")
    table_path = os.path.join(args.output_dir, "table_cross_task.tex")
    with open(table_path, "w") as f:
        f.write(cross_table)
    print(f"  Cross-task table: {table_path}")

    # Export CSV and JSON
    csv_path = os.path.join(args.output_dir, "all_results.csv")
    export_csv(results, base_results, csv_path)
    print(f"  CSV: {csv_path}")

    json_path = os.path.join(args.output_dir, "all_results.json")
    export_json(results, base_results, json_path)
    print(f"  JSON: {json_path}")

    # Print the main table to stdout for quick inspection
    print("\n" + "=" * 70)
    print("MAIN TABLE (LaTeX)")
    print("=" * 70)
    print(main_table)

    print("\nDone!")


if __name__ == "__main__":
    main()
