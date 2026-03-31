#!/usr/bin/env python3
"""
Generate publication-quality figures for the NeurIPS 2026 paper.

Reads aggregated results JSON and produces:
  1. Scaling curves (accuracy vs model scale, one line per algorithm)
  2. DPO variant bar chart with error bars and significance markers
  3. Algorithm x task heatmap
  4. Training dynamics plots (from training logs)
  5. Algorithm family comparison (grouped bar chart)

Usage:
    python scripts/generate_figures.py
    python scripts/generate_figures.py --results-json /path/to/all_results.json
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────────────────

# NeurIPS-friendly settings
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Color palette: distinguishable, colorblind-friendly
COLORS = {
    "sft": "#888888",
    "dpo": "#1f77b4",
    "simpo": "#ff7f0e",
    "ipo": "#2ca02c",
    "kto": "#d62728",
    "hinge": "#9467bd",
    "orpo": "#8c564b",
    "cpo": "#e377c2",
    "sgrpo": "#17becf",
    "gspo": "#bcbd22",
    "cispo": "#aec7e8",
    "ppo": "#ff9896",
    "base": "#333333",
}

MARKERS = {
    "sft": "s",
    "dpo": "o",
    "simpo": "D",
    "ipo": "^",
    "kto": "v",
    "hinge": "<",
    "orpo": ">",
    "sgrpo": "P",
    "gspo": "X",
    "ppo": "*",
}

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
}

SCALE_PARAMS = {"0.5B": 0.5, "1.5B": 1.5, "3B": 3.0, "7B": 7.0}
SCALE_ORDER = ["0.5B", "1.5B", "3B", "7B"]

RESULTS_JSON = "/home/ec2-user/fsx/oxrl_results/aggregated/all_results.json"
OUTPUT_DIR = "/home/ec2-user/fsx/oxRL/docs/figures"
LOG_DIR = "/home/ec2-user/fsx/oxrl_logs/neurips2026"


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────


def load_results(results_json: str) -> Dict:
    """Load aggregated results."""
    with open(results_json) as f:
        return json.load(f)


def get_alg_scale_data(data: Dict, task: str = "gsm8k"):
    """Extract {alg: {scale: (mean, std, n)}} for a given task."""
    out = {}
    for key, info in data.get("experiments", {}).items():
        if info["task"] != task:
            continue
        alg = info["algorithm"]
        scale = info["scale"]
        if alg not in out:
            out[alg] = {}
        out[alg][scale] = (info["mean"], info["std"], info["n_seeds"])

    # Add base model
    base = {}
    for bkey, bacc in data.get("base_models", {}).items():
        parts = bkey.split("_")
        if parts[-1] == task:
            scale = parts[0]
            base[scale] = (bacc, 0.0, 1)
    if base:
        out["base"] = base

    return out


# ──────────────────────────────────────────────────────────────────────
# Figure 1: Scaling curves
# ──────────────────────────────────────────────────────────────────────


def plot_scaling_curves(data: Dict, output_dir: str, task: str = "gsm8k"):
    """Plot accuracy vs model scale for each algorithm."""
    alg_data = get_alg_scale_data(data, task)
    if not alg_data:
        print(f"  No data for scaling curves ({task})")
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    # Plot order: base first (dashed), then offline, then online
    plot_order = ["base", "sft", "dpo", "simpo", "ipo", "kto", "hinge", "sgrpo", "gspo", "ppo"]
    plot_order = [a for a in plot_order if a in alg_data]

    x_params = np.array([SCALE_PARAMS[s] for s in SCALE_ORDER])

    for alg in plot_order:
        scales = alg_data[alg]
        x_vals, y_vals, y_errs = [], [], []
        for scale in SCALE_ORDER:
            if scale in scales:
                mean, std, n = scales[scale]
                x_vals.append(SCALE_PARAMS[scale])
                y_vals.append(mean * 100)
                y_errs.append(std * 100)

        if not x_vals:
            continue

        color = COLORS.get(alg, "#666666")
        marker = MARKERS.get(alg, "o")
        label = ALG_DISPLAY.get(alg, alg)
        ls = "--" if alg == "base" else "-"
        alpha = 0.6 if alg == "base" else 0.9

        ax.errorbar(
            x_vals, y_vals, yerr=y_errs,
            label=label, color=color, marker=marker,
            linestyle=ls, alpha=alpha, markersize=5,
            capsize=2, linewidth=1.5,
        )

    ax.set_xlabel("Model Scale (Billion Parameters)")
    ax.set_ylabel("GSM8K Accuracy (%)")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1.5, 3, 7])
    ax.get_xaxis().set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.legend(loc="best", ncol=2, framealpha=0.8)
    ax.set_title("Post-Training Algorithm Scaling on GSM8K")

    path = os.path.join(output_dir, f"scaling_curves_{task}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Scaling curves: {path}")

    # Also save PNG for quick viewing
    path_png = path.replace(".pdf", ".png")
    fig2, ax2 = plt.subplots(figsize=(5.5, 3.8))
    for alg in plot_order:
        scales = alg_data[alg]
        x_vals, y_vals, y_errs = [], [], []
        for scale in SCALE_ORDER:
            if scale in scales:
                mean, std, n = scales[scale]
                x_vals.append(SCALE_PARAMS[scale])
                y_vals.append(mean * 100)
                y_errs.append(std * 100)
        if not x_vals:
            continue
        color = COLORS.get(alg, "#666666")
        marker = MARKERS.get(alg, "o")
        label = ALG_DISPLAY.get(alg, alg)
        ls = "--" if alg == "base" else "-"
        alpha = 0.6 if alg == "base" else 0.9
        ax2.errorbar(x_vals, y_vals, yerr=y_errs, label=label, color=color,
                     marker=marker, linestyle=ls, alpha=alpha, markersize=5,
                     capsize=2, linewidth=1.5)
    ax2.set_xlabel("Model Scale (Billion Parameters)")
    ax2.set_ylabel("GSM8K Accuracy (%)")
    ax2.set_xscale("log")
    ax2.set_xticks([0.5, 1.5, 3, 7])
    ax2.get_xaxis().set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax2.legend(loc="best", ncol=2, framealpha=0.8)
    ax2.set_title("Post-Training Algorithm Scaling on GSM8K")
    fig2.savefig(path_png)
    plt.close(fig2)


# ──────────────────────────────────────────────────────────────────────
# Figure 2: DPO variant comparison
# ──────────────────────────────────────────────────────────────────────


def plot_dpo_variants(data: Dict, output_dir: str, task: str = "gsm8k"):
    """Bar chart of DPO variants at 1.5B scale with error bars."""
    exps = data.get("experiments", {})
    variants = {}

    # Collect all results at 1.5B
    for key, info in exps.items():
        if info["task"] == task and info["scale"] == "1.5B":
            alg = info["algorithm"]
            variants[alg] = info

    if len(variants) < 3:
        print(f"  Not enough DPO variant data ({len(variants)} algorithms found)")
        return

    # Sort by mean accuracy
    sorted_variants = sorted(variants.items(), key=lambda x: -x[1]["mean"])

    fig, ax = plt.subplots(figsize=(7, 3.5))

    names = [ALG_DISPLAY.get(alg, alg) for alg, _ in sorted_variants]
    means = [info["mean"] * 100 for _, info in sorted_variants]
    stds = [info["std"] * 100 for _, info in sorted_variants]

    # Color: DPO variants in blue shades, DPO itself highlighted
    colors = []
    for alg, _ in sorted_variants:
        if alg == "dpo":
            colors.append("#d62728")  # Red for vanilla DPO
        elif alg in ("simpo", "orpo", "cpo"):
            colors.append("#ff7f0e")  # Orange for reference-free
        elif alg in ("sgrpo", "gspo", "ppo", "cispo"):
            colors.append("#17becf")  # Cyan for online RL
        elif alg == "sft":
            colors.append("#888888")  # Grey for SFT
        else:
            colors.append("#1f77b4")  # Blue for DPO variants

    bars = ax.barh(range(len(names)), means, xerr=stds, color=colors,
                   capsize=2, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("GSM8K Accuracy (%)")
    ax.set_title("DPO Variant Comparison at 1.5B Scale")

    # Add DPO reference line
    dpo_mean = variants.get("dpo", {}).get("mean", 0) * 100
    if dpo_mean > 0:
        ax.axvline(x=dpo_mean, color="#d62728", linestyle="--", alpha=0.5,
                   label=f"DPO baseline ({dpo_mean:.1f}%)")
        ax.legend(loc="lower right")

    path = os.path.join(output_dir, f"dpo_variants_{task}.pdf")
    fig.savefig(path)
    path_png = path.replace(".pdf", ".png")
    fig.savefig(path_png)
    plt.close(fig)
    print(f"  DPO variants: {path}")


# ──────────────────────────────────────────────────────────────────────
# Figure 3: Algorithm x Task heatmap
# ──────────────────────────────────────────────────────────────────────


def plot_task_heatmap(data: Dict, output_dir: str):
    """Heatmap: algorithm (rows) x task (columns), showing relative improvement over base."""
    tasks = ["gsm8k", "math", "mbpp"]
    task_display = {"gsm8k": "GSM8K", "math": "MATH", "mbpp": "MBPP"}
    algs = ["sft", "dpo", "simpo", "ipo", "kto", "hinge", "sgrpo", "gspo", "ppo"]

    exps = data.get("experiments", {})
    base = data.get("base_models", {})

    # Build the matrix: relative improvement over base
    matrix = np.full((len(algs), len(tasks)), np.nan)

    for i, alg in enumerate(algs):
        for j, task in enumerate(tasks):
            # Try to find results (any scale -- prefer 1.5B)
            for scale in ["1.5B", "0.5B", "3B", "7B"]:
                key = f"{alg}_{scale}_{task}"
                if key in exps:
                    mean = exps[key]["mean"]
                    base_key = f"{scale}_{task}"
                    base_acc = base.get(base_key, 0)
                    if base_acc > 0:
                        matrix[i, j] = (mean - base_acc) / base_acc * 100
                    else:
                        matrix[i, j] = mean * 100
                    break

    # Only plot if we have enough data
    valid_cells = np.sum(~np.isnan(matrix))
    if valid_cells < 3:
        print(f"  Not enough data for heatmap ({valid_cells} cells)")
        return

    fig, ax = plt.subplots(figsize=(4, 4.5))
    masked = np.ma.masked_invalid(matrix)

    im = ax.imshow(masked, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([task_display[t] for t in tasks])
    ax.set_yticks(range(len(algs)))
    ax.set_yticklabels([ALG_DISPLAY.get(a, a) for a in algs])

    # Annotate cells
    for i in range(len(algs)):
        for j in range(len(tasks)):
            if not np.isnan(matrix[i, j]):
                val = matrix[i, j]
                text = f"{val:+.1f}%"
                color = "white" if abs(val) > 30 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="Relative Improvement (%)")
    ax.set_title("Relative Improvement over Base Model")

    path = os.path.join(output_dir, "task_heatmap.pdf")
    fig.savefig(path)
    path_png = path.replace(".pdf", ".png")
    fig.savefig(path_png)
    plt.close(fig)
    print(f"  Task heatmap: {path}")


# ──────────────────────────────────────────────────────────────────────
# Figure 4: Training dynamics
# ──────────────────────────────────────────────────────────────────────


def parse_training_log(log_path: str) -> Dict:
    """Parse a training log file for loss and progress information."""
    losses = []
    epochs = []
    val_losses = []

    with open(log_path) as f:
        for line in f:
            # Match progress bar output: "Epoch X/Y: ZZ%|...|step/total [time, loss=L]"
            m = re.search(r"Epoch (\d+)/(\d+):\s*\d+%.*?loss=([\d.]+)", line)
            if m:
                epoch = int(m.group(1))
                loss = float(m.group(3))
                losses.append(loss)
                epochs.append(epoch)

            # Match validation loss
            m = re.search(r"Epoch (\d+), Validation Loss: ([\d.]+)", line)
            if m:
                epoch = int(m.group(1))
                val_loss = float(m.group(2))
                val_losses.append((epoch, val_loss))

    return {
        "losses": losses,
        "epochs": epochs,
        "val_losses": val_losses,
    }


def plot_training_dynamics(output_dir: str, log_dir: str):
    """Plot training loss curves for representative algorithms."""
    # Look for 0.5B logs for offline and online methods
    target_logs = {
        "DPO": "core_math_gsm8k_dpo_qwen05b_gsm8k_s42.log",
        "SimPO": "core_math_gsm8k_simpo_qwen05b_gsm8k_s42.log",
        "SFT": "core_math_gsm8k_sft_qwen05b_gsm8k_s42.log",
        "SGRPO": "core_math_gsm8k_sgrpo_qwen05b_gsm8k_s42.log",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    found_any = False
    for name, logfile in target_logs.items():
        log_path = os.path.join(log_dir, logfile)
        if not os.path.exists(log_path):
            continue

        parsed = parse_training_log(log_path)
        if not parsed["losses"]:
            continue

        found_any = True
        losses = parsed["losses"]
        color = COLORS.get(name.lower(), "#666666")

        # Smooth losses for plotting
        window = min(20, len(losses) // 5) if len(losses) > 20 else 1
        if window > 1:
            smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        else:
            smoothed = np.array(losses)

        axes[0].plot(range(len(smoothed)), smoothed, label=name, color=color,
                     alpha=0.8, linewidth=1.2)

        # Validation loss
        if parsed["val_losses"]:
            ve, vl = zip(*parsed["val_losses"])
            axes[1].plot(ve, vl, label=name, color=color, marker="o",
                        markersize=4, linewidth=1.5)

    if not found_any:
        print("  No training logs found for dynamics plot")
        plt.close(fig)
        return

    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss (0.5B)")
    axes[0].legend()

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Loss")
    axes[1].set_title("Validation Loss (0.5B)")
    axes[1].legend()

    fig.tight_layout()
    path = os.path.join(output_dir, "training_dynamics.pdf")
    fig.savefig(path)
    path_png = path.replace(".pdf", ".png")
    fig.savefig(path_png)
    plt.close(fig)
    print(f"  Training dynamics: {path}")


# ──────────────────────────────────────────────────────────────────────
# Figure 5: Algorithm family grouped bar chart
# ──────────────────────────────────────────────────────────────────────


def plot_family_comparison(data: Dict, output_dir: str, task: str = "gsm8k"):
    """Grouped bar chart: Algorithm families (SFT, Offline Pref, Online RL) across scales."""
    alg_data = get_alg_scale_data(data, task)
    if not alg_data:
        print(f"  No data for family comparison ({task})")
        return

    families = {
        "SFT": ["sft"],
        "Offline Pref.\n(best)": ["dpo", "simpo", "ipo", "kto", "hinge", "orpo"],
        "Online RL\n(best)": ["sgrpo", "gspo", "ppo"],
    }

    fig, ax = plt.subplots(figsize=(5, 3.5))

    x = np.arange(len(SCALE_ORDER))
    width = 0.22
    family_colors = ["#888888", "#1f77b4", "#17becf"]

    for fidx, (fname, alg_list) in enumerate(families.items()):
        means, errs = [], []
        for scale in SCALE_ORDER:
            # Take the best algorithm in this family at this scale
            best_mean = 0
            best_std = 0
            for alg in alg_list:
                if alg in alg_data and scale in alg_data[alg]:
                    m, s, n = alg_data[alg][scale]
                    if m > best_mean:
                        best_mean = m
                        best_std = s
            means.append(best_mean * 100)
            errs.append(best_std * 100)

        offset = (fidx - 1) * width
        ax.bar(x + offset, means, width, yerr=errs, label=fname,
               color=family_colors[fidx], capsize=2, edgecolor="white")

    # Add base model as reference line per scale
    for i, scale in enumerate(SCALE_ORDER):
        if "base" in alg_data and scale in alg_data["base"]:
            base_acc = alg_data["base"][scale][0] * 100
            ax.plot([i - 0.4, i + 0.4], [base_acc, base_acc], "k--", alpha=0.4,
                    linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(SCALE_ORDER)
    ax.set_xlabel("Model Scale")
    ax.set_ylabel("GSM8K Accuracy (%)")
    ax.set_title("Algorithm Family Performance Across Scales")
    ax.legend(loc="upper left")

    path = os.path.join(output_dir, f"family_comparison_{task}.pdf")
    fig.savefig(path)
    path_png = path.replace(".pdf", ".png")
    fig.savefig(path_png)
    plt.close(fig)
    print(f"  Family comparison: {path}")


# ──────────────────────────────────────────────────────────────────────
# Figure 6: Seed invariance visualization
# ──────────────────────────────────────────────────────────────────────


def plot_seed_invariance(data: Dict, output_dir: str, task: str = "gsm8k"):
    """Visualize per-seed accuracies to show seed invariance at 0.5B."""
    exps = data.get("experiments", {})

    # Collect per-seed accuracies for 0.5B algorithms
    alg_seeds = {}
    for key, info in exps.items():
        if info["task"] == task and info["scale"] == "0.5B":
            alg = info["algorithm"]
            accs = info.get("accuracies", [info["mean"]])
            alg_seeds[alg] = [a * 100 for a in accs]

    if not alg_seeds:
        print("  No per-seed data for seed invariance plot")
        return

    # Filter to algorithms with 3+ seeds
    alg_seeds = {k: v for k, v in alg_seeds.items() if len(v) >= 2}
    if not alg_seeds:
        print("  Not enough multi-seed data for seed invariance plot")
        return

    # Sort by mean
    sorted_algs = sorted(alg_seeds.items(), key=lambda x: -np.mean(x[1]))

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    y_positions = range(len(sorted_algs))
    for i, (alg, accs) in enumerate(sorted_algs):
        color = COLORS.get(alg, "#666666")
        display = ALG_DISPLAY.get(alg, alg)

        # Plot individual seed points
        for j, acc in enumerate(accs):
            ax.scatter(acc, i, color=color, s=40, zorder=3,
                      marker=["o", "s", "D", "^", "v"][j % 5],
                      edgecolors="black", linewidths=0.5)

        # Plot mean
        mean_acc = np.mean(accs)
        ax.plot([mean_acc], [i], marker="|", color="black", markersize=12,
                markeredgewidth=2, zorder=4)

        # Show std annotation
        std = np.std(accs, ddof=1) if len(accs) > 1 else 0
        ax.annotate(f"$\\sigma={std:.2f}$", xy=(mean_acc + 0.3, i),
                   fontsize=7, va="center")

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([ALG_DISPLAY.get(a, a) for a, _ in sorted_algs])
    ax.invert_yaxis()
    ax.set_xlabel("GSM8K Accuracy (%)")
    ax.set_title("Seed Invariance at 0.5B: Per-Seed Accuracy (3 seeds)")

    # Add legend for seed markers
    from matplotlib.lines import Line2D
    seed_labels = ["Seed 42", "Seed 123", "Seed 456"]
    seed_markers = ["o", "s", "D"]
    legend_elements = [Line2D([0], [0], marker=m, color="w", markerfacecolor="#666",
                             markersize=6, label=l, markeredgecolor="black",
                             markeredgewidth=0.5)
                      for m, l in zip(seed_markers, seed_labels)]
    legend_elements.append(Line2D([0], [0], marker="|", color="black",
                                  markersize=8, markeredgewidth=2, label="Mean",
                                  linestyle="None"))
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7)

    fig.tight_layout()
    path = os.path.join(output_dir, f"seed_invariance_{task}.pdf")
    fig.savefig(path)
    path_png = path.replace(".pdf", ".png")
    fig.savefig(path_png)
    plt.close(fig)
    print(f"  Seed invariance: {path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate NeurIPS paper figures")
    parser.add_argument(
        "--results-json", type=str, default=RESULTS_JSON,
        help="Path to aggregated all_results.json"
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help="Output directory for figures"
    )
    parser.add_argument(
        "--log-dir", type=str, default=LOG_DIR,
        help="Directory containing training logs"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from {args.results_json}...")
    try:
        data = load_results(args.results_json)
    except FileNotFoundError:
        print(f"Results file not found: {args.results_json}")
        print("Run aggregate_results.py first.")
        sys.exit(1)

    print(f"Generating figures in {args.output_dir}...\n")

    # Generate all figures
    plot_scaling_curves(data, args.output_dir, task="gsm8k")
    plot_dpo_variants(data, args.output_dir, task="gsm8k")
    plot_task_heatmap(data, args.output_dir)
    plot_training_dynamics(args.output_dir, args.log_dir)
    plot_family_comparison(data, args.output_dir, task="gsm8k")
    plot_seed_invariance(data, args.output_dir, task="gsm8k")

    print("\nDone! Figures saved to:", args.output_dir)


if __name__ == "__main__":
    main()
