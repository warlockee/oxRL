#!/usr/bin/env python3
"""
Patch the NeurIPS paper with latest aggregated results.

Reads aggregated all_results.json and updates the main results table,
replacing placeholder cells with actual numbers.

Usage:
    python scripts/patch_paper_results.py
    python scripts/patch_paper_results.py --dry-run   # Show changes without writing
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

RESULTS_JSON = "/home/ec2-user/fsx/oxrl_results/aggregated/all_results.json"
PAPER_PATH = "/home/ec2-user/fsx/oxRL/docs/neurips2026_paper.tex"

SCALE_MAP = {"qwen05b": "0.5B", "qwen15b": "1.5B", "qwen3b": "3B", "qwen7b": "7B"}
ALG_MAP = {
    "sft": "SFT", "dpo": "DPO", "simpo": "SimPO", "ipo": "IPO",
    "kto": "KTO", "hinge": "Hinge", "orpo": "ORPO",
    "sgrpo": "SGRPO", "gspo": "GSPO", "cispo": "CISPO", "ppo": "PPO",
}


def load_results(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def format_cell(mean: float, std: float, n_seeds: int, bold: bool = False) -> str:
    """Format a table cell: mean ± std (or just mean if n=1)."""
    m = mean * 100
    if n_seeds > 1:
        s = std * 100
        if s < 0.01:
            text = f"{m:.2f}" + r"$_{\pm 0.0}$"
        else:
            text = f"{m:.2f}" + r"$_{\pm " + f"{s:.1f}" + r"}$"
    else:
        text = f"{m:.2f}"

    if bold:
        text = r"\textbf{" + text + "}"
    return text


def patch_main_table(paper: str, data: Dict) -> str:
    """Replace placeholder cells in the main results table with actual numbers."""
    exps = data.get("experiments", {})
    base = data.get("base_models", {})

    # Build lookup: (alg, scale) -> (mean, std, n)
    lookup = {}
    for key, info in exps.items():
        if info["task"] == "gsm8k":
            lookup[(info["algorithm"], info["scale"])] = (
                info["mean"], info["std"], info["n_seeds"]
            )

    # Find best per scale (for bolding)
    best_per_scale = {}
    for scale in ["0.5B", "1.5B", "3B", "7B"]:
        best = 0
        for (a, s), (m, _, _) in lookup.items():
            if s == scale and m > best:
                best = m
        best_per_scale[scale] = best

    # Patch each algorithm row
    alg_row_names = {
        "SFT": "sft", "DPO": "dpo", "SimPO": "simpo", "IPO": "ipo",
        "KTO": "kto", "SGRPO": "sgrpo", "GSPO": "gspo",
        "CISPO": "cispo", "PPO": "ppo",
    }

    changes = 0
    for display_name, alg_key in alg_row_names.items():
        for scale in ["1.5B", "3B", "7B"]:
            result = lookup.get((alg_key, scale))
            if result is None:
                continue

            mean, std, n = result
            is_best = abs(mean - best_per_scale.get(scale, 0)) < 1e-6
            cell = format_cell(mean, std, n, bold=is_best)

            # Find and replace the placeholder in the row
            # Match pattern: "DISPLAY_NAME & ... & \placeholder{--} & ..."
            # We need to find the correct column
            col_idx = {"1.5B": 1, "3B": 2, "7B": 3}[scale]

            # Use regex to find the row and replace the n-th placeholder
            pattern = re.escape(display_name)
            # Find the line containing this algorithm
            lines = paper.split("\n")
            for i, line in enumerate(lines):
                # Match line starting with the display name (possibly with bold/markers)
                if line.strip().startswith(display_name) and r"\placeholder{--}" in line:
                    # Count which placeholder to replace (1st = 1.5B, 2nd = 3B, 3rd = 7B)
                    placeholder = r"\placeholder{--}"
                    parts = line.split(placeholder)
                    if len(parts) > col_idx:
                        # Replace the col_idx-th placeholder
                        new_parts = list(parts)
                        new_parts.insert(col_idx, cell)
                        new_line = placeholder.join(new_parts[:col_idx]) + cell + placeholder.join(new_parts[col_idx:])
                        # Simpler: replace n-th occurrence
                        count = 0
                        new_line = line
                        for match in re.finditer(re.escape(placeholder), line):
                            count += 1
                            if count == col_idx:
                                new_line = line[:match.start()] + cell + line[match.end():]
                                break
                        if new_line != line:
                            lines[i] = new_line
                            changes += 1
                    break

    if changes > 0:
        paper = "\n".join(lines)

    return paper, changes


def main():
    parser = argparse.ArgumentParser(description="Patch paper with latest results")
    parser.add_argument("--dry-run", action="store_true", help="Show changes only")
    parser.add_argument("--results", default=RESULTS_JSON)
    parser.add_argument("--paper", default=PAPER_PATH)
    args = parser.parse_args()

    print(f"Loading results from {args.results}...")
    data = load_results(args.results)

    exps = data.get("experiments", {})
    scales_with_data = set()
    for key, info in exps.items():
        if info["task"] == "gsm8k":
            scales_with_data.add(info["scale"])

    print(f"Scales with GSM8K data: {sorted(scales_with_data)}")

    print(f"\nReading paper from {args.paper}...")
    paper = Path(args.paper).read_text()

    new_paper, changes = patch_main_table(paper, data)

    print(f"Made {changes} cell replacements in main table")

    if changes > 0 and not args.dry_run:
        Path(args.paper).write_text(new_paper)
        print("Paper updated!")
    elif args.dry_run:
        print("[DRY RUN] No changes written")


if __name__ == "__main__":
    main()
