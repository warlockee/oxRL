"""
Generate a NeurIPS-compliant reproduction section for README.md.

Papers With Code ML Code Completeness Checklist requires:
  1. Specification of dependencies
  2. Training code
  3. Evaluation code
  4. Pre-trained models
  5. README with results table + commands to reproduce
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import requests


def list_hf_repos(
    username: str, hf_token: Optional[str] = None
) -> Dict[str, List[Dict]]:
    """List all models and datasets for a HuggingFace user."""
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    models = requests.get(
        f"https://huggingface.co/api/models?author={username}&sort=lastModified&direction=-1&limit=100",
        headers=headers, timeout=30,
    ).json()

    datasets = requests.get(
        f"https://huggingface.co/api/datasets?author={username}&sort=lastModified&direction=-1&limit=100",
        headers=headers, timeout=30,
    ).json()

    return {"models": models, "datasets": datasets}


def generate_reproduction_section(
    paper_title: str,
    paper_description: str,
    results_tables: List[Dict],
    training_commands: List[Dict],
    eval_commands: List[str],
    checkpoints: List[Dict],
    datasets: List[Dict],
    hardware_description: str,
) -> str:
    """Generate a markdown reproduction section.

    Args:
        paper_title: Full paper title.
        paper_description: One-line description of the study scope.
        results_tables: List of dicts with keys: title, headers, rows.
            Each row is a list of cell values.
        training_commands: List of dicts with keys: description, command.
        eval_commands: List of command strings.
        checkpoints: List of dicts with keys: name, url, description.
        datasets: List of dicts with keys: name, url, description.
        hardware_description: Free-text hardware and compute summary.

    Returns:
        Markdown string for the reproduction section.
    """
    lines = []
    lines.append(f"## Reproducing Paper Results (NeurIPS 2026)")
    lines.append("")
    lines.append(
        f"The paper *\"{paper_title}\"* {paper_description}"
    )
    lines.append("")

    # Results tables.
    for table in results_tables:
        lines.append(f"### {table['title']}")
        lines.append("")
        headers = table["headers"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(":---:" if i > 0 else ":---" for i in range(len(headers))) + "|")
        for row in table["rows"]:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        lines.append("")
        if "caption" in table:
            lines.append(table["caption"])
            lines.append("")

    # Training commands.
    if training_commands:
        lines.append("### Training Commands")
        lines.append("")
        lines.append("```bash")
        for cmd in training_commands:
            lines.append(f"# {cmd['description']}")
            lines.append(cmd["command"])
            lines.append("")
        lines.append("```")
        lines.append("")

    # Evaluation commands.
    if eval_commands:
        lines.append("### Evaluation")
        lines.append("")
        lines.append("```bash")
        for cmd in eval_commands:
            lines.append(cmd)
            lines.append("")
        lines.append("```")
        lines.append("")

    # Pre-trained checkpoints.
    if checkpoints:
        lines.append("### Pre-trained Checkpoints")
        lines.append("")
        lines.append("All trained checkpoints from the paper are available on HuggingFace:")
        lines.append("")
        lines.append("| Checkpoint | Models |")
        lines.append("|:---|:---|")
        for ckpt in checkpoints:
            lines.append(f"| [{ckpt['name']}]({ckpt['url']}) | {ckpt['description']} |")
        lines.append("")

    # Datasets.
    if datasets:
        lines.append("### Datasets")
        lines.append("")
        lines.append("| Dataset | Description |")
        lines.append("|:---|:---|")
        for ds in datasets:
            lines.append(f"| [{ds['name']}]({ds['url']}) | {ds['description']} |")
        lines.append("")

    # Hardware.
    if hardware_description:
        lines.append("### Hardware")
        lines.append("")
        lines.append(hardware_description)
        lines.append("")

    return "\n".join(lines)


def check_readme_completeness(readme_path: str) -> Dict:
    """Check a README against the Papers With Code ML Code Completeness Checklist.

    Returns a dict mapping each checklist item to PASS/FAIL.
    """
    text = Path(readme_path).read_text()
    text_lower = text.lower()

    checklist = {}

    # 1. Dependencies.
    checklist["dependencies"] = bool(
        re.search(r"requirements\.txt|pip install|setup\.py|pyproject\.toml", text)
    )

    # 2. Training code.
    checklist["training_code"] = bool(
        re.search(r"train|training command", text_lower)
        and re.search(r"```", text)
    )

    # 3. Evaluation code.
    checklist["evaluation_code"] = bool(
        re.search(r"eval|evaluat", text_lower)
        and re.search(r"```", text)
    )

    # 4. Pre-trained models.
    checklist["pretrained_models"] = bool(
        re.search(r"checkpoint|pre.?trained|huggingface\.co.*ckpt", text_lower)
    )

    # 5. Results table.
    checklist["results_table"] = bool(
        re.search(r"\|.*\|.*\|", text)
        and re.search(r"accuracy|%|result|gsm8k|math", text_lower)
    )

    return checklist
