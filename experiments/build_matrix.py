"""Generate the full (model, algorithm, seed) configuration matrix from the paper.

Usage: python experiments/build_matrix.py --out experiments/configs

Each YAML is callable directly via:
    oxrl train --config experiments/configs/<model>__<algo>__seed<seed>.yaml

The matrix exactly mirrors the paper's run accounting (Table 17 / Appendix I);
re-running every config end-to-end reproduces the per-seed numbers in
Tables 1, 2, 6, 7, 18, 19.
"""

import argparse
import os
from pathlib import Path

from oxrl.swarm.config_generator import generate_config, save_config


CORE_QWEN_SCALES = [
    ("Qwen/Qwen2.5-0.5B-Instruct", 0.5),
    ("Qwen/Qwen2.5-1.5B-Instruct", 1.5),
    ("Qwen/Qwen2.5-3B-Instruct", 3.0),
    ("Qwen/Qwen2.5-7B-Instruct", 7.0),
    ("Qwen/Qwen2.5-14B-Instruct", 14.0),
]

CORE_GEMMA_SCALES = [
    ("google/gemma-3-1b-it", 1.0),
    ("google/gemma-3-4b-it", 4.0),
    ("google/gemma-3-12b-it", 12.0),
]

CORE_OFFLINE = ["sft", "dpo", "ipo", "kto", "simpo"]
ONLINE_RL = ["sgrpo"]
DPO_VARIANTS_20 = [
    "dpo", "ipo", "simpo", "kto", "hinge", "cpo", "orpo", "rdpo",
    "cdpo", "betadpo", "caldpo", "dpop", "odpo", "exo", "alphapo",
    "apo", "sppo", "robust_dpo", "gpo", "focalpo",
]
DPO_STRATIFIED_5 = ["exo", "odpo", "gpo", "dpop", "orpo"]  # +vanilla dpo
LR_SWEEP_LRS = [5e-7, 1e-6, 5e-6, 1e-5]


def slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def emit(out_dir: Path, model_id: str, algo: str, seed: int,
         param_count_b: float, lr: float | None = None,
         tag: str | None = None, **overrides) -> Path:
    cfg = generate_config(model_name=model_id, task="math",
                          param_count_b=param_count_b)
    cfg["train"]["alg_name"] = algo
    cfg["run"]["seed"] = seed
    if lr is not None:
        cfg["train"]["lr"] = lr
    if tag is not None:
        cfg["run"]["experiment_id"] = (
            f"{cfg['run']['experiment_id']}__{tag}__seed{seed}"
        )
    # User-supplied overrides land directly under run/train via dot-paths
    for k, v in overrides.items():
        if "." in k:
            section, key = k.split(".", 1)
            cfg.setdefault(section, {})[key] = v
        else:
            cfg.setdefault("run", {})[k] = v
    fn = out_dir / f"{slug(model_id)}__{algo}__seed{seed}.yaml"
    save_config(cfg, str(fn))
    return fn


def build(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0

    # Core Qwen 2.5 — 5 scales × {6 offline + 1 online} × seeds
    seeds_default = [42, 123, 456]
    seeds_7b = [42, 456, 789, 1024, 1337]
    for model, b in CORE_QWEN_SCALES:
        seeds = seeds_7b if b == 7.0 else seeds_default
        for algo in CORE_OFFLINE + ONLINE_RL:
            for s in seeds:
                emit(out_dir, model, algo, s, b)
                n += 1

    # Gemma 3 — 3 scales × 3 algos × 3 seeds
    for model, b in CORE_GEMMA_SCALES:
        for algo in ["sft", "dpo", "simpo"]:
            for s in seeds_default:
                emit(out_dir, model, algo, s, b)
                n += 1

    # DPO variants @ 1.5B — 20 × 5
    for v in DPO_VARIANTS_20:
        for s in [42, 123, 456, 789, 1024]:
            emit(out_dir, "Qwen/Qwen2.5-1.5B-Instruct", v, s, 1.5)
            n += 1
    # Stratified DPO variants @ 7B — 5 × 3 (+ vanilla DPO 5 seeds, already above)
    for v in DPO_STRATIFIED_5:
        for s in seeds_default:
            emit(out_dir, "Qwen/Qwen2.5-7B-Instruct", v, s, 7.0)
            n += 1

    # LR sweeps for SP-RFT and DPO at 1.5B / 3B / 7B
    for model, b in [("Qwen/Qwen2.5-1.5B-Instruct", 1.5),
                     ("Qwen/Qwen2.5-3B-Instruct", 3.0),
                     ("Qwen/Qwen2.5-7B-Instruct", 7.0)]:
        for lr in LR_SWEEP_LRS:
            for algo in ["sft", "dpo"]:
                for s in seeds_default:
                    emit(out_dir, model, algo, s, b, lr=lr,
                         tag=f"lrsweep_{lr:.0e}")
                    n += 1

    # Fixed-dataset control: 1.5B trained on 14B-generated data
    for algo in ["sft", "dpo"]:
        for s in seeds_default:
            emit(out_dir, "Qwen/Qwen2.5-1.5B-Instruct", algo, s, 1.5,
                 self_play_source="qwen2.5-14b-instruct",
                 tag="fixed_dataset_14B_data")
            n += 1

    # Gold-SFT control: MetaMathQA at 1.5B and 7B
    for model, b in [("Qwen/Qwen2.5-1.5B-Instruct", 1.5),
                     ("Qwen/Qwen2.5-7B-Instruct", 7.0)]:
        for s in seeds_default:
            emit(out_dir, model, "sft", s, b,
                 dataset_override="metamathqa",
                 tag="gold_sft_metamathqa")
            n += 1

    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory for generated YAMLs.")
    args = ap.parse_args()
    n = build(args.out)
    print(f"Wrote {n} configs to {args.out}")


if __name__ == "__main__":
    main()
