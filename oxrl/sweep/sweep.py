"""
Experiment grid generation for NeurIPS 2026.

Generates YAML config files for all (algorithm x model x seed x task)
combinations in the experiment plan. Uses oxrl.swarm.config_generator as
the base config factory and applies algorithm-specific overrides.

Usage:
    # Generate the full Tier 1 math grid:
    python -m oxrl.sweep.sweep \
        --experiment core_math \
        --output-dir /home/ec2-user/fsx/oxrl_configs/neurips2026

    # Generate just the DPO variant sweep:
    python -m oxrl.sweep.sweep \
        --experiment dpo_variants \
        --output-dir /home/ec2-user/fsx/oxrl_configs/neurips2026

    # List available experiment grids:
    python -m oxrl.sweep.sweep --list
"""
import argparse
import itertools
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from oxrl.swarm.config_generator import generate_config, save_config


# ──────────────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────────────

MODELS = {
    "qwen05b":  {"name": "Qwen/Qwen2.5-0.5B-Instruct", "params_b": 0.5},
    "qwen1.5b": {"name": "Qwen/Qwen2.5-1.5B-Instruct", "params_b": 1.5},
    "qwen3b":   {"name": "Qwen/Qwen2.5-3B-Instruct",   "params_b": 3.0},
    "qwen7b":   {"name": "Qwen/Qwen2.5-7B-Instruct",   "params_b": 7.0},
    "coder1.5b": {"name": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "params_b": 1.5},
    "coder7b":   {"name": "Qwen/Qwen2.5-Coder-7B-Instruct",   "params_b": 7.0},
}

# ──────────────────────────────────────────────────────────────────────
# Algorithm registry
# ──────────────────────────────────────────────────────────────────────

# RL algorithms (use main_rl.py, online generation)
RL_ALGORITHMS = ["sgrpo", "gspo", "cispo", "ppo"]

# SL algorithms (use main_sl.py, offline preference data required)
# These need pre-generated preference datasets.
SL_PREF_ALGORITHMS = ["dpo", "simpo", "kto", "ipo"]

# SL algorithms (use main_sl.py, prompt-response data for SFT)
SL_SFT_ALGORITHMS = ["sft"]

# DPO variants for Experiment 3
DPO_VARIANT_ALGORITHMS = [
    "dpo", "ipo", "simpo", "kto", "orpo", "cpo", "alphapo",
    "rdpo", "cdpo", "betadpo", "caldpo", "sppo", "apo", "hinge",
    "robust_dpo", "exo", "odpo", "dpop", "focalpo", "gpo",
]

# ──────────────────────────────────────────────────────────────────────
# Task registry
# ──────────────────────────────────────────────────────────────────────

TASKS = {
    "gsm8k": {
        "task": "math",
        "reward_func": "gsm8k_reward_func",
        "eval_tasks": ["gsm8k"],
    },
    "math": {
        "task": "math-hard",
        "reward_func": "math_reward_func",
        "eval_tasks": ["math"],
    },
    "mbpp": {
        "task": "code",
        "reward_func": "code_reward_func",
        "eval_tasks": ["mbpp"],
    },
}


# ──────────────────────────────────────────────────────────────────────
# Experiment grid definitions
# ──────────────────────────────────────────────────────────────────────

def _get_data_dir() -> str:
    return "/home/ec2-user/fsx/oxrl_data/neurips2026"

def _get_checkpoint_dir() -> str:
    return "/home/ec2-user/fsx/oxrl_checkpoints/neurips2026"


EXPERIMENTS = {
    "core_math_gsm8k": {
        "description": "Exp 1a: 9 algorithms x 4 scales on GSM8K (3 seeds)",
        "models": ["qwen05b", "qwen1.5b", "qwen3b", "qwen7b"],
        "algorithms": RL_ALGORITHMS + SL_PREF_ALGORITHMS + SL_SFT_ALGORITHMS,
        "tasks": ["gsm8k"],
        "seeds": [42, 123, 456],
    },
    "core_math_hard": {
        "description": "Exp 1b: 9 algorithms x 4 scales on MATH (3 seeds)",
        "models": ["qwen05b", "qwen1.5b", "qwen3b", "qwen7b"],
        "algorithms": RL_ALGORITHMS + SL_PREF_ALGORITHMS + SL_SFT_ALGORITHMS,
        "tasks": ["math"],
        "seeds": [42, 123, 456],
    },
    "core_code": {
        "description": "Exp 2: 9 algorithms x 2 coder scales on MBPP (3 seeds)",
        "models": ["coder1.5b", "coder7b"],
        "algorithms": RL_ALGORITHMS + SL_PREF_ALGORITHMS + SL_SFT_ALGORITHMS,
        "tasks": ["mbpp"],
        "seeds": [42, 123, 456],
    },
    "dpo_variants": {
        "description": "Exp 3: 20 DPO variants at 1.5B on GSM8K (5 seeds)",
        "models": ["qwen1.5b"],
        "algorithms": DPO_VARIANT_ALGORITHMS,
        "tasks": ["gsm8k"],
        "seeds": [42, 123, 456, 789, 1024],
    },
    "ablation_nsamples": {
        "description": "Ablation A: n_samples sweep for SGRPO at 1.5B",
        "models": ["qwen1.5b"],
        "algorithms": ["sgrpo"],
        "tasks": ["gsm8k"],
        "seeds": [42, 123, 456],
        "overrides": [
            {"rollout": {"n_samples": n}} for n in [1, 2, 4, 8, 16, 32]
        ],
    },
    "ablation_kl": {
        "description": "Ablation D: KL coefficient sweep for SGRPO at 1.5B",
        "models": ["qwen1.5b"],
        "algorithms": ["sgrpo"],
        "tasks": ["gsm8k"],
        "seeds": [42, 123, 456],
        "overrides": [
            {"train": {"kl_coeff": kl}} for kl in [0.0, 0.001, 0.01, 0.1]
        ],
    },
    "ablation_lr": {
        "description": "Ablation E: LR sweep for top algorithms at 1.5B",
        "models": ["qwen1.5b"],
        "algorithms": ["sgrpo", "dpo", "ppo"],
        "tasks": ["gsm8k"],
        "seeds": [42, 123, 456],
        "overrides": [
            {"train": {"lr": lr}} for lr in [5e-7, 1e-6, 5e-6]
        ],
    },
}


# ──────────────────────────────────────────────────────────────────────
# Config generation
# ──────────────────────────────────────────────────────────────────────

def _method_for_algorithm(alg: str) -> str:
    """Return 'rl' or 'sl' based on the algorithm name."""
    if alg in RL_ALGORITHMS:
        return "rl"
    return "sl"


def _apply_overrides(config: Dict, overrides: Dict) -> Dict:
    """Deep-merge overrides into a config dict."""
    for section, values in overrides.items():
        if section not in config:
            config[section] = {}
        if isinstance(values, dict):
            config[section].update(values)
        else:
            config[section] = values
    return config


def _data_path_for(model_slug: str, task_name: str, alg: str) -> Tuple[str, str]:
    """Return (train_data_path, val_data_path) for an algorithm.

    RL algorithms use prompt-only data.
    SL preference algorithms use pre-generated preference data.
    SFT uses prompt-response data.
    """
    data_dir = _get_data_dir()
    method = _method_for_algorithm(alg)

    if method == "rl":
        # RL uses prompt-only data (the same as the original oxrl data dir)
        # Fall back to the standard data/ dir for prompt-only datasets
        train = f"./data/{task_name}_{model_slug}_wsp_train.parquet"
        val = f"./data/{task_name}_{model_slug}_wsp_test.parquet"
    elif alg == "sft":
        # SFT uses prompt-response data
        train = f"{data_dir}/{task_name}_{model_slug}_sft_train.parquet"
        val = f"{data_dir}/{task_name}_{model_slug}_sft_test.parquet"
    else:
        # Preference algorithms use generated preference data
        train = f"{data_dir}/{task_name}_{model_slug}_prefs_train.parquet"
        val = f"{data_dir}/{task_name}_{model_slug}_prefs_test.parquet"

    return train, val


def generate_experiment_configs(
    experiment_name: str,
    output_dir: str,
    dry_run: bool = False,
) -> List[Dict]:
    """Generate all config files for an experiment grid.

    Returns a list of dicts with metadata about each generated config:
        [{"experiment_id": "...", "config_path": "...", "method": "rl|sl", ...}]
    """
    if experiment_name not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment: {experiment_name}. "
            f"Available: {list(EXPERIMENTS.keys())}"
        )

    exp = EXPERIMENTS[experiment_name]
    models = exp["models"]
    algorithms = exp["algorithms"]
    tasks = exp["tasks"]
    seeds = exp["seeds"]
    override_list = exp.get("overrides", [None])

    os.makedirs(output_dir, exist_ok=True)
    generated = []

    for model_key, alg, task_name, seed in itertools.product(
        models, algorithms, tasks, seeds
    ):
        for override_idx, overrides in enumerate(
            override_list if override_list[0] is not None else [None]
        ):
            model_info = MODELS[model_key]
            model_name = model_info["name"]
            params_b = model_info["params_b"]
            model_slug = model_name.split("/")[-1].lower()
            task_info = TASKS[task_name]

            # Build experiment ID
            exp_id_parts = [experiment_name, alg, model_key, task_name, f"s{seed}"]
            if overrides is not None:
                # Encode override values into the ID
                for section, values in overrides.items():
                    for k, v in values.items():
                        exp_id_parts.append(f"{k}{v}")
            experiment_id = "_".join(str(p) for p in exp_id_parts)
            # Sanitize: remove dots, slashes, etc.
            experiment_id = re.sub(r"[^a-zA-Z0-9_\-]", "", experiment_id)

            method = _method_for_algorithm(alg)
            checkpoint_dir = _get_checkpoint_dir()

            # Generate base config
            config = generate_config(
                model_name=model_name,
                task=task_info["task"],
                param_count_b=params_b,
                data_dir="./data",
                checkpoint_dir=checkpoint_dir,
                experiment_id=experiment_id,
            )

            # Set algorithm
            config["train"]["alg_name"] = alg

            # Set seed
            config["run"]["seed"] = seed

            # Assign unique ray_master_port to avoid port conflicts
            # when running multiple RL jobs in parallel.
            # Hash the experiment_id to get a deterministic port in [29500, 39500).
            port_offset = hash(experiment_id) % 10000
            config["run"]["ray_master_port"] = 29500 + port_offset

            # Set proper data paths
            train_path, val_path = _data_path_for(model_slug, task_name, alg)
            config["data"]["train_files_path"] = train_path
            config["data"]["val_files_path"] = val_path

            # Set reward function
            config["reward"]["reward_func"] = task_info["reward_func"]

            # Set training epochs and steps for the benchmark
            # RL: train_steps_per_epoch = number of FULL PASSES through replay buffer.
            # 1 pass/epoch x 3 epochs = 3 full passes (standard GRPO recipe).
            # SL: overridden below to use micro_batches_per_epoch instead.
            config["train"]["total_number_of_epochs"] = 3
            config["train"]["train_steps_per_epoch"] = 1

            # Algorithm-specific overrides for SL methods
            if method == "sl":
                # SL methods use main_sl.py with micro_batches_per_epoch
                # Set to ~1 full pass per epoch (dataset_size / effective_batch)
                # With ~6400 pairs, batch=2, grad_accum=8 -> effective_batch=16
                # -> ~400 micro_batches per epoch for 1 full pass
                config["train"]["train_steps_per_epoch"] = None
                config["train"]["micro_batches_per_epoch"] = 400

                # Preference methods need ref_model for DPO/KTO/IPO
                if alg not in ("sft", "simpo", "orpo", "cpo", "alphapo",
                               "cposimpo"):
                    config["model"]["ref_model"] = model_name

            # Apply experiment-specific overrides
            if overrides is not None:
                config = _apply_overrides(config, overrides)

            # Write config
            config_dir = os.path.join(output_dir, experiment_id)
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, "config.yaml")

            if not dry_run:
                save_config(config, config_path)

            generated.append({
                "experiment_id": experiment_id,
                "experiment_name": experiment_name,
                "config_path": config_path,
                "method": method,
                "algorithm": alg,
                "model": model_name,
                "model_key": model_key,
                "task": task_name,
                "seed": seed,
                "params_b": params_b,
            })

    return generated


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate experiment sweep configs for NeurIPS 2026."
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help=f"Experiment grid name. Use --list to see options.",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="/home/ec2-user/fsx/oxrl_configs/neurips2026",
        help="Root directory for generated config files.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available experiment grids and exit.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Count configs without writing files.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available experiment grids:")
        print(f"{'Name':<25s} {'Runs':>6s}  Description")
        print("-" * 80)
        for name, exp in EXPERIMENTS.items():
            n_models = len(exp["models"])
            n_algs = len(exp["algorithms"])
            n_tasks = len(exp["tasks"])
            n_seeds = len(exp["seeds"])
            n_overrides = len(exp.get("overrides", [None]))
            n_runs = n_models * n_algs * n_tasks * n_seeds * n_overrides
            print(f"{name:<25s} {n_runs:>6d}  {exp['description']}")
        print()
        total = sum(
            len(e["models"]) * len(e["algorithms"]) * len(e["tasks"])
            * len(e["seeds"]) * len(e.get("overrides", [None]))
            for e in EXPERIMENTS.values()
        )
        print(f"Total runs across all experiments: {total}")
        sys.exit(0)

    if not args.experiment:
        parser.error("--experiment is required (or use --list)")

    # Support "all" to generate everything
    if args.experiment == "all":
        exp_names = list(EXPERIMENTS.keys())
    else:
        exp_names = [args.experiment]

    total_generated = 0
    for exp_name in exp_names:
        print(f"\n{'='*60}")
        print(f"Generating configs for: {exp_name}")
        print(f"{'='*60}")

        configs = generate_experiment_configs(
            experiment_name=exp_name,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
        total_generated += len(configs)

        print(f"Generated {len(configs)} configs")
        if configs:
            print(f"  Models:     {sorted(set(c['model_key'] for c in configs))}")
            print(f"  Algorithms: {sorted(set(c['algorithm'] for c in configs))}")
            print(f"  Tasks:      {sorted(set(c['task'] for c in configs))}")
            print(f"  Seeds:      {sorted(set(c['seed'] for c in configs))}")

    # Write manifest
    if not args.dry_run:
        manifest_path = os.path.join(args.output_dir, "manifest.yaml")
        manifest = {
            "experiments": exp_names,
            "total_configs": total_generated,
            "output_dir": args.output_dir,
        }
        with open(manifest_path, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False)
        print(f"\nManifest written to {manifest_path}")

    print(f"\nTotal configs generated: {total_generated}")


if __name__ == "__main__":
    main()
