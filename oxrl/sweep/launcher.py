"""
Experiment launcher for NeurIPS 2026 sweeps.

Supports two modes:
    1. Sequential: Run experiments one at a time on the current machine.
    2. SLURM: Submit each experiment as a SLURM job.

Usage:
    # Sequential launch (for debugging or single-node):
    python -m oxrl.sweep.launcher \
        --config-dir /home/ec2-user/fsx/oxrl_configs/neurips2026 \
        --mode sequential \
        --experiment core_math_gsm8k

    # SLURM launch:
    python -m oxrl.sweep.launcher \
        --config-dir /home/ec2-user/fsx/oxrl_configs/neurips2026 \
        --mode slurm \
        --experiment core_math_gsm8k \
        --partition gpu \
        --gpus-per-node 8 \
        --time 04:00:00

    # Resume from a specific experiment (skip completed ones):
    python -m oxrl.sweep.launcher \
        --config-dir /home/ec2-user/fsx/oxrl_configs/neurips2026 \
        --mode sequential \
        --experiment core_math_gsm8k \
        --skip-completed
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# ──────────────────────────────────────────────────────────────────────
# Config discovery
# ──────────────────────────────────────────────────────────────────────

def find_configs(config_dir: str, experiment_filter: Optional[str] = None) -> List[Dict]:
    """Find all experiment configs under a config directory.

    Returns list of dicts with config_path, experiment_id, and method.
    """
    configs = []
    root = Path(config_dir)

    if not root.exists():
        print(f"[launcher] ERROR: Config directory does not exist: {config_dir}")
        return configs

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue

        config_path = subdir / "config.yaml"
        if not config_path.exists():
            continue

        experiment_id = subdir.name

        # Apply experiment filter
        if experiment_filter and not experiment_id.startswith(experiment_filter):
            continue

        # Determine method (rl or sl) from the config
        with open(config_path) as f:
            config = yaml.safe_load(f)

        alg_name = config.get("train", {}).get("alg_name", "")
        method = _infer_method(alg_name)

        configs.append({
            "experiment_id": experiment_id,
            "config_path": str(config_path),
            "method": method,
            "algorithm": alg_name,
        })

    return configs


def _infer_method(alg_name: str) -> str:
    """Infer whether an algorithm uses main_rl.py or main_sl.py."""
    rl_algs = {"sgrpo", "gspo", "cispo", "ppo", "rlhf", "rlaif"}
    return "rl" if alg_name.lower() in rl_algs else "sl"


# ──────────────────────────────────────────────────────────────────────
# Completion tracking
# ──────────────────────────────────────────────────────────────────────

def is_completed(experiment_id: str, results_dir: str) -> bool:
    """Check if an experiment has already completed by looking for results."""
    result_path = Path(results_dir) / experiment_id / "metrics.json"
    return result_path.exists()


# ──────────────────────────────────────────────────────────────────────
# Sequential launcher
# ──────────────────────────────────────────────────────────────────────

def launch_sequential(
    configs: List[Dict],
    results_dir: str,
    skip_completed: bool = False,
    log_dir: Optional[str] = None,
) -> Dict:
    """Run experiments one at a time on the current machine.

    Returns summary dict with counts of succeeded, failed, skipped.
    """
    total = len(configs)
    succeeded = 0
    failed = 0
    skipped = 0
    failures = []

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    for i, config_info in enumerate(configs):
        exp_id = config_info["experiment_id"]
        config_path = config_info["config_path"]
        method = config_info["method"]

        # Skip completed experiments
        if skip_completed and is_completed(exp_id, results_dir):
            print(f"[{i+1}/{total}] SKIP (completed): {exp_id}")
            skipped += 1
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] Launching: {exp_id}")
        print(f"  Config:    {config_path}")
        print(f"  Method:    {method}")
        print(f"  Algorithm: {config_info['algorithm']}")
        print(f"{'='*60}")

        # Build command and env
        env = os.environ.copy()
        if method == "rl":
            cmd = [
                sys.executable, "-m", "oxrl.main_rl",
                "--config-file", config_path,
                "--experiment_id", exp_id,
            ]
        else:
            # SL uses DeepSpeed distributed training.
            # Each job needs a unique MASTER_PORT to avoid EADDRINUSE
            # when running multiple SL jobs in parallel.
            # Use the ray_master_port from config to derive a unique port.
            with open(config_path) as _cf:
                _cfg = yaml.safe_load(_cf)
            sl_port = _cfg.get("run", {}).get("ray_master_port", 29500)
            env["MASTER_PORT"] = str(sl_port)
            env["MASTER_ADDR"] = "127.0.0.1"
            cmd = [
                sys.executable, "-m", "deepspeed.launcher.runner",
                "--module",
                "--master_port", str(sl_port),
                "oxrl.main_sl",
                "--config-file", config_path,
                "--experiment_id", exp_id,
            ]

        # Log file
        log_file = None
        if log_dir:
            log_path = os.path.join(log_dir, f"{exp_id}.log")
            log_file = open(log_path, "w")

        t0 = time.time()
        try:
            result = subprocess.run(
                cmd,
                stdout=log_file or sys.stdout,
                stderr=subprocess.STDOUT if log_file else sys.stderr,
                env=env,
                timeout=7200,  # 2 hour timeout per run
            )
            elapsed = time.time() - t0

            if result.returncode == 0:
                print(f"  SUCCESS ({elapsed:.0f}s)")
                succeeded += 1

                # Write completion marker
                marker_dir = os.path.join(results_dir, exp_id)
                os.makedirs(marker_dir, exist_ok=True)
                marker = {
                    "experiment_id": exp_id,
                    "status": "completed",
                    "elapsed_seconds": round(elapsed, 2),
                    "config_path": config_path,
                }
                with open(os.path.join(marker_dir, "metrics.json"), "w") as f:
                    json.dump(marker, f, indent=2)
            else:
                print(f"  FAILED (exit code {result.returncode}, {elapsed:.0f}s)")
                failed += 1
                failures.append(exp_id)

        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"  TIMEOUT after {elapsed:.0f}s")
            failed += 1
            failures.append(exp_id)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR: {e} ({elapsed:.0f}s)")
            failed += 1
            failures.append(exp_id)
        finally:
            if log_file:
                log_file.close()

    return {
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "failures": failures,
    }


# ──────────────────────────────────────────────────────────────────────
# SLURM launcher
# ──────────────────────────────────────────────────────────────────────

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{job_name}.out
#SBATCH --error={log_dir}/{job_name}.err
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
{extra_sbatch}

# Environment setup
source ~/.bashrc
{conda_activate}

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export DS_SKIP_CUDA_CHECK=1

echo "Job: {job_name}"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Start: $(date)"

# Run experiment
{command}

echo "End: $(date)"
echo "Exit code: $?"
"""


def launch_slurm(
    configs: List[Dict],
    results_dir: str,
    skip_completed: bool = False,
    log_dir: Optional[str] = None,
    partition: str = "gpu",
    gpus_per_node: int = 8,
    cpus_per_task: int = 16,
    mem: str = "200G",
    time_limit: str = "04:00:00",
    conda_env: Optional[str] = None,
    extra_sbatch: str = "",
) -> Dict:
    """Submit each experiment as a SLURM job.

    Returns summary dict with counts and job IDs.
    """
    script_dir = os.path.join(log_dir or "/tmp/oxrl_slurm", "scripts")
    os.makedirs(script_dir, exist_ok=True)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    total = len(configs)
    submitted = 0
    skipped = 0
    job_ids = []

    conda_activate = f"conda activate {conda_env}" if conda_env else ""

    for i, config_info in enumerate(configs):
        exp_id = config_info["experiment_id"]
        config_path = config_info["config_path"]
        method = config_info["method"]

        if skip_completed and is_completed(exp_id, results_dir):
            skipped += 1
            continue

        # Build training command
        if method == "rl":
            command = (
                f"{sys.executable} -m oxrl.main_rl "
                f"--config-file {config_path} "
                f"--experiment_id {exp_id}"
            )
        else:
            command = (
                f"{sys.executable} -m oxrl.main_sl "
                f"--config-file {config_path} "
                f"--experiment_id {exp_id}"
            )

        # Generate SLURM script
        script_content = SLURM_TEMPLATE.format(
            job_name=exp_id,
            log_dir=log_dir or "/tmp/oxrl_slurm",
            partition=partition,
            gpus_per_node=gpus_per_node,
            cpus_per_task=cpus_per_task,
            mem=mem,
            time_limit=time_limit,
            extra_sbatch=extra_sbatch,
            conda_activate=conda_activate,
            command=command,
        )

        script_path = os.path.join(script_dir, f"{exp_id}.sh")
        with open(script_path, "w") as f:
            f.write(script_content)

        # Submit
        try:
            result = subprocess.run(
                ["sbatch", script_path],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                # Parse job ID from "Submitted batch job 12345"
                job_id = result.stdout.strip().split()[-1]
                job_ids.append(job_id)
                submitted += 1
                print(f"[{i+1}/{total}] Submitted {exp_id} -> Job {job_id}")
            else:
                print(f"[{i+1}/{total}] FAILED to submit {exp_id}: {result.stderr.strip()}")
        except FileNotFoundError:
            print("[launcher] ERROR: sbatch not found. Is SLURM available?")
            sys.exit(1)
        except Exception as e:
            print(f"[{i+1}/{total}] ERROR submitting {exp_id}: {e}")

    return {
        "total": total,
        "submitted": submitted,
        "skipped": skipped,
        "job_ids": job_ids,
    }


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Launch NeurIPS 2026 experiment sweeps."
    )
    parser.add_argument(
        "--config-dir", type=str, required=True,
        help="Directory containing experiment configs (from sweep.py).",
    )
    parser.add_argument(
        "--mode", type=str, choices=["sequential", "slurm"], default="sequential",
        help="Launch mode: sequential (local) or slurm.",
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Filter configs by experiment name prefix.",
    )
    parser.add_argument(
        "--results-dir", type=str,
        default="/home/ec2-user/fsx/oxrl_results/neurips2026",
        help="Directory for tracking completed experiments.",
    )
    parser.add_argument(
        "--log-dir", type=str,
        default="/home/ec2-user/fsx/oxrl_logs/neurips2026",
        help="Directory for experiment logs.",
    )
    parser.add_argument(
        "--skip-completed", action="store_true",
        help="Skip experiments that have already completed.",
    )

    # SLURM-specific args
    parser.add_argument("--partition", type=str, default="gpu")
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--cpus-per-task", type=int, default=16)
    parser.add_argument("--mem", type=str, default="200G")
    parser.add_argument("--time", type=str, default="04:00:00")
    parser.add_argument("--conda-env", type=str, default=None)
    parser.add_argument("--extra-sbatch", type=str, default="")

    args = parser.parse_args()

    # Find configs
    configs = find_configs(args.config_dir, experiment_filter=args.experiment)
    if not configs:
        print(f"[launcher] No configs found in {args.config_dir}")
        if args.experiment:
            print(f"  (filtered by prefix: {args.experiment})")
        sys.exit(1)

    print(f"[launcher] Found {len(configs)} experiment configs")
    print(f"  Algorithms: {sorted(set(c['algorithm'] for c in configs))}")
    print(f"  Methods:    {sorted(set(c['method'] for c in configs))}")

    os.makedirs(args.results_dir, exist_ok=True)

    # Launch
    if args.mode == "sequential":
        summary = launch_sequential(
            configs=configs,
            results_dir=args.results_dir,
            skip_completed=args.skip_completed,
            log_dir=args.log_dir,
        )
        print(f"\n{'='*60}")
        print(f"[launcher] Sequential launch complete")
        print(f"  Total:     {summary['total']}")
        print(f"  Succeeded: {summary['succeeded']}")
        print(f"  Failed:    {summary['failed']}")
        print(f"  Skipped:   {summary['skipped']}")
        if summary['failures']:
            print(f"  Failed experiments:")
            for f in summary['failures']:
                print(f"    - {f}")

    elif args.mode == "slurm":
        summary = launch_slurm(
            configs=configs,
            results_dir=args.results_dir,
            skip_completed=args.skip_completed,
            log_dir=args.log_dir,
            partition=args.partition,
            gpus_per_node=args.gpus_per_node,
            cpus_per_task=args.cpus_per_task,
            mem=args.mem,
            time_limit=args.time,
            conda_env=args.conda_env,
            extra_sbatch=args.extra_sbatch,
        )
        print(f"\n{'='*60}")
        print(f"[launcher] SLURM launch complete")
        print(f"  Total:     {summary['total']}")
        print(f"  Submitted: {summary['submitted']}")
        print(f"  Skipped:   {summary['skipped']}")
        if summary['job_ids']:
            print(f"  Job IDs:   {', '.join(summary['job_ids'][:10])}")
            if len(summary['job_ids']) > 10:
                print(f"             ... and {len(summary['job_ids'])-10} more")


if __name__ == "__main__":
    main()
