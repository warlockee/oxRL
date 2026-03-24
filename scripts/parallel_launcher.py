#!/usr/bin/env python3
"""
Parallel experiment launcher for 8xH100 single-node.

Schedules RL jobs (2 GPUs each) and SL jobs (1 GPU each) in parallel,
maximizing GPU utilization. Includes eval after each completed training run.

Usage:
    # Launch all 0.5B GSM8K experiments:
    python scripts/parallel_launcher.py --experiment core_math_gsm8k --model qwen05b

    # Launch everything:
    python scripts/parallel_launcher.py --experiment all

    # Dry run (show what would be launched):
    python scripts/parallel_launcher.py --experiment core_math_gsm8k --dry-run
"""
import argparse
import json
import os
import subprocess
import sys
import time
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

CONFIG_DIR = "/home/ec2-user/fsx/oxrl_configs/neurips2026"
RESULTS_DIR = "/home/ec2-user/fsx/oxrl_results/neurips2026"
LOG_DIR = "/home/ec2-user/fsx/oxrl_logs/neurips2026"
PYTHON = "/opt/pytorch/bin/python3"
WORK_DIR = "/home/ec2-user/fsx/oxRL"

TOTAL_GPUS = 8
RL_ALGS = {"sgrpo", "gspo", "cispo", "ppo"}


# ──────────────────────────────────────────────────────────────────────
# Job discovery
# ──────────────────────────────────────────────────────────────────────

def find_jobs(
    config_dir: str,
    experiment_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
) -> List[Dict]:
    """Find experiment configs and classify as RL (2 GPU) or SL (1 GPU)."""
    jobs = []
    root = Path(config_dir)
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        config_path = subdir / "config.yaml"
        if not config_path.exists():
            continue

        exp_id = subdir.name

        if experiment_filter and experiment_filter != "all":
            if not exp_id.startswith(experiment_filter):
                continue
        if model_filter:
            if model_filter not in exp_id:
                continue

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        alg = cfg.get("train", {}).get("alg_name", "").lower()
        is_rl = alg in RL_ALGS
        port = cfg.get("run", {}).get("ray_master_port", 29500)

        if is_rl:
            # RL jobs: read GPU allocation from config.
            # Physical GPUs needed = training_gpus (rollout engines colocate
            # via fractional Ray GPU allocation when needed).
            training_gpus = cfg.get("run", {}).get("training_gpus", 2)
            gpus_needed = training_gpus
        else:
            gpus_needed = 1

        jobs.append({
            "experiment_id": exp_id,
            "config_path": str(config_path),
            "algorithm": alg,
            "is_rl": is_rl,
            "gpus_needed": gpus_needed,
            "port": port,
        })
    return jobs


def is_completed(exp_id: str) -> bool:
    """Check if experiment has a completion marker."""
    marker = Path(RESULTS_DIR) / exp_id / "metrics.json"
    if not marker.exists():
        return False
    try:
        with open(marker) as f:
            data = json.load(f)
        return data.get("status") == "completed"
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────
# GPU scheduler
# ──────────────────────────────────────────────────────────────────────

class GPUScheduler:
    """Thread-safe GPU allocation for parallel job execution."""

    def __init__(self, num_gpus: int = 8):
        self.lock = threading.Lock()
        self.available = list(range(num_gpus))  # [0,1,2,...,7]

    def allocate(self, n: int) -> Optional[List[int]]:
        """Try to allocate n contiguous GPUs. Returns list of GPU IDs or None."""
        with self.lock:
            if len(self.available) < n:
                return None
            gpus = self.available[:n]
            self.available = self.available[n:]
            return gpus

    def release(self, gpus: List[int]):
        """Return GPUs to the pool."""
        with self.lock:
            self.available.extend(gpus)
            self.available.sort()

    def free_count(self) -> int:
        with self.lock:
            return len(self.available)


# ──────────────────────────────────────────────────────────────────────
# Job execution
# ──────────────────────────────────────────────────────────────────────

def run_job(job: Dict, gpus: List[int], scheduler: GPUScheduler, stats: Dict):
    """Run a single training job on assigned GPUs. Called in a thread."""
    exp_id = job["experiment_id"]
    config_path = job["config_path"]
    is_rl = job["is_rl"]
    gpu_str = ",".join(str(g) for g in gpus)

    log_path = os.path.join(LOG_DIR, f"{exp_id}.log")
    os.makedirs(LOG_DIR, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_str

    if is_rl:
        cmd = [
            PYTHON, "-m", "oxrl.main_rl",
            "--config-file", config_path,
            "--experiment_id", exp_id,
        ]
    else:
        # SL: use deepspeed launcher with unique port and specific GPU
        port = job["port"]
        gpu_include = ",".join(str(g) for g in gpus)
        env["MASTER_PORT"] = str(port)
        env["MASTER_ADDR"] = "127.0.0.1"
        cmd = [
            PYTHON, "-m", "deepspeed.launcher.runner",
            f"--include=localhost:{gpu_include}",
            "--master_port", str(port),
            "--module",
            "oxrl.main_sl",
            "--config-file", config_path,
            "--experiment_id", exp_id,
        ]

    t0 = time.time()
    status = "failed"
    try:
        with open(log_path, "w") as lf:
            result = subprocess.run(
                cmd, stdout=lf, stderr=subprocess.STDOUT,
                env=env, cwd=WORK_DIR,
                timeout=14400,  # 4 hour timeout
            )
        elapsed = time.time() - t0

        if result.returncode == 0:
            status = "completed"
            # Write completion marker
            marker_dir = os.path.join(RESULTS_DIR, exp_id)
            os.makedirs(marker_dir, exist_ok=True)
            with open(os.path.join(marker_dir, "metrics.json"), "w") as f:
                json.dump({
                    "experiment_id": exp_id,
                    "status": "completed",
                    "elapsed_seconds": round(elapsed, 2),
                    "gpus": gpus,
                }, f, indent=2)
        else:
            status = f"failed(rc={result.returncode})"

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        status = "timeout"
    except Exception as e:
        elapsed = time.time() - t0
        status = f"error({e})"

    # Log result
    elapsed = time.time() - t0
    with stats["lock"]:
        stats["completed" if status == "completed" else "failed"] += 1
        total_done = stats["completed"] + stats["failed"]
        total = stats["total"]

    tag = "OK" if status == "completed" else "FAIL"
    print(f"  [{tag}] [{total_done}/{total}] {exp_id} gpu={gpu_str} {elapsed:.0f}s {status}")

    # Release GPUs
    scheduler.release(gpus)


# ──────────────────────────────────────────────────────────────────────
# Main orchestrator
# ──────────────────────────────────────────────────────────────────────

def run_parallel(jobs: List[Dict], skip_completed: bool = True):
    """Run all jobs in parallel, respecting GPU constraints."""
    if skip_completed:
        pending = [j for j in jobs if not is_completed(j["experiment_id"])]
        skipped = len(jobs) - len(pending)
        if skipped > 0:
            print(f"  Skipping {skipped} already-completed jobs")
    else:
        pending = list(jobs)

    if not pending:
        print("  No pending jobs")
        return

    # Sort: SL jobs first (quick, 1 GPU), then RL jobs (slow, 2 GPUs)
    # This fills GPUs faster at the start
    pending.sort(key=lambda j: (j["is_rl"], j["experiment_id"]))

    scheduler = GPUScheduler(TOTAL_GPUS)
    stats = {
        "lock": threading.Lock(),
        "total": len(pending),
        "completed": 0,
        "failed": 0,
    }
    threads = []

    print(f"\n{'='*60}")
    print(f"Launching {len(pending)} jobs ({sum(1 for j in pending if not j['is_rl'])} SL + {sum(1 for j in pending if j['is_rl'])} RL)")
    print(f"{'='*60}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    job_idx = 0
    while job_idx < len(pending) or any(t.is_alive() for t in threads):
        # Try to launch the next job if GPUs are available
        while job_idx < len(pending):
            job = pending[job_idx]
            gpus = scheduler.allocate(job["gpus_needed"])
            if gpus is None:
                break  # No GPUs available, wait

            job_idx += 1
            t = threading.Thread(
                target=run_job,
                args=(job, gpus, scheduler, stats),
                daemon=True,
            )
            t.start()
            threads.append(t)
            gpu_str = ",".join(str(g) for g in gpus)
            print(f"  [START] {job['experiment_id']} gpu={gpu_str} alg={job['algorithm']}")

        # Clean up finished threads and wait briefly
        threads = [t for t in threads if t.is_alive()]
        time.sleep(2)

    # Wait for all threads
    for t in threads:
        t.join()

    print(f"\n{'='*60}")
    print(f"DONE: {stats['completed']} completed, {stats['failed']} failed out of {stats['total']}")
    print(f"{'='*60}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parallel experiment launcher for 8xH100")
    parser.add_argument("--experiment", type=str, default="all",
                        help="Experiment prefix filter (e.g., core_math_gsm8k) or 'all'")
    parser.add_argument("--model", type=str, default=None,
                        help="Model filter (e.g., qwen05b, qwen15b)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be launched without running")
    parser.add_argument("--skip-completed", action="store_true", default=True,
                        help="Skip already-completed experiments")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-run even completed experiments")
    args = parser.parse_args()

    skip = not args.no_skip

    jobs = find_jobs(CONFIG_DIR, args.experiment, args.model)
    if not jobs:
        print(f"No jobs found for experiment={args.experiment} model={args.model}")
        sys.exit(1)

    rl_jobs = [j for j in jobs if j["is_rl"]]
    sl_jobs = [j for j in jobs if not j["is_rl"]]
    pending = [j for j in jobs if not is_completed(j["experiment_id"])] if skip else jobs

    print(f"Found {len(jobs)} total jobs: {len(sl_jobs)} SL + {len(rl_jobs)} RL")
    print(f"Pending: {len(pending)} (skipping {len(jobs) - len(pending)} completed)")
    print(f"Algorithms: {sorted(set(j['algorithm'] for j in jobs))}")

    if args.dry_run:
        print("\nDry run — would launch:")
        for j in pending:
            print(f"  {j['experiment_id']}  alg={j['algorithm']}  gpus={j['gpus_needed']}")
        return

    run_parallel(jobs, skip_completed=skip)


if __name__ == "__main__":
    main()
