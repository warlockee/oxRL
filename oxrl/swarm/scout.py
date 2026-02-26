"""
scout.py — The Scout agent that onboards models into oxRL.

Iterates through the model queue in manifest.json and for each model:
  1. DISCOVER  — Pull model info from HF, verify chat_template, check param count
  2. PREPROCESS — Run the appropriate preprocessing script (skip if data exists)
  3. GENERATE  — Auto-create config.yaml via config_generator
  4. TRAIN     — Launch main_rl.py with timeout
  5. EVALUATE  — Parse training log for reward metrics
  6. ARCHIVE   — Save results.json or mark failure in manifest
  7. GC        — Clean up caches (HF hub, vLLM, DeepSpeed, torch extensions)
  8. UPDATE    — Write status + metrics back to manifest.json

CLI:
    python swarm/scout.py                                        # all queued
    python swarm/scout.py --model "Qwen/Qwen2.5-0.5B-Instruct" --task math
    python swarm/scout.py --tier 1                               # tier 1 only
    python swarm/scout.py --dry-run                              # preview
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
DATA_DIR = Path("/ceph/workspace/oxrl/data")
CHECKPOINT_DIR = Path("/ceph/workspace/oxrl/ckps")
MANIFEST_PATH = PROJECT_ROOT / "oxrl" / "swarm" / "manifest.json"
ONBOARDED_DIR = PROJECT_ROOT / "registry"

# ---------------------------------------------------------------------------
# Imports from sibling modules
# ---------------------------------------------------------------------------
sys.path.insert(0, str(PROJECT_ROOT))

from oxrl.swarm.config_generator import generate_config, save_config, make_slug
from oxrl.swarm.model_registry import check_chat_template, get_model_info

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][scout][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scout")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_ATTEMPTS = 3

# Dataset -> preprocessing script mapping
DATASET_SCRIPT_MAP = {
    "gsm8k": "oxrl/preprocessing/gsm8k.py",
    "math_hard": "oxrl/preprocessing/math_hard.py",
    "mbpp": "oxrl/preprocessing/mbpp.py",
    "ultrafeedback": "oxrl/preprocessing/ultrafeedback.py",
    "vision_dummy": "oxrl/preprocessing/vision_dummy.py",
    "audio_dummy": "oxrl/preprocessing/audio_dummy.py",
    "openr1_math": "oxrl/preprocessing/openr1_math.py",
    "gpqa": "oxrl/preprocessing/gpqa.py",
}

# Timeout per tier (seconds): tier -> timeout_seconds
# <= 1.5B -> 30 min, 3-4B -> 1 hr, 7B -> 3 hr
TIER_TIMEOUT = {
    1: 30 * 60,
    2: 60 * 60,
    3: 3 * 60 * 60,
}


# ===================================================================
# Manifest I/O (atomic writes to avoid races with bugfixer)
# ===================================================================

def _load_manifest() -> dict:
    """Load manifest.json, returning the parsed dict."""
    with open(MANIFEST_PATH, "r") as f:
        return json.load(f)


def _save_manifest(manifest: dict) -> None:
    """Atomically write manifest.json via write-to-temp + rename."""
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=MANIFEST_PATH.parent, suffix=".tmp", prefix=".manifest_"
    )
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, MANIFEST_PATH)
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _update_model_entry(model_id: str, updates: dict) -> None:
    """Load manifest, patch a single model entry, and save atomically."""
    manifest = _load_manifest()
    if model_id in manifest["models"]:
        manifest["models"][model_id].update(updates)
    _save_manifest(manifest)


# ===================================================================
# Step 1: DISCOVER
# ===================================================================

def step_discover(model_id: str, entry: dict) -> dict:
    """Pull model info from HF Hub, verify chat_template, return info dict.

    Raises RuntimeError on any disqualifying issue.
    """
    logger.info("[DISCOVER] Fetching HF info for %s", model_id)
    info = get_model_info(model_id)

    if "error" in info:
        raise RuntimeError(f"HF Hub lookup failed: {info['error']}")

    if not info.get("has_chat_template"):
        raise RuntimeError(f"Model {model_id} has no chat_template in tokenizer")

    param_b = info.get("param_count_b")
    if param_b is None:
        # Fall back to manifest value if HF does not expose param count
        param_b = entry.get("param_count_b")
        if param_b is None:
            raise RuntimeError(f"Cannot determine param count for {model_id}")
        logger.info(
            "[DISCOVER] HF did not report param count; using manifest value %.2fB",
            param_b,
        )
    else:
        logger.info("[DISCOVER] %s — %.2fB params, tier %s", model_id, param_b, info.get("tier"))

    return {
        "param_count_b": param_b,
        "tier": info.get("tier", entry.get("tier", 0)),
        "hf_downloads": info.get("downloads", 0),
    }


# ===================================================================
# Step 2: PREPROCESS
# ===================================================================

def _expected_parquet_paths(dataset: str, model_slug: str) -> tuple[Path, Path]:
    """Return (train_parquet, test_parquet) paths for the given dataset + slug."""
    train = DATA_DIR / f"{dataset}_{model_slug}_wsp_train.parquet"
    test = DATA_DIR / f"{dataset}_{model_slug}_wsp_test.parquet"
    return train, test


def step_preprocess(dataset: str, model_slug: str) -> None:
    """Run the preprocessing script if parquet files do not already exist.

    Raises RuntimeError on subprocess failure.
    """
    train_path, test_path = _expected_parquet_paths(dataset, model_slug)

    if train_path.exists() and test_path.exists():
        logger.info(
            "[PREPROCESS] Data already exists, skipping: %s", train_path.parent
        )
        return

    script = DATASET_SCRIPT_MAP.get(dataset)
    if script is None:
        # Dynamic check: Does a script with this name exist in preprocessing?
        dynamic_script = f"oxrl/preprocessing/{dataset}.py"
        if (PROJECT_ROOT / dynamic_script).exists():
            script = dynamic_script
            logger.info("[PREPROCESS] Found dynamic preprocessor for %s: %s", dataset, script)
        else:
            raise RuntimeError(
                f"MISSING_PREPROCESSOR: No script found for dataset {dataset!r}. "
                f"Please create oxrl/preprocessing/{dataset}.py"
            )

    script_path = PROJECT_ROOT / script
    if not script_path.exists():
        raise RuntimeError(f"Preprocessing script not found: {script_path}")

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    use_system_prompt = "True"
    if "gemma" in model_slug:
        use_system_prompt = "False"

    cmd = [
        sys.executable,
        str(script_path),
        "--local_dir", str(DATA_DIR),
        "--run_id", model_slug,
        "--use_system_prompt", use_system_prompt,
    ]
    logger.info("[PREPROCESS] Running: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=600,  # 10 min preprocessing timeout
    )

    if result.returncode != 0:
        logger.error("[PREPROCESS] STDERR:\n%s", result.stderr)
        raise RuntimeError(
            f"Preprocessing failed (exit {result.returncode}): {result.stderr[-500:]}"
        )

    # Verify the files were actually created
    if not train_path.exists():
        raise RuntimeError(f"Preprocessing completed but train parquet missing: {train_path}")
    if not test_path.exists():
        raise RuntimeError(f"Preprocessing completed but test parquet missing: {test_path}")

    logger.info("[PREPROCESS] Data ready: %s", train_path)


# ===================================================================
# Step 3: GENERATE config
# ===================================================================

def step_generate_config(
    model_id: str, task: str, param_count_b: float, model_slug: str
) -> str:
    """Generate config.yaml and save to onboarded/{model_slug}/config.yaml.

    Returns the absolute path to the saved config file.
    """
    logger.info("[GENERATE] Creating config for %s (task=%s)", model_id, task)

    config = generate_config(
        model_name=model_id,
        task=task,
        param_count_b=param_count_b,
        data_dir=str(DATA_DIR),
        checkpoint_dir=str(CHECKPOINT_DIR),
    )

    out_dir = ONBOARDED_DIR / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.yaml"
    save_config(config, str(config_path))

    logger.info("[GENERATE] Config saved: %s", config_path)
    return str(config_path)


# ===================================================================
# Step 4: TRAIN
# ===================================================================

def _get_timeout(param_count_b: float) -> int:
    """Return timeout in seconds based on model size."""
    if param_count_b <= 1.5:
        return TIER_TIMEOUT[1]
    elif param_count_b <= 4.0:
        return TIER_TIMEOUT[2]
    else:
        return TIER_TIMEOUT[3]


def step_train(config_path: str, model_slug: str, param_count_b: float) -> str:
    """Launch main_rl.py, stream output to both stdout and a log file.

    Returns the path to the training log.
    Raises RuntimeError on training failure or timeout.
    """
    log_dir = ONBOARDED_DIR / model_slug
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"

    timeout = _get_timeout(param_count_b)
    logger.info(
        "[TRAIN] Launching training — config=%s, timeout=%ds", config_path, timeout
    )

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "main_rl.py"),
        "--config-file", config_path,
    ]

    # Inject environment variables to bypass CUDA checks if nvcc is missing
    env = os.environ.copy()
    env["DS_SKIP_CUDA_CHECK"] = "1"

    # Stream output to both log file and stdout in real-time
    start_time = time.monotonic()
    try:
        with open(log_path, "w") as log_fh:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(PROJECT_ROOT),
                env=env,
            )

            for line in process.stdout:
                # Write to log file
                log_fh.write(line)
                log_fh.flush()
                # Also stream to stdout
                sys.stdout.write(line)
                sys.stdout.flush()

                # Check timeout
                elapsed = time.monotonic() - start_time
                if elapsed > timeout:
                    process.kill()
                    process.wait()
                    raise subprocess.TimeoutExpired(cmd, timeout)

            process.wait()

            # Final timeout check (in case the process finished right at the edge)
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)

    except subprocess.TimeoutExpired:
        logger.error("[TRAIN] Training timed out after %ds", timeout)
        raise RuntimeError(f"Training timed out after {timeout}s")

    elapsed = time.monotonic() - start_time
    logger.info("[TRAIN] Training completed in %.0fs — log: %s", elapsed, log_path)
    return str(log_path)


# ===================================================================
# Step 5: EVALUATE
# ===================================================================

def step_evaluate(log_path: str) -> dict:
    """Parse the training log for reward_mean values.

    Returns a dict with:
        success (bool), reward_first (float|None), reward_final (float|None),
        reward_values (list[float]), error (str|None)
    """
    logger.info("[EVALUATE] Parsing log: %s", log_path)

    reward_values = []
    oom_detected = False

    try:
        with open(log_path, "r") as f:
            for line in f:
                # Look for reward metrics in log output
                # Common patterns: "reward_mean: 0.123", "reward_mean=0.123",
                # "'reward_mean': 0.123", "avg_reward=0.123", "avg_reward: 0.123"
                matches = re.findall(
                    r"(?:reward_mean|avg_reward)['\"]?\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                    line,
                )
                for m in matches:
                    try:
                        reward_values.append(float(m))
                    except ValueError:
                        pass

                # Detect OOM
                if "out of memory" in line.lower() or "CUDA OOM" in line:
                    oom_detected = True
    except FileNotFoundError:
        return {
            "success": False,
            "reward_first": None,
            "reward_final": None,
            "reward_values": [],
            "error": f"Log file not found: {log_path}",
        }

    if oom_detected:
        return {
            "success": False,
            "reward_first": reward_values[0] if reward_values else None,
            "reward_final": reward_values[-1] if reward_values else None,
            "reward_values": reward_values,
            "error": "OOM detected during training",
        }

    if not reward_values:
        return {
            "success": False,
            "reward_first": None,
            "reward_final": None,
            "reward_values": [],
            "error": "No reward_mean values found in log",
        }

    reward_first = reward_values[0]
    reward_final = reward_values[-1]
    improved = reward_final > reward_first

    logger.info(
        "[EVALUATE] reward: first=%.4f, final=%.4f, improved=%s (%d data points)",
        reward_first,
        reward_final,
        improved,
        len(reward_values),
    )

    # Consider training successful if it completed and produced reward values.
    # Improvement is noted but not required for onboarding success.
    return {
        "success": True,
        "reward_first": reward_first,
        "reward_final": reward_final,
        "reward_values": reward_values,
        "error": None,
    }


# ===================================================================
# Step 6: ARCHIVE
# ===================================================================

def step_archive(model_id: str, model_slug: str, eval_result: dict,
                 config_path: str, discover_info: dict) -> None:
    """On success, save results.json. On failure, update manifest with error."""
    out_dir = ONBOARDED_DIR / model_slug

    if eval_result["success"]:
        results = {
            "model_id": model_id,
            "model_slug": model_slug,
            "status": "onboarded",
            "config_path": config_path,
            "param_count_b": discover_info.get("param_count_b"),
            "tier": discover_info.get("tier"),
            "reward_first": eval_result["reward_first"],
            "reward_final": eval_result["reward_final"],
            "reward_improvement": (
                eval_result["reward_final"] - eval_result["reward_first"]
                if eval_result["reward_first"] is not None
                else None
            ),
            "reward_values": eval_result["reward_values"],
            "onboarded_at": datetime.now(timezone.utc).isoformat(),
        }
        results_path = out_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
            f.write("\n")
        logger.info("[ARCHIVE] Results saved: %s", results_path)
    else:
        logger.warning(
            "[ARCHIVE] Model %s failed: %s", model_id, eval_result.get("error")
        )


# ===================================================================
# Step 7: GC (Garbage Collection)
# ===================================================================

def step_gc(model_id: str) -> None:
    """Clean up caches after each model run."""
    logger.info("[GC] Cleaning caches for %s", model_id)

    # 1. Delete HF model cache: ~/.cache/huggingface/hub/models--{org}--{model}
    #    HF stores models in directories like models--Qwen--Qwen2.5-0.5B-Instruct
    hf_cache_name = "models--" + model_id.replace("/", "--")
    hf_cache_path = Path.home() / ".cache" / "huggingface" / "hub" / hf_cache_name
    if hf_cache_path.exists():
        logger.info("[GC] Removing HF cache: %s", hf_cache_path)
        shutil.rmtree(hf_cache_path, ignore_errors=True)

    # 2. Delete intermediate checkpoints (keep only final)
    #    Checkpoints live under CHECKPOINT_DIR/{experiment_id}/
    #    We look for directories matching the model slug pattern.
    model_slug = make_slug(model_id)
    for ckp_dir in CHECKPOINT_DIR.glob(f"{model_slug}_*"):
        if ckp_dir.is_dir():
            # Keep the directory itself but remove intermediate checkpoint subdirs
            # Typical pattern: step_100/, step_200/, ..., final/
            subdirs = sorted(ckp_dir.glob("step_*"))
            if len(subdirs) > 1:
                # Keep the last one (highest step number), delete the rest
                for subdir in subdirs[:-1]:
                    logger.info("[GC] Removing intermediate checkpoint: %s", subdir)
                    shutil.rmtree(subdir, ignore_errors=True)

    # 3. Delete vLLM cache
    vllm_cache = Path.home() / ".cache" / "vllm"
    if vllm_cache.exists():
        logger.info("[GC] Removing vLLM cache: %s", vllm_cache)
        shutil.rmtree(vllm_cache, ignore_errors=True)

    # 4. Delete DeepSpeed temp files
    for ds_dir in Path("/tmp").glob("deepspeed_*"):
        logger.info("[GC] Removing DeepSpeed temp: %s", ds_dir)
        shutil.rmtree(ds_dir, ignore_errors=True)

    # 5. Delete torch extensions
    torch_ext = Path("/tmp/torch_extensions")
    if torch_ext.exists():
        logger.info("[GC] Removing torch extensions: %s", torch_ext)
        shutil.rmtree(torch_ext, ignore_errors=True)

    logger.info("[GC] Cache cleanup complete")


# ===================================================================
# Step 8: UPDATE manifest
# ===================================================================

def step_update_manifest(model_id: str, eval_result: dict, config_path: str,
                         discover_info: dict) -> None:
    """Write status + metrics back to manifest.json."""
    now = datetime.now(timezone.utc).isoformat()

    if eval_result["success"]:
        updates = {
            "status": "onboarded",
            "config_path": config_path,
            "param_count_b": discover_info.get("param_count_b"),
            "hf_downloads": discover_info.get("hf_downloads"),
            "tier": discover_info.get("tier"),
            "error_log": None,
            "onboarded_at": now,
        }
        # Update the counters
        manifest = _load_manifest()
        manifest["total_onboarded"] = manifest.get("total_onboarded", 0) + 1
        if model_id in manifest["models"]:
            manifest["models"][model_id].update(updates)
            manifest["models"][model_id]["attempts"] = (
                manifest["models"][model_id].get("attempts", 0) + 1
            )
        _save_manifest(manifest)
        logger.info("[UPDATE] Manifest updated: %s -> onboarded", model_id)
    else:
        error_msg = eval_result.get("error", "Unknown failure")
        manifest = _load_manifest()
        entry = manifest["models"].get(model_id, {})
        attempts = entry.get("attempts", 0) + 1
        updates = {
            "status": "failed",
            "error_log": error_msg,
            "attempts": attempts,
            "param_count_b": discover_info.get("param_count_b", entry.get("param_count_b")),
            "hf_downloads": discover_info.get("hf_downloads"),
            "tier": discover_info.get("tier", entry.get("tier")),
        }
        manifest["total_failed"] = manifest.get("total_failed", 0) + 1
        if model_id in manifest["models"]:
            manifest["models"][model_id].update(updates)
        _save_manifest(manifest)
        logger.info("[UPDATE] Manifest updated: %s -> failed (attempt %d)", model_id, attempts)


# ===================================================================
# Orchestrator: run a single model through the full pipeline
# ===================================================================

def _task_for_entry(entry: dict) -> str:
    """Determine the task string from a manifest entry.

    The manifest stores 'tasks' as a list and 'dataset' / 'reward_func' directly.
    We reverse-map from the dataset to the config_generator task key.
    """
    from oxrl.swarm.config_generator import TASK_MAP

    dataset = entry.get("dataset", "")
    reward_func = entry.get("reward_func", "")

    # Direct reverse lookup: find the task whose dataset and reward_func match
    for task_name, task_info in TASK_MAP.items():
        if task_info["dataset"] == dataset and task_info["reward_func"] == reward_func:
            return task_name

    # Fallback: use the first item in "tasks" list
    tasks = entry.get("tasks", [])
    if tasks:
        return tasks[0]

    raise RuntimeError(f"Cannot determine task for entry: {entry}")


def onboard_model(model_id: str, entry: dict, dry_run: bool = False) -> bool:
    """Run the full onboarding pipeline for a single model.

    Returns True if the model was successfully onboarded, False otherwise.
    """
    model_slug = make_slug(model_id)
    dataset = entry.get("dataset", "gsm8k")
    task = _task_for_entry(entry)

    logger.info("=" * 70)
    logger.info("ONBOARDING: %s (slug=%s, task=%s, dataset=%s)", model_id, model_slug, task, dataset)
    logger.info("=" * 70)

    if dry_run:
        logger.info("[DRY-RUN] Would process %s (tier %s, %.2fB params)",
                     model_id, entry.get("tier"), entry.get("param_count_b", 0))
        train_pq, test_pq = _expected_parquet_paths(dataset, model_slug)
        logger.info("[DRY-RUN] Data: train=%s (exists=%s), test=%s (exists=%s)",
                     train_pq, train_pq.exists(), test_pq, test_pq.exists())
        config_dir = ONBOARDED_DIR / model_slug / "config.yaml"
        logger.info("[DRY-RUN] Config would be saved to: %s", config_dir)
        logger.info("[DRY-RUN] Timeout: %ds", _get_timeout(entry.get("param_count_b", 0)))
        return True

    # Track discovery info across steps; initialize with manifest defaults
    discover_info = {
        "param_count_b": entry.get("param_count_b"),
        "tier": entry.get("tier"),
        "hf_downloads": entry.get("hf_downloads"),
    }
    config_path = None

    try:
        # --- Step 1: DISCOVER ---
        discover_info = step_discover(model_id, entry)
        param_count_b = discover_info["param_count_b"]

        # Mark as running
        _update_model_entry(model_id, {"status": "running"})

        # --- Step 2: PREPROCESS ---
        step_preprocess(dataset, model_slug)

        # --- Step 3: GENERATE ---
        config_path = step_generate_config(model_id, task, param_count_b, model_slug)

        # Update manifest with config path
        _update_model_entry(model_id, {"config_path": config_path})

        # --- Step 4: TRAIN ---
        log_path = step_train(config_path, model_slug, param_count_b)

        # --- Step 5: EVALUATE ---
        eval_result = step_evaluate(log_path)

    except Exception as exc:
        # Any exception in steps 1-5 is a failure
        logger.error("Pipeline failed for %s: %s", model_id, exc, exc_info=True)
        eval_result = {
            "success": False,
            "reward_first": None,
            "reward_final": None,
            "reward_values": [],
            "error": str(exc),
        }

    # --- Step 6: ARCHIVE ---
    step_archive(model_id, model_slug, eval_result, config_path or "", discover_info)

    # --- Step 7: GC ---
    try:
        step_gc(model_id)
    except Exception as exc:
        logger.warning("[GC] Cleanup failed (non-fatal): %s", exc)

    # --- Step 8: UPDATE ---
    step_update_manifest(model_id, eval_result, config_path or "", discover_info)

    if eval_result["success"]:
        logger.info("ONBOARDED: %s", model_id)
    else:
        logger.warning("FAILED: %s — %s", model_id, eval_result.get("error"))
        
        # New: Framework bug detection signal
        from oxrl.swarm.bug_reporter import is_framework_error
        if is_framework_error(eval_result.get("error")):
            logger.error("=" * 70)
            logger.error("🚨 FRAMEWORK_BUG_DETECTED: This failure looks like a bug in oxRL.")
            logger.error("AI AGENT ACTION: Please run `oxrl report --model %s` and submit a GitHub issue.", model_id)
            logger.error("=" * 70)

    return eval_result["success"]


# ===================================================================
# Queue builder: select which models to process
# ===================================================================

def build_queue(
    manifest: dict,
    model_filter: Optional[str] = None,
    tier_filter: Optional[int] = None,
    task_filter: Optional[str] = None,
) -> list[tuple[str, dict]]:
    """Build an ordered list of (model_id, entry) to process.

    Respects:
    - Only models with status="queued" and attempts < MAX_ATTEMPTS
    - Optional filters for specific model, tier, or task
    - Ordering: tier 1 first, then 2, then 3; within each tier, manifest order
    """
    queue = []

    for model_id, entry in manifest.get("models", {}).items():
        # Status filter: only process queued (or failed with retries left)
        status = entry.get("status", "queued")
        attempts = entry.get("attempts", 0)

        if status not in ("queued", "failed"):
            continue
        if attempts >= MAX_ATTEMPTS:
            continue

        # Model filter
        if model_filter and model_id != model_filter:
            continue

        # Tier filter
        if tier_filter is not None and entry.get("tier") != tier_filter:
            continue

        # Task filter: check both the 'tasks' list and 'dataset' field
        if task_filter:
            tasks = entry.get("tasks", [])
            dataset = entry.get("dataset", "")
            if task_filter not in tasks and task_filter != dataset:
                continue

        queue.append((model_id, entry))

    # Sort by tier (ascending), then by original manifest order (stable sort)
    queue.sort(key=lambda item: item[1].get("tier", 99))

    return queue


# ===================================================================
# Main entry point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Scout agent — onboard models into oxRL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python swarm/scout.py                                        # all queued models
  python swarm/scout.py --model "Qwen/Qwen2.5-0.5B-Instruct" --task math
  python swarm/scout.py --tier 1                               # tier 1 only
  python swarm/scout.py --dry-run                              # preview
""",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Process only this specific model (HF model ID)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Filter by task (math, math-hard, code, instruct, reasoning)",
    )
    parser.add_argument(
        "--tier",
        type=int,
        default=None,
        choices=[1, 2, 3],
        help="Process only models of this tier",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be done without executing anything",
    )
    args = parser.parse_args()

    logger.info("Scout agent starting")
    logger.info("  Project root : %s", PROJECT_ROOT)
    logger.info("  Data dir     : %s", DATA_DIR)
    logger.info("  Checkpoint   : %s", CHECKPOINT_DIR)
    logger.info("  Manifest     : %s", MANIFEST_PATH)
    logger.info("  Onboarded dir: %s", ONBOARDED_DIR)

    # Load manifest
    if not MANIFEST_PATH.exists():
        logger.error("Manifest not found: %s", MANIFEST_PATH)
        sys.exit(1)

    manifest = _load_manifest()

    # Build queue
    queue = build_queue(
        manifest,
        model_filter=args.model,
        tier_filter=args.tier,
        task_filter=args.task,
    )

    if not queue:
        logger.info("No models to process. Queue is empty.")
        return

    logger.info("Queue: %d model(s) to process", len(queue))
    for i, (mid, entry) in enumerate(queue, 1):
        logger.info(
            "  %2d. %-50s tier=%s  params=%.2fB  dataset=%s  attempts=%d",
            i,
            mid,
            entry.get("tier"),
            entry.get("param_count_b", 0),
            entry.get("dataset"),
            entry.get("attempts", 0),
        )

    if args.dry_run:
        logger.info("-" * 70)
        logger.info("DRY RUN — walking through pipeline without execution")
        logger.info("-" * 70)

    # Ensure onboarded directory exists
    ONBOARDED_DIR.mkdir(parents=True, exist_ok=True)

    # Process models sequentially
    results = {"onboarded": 0, "failed": 0, "skipped": 0}

    for model_id, entry in queue:
        # Re-check attempts from manifest (bugfixer may have updated it)
        fresh_manifest = _load_manifest()
        fresh_entry = fresh_manifest.get("models", {}).get(model_id, entry)
        if fresh_entry.get("attempts", 0) >= MAX_ATTEMPTS:
            logger.info("Skipping %s — max attempts (%d) reached", model_id, MAX_ATTEMPTS)
            results["skipped"] += 1
            continue
        if fresh_entry.get("status") == "onboarded":
            logger.info("Skipping %s — already onboarded", model_id)
            results["skipped"] += 1
            continue

        success = onboard_model(model_id, fresh_entry, dry_run=args.dry_run)

        if args.dry_run:
            continue

        if success:
            results["onboarded"] += 1
        else:
            results["failed"] += 1

    # Summary
    logger.info("=" * 70)
    logger.info("Scout run complete")
    logger.info(
        "  Onboarded: %d | Failed: %d | Skipped: %d",
        results["onboarded"],
        results["failed"],
        results["skipped"],
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
