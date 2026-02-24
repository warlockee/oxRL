"""
bugfixer.py -- Bug-fixer agent that monitors failed models and applies fixes.

Scans manifest.json for models with status=="failed", reads their train.log,
classifies the failure, and either applies automated config fixes (re-queuing
the model) or marks it as "skipped" when no automated fix is possible.

Failure taxonomy:
    oom              - CUDA out of memory
    chat_template    - Missing chat template in tokenizer
    vllm_load        - vLLM cannot load the model
    reward_zero      - Reward stuck at 0.0 for >10 steps
    nan_loss         - NaN values in training loss
    timeout          - DeepSpeed hang / killed with no error
    pad_token        - Tokenizer padding assertion errors
    config_validation - Pydantic ValidationError in config
    unknown          - Unrecognized failure pattern

Usage:
    python swarm/bugfixer.py                       # Scan and fix all failures
    python swarm/bugfixer.py --model "Qwen/..."    # Fix specific model
    python swarm/bugfixer.py --dry-run              # Show what would be fixed
"""

import argparse
import fcntl
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Resolve project paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

sys.path.insert(0, _PROJECT_ROOT)

from swarm.config_generator import generate_config, save_config, make_slug  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MANIFEST_PATH = os.path.join(_SCRIPT_DIR, "manifest.json")
DEFAULT_ONBOARDED_DIR = os.path.join(_PROJECT_ROOT, "onboarded")
DEFAULT_MAX_RETRIES = 3

# Minimum number of training steps before we diagnose "reward always zero"
_REWARD_ZERO_MIN_STEPS = 10


# =========================================================================
# Manifest I/O (with file locking)
# =========================================================================

def load_manifest(path: str) -> dict:
    """Load manifest.json with a shared (read) file lock."""
    with open(path, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_SH)
        try:
            data = json.load(fh)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    return data


def _save_manifest(path: str, data: dict) -> None:
    """Write manifest.json atomically with an exclusive lock."""
    with open(path, "r+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0)
            json.dump(data, fh, indent=2)
            fh.truncate()
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def update_manifest(path: str, model_name: str, **updates) -> None:
    """Update a model's entry in manifest.json with file locking.

    Parameters
    ----------
    path : str
        Path to manifest.json.
    model_name : str
        The HuggingFace model identifier (key in ``manifest["models"]``).
    **updates
        Arbitrary key-value pairs to merge into the model's dict.
    """
    manifest = load_manifest(path)
    if model_name not in manifest["models"]:
        raise KeyError(f"Model {model_name!r} not found in manifest")

    manifest["models"][model_name].update(updates)

    # Recompute aggregate counters
    statuses = [m["status"] for m in manifest["models"].values()]
    manifest["total_failed"] = statuses.count("failed")
    manifest["total_onboarded"] = statuses.count("completed")

    _save_manifest(path, manifest)
    logger.debug("Manifest updated for %s: %s", model_name, updates)


# =========================================================================
# Log parsing helpers
# =========================================================================

def _read_log(log_path: str) -> str:
    """Read the full contents of a log file. Returns empty string if missing."""
    if not os.path.isfile(log_path):
        logger.warning("Log file does not exist: %s", log_path)
        return ""
    with open(log_path, "r", errors="replace") as fh:
        return fh.read()


def parse_reward_metrics(log_path: str) -> list[float]:
    """Extract all ``reward_mean`` values from a training log.

    Looks for patterns like ``reward_mean: 0.0`` or ``"reward_mean": 0.0``
    in each line.
    """
    values: list[float] = []
    log_text = _read_log(log_path)
    if not log_text:
        return values

    # Match both YAML-style and JSON-style output
    pattern = re.compile(r"reward_mean[\"']?\s*[:=]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
    for match in pattern.finditer(log_text):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue
    return values


def _check_log_is_empty_kill(log_text: str) -> bool:
    """Heuristic: log is very short / has no traceback but the process was
    killed.  Indicates a DeepSpeed hang that hit a timeout."""
    if not log_text.strip():
        return True
    # If there is no Python traceback and the log just ends, treat as timeout.
    has_traceback = "Traceback" in log_text or "Error" in log_text
    return not has_traceback


# =========================================================================
# Failure classification
# =========================================================================

def classify_failure(log_path: str) -> str:
    """Read train.log, return failure type string.

    Returns one of:
        ``'oom'``, ``'chat_template'``, ``'vllm_load'``, ``'reward_zero'``,
        ``'nan_loss'``, ``'timeout'``, ``'pad_token'``, ``'config_validation'``,
        ``'unknown'``
    """
    log_text = _read_log(log_path)

    if not log_text:
        # No log at all -- likely killed before anything was written
        return "timeout"

    # ----- OOM -----
    if "CUDA out of memory" in log_text or "OutOfMemoryError" in log_text:
        return "oom"

    # ----- Chat template -----
    if "apply_chat_template" in log_text or "chat_template" in log_text:
        # Distinguish between a genuine chat-template error and a simple
        # mention.  Look for traceback context.
        if ("Error" in log_text or "Traceback" in log_text) and (
            "apply_chat_template" in log_text or "chat_template" in log_text
        ):
            return "chat_template"

    # ----- vLLM load -----
    if "Cannot load model" in log_text or "Error loading model" in log_text:
        return "vllm_load"

    # ----- Config validation -----
    if "ValidationError" in log_text:
        return "config_validation"

    # ----- Pad token -----
    if ("pad_token" in log_text or "padding" in log_text) and (
        "AssertionError" in log_text or "assertion" in log_text.lower()
    ):
        return "pad_token"

    # ----- NaN loss -----
    nan_pattern = re.compile(r"loss[\"']?\s*[:=]\s*nan", re.IGNORECASE)
    if nan_pattern.search(log_text) or "NaN" in log_text:
        return "nan_loss"

    # ----- Reward always zero -----
    reward_values = parse_reward_metrics(log_path)
    if len(reward_values) >= _REWARD_ZERO_MIN_STEPS and all(v == 0.0 for v in reward_values):
        return "reward_zero"

    # ----- Timeout / DeepSpeed hang -----
    if _check_log_is_empty_kill(log_text):
        return "timeout"

    return "unknown"


# =========================================================================
# Fix generation
# =========================================================================

def get_fix(failure_type: str, model_info: dict) -> dict:
    """Return a fix descriptor dict.

    Parameters
    ----------
    failure_type : str
        One of the failure type strings produced by :func:`classify_failure`.
    model_info : dict
        The model's entry from ``manifest["models"]``.

    Returns
    -------
    dict
        ``action`` is one of ``"skip"``, ``"adjust_config"``,
        ``"patch_framework"``.  ``changes`` carries the specific
        modifications to apply.  ``reason`` is a human-readable note.
    """
    config_path = model_info.get("config_path")

    if failure_type == "oom":
        return _fix_oom(config_path)

    if failure_type == "chat_template":
        return {
            "action": "skip",
            "changes": {},
            "reason": "Chat template missing in tokenizer -- cannot fix automatically.",
        }

    if failure_type == "vllm_load":
        return _fix_vllm_load(config_path)

    if failure_type == "reward_zero":
        return {
            "action": "skip",
            "changes": {},
            "reason": (
                "Reward is 0.0 for all logged steps. "
                "Likely a data preprocessing issue -- check that answers are valid."
            ),
        }

    if failure_type == "nan_loss":
        return _fix_nan_loss(config_path)

    if failure_type == "timeout":
        return _fix_timeout(config_path)

    if failure_type == "pad_token":
        return {
            "action": "skip",
            "changes": {},
            "reason": "Pad-token / padding assertion error -- handled by setup.py at framework level.",
        }

    if failure_type == "config_validation":
        return {
            "action": "patch_framework",
            "changes": {"rerun_config_generator": True},
            "reason": "Pydantic ValidationError -- re-generate config via config_generator.",
        }

    # unknown
    return {
        "action": "skip",
        "changes": {},
        "reason": f"Unknown failure type ({failure_type!r}) -- manual inspection required.",
    }


# ---------------------------------------------------------------------------
# Per-failure fix helpers
# ---------------------------------------------------------------------------

def _load_config_yaml(config_path: Optional[str]) -> dict:
    """Load a YAML config file.  Returns empty dict if path is missing."""
    if not config_path or not os.path.isfile(config_path):
        return {}
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh) or {}


def _fix_oom(config_path: Optional[str]) -> dict:
    """Halve batch sizes.  If already at 1, enable CPU offload."""
    cfg = _load_config_yaml(config_path)
    changes: dict = {}

    train_bs = cfg.get("train", {}).get("train_batch_size_per_gpu", 1)
    rollout_bs = cfg.get("rollout", {}).get("rollout_batch_size_per_gpu", 1)
    n_samples = cfg.get("rollout", {}).get("n_samples", 1)

    all_at_minimum = (train_bs <= 1 and rollout_bs <= 1 and n_samples <= 1)

    if all_at_minimum:
        # Escalate: enable CPU offload
        zero_cfg = cfg.get("deepspeed", {}).get("zero_optimization", {})
        offload_opt = zero_cfg.get("offload_optimizer", {})
        offload_param = zero_cfg.get("offload_param", {})

        opt_device = offload_opt.get("device", "none") if isinstance(offload_opt, dict) else "none"
        param_device = offload_param.get("device", "none") if isinstance(offload_param, dict) else "none"

        if opt_device == "none":
            changes["deepspeed.zero_optimization.offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
            }
        elif param_device == "none":
            changes["deepspeed.zero_optimization.offload_param"] = {
                "device": "cpu",
                "pin_memory": True,
            }
        else:
            # Both already offloaded -- nothing more we can do automatically
            return {
                "action": "skip",
                "changes": {},
                "reason": (
                    "OOM with batch sizes at 1 and full CPU offload already enabled. "
                    "Model may be too large for available GPU memory."
                ),
            }
    else:
        changes["train.train_batch_size_per_gpu"] = max(1, train_bs // 2)
        changes["rollout.rollout_batch_size_per_gpu"] = max(1, rollout_bs // 2)
        changes["rollout.n_samples"] = max(1, n_samples // 2)

    return {
        "action": "adjust_config",
        "changes": changes,
        "reason": "OOM -- reducing batch sizes or enabling CPU offload.",
    }


def _fix_vllm_load(config_path: Optional[str]) -> dict:
    """Set trust_remote_code=True.  If already set, skip."""
    cfg = _load_config_yaml(config_path)
    trust = cfg.get("model", {}).get("trust_remote_code", False)

    if trust:
        return {
            "action": "skip",
            "changes": {},
            "reason": "vLLM load failure with trust_remote_code already True -- skipping.",
        }

    return {
        "action": "adjust_config",
        "changes": {"model.trust_remote_code": True},
        "reason": "vLLM load failure -- enabling trust_remote_code.",
    }


def _fix_nan_loss(config_path: Optional[str]) -> dict:
    """Reduce learning rate by 10x, increase clip_grad_norm to 5.0."""
    cfg = _load_config_yaml(config_path)
    current_lr = cfg.get("train", {}).get("lr", 1e-6)

    return {
        "action": "adjust_config",
        "changes": {
            "train.lr": current_lr / 10.0,
            "train.clip_grad_norm": 5.0,
        },
        "reason": "NaN loss detected -- reducing LR by 10x and raising clip_grad_norm to 5.0.",
    }


def _fix_timeout(config_path: Optional[str]) -> dict:
    """Reduce batch sizes and try different ZeRO settings."""
    cfg = _load_config_yaml(config_path)
    changes: dict = {}

    train_bs = cfg.get("train", {}).get("train_batch_size_per_gpu", 1)
    rollout_bs = cfg.get("rollout", {}).get("rollout_batch_size_per_gpu", 1)

    changes["train.train_batch_size_per_gpu"] = max(1, train_bs // 2)
    changes["rollout.rollout_batch_size_per_gpu"] = max(1, rollout_bs // 2)

    # Try ZeRO stage 2 if currently at stage 3 (reduces communication overhead
    # that can trigger hangs)
    zero_stage = cfg.get("deepspeed", {}).get("zero_optimization", {}).get("stage", 3)
    if zero_stage == 3:
        changes["deepspeed.zero_optimization.stage"] = 2

    return {
        "action": "adjust_config",
        "changes": changes,
        "reason": "Timeout / DeepSpeed hang -- reducing batch sizes and ZeRO stage.",
    }


# =========================================================================
# Config patching
# =========================================================================

def _set_nested(d: dict, dotted_key: str, value) -> None:
    """Set a value in a nested dict using a dot-separated key path.

    Example: ``_set_nested(cfg, "train.lr", 1e-7)`` sets ``cfg["train"]["lr"]``.
    """
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def apply_config_fix(fix: dict, model_info: dict) -> str:
    """Apply fix to config YAML, return path to the updated config file.

    Parameters
    ----------
    fix : dict
        Fix descriptor from :func:`get_fix`.  ``fix["changes"]`` maps
        dotted-key paths (e.g. ``"train.lr"``) to new values.
    model_info : dict
        The model's manifest entry.  Must contain ``config_path``.

    Returns
    -------
    str
        Path to the (overwritten) config YAML file.
    """
    config_path = model_info.get("config_path")
    if not config_path or not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Config file not found for patching: {config_path!r}"
        )

    cfg = _load_config_yaml(config_path)

    for dotted_key, value in fix["changes"].items():
        _set_nested(cfg, dotted_key, value)
        logger.info("  patched %s = %r", dotted_key, value)

    save_config(cfg, config_path)
    logger.info("Config saved: %s", config_path)
    return config_path


# =========================================================================
# Framework-level fix (config re-generation)
# =========================================================================

def _handle_config_validation_fix(model_name: str, model_info: dict, onboarded_dir: str) -> Optional[str]:
    """Re-generate the config via config_generator when a ValidationError occurred.

    Returns the path to the newly written config, or None on failure.
    """
    task = model_info.get("tasks", ["math"])[0]
    param_count_b = model_info.get("param_count_b", 1.0)
    slug = make_slug(model_name)
    config_dir = os.path.join(onboarded_dir, slug)
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "config.yaml")

    try:
        cfg = generate_config(
            model_name=model_name,
            task=task,
            param_count_b=param_count_b,
        )
        save_config(cfg, config_path)
        logger.info("Re-generated config for %s -> %s", model_name, config_path)
        return config_path
    except Exception as exc:
        logger.error("Failed to re-generate config for %s: %s", model_name, exc)
        return None


# =========================================================================
# Framework issue logging
# =========================================================================

def log_framework_issue(fix: dict, model_name: str = "") -> None:
    """Log a framework-level issue that requires manual attention."""
    logger.warning(
        "FRAMEWORK ISSUE for %s: %s | changes=%s",
        model_name,
        fix.get("reason", ""),
        fix.get("changes", {}),
    )


# =========================================================================
# Main execution loop
# =========================================================================

def run(
    manifest_path: str = DEFAULT_MANIFEST_PATH,
    onboarded_dir: str = DEFAULT_ONBOARDED_DIR,
    max_retries: int = DEFAULT_MAX_RETRIES,
    target_model: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """Scan manifest for failed models and apply fixes.

    Parameters
    ----------
    manifest_path : str
        Path to ``manifest.json``.
    onboarded_dir : str
        Root directory containing per-model subdirectories with ``train.log``
        and ``config.yaml`` files.
    max_retries : int
        Maximum number of retry attempts per model before giving up.
    target_model : str, optional
        If given, only process this specific model.
    dry_run : bool
        If True, report planned fixes without modifying anything.

    Returns
    -------
    dict
        Summary: ``{"fixed": [...], "skipped": [...], "errors": [...]}``.
    """
    summary: dict = {"fixed": [], "skipped": [], "errors": []}
    iteration = 0

    while True:
        iteration += 1
        manifest = load_manifest(manifest_path)
        models = manifest.get("models", {})

        # Collect actionable failures
        failed: list[str] = []
        for name, info in models.items():
            if info.get("status") != "failed":
                continue
            if info.get("attempts", 0) >= max_retries:
                continue
            if target_model and name != target_model:
                continue
            failed.append(name)

        if not failed:
            logger.info("No more actionable failures (iteration %d).", iteration)
            break

        progress_made = False

        for model_name in failed:
            # Re-read manifest each iteration in case another model's fix
            # changed it.
            manifest = load_manifest(manifest_path)
            model_info = manifest["models"][model_name]
            attempts = model_info.get("attempts", 0)

            slug = make_slug(model_name)
            log_path = os.path.join(onboarded_dir, slug, "train.log")

            failure_type = classify_failure(log_path)
            fix = get_fix(failure_type, model_info)

            logger.info(
                "[%s] failure=%s action=%s reason=%s (attempt %d/%d)",
                model_name,
                failure_type,
                fix["action"],
                fix["reason"],
                attempts + 1,
                max_retries,
            )

            if dry_run:
                print(f"[DRY-RUN] {model_name}")
                print(f"  failure : {failure_type}")
                print(f"  action  : {fix['action']}")
                print(f"  changes : {fix['changes']}")
                print(f"  reason  : {fix['reason']}")
                print()
                # Record in summary but don't modify anything
                if fix["action"] == "skip":
                    summary["skipped"].append(model_name)
                else:
                    summary["fixed"].append(model_name)
                continue

            # ---- Skip ----
            if fix["action"] == "skip":
                update_manifest(
                    manifest_path,
                    model_name,
                    status="skipped",
                    error_log=fix["reason"],
                )
                summary["skipped"].append(model_name)
                logger.info("Skipped %s: %s", model_name, fix["reason"])
                progress_made = True
                continue

            # ---- Adjust config ----
            if fix["action"] == "adjust_config":
                try:
                    new_config = apply_config_fix(fix, model_info)
                    update_manifest(
                        manifest_path,
                        model_name,
                        status="queued",
                        attempts=attempts + 1,
                        config_path=new_config,
                        error_log=f"Auto-fix applied: {fix['reason']}",
                    )
                    summary["fixed"].append(model_name)
                    logger.info("Re-queued %s (attempt %d)", model_name, attempts + 1)
                    progress_made = True
                except Exception as exc:
                    logger.error("Failed to apply fix to %s: %s", model_name, exc)
                    update_manifest(
                        manifest_path,
                        model_name,
                        status="skipped",
                        error_log=f"Bugfixer error: {exc}",
                    )
                    summary["errors"].append(model_name)
                    progress_made = True
                continue

            # ---- Patch framework ----
            if fix["action"] == "patch_framework":
                log_framework_issue(fix, model_name)

                # For config_validation, attempt re-generation
                if fix["changes"].get("rerun_config_generator"):
                    new_path = _handle_config_validation_fix(
                        model_name, model_info, onboarded_dir
                    )
                    if new_path:
                        update_manifest(
                            manifest_path,
                            model_name,
                            status="queued",
                            attempts=attempts + 1,
                            config_path=new_path,
                            error_log=f"Config re-generated: {fix['reason']}",
                        )
                        summary["fixed"].append(model_name)
                        progress_made = True
                    else:
                        update_manifest(
                            manifest_path,
                            model_name,
                            status="skipped",
                            error_log=f"Config re-generation failed: {fix['reason']}",
                        )
                        summary["errors"].append(model_name)
                        progress_made = True
                else:
                    # Generic framework issue -- skip for now
                    update_manifest(
                        manifest_path,
                        model_name,
                        status="skipped",
                        error_log=f"Framework issue (manual fix needed): {fix['reason']}",
                    )
                    summary["skipped"].append(model_name)
                    progress_made = True
                continue

        # Safety: break if nothing changed in this iteration to avoid infinite
        # loops (e.g. dry-run mode, or all models processed).
        if not progress_made:
            break

    return summary


# =========================================================================
# CLI
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="oxRL Bug-fixer: scan manifest.json for failed models and apply fixes.",
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST_PATH,
        help=f"Path to manifest.json (default: {DEFAULT_MANIFEST_PATH})",
    )
    parser.add_argument(
        "--onboarded-dir",
        default=DEFAULT_ONBOARDED_DIR,
        help=f"Root dir for onboarded models (default: {DEFAULT_ONBOARDED_DIR})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Fix a specific model (HF model id, e.g. 'Qwen/Qwen2.5-0.5B-Instruct')",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Max retry attempts per model (default: {DEFAULT_MAX_RETRIES})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without making changes.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("=" * 60)
    print("oxRL Bug-fixer Agent")
    print("=" * 60)
    print(f"  manifest    : {args.manifest}")
    print(f"  onboarded   : {args.onboarded_dir}")
    print(f"  target model: {args.model or '(all failed)'}")
    print(f"  max retries : {args.max_retries}")
    print(f"  dry run     : {args.dry_run}")
    print("=" * 60)
    print()

    summary = run(
        manifest_path=args.manifest,
        onboarded_dir=args.onboarded_dir,
        max_retries=args.max_retries,
        target_model=args.model,
        dry_run=args.dry_run,
    )

    # Print summary
    print()
    print("-" * 60)
    print("Summary")
    print("-" * 60)
    print(f"  Fixed (re-queued) : {len(summary['fixed'])}")
    for m in summary["fixed"]:
        print(f"    - {m}")
    print(f"  Skipped           : {len(summary['skipped'])}")
    for m in summary["skipped"]:
        print(f"    - {m}")
    print(f"  Errors            : {len(summary['errors'])}")
    for m in summary["errors"]:
        print(f"    - {m}")
    print("-" * 60)


if __name__ == "__main__":
    main()
