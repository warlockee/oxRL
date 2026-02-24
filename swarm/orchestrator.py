"""
orchestrator.py -- Top-level orchestrator that dispatches scout and bugfixer agents.

Reads manifest.json, runs the scout on queued models, dispatches the bugfixer
on failures, and loops until all models are processed or max iterations are hit.

Main loop:
    1. Scout processes all queued models (one at a time, sequentially).
    2. Bugfixer scans failures, classifies them, applies fixes, re-queues fixable ones.
    3. If bugfixer re-queued any models, loop back to step 1.
    4. Stop when no queued or fixable-failed models remain, or max iterations reached.

CLI:
    python swarm/orchestrator.py                    # Run full pipeline
    python swarm/orchestrator.py --tier 1           # Only tier 1
    python swarm/orchestrator.py --dry-run           # Preview plan
    python swarm/orchestrator.py --max-iterations 3  # Limit scout-bugfixer loops
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/ceph/workspace/erik/oxRL")
MANIFEST_PATH = PROJECT_ROOT / "swarm" / "manifest.json"
ONBOARDED_DIR = PROJECT_ROOT / "onboarded"

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][orchestrator][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("orchestrator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MAX_ITERATIONS = 5
MAX_ATTEMPTS = 3

# Status values used in manifest
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_FAILED = "failed"
STATUS_ONBOARDED = "onboarded"
STATUS_SKIPPED = "skipped"

ALL_STATUSES = [STATUS_QUEUED, STATUS_RUNNING, STATUS_FAILED, STATUS_ONBOARDED, STATUS_SKIPPED]


# ===================================================================
# Manifest helpers
# ===================================================================

def load_manifest() -> dict:
    """Load manifest.json, returning the parsed dict."""
    with open(MANIFEST_PATH, "r") as f:
        return json.load(f)


def count_by_status(manifest: dict) -> dict:
    """Count models by status.

    Returns a dict like {"queued": 3, "running": 0, "onboarded": 2, "failed": 1, "skipped": 0}.
    """
    counts = {s: 0 for s in ALL_STATUSES}
    for _model_id, entry in manifest.get("models", {}).items():
        status = entry.get("status", STATUS_QUEUED)
        if status in counts:
            counts[status] += 1
        else:
            counts[status] = counts.get(status, 0) + 1
    return counts


def _get_queued_models(manifest: dict, tier: Optional[int] = None) -> list[str]:
    """Return model IDs with status 'queued', optionally filtered by tier."""
    result = []
    for model_id, entry in manifest.get("models", {}).items():
        if entry.get("status") != STATUS_QUEUED:
            continue
        if tier is not None and entry.get("tier") != tier:
            continue
        result.append(model_id)
    return result


def _get_failed_models(manifest: dict, tier: Optional[int] = None) -> list[str]:
    """Return model IDs with status 'failed' and attempts < MAX_ATTEMPTS."""
    result = []
    for model_id, entry in manifest.get("models", {}).items():
        if entry.get("status") != STATUS_FAILED:
            continue
        if entry.get("attempts", 0) >= MAX_ATTEMPTS:
            continue
        if tier is not None and entry.get("tier") != tier:
            continue
        result.append(model_id)
    return result


# ===================================================================
# Status table / summary
# ===================================================================

def print_status_table(manifest: dict) -> None:
    """Print a formatted table showing all models and their current state."""
    models = manifest.get("models", {})
    if not models:
        logger.info("No models in manifest.")
        return

    # Column widths
    col_model = 50
    col_status = 12
    col_tier = 6
    col_params = 10
    col_attempts = 10
    col_error = 40

    header = (
        f"  {'Model':<{col_model}} "
        f"{'Status':<{col_status}} "
        f"{'Tier':<{col_tier}} "
        f"{'Params':<{col_params}} "
        f"{'Attempts':<{col_attempts}} "
        f"{'Error':<{col_error}}"
    )
    separator = "  " + "-" * (col_model + col_status + col_tier + col_params + col_attempts + col_error + 5)

    print()
    print(separator)
    print(header)
    print(separator)

    for model_id, entry in models.items():
        status = entry.get("status", "?")
        tier = entry.get("tier", "?")
        params = entry.get("param_count_b")
        params_str = f"{params:.1f}B" if params is not None else "?"
        attempts = entry.get("attempts", 0)
        error = entry.get("error_log") or ""
        # Truncate error for table display
        if len(error) > col_error:
            error = error[:col_error - 3] + "..."

        # Status indicator
        if status == STATUS_ONBOARDED:
            status_display = "DONE"
        elif status == STATUS_FAILED:
            status_display = "FAILED"
        elif status == STATUS_QUEUED:
            status_display = "QUEUED"
        elif status == STATUS_RUNNING:
            status_display = "RUNNING"
        elif status == STATUS_SKIPPED:
            status_display = "SKIPPED"
        else:
            status_display = status.upper()

        print(
            f"  {model_id:<{col_model}} "
            f"{status_display:<{col_status}} "
            f"{str(tier):<{col_tier}} "
            f"{params_str:<{col_params}} "
            f"{str(attempts):<{col_attempts}} "
            f"{error:<{col_error}}"
        )

    print(separator)
    print()


def print_summary(manifest: dict) -> None:
    """Print a final summary of all model statuses."""
    counts = count_by_status(manifest)
    total = sum(counts.values())

    print()
    print("=" * 70)
    print("ORCHESTRATOR -- FINAL SUMMARY")
    print("=" * 70)
    print(f"  Total models  : {total}")
    print(f"  Onboarded     : {counts.get(STATUS_ONBOARDED, 0)}")
    print(f"  Failed        : {counts.get(STATUS_FAILED, 0)}")
    print(f"  Skipped       : {counts.get(STATUS_SKIPPED, 0)}")
    print(f"  Queued        : {counts.get(STATUS_QUEUED, 0)}")
    print(f"  Running       : {counts.get(STATUS_RUNNING, 0)}")
    print("=" * 70)

    # List onboarded models
    models = manifest.get("models", {})
    onboarded = [mid for mid, e in models.items() if e.get("status") == STATUS_ONBOARDED]
    if onboarded:
        print()
        print("  Onboarded models:")
        for mid in onboarded:
            print(f"    + {mid}")

    # List failed models (exhausted retries)
    permanently_failed = [
        mid for mid, e in models.items()
        if e.get("status") == STATUS_FAILED and e.get("attempts", 0) >= MAX_ATTEMPTS
    ]
    if permanently_failed:
        print()
        print("  Permanently failed (max attempts reached):")
        for mid in permanently_failed:
            err = models[mid].get("error_log", "")
            print(f"    x {mid} -- {err}")

    # List skipped models
    skipped = [mid for mid, e in models.items() if e.get("status") == STATUS_SKIPPED]
    if skipped:
        print()
        print("  Skipped (unfixable):")
        for mid in skipped:
            err = models[mid].get("error_log", "")
            print(f"    - {mid} -- {err}")

    print()


# ===================================================================
# Agent dispatchers
# ===================================================================

def run_scout(tier: Optional[int] = None, dry_run: bool = False) -> None:
    """Import and run the scout agent on all queued models.

    The scout processes models one at a time internally (its main() reads
    the manifest and iterates). We call its main logic directly by building
    the queue and processing each model via onboard_model().
    """
    from swarm.scout import _load_manifest as scout_load, build_queue, onboard_model

    logger.info("Dispatching scout agent (tier=%s, dry_run=%s)", tier, dry_run)

    manifest = scout_load()
    queue = build_queue(manifest, tier_filter=tier)

    if not queue:
        logger.info("Scout: no queued models to process.")
        return

    logger.info("Scout: %d model(s) in queue", len(queue))
    for i, (mid, entry) in enumerate(queue, 1):
        logger.info(
            "  %2d. %-50s tier=%s  params=%.2fB  dataset=%s",
            i, mid,
            entry.get("tier"),
            entry.get("param_count_b", 0),
            entry.get("dataset"),
        )

    for model_id, entry in queue:
        # Re-read manifest before each model (bugfixer or previous iteration
        # may have changed statuses)
        fresh_manifest = scout_load()
        fresh_entry = fresh_manifest.get("models", {}).get(model_id, entry)

        # Skip if no longer queued (e.g. already onboarded or skipped)
        if fresh_entry.get("status") not in (STATUS_QUEUED, STATUS_FAILED):
            logger.info("Scout: skipping %s (status=%s)", model_id, fresh_entry.get("status"))
            continue
        if fresh_entry.get("attempts", 0) >= MAX_ATTEMPTS:
            logger.info("Scout: skipping %s (max attempts reached)", model_id)
            continue

        success = onboard_model(model_id, fresh_entry, dry_run=dry_run)

        if dry_run:
            continue

        if success:
            logger.info("Scout: %s -> onboarded", model_id)
        else:
            logger.info("Scout: %s -> failed", model_id)

        # Print progress after each model
        current_manifest = load_manifest()
        print_status_table(current_manifest)


def run_bugfixer(tier: Optional[int] = None, dry_run: bool = False) -> dict:
    """Import and run the bugfixer agent on all failed models.

    Returns the bugfixer summary dict: {"fixed": [...], "skipped": [...], "errors": [...]}.
    """
    from swarm.bugfixer import run as bugfixer_run

    logger.info("Dispatching bugfixer agent (dry_run=%s)", dry_run)

    summary = bugfixer_run(
        manifest_path=str(MANIFEST_PATH),
        onboarded_dir=str(ONBOARDED_DIR),
        max_retries=MAX_ATTEMPTS,
        dry_run=dry_run,
    )

    fixed_count = len(summary.get("fixed", []))
    skipped_count = len(summary.get("skipped", []))
    error_count = len(summary.get("errors", []))

    logger.info(
        "Bugfixer complete: %d fixed (re-queued), %d skipped, %d errors",
        fixed_count, skipped_count, error_count,
    )

    return summary


# ===================================================================
# Dry-run preview
# ===================================================================

def preview_plan(manifest: dict, tier: Optional[int] = None) -> None:
    """Print a preview of what the orchestrator would do without executing."""
    print()
    print("=" * 70)
    print("ORCHESTRATOR -- DRY RUN PREVIEW")
    print("=" * 70)

    counts = count_by_status(manifest)
    print(f"  Current status counts: {counts}")
    print()

    queued = _get_queued_models(manifest, tier=tier)
    failed = _get_failed_models(manifest, tier=tier)

    if not queued and not failed:
        print("  Nothing to do. All models are either onboarded, skipped, or exhausted retries.")
        print()
        return

    if queued:
        print(f"  Phase 1 -- Scout would process {len(queued)} queued model(s):")
        models = manifest.get("models", {})
        for mid in queued:
            entry = models[mid]
            print(
                f"    -> {mid} (tier={entry.get('tier')}, "
                f"params={entry.get('param_count_b', '?')}B, "
                f"dataset={entry.get('dataset')}, "
                f"attempt={entry.get('attempts', 0)})"
            )
        print()

    if failed:
        print(f"  Phase 2 -- Bugfixer would examine {len(failed)} failed model(s):")
        models = manifest.get("models", {})
        for mid in failed:
            entry = models[mid]
            print(
                f"    -> {mid} (error={entry.get('error_log', 'unknown')}, "
                f"attempt={entry.get('attempts', 0)}/{MAX_ATTEMPTS})"
            )
        print()

    print("  The scout->bugfixer loop would repeat until no queued/fixable models remain")
    print("  (or max iterations reached).")
    print("=" * 70)
    print()


# ===================================================================
# Main orchestrator loop
# ===================================================================

def run(
    tier: Optional[int] = None,
    dry_run: bool = False,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> None:
    """Run the full orchestrator loop: scout -> bugfixer -> repeat.

    Parameters
    ----------
    tier : int, optional
        Only process models of this tier (1, 2, or 3).
    dry_run : bool
        If True, preview the plan without executing anything.
    max_iterations : int
        Maximum number of scout->bugfixer loop iterations to prevent
        infinite loops.
    """
    start_time = time.monotonic()

    logger.info("=" * 70)
    logger.info("ORCHESTRATOR STARTING")
    logger.info("=" * 70)
    logger.info("  Manifest       : %s", MANIFEST_PATH)
    logger.info("  Onboarded dir  : %s", ONBOARDED_DIR)
    logger.info("  Tier filter    : %s", tier if tier is not None else "(all)")
    logger.info("  Max iterations : %d", max_iterations)
    logger.info("  Dry run        : %s", dry_run)

    if not MANIFEST_PATH.exists():
        logger.error("Manifest not found: %s", MANIFEST_PATH)
        sys.exit(1)

    # Initial status
    manifest = load_manifest()
    counts = count_by_status(manifest)
    logger.info("  Initial status : %s", counts)
    print_status_table(manifest)

    # Dry-run mode: just preview and exit
    if dry_run:
        preview_plan(manifest, tier=tier)
        # Also run scout and bugfixer in dry-run mode so they print their plans
        run_scout(tier=tier, dry_run=True)
        run_bugfixer(tier=tier, dry_run=True)
        return

    # Main loop
    for iteration in range(1, max_iterations + 1):
        logger.info("-" * 70)
        logger.info("ITERATION %d / %d", iteration, max_iterations)
        logger.info("-" * 70)

        # Re-read manifest at the start of each iteration
        manifest = load_manifest()
        queued = _get_queued_models(manifest, tier=tier)
        failed = _get_failed_models(manifest, tier=tier)

        if not queued and not failed:
            logger.info("No queued or fixable-failed models remaining. Done.")
            break

        # ---- Phase 1: Run scout on queued models ----
        if queued:
            logger.info(
                "Phase 1: Scout processing %d queued model(s)", len(queued)
            )
            run_scout(tier=tier, dry_run=False)
        else:
            logger.info("Phase 1: No queued models. Skipping scout.")

        # ---- Phase 2: Run bugfixer on failures ----
        manifest = load_manifest()  # Re-read after scout
        failed = _get_failed_models(manifest, tier=tier)

        if failed:
            logger.info(
                "Phase 2: Bugfixer examining %d failed model(s)", len(failed)
            )
            bugfixer_summary = run_bugfixer(tier=tier, dry_run=False)

            # Print status after bugfixer
            manifest = load_manifest()
            print_status_table(manifest)
        else:
            logger.info("Phase 2: No failed models. Skipping bugfixer.")

        # ---- Phase 3: Check if bugfixer re-queued anything ----
        manifest = load_manifest()
        newly_queued = _get_queued_models(manifest, tier=tier)

        if not newly_queued:
            logger.info("No models re-queued by bugfixer. Orchestrator loop complete.")
            break

        logger.info(
            "%d model(s) re-queued by bugfixer. Continuing to next iteration.",
            len(newly_queued),
        )

    else:
        # max_iterations exhausted
        logger.warning(
            "Max iterations (%d) reached. Some models may still be queued or failed.",
            max_iterations,
        )

    # Final summary
    elapsed = time.monotonic() - start_time
    manifest = load_manifest()
    logger.info("Orchestrator finished in %.1f seconds.", elapsed)
    print_summary(manifest)


# ===================================================================
# CLI
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="oxRL Orchestrator -- dispatch scout and bugfixer agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python swarm/orchestrator.py                    # Run full pipeline
  python swarm/orchestrator.py --tier 1           # Only tier 1
  python swarm/orchestrator.py --dry-run           # Preview plan
  python swarm/orchestrator.py --max-iterations 3  # Limit scout-bugfixer loops
""",
    )
    parser.add_argument(
        "--tier",
        type=int,
        default=None,
        choices=[1, 2, 3],
        help="Only process models of this tier",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Preview what would be done without executing anything",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Max scout->bugfixer loop iterations (default: {DEFAULT_MAX_ITERATIONS})",
    )
    args = parser.parse_args()

    run(
        tier=args.tier,
        dry_run=args.dry_run,
        max_iterations=args.max_iterations,
    )


if __name__ == "__main__":
    main()
