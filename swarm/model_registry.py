"""
model_registry.py — Discover and rank candidate models from HuggingFace Hub.

Queries the HF Hub for text-generation models that are compatible with oxRL
(AutoModelForCausalLM + chat_template), filters by parameter count to fit
within a 2xA100-40GB budget (1 train + 1 rollout GPU), and assigns tiers.

Tiers:
    1 — up to 1.5B params
    2 — 3B to 4B params
    3 — up to 7B params
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier boundaries (in billions of parameters)
# ---------------------------------------------------------------------------
_TIER_BOUNDARIES = [
    (1, 1.5),   # Tier 1: param_count_b <= 1.5
    (2, 4.0),   # Tier 2: 1.5 < param_count_b <= 4.0
    (3, 7.0),   # Tier 3: 4.0 < param_count_b <= 7.0
]

# HF Hub pipeline tag that corresponds to causal LM / AutoModelForCausalLM
_PIPELINE_TAG = "text-generation"

# Hard ceiling — models above this are too large for 2xA100-40GB
_DEFAULT_MAX_PARAMS_B = 7.0


def get_tier(param_count_b: float) -> int:
    """Return tier (1, 2, or 3) based on parameter count in billions.

    Tier 1: <= 1.5B
    Tier 2: > 1.5B and <= 4B  (covers 3B-4B range)
    Tier 3: > 4B  and <= 7B

    Returns 0 if the model does not fit any tier (e.g. > 7B or <= 0).
    """
    if param_count_b <= 0:
        return 0
    for tier, upper in _TIER_BOUNDARIES:
        if param_count_b <= upper:
            return tier
    return 0


def check_chat_template(model_id: str) -> bool:
    """Check whether a model's tokenizer defines a ``chat_template``.

    Fetches ``tokenizer_config.json`` from the Hub and looks for the
    ``chat_template`` key.  Returns *False* on any error (network failure,
    gated model, missing file, etc.) so callers can treat the model as
    incompatible without crashing.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("huggingface_hub is not installed — cannot check chat template.")
        return False

    try:
        path = hf_hub_download(
            repo_id=model_id,
            filename="tokenizer_config.json",
            repo_type="model",
        )
        with open(path, "r") as f:
            config = json.load(f)
        has_template = bool(config.get("chat_template"))
        return has_template
    except Exception as exc:
        # Gated models, private repos, network errors, missing file, etc.
        logger.debug("Could not fetch tokenizer_config.json for %s: %s", model_id, exc)
        return False


def _extract_param_count_b(model_info) -> Optional[float]:
    """Extract parameter count (in billions) from an HF ModelInfo object.

    HF Hub exposes ``safetensors.total`` (exact param count) on many models.
    Falls back to the ``params`` field when available.  Returns *None* when
    neither source is present.
    """
    # Preferred: safetensors metadata carries an exact count
    safetensors = getattr(model_info, "safetensors", None)
    if safetensors is not None:
        # safetensors can be a dict with 'total' or 'parameters' sub-dicts
        total = None
        if isinstance(safetensors, dict):
            total = safetensors.get("total")
            if total is None:
                params = safetensors.get("parameters", {})
                if isinstance(params, dict):
                    total = sum(params.values())
        if total is not None and total > 0:
            return total / 1e9

    # Fallback to the direct attribute the API sometimes provides
    # (added in newer huggingface_hub versions)
    params = getattr(model_info, "params", None)
    if params is not None and params > 0:
        return params / 1e9

    return None


def get_model_info(model_id: str) -> dict:
    """Retrieve detailed information for a single model from HuggingFace Hub.

    Returns a dict with:
        model_id          (str)   — HF repo id
        param_count_b     (float | None)
        downloads         (int)
        pipeline_tag      (str | None)
        has_chat_template (bool)
        tier              (int)   — 0 if param count unknown or out of range
        library_name      (str | None)
        tags              (list[str])
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub is not installed.")
        return {"model_id": model_id, "error": "huggingface_hub not installed"}

    api = HfApi()
    try:
        info = api.model_info(model_id)
    except Exception as exc:
        logger.warning("Failed to fetch model info for %s: %s", model_id, exc)
        return {"model_id": model_id, "error": str(exc)}

    param_b = _extract_param_count_b(info)
    has_template = check_chat_template(model_id)
    tier = get_tier(param_b) if param_b is not None else 0

    return {
        "model_id": model_id,
        "param_count_b": round(param_b, 2) if param_b is not None else None,
        "downloads": getattr(info, "downloads", 0) or 0,
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "has_chat_template": has_template,
        "tier": tier,
        "library_name": getattr(info, "library_name", None),
        "tags": list(getattr(info, "tags", []) or []),
    }


def discover_models(
    max_params_b: float = _DEFAULT_MAX_PARAMS_B,
    limit: int = 50,
) -> list[dict]:
    """Query HuggingFace Hub for candidate causal-LM models.

    Filters:
        - pipeline_tag = text-generation (causal LM)
        - parameter count <= *max_params_b* billion
        - tokenizer must define a ``chat_template``

    Returns a list of dicts sorted by downloads (most popular first), each
    containing: ``model_id``, ``param_count_b``, ``downloads``,
    ``has_chat_template``, ``tier``.

    At most *limit* models are returned.  The function gracefully handles
    network errors and gated / inaccessible models by skipping them.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub is not installed — pip install huggingface_hub")
        return []

    api = HfApi()

    # Fetch a larger pool so we can apply our own filters.
    # sort="downloads" gives us the most popular models first.
    fetch_limit = limit * 10  # over-fetch to account for filtering
    try:
        models = api.list_models(
            pipeline_tag=_PIPELINE_TAG,
            sort="downloads",
            direction=-1,
            limit=fetch_limit,
            library="transformers",
        )
    except Exception as exc:
        logger.error("Failed to query HuggingFace Hub: %s", exc)
        return []

    candidates: list[dict] = []

    for info in models:
        if len(candidates) >= limit:
            break

        model_id = info.id if hasattr(info, "id") else str(info)

        # --- Parameter count filter ---
        param_b = _extract_param_count_b(info)
        if param_b is None:
            # Cannot determine size — skip to be safe
            logger.debug("Skipping %s — unknown param count", model_id)
            continue
        if param_b > max_params_b:
            logger.debug("Skipping %s — %.1fB params exceeds %.1fB limit", model_id, param_b, max_params_b)
            continue

        tier = get_tier(param_b)
        if tier == 0:
            logger.debug("Skipping %s — %.2fB does not fit any tier", model_id, param_b)
            continue

        # --- Chat template filter ---
        has_template = check_chat_template(model_id)
        if not has_template:
            logger.debug("Skipping %s — no chat_template", model_id)
            continue

        downloads = getattr(info, "downloads", 0) or 0

        candidates.append({
            "model_id": model_id,
            "param_count_b": round(param_b, 2),
            "downloads": downloads,
            "has_chat_template": True,
            "tier": tier,
        })

    # Ensure sorted by downloads descending (HF API usually returns this
    # order, but let's be explicit after filtering).
    candidates.sort(key=lambda m: m["downloads"], reverse=True)

    return candidates


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("=" * 60)
    print("oxRL Model Registry — Discovering candidate models")
    print("=" * 60)

    models = discover_models(max_params_b=7.0, limit=20)

    if not models:
        print("\nNo models found (check network / huggingface_hub installation).")
    else:
        print(f"\nFound {len(models)} candidate model(s):\n")
        for i, m in enumerate(models, 1):
            print(
                f"  {i:>2}. {m['model_id']:<50s} "
                f"| {m['param_count_b']:>5.2f}B "
                f"| tier {m['tier']} "
                f"| {m['downloads']:>12,} downloads"
            )

    # Also demonstrate single-model lookup
    print("\n" + "-" * 60)
    print("Single model info (Qwen/Qwen2.5-0.5B-Instruct):")
    print("-" * 60)
    info = get_model_info("Qwen/Qwen2.5-0.5B-Instruct")
    for k, v in info.items():
        print(f"  {k}: {v}")
