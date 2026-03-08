"""
Loss function registry for policy-gradient algorithms.

Each loss variant is a pure function with identical signature:
    (logprobs, old_logprobs, advantages, mask, entropies, ref_logprobs,
     clip_low, clip_high, ent_coeff, kl_coeff) -> (loss, metrics)

Usage:
    from oxrl.algs.losses import get_loss_fn
    loss_fn = get_loss_fn("sgrpo")
    loss, metrics = loss_fn(logprobs, old_logprobs, ...)
"""
from oxrl.algs.losses.sgrpo import compute_sgrpo_loss
from oxrl.algs.losses.gspo import compute_gspo_loss
from oxrl.algs.losses.cispo import compute_cispo_loss

LOSS_REGISTRY = {
    "sgrpo": compute_sgrpo_loss,
    "gspo": compute_gspo_loss,
    "cispo": compute_cispo_loss,
    # PPO uses the same token-level clipped surrogate as SGRPO
    "ppo": compute_sgrpo_loss,
}


def get_loss_fn(variant: str):
    """Look up a loss function by name.

    Checks the built-in registry first, then falls back to any custom
    losses registered via :func:`oxrl.models.research_adapters.register_loss`.
    Raises ValueError if the variant is not found in either registry.
    """
    if variant in LOSS_REGISTRY:
        return LOSS_REGISTRY[variant]
    # Lazy import to avoid circular dependencies
    from oxrl.models.research_adapters import get_custom_loss_fn
    custom = get_custom_loss_fn(variant)
    if custom is not None:
        return custom
    raise ValueError(
        f"Unknown loss variant: {variant!r}. "
        f"Available: {sorted(LOSS_REGISTRY.keys())}"
    )
