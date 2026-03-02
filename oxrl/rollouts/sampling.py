"""
Sampling parameter construction and on-policy validation.

Pure functions — no model state, no vLLM engine references.
"""
from vllm import SamplingParams


def validate_on_policy_params(
    temperature: float,
    top_p: float,
    top_k: int,
    n_samples: int,
    max_tokens: int,
    stop,
    stop_token_ids,
    ignore_eos: bool,
) -> None:
    """Enforce that sampling params stay in on-policy regime.

    Raises ValueError if any parameter would bias the sampling distribution
    away from the true policy.
    """
    if temperature != 1.0:
        raise ValueError("Strict on-policy requires temperature = 1.0 (no scaling).")

    if top_p != 1.0:
        raise ValueError("Strict on-policy requires top_p = 1.0 (no nucleus truncation).")

    if top_k != -1:
        raise ValueError("Strict on-policy requires top_k = -1 (no top-k truncation).")

    if n_samples < 1:
        raise ValueError("Strict on-policy requires n_samples >= 1.")

    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0.")

    if stop is not None or stop_token_ids is not None or ignore_eos:
        raise ValueError(
            "Strict on-policy requires stop=None, stop_token_ids=None, ignore_eos=False "
            "(these change the trajectory distribution)."
        )


def make_sampling_params(
    seed: int,
    n_samples: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    stop,
    stop_token_ids,
    ignore_eos: bool,
    prompt_logprobs: bool,
    force_strict_on_policy: bool,
) -> SamplingParams:
    """Build vLLM SamplingParams, optionally validating on-policy constraints."""
    if force_strict_on_policy:
        validate_on_policy_params(
            temperature, top_p, top_k, n_samples, max_tokens,
            stop, stop_token_ids, ignore_eos,
        )

    return SamplingParams(
        seed=seed,
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=0.0,
        max_tokens=max_tokens,
        stop=stop,
        stop_token_ids=stop_token_ids,
        ignore_eos=ignore_eos,
        # Neutral penalties and no shaping
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        logit_bias=None,
        allowed_token_ids=None,
        bad_words=None,
        logits_processors=None,
        # Returns required info
        logprobs=1,
        prompt_logprobs=(1 if prompt_logprobs else None),
    )
