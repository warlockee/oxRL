"""
Log-probability extraction from vLLM output.

Pure function — no engine state, no model references.
"""
import torch
from typing import List, Any


def extract_logprobs(
    response_ids: List[int],
    logprobs_by_pos: Any,
) -> torch.Tensor:
    """Extract per-token logprobs aligned with response_ids.

    Args:
        response_ids:    List of token IDs in the generated response.
        logprobs_by_pos: List of dicts {token_id -> logprob_info} from vLLM.

    Returns:
        Float32 CPU tensor of shape [len(response_ids)].
    """
    if logprobs_by_pos is None:
        raise ValueError("logprobs_by_pos must not be None.")

    if not isinstance(logprobs_by_pos, list):
        raise TypeError(f"logprobs_by_pos must be a list, got {type(logprobs_by_pos)}")

    if len(response_ids) != len(logprobs_by_pos):
        raise ValueError(
            f"logprobs_by_pos must have the same len as response_ids. "
            f"Got {len(logprobs_by_pos)} vs {len(response_ids)}."
        )

    token_logprobs = []
    for t_id, lgp_dict in zip(response_ids, logprobs_by_pos):
        if lgp_dict is None:
            raise ValueError(f"No logprobs for token {t_id} in {response_ids}.")

        key = t_id
        if key not in lgp_dict and str(key) in lgp_dict:
            key = str(key)

        if key not in lgp_dict:
            raise ValueError(f"No logprobs for token {t_id} in {response_ids}.")

        v = lgp_dict[key]
        if hasattr(v, "logprob"):
            token_logprobs.append(float(v.logprob))
        elif isinstance(v, (int, float)):
            token_logprobs.append(float(v))
        elif isinstance(v, dict) and "logprob" in v:
            token_logprobs.append(float(v["logprob"]))
        else:
            raise TypeError(f"Unexpected logprob type: {type(v)}")

    return torch.tensor(token_logprobs, dtype=torch.float32, device="cpu")
