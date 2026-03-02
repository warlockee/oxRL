"""
Reward scoring and z-score normalization for rollout groups.

Pure functions — no model state, no engine references.
"""
import torch
import numpy as np
from typing import List, Dict, Any, Callable, Tuple


def score_response(
    reward_func: Callable,
    prompt_ids: List[int],
    response_ids: List[int],
    finish_reason: str,
    metadata: dict = None,
) -> Tuple[torch.Tensor, bool]:
    """Score a single response using the reward function.

    Returns:
        (rewards, is_per_token) where rewards is float32 CPU tensor of len(response_ids).
    """
    with torch.no_grad():
        rewards, is_per_token = reward_func(
            prompt_ids, response_ids, finish_reason, metadata=metadata
        )

    if isinstance(rewards, torch.Tensor):
        rewards = rewards.to(dtype=torch.float32, device="cpu")
    else:
        rewards = torch.tensor(rewards, dtype=torch.float32, device="cpu")

    if rewards.numel() != len(response_ids):
        raise ValueError(
            f"score_response must return len={len(response_ids)} rewards, "
            f"got {rewards.numel()}"
        )

    return rewards, is_per_token


def normalize_rewards(
    samples: List[Dict[str, Any]],
    stats: Dict[str, List],
    prompt_len: int,
    is_per_token: bool,
    eps_reward_norm: float,
    reward_broadcast: bool,
) -> None:
    """Z-score normalize rewards across a group of samples for one prompt.

    Mutates samples in-place: sets 'zscores' and 'pred_zscores' keys.
    """
    denom = len(samples)
    if len(samples) > 1:
        rewards_array = np.array(stats["rewards"])
        mean_scores = rewards_array.sum() / denom
        std_scores = np.sqrt(
            ((rewards_array - mean_scores) ** 2).sum() / max(1, denom - 1)
        )
    else:
        mean_scores = 0.0
        std_scores = 1.0 - eps_reward_norm

    if is_per_token:
        raise ValueError(
            "per token rewards are not supported yet as normalization "
            "is done assuming per response rewards"
        )

    for sample in samples:
        zscore = torch.zeros_like(sample["rewards"], dtype=torch.float)
        zscore[-1] = (sample["rewards"][-1] - mean_scores) / (
            std_scores + eps_reward_norm
        )
        sample["zscores"] = zscore
        if reward_broadcast:
            sample["zscores"][prompt_len:] = zscore[-1]

        # prediction-aligned zscores
        pred_zscores = torch.zeros_like(sample["rewards"], dtype=torch.float)
        pred_start = prompt_len - 1
        pred_end = len(sample["rewards"]) - 1
        pred_zscores[pred_start:pred_end] = sample["zscores"][prompt_len:]
        sample["pred_zscores"] = pred_zscores
