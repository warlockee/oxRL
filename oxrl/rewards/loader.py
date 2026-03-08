"""Pluggable reward backend loader.

Resolves reward function strings to RewardBackend instances, supports:
- Builtin names (e.g. "gsm8k_reward_func") from oxrl.rewards
- Dotted module paths (e.g. "my_project.rewards.custom_func")
- Composite rewards ("composite") combining multiple sub-rewards
- LLM-as-judge ("llm_judge_reward_func") using external LLM API
- Reward model ("rm_reward_func") using a trained reward model
"""
import importlib
from typing import Callable, List, Optional, Tuple

import torch

from oxrl.rewards.backend import RewardBackend, FunctionRewardBackend


def create_reward_backend(
    reward_func: str,
    composite_rewards=None,
    llm_judge_config=None,
    reward_model_config=None,
) -> RewardBackend:
    """Create a RewardBackend from a reward function string.

    Args:
        reward_func: Name or dotted path of the reward function.
        composite_rewards: List of CompositeRewardEntry for composite mode.
        llm_judge_config: LLMJudgeConfig for LLM-as-judge mode.
        reward_model_config: RewardModelConfig for reward model mode.

    Returns:
        RewardBackend instance.
    """
    if reward_func == "composite":
        if not composite_rewards:
            raise ValueError(
                "reward_func='composite' requires non-empty composite_rewards config"
            )
        sub_backends = []
        for entry in composite_rewards:
            sub = _resolve_to_backend(entry.func)
            sub_backends.append((sub, entry.weight))
        return CompositeRewardBackend(sub_backends)

    if reward_func == "llm_judge_reward_func":
        if llm_judge_config is None:
            raise ValueError(
                "reward_func='llm_judge_reward_func' requires llm_judge config"
            )
        from oxrl.rewards.llm_judge import LLMJudgeReward
        return LLMJudgeReward(llm_judge_config)

    if reward_func == "rm_reward_func" and reward_model_config is not None:
        from oxrl.rewards.rm_reward import RewardModelBackend
        return RewardModelBackend(
            model_path=reward_model_config.model_path,
            device=reward_model_config.device,
        )

    return _resolve_to_backend(reward_func)


def resolve_reward_func(
    reward_func: str,
    composite_rewards=None,
    llm_judge_config=None,
    reward_model_config=None,
) -> RewardBackend:
    """Backward-compatible alias for create_reward_backend."""
    return create_reward_backend(
        reward_func=reward_func,
        composite_rewards=composite_rewards,
        llm_judge_config=llm_judge_config,
        reward_model_config=reward_model_config,
    )


def _resolve_single(name: str) -> Callable:
    """Resolve a single reward function name or dotted path to a callable."""
    if "." in name:
        module_path, func_name = name.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, func_name)
    else:
        module = importlib.import_module("oxrl.rewards")
        return getattr(module, name)


def _resolve_to_backend(name: str) -> RewardBackend:
    """Resolve a name to a RewardBackend (wrapping bare functions)."""
    obj = _resolve_single(name)
    if isinstance(obj, RewardBackend):
        return obj
    return FunctionRewardBackend(obj)


class CompositeRewardBackend(RewardBackend):
    """Combines multiple reward backends with weighted averaging.

    Callable and picklable for use with Ray actors.
    """

    def __init__(self, sub_funcs: List[Tuple[RewardBackend, float]]):
        if not sub_funcs:
            raise ValueError("CompositeRewardBackend requires at least one sub-function")
        self.sub_funcs = sub_funcs
        total_weight = sum(w for _, w in sub_funcs)
        if total_weight <= 0:
            raise ValueError("Total weight must be positive")
        self.normalized_weights = [w / total_weight for _, w in sub_funcs]

    def setup(self, config=None):
        for backend, _ in self.sub_funcs:
            backend.setup(config)

    def cleanup(self):
        for backend, _ in self.sub_funcs:
            backend.cleanup()

    def __call__(self, prompt_ids, response_ids, finish_reason, metadata=None):
        resp_len = len(response_ids)
        if resp_len == 0:
            return torch.zeros((0,), dtype=torch.float32), False

        combined = torch.zeros((resp_len,), dtype=torch.float32)

        for (fn, _), weight in zip(self.sub_funcs, self.normalized_weights):
            reward_tensor, is_per_token = fn(
                prompt_ids, response_ids, finish_reason, metadata
            )
            if is_per_token:
                raise ValueError(
                    f"CompositeRewardBackend does not support per-token sub-rewards "
                    f"(got per-token from {fn})"
                )
            combined += weight * reward_tensor

        return combined, False


# Backward-compatible alias
CompositeReward = CompositeRewardBackend
