"""
Research adapters — decorator-based registries for custom losses,
algorithms, and reward shapers.

Researchers can register extensions from their own modules without
modifying any oxRL core file.  A single ``research.module`` config
field auto-imports the user module at startup so that decorators fire.

Usage::

    # my_research/extensions.py
    from oxrl.models.research_adapters import register_loss

    @register_loss("dr_grpo")
    def compute_dr_grpo_loss(logprobs, old_logprobs, advantages, mask,
                             entropies, ref_logprobs, clip_low, clip_high,
                             ent_coeff, kl_coeff):
        ...
"""
import importlib
from typing import Callable, Dict, Optional

import torch

from oxrl.rewards.backend import RewardBackend

# ---------------------------------------------------------------------------
# Loss registry
# ---------------------------------------------------------------------------

_CUSTOM_LOSS_REGISTRY: Dict[str, Callable] = {}


def register_loss(name: str):
    """Decorator to register a custom loss function with the standard signature."""
    def decorator(fn: Callable) -> Callable:
        _CUSTOM_LOSS_REGISTRY[name] = fn
        return fn
    return decorator


def get_custom_loss_fn(name: str) -> Optional[Callable]:
    """Look up a custom-registered loss by name (returns None if not found)."""
    return _CUSTOM_LOSS_REGISTRY.get(name)


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

_CUSTOM_ALGORITHM_REGISTRY: Dict[str, type] = {}


def register_algorithm(name: str):
    """Decorator to register a custom algorithm class (BaseAlgorithm subclass).

    ``@ray.remote`` is applied automatically if missing when the algorithm
    is retrieved via :func:`get_algorithm_class`.
    """
    def decorator(cls: type) -> type:
        _CUSTOM_ALGORITHM_REGISTRY[name] = cls
        return cls
    return decorator


def get_custom_algorithm(name: str) -> Optional[type]:
    """Look up a custom-registered algorithm by name (returns None if not found)."""
    return _CUSTOM_ALGORITHM_REGISTRY.get(name)


# ---------------------------------------------------------------------------
# Reward shapers
# ---------------------------------------------------------------------------

class RewardShaper(RewardBackend):
    """Wraps an inner ``RewardBackend`` and transforms its output.

    Subclass and override :meth:`shape` for custom shaping.
    """

    def __init__(self, inner: RewardBackend, **kwargs):
        self.inner = inner
        self.kwargs = kwargs

    def __call__(self, prompt_ids, response_ids, finish_reason, metadata=None):
        reward_tensor, is_per_token = self.inner(
            prompt_ids, response_ids, finish_reason, metadata,
        )
        return self.shape(
            reward_tensor, is_per_token,
            prompt_ids, response_ids, finish_reason, metadata,
        )

    def shape(self, reward_tensor, is_per_token, prompt_ids, response_ids,
              finish_reason, metadata):
        """Override to transform rewards. Default is identity."""
        return reward_tensor, is_per_token

    def setup(self, config=None):
        self.inner.setup(config)

    def cleanup(self):
        self.inner.cleanup()


class LengthPenaltyShaper(RewardShaper):
    """Adds a per-token length penalty to the final (sparse) reward.

    ``shaped_reward = reward - penalty_per_token * num_response_tokens``
    """

    def __init__(self, inner: RewardBackend, penalty_per_token: float = 0.001):
        super().__init__(inner, penalty_per_token=penalty_per_token)
        self.penalty_per_token = penalty_per_token

    def shape(self, reward_tensor, is_per_token, prompt_ids, response_ids,
              finish_reason, metadata):
        penalty = self.penalty_per_token * len(response_ids)
        return reward_tensor - penalty, is_per_token


class ClampRewardShaper(RewardShaper):
    """Clamps rewards to ``[min_val, max_val]``."""

    def __init__(self, inner: RewardBackend, min_val: float = -1.0,
                 max_val: float = 1.0):
        super().__init__(inner, min_val=min_val, max_val=max_val)
        self.min_val = min_val
        self.max_val = max_val

    def shape(self, reward_tensor, is_per_token, prompt_ids, response_ids,
              finish_reason, metadata):
        return torch.clamp(reward_tensor, self.min_val, self.max_val), is_per_token


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------

def load_research_module(module_path: str):
    """Import a user module so its ``@register_*`` decorators execute."""
    importlib.import_module(module_path)
