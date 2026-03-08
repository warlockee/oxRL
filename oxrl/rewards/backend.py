"""Reward backend abstraction.

RewardBackend ABC defines the uniform interface for all reward backends.
FunctionRewardBackend adapts bare reward functions to the interface.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch


class RewardBackend(ABC):
    """Abstract base class for reward backends.

    All reward backends must implement __call__ with the standard reward
    function signature. setup() and cleanup() are optional lifecycle hooks
    with no-op defaults.
    """

    @abstractmethod
    def __call__(
        self,
        prompt_ids: List[int],
        response_ids: List[int],
        finish_reason: str,
        metadata: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, bool]:
        """Score a response.

        Args:
            prompt_ids: Token IDs of the prompt.
            response_ids: Token IDs of the response.
            finish_reason: Why generation stopped (e.g. "stop", "length").
            metadata: Optional dict with extra info (e.g. prompt_text).

        Returns:
            (rewards, is_per_token): rewards tensor of shape [len(response_ids)],
            and bool indicating if rewards are per-token or sparse.
        """
        ...

    def setup(self, config=None):
        """Optional initialization hook. No-op by default."""
        pass

    def cleanup(self):
        """Optional teardown hook. No-op by default."""
        pass


class FunctionRewardBackend(RewardBackend):
    """Thin adapter wrapping a bare reward function as a RewardBackend.

    Stores a reference to the function, which should be a module-level
    callable for picklability.
    """

    def __init__(self, func):
        self._func = func

    def __call__(self, prompt_ids, response_ids, finish_reason, metadata=None):
        return self._func(prompt_ids, response_ids, finish_reason, metadata)

    def __repr__(self):
        name = getattr(self._func, "__name__", repr(self._func))
        return f"FunctionRewardBackend({name})"
