import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseAlgorithm(ABC):
    """
    Abstract Base Class for all oxRL algorithms.
    Provides a standardized interface for LLMs to implement new methods (like DPO).
    """
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Barrier method to ensure all Ray actors are initialized."""
        pass

    @abstractmethod
    def train_step(self, *args, **kwargs) -> Dict[str, float]:
        """Perform a single optimization step and return metrics."""
        pass

    @abstractmethod
    def save_checkpoint(self, output_dir: str, tag: str, state_dict_ref=None):
        """Save model weights and configuration.

        If state_dict_ref is provided (a Ray object ref), rank 0 retrieves and
        writes the pre-gathered state dict instead of doing ZeRO-3 gather.
        """
        pass

    def gather_state_dict(self) -> Optional[dict]:
        """Gather ZeRO-3 partitioned weights into a single state dict in memory.

        Collective operation — must be called on ALL ranks.
        Returns the state dict on rank 0, None on other ranks.
        Includes '__model_config_dict__' key with model config on rank 0.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement gather_state_dict. "
            "Falling back to disk-based checkpoint flow."
        )
