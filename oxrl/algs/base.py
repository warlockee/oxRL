import torch
from abc import ABC, abstractmethod
from typing import Dict, Any

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
    def save_checkpoint(self, output_dir: str, tag: str):
        """Save model weights and configuration."""
        pass
