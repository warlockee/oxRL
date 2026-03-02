"""
Tensor utility functions: shape validation, padding, dtype conversion.

Pure functions — no model state, no side effects.
"""
import torch


def ensure_1d(x: torch.Tensor, name: str) -> torch.Tensor:
    """Assert that x is 1-D; raise ValueError otherwise."""
    if x.dim() != 1:
        raise ValueError(f"Expected {name} to be 1D, got {x.dim()}D")
    return x


def pad_1d_to_length(
    x: torch.Tensor, pad_value: float, target_len: int
) -> torch.Tensor:
    """Pad or truncate 1-D tensor x to exactly target_len elements."""
    seq_len = x.numel()

    if seq_len > target_len:
        return x[:target_len]

    if seq_len < target_len:
        pad = torch.full(
            (target_len - seq_len,),
            pad_value,
            dtype=x.dtype,
            device=x.device,
        )
        return torch.cat([x, pad], dim=0)

    return x
