"""
Shared helpers for all policy-gradient loss variants.

Pure functions — no class state, no side effects.
Input:  standard tensors [B, T-1].
Output: preprocessed tensors ready for surrogate computation.
"""
import torch
from typing import Tuple


def prepare_loss_inputs(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared preprocessing for all loss variants.

    Returns:
        adv:      [B, T-1] detached, float32
        mask:     [B, T-1] binary float
        denom:    scalar, sum of mask clamped >= 1
        logratio: [B, T-1] float32, log(pi / pi_old)
        ratio:    [B, T-1] float32, pi / pi_old
    """
    device = logprobs.device
    dtype = logprobs.dtype

    adv = advantages.detach().to(torch.float32)
    mask = (mask.to(device=device) > 0.5).to(dtype=dtype)
    denom = mask.sum().clamp(min=1.0)

    logratio = (logprobs - old_logprobs).to(torch.float32)
    ratio = torch.exp(logratio)

    return adv, mask, denom, logratio, ratio


def compute_kl_distance(
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
) -> torch.Tensor:
    """Variance-reduced KL divergence: log(pi/pi_ref) + pi_ref/pi - 1."""
    log_ratio = logprobs - ref_logprobs
    ratio_inv = torch.exp(ref_logprobs - logprobs)
    return log_ratio + ratio_inv - 1


def compute_entropy_and_kl(
    logprobs: torch.Tensor,
    entropies: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: torch.Tensor,
    denom: torch.Tensor,
    ent_coeff: float,
    kl_coeff: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute entropy bonus and KL penalty terms.

    Returns:
        loss_ent: scalar (positive; caller subtracts ent_coeff * loss_ent)
        kl_ref:   scalar (positive; caller adds kl_coeff * kl_ref)
    """
    device = logprobs.device
    dtype = logprobs.dtype

    loss_ent = torch.tensor(0.0, device=device, dtype=dtype)
    kl_ref = torch.tensor(0.0, device=device, dtype=dtype)

    if entropies is not None and ent_coeff > 0.0:
        loss_ent = (entropies * mask).sum() / denom

    if ref_logprobs is not None and kl_coeff > 0.0:
        kl_dist = compute_kl_distance(logprobs, ref_logprobs)
        kl_ref = (kl_dist * mask).sum() / denom

    return loss_ent, kl_ref


def token_level_metrics(
    ratio: torch.Tensor,
    logratio: torch.Tensor,
    mask: torch.Tensor,
    denom: torch.Tensor,
    clip_low: float,
    clip_high: float,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Token-level clip fraction and approx KL (used by sgrpo, cispo, ppo).

    Returns:
        clipfrac:  scalar
        approx_kl: scalar
    """
    clipped_mask = (ratio > (1.0 + clip_high)) | (ratio < (1.0 - clip_low))
    clipfrac = (clipped_mask.to(dtype=dtype) * mask).sum() / denom

    ratio_inv = torch.exp(-logratio)
    approx_kl_t = logratio + ratio_inv - 1.0
    approx_kl = (approx_kl_t.to(dtype=dtype) * mask).sum() / denom

    return clipfrac, approx_kl
