"""
CISPO: Conservative Indirect Surrogate Policy Optimization.

Uses the clipped ratio as a detached weight on log-probabilities (policy-
gradient style). More conservative than SGRPO because the ratio does NOT
flow gradients — only the log-prob term receives gradient signal.

Use when SGRPO shows reward hacking or unstable updates.
"""
import torch
from typing import Dict, Tuple

from oxrl.algs.losses.common import (
    prepare_loss_inputs,
    compute_entropy_and_kl,
    token_level_metrics,
)


def compute_cispo_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    entropies: torch.Tensor,
    ref_logprobs: torch.Tensor,
    clip_low: float,
    clip_high: float,
    ent_coeff: float,
    kl_coeff: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Conservative indirect policy loss (CISPO).

    All inputs are [B, T-1] except scalars.
    Returns (loss_total, metrics_dict).
    """
    adv, mask, denom, logratio, ratio = prepare_loss_inputs(
        logprobs, old_logprobs, advantages, mask
    )

    # Detached clipped ratio * logratio (no grad through ratio weight)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
    loss_pi = -(clipped_ratio.detach() * logratio * adv * mask).sum() / denom

    # Entropy bonus + KL penalty
    loss_ent, kl_ref = compute_entropy_and_kl(
        logprobs, entropies, ref_logprobs, mask, denom, ent_coeff, kl_coeff
    )

    loss_total = loss_pi - ent_coeff * loss_ent + kl_coeff * kl_ref

    # Metrics (token-level, same as sgrpo)
    with torch.no_grad():
        clipfrac, approx_kl = token_level_metrics(
            ratio, logratio, mask, denom, clip_low, clip_high, logprobs.dtype
        )
        metrics = {
            "clipfrac": clipfrac.item(),
            "kl_old": approx_kl.item(),
            "loss_ent": loss_ent.item(),
            "loss_pi": loss_pi.item(),
            "loss_total": loss_total.item(),
            "kl_ref": kl_ref.item(),
        }

    return loss_total, metrics
