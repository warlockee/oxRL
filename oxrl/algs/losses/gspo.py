"""
GSPO: Sequence-level clipped surrogate loss for MoE models.

Averages log-ratios across the sequence before clipping, so per-token
MoE expert-routing noise cancels out. Use for MoE models like
Qwen3-MoE and DeepSeek-V3 where token-level log-probs between vLLM
and HF can diverge due to different expert-selection implementations.
"""
import torch
from typing import Dict, Tuple

from oxrl.algs.losses.common import (
    prepare_loss_inputs,
    compute_entropy_and_kl,
)


def compute_gspo_loss(
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
    """Sequence-level clipped surrogate (GSPO).

    All inputs are [B, T-1] except scalars.
    Returns (loss_total, metrics_dict).
    """
    adv, mask, denom, logratio, ratio = prepare_loss_inputs(
        logprobs, old_logprobs, advantages, mask
    )

    # Sequence-level: average log-ratio per sequence, then clip
    seq_lens = mask.sum(dim=-1).clamp(min=1.0)                        # [B]
    seq_logratio = (logratio * mask).sum(dim=-1) / seq_lens           # [B]
    seq_ratio = torch.exp(seq_logratio)                               # [B]
    seq_adv = (adv * mask).sum(dim=-1) / seq_lens                     # [B]

    seq_unclipped = seq_ratio * seq_adv                               # [B]
    seq_clip_adv = torch.clamp(
        seq_ratio, 1.0 - clip_low, 1.0 + clip_high
    ) * seq_adv                                                       # [B]
    loss_pi = -torch.minimum(seq_unclipped, seq_clip_adv).mean()      # scalar

    # Entropy bonus + KL penalty
    loss_ent, kl_ref = compute_entropy_and_kl(
        logprobs, entropies, ref_logprobs, mask, denom, ent_coeff, kl_coeff
    )

    loss_total = loss_pi - ent_coeff * loss_ent + kl_coeff * kl_ref

    # Sequence-level metrics
    with torch.no_grad():
        dtype = logprobs.dtype
        seq_clipped = (seq_ratio > (1.0 + clip_high)) | (seq_ratio < (1.0 - clip_low))
        clipfrac = seq_clipped.to(dtype=dtype).mean()
        seq_ratio_inv = torch.exp(-seq_logratio)
        approx_kl = (seq_logratio + seq_ratio_inv - 1.0).to(dtype=dtype).mean()

        metrics = {
            "clipfrac": clipfrac.item(),
            "kl_old": approx_kl.item(),
            "loss_ent": loss_ent.item(),
            "loss_pi": loss_pi.item(),
            "loss_total": loss_total.item(),
            "kl_ref": kl_ref.item(),
        }

    return loss_total, metrics
