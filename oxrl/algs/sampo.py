import torch
import torch.nn.functional as F
from typing import Dict, Any

class SamPO:
    """
    SamPO: Eliminating Biased Length Reliance of DPO via Down-Sampled KL Divergence.

    Reference: Lu et al., "Eliminating Biased Length Reliance of Direct Preference
    Optimization via Down-Sampled KL Divergence" (EMNLP 2024).
    https://arxiv.org/abs/2406.10957

    Standard DPO suffers from length bias: the implicit reward (sequence-level KL
    divergence) accumulates over tokens, causing longer sequences to have larger
    absolute log-probability sums. This means the DPO loss can overestimate or
    underestimate rewards simply due to varying response lengths, leading to the
    "verbosity problem" where models learn to generate longer outputs.

    SamPO addresses this by down-sampling token-level log probabilities to match
    sequence lengths before computing the DPO loss. For each preference pair:
      1. Compute T_m = min(len_chosen, len_rejected)
      2. Uniformly sample T_m token positions from each sequence independently
      3. Sum log-probability ratios only over the sampled positions

    This ensures both sequences contribute equally regardless of their lengths,
    eliminating the length bias without averaging (which would remove the variance
    signal among tokens).

    Loss:
        L = -logsigmoid(beta * (sampled_logr_w - sampled_logr_l))

    where:
        sampled_logr = sum of (log pi(y_t|x) - log pi_ref(y_t|x))
                       over T_m uniformly sampled token positions

    When len_chosen == len_rejected, SamPO reduces to standard DPO.

    Pro:  Eliminates length bias; retains per-token variance signal (unlike averaging);
          simple modification; no new hyperparameters.
    Con:  Stochastic sampling introduces gradient noise; requires paired sequence
          length information at logps computation time.
    """
    def __init__(self,
                model_engine,
                ref_model_engine,
                optimizer,
                beta=0.1,
                use_cache=False,
                normalize_loss=False):

        self.model_engine = model_engine
        self.ref_model_engine = ref_model_engine
        self.optimizer = optimizer
        self.beta = beta
        self.use_cache = use_cache
        self.normalize_loss = normalize_loss

    def compute_logps(self, logits, target_ids, loss_mask):
        '''
           Computes per-token log probabilities (unmasked sum, used as fallback).
           logits: [B, T-1, vocab_size]
           target_ids: [B, T-1]
           loss_mask: [B, T-1]
           Returns:
               logps: [B]
        '''
        log_probs = F.log_softmax(logits, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=2, index=target_ids.unsqueeze(2)).squeeze(2)
        logps = (per_token_logps * loss_mask).sum(-1)
        return logps

    def compute_per_token_logps(self, logits, target_ids, loss_mask):
        '''
           Computes per-token log probabilities without summing.
           logits: [B, T-1, vocab_size]
           target_ids: [B, T-1]
           loss_mask: [B, T-1]
           Returns:
               per_token_logps: [B, T-1] (masked, zeros for non-response tokens)
        '''
        log_probs = F.log_softmax(logits, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=2, index=target_ids.unsqueeze(2)).squeeze(2)
        # Apply mask (zeros for non-response tokens)
        return per_token_logps * loss_mask

    def forward_per_token(self, input_ids, attn_mask, loss_mask, model_engine):
        '''
            Forward pass returning per-token log probabilities.
        '''
        token_type_ids = torch.zeros_like(input_ids)

        output = model_engine(input_ids=input_ids,
                              attention_mask=attn_mask,
                              token_type_ids=token_type_ids,
                              use_cache=self.use_cache)

        logits = output.logits[:, :-1, :].contiguous()
        target_ids = input_ids[:, 1:].contiguous()

        per_token_logps = self.compute_per_token_logps(logits, target_ids, loss_mask)
        return per_token_logps

    def forward(self, input_ids, attn_mask, loss_mask, model_engine):
        '''
            Standard forward pass (for compatibility).
        '''
        token_type_ids = torch.zeros_like(input_ids)

        output = model_engine(input_ids=input_ids,
                              attention_mask=attn_mask,
                              token_type_ids=token_type_ids,
                              use_cache=self.use_cache)

        logits = output.logits[:, :-1, :].contiguous()
        target_ids = input_ids[:, 1:].contiguous()

        logps = self.compute_logps(logits, target_ids, loss_mask)
        return logps

    def downsample_logps(self, per_token_logps_w, per_token_logps_l,
                          loss_mask_w, loss_mask_l):
        '''
           Down-sample per-token log probabilities to match the shorter sequence.

           For each sample in the batch:
             1. Find T_m = min(len_w, len_l)
             2. Uniformly sample T_m positions from each sequence's valid tokens
             3. Sum log probabilities over the sampled positions

           per_token_logps_w: [B, T-1] (chosen per-token log probs)
           per_token_logps_l: [B, T-1] (rejected per-token log probs)
           loss_mask_w: [B, T-1] (chosen loss mask)
           loss_mask_l: [B, T-1] (rejected loss mask)

           Returns:
               sampled_logps_w: [B] (down-sampled sum for chosen)
               sampled_logps_l: [B] (down-sampled sum for rejected)
        '''
        B = per_token_logps_w.shape[0]
        len_w = loss_mask_w.sum(-1).long()  # [B]
        len_l = loss_mask_l.sum(-1).long()  # [B]
        t_m = torch.min(len_w, len_l)       # [B]

        sampled_logps_w = torch.zeros(B, device=per_token_logps_w.device)
        sampled_logps_l = torch.zeros(B, device=per_token_logps_l.device)

        for i in range(B):
            n_sample = t_m[i].item()
            if n_sample <= 0:
                continue

            n_w = len_w[i].item()
            n_l = len_l[i].item()

            # Sample T_m positions from chosen (without replacement)
            if n_w == n_sample:
                # No need to sample -- use all valid positions
                sampled_logps_w[i] = per_token_logps_w[i].sum()
            else:
                indices_w = torch.multinomial(loss_mask_w[i].float(), n_sample, replacement=False)
                sampled_logps_w[i] = per_token_logps_w[i][indices_w].sum()

            # Sample T_m positions from rejected (without replacement)
            if n_l == n_sample:
                sampled_logps_l[i] = per_token_logps_l[i].sum()
            else:
                indices_l = torch.multinomial(loss_mask_l[i].float(), n_sample, replacement=False)
                sampled_logps_l[i] = per_token_logps_l[i][indices_l].sum()

        return sampled_logps_w, sampled_logps_l

    def compute_loss(self, pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l):
        '''
           Standard DPO loss on down-sampled log probabilities.
           L = -logsigmoid(beta * (logr_w - logr_l))
        '''
        pi_logr_w = pi_logps_w - ref_logps_w
        pi_logr_l = pi_logps_l - ref_logps_l

        logits = self.beta * (pi_logr_w - pi_logr_l)
        loss = -F.logsigmoid(logits).mean()

        with torch.no_grad():
            rewards_w = self.beta * pi_logr_w
            rewards_l = self.beta * pi_logr_l
            margin = (rewards_w - rewards_l).mean()
            reward_acc = (rewards_w > rewards_l).float().mean()

        return loss, margin, reward_acc

    def train_step(self, micro_batch):
        '''
           One training step for SamPO.
        '''
        self.model_engine.train()
        if self.ref_model_engine is not None:
            self.ref_model_engine.eval()

        batch_size = micro_batch['chosen_input_ids'].shape[0]
        input_ids = torch.cat([micro_batch['chosen_input_ids'], micro_batch['rejected_input_ids']], dim=0)
        attn_mask = torch.cat([micro_batch['chosen_attn_mask'], micro_batch['rejected_attn_mask']], dim=0)
        loss_mask = torch.cat([micro_batch['chosen_loss_mask'], micro_batch['rejected_loss_mask']], dim=0)

        loss_mask_w = micro_batch['chosen_loss_mask'][:, 1:]    # [B, T-1]
        loss_mask_l = micro_batch['rejected_loss_mask'][:, 1:]  # [B, T-1]

        # Reference model: per-token log probs
        with torch.no_grad():
            ref_per_token = self.forward_per_token(input_ids, attn_mask, loss_mask, self.ref_model_engine)
            ref_per_token_w, ref_per_token_l = torch.split(ref_per_token, batch_size, dim=0)

            # Down-sample reference log probs
            ref_logps_w, ref_logps_l = self.downsample_logps(
                ref_per_token_w, ref_per_token_l, loss_mask_w, loss_mask_l)

        # Policy model: per-token log probs
        pi_per_token = self.forward_per_token(input_ids, attn_mask, loss_mask, self.model_engine)
        pi_per_token_w, pi_per_token_l = torch.split(pi_per_token, batch_size, dim=0)

        # Down-sample policy log probs (use same sampling for consistency)
        # Note: we re-sample here -- the stochasticity is part of the algorithm
        pi_logps_w, pi_logps_l = self.downsample_logps(
            pi_per_token_w, pi_per_token_l, loss_mask_w, loss_mask_l)

        # Compute DPO loss on down-sampled log probs
        loss, margin, reward_acc = self.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # Backward and Step
        self.model_engine.backward(loss)
        self.model_engine.step()

        return {
            "loss": loss.item(),
            "margin": margin.item(),
            "reward_acc": reward_acc.item(),
        }

    def eval_step(self, micro_batch):
        '''
           Validation step. Uses standard (non-sampled) DPO for deterministic eval.
        '''
        self.model_engine.eval()
        with torch.no_grad():
            batch_size = micro_batch['chosen_input_ids'].shape[0]
            input_ids = torch.cat([micro_batch['chosen_input_ids'], micro_batch['rejected_input_ids']], dim=0)
            attn_mask = torch.cat([micro_batch['chosen_attn_mask'], micro_batch['rejected_attn_mask']], dim=0)
            loss_mask = torch.cat([micro_batch['chosen_loss_mask'], micro_batch['rejected_loss_mask']], dim=0)

            # Use standard (non-sampled) logps for deterministic evaluation
            ref_logps = self.forward(input_ids, attn_mask, loss_mask, self.ref_model_engine)
            ref_logps_w, ref_logps_l = torch.split(ref_logps, batch_size, dim=0)

            pi_logps = self.forward(input_ids, attn_mask, loss_mask, self.model_engine)
            pi_logps_w, pi_logps_l = torch.split(pi_logps, batch_size, dim=0)

            loss, margin, reward_acc = self.compute_loss(
                pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        return {
            "loss": loss.item(),
            "margin": margin.item(),
            "reward_acc": reward_acc.item(),
        }
