import torch
import torch.nn.functional as F
from typing import Dict, Any

class WPO:
    """
    WPO: Weighted Preference Optimization.

    Reference: "WPO: Enhancing RLHF with Weighted Preference Optimization"
    (Zhou et al., EMNLP 2024). https://arxiv.org/abs/2406.11827

    WPO addresses the distributional gap between the off-policy data used for
    training and the current policy. It reweights preference pairs by their
    probability under the current policy, simulating on-policy learning from
    off-policy data.

    The key modification is a per-sample importance weight applied to the DPO
    loss:

        weight_i = clamp(exp(avg_logp_w_i + avg_logp_l_i), max=1)
        L_wpo = sum_i weight_i * L_dpo_i / sum_i weight_i

    where avg_logp_w and avg_logp_l are the average per-token log probabilities
    of the chosen and rejected responses under the current policy.

    Intuition: If the current policy assigns low probability to both the chosen
    and rejected response (the pair is "off-policy"), the weight is small,
    reducing that pair's influence on the gradient. If both responses are likely
    under the current policy (the pair is "on-policy"), the weight is close to 1.

    The weight is clamped to max=1 to prevent any single sample from dominating.

    When all weights are 1 (all data is on-policy), WPO reduces to standard DPO.

    Pro:  Better stability and less reward model overoptimization than DPO;
          consistent performance over more training epochs.
    Con:  Slightly more compute per step (need average log probs for weighting);
          extra clamping hyperparameter (fixed at 1 following the paper).
    """
    def __init__(self,
                model_engine,
                ref_model_engine,
                optimizer,
                beta=0.1,
                label_smoothing=0.0,
                use_cache=False,
                normalize_loss=False):

        self.model_engine = model_engine
        self.ref_model_engine = ref_model_engine
        self.optimizer = optimizer
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.use_cache = use_cache
        self.normalize_loss = normalize_loss

    def compute_logps(self, logits, target_ids, loss_mask):
        '''
           Computes log probabilities for the given logits and targets.
           logits: [B, T-1, vocab_size]
           target_ids: [B, T-1]
           loss_mask: [B, T-1]
           Returns:
               logps: [B] (sum of log probs)
        '''
        log_probs = F.log_softmax(logits, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=2, index=target_ids.unsqueeze(2)).squeeze(2)

        # Apply mask and sum across sequence length
        logps = (per_token_logps * loss_mask).sum(-1)
        return logps

    def compute_avg_logps(self, logits, target_ids, loss_mask):
        '''
           Computes average per-token log probabilities for WPO weighting.
           logits: [B, T-1, vocab_size]
           target_ids: [B, T-1]
           loss_mask: [B, T-1]
           Returns:
               avg_logps: [B] (mean of log probs over non-masked tokens)
        '''
        log_probs = F.log_softmax(logits, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=2, index=target_ids.unsqueeze(2)).squeeze(2)

        # Average over non-masked tokens
        mask_sum = loss_mask.sum(-1).clamp(min=1.0)
        avg_logps = (per_token_logps * loss_mask).sum(-1) / mask_sum
        return avg_logps

    def forward(self, input_ids, attn_mask, loss_mask, model_engine):
        '''
            Forward pass through the given model engine.
            Returns both sum and average log probabilities.
        '''
        token_type_ids = torch.zeros_like(input_ids)

        output = model_engine(input_ids=input_ids,
                              attention_mask=attn_mask,
                              token_type_ids=token_type_ids,
                              use_cache=self.use_cache)

        logits = output.logits[:, :-1, :].contiguous()
        target_ids = input_ids[:, 1:].contiguous()

        logps = self.compute_logps(logits, target_ids, loss_mask)
        avg_logps = self.compute_avg_logps(logits, target_ids, loss_mask)
        return logps, avg_logps

    def compute_loss(self, pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l,
                     pi_avg_logps_w, pi_avg_logps_l):
        '''
           WPO Loss:
           weight = clamp(exp(avg_logp_w + avg_logp_l), max=1)
           L = weighted_mean(-logsigmoid(beta * (logr_w - logr_l)))
        '''
        pi_logr_w = pi_logps_w - ref_logps_w
        pi_logr_l = pi_logps_l - ref_logps_l
        logits = self.beta * (pi_logr_w - pi_logr_l)

        # Compute per-sample importance weights
        # weight = clamp(exp(avg_logp_chosen + avg_logp_rejected), max=1)
        weights = torch.clamp(torch.exp(pi_avg_logps_w + pi_avg_logps_l), max=1.0)

        # DPO loss with optional label smoothing
        if self.label_smoothing > 0:
            per_sample_loss = -(
                (1 - self.label_smoothing) * F.logsigmoid(logits) +
                self.label_smoothing * F.logsigmoid(-logits)
            )
        else:
            per_sample_loss = -F.logsigmoid(logits)

        # Weighted mean
        weight_sum = weights.sum().clamp(min=1e-8)
        loss = (weights * per_sample_loss).sum() / weight_sum

        with torch.no_grad():
            rewards_w = self.beta * pi_logr_w
            rewards_l = self.beta * pi_logr_l
            margin = (rewards_w - rewards_l).mean()
            reward_acc = (rewards_w > rewards_l).float().mean()
            avg_weight = weights.mean()

        return loss, margin, reward_acc, avg_weight

    def train_step(self, micro_batch):
        '''
           One training step for WPO.
        '''
        self.model_engine.train()
        if self.ref_model_engine is not None:
            self.ref_model_engine.eval()

        batch_size = micro_batch['chosen_input_ids'].shape[0]
        input_ids = torch.cat([micro_batch['chosen_input_ids'], micro_batch['rejected_input_ids']], dim=0)
        attn_mask = torch.cat([micro_batch['chosen_attn_mask'], micro_batch['rejected_attn_mask']], dim=0)
        loss_mask = torch.cat([micro_batch['chosen_loss_mask'], micro_batch['rejected_loss_mask']], dim=0)

        # Reference model logps (only need sum, not avg)
        with torch.no_grad():
            ref_logps, _ = self.forward(input_ids, attn_mask, loss_mask, self.ref_model_engine)
            ref_logps_w, ref_logps_l = torch.split(ref_logps, batch_size, dim=0)

        # Policy model logps (need both sum and avg)
        pi_logps, pi_avg_logps = self.forward(input_ids, attn_mask, loss_mask, self.model_engine)
        pi_logps_w, pi_logps_l = torch.split(pi_logps, batch_size, dim=0)
        pi_avg_logps_w, pi_avg_logps_l = torch.split(pi_avg_logps, batch_size, dim=0)

        # Compute WPO loss
        loss, margin, reward_acc, avg_weight = self.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l,
            pi_avg_logps_w, pi_avg_logps_l)

        # Backward and Step
        self.model_engine.backward(loss)
        self.model_engine.step()

        return {
            "loss": loss.item(),
            "margin": margin.item(),
            "reward_acc": reward_acc.item(),
            "avg_weight": avg_weight.item(),
        }

    def eval_step(self, micro_batch):
        '''
           Validation step.
        '''
        self.model_engine.eval()
        with torch.no_grad():
            batch_size = micro_batch['chosen_input_ids'].shape[0]
            input_ids = torch.cat([micro_batch['chosen_input_ids'], micro_batch['rejected_input_ids']], dim=0)
            attn_mask = torch.cat([micro_batch['chosen_attn_mask'], micro_batch['rejected_attn_mask']], dim=0)
            loss_mask = torch.cat([micro_batch['chosen_loss_mask'], micro_batch['rejected_loss_mask']], dim=0)

            ref_logps, _ = self.forward(input_ids, attn_mask, loss_mask, self.ref_model_engine)
            ref_logps_w, ref_logps_l = torch.split(ref_logps, batch_size, dim=0)

            pi_logps, pi_avg_logps = self.forward(input_ids, attn_mask, loss_mask, self.model_engine)
            pi_logps_w, pi_logps_l = torch.split(pi_logps, batch_size, dim=0)
            pi_avg_logps_w, pi_avg_logps_l = torch.split(pi_avg_logps, batch_size, dim=0)

            loss, margin, reward_acc, avg_weight = self.compute_loss(
                pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l,
                pi_avg_logps_w, pi_avg_logps_l)

        return {
            "loss": loss.item(),
            "margin": margin.item(),
            "reward_acc": reward_acc.item(),
            "avg_weight": avg_weight.item(),
        }
