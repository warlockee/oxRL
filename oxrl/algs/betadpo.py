import torch
import torch.nn.functional as F
from typing import Dict, Any

class BetaDPO:
    """
    beta-DPO: Direct Preference Optimization with Dynamic Beta.

    Reference: "beta-DPO: Direct Preference Optimization with Dynamic beta"
    (Wu et al., NeurIPS 2024). https://arxiv.org/abs/2407.08639

    beta-DPO extends standard DPO by dynamically calibrating the beta
    parameter at the sample level based on data quality. The key insight
    is that optimal beta values vary with the informativeness of each
    pairwise comparison. Samples with advantage gaps far from the running
    mean need different regularization strengths.

    Dynamic beta formula:
        beta_i = beta * (1 + alpha * (A_i - mu_A))

    where:
        A_i = log_ratio_w - log_ratio_l  (advantage gap for sample i)
        mu_A = exponential moving average of advantage gaps across batches
        alpha = scaling factor controlling beta adaptation strength
        beta = base beta parameter (same as standard DPO)

    Loss:
        L = -mean(logsigmoid(beta_i * (log_ratio_w - log_ratio_l)))

    When alpha=0, this reduces to standard DPO.

    Pro:  Adapts regularization strength per-sample based on data quality;
          more robust to heterogeneous preference data; handles outliers.
    Con:  Introduces one additional hyperparameter (alpha); requires running
          statistics tracking.
    """
    def __init__(self,
                model_engine,
                ref_model_engine,
                optimizer,
                beta=0.1,
                betadpo_alpha=0.5,
                betadpo_ema_gamma=0.9,
                use_cache=False,
                normalize_loss=False):

        self.model_engine = model_engine
        self.ref_model_engine = ref_model_engine
        self.optimizer = optimizer
        self.beta = beta
        self.alpha = betadpo_alpha
        self.ema_gamma = betadpo_ema_gamma
        self.use_cache = use_cache
        self.normalize_loss = normalize_loss

        # Running statistics for advantage gap (exponential moving average)
        self._gap_mean = 0.0
        self._initialized = False

    def compute_logps(self, logits, target_ids, loss_mask):
        '''
           Computes log probabilities for the given logits and targets.
           logits: [B, T-1, vocab_size]
           target_ids: [B, T-1]
           loss_mask: [B, T-1]
           Returns:
               logps: [B]
        '''
        log_probs = F.log_softmax(logits, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=2, index=target_ids.unsqueeze(2)).squeeze(2)

        # Apply mask and sum across sequence length
        logps = (per_token_logps * loss_mask).sum(-1)
        return logps

    def forward(self, input_ids, attn_mask, loss_mask, model_engine):
        '''
            Forward pass through the given model engine.
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

    def _update_gap_stats(self, gap):
        """Update exponential moving average of the advantage gap."""
        batch_mean = gap.mean().item()
        if not self._initialized:
            self._gap_mean = batch_mean
            self._initialized = True
        else:
            self._gap_mean = (
                self.ema_gamma * self._gap_mean
                + (1 - self.ema_gamma) * batch_mean
            )

    def compute_loss(self, pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l):
        '''
           beta-DPO Loss with dynamic per-sample beta:
           A_i = log_ratio_w_i - log_ratio_l_i
           beta_i = beta * (1 + alpha * (A_i - gap_mean))
           loss = -mean(logsigmoid(beta_i * A_i))
        '''
        pi_logr_w = pi_logps_w - ref_logps_w
        pi_logr_l = pi_logps_l - ref_logps_l

        # Advantage gap per sample
        advantage_gap = pi_logr_w - pi_logr_l

        # Dynamic per-sample beta
        # beta_i = beta * (1 + alpha * (A_i - gap_mean))
        dynamic_beta = self.beta * (
            1.0 + self.alpha * (advantage_gap - self._gap_mean)
        )
        # Clamp to prevent negative or extremely small beta
        dynamic_beta = torch.clamp(dynamic_beta, min=1e-3)

        # Update running statistics
        self._update_gap_stats(advantage_gap.detach())

        # Per-sample weighted DPO loss
        logits = dynamic_beta * advantage_gap
        loss = -F.logsigmoid(logits).mean()

        with torch.no_grad():
            rewards_w = self.beta * pi_logr_w
            rewards_l = self.beta * pi_logr_l
            margin = (rewards_w - rewards_l).mean()
            reward_acc = (rewards_w > rewards_l).float().mean()
            mean_dynamic_beta = dynamic_beta.mean()

        return loss, margin, reward_acc, mean_dynamic_beta

    def train_step(self, micro_batch):
        '''
           One training step for beta-DPO.
        '''
        self.model_engine.train()
        if self.ref_model_engine is not None:
            self.ref_model_engine.eval()

        batch_size = micro_batch['chosen_input_ids'].shape[0]
        input_ids = torch.cat([micro_batch['chosen_input_ids'], micro_batch['rejected_input_ids']], dim=0)
        attn_mask = torch.cat([micro_batch['chosen_attn_mask'], micro_batch['rejected_attn_mask']], dim=0)
        loss_mask = torch.cat([micro_batch['chosen_loss_mask'], micro_batch['rejected_loss_mask']], dim=0)

        # Reference model logps
        with torch.no_grad():
            ref_logps = self.forward(input_ids, attn_mask, loss_mask, self.ref_model_engine)
            ref_logps_w, ref_logps_l = torch.split(ref_logps, batch_size, dim=0)

        # Policy model logps
        pi_logps = self.forward(input_ids, attn_mask, loss_mask, self.model_engine)
        pi_logps_w, pi_logps_l = torch.split(pi_logps, batch_size, dim=0)

        # Compute beta-DPO loss
        loss, margin, reward_acc, mean_beta = self.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # Backward and Step
        self.model_engine.backward(loss)
        self.model_engine.step()

        return {
            "loss": loss.item(),
            "margin": margin.item(),
            "reward_acc": reward_acc.item(),
            "dynamic_beta": mean_beta.item(),
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

            ref_logps = self.forward(input_ids, attn_mask, loss_mask, self.ref_model_engine)
            ref_logps_w, ref_logps_l = torch.split(ref_logps, batch_size, dim=0)

            pi_logps = self.forward(input_ids, attn_mask, loss_mask, self.model_engine)
            pi_logps_w, pi_logps_l = torch.split(pi_logps, batch_size, dim=0)

            loss, margin, reward_acc, mean_beta = self.compute_loss(
                pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        return {
            "loss": loss.item(),
            "margin": margin.item(),
            "reward_acc": reward_acc.item(),
            "dynamic_beta": mean_beta.item(),
        }
