import torch
import torch.nn.functional as F
from typing import Dict, Any

class AlphaDPO:
    """
    AlphaDPO: Adaptive Reward Margin for Direct Preference Optimization.

    Reference: Wu et al., "AlphaDPO: Adaptive Reward Margin is What Direct
    Preference Optimization Needs" (ICML 2025). https://arxiv.org/abs/2410.10148

    Standard DPO uses a fixed reference model, leading to a static implicit
    reward. SimPO avoids the reference model but assumes a fixed target reward
    margin. AlphaDPO bridges both approaches by introducing an adaptive,
    instance-level reward margin derived from an implicit reference model
    that interpolates between policy-driven specialization and uniform
    exploration.

    The implicit reference model is:
        pi_hat_ref(y|x) proportional to U(y|x) * (pi_theta(y|x) / pi_ref(y|x))^alpha

    This leads to an adaptive margin computed as:
        M(x, y_w, y_l) = pi_logratio_w - pi_logratio_l - (ref_logratio_w - ref_logratio_l)
                        = (pi_logr_w - ref_logr_w) - (pi_logr_l - ref_logr_l)

    which is Z-score normalized across the batch and scaled by alpha.

    The final loss is:
        L = -logsigmoid(beta * (u - sg[alpha * M_normalized + gamma_beta_ratio]))

    where:
        u = (beta/|y_w|)*log pi(y_w|x) - (beta/|y_l|)*log pi(y_l|x)
            (length-normalized policy log-prob difference)
        M_normalized = (gap - EMA_mean) / EMA_std  (Z-score of gap)
        gap = pi_logratios - ref_logratios (using length-normalized logps)
        sg[.] = stop-gradient
        gamma_beta_ratio = gamma / beta (base margin scaled by beta)

    Key differences from existing methods:
    - Unlike DPO: uses length-normalized logps and adaptive margin
    - Unlike SimPO: uses a reference model for the adaptive margin
    - Unlike AlphaPO (already in oxRL): AlphaPO transforms the reward shape
      via (1-p^{-alpha})/alpha; AlphaDPO uses Z-score normalized log-ratio
      gaps for adaptive margins. Completely different methods.

    alpha = 0 -> margin is just gamma_beta_ratio (similar to SimPO)
    alpha > 0 -> margin adapts based on per-sample preference strength

    Pro:  Adaptive per-sample margin; principled interpolation between
          exploration and exploitation; strong empirical results on
          AlpacaEval 2 and Arena-Hard.
    Con:  Requires reference model; EMA statistics need warm-up period;
          three hyperparameters (alpha, gamma_beta_ratio, beta) to tune.
    """
    def __init__(self,
                model_engine,
                ref_model_engine,
                optimizer,
                beta=2.5,
                alpha_dpo_alpha=0.1,
                alpha_dpo_gamma_beta_ratio=0.3,
                alpha_dpo_ema_decay=0.99,
                use_cache=False,
                normalize_loss=False):

        self.model_engine = model_engine
        self.ref_model_engine = ref_model_engine
        self.optimizer = optimizer
        self.beta = beta
        self.alpha = alpha_dpo_alpha
        self.gamma_beta_ratio = alpha_dpo_gamma_beta_ratio
        self.ema_decay = alpha_dpo_ema_decay
        self.use_cache = use_cache
        self.normalize_loss = normalize_loss

        # EMA running statistics for Z-score normalization
        self._ema_mean = 0.0
        self._ema_var = 1.0
        self._ema_initialized = False

    def compute_logps(self, logits, target_ids, loss_mask):
        '''
           Computes length-normalized log probabilities for the given logits and targets.
           logits: [B, T-1, vocab_size]
           target_ids: [B, T-1]
           loss_mask: [B, T-1]
           Returns:
               logps: [B]  (length-normalized)
        '''
        log_probs = F.log_softmax(logits, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=2, index=target_ids.unsqueeze(2)).squeeze(2)

        # Apply mask, sum, and length-normalize
        logps = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)
        return logps

    def forward(self, input_ids, attn_mask, loss_mask, model_engine):
        '''
            Forward pass through the given model engine.
            Returns length-normalized log probabilities.
        '''
        token_type_ids = torch.zeros_like(input_ids)

        output = model_engine(input_ids=input_ids,
                              attention_mask=attn_mask,
                              token_type_ids=token_type_ids,
                              use_cache=self.use_cache)

        # [B, T, V] -> [B, T-1, V]
        logits = output.logits[:, :-1, :].contiguous()
        # [B, T] -> [B, T-1]
        target_ids = input_ids[:, 1:].contiguous()

        logps = self.compute_logps(logits, target_ids, loss_mask)
        return logps

    def _update_ema(self, gap):
        '''
           Update exponential moving average of gap mean and variance.
           gap: [B] tensor of per-sample log-ratio gaps.
        '''
        batch_mean = gap.mean().item()
        batch_var = gap.var().item() if gap.numel() > 1 else 1.0

        if not self._ema_initialized:
            self._ema_mean = batch_mean
            self._ema_var = max(batch_var, 1e-8)
            self._ema_initialized = True
        else:
            d = self.ema_decay
            self._ema_mean = d * self._ema_mean + (1 - d) * batch_mean
            self._ema_var = d * self._ema_var + (1 - d) * batch_var

    def compute_loss(self, pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l):
        '''
           AlphaDPO Loss:
           1. Compute policy and reference log-ratio differences
           2. Compute gap = pi_logratios - ref_logratios
           3. Z-score normalize gap using EMA statistics
           4. Compute adaptive margin: alpha * normalized_gap + gamma_beta_ratio
           5. Loss = -logsigmoid(beta * (pi_logratios - sg[margin]))

           All log probabilities are length-normalized.
        '''
        # Policy log-ratio difference (length-normalized)
        pi_logratios = pi_logps_w - pi_logps_l

        # Reference log-ratio difference (length-normalized)
        ref_logratios = ref_logps_w - ref_logps_l

        # Gap between policy and reference log-ratio differences
        gap = pi_logratios - ref_logratios

        # Update EMA statistics (no gradient)
        with torch.no_grad():
            self._update_ema(gap.detach())

        # Z-score normalize the gap
        ema_std = max(self._ema_var, 1e-8) ** 0.5
        gap_normalized = (gap.detach() - self._ema_mean) / ema_std

        # Adaptive margin: alpha * normalized_gap + gamma_beta_ratio
        # Stop-gradient: the margin is a target, not optimized through
        adaptive_margin = self.alpha * gap_normalized + self.gamma_beta_ratio

        # Final logits: pi_logratios - adaptive_margin (both are in the same space)
        logits = self.beta * (pi_logratios - adaptive_margin)

        loss = -F.logsigmoid(logits).mean()

        with torch.no_grad():
            rewards_w = self.beta * pi_logps_w
            rewards_l = self.beta * pi_logps_l
            margin = (rewards_w - rewards_l).mean()
            reward_acc = (pi_logps_w > pi_logps_l).float().mean()

        return loss, margin, reward_acc

    def train_step(self, micro_batch):
        '''
           One training step for AlphaDPO.
        '''
        self.model_engine.train()
        if self.ref_model_engine is not None:
            self.ref_model_engine.eval()

        batch_size = micro_batch['chosen_input_ids'].shape[0]
        input_ids = torch.cat([micro_batch['chosen_input_ids'], micro_batch['rejected_input_ids']], dim=0)
        attn_mask = torch.cat([micro_batch['chosen_attn_mask'], micro_batch['rejected_attn_mask']], dim=0)
        loss_mask = torch.cat([micro_batch['chosen_loss_mask'], micro_batch['rejected_loss_mask']], dim=0)

        # Reference model logps (length-normalized)
        with torch.no_grad():
            ref_logps = self.forward(input_ids, attn_mask, loss_mask, self.ref_model_engine)
            ref_logps_w, ref_logps_l = torch.split(ref_logps, batch_size, dim=0)

        # Policy model logps (length-normalized)
        pi_logps = self.forward(input_ids, attn_mask, loss_mask, self.model_engine)
        pi_logps_w, pi_logps_l = torch.split(pi_logps, batch_size, dim=0)

        # Compute AlphaDPO loss
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

            loss, margin, reward_acc = self.compute_loss(
                pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        return {
            "loss": loss.item(),
            "margin": margin.item(),
            "reward_acc": reward_acc.item(),
        }
