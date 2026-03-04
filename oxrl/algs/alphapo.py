import torch
import torch.nn.functional as F
from typing import Dict, Any

class AlphaPO:
    """
    AlphaPO: Reward Shape Matters for LLM Alignment.

    Reference: "AlphaPO -- Reward shape matters for LLM alignment"
    (Gupta et al., 2025). https://arxiv.org/abs/2501.03884

    AlphaPO generalizes SimPO by introducing an alpha parameter that
    controls the shape of the implicit reward function. The key insight
    is that different reward shapes lead to different training dynamics
    and alignment performance.

    Reward transformation:
        When alpha != 0:  r(p) = (1 - p^{-alpha}) / alpha
        When alpha == 0:  r(p) = log(p)  (reduces to SimPO)

    where p = exp(avg_logps) is the average token probability.

    This transformation helps maintain fine-grained control over
    likelihood displacement and overoptimization.

    Like SimPO, AlphaPO is reference-free (no ref_model needed) and
    uses length-normalized log probabilities.
    """
    def __init__(self,
                model_engine,
                optimizer,
                beta=0.1,
                gamma=0.5,
                alpha=1.0,
                use_cache=False):

        self.model_engine = model_engine
        self.optimizer = optimizer
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.use_cache = use_cache

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
        # [B, T-1] * [B, T-1] -> [B, T-1] -> [B]
        logps = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)
        return logps

    def forward(self, input_ids, attn_mask, loss_mask):
        '''
            Forward pass through the model engine.
        '''
        token_type_ids = torch.zeros_like(input_ids)

        output = self.model_engine(input_ids=input_ids,
                                   attention_mask=attn_mask,
                                   token_type_ids=token_type_ids,
                                   use_cache=self.use_cache)

        # [B, T, V] -> [B, T-1, V]
        logits = output.logits[:, :-1, :].contiguous()
        # [B, T] -> [B, T-1]
        target_ids = input_ids[:, 1:].contiguous()

        logps = self.compute_logps(logits, target_ids, loss_mask)
        return logps

    def logps_to_rewards(self, logps):
        '''
           Transform length-normalized log-probs into rewards using AlphaPO.

           When alpha != 0:  r = (1 - p^{-alpha}) / alpha
           When alpha == 0:  r = log(p) = logps  (standard SimPO)

           where p = exp(logps).

           The clamp prevents numerical overflow for very small probabilities.
        '''
        if self.alpha == 0.0:
            # Reduce to standard SimPO: reward = log(p) = logps
            return logps

        # p = exp(logps), need p^{-alpha} = exp(-alpha * logps)
        # Clamp to prevent overflow: if logps is very negative (p near 0),
        # -alpha * logps can be very large.
        neg_alpha_logps = (-self.alpha * logps).clamp(max=50.0)
        p_neg_alpha = torch.exp(neg_alpha_logps)
        rewards = (1.0 - p_neg_alpha) / self.alpha
        return rewards

    def compute_loss(self, pi_logps_w, pi_logps_l):
        '''
           AlphaPO Loss: -log(sigmoid(beta * (r_w - r_l) - gamma))

           where r_w, r_l are the alpha-transformed rewards.
        '''
        # Transform logps to rewards using AlphaPO
        r_w = self.logps_to_rewards(pi_logps_w)
        r_l = self.logps_to_rewards(pi_logps_l)

        logits = self.beta * (r_w - r_l) - self.gamma
        loss = -F.logsigmoid(logits).mean()

        with torch.no_grad():
            margin = (r_w - r_l).mean()
            logps_margin = (pi_logps_w - pi_logps_l).mean()

        return loss, margin, logps_margin

    def train_step(self, micro_batch):
        '''
           One training step for AlphaPO.
           micro_batch contains: chosen_input_ids, rejected_input_ids, ...
        '''
        self.model_engine.train()

        # Combine chosen and rejected inputs for a single forward pass
        batch_size = micro_batch['chosen_input_ids'].shape[0]
        input_ids = torch.cat([micro_batch['chosen_input_ids'], micro_batch['rejected_input_ids']], dim=0)
        attn_mask = torch.cat([micro_batch['chosen_attn_mask'], micro_batch['rejected_attn_mask']], dim=0)
        loss_mask = torch.cat([micro_batch['chosen_loss_mask'], micro_batch['rejected_loss_mask']], dim=0)

        # Policy model logps (length-normalized)
        pi_logps = self.forward(input_ids, attn_mask, loss_mask)
        pi_logps_w, pi_logps_l = torch.split(pi_logps, batch_size, dim=0)

        # Compute AlphaPO loss
        loss, margin, logps_margin = self.compute_loss(pi_logps_w, pi_logps_l)

        # Backward and Step
        self.model_engine.backward(loss)
        self.model_engine.step()

        return {"loss": loss.item(), "margin": margin.item(), "logps_margin": logps_margin.item()}

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

            pi_logps = self.forward(input_ids, attn_mask, loss_mask)
            pi_logps_w, pi_logps_l = torch.split(pi_logps, batch_size, dim=0)

            loss, margin, logps_margin = self.compute_loss(pi_logps_w, pi_logps_l)

        return {"loss": loss.item(), "margin": margin.item(), "logps_margin": logps_margin.item()}
