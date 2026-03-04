import torch
import torch.nn.functional as F
from typing import Dict, Any

class ChiPO:
    """
    Chi-Squared Preference Optimization (Chi-PO / XPO).

    Reference: Huang et al., "Correcting the Mythos of KL-Regularization:
    Direct Alignment without Overoptimization via Chi-Squared Preference
    Optimization" (ICLR 2025).
    https://arxiv.org/abs/2407.13399

    Standard DPO uses the logarithmic link function phi(z) = log(z), which
    corresponds to KL-divergence regularization. However, KL-regularization
    can lead to overoptimization because it does not sufficiently penalize
    policies that deviate far from the reference in low-density regions.

    Chi-PO replaces the log link with phi(z) = z + log(z), which corresponds
    to chi-squared divergence regularization. This "mixed" link function
    provides stronger penalization for deviations from the reference policy,
    implicitly implementing pessimism in the face of uncertainty.

    The loss is identical to DPO except for the link function:
        DPO:   logits = beta * (log(pi_w/ref_w) - log(pi_l/ref_l))
        ChiPO: logits = beta * (phi(pi_w/ref_w) - phi(pi_l/ref_l))
               where phi(z) = z + log(z)
               i.e., logits = beta * ((r_w + logr_w) - (r_l + logr_l))
               where r = pi/ref (probability ratio) and logr = log(pi/ref)

    This is truly a one-line change from DPO. When the policy is close to
    the reference (r ~= 1), phi(z) ~= 1 + log(z) ~= log(z) + const, so
    Chi-PO behaves similarly to DPO. But as r deviates, the linear term
    provides stronger gradient signal than log alone.

    Pro:  Provably alleviates overoptimization; simple one-line change;
          no new hyperparameters; strong theoretical guarantees.
    Con:  exp() of log-ratios can be numerically unstable for very large
          |logr| values; may need gradient clipping in practice.
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
           Computes log probabilities for the given logits and targets.
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

    def chi_link(self, logr):
        '''
           Chi-squared link function: phi(z) = z + log(z)
           where z = exp(logr) = pi/ref.

           phi(exp(logr)) = exp(logr) + logr

           Args:
               logr: log probability ratio log(pi/ref), shape [B]
           Returns:
               phi_value: exp(logr) + logr, shape [B]
        '''
        # Clamp logr to prevent numerical overflow in exp
        logr_clamped = torch.clamp(logr, min=-20.0, max=20.0)
        return torch.exp(logr_clamped) + logr

    def compute_loss(self, pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l):
        '''
           Chi-PO Loss: -logsigmoid(beta * (phi(r_w) - phi(r_l)))
           where phi(z) = z + log(z) and r = pi/ref.

           Equivalent to DPO with a modified link function.
        '''
        logr_w = pi_logps_w - ref_logps_w  # log(pi_w/ref_w)
        logr_l = pi_logps_l - ref_logps_l  # log(pi_l/ref_l)

        # Apply chi-squared link function: phi(exp(logr)) = exp(logr) + logr
        phi_w = self.chi_link(logr_w)
        phi_l = self.chi_link(logr_l)

        logits = self.beta * (phi_w - phi_l)
        loss = -F.logsigmoid(logits).mean()

        with torch.no_grad():
            rewards_w = self.beta * logr_w
            rewards_l = self.beta * logr_l
            margin = (rewards_w - rewards_l).mean()
            reward_acc = (rewards_w > rewards_l).float().mean()

        return loss, margin, reward_acc

    def train_step(self, micro_batch):
        '''
           One training step for Chi-PO.
        '''
        self.model_engine.train()
        if self.ref_model_engine is not None:
            self.ref_model_engine.eval()

        batch_size = micro_batch['chosen_input_ids'].shape[0]
        input_ids = torch.cat([micro_batch['chosen_input_ids'], micro_batch['rejected_input_ids']], dim=0)
        attn_mask = torch.cat([micro_batch['chosen_attn_mask'], micro_batch['rejected_attn_mask']], dim=0)
        loss_mask = torch.cat([micro_batch['chosen_loss_mask'], micro_batch['rejected_loss_mask']], dim=0)

        with torch.no_grad():
            ref_logps = self.forward(input_ids, attn_mask, loss_mask, self.ref_model_engine)
            ref_logps_w, ref_logps_l = torch.split(ref_logps, batch_size, dim=0)

        pi_logps = self.forward(input_ids, attn_mask, loss_mask, self.model_engine)
        pi_logps_w, pi_logps_l = torch.split(pi_logps, batch_size, dim=0)

        loss, margin, reward_acc = self.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

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
