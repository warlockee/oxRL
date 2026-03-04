import torch
import torch.nn.functional as F
from typing import Dict, Any

class HDPO:
    """
    H-DPO: Entropy Controllable Direct Preference Optimization.

    Reference: "Entropy Controllable Direct Preference Optimization"
    (2024). https://arxiv.org/abs/2411.07595

    Standard DPO implicitly determines the entropy of the resulting policy
    through the beta parameter and reference model. H-DPO introduces an
    additional parameter alpha that explicitly controls the entropy of the
    optimized policy.

    Loss:
        L = -logsigmoid(alpha * beta * log(pi/pi_ref)(y_w|x)
                        - beta * log(pi_ref)(y_w|x) / log(pi_ref)(y_l|x))

    Equivalently (and more simply):
        L = -logsigmoid(alpha * beta * logr_w - alpha * beta * logr_l
                        + (alpha - 1) * beta * (logr_w_ref_component))

    The simplest implementation: multiply the policy log-ratio coefficient
    by alpha:
        h = alpha * beta * (pi_logr_w - pi_logr_l)
            - (alpha - 1) * beta * (ref_logr_w - ref_logr_l)

    But even simpler from the paper: the DPO logits become:
        h = alpha * beta * log(pi(y_w)/pi(y_l))
            - beta * log(pi_ref(y_w)/pi_ref(y_l))

    Where:
        alpha < 1: Lower entropy (sharper, more mode-seeking). The policy
                   concentrates more on high-reward modes.
        alpha = 1: Standard DPO (no entropy modification).
        alpha > 1: Higher entropy (more diverse, more exploratory). The
                   policy spreads probability more broadly.

    The optimal policy under H-DPO is:
        pi*(y|x) = (1/Z) * pi_ref(y|x)^(1/alpha) * exp(r(x,y)/(alpha*beta))

    Pro:  Very simple modification (one extra multiplier); explicit entropy
          control prevents mode collapse or over-exploration.
    Con:  Requires tuning alpha; the effect of alpha depends on the task
          and reference model quality.
    """
    def __init__(self,
                model_engine,
                ref_model_engine,
                optimizer,
                beta=0.1,
                hdpo_alpha=1.0,
                use_cache=False,
                normalize_loss=False):

        self.model_engine = model_engine
        self.ref_model_engine = ref_model_engine
        self.optimizer = optimizer
        self.beta = beta
        self.hdpo_alpha = hdpo_alpha
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

    def compute_loss(self, pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l):
        '''
           H-DPO Loss:
           h = alpha * beta * log(pi(y_w)/pi(y_l))
               - beta * log(pi_ref(y_w)/pi_ref(y_l))
           L = -logsigmoid(h)
        '''
        # Policy log ratios (pi relative to uniform, then combined)
        pi_log_ratio = pi_logps_w - pi_logps_l
        ref_log_ratio = ref_logps_w - ref_logps_l

        # H-DPO: multiply policy coefficient by alpha
        logits = self.hdpo_alpha * self.beta * pi_log_ratio - self.beta * ref_log_ratio

        loss = -F.logsigmoid(logits).mean()

        with torch.no_grad():
            pi_logr_w = pi_logps_w - ref_logps_w
            pi_logr_l = pi_logps_l - ref_logps_l
            rewards_w = self.beta * pi_logr_w
            rewards_l = self.beta * pi_logr_l
            margin = (rewards_w - rewards_l).mean()
            reward_acc = (rewards_w > rewards_l).float().mean()

        return loss, margin, reward_acc

    def train_step(self, micro_batch):
        '''
           One training step for H-DPO.
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

        # Compute H-DPO loss
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
