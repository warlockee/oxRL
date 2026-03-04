import torch
import torch.nn.functional as F
from typing import Dict, Any

class DPOShift:
    """
    DPO-Shift: Shifting the Distribution of Direct Preference Optimization.

    Reference: "DPO-Shift: Shifting the Distribution of Direct Preference
    Optimization" (2025). https://arxiv.org/abs/2502.07599

    Standard DPO suffers from "likelihood displacement" -- the probability
    of chosen responses often decreases during training. This happens because
    DPO's gradient simultaneously pushes up chosen probability and pushes down
    rejected probability equally, and the push-down on rejected can dominate.

    DPO-Shift addresses this by scaling down the rejected term:

        h = beta * logr_w - shift_lambda * beta * logr_l

    where shift_lambda in (0, 1) reduces the influence of the rejected
    response on the gradient. This creates a controllable trade-off:
    - Lower shift_lambda: chosen probability increases more (less displacement)
      but reward margin may decrease.
    - shift_lambda = 1.0: standard DPO.

    The insight is that by reducing the "confrontation" between chosen and
    rejected responses, the model can increase chosen probability without
    as much interference from the rejected term.

    Loss:
        L = -logsigmoid(beta * logr_w - shift_lambda * beta * logr_l)

    where:
        logr_w = log(pi(y_w|x) / pi_ref(y_w|x))
        logr_l = log(pi(y_l|x) / pi_ref(y_l|x))
        shift_lambda in (0, 1]

    When shift_lambda = 1, this reduces to standard DPO.

    Pro:  Simple modification; addresses likelihood displacement; controllable
          trade-off between chosen probability and reward margin.
    Con:  Smaller shift_lambda reduces reward margin; optimal value is
          task-dependent.
    """
    def __init__(self,
                model_engine,
                ref_model_engine,
                optimizer,
                beta=0.1,
                shift_lambda=0.5,
                use_cache=False,
                normalize_loss=False):

        self.model_engine = model_engine
        self.ref_model_engine = ref_model_engine
        self.optimizer = optimizer
        self.beta = beta
        self.shift_lambda = shift_lambda
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
           DPO-Shift Loss:
           h = beta * logr_w - shift_lambda * beta * logr_l
           L = -logsigmoid(h)
        '''
        pi_logr_w = pi_logps_w - ref_logps_w
        pi_logr_l = pi_logps_l - ref_logps_l

        # DPO-Shift: scale rejected term by shift_lambda
        logits = self.beta * pi_logr_w - self.shift_lambda * self.beta * pi_logr_l
        loss = -F.logsigmoid(logits).mean()

        with torch.no_grad():
            rewards_w = self.beta * pi_logr_w
            rewards_l = self.beta * pi_logr_l
            margin = (rewards_w - rewards_l).mean()
            reward_acc = (rewards_w > rewards_l).float().mean()

        return loss, margin, reward_acc

    def train_step(self, micro_batch):
        '''
           One training step for DPO-Shift.
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

        # Compute DPO-Shift loss
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
