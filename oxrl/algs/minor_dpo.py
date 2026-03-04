import torch
import torch.nn.functional as F
from typing import Dict, Any

class MinorDPO:
    """
    Minor DPO: DPO with Clamped Reject Penalty.

    Reference: "Minor DPO reject penalty to increase training robustness"
    (Xie & Chen, 2024). arXiv:2408.09834.

    Standard DPO uses the full log-ratio for both chosen and rejected:
        logits = beta * (logr_w - logr_l)

    This can lead to an over-penalty problem: once the model has successfully
    learned to assign lower probability to rejected responses than the reference
    model (logr_l < 0), DPO continues to push rejected probabilities even lower,
    wasting optimization capacity and potentially destabilizing training.

    Minor DPO clamps the rejected log-ratio to be non-negative:
        logits = beta * (logr_w - max(0, logr_l))

    This means the rejected penalty only applies when pi(y_l|x) > pi_ref(y_l|x).
    Once the model's rejected probability drops below the reference, the penalty
    stops, allowing the optimizer to focus on improving chosen responses.

    Loss:
        L = -logsigmoid(beta * (logr_w - max(0, logr_l))).mean()

    where logr_w = log pi(y_w|x) - log pi_ref(y_w|x)
          logr_l = log pi(y_l|x) - log pi_ref(y_l|x)

    Key properties:
        - No new hyperparameters (same interface as DPO)
        - One-line change from DPO: clamp rejected log-ratio
        - Eases over-penalty on rejected responses
        - Focuses optimization on improving chosen responses
        - More robust training, especially with small edit-distance pairs

    Pro:  Reduces over-optimization of rejected responses; more robust training.
    Con:  May converge slightly slower since rejected penalty is weaker.
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

    def compute_loss(self, pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l):
        '''
           Minor DPO Loss: -logsigmoid(beta * (logr_w - max(0, logr_l)))
           where logr = log(pi / pi_ref)
        '''
        pi_logr_w = pi_logps_w - ref_logps_w
        pi_logr_l = pi_logps_l - ref_logps_l

        # Minor DPO: clamp rejected log-ratio to be non-negative
        # This stops the rejected penalty once pi(y_l) < pi_ref(y_l)
        clamped_logr_l = torch.clamp(pi_logr_l, min=0.0)

        logits = self.beta * (pi_logr_w - clamped_logr_l)
        loss = -F.logsigmoid(logits).mean()

        with torch.no_grad():
            rewards_w = self.beta * pi_logr_w
            rewards_l = self.beta * pi_logr_l
            margin = (rewards_w - rewards_l).mean()
            reward_acc = (rewards_w > rewards_l).float().mean()

        return loss, margin, reward_acc

    def train_step(self, micro_batch):
        '''
           One training step for Minor DPO.
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
