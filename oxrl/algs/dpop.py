import torch
import torch.nn.functional as F
from typing import Dict, Any

class DPOP:
    """
    DPOP: DPO-Positive (Smaug).

    Reference: "Smaug: Fixing Failure Modes of Preference Optimisation with
    DPO-Positive" (Pal et al., 2024). https://arxiv.org/abs/2402.13228

    Standard DPO can reduce the likelihood of preferred responses, which is
    a well-known failure mode. DPOP adds a penalty term that prevents the
    model from reducing the chosen response probability below the reference
    model's probability.

    Loss:
        L_dpop = -logsigmoid(beta * (logr_w - logr_l
                                      - lambda * max(0, -logr_w)))

    where:
        logr_w = log(pi(y_w|x) / pi_ref(y_w|x))
        logr_l = log(pi(y_l|x) / pi_ref(y_l|x))
        lambda = penalty strength (dpop_lambda)

    The penalty max(0, -logr_w) is zero when pi(y_w|x) >= pi_ref(y_w|x),
    and positive when the chosen probability drops below reference. This
    ensures the model maintains high likelihood for preferred responses.

    When lambda=0, this reduces to standard DPO.

    Pro:  Prevents chosen likelihood degradation; especially effective with
          low-edit-distance preference pairs.
    Con:  May slow convergence by preventing some beneficial exploration;
          extra hyperparameter lambda to tune.
    """
    def __init__(self,
                model_engine,
                ref_model_engine,
                optimizer,
                beta=0.1,
                dpop_lambda=5.0,
                use_cache=False,
                normalize_loss=False):

        self.model_engine = model_engine
        self.ref_model_engine = ref_model_engine
        self.optimizer = optimizer
        self.beta = beta
        self.dpop_lambda = dpop_lambda
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
           DPOP Loss:
           L = -logsigmoid(beta * (logr_w - logr_l - lambda * max(0, -logr_w)))
        '''
        pi_logr_w = pi_logps_w - ref_logps_w
        pi_logr_l = pi_logps_l - ref_logps_l

        # Penalty: max(0, -logr_w) = max(0, log(pi_ref/pi_theta))
        # Active when pi(y_w|x) < pi_ref(y_w|x)
        penalty = torch.relu(-pi_logr_w)

        logits = self.beta * (pi_logr_w - pi_logr_l - self.dpop_lambda * penalty)
        loss = -F.logsigmoid(logits).mean()

        with torch.no_grad():
            rewards_w = self.beta * pi_logr_w
            rewards_l = self.beta * pi_logr_l
            margin = (rewards_w - rewards_l).mean()
            reward_acc = (rewards_w > rewards_l).float().mean()

        return loss, margin, reward_acc

    def train_step(self, micro_batch):
        '''
           One training step for DPOP.
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

        # Compute DPOP loss
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
