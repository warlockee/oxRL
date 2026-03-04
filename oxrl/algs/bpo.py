import torch
import torch.nn.functional as F
from typing import Dict, Any

class BPO:
    """
    BPO: Balanced Preference Optimization.

    Reference: Sun et al., "BPO: Revisiting Preference Modeling in Direct
    Preference Optimization" (2025). https://arxiv.org/abs/2506.03557

    Standard DPO uses the relative margin (logr_w - logr_l) which treats
    the chosen improvement and rejected suppression equally. However, this
    can lead to "Degraded Chosen Responses" (DCR) where the model focuses
    too much on pushing down rejected responses while letting chosen
    response quality degrade.

    BPO addresses this by replacing the relative margin with a balanced
    margin that dynamically prioritizes the weaker component:

        balanced_logits = min(logr_w, -logr_l)

    This means:
    - When logr_w < -logr_l: the chosen improvement is the bottleneck,
      so the gradient focuses on improving chosen responses.
    - When -logr_l < logr_w: the rejected suppression is the bottleneck,
      so the gradient focuses on pushing down rejected responses.

    The "alpha" variant adds a balance_factor that scales the rejected
    term before taking the minimum:

        balanced_logits = min(logr_w, balance_factor * (-logr_l))

    With balance_factor < 1.0, this biases toward prioritizing the chosen
    improvement (since the rejected term is scaled down, it's less likely
    to be the minimum).

    Loss:
        L = -logsigmoid(beta * min(logr_w, balance_factor * (-logr_l)))

    where:
        logr_w = log(pi(y_w|x) / pi_ref(y_w|x))
        logr_l = log(pi(y_l|x) / pi_ref(y_l|x))
        balance_factor in (0, 1] controls chosen-rejected trade-off

    When balance_factor = 1.0 and min always selects logr_w - logr_l
    (i.e., logr_w < -logr_l), this approaches standard DPO behavior.

    Pro:  Simple one-line modification; prevents chosen degradation;
          dynamically balances optimization focus; no new dependencies.
    Con:  min() makes loss non-smooth (gradient discontinuity at boundary);
          balance_factor requires tuning per task; may under-suppress
          rejected responses in some cases.
    """
    def __init__(self,
                model_engine,
                ref_model_engine,
                optimizer,
                beta=0.1,
                balance_factor=0.3,
                use_cache=False,
                normalize_loss=False):

        self.model_engine = model_engine
        self.ref_model_engine = ref_model_engine
        self.optimizer = optimizer
        self.beta = beta
        self.balance_factor = balance_factor
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
           BPO Loss:
           logr_w = pi_logps_w - ref_logps_w
           logr_l = pi_logps_l - ref_logps_l
           balanced_logits = min(logr_w, balance_factor * (-logr_l))
           L = -logsigmoid(beta * balanced_logits)
        '''
        logr_w = pi_logps_w - ref_logps_w  # chosen log-ratio (want positive)
        logr_l = pi_logps_l - ref_logps_l  # rejected log-ratio (want negative, so -logr_l positive)

        # Balanced margin: min of chosen improvement and scaled rejected suppression
        balanced_logits = torch.min(logr_w, self.balance_factor * (-logr_l))

        logits = self.beta * balanced_logits
        loss = -F.logsigmoid(logits).mean()

        with torch.no_grad():
            rewards_w = self.beta * logr_w
            rewards_l = self.beta * logr_l
            margin = (rewards_w - rewards_l).mean()
            reward_acc = (rewards_w > rewards_l).float().mean()

        return loss, margin, reward_acc

    def train_step(self, micro_batch):
        '''
           One training step for BPO.
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

        # Compute BPO loss
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
