import torch
import torch.nn.functional as F
from typing import Dict, Any

class CalDPO:
    """
    Cal-DPO: Calibrated Direct Preference Optimization.

    Reference: "Cal-DPO: Calibrated Direct Preference Optimization for
    Language Model Alignment" (Xiao et al., NeurIPS 2024).
    https://arxiv.org/abs/2412.14516

    Cal-DPO extends DPO by adding a calibration loss that anchors the
    learned implicit rewards to reasonable absolute scales. Standard DPO
    only optimizes relative reward differences; Cal-DPO additionally
    constrains the chosen reward toward +1/(2*beta) and the rejected
    reward toward -1/(2*beta).

    Loss:
        L_CalDPO = L_BT + lambda * L_Cal

    where:
        L_BT  = -logsigmoid(log_ratio_w - log_ratio_l)
                (Bradley-Terry preference loss, no beta scaling)
        L_Cal = (log_ratio_w - 1/(2*beta))^2
              + (log_ratio_l + 1/(2*beta))^2
                (calibration regression loss)

    log_ratio = log(pi(y|x) / pi_ref(y|x))

    When lambda=0, this reduces to the basic BT preference loss.

    Pro:  Prevents chosen response likelihood from decreasing during
          training; calibrates implicit rewards to ground-truth scale.
    Con:  Introduces additional hyperparameter (lambda); slightly more
          compute per step due to the regression term.
    """
    def __init__(self,
                model_engine,
                ref_model_engine,
                optimizer,
                beta=0.1,
                caldpo_lambda=1.0,
                use_cache=False,
                normalize_loss=False):

        self.model_engine = model_engine
        self.ref_model_engine = ref_model_engine
        self.optimizer = optimizer
        self.beta = beta
        self.caldpo_lambda = caldpo_lambda
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
           Cal-DPO Loss:
           L = L_BT + lambda * L_Cal

           L_BT  = -logsigmoid(log_ratio_w - log_ratio_l)
           L_Cal = (log_ratio_w - 1/(2*beta))^2
                 + (log_ratio_l + 1/(2*beta))^2
        '''
        log_ratio_w = pi_logps_w - ref_logps_w
        log_ratio_l = pi_logps_l - ref_logps_l

        # Bradley-Terry preference loss (no beta scaling per Cal-DPO paper)
        bt_logits = log_ratio_w - log_ratio_l
        loss_bt = -F.logsigmoid(bt_logits).mean()

        # Calibration loss: anchor rewards to +/- 1/(2*beta)
        target_w = 1.0 / (2.0 * self.beta)
        target_l = -1.0 / (2.0 * self.beta)
        loss_cal = (
            (log_ratio_w - target_w).pow(2)
            + (log_ratio_l - target_l).pow(2)
        ).mean()

        # Combined loss
        loss = loss_bt + self.caldpo_lambda * loss_cal

        with torch.no_grad():
            rewards_w = self.beta * log_ratio_w
            rewards_l = self.beta * log_ratio_l
            margin = (rewards_w - rewards_l).mean()
            reward_acc = (rewards_w > rewards_l).float().mean()

        return loss, margin, reward_acc, loss_bt, loss_cal

    def train_step(self, micro_batch):
        '''
           One training step for Cal-DPO.
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

        # Compute Cal-DPO loss
        loss, margin, reward_acc, loss_bt, loss_cal = self.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # Backward and Step
        self.model_engine.backward(loss)
        self.model_engine.step()

        return {
            "loss": loss.item(),
            "margin": margin.item(),
            "reward_acc": reward_acc.item(),
            "loss_bt": loss_bt.item(),
            "loss_cal": loss_cal.item(),
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

            loss, margin, reward_acc, loss_bt, loss_cal = self.compute_loss(
                pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        return {
            "loss": loss.item(),
            "margin": margin.item(),
            "reward_acc": reward_acc.item(),
            "loss_bt": loss_bt.item(),
            "loss_cal": loss_cal.item(),
        }
