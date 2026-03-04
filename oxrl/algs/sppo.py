import torch
import torch.nn.functional as F
from typing import Dict, Any

class SPPO:
    """
    SPPO: Self-Play Preference Optimization (hard label variant).

    Reference: "Self-Play Preference Optimization for Language Model
    Alignment" (Wu et al., 2024). https://arxiv.org/abs/2405.00675

    SPPO formulates language model alignment as a two-player constant-sum
    game and finds the Nash equilibrium policy through iterative updates.
    The loss pushes chosen response log-ratios toward +1/(2*beta) and
    rejected response log-ratios toward -1/(2*beta), independently
    adjusting both rather than only the relative gap (as DPO does).

    This is the "hard label" variant (sppo_hard in TRL) where the win
    probability P(y_w > y_l | x) is set to 1.0 for the chosen response.

    Loss:
        L_SPPO = (log_ratio_w - 1/(2*beta))^2
               + (log_ratio_l + 1/(2*beta))^2

    where:
        log_ratio = log(pi(y|x) / pi_ref(y|x))

    Pro:  Increases chosen likelihood AND decreases rejected likelihood
          (unlike DPO which only maximizes their gap); converges to
          Nash equilibrium with theoretical guarantees.
    Con:  Requires a reference model; squared loss can be less stable
          than log-sigmoid loss with outlier log-ratios.
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
           SPPO Hard Label Loss:
           L = (log_ratio_w - 1/(2*beta))^2 + (log_ratio_l + 1/(2*beta))^2
        '''
        log_ratio_w = pi_logps_w - ref_logps_w
        log_ratio_l = pi_logps_l - ref_logps_l

        # Target values from SPPO Nash equilibrium
        target_w = 1.0 / (2.0 * self.beta)
        target_l = -1.0 / (2.0 * self.beta)

        # Squared error regression loss
        loss = (
            (log_ratio_w - target_w).pow(2)
            + (log_ratio_l - target_l).pow(2)
        ).mean()

        with torch.no_grad():
            rewards_w = self.beta * log_ratio_w
            rewards_l = self.beta * log_ratio_l
            margin = (rewards_w - rewards_l).mean()
            reward_acc = (rewards_w > rewards_l).float().mean()

        return loss, margin, reward_acc

    def train_step(self, micro_batch):
        '''
           One training step for SPPO.
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

        # Compute SPPO loss
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
