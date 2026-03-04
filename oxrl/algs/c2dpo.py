import torch
import torch.nn.functional as F
from typing import Dict, Any

class C2DPO:
    """
    C2-DPO: Constrained Controlled Direct Preference Optimization.

    Reference: "C2-DPO: Constrained Controlled Direct Preference Optimization"
    (2025). arXiv:2502.17507.

    Standard DPO can overoptimize by increasing chosen probability AND
    decreasing rejected probability without bound. This diverges the policy
    far from the reference model, leading to degenerate behavior.

    C2-DPO adds a constraint regularizer that penalizes the squared sum of
    log-ratios, keeping the total deviation from the reference bounded:

        L = L_DPO + lambda * (logr_w + logr_l)^2

    where:
        L_DPO = -logsigmoid(beta * (logr_w - logr_l)).mean()
        constraint = (logr_w + logr_l)^2

    The constraint term pushes (logr_w + logr_l) toward zero. This means
    that if the model increases the chosen log-ratio, it must correspondingly
    increase the rejected log-ratio by a similar amount, and vice versa.
    This prevents the model from simultaneously pushing chosen up and
    rejected down without limit.

    Hyperparameters:
        beta:    KL constraint strength (same as DPO)
        c2_lambda: Weight of the constraint penalty (default 2e-4, as in paper)

    Pro:  Prevents overoptimization; simple L2 penalty; well-motivated
          theoretically; no architectural changes.
    Con:  Adds one hyperparameter (lambda); may slow convergence slightly
          due to constraint.
    """
    def __init__(self,
                model_engine,
                ref_model_engine,
                optimizer,
                beta=0.1,
                c2_lambda=2e-4,
                use_cache=False,
                normalize_loss=False):

        self.model_engine = model_engine
        self.ref_model_engine = ref_model_engine
        self.optimizer = optimizer
        self.beta = beta
        self.c2_lambda = c2_lambda
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
           C2-DPO Loss: L_DPO + lambda * (logr_w + logr_l)^2

           The constraint penalty keeps the sum of log-ratios near zero,
           preventing the model from diverging too far from the reference.
        '''
        pi_logr_w = pi_logps_w - ref_logps_w
        pi_logr_l = pi_logps_l - ref_logps_l

        # 1. Standard DPO preference loss
        dpo_logits = self.beta * (pi_logr_w - pi_logr_l)
        dpo_loss = -F.logsigmoid(dpo_logits).mean()

        # 2. Constraint penalty: penalize squared sum of log-ratios
        constraint = ((pi_logr_w + pi_logr_l) ** 2).mean()

        # 3. Combined loss
        loss = dpo_loss + self.c2_lambda * constraint

        with torch.no_grad():
            rewards_w = self.beta * pi_logr_w
            rewards_l = self.beta * pi_logr_l
            margin = (rewards_w - rewards_l).mean()
            reward_acc = (rewards_w > rewards_l).float().mean()

        return loss, dpo_loss, constraint, margin, reward_acc

    def train_step(self, micro_batch):
        '''
           One training step for C2-DPO.
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

        loss, dpo_loss, constraint, margin, reward_acc = self.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        self.model_engine.backward(loss)
        self.model_engine.step()

        return {
            "loss": loss.item(),
            "dpo_loss": dpo_loss.item(),
            "constraint": constraint.item(),
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

            loss, dpo_loss, constraint, margin, reward_acc = self.compute_loss(
                pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        return {
            "loss": loss.item(),
            "dpo_loss": dpo_loss.item(),
            "constraint": constraint.item(),
            "margin": margin.item(),
            "reward_acc": reward_acc.item(),
        }
