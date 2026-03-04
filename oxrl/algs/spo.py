import torch
import torch.nn.functional as F
from typing import Dict, Any

class SPO:
    """
    SPO: Self Preference Optimization with Self Regularization.

    Reference: "SPO: Self Preference Optimization with Self Regularization"
    (EMNLP 2025 Findings).

    Standard DPO uses -logsigmoid(x) as its loss function, which is monotonically
    decreasing and unbounded below. This means the model is incentivized to push
    the log-probability ratio arbitrarily large, leading to overoptimization.

    SPO replaces -logsigmoid(x) with -SiLU(x) = -x * sigmoid(x), the negated
    Sigmoid Linear Unit. Unlike logsigmoid, SiLU has a finite minimum at x ~ -1.28
    (where its value is ~ -0.278), which naturally prevents the model from
    excessively amplifying the chosen-rejected probability ratio.

    Key properties:
        - SiLU(x) = x * sigmoid(x)
        - min SiLU(x) ~ -0.278 at x ~ -1.28
        - SiLU is an upper bound of logsigmoid: SiLU(x) >= logsigmoid(x)
        - Optimizing SPO implicitly optimizes DPO (since SPO loss >= DPO loss)

    Loss:
        L_SPO = -SiLU(beta * (logr_w - logr_l)).mean()
             = -(beta * (logr_w - logr_l) * sigmoid(beta * (logr_w - logr_l))).mean()

    This is a one-line change from DPO: replace F.logsigmoid with F.silu.

    Pro:  Prevents overoptimization by having a bounded loss; implicit self
          regularization; no new hyperparameters; upper bound of DPO loss.
    Con:  The bounded minimum means the loss floor is reached sooner, potentially
          limiting how far the model can be pushed from the reference.
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
           SPO Loss: -SiLU(beta * (log_ratio_w - log_ratio_l))
           where SiLU(x) = x * sigmoid(x)
        '''
        pi_logr_w = pi_logps_w - ref_logps_w
        pi_logr_l = pi_logps_l - ref_logps_l

        logits = self.beta * (pi_logr_w - pi_logr_l)

        # SPO: replace -logsigmoid with -SiLU
        # SiLU(x) = x * sigmoid(x), so -SiLU(logits) = -(logits * sigmoid(logits))
        loss = -F.silu(logits).mean()

        with torch.no_grad():
            rewards_w = self.beta * pi_logr_w
            rewards_l = self.beta * pi_logr_l
            margin = (rewards_w - rewards_l).mean()
            reward_acc = (rewards_w > rewards_l).float().mean()

        return loss, margin, reward_acc

    def train_step(self, micro_batch):
        '''
           One training step for SPO.
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
