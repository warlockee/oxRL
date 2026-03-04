import torch
import torch.nn.functional as F
from typing import Dict, Any

class RobustDPO:
    """
    Robust DPO: Provably Robust Direct Preference Optimization.

    Reference: "Provably Robust DPO: Aligning Language Models with Noisy
    Feedback" (Chowdhury et al., 2024). https://arxiv.org/abs/2403.00409
    Published at ICML 2024.

    Robust DPO provides an unbiased estimate of the DPO loss under noisy
    preference labels by applying a correction factor. While cDPO uses a
    weighted average of forward and reversed losses, Robust DPO normalizes
    by 1/(1-2*epsilon) to make the loss an unbiased estimator of the
    clean-label DPO loss.

    Loss:
        L_robust = -[(1 - eps) * logsigmoid(beta * h)
                    + eps * logsigmoid(-beta * h)] / (1 - 2 * eps)

    where:
        h = log_ratio_w - log_ratio_l
        log_ratio = log(pi(y|x) / pi_ref(y|x))
        eps = label flip probability

    When eps=0, this reduces to standard DPO.
    Unlike cDPO, the 1/(1-2*eps) factor makes this an unbiased estimator:
        E[L_robust(noisy)] = L_DPO(clean)

    Pro:  Provably unbiased under label noise; convergence guarantees.
    Con:  Loss magnitude increases as eps -> 0.5 (more noise = larger loss);
          requires knowledge or estimate of the noise rate.
    """
    def __init__(self,
                model_engine,
                ref_model_engine,
                optimizer,
                beta=0.1,
                label_smoothing=0.1,
                use_cache=False,
                normalize_loss=False):

        self.model_engine = model_engine
        self.ref_model_engine = ref_model_engine
        self.optimizer = optimizer
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.use_cache = use_cache
        self.normalize_loss = normalize_loss

        # Validate epsilon is in [0, 0.5)
        assert 0.0 <= self.label_smoothing < 0.5, \
            f"label_smoothing must be in [0, 0.5), got {self.label_smoothing}"

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
           Robust DPO Loss (unbiased under label noise):
           L = -[(1 - eps) * logsigmoid(beta * h)
               + eps * logsigmoid(-beta * h)] / (1 - 2 * eps)
        '''
        pi_logr_w = pi_logps_w - ref_logps_w
        pi_logr_l = pi_logps_l - ref_logps_l

        logits = self.beta * (pi_logr_w - pi_logr_l)

        eps = self.label_smoothing
        numerator = (1 - eps) * F.logsigmoid(logits) + eps * F.logsigmoid(-logits)

        if eps > 0:
            loss = -(numerator / (1 - 2 * eps)).mean()
        else:
            # eps=0: standard DPO (avoid division by 1.0 for clarity)
            loss = -numerator.mean()

        with torch.no_grad():
            rewards_w = self.beta * pi_logr_w
            rewards_l = self.beta * pi_logr_l
            margin = (rewards_w - rewards_l).mean()
            reward_acc = (rewards_w > rewards_l).float().mean()

        return loss, margin, reward_acc

    def train_step(self, micro_batch):
        '''
           One training step for Robust DPO.
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

        # Compute robust DPO loss
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
