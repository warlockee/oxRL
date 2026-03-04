import torch
import torch.nn.functional as F
from typing import Dict, Any

class RDPO:
    """
    R-DPO: Robust Direct Preference Optimization with Length Regularization.

    Reference: "Disentangling Length from Quality in Direct Preference Optimization"
    (Park et al., 2024). https://arxiv.org/abs/2403.19159

    R-DPO extends DPO with a length regularization term that prevents the model
    from exploiting length biases in the preference data. The key insight is that
    standard DPO can learn to generate longer/shorter responses simply because
    the training data has a length-quality correlation, rather than learning the
    underlying quality features.

    Loss:
        L_RDPO = -logsigmoid(beta * (log_ratio_w - log_ratio_l)
                              - alpha * (len_w - len_l))

    Where:
        - log_ratio = log(pi / pi_ref) for chosen (w) and rejected (l)
        - len_w, len_l = number of valid response tokens in chosen and rejected
        - alpha = length regularization coefficient

    The regularization term acts as a per-example learning rate modifier:
        - Up-weights gradient when chosen is shorter than rejected
        - Down-weights gradient when chosen is longer than rejected
    This forces the model to learn quality features beyond length.
    """
    def __init__(self,
                model_engine,
                ref_model_engine,
                optimizer,
                beta=0.1,
                alpha=0.01,
                use_cache=False,
                normalize_loss=False):

        self.model_engine = model_engine
        self.ref_model_engine = ref_model_engine
        self.optimizer = optimizer
        self.beta = beta
        self.alpha = alpha  # length regularization coefficient
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
        # [B, T-1] * [B, T-1] -> [B, T-1] -> [B]
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

        # [B, T, V] -> [B, T-1, V]
        logits = output.logits[:, :-1, :].contiguous()
        # [B, T] -> [B, T-1]
        target_ids = input_ids[:, 1:].contiguous()

        logps = self.compute_logps(logits, target_ids, loss_mask)
        return logps

    def compute_loss(self, pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l, len_w, len_l):
        '''
           R-DPO Loss with length regularization:
           L = -logsigmoid(beta * (log_ratio_w - log_ratio_l)
                           - alpha * (len_w - len_l))

           len_w, len_l: [B] tensors with number of valid (non-padding) response tokens
        '''
        pi_logr_w = pi_logps_w - ref_logps_w
        pi_logr_l = pi_logps_l - ref_logps_l

        # Standard DPO logits
        dpo_logits = self.beta * (pi_logr_w - pi_logr_l)

        # Length regularization term
        length_penalty = self.alpha * (len_w - len_l)

        # R-DPO: subtract length penalty from logits
        logits = dpo_logits - length_penalty

        loss = -F.logsigmoid(logits).mean()

        with torch.no_grad():
            rewards_w = self.beta * pi_logr_w
            rewards_l = self.beta * pi_logr_l
            margin = (rewards_w - rewards_l).mean()
            avg_length_diff = (len_w - len_l).float().mean()

        return loss, margin, avg_length_diff

    def train_step(self, micro_batch):
        '''
           One training step for R-DPO.
           micro_batch contains: chosen_input_ids, rejected_input_ids, ...
        '''
        self.model_engine.train()
        if self.ref_model_engine is not None:
            self.ref_model_engine.eval()

        # Combine chosen and rejected inputs for a single forward pass
        batch_size = micro_batch['chosen_input_ids'].shape[0]
        input_ids = torch.cat([micro_batch['chosen_input_ids'], micro_batch['rejected_input_ids']], dim=0)
        attn_mask = torch.cat([micro_batch['chosen_attn_mask'], micro_batch['rejected_attn_mask']], dim=0)
        loss_mask = torch.cat([micro_batch['chosen_loss_mask'], micro_batch['rejected_loss_mask']], dim=0)

        # Compute response lengths (number of valid tokens per response)
        # loss_mask[:, :] has 1s for response tokens, 0s for prompt/padding
        chosen_len = micro_batch['chosen_loss_mask'].sum(-1).float()  # [B]
        rejected_len = micro_batch['rejected_loss_mask'].sum(-1).float()  # [B]

        # 1. Reference model logps
        with torch.no_grad():
            ref_logps = self.forward(input_ids, attn_mask, loss_mask, self.ref_model_engine)
            ref_logps_w, ref_logps_l = torch.split(ref_logps, batch_size, dim=0)

        # 2. Policy model logps
        pi_logps = self.forward(input_ids, attn_mask, loss_mask, self.model_engine)
        pi_logps_w, pi_logps_l = torch.split(pi_logps, batch_size, dim=0)

        # 3. Compute R-DPO loss with length regularization
        loss, margin, avg_len_diff = self.compute_loss(
            pi_logps_w, pi_logps_l,
            ref_logps_w, ref_logps_l,
            chosen_len.to(pi_logps_w.device),
            rejected_len.to(pi_logps_l.device),
        )

        # 4. Backward and Step
        self.model_engine.backward(loss)
        self.model_engine.step()

        return {
            "loss": loss.item(),
            "margin": margin.item(),
            "avg_length_diff": avg_len_diff.item(),
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

            chosen_len = micro_batch['chosen_loss_mask'].sum(-1).float()
            rejected_len = micro_batch['rejected_loss_mask'].sum(-1).float()

            ref_logps = self.forward(input_ids, attn_mask, loss_mask, self.ref_model_engine)
            ref_logps_w, ref_logps_l = torch.split(ref_logps, batch_size, dim=0)

            pi_logps = self.forward(input_ids, attn_mask, loss_mask, self.model_engine)
            pi_logps_w, pi_logps_l = torch.split(pi_logps, batch_size, dim=0)

            loss, margin, avg_len_diff = self.compute_loss(
                pi_logps_w, pi_logps_l,
                ref_logps_w, ref_logps_l,
                chosen_len.to(pi_logps_w.device),
                rejected_len.to(pi_logps_l.device),
            )

        return {
            "loss": loss.item(),
            "margin": margin.item(),
            "avg_length_diff": avg_len_diff.item(),
        }
