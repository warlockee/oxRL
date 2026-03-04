import torch
import torch.nn.functional as F
from typing import Dict, Any

class EXO:
    """
    EXO: Efficient Exact Optimization via Reverse KL Preference Optimization.

    Reference: "Towards Efficient Exact Optimization of Language Model
    Alignment" (Ji et al., 2024). https://arxiv.org/abs/2402.00856
    Published at ICML 2024.

    EXO minimizes the reverse KL divergence between the model's preference
    distribution and the label distribution, rather than the forward KL
    (log-sigmoid) used in standard DPO. This yields mode-seeking behavior,
    which is theoretically better for capturing the correct modes of the
    optimal policy.

    This implementation is the simplified pairwise variant (exo_pair) with
    label smoothing epsilon, as used in TRL (Eq. 16 of the paper).

    Loss:
        L_exo = D_KL(q_theta || p_label)
              = q_w * (log(q_w) - log(p_w)) + q_l * (log(q_l) - log(p_l))

    where:
        q_w = sigmoid(beta * h)   (model's prob of choosing winner)
        q_l = sigmoid(-beta * h)  (model's prob of choosing loser)
        p_w = 1 - epsilon         (label prob of winner)
        p_l = epsilon             (label prob of loser)
        h = log_ratio_w - log_ratio_l

    Pro:  Mode-seeking behavior (better mode coverage than DPO);
          approaches PPO as K (number of samples) grows.
    Con:  Requires epsilon > 0 to avoid log(0); slightly more complex
          gradient landscape than DPO.
    """
    def __init__(self,
                model_engine,
                ref_model_engine,
                optimizer,
                beta=0.1,
                exo_epsilon=1e-3,
                use_cache=False,
                normalize_loss=False):

        self.model_engine = model_engine
        self.ref_model_engine = ref_model_engine
        self.optimizer = optimizer
        self.beta = beta
        self.exo_epsilon = exo_epsilon
        self.use_cache = use_cache
        self.normalize_loss = normalize_loss

        # Validate epsilon is strictly positive and < 0.5
        assert 0.0 < self.exo_epsilon < 0.5, \
            f"exo_epsilon must be in (0, 0.5), got {self.exo_epsilon}"

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
           EXO Pairwise Loss (reverse KL):
           L = q_w * (log(q_w) - log(p_w)) + q_l * (log(q_l) - log(p_l))
        '''
        pi_logr_w = pi_logps_w - ref_logps_w
        pi_logr_l = pi_logps_l - ref_logps_l

        delta = self.beta * (pi_logr_w - pi_logr_l)

        # Model's preference distribution
        q_w = torch.sigmoid(delta)          # prob of choosing winner
        log_q_w = F.logsigmoid(delta)
        q_l = torch.sigmoid(-delta)         # prob of choosing loser
        log_q_l = F.logsigmoid(-delta)

        # Label distribution
        eps = self.exo_epsilon
        log_p_w = torch.log(torch.tensor(1.0 - eps, device=delta.device, dtype=delta.dtype))
        log_p_l = torch.log(torch.tensor(eps, device=delta.device, dtype=delta.dtype))

        # Reverse KL: D_KL(q || p) = sum_i q_i * (log q_i - log p_i)
        loss = (q_w * (log_q_w - log_p_w) + q_l * (log_q_l - log_p_l)).mean()

        with torch.no_grad():
            rewards_w = self.beta * pi_logr_w
            rewards_l = self.beta * pi_logr_l
            margin = (rewards_w - rewards_l).mean()
            reward_acc = (rewards_w > rewards_l).float().mean()

        return loss, margin, reward_acc

    def train_step(self, micro_batch):
        '''
           One training step for EXO.
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

        # Compute EXO loss
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
