import torch
import torch.nn.functional as F
from typing import Dict, Any

class DrDPO:
    """
    Dr. DPO: Distributionally Robustifying Direct Preference Optimization.

    Reference: Wu et al., "Towards Robust Alignment of Language Models:
    Distributionally Robustifying Direct Preference Optimization" (ICLR 2025).
    https://arxiv.org/abs/2407.07880

    Standard DPO is sensitive to noise in preference data and can overfit to
    noisy labels. While DPO has some inherent pointwise noise robustness through
    its beta parameter, it lacks pairwise robustness -- the ability to handle
    cases where the preference ordering itself is wrong.

    Dr. DPO addresses this through a distributionally robust optimization (DRO)
    formulation that optimizes against worst-case pairwise scenarios. Instead of
    simply averaging per-sample DPO losses, Dr. DPO uses a log-sum-exp (LSE)
    aggregation that implicitly reweights samples based on their difficulty:

        L_DrDPO = -beta_prime * log( mean( exp( -L_DPO_per_sample / beta_prime ) ) )

    where L_DPO_per_sample = -logsigmoid(beta * (logr_w - logr_l)) for each sample.

    The beta_prime (mode_weight) parameter controls the robustness-performance
    trade-off:
        - beta_prime -> inf: reduces to standard DPO (mean of losses)
        - beta_prime -> 0: focuses on worst-case (hardest) samples
        - Typical values: 0.5 - 2.0, default 1.0

    This modification requires only a single line of code change from DPO
    (the aggregation step), making it one of the simplest robustness improvements
    for DPO.

    Pro:  Robust to noisy preference labels; near-zero computational overhead;
          single new hyperparameter; theoretically grounded in DRO.
    Con:  beta_prime tuning may be dataset-dependent; with clean data, can
          slightly underperform standard DPO due to added conservatism.
    """
    def __init__(self,
                model_engine,
                ref_model_engine,
                optimizer,
                beta=0.1,
                beta_prime=1.0,
                use_cache=False,
                normalize_loss=False):

        self.model_engine = model_engine
        self.ref_model_engine = ref_model_engine
        self.optimizer = optimizer
        self.beta = beta
        self.beta_prime = beta_prime
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
           Dr. DPO Loss: DRO aggregation of per-sample DPO losses.

           Standard DPO: L = mean(-logsigmoid(beta * (logr_w - logr_l)))
           Dr. DPO:      L = -beta' * log(mean(exp(-per_sample_loss / beta')))

           where per_sample_loss = -logsigmoid(beta * (logr_w - logr_l))
        '''
        pi_logr_w = pi_logps_w - ref_logps_w
        pi_logr_l = pi_logps_l - ref_logps_l

        logits = self.beta * (pi_logr_w - pi_logr_l)

        # Per-sample DPO losses (not yet averaged)
        per_sample_losses = -F.logsigmoid(logits)  # [B]

        # Dr. DPO: DRO log-sum-exp aggregation
        # L = -beta' * log(mean(exp(-loss / beta')))
        loss = -self.beta_prime * torch.log(
            torch.mean(torch.exp(-per_sample_losses / self.beta_prime))
        )

        with torch.no_grad():
            rewards_w = self.beta * pi_logr_w
            rewards_l = self.beta * pi_logr_l
            margin = (rewards_w - rewards_l).mean()
            reward_acc = (rewards_w > rewards_l).float().mean()

        return loss, margin, reward_acc

    def train_step(self, micro_batch):
        '''
           One training step for Dr. DPO.
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
