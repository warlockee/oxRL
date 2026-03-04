import torch
import torch.nn.functional as F
from typing import Dict, Any

class AOT:
    """
    AOT: Alignment via Optimal Transport.

    Reference: "Distributional Preference Alignment of LLMs via Optimal
    Transport" (2024). https://arxiv.org/abs/2406.05882

    AOT enforces first-order stochastic dominance of the chosen reward
    distribution over the rejected reward distribution by matching
    sorted quantiles through optimal transport with a convex cost.

    Unlike standard DPO which optimizes sample-level log-ratios, AOT
    sorts the within-batch log-likelihood ratios for chosen and rejected
    responses separately, then computes a DPO-style loss on the matched
    quantile pairs. This ensures that at every quantile, the chosen
    response distribution dominates the rejected one.

    Loss:
        logratios_w = log(pi(y_w|x) / pi_ref(y_w|x))  [sorted ascending]
        logratios_l = log(pi(y_l|x) / pi_ref(y_l|x))  [sorted ascending]
        delta = logratios_w_sorted - logratios_l_sorted
        L = -logsigmoid(beta * delta).mean()

    When batch_size=1, this reduces to standard DPO since sorting is
    a no-op on a single element.

    Pro:  Provides distributional-level preference ordering rather than
          sample-level; more robust to outliers; smooth optimization.
    Con:  Requires sorting within batch (breaks sample correspondence);
          needs larger batch sizes to be effective.
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
           AOT Loss:
           1. Compute per-sample log-ratios for chosen and rejected
           2. Sort each independently (ascending)
           3. Compute DPO-style loss on matched quantile pairs
        '''
        # Per-sample log-likelihood ratios
        logratios_w = pi_logps_w - ref_logps_w
        logratios_l = pi_logps_l - ref_logps_l

        # Sort independently to match quantiles (optimal coupling for 1D OT)
        logratios_w_sorted, _ = torch.sort(logratios_w, dim=0)
        logratios_l_sorted, _ = torch.sort(logratios_l, dim=0)

        # DPO-style loss on matched quantile pairs
        delta = logratios_w_sorted - logratios_l_sorted
        loss = -F.logsigmoid(self.beta * delta).mean()

        with torch.no_grad():
            rewards_w = self.beta * logratios_w
            rewards_l = self.beta * logratios_l
            margin = (rewards_w - rewards_l).mean()
            reward_acc = (rewards_w > rewards_l).float().mean()

        return loss, margin, reward_acc

    def train_step(self, micro_batch):
        '''
           One training step for AOT.
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

        # Compute AOT loss
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
