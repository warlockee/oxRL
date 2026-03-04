import torch
import torch.nn.functional as F
from typing import Dict, Any

class CPOSimPO:
    """
    CPO-SimPO: Contrastive Preference Optimization with SimPO reward formulation.

    Reference: Xu et al., "Contrastive Preference Optimization: Pushing the
    Boundaries of LLM Performance in Machine Translation" (ICML 2024) combined
    with Meng et al., "SimPO: Simple Preference Optimization with a Reference-
    Free Reward" (NeurIPS 2024).

    GitHub: https://github.com/fe1ixxu/CPO_SIMPO

    CPO-SimPO merges two complementary ideas:
      - SimPO's length-normalized average log-probability as the implicit
        reward, plus a target reward margin gamma.
      - CPO's behavioral cloning (BC) regularizer: an NLL loss on the chosen
        response that prevents the model from drifting too far from the
        preferred data distribution.

    Like both CPO and SimPO individually, CPO-SimPO is **reference-free** --
    it does not need a frozen reference model.

    Loss:
        L = L_pref + alpha * L_nll

    where:
        L_pref = -logsigmoid(beta * avg_logp_w - beta * avg_logp_l - gamma)
        L_nll  = NLL on chosen response (cross-entropy, masked mean)

        avg_logp = (1/|y|) * sum_t log pi(y_t | x, y_<t)

    Hyperparameters:
        beta:   Scaling for the implicit reward (temperature). Default 2.0.
        gamma:  Target reward margin (SimPO). Default 0.5.
        alpha:  Weight of BC/NLL regularization (CPO). Default 1.0.

    When alpha=0, this reduces to SimPO.
    When gamma=0 and length normalization is removed, this reduces to CPO.

    Pro:  Reference-free; length-aware; BC regularization prevents collapse;
          combines benefits of SimPO (length normalization, margin) and CPO
          (behavioral cloning).
    Con:  Three hyperparameters to tune (beta, gamma, alpha).
    """
    def __init__(self,
                model_engine,
                optimizer,
                beta=2.0,
                gamma=0.5,
                cposimpo_alpha=1.0,
                use_cache=False,
                normalize_loss=False):

        self.model_engine = model_engine
        self.optimizer = optimizer
        self.beta = beta
        self.gamma = gamma
        self.cposimpo_alpha = cposimpo_alpha
        self.use_cache = use_cache
        self.normalize_loss = normalize_loss

        # NLL loss for BC regularization
        self.nll_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    def compute_logps(self, logits, target_ids, loss_mask):
        '''
           Computes length-normalized log probabilities.
           logits: [B, T-1, vocab_size]
           target_ids: [B, T-1]
           loss_mask: [B, T-1]
           Returns:
               logps: [B]  (length-normalized average)
        '''
        log_probs = F.log_softmax(logits, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=2, index=target_ids.unsqueeze(2)).squeeze(2)

        # Apply mask, sum, and length-normalize (SimPO-style)
        logps = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)
        return logps

    def forward(self, input_ids, attn_mask, loss_mask):
        '''
            Forward pass. Returns length-normalized logps and raw logits/targets
            for computing NLL loss on chosen responses.
        '''
        token_type_ids = torch.zeros_like(input_ids)

        output = self.model_engine(input_ids=input_ids,
                                   attention_mask=attn_mask,
                                   token_type_ids=token_type_ids,
                                   use_cache=self.use_cache)

        logits = output.logits[:, :-1, :].contiguous()
        target_ids = input_ids[:, 1:].contiguous()

        logps = self.compute_logps(logits, target_ids, loss_mask)
        return logps, logits, target_ids

    def compute_nll_loss(self, logits, target_ids, loss_mask):
        '''
           NLL loss on chosen responses (BC regularization).
           logits: [B, T-1, vocab_size]
           target_ids: [B, T-1]
           loss_mask: [B, T-1]
           Returns: scalar
        '''
        B, T_minus_1, V = logits.shape
        per_token_loss = self.nll_loss_fn(logits.view(-1, V), target_ids.view(-1))
        per_token_loss = per_token_loss.view(B, T_minus_1)

        # Masked mean
        nll_loss = (per_token_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
        return nll_loss

    def compute_loss(self, pi_logps_w, pi_logps_l, chosen_logits, chosen_targets, chosen_mask):
        '''
           CPO-SimPO Loss:
             L = L_pref + alpha * L_nll

           L_pref = -logsigmoid(beta * avg_logp_w - beta * avg_logp_l - gamma)
           L_nll = masked mean NLL on chosen tokens
        '''
        # 1. Preference loss (SimPO-style: length-normalized, with margin)
        pref_logits = self.beta * (pi_logps_w - pi_logps_l) - self.gamma
        pref_loss = -F.logsigmoid(pref_logits).mean()

        # 2. BC regularization (NLL on chosen)
        nll_loss = self.compute_nll_loss(chosen_logits, chosen_targets, chosen_mask)

        # 3. Combined loss
        total_loss = pref_loss + self.cposimpo_alpha * nll_loss

        with torch.no_grad():
            margin = (pi_logps_w - pi_logps_l).mean()
            reward_acc = (pi_logps_w > pi_logps_l).float().mean()

        return total_loss, pref_loss, nll_loss, margin, reward_acc

    def train_step(self, micro_batch):
        '''
           One training step for CPO-SimPO.
        '''
        self.model_engine.train()

        batch_size = micro_batch['chosen_input_ids'].shape[0]
        input_ids = torch.cat([micro_batch['chosen_input_ids'], micro_batch['rejected_input_ids']], dim=0)
        attn_mask = torch.cat([micro_batch['chosen_attn_mask'], micro_batch['rejected_attn_mask']], dim=0)
        loss_mask = torch.cat([micro_batch['chosen_loss_mask'], micro_batch['rejected_loss_mask']], dim=0)

        # Single forward pass (no reference model needed)
        pi_logps, logits, target_ids = self.forward(input_ids, attn_mask, loss_mask)
        pi_logps_w, pi_logps_l = torch.split(pi_logps, batch_size, dim=0)

        # Extract chosen logits/targets/mask for NLL term
        logits_w, _ = torch.split(logits, batch_size, dim=0)
        targets_w, _ = torch.split(target_ids, batch_size, dim=0)
        loss_mask_w, _ = torch.split(loss_mask, batch_size, dim=0)

        # Compute CPO-SimPO loss
        loss, pref_loss, nll_loss, margin, reward_acc = self.compute_loss(
            pi_logps_w, pi_logps_l, logits_w, targets_w, loss_mask_w
        )

        # Backward and Step
        self.model_engine.backward(loss)
        self.model_engine.step()

        return {
            "loss": loss.item(),
            "pref_loss": pref_loss.item(),
            "nll_loss": nll_loss.item(),
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

            pi_logps, logits, target_ids = self.forward(input_ids, attn_mask, loss_mask)
            pi_logps_w, pi_logps_l = torch.split(pi_logps, batch_size, dim=0)

            logits_w, _ = torch.split(logits, batch_size, dim=0)
            targets_w, _ = torch.split(target_ids, batch_size, dim=0)
            loss_mask_w, _ = torch.split(loss_mask, batch_size, dim=0)

            loss, pref_loss, nll_loss, margin, reward_acc = self.compute_loss(
                pi_logps_w, pi_logps_l, logits_w, targets_w, loss_mask_w
            )

        return {
            "loss": loss.item(),
            "pref_loss": pref_loss.item(),
            "nll_loss": nll_loss.item(),
            "margin": margin.item(),
            "reward_acc": reward_acc.item(),
        }
