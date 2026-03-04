import torch
import torch.nn.functional as F
from typing import Dict, Any

class DPNLL:
    """
    DPO with NLL Regularization (DPO+NLL).

    Combines standard DPO loss with a negative log-likelihood (NLL) loss on the
    chosen responses. This is one of the most widely-used practical improvements
    to DPO, deployed in production systems at Meta, and supported by frameworks
    like LLaMA-Factory and Axolotl.

    Reference: Pang et al., "Iterative Reasoning Preference Optimization"
    (NeurIPS 2024); also widely used in RLHF pipelines without a single
    canonical paper.

    The NLL term explicitly increases the probability of chosen responses,
    addressing DPO's well-known failure mode where chosen log-probabilities
    decrease during training (likelihood displacement). Without NLL, DPO
    primarily learns by decreasing rejected probabilities rather than
    increasing chosen probabilities.

    Loss:
        L = L_DPO + alpha * L_NLL

    where:
        L_DPO = -logsigmoid(beta * (logr_w - logr_l)).mean()
        L_NLL = NLL(chosen_logits, chosen_targets) [masked mean over chosen tokens]

    Hyperparameters:
        beta:  KL constraint strength (same as DPO)
        alpha: Weight of the NLL regularization term (default 1.0)

    Pro:  Prevents chosen probability decrease; stabilizes training; simple to
          implement; widely validated in production.
    Con:  Adds an extra hyperparameter (alpha) to tune.
    """
    def __init__(self,
                model_engine,
                ref_model_engine,
                optimizer,
                beta=0.1,
                dpnll_alpha=1.0,
                use_cache=False,
                normalize_loss=False):

        self.model_engine = model_engine
        self.ref_model_engine = ref_model_engine
        self.optimizer = optimizer
        self.beta = beta
        self.dpnll_alpha = dpnll_alpha
        self.use_cache = use_cache
        self.normalize_loss = normalize_loss

        # NLL loss for the chosen-response regularization term
        self.nll_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

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
            Returns logps and also the raw logits (needed for NLL loss).
        '''
        token_type_ids = torch.zeros_like(input_ids)

        output = model_engine(input_ids=input_ids,
                              attention_mask=attn_mask,
                              token_type_ids=token_type_ids,
                              use_cache=self.use_cache)

        logits = output.logits[:, :-1, :].contiguous()
        target_ids = input_ids[:, 1:].contiguous()

        logps = self.compute_logps(logits, target_ids, loss_mask)
        return logps, logits, target_ids

    def forward_ref(self, input_ids, attn_mask, loss_mask):
        '''
            Forward pass through the reference model.
            Only returns logps (no raw logits needed).
        '''
        token_type_ids = torch.zeros_like(input_ids)

        output = self.ref_model_engine(input_ids=input_ids,
                                       attention_mask=attn_mask,
                                       token_type_ids=token_type_ids,
                                       use_cache=self.use_cache)

        logits = output.logits[:, :-1, :].contiguous()
        target_ids = input_ids[:, 1:].contiguous()

        logps = self.compute_logps(logits, target_ids, loss_mask)
        return logps

    def compute_nll_loss(self, logits, target_ids, loss_mask):
        '''
           Compute negative log-likelihood (NLL) loss on chosen responses.
           logits: [B, T-1, vocab_size]
           target_ids: [B, T-1]
           loss_mask: [B, T-1]
           Returns:
               nll_loss: scalar
        '''
        B, T_minus_1, V = logits.shape
        per_token_loss = self.nll_loss_fn(logits.view(-1, V), target_ids.view(-1))
        per_token_loss = per_token_loss.view(B, T_minus_1)

        # Masked mean
        nll_loss = (per_token_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
        return nll_loss

    def compute_loss(self, pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l,
                     chosen_logits, chosen_targets, chosen_mask):
        '''
           DPO+NLL Loss: L_DPO + alpha * L_NLL

           L_DPO: standard DPO preference loss
           L_NLL: NLL on chosen responses (prevents chosen probability decrease)
        '''
        pi_logr_w = pi_logps_w - ref_logps_w
        pi_logr_l = pi_logps_l - ref_logps_l

        logits = self.beta * (pi_logr_w - pi_logr_l)

        # 1. DPO preference loss
        dpo_loss = -F.logsigmoid(logits).mean()

        # 2. NLL regularization on chosen responses
        nll_loss = self.compute_nll_loss(chosen_logits, chosen_targets, chosen_mask)

        # 3. Combined loss
        total_loss = dpo_loss + self.dpnll_alpha * nll_loss

        with torch.no_grad():
            rewards_w = self.beta * pi_logr_w
            rewards_l = self.beta * pi_logr_l
            margin = (rewards_w - rewards_l).mean()
            reward_acc = (rewards_w > rewards_l).float().mean()

        return total_loss, dpo_loss, nll_loss, margin, reward_acc

    def train_step(self, micro_batch):
        '''
           One training step for DPO+NLL.
        '''
        self.model_engine.train()
        if self.ref_model_engine is not None:
            self.ref_model_engine.eval()

        batch_size = micro_batch['chosen_input_ids'].shape[0]
        input_ids = torch.cat([micro_batch['chosen_input_ids'], micro_batch['rejected_input_ids']], dim=0)
        attn_mask = torch.cat([micro_batch['chosen_attn_mask'], micro_batch['rejected_attn_mask']], dim=0)
        loss_mask = torch.cat([micro_batch['chosen_loss_mask'], micro_batch['rejected_loss_mask']], dim=0)

        # 1. Reference model logps (no raw logits needed)
        with torch.no_grad():
            ref_logps = self.forward_ref(input_ids, attn_mask, loss_mask)
            ref_logps_w, ref_logps_l = torch.split(ref_logps, batch_size, dim=0)

        # 2. Policy model logps + raw logits (for NLL)
        pi_logps, logits_all, target_ids_all = self.forward(input_ids, attn_mask, loss_mask, self.model_engine)
        pi_logps_w, pi_logps_l = torch.split(pi_logps, batch_size, dim=0)

        # 3. Extract chosen logits/targets/mask for the NLL term
        logits_w, _ = torch.split(logits_all, batch_size, dim=0)
        targets_w, _ = torch.split(target_ids_all, batch_size, dim=0)
        loss_mask_w, _ = torch.split(loss_mask, batch_size, dim=0)

        # 4. Compute combined loss
        loss, dpo_loss, nll_loss, margin, reward_acc = self.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l,
            logits_w, targets_w, loss_mask_w)

        self.model_engine.backward(loss)
        self.model_engine.step()

        return {
            "loss": loss.item(),
            "dpo_loss": dpo_loss.item(),
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

            ref_logps = self.forward_ref(input_ids, attn_mask, loss_mask)
            ref_logps_w, ref_logps_l = torch.split(ref_logps, batch_size, dim=0)

            pi_logps, logits_all, target_ids_all = self.forward(input_ids, attn_mask, loss_mask, self.model_engine)
            pi_logps_w, pi_logps_l = torch.split(pi_logps, batch_size, dim=0)

            logits_w, _ = torch.split(logits_all, batch_size, dim=0)
            targets_w, _ = torch.split(target_ids_all, batch_size, dim=0)
            loss_mask_w, _ = torch.split(loss_mask, batch_size, dim=0)

            loss, dpo_loss, nll_loss, margin, reward_acc = self.compute_loss(
                pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l,
                logits_w, targets_w, loss_mask_w)

        return {
            "loss": loss.item(),
            "dpo_loss": dpo_loss.item(),
            "nll_loss": nll_loss.item(),
            "margin": margin.item(),
            "reward_acc": reward_acc.item(),
        }
