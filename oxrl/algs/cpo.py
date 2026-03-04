import torch
import torch.nn.functional as F
from typing import Dict, Any

class CPO:
    """
    Contrastive Preference Optimization (CPO).

    Reference: "Contrastive Preference Optimization: Pushing the Boundaries of
    LLM Performance in Machine Translation" (Xu et al., ICML 2024).
    https://arxiv.org/abs/2401.08417

    CPO is a reference-free preference optimization method that combines:
      1. A preference loss (Bradley-Terry sigmoid on policy log-probs), and
      2. A behavioral cloning (BC) regularizer (NLL on chosen responses).

    Loss:  L_CPO = L_preference + alpha * L_nll

    Unlike DPO, CPO does not require a reference model. The preference logits
    are computed directly from the policy: logits = logps_chosen - logps_rejected.
    The BC regularizer keeps the policy close to the data distribution, preventing
    the model from collapsing to degenerate solutions.

    Supports loss_type variants:
      - "sigmoid" (default): standard DPO-style Bradley-Terry loss
      - "hinge": SLiC-style hinge loss
    """
    def __init__(self,
                model_engine,
                optimizer,
                beta=0.1,
                cpo_alpha=1.0,
                loss_type="sigmoid",
                label_smoothing=0.0,
                use_cache=False,
                normalize_loss=False):

        self.model_engine = model_engine
        self.optimizer = optimizer
        self.beta = beta
        self.cpo_alpha = cpo_alpha
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.use_cache = use_cache
        self.normalize_loss = normalize_loss

        # NLL loss for the BC regularization term
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

        # Apply mask and sum across sequence length
        # [B, T-1] * [B, T-1] -> [B, T-1] -> [B]
        logps = (per_token_logps * loss_mask).sum(-1)
        return logps

    def forward(self, input_ids, attn_mask, loss_mask):
        '''
            Forward pass through the model engine.
            Returns logps and also the raw logits (needed for NLL loss).
        '''
        token_type_ids = torch.zeros_like(input_ids)

        output = self.model_engine(input_ids=input_ids,
                                   attention_mask=attn_mask,
                                   token_type_ids=token_type_ids,
                                   use_cache=self.use_cache)

        # [B, T, V] -> [B, T-1, V]
        logits = output.logits[:, :-1, :].contiguous()
        # [B, T] -> [B, T-1]
        target_ids = input_ids[:, 1:].contiguous()

        logps = self.compute_logps(logits, target_ids, loss_mask)
        return logps, logits, target_ids

    def compute_nll_loss(self, logits, target_ids, loss_mask):
        '''
           Compute negative log-likelihood (NLL) loss on chosen responses.
           This is the behavioral cloning (BC) regularization term.
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

    def compute_loss(self, pi_logps_w, pi_logps_l, chosen_logits, chosen_targets, chosen_mask):
        '''
           CPO Loss: L_preference + alpha * L_nll

           L_preference depends on loss_type:
             - "sigmoid": -logsigmoid(beta * (logps_w - logps_l))  (with optional label smoothing)
             - "hinge": relu(1 - beta * (logps_w - logps_l))

           L_nll: NLL on chosen responses (BC regularizer)
        '''
        logits = pi_logps_w - pi_logps_l

        # 1. Preference loss
        if self.loss_type == "sigmoid":
            pref_losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            pref_losses = torch.relu(1 - self.beta * logits)
        else:
            raise ValueError(f"Unknown CPO loss_type: {self.loss_type}. Use 'sigmoid' or 'hinge'.")

        pref_loss = pref_losses.mean()

        # 2. BC regularization (NLL on chosen)
        nll_loss = self.compute_nll_loss(chosen_logits, chosen_targets, chosen_mask)

        # 3. Combined CPO loss
        total_loss = pref_loss + self.cpo_alpha * nll_loss

        with torch.no_grad():
            margin = (pi_logps_w - pi_logps_l).mean()
            reward_acc = (pi_logps_w > pi_logps_l).float().mean()

        return total_loss, pref_loss, nll_loss, margin, reward_acc

    def train_step(self, micro_batch):
        '''
           One training step for CPO.
           micro_batch contains: chosen_input_ids, rejected_input_ids, ...
        '''
        self.model_engine.train()

        # Combine chosen and rejected inputs for a single forward pass
        batch_size = micro_batch['chosen_input_ids'].shape[0]
        input_ids = torch.cat([micro_batch['chosen_input_ids'], micro_batch['rejected_input_ids']], dim=0)
        attn_mask = torch.cat([micro_batch['chosen_attn_mask'], micro_batch['rejected_attn_mask']], dim=0)
        loss_mask = torch.cat([micro_batch['chosen_loss_mask'], micro_batch['rejected_loss_mask']], dim=0)

        # Policy model forward pass (no reference model needed)
        pi_logps, logits, target_ids = self.forward(input_ids, attn_mask, loss_mask)
        pi_logps_w, pi_logps_l = torch.split(pi_logps, batch_size, dim=0)

        # Extract chosen logits/targets/mask for the NLL term
        logits_w, _ = torch.split(logits, batch_size, dim=0)
        targets_w, _ = torch.split(target_ids, batch_size, dim=0)
        loss_mask_w, _ = torch.split(loss_mask, batch_size, dim=0)

        # Compute CPO loss
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
