import torch
import torch.nn.functional as F
from typing import Dict, Any

class APO:
    """
    APO: Anchored Preference Optimization.

    Reference: "Anchored Preference Optimization and Contrastive Revisions:
    Addressing Underspecification in Alignment"
    (Mita et al., TACL 2024). https://arxiv.org/abs/2408.06266

    APO provides fine-grained control over individual response likelihoods
    during alignment, unlike DPO which only constrains the reward difference.
    APO "anchors" each side of the comparison to predictably increase or
    decrease depending on relative quality.

    Two variants:

    APO-zero:
        L = -sigmoid(beta * logr_w) + sigmoid(beta * logr_l)
        Increases chosen likelihood AND decreases rejected likelihood.

    APO-down:
        L = sigmoid(beta * logr_w) - sigmoid(beta * logr_w - beta * logr_l)
        Decreases both likelihoods, but decreases rejected MORE.
        Useful when winning outputs are already from a weaker model.

    where:
        logr = log(pi(y|x) / pi_ref(y|x))

    Pro:  Fine-grained control over absolute likelihood changes (not just
          relative); handles underspecified preference data well.
    Con:  Requires choosing between apo_zero and apo_down modes;
          sigmoid saturation can slow convergence for large rewards.
    """
    def __init__(self,
                model_engine,
                ref_model_engine,
                optimizer,
                beta=0.1,
                apo_mode="zero",
                use_cache=False,
                normalize_loss=False):

        self.model_engine = model_engine
        self.ref_model_engine = ref_model_engine
        self.optimizer = optimizer
        self.beta = beta
        self.apo_mode = apo_mode
        self.use_cache = use_cache
        self.normalize_loss = normalize_loss

        if self.apo_mode not in ("zero", "down"):
            raise ValueError(
                f"Unknown apo_mode: {self.apo_mode}. Must be 'zero' or 'down'."
            )

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
           APO Loss:
           r_w = beta * (pi_logps_w - ref_logps_w)
           r_l = beta * (pi_logps_l - ref_logps_l)

           APO-zero: L = -sigmoid(r_w) + sigmoid(r_l)
           APO-down: L = sigmoid(r_w) - sigmoid(r_w - r_l)
        '''
        r_w = self.beta * (pi_logps_w - ref_logps_w)
        r_l = self.beta * (pi_logps_l - ref_logps_l)

        if self.apo_mode == "zero":
            loss = (-torch.sigmoid(r_w) + torch.sigmoid(r_l)).mean()
        else:  # apo_mode == "down"
            loss = (torch.sigmoid(r_w) - torch.sigmoid(r_w - r_l)).mean()

        with torch.no_grad():
            margin = (r_w - r_l).mean()
            reward_acc = (r_w > r_l).float().mean()

        return loss, margin, reward_acc

    def train_step(self, micro_batch):
        '''
           One training step for APO.
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

        # Compute APO loss
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
