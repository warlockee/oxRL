import torch
import torch.nn.functional as F
from typing import Dict, Any

class NCA:
    """
    NCA: Noise Contrastive Alignment (pairwise variant).

    Reference: "Noise Contrastive Alignment of Language Models with
    Explicit Rewards" (NeurIPS 2024). https://arxiv.org/abs/2402.05369

    NCA optimizes absolute likelihood values for each response rather
    than just relative comparisons (as DPO does). The pairwise NCA loss
    has two components: (1) maximizing the sigmoid of the chosen reward,
    and (2) regularizing both rewards to prevent unbounded growth.

    Unlike DPO, NCA prevents the chosen response likelihood from
    decreasing during training, making it more effective for reasoning
    tasks where preserving ground-truth response probability is valuable.

    Loss:
        L_NCA = -logsigmoid(r_w)
              - 0.5 * [logsigmoid(-r_w) + logsigmoid(-r_l)]

    where:
        r = beta * log(pi(y|x) / pi_ref(y|x))

    Note: For K=2 responses, InfoNCA reduces to DPO exactly. This
    implementation is the NCA variant, which operates on absolute rewards.

    Pro:  Prevents chosen likelihood from decreasing; handles absolute
          reward calibration; outperforms DPO on reasoning tasks.
    Con:  Slightly more complex gradient landscape; one extra logsigmoid
          term per sample.
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
           NCA Pairwise Loss:
           L = -logsigmoid(r_w)
             - 0.5 * [logsigmoid(-r_w) + logsigmoid(-r_l)]

           where r = beta * log(pi / pi_ref)
        '''
        r_w = self.beta * (pi_logps_w - ref_logps_w)
        r_l = self.beta * (pi_logps_l - ref_logps_l)

        # NCA loss: positive term + regularization
        loss = (
            -F.logsigmoid(r_w)
            - 0.5 * (F.logsigmoid(-r_w) + F.logsigmoid(-r_l))
        ).mean()

        with torch.no_grad():
            margin = (r_w - r_l).mean()
            reward_acc = (r_w > r_l).float().mean()

        return loss, margin, reward_acc

    def train_step(self, micro_batch):
        '''
           One training step for NCA.
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

        # Compute NCA loss
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
