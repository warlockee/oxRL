import torch
import torch.nn.functional as F
from typing import Dict, Any

class KTO:
    def __init__(self,
                model_engine,
                ref_model_engine,
                optimizer,
                beta=0.1,
                lambda_p=1.0,
                lambda_n=1.0,
                use_cache=False,
                normalize_loss=False):

        self.model_engine = model_engine
        self.ref_model_engine = ref_model_engine
        self.optimizer = optimizer
        self.beta = beta
        self.lambda_p = lambda_p # Weight for positive samples
        self.lambda_n = lambda_n # Weight for negative samples
        self.use_cache = use_cache
        self.normalize_loss = normalize_loss

        # Moving average for KL divergence (used in KTO)
        self.kl_ema = 0.0
        self.kl_alpha = 0.1

    def compute_logps(self, logits, target_ids, loss_mask):
        log_probs = F.log_softmax(logits, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=2, index=target_ids.unsqueeze(2)).squeeze(2)
        logps = (per_token_logps * loss_mask).sum(-1)
        return logps

    def forward(self, input_ids, attn_mask, loss_mask, model_engine):
        token_type_ids = torch.zeros_like(input_ids)
        output = model_engine(input_ids=input_ids,
                              attention_mask=attn_mask,
                              token_type_ids=token_type_ids,
                              use_cache=self.use_cache)
        logits = output.logits[:, :-1, :].contiguous()
        target_ids = input_ids[:, 1:].contiguous()
        logps = self.compute_logps(logits, target_ids, loss_mask)
        return logps

    def compute_loss(self, pi_logps, ref_logps, labels, batch_size):
        '''
           KTO Loss with moving average KL term.
           labels: 1 for positive, -1 for negative
           batch_size: number of chosen samples (first batch_size elements are chosen)
        '''
        # log_ratio = log(pi / pi_ref)
        log_ratio = pi_logps - ref_logps

        # KL baseline computed over chosen (desirable) examples only,
        # per the KTO paper.
        batch_kl = log_ratio[:batch_size].mean().detach()
        self.kl_ema = (1 - self.kl_alpha) * self.kl_ema + self.kl_alpha * batch_kl

        logits = self.beta * (log_ratio - self.kl_ema)

        # labels are 1 or -1
        # z * logits
        loss_elements = -F.logsigmoid(labels * logits)

        # Apply weights based on labels
        weights = torch.where(labels > 0, self.lambda_p, self.lambda_n)
        loss = (loss_elements * weights).mean()

        return loss

    def train_step(self, micro_batch):
        self.model_engine.train()
        if self.ref_model_engine is not None:
            self.ref_model_engine.eval()

        # Combine chosen and rejected for a single forward pass
        batch_size = micro_batch['chosen_input_ids'].shape[0]
        input_ids = torch.cat([micro_batch['chosen_input_ids'], micro_batch['rejected_input_ids']], dim=0)
        attn_mask = torch.cat([micro_batch['chosen_attn_mask'], micro_batch['rejected_attn_mask']], dim=0)
        loss_mask = torch.cat([micro_batch['chosen_loss_mask'], micro_batch['rejected_loss_mask']], dim=0)

        with torch.no_grad():
            ref_logps = self.forward(input_ids, attn_mask, loss_mask, self.ref_model_engine)

        pi_logps = self.forward(input_ids, attn_mask, loss_mask, self.model_engine)

        # Labels: chosen=1, rejected=-1
        labels = torch.cat([torch.ones(batch_size, device=pi_logps.device), -torch.ones(batch_size, device=pi_logps.device)], dim=0)

        loss = self.compute_loss(pi_logps, ref_logps, labels, batch_size)

        self.model_engine.backward(loss)
        self.model_engine.step()

        return {"loss": loss.item(), "kl_ema": float(self.kl_ema)}

    def eval_step(self, micro_batch):
        self.model_engine.eval()
        with torch.no_grad():
            batch_size = micro_batch['chosen_input_ids'].shape[0]
            input_ids = torch.cat([micro_batch['chosen_input_ids'], micro_batch['rejected_input_ids']], dim=0)
            attn_mask = torch.cat([micro_batch['chosen_attn_mask'], micro_batch['rejected_attn_mask']], dim=0)
            loss_mask = torch.cat([micro_batch['chosen_loss_mask'], micro_batch['rejected_loss_mask']], dim=0)

            ref_logps = self.forward(input_ids, attn_mask, loss_mask, self.ref_model_engine)
            pi_logps = self.forward(input_ids, attn_mask, loss_mask, self.model_engine)

            labels = torch.cat([torch.ones(batch_size, device=pi_logps.device), -torch.ones(batch_size, device=pi_logps.device)], dim=0)
            loss = self.compute_loss(pi_logps, ref_logps, labels, batch_size)
        return {"loss": loss.item()}
