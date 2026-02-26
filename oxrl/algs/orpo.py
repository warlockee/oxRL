import torch
import torch.nn.functional as F
from typing import Dict, Any

class ORPO:
    def __init__(self,
                model_engine,
                optimizer,
                beta=0.1,
                use_cache=False,
                normalize_loss=False):

        self.model_engine = model_engine
        self.optimizer = optimizer
        self.beta = beta
        self.use_cache = use_cache
        self.normalize_loss = normalize_loss
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    def compute_logps(self, logits, target_ids, loss_mask):
        log_probs = F.log_softmax(logits, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=2, index=target_ids.unsqueeze(2)).squeeze(2)
        logps = (per_token_logps * loss_mask).sum(-1)
        return logps

    def forward(self, input_ids, attn_mask, loss_mask):
        output = self.model_engine(input_ids=input_ids,
                                   attention_mask=attn_mask,
                                   use_cache=self.use_cache)
        logits = output.logits[:, :-1, :].contiguous()
        target_ids = input_ids[:, 1:].contiguous()
        logps = self.compute_logps(logits, target_ids, loss_mask)
        return logps, logits, target_ids

    def compute_loss(self, pi_logps_w, pi_logps_l, sft_logits, sft_targets, sft_mask):
        # 1. SFT Loss (only on chosen)
        # sft_logits: [B, T-1, V], sft_targets: [B, T-1], sft_mask: [B, T-1]
        B, T_minus_1, V = sft_logits.shape
        per_token_sft_loss = self.loss_fn(sft_logits.view(-1, V), sft_targets.view(-1))
        sft_loss = (per_token_sft_loss * sft_mask.view(-1)).sum() / sft_mask.sum().clamp(min=1.0)

        # 2. ORPO Loss
        # log odds: log(p / (1-p)) = log(p) - log(1-p)
        # for log(p) close to 0, log(1-p) = log(1 - exp(log(p))) can be unstable
        # but ORPO uses avg log-probs: p_hat = exp(avg_log_probs)
        
        # Calculate log odds for chosen and rejected
        log_odds_w = pi_logps_w - torch.log1p(-torch.exp(pi_logps_w).clamp(max=1.0 - 1e-7))
        log_odds_l = pi_logps_l - torch.log1p(-torch.exp(pi_logps_l).clamp(max=1.0 - 1e-7))
        
        logits = log_odds_w - log_odds_l
        or_loss = -F.logsigmoid(logits).mean()
        
        total_loss = sft_loss + self.beta * or_loss
        
        return total_loss, sft_loss, or_loss

    def train_step(self, micro_batch):
        self.model_engine.train()
        
        # Combine chosen and rejected for a single forward pass
        batch_size = micro_batch['chosen_input_ids'].shape[0]
        input_ids = torch.cat([micro_batch['chosen_input_ids'], micro_batch['rejected_input_ids']], dim=0)
        attn_mask = torch.cat([micro_batch['chosen_attn_mask'], micro_batch['rejected_attn_mask']], dim=0)
        loss_mask = torch.cat([micro_batch['chosen_loss_mask'], micro_batch['rejected_loss_mask']], dim=0)

        pi_logps, logits, targets = self.forward(input_ids, attn_mask, loss_mask)
        
        # Split the results back
        pi_logps_w, pi_logps_l = torch.split(pi_logps, batch_size, dim=0)
        logits_w, _ = torch.split(logits, batch_size, dim=0)
        targets_w, _ = torch.split(targets, batch_size, dim=0)
        loss_mask_w, _ = torch.split(loss_mask, batch_size, dim=0)

        loss, sft_loss, or_loss = self.compute_loss(pi_logps_w, pi_logps_l, logits_w, targets_w, loss_mask_w)
        
        self.model_engine.backward(loss)
        self.model_engine.step()
        
        return {"loss": loss.item(), "sft_loss": sft_loss.item(), "or_loss": or_loss.item()}

    def eval_step(self, micro_batch):
        self.model_engine.eval()
        with torch.no_grad():
            batch_size = micro_batch['chosen_input_ids'].shape[0]
            input_ids = torch.cat([micro_batch['chosen_input_ids'], micro_batch['rejected_input_ids']], dim=0)
            attn_mask = torch.cat([micro_batch['chosen_attn_mask'], micro_batch['rejected_attn_mask']], dim=0)
            loss_mask = torch.cat([micro_batch['chosen_loss_mask'], micro_batch['rejected_loss_mask']], dim=0)

            pi_logps, logits, targets = self.forward(input_ids, attn_mask, loss_mask)
            
            pi_logps_w, pi_logps_l = torch.split(pi_logps, batch_size, dim=0)
            logits_w, _ = torch.split(logits, batch_size, dim=0)
            targets_w, _ = torch.split(targets, batch_size, dim=0)
            loss_mask_w, _ = torch.split(loss_mask, batch_size, dim=0)

            loss, sft_loss, or_loss = self.compute_loss(pi_logps_w, pi_logps_l, logits_w, targets_w, loss_mask_w)
        return {"loss": loss.item(), "sft_loss": sft_loss.item(), "or_loss": or_loss.item()}
