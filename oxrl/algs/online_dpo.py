import torch
import torch.nn.functional as F
from typing import Dict, Any

class OnlineDPO:
    """Online DPO: DPO with on-the-fly rejection generation. The online generation of rejected samples happens upstream in the data pipeline. This class handles the DPO optimization step."""

    def __init__(self, model_engine, ref_model_engine, optimizer, beta=0.1, use_cache=False, normalize_loss=False):
        self.model_engine = model_engine
        self.ref_model_engine = ref_model_engine
        self.optimizer = optimizer
        self.beta = beta
        self.use_cache = use_cache

    def compute_logps(self, logits, target_ids, loss_mask):
        log_probs = F.log_softmax(logits, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=2, index=target_ids.unsqueeze(2)).squeeze(2)
        logps = (per_token_logps * loss_mask).sum(-1)
        return logps

    def forward(self, input_ids, attn_mask, loss_mask, model_engine):
        token_type_ids = torch.zeros_like(input_ids)
        output = model_engine(input_ids=input_ids, attention_mask=attn_mask,
                              token_type_ids=token_type_ids, use_cache=self.use_cache)
        logits = output.logits[:, :-1, :].contiguous()
        target_ids = input_ids[:, 1:].contiguous()
        logps = self.compute_logps(logits, target_ids, loss_mask)
        return logps

    def compute_loss(self, pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l):
        pi_logr_w = pi_logps_w - ref_logps_w
        pi_logr_l = pi_logps_l - ref_logps_l
        logits = self.beta * (pi_logr_w - pi_logr_l)
        loss = -F.logsigmoid(logits).mean()
        with torch.no_grad():
            rewards_w = self.beta * pi_logr_w
            rewards_l = self.beta * pi_logr_l
            margin = (rewards_w - rewards_l).mean()
        return loss, margin

    def train_step(self, micro_batch):
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
        loss, margin = self.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        self.model_engine.backward(loss)
        self.model_engine.step()
        return {"loss": loss.item(), "margin": margin.item()}

    def eval_step(self, micro_batch):
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
            loss, margin = self.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        return {"loss": loss.item(), "margin": margin.item()}
