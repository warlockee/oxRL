import torch
import torch.nn.functional as F
from typing import Dict, Any

class DPO:
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
        self.ref_cache = None

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

    def forward(self, input_ids, attn_mask, loss_mask, model_engine):
        '''
            Forward pass through the given model engine.
        '''
        # Token type IDs might be needed for some models
        token_type_ids = torch.zeros_like(input_ids)

        output = model_engine(input_ids=input_ids,
                              attention_mask=attn_mask,
                              token_type_ids=token_type_ids,
                              use_cache=self.use_cache)

        # [B, T, V] -> [B, T-1, V]
        logits = output.logits[:, :-1, :].contiguous()
        # [B, T] -> [B, T-1]
        target_ids = input_ids[:, 1:].contiguous()
        
        logps = self.compute_logps(logits, target_ids, loss_mask)
        return logps

    @staticmethod
    def _row_key(input_ids_row, loss_mask_row):
        import hashlib
        h = hashlib.sha1()
        h.update(input_ids_row.detach().cpu().numpy().tobytes())
        h.update(loss_mask_row.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def build_ref_cache(self, dataloader, device, extra_loaders=()):
        '''
            Precompute reference logps with the *initial* policy weights,
            which are exactly the reference policy (must run before the first
            optimizer step). Avoids holding a second full model in memory —
            required at 32B+. Covers the FULL train and val datasets (the
            distributed sampler reshuffles and drops different tail rows every
            epoch, so a single epoch-0 pass would leave holes).
        '''
        from torch.utils.data import DataLoader
        # ZeRO-3 forwards are collective: every rank must run every batch in
        # lockstep, so all ranks compute the full (identical) cache. The pass
        # is no-grad, so a larger batch keeps it fast.
        datasets = [dataloader.dataset] + [ld.dataset for ld in extra_loaders]
        self.model_engine.eval()
        cache = {}
        with torch.no_grad():
            for ds in datasets:
                full_loader = DataLoader(ds, batch_size=8,
                                         shuffle=False, num_workers=2, pin_memory=True)
                for micro_batch in full_loader:
                    micro_batch = {k: v.to(device) for k, v in micro_batch.items()}
                    bs = micro_batch['chosen_input_ids'].shape[0]
                    input_ids = torch.cat([micro_batch['chosen_input_ids'], micro_batch['rejected_input_ids']], dim=0)
                    attn_mask = torch.cat([micro_batch['chosen_attn_mask'], micro_batch['rejected_attn_mask']], dim=0)
                    loss_mask = torch.cat([micro_batch['chosen_loss_mask'], micro_batch['rejected_loss_mask']], dim=0)
                    logps = self.forward(input_ids, attn_mask, loss_mask, self.model_engine)
                    for i in range(2 * bs):
                        cache[self._row_key(input_ids[i], loss_mask[i])] = logps[i].item()
        self.ref_cache = cache

    def ref_forward(self, input_ids, attn_mask, loss_mask):
        '''
            Reference logps. Priority: separate ref engine; precomputed cache
            (initial-weights logps, exact); PEFT adapter-disable (LoRA
            policies). All three are mathematically the same reference.
        '''
        if self.ref_model_engine is not None:
            return self.forward(input_ids, attn_mask, loss_mask, self.ref_model_engine)
        if self.ref_cache is not None:
            vals = [self.ref_cache[self._row_key(input_ids[i], loss_mask[i])]
                    for i in range(input_ids.shape[0])]
            return torch.tensor(vals, device=input_ids.device, dtype=torch.float32)
        m = getattr(self.model_engine, 'module', self.model_engine)
        if not hasattr(m, 'disable_adapter'):
            raise RuntimeError("DPO without a ref_model requires a PEFT/LoRA policy or a prebuilt ref cache")
        with m.disable_adapter():
            return self.forward(input_ids, attn_mask, loss_mask, self.model_engine)

    def compute_loss(self, pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l):
        '''
           DPO Loss: -log(sigmoid(beta * (log_ratio_w - log_ratio_l)))
           log_ratio = log(pi / pi_ref)
        '''
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
        '''
           One training step for DPO.
           micro_batch contains: chosen_input_ids, rejected_input_ids, ...
        '''
        self.model_engine.train()
        if self.ref_model_engine is not None:
            self.ref_model_engine.eval()

        # Combine chosen and rejected inputs for a single forward pass
        batch_size = micro_batch['chosen_input_ids'].shape[0]
        input_ids = torch.cat([micro_batch['chosen_input_ids'], micro_batch['rejected_input_ids']], dim=0)
        attn_mask = torch.cat([micro_batch['chosen_attn_mask'], micro_batch['rejected_attn_mask']], dim=0)
        loss_mask = torch.cat([micro_batch['chosen_loss_mask'], micro_batch['rejected_loss_mask']], dim=0)

        # 1. Reference model logps
        with torch.no_grad():
            ref_logps = self.ref_forward(input_ids, attn_mask, loss_mask)
            ref_logps_w, ref_logps_l = torch.split(ref_logps, batch_size, dim=0)

        # 2. Policy model logps
        pi_logps = self.forward(input_ids, attn_mask, loss_mask, self.model_engine)
        pi_logps_w, pi_logps_l = torch.split(pi_logps, batch_size, dim=0)

        # 3. Compute DPO loss
        loss, margin = self.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # 4. Backward and Step
        self.model_engine.backward(loss)
        self.model_engine.step()

        return {"loss": loss.item(), "margin": margin.item()}

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

            ref_logps = self.ref_forward(input_ids, attn_mask, loss_mask)
            ref_logps_w, ref_logps_l = torch.split(ref_logps, batch_size, dim=0)
            
            pi_logps = self.forward(input_ids, attn_mask, loss_mask, self.model_engine)
            pi_logps_w, pi_logps_l = torch.split(pi_logps, batch_size, dim=0)
            
            loss, margin = self.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        return {"loss": loss.item(), "margin": margin.item()}
