import torch
import torch.nn.functional as F

class KD:
    def __init__(self,
                model_engine,
                ref_model_engine,
                optimizer,
                alpha=0.5,
                temperature=2.0,
                use_cache=False,
                normalize_loss=False):

        self.model_engine = model_engine
        self.ref_model_engine = ref_model_engine  # frozen teacher model
        self.optimizer = optimizer
        self.alpha = alpha
        self.temperature = temperature
        self.use_cache = use_cache
        self.normalize_loss = normalize_loss

        # use cross entropy loss for hard labels
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, batch, model_engine):
        '''
            Forward pass through the given model engine.
            batch['input_ids/attn_mask'] are [B, T]
            batch['position_ids'] is [B, T] or None
            Returns:
                logits: [B, T-1, vocab_size]
                target_ids: [B, T-1]
                loss_mask: [B, T-1]
        '''
        input_ids = batch['input_ids']
        att_mask  = batch['attn_mask']

        pos_ids = batch.get('position_ids', None)
        if pos_ids is not None:
            pos_ids = pos_ids.to(att_mask.device)

        token_type_ids = torch.zeros_like(input_ids)
        output = model_engine(input_ids=input_ids,
                              attention_mask=att_mask,
                              position_ids=pos_ids,
                              token_type_ids=token_type_ids,
                              use_cache=self.use_cache)

        # [B, T, vocab_size] -> [B, T-1, vocab_size]
        logits = output.logits[:, :-1, :].contiguous()
        # [B, T] -> [B, T-1]
        target_ids = input_ids[:, 1:].contiguous()
        # [B, T-1]
        loss_mask = batch['loss_mask'].contiguous()

        return logits, target_ids, loss_mask

    def compute_loss(self, student_logits, teacher_logits, target_ids, loss_mask):
        r'''
            Computes the Knowledge Distillation loss combining:
            1. CE loss on hard labels (same as SFT)
            2. KL divergence between teacher and student soft distributions

            L = alpha * CE(student, y) + (1 - alpha) * T^2 * KL(teacher_soft || student_soft)

            student_logits: [B, T-1, vocab_size]
            teacher_logits: [B, T-1, vocab_size]
            target_ids: [B, T-1]
            loss_mask: [B, T-1]

            Returns:
                (total_loss, ce_loss, kl_loss)
        '''
        _, _, vocab_size = student_logits.shape

        # ---- CE loss on hard labels (same as SFT.compute_loss) ----
        flat_logits = student_logits.view(-1, vocab_size)
        flat_targets = target_ids.view(-1)
        per_token_ce = self.loss_fn(flat_logits, flat_targets)

        flat_mask = loss_mask.view(-1).to(dtype=per_token_ce.dtype)
        masked_ce = per_token_ce * flat_mask
        ce_loss = masked_ce.sum()

        if self.normalize_loss:
            total_possible_tokens = flat_logits.shape[0]
            if total_possible_tokens == 0:
                raise ValueError("Cannot compute loss: total_possible_tokens is 0")
            ce_loss = ce_loss / total_possible_tokens

        # ---- KL divergence on soft labels ----
        T = self.temperature

        # teacher_soft: softmax(teacher_logits / T)  [B, T-1, vocab_size]
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        # student_soft: log_softmax(student_logits / T)  [B, T-1, vocab_size]
        student_log_soft = F.log_softmax(student_logits / T, dim=-1)

        # KL(teacher_soft || student_soft) per token: [B, T-1, vocab_size] -> sum over vocab -> [B, T-1]
        per_token_kl = F.kl_div(student_log_soft, teacher_soft, reduction='none').sum(dim=-1)

        # Apply mask
        masked_kl = per_token_kl * loss_mask.to(dtype=per_token_kl.dtype)
        kl_loss = masked_kl.sum()

        if self.normalize_loss:
            total_possible_tokens = loss_mask.numel()
            if total_possible_tokens == 0:
                raise ValueError("Cannot compute loss: total_possible_tokens is 0")
            kl_loss = kl_loss / total_possible_tokens

        # Scale KL by T^2
        kl_loss = (T ** 2) * kl_loss

        # ---- Total loss ----
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss

        return total_loss, ce_loss, kl_loss

    def train_step(self, micro_batch):
        '''
           One training step for Knowledge Distillation.
           Uses the frozen teacher (ref_model_engine) to produce soft targets,
           then trains the student (model_engine) with combined CE + KL loss.
        '''
        self.model_engine.train()
        self.ref_model_engine.eval()

        # 1. Teacher forward (no gradients needed)
        with torch.no_grad():
            teacher_logits, _, _ = self.forward(micro_batch, self.ref_model_engine)

        # 2. Student forward
        student_logits, target_ids, loss_mask = self.forward(micro_batch, self.model_engine)

        # 3. Compute combined loss
        total_loss, ce_loss, kl_loss = self.compute_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            target_ids=target_ids,
            loss_mask=loss_mask
        )

        # 4. Backward step
        self.model_engine.backward(total_loss)

        # 5. Optimizer step
        self.model_engine.step()

        return {
            "loss": float(total_loss.item()),
            "ce_loss": float(ce_loss.item()),
            "kl_loss": float(kl_loss.item())
        }

    def eval_step(self, micro_batch):
        '''
           Validation step for Knowledge Distillation.
        '''
        self.model_engine.eval()
        self.ref_model_engine.eval()

        with torch.no_grad():
            # Teacher forward
            teacher_logits, _, _ = self.forward(micro_batch, self.ref_model_engine)

            # Student forward
            student_logits, target_ids, loss_mask = self.forward(micro_batch, self.model_engine)

            # Compute combined loss
            total_loss, ce_loss, kl_loss = self.compute_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                target_ids=target_ids,
                loss_mask=loss_mask
            )

        return {
            "loss": float(total_loss.item()),
            "ce_loss": float(ce_loss.item()),
            "kl_loss": float(kl_loss.item())
        }
