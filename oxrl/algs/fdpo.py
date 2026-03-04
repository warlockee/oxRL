import torch
import torch.nn.functional as F
from typing import Dict, Any

class FDPO:
    """
    f-DPO: f-Divergence Preference Optimization.

    Reference: "Beyond Reverse KL: Generalizing Direct Preference Optimization
    with Diverse Divergence Constraints" (Wang et al., ICLR 2024).
    https://arxiv.org/abs/2309.16526

    Standard DPO implicitly optimizes a reverse KL divergence constraint between
    the policy and reference. f-DPO generalizes this to other f-divergences,
    which changes the implicit regularization behavior:

    - "reverse_kl" (default): Standard DPO. Mode-seeking -- the policy
      concentrates on high-probability reference modes. Can miss some modes.
    - "forward_kl": Mode-covering -- the policy tries to cover all reference
      modes. Reduces reward hacking but may be conservative.
    - "js_divergence": Jensen-Shannon divergence. Symmetric, balanced between
      mode-seeking and mode-covering.
    - "alpha_divergence": Interpolates between forward and reverse KL via
      alpha parameter. alpha->0 = forward KL, alpha->1 = reverse KL.

    Implementation:
    The f-divergence type modifies how log-ratios are transformed before
    the logistic loss. Given:
        logr_w = log(pi(y_w|x) / pi_ref(y_w|x))
        logr_l = log(pi(y_l|x) / pi_ref(y_l|x))

    Reverse KL: h = beta * (logr_w - logr_l)          [standard DPO]
    Forward KL: h = beta * (logr_w - logr_l)
                    + beta * (exp(-logr_w) - exp(-logr_l))
                    [adds correction term for mode-covering]
    JS:         h = beta * (logr_w - logr_l)
                    - (softplus(logr_w) - softplus(logr_l))
                    [JS divergence adjustment]
    Alpha:      h = (cap_exp(logr_l * -alpha) - cap_exp(logr_w * -alpha))
                    / alpha
                    [Renyi/alpha-divergence interpolation]

    Loss: L = -logsigmoid(h)

    Pro:  Different divergences suit different alignment needs; forward KL
          reduces reward hacking; JS is a balanced choice.
    Con:  Optimal divergence is task-dependent; alpha divergence adds a
          hyperparameter; forward KL may be too conservative for some tasks.
    """
    def __init__(self,
                model_engine,
                ref_model_engine,
                optimizer,
                beta=0.1,
                fdpo_divergence="reverse_kl",
                fdpo_alpha=0.5,
                use_cache=False,
                normalize_loss=False):

        self.model_engine = model_engine
        self.ref_model_engine = ref_model_engine
        self.optimizer = optimizer
        self.beta = beta
        self.fdpo_divergence = fdpo_divergence
        self.fdpo_alpha = fdpo_alpha
        self.use_cache = use_cache
        self.normalize_loss = normalize_loss

        valid_types = ["reverse_kl", "forward_kl", "js_divergence", "alpha_divergence"]
        if self.fdpo_divergence not in valid_types:
            raise ValueError(
                f"Invalid fdpo_divergence '{self.fdpo_divergence}'. "
                f"Must be one of: {valid_types}"
            )

    @staticmethod
    def _cap_exp(x, cap=50.0):
        """Capped exponential to avoid numerical overflow."""
        return torch.exp(x.clamp(max=cap))

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
           f-DPO Loss with configurable f-divergence.
        '''
        logr_w = pi_logps_w - ref_logps_w  # log(pi/pi_ref) for chosen
        logr_l = pi_logps_l - ref_logps_l  # log(pi/pi_ref) for rejected

        # Standard logit difference
        logits = self.beta * (logr_w - logr_l)

        if self.fdpo_divergence == "reverse_kl":
            # Standard DPO: no modification needed
            pass
        elif self.fdpo_divergence == "forward_kl":
            # Forward KL adds: beta * (exp(-logr_w) - exp(-logr_l))
            # This comes from f'(u) = -1/u for forward KL: the derivative
            # applied to the log-ratio gives exp(-logr) correction
            logits = logits + self.beta * (self._cap_exp(-logr_w) - self._cap_exp(-logr_l))
        elif self.fdpo_divergence == "js_divergence":
            # JS divergence: subtract softplus corrections
            # softplus(logr) = log(1 + exp(logr)) = log(1 + pi/pi_ref)
            logits = logits - (F.softplus(logr_w) - F.softplus(logr_l))
        elif self.fdpo_divergence == "alpha_divergence":
            # Alpha divergence: (cap_exp(logr_l * -alpha) - cap_exp(logr_w * -alpha)) / alpha
            alpha = self.fdpo_alpha
            logits = (self._cap_exp(logr_l * -alpha) - self._cap_exp(logr_w * -alpha)) / alpha

        loss = -F.logsigmoid(logits).mean()

        with torch.no_grad():
            rewards_w = self.beta * logr_w
            rewards_l = self.beta * logr_l
            margin = (rewards_w - rewards_l).mean()
            reward_acc = (rewards_w > rewards_l).float().mean()

        return loss, margin, reward_acc

    def train_step(self, micro_batch):
        '''
           One training step for f-DPO.
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

        # Compute f-DPO loss
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
