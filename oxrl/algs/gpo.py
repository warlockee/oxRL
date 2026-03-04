import torch
import torch.nn.functional as F
from typing import Dict, Any

class GPO:
    """
    GPO: Generalized Preference Optimization.

    Reference: "Generalized Preference Optimization: A Unified Approach to
    Offline Alignment" (Tang et al., ICML 2024).
    https://arxiv.org/abs/2402.05749

    GPO unifies offline preference optimization methods through the lens of
    binary classification with different convex loss functions. Given the
    log-ratio difference:

        h = beta * (log(pi/pi_ref)(y_w|x) - log(pi/pi_ref)(y_l|x))

    the GPO loss is f(h), where f is a convex function. Different choices of
    f recover existing methods and yield novel ones:

    Existing methods (for reference -- use original implementations instead):
        - "logistic":  f(h) = log(1 + exp(-h))    [DPO]
        - "hinge":     f(h) = max(0, 1 - h)       [SLiC]
        - "squared":   f(h) = (h - 1)^2           [IPO variant]

    Novel loss functions proposed by GPO (main value-add of this class):
        - "exponential":         f(h) = exp(-h)
        - "truncated_quadratic": f(h) = max(0, 1-h)^2
        - "savage":              f(h) = 1 / (1 + exp(h))^2

    All losses encourage the model to increase the log-ratio margin between
    chosen and rejected responses, but differ in how they weight samples
    near vs far from the decision boundary.

    - "exponential": Aggressively penalizes incorrectly ranked pairs; decays
      smoothly for well-separated pairs. Similar to AdaBoost's loss.
    - "truncated_quadratic": Like hinge but smoother; zero loss once margin
      exceeds 1, quadratic penalty near boundary. Good for noisy data.
    - "savage": Very aggressive near boundary (like sigmoid^2). Focuses
      heavily on hard-to-separate pairs.

    The default is "exponential" since it is the most novel and showed
    competitive results in the paper.

    Pro:  Unified framework; novel losses can outperform DPO on specific tasks.
    Con:  Optimal loss type is task-dependent; exponential can be numerically
          sensitive for very large margins.
    """
    def __init__(self,
                model_engine,
                ref_model_engine,
                optimizer,
                beta=0.1,
                gpo_loss_type="exponential",
                use_cache=False,
                normalize_loss=False):

        self.model_engine = model_engine
        self.ref_model_engine = ref_model_engine
        self.optimizer = optimizer
        self.beta = beta
        self.gpo_loss_type = gpo_loss_type
        self.use_cache = use_cache
        self.normalize_loss = normalize_loss

        valid_types = ["logistic", "exponential", "truncated_quadratic", "savage"]
        if self.gpo_loss_type not in valid_types:
            raise ValueError(
                f"Invalid gpo_loss_type '{self.gpo_loss_type}'. "
                f"Must be one of: {valid_types}"
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
           GPO Loss: f(beta * (logr_w - logr_l))
           where f depends on gpo_loss_type.
        '''
        pi_logr_w = pi_logps_w - ref_logps_w
        pi_logr_l = pi_logps_l - ref_logps_l
        h = self.beta * (pi_logr_w - pi_logr_l)

        if self.gpo_loss_type == "logistic":
            # f(h) = log(1 + exp(-h)) = softplus(-h)
            loss = F.softplus(-h).mean()
        elif self.gpo_loss_type == "exponential":
            # f(h) = exp(-h)
            loss = torch.exp(-h).mean()
        elif self.gpo_loss_type == "truncated_quadratic":
            # f(h) = max(0, 1-h)^2
            loss = (torch.relu(1.0 - h) ** 2).mean()
        elif self.gpo_loss_type == "savage":
            # f(h) = 1 / (1 + exp(h))^2 = sigmoid(-h)^2
            loss = (torch.sigmoid(-h) ** 2).mean()

        with torch.no_grad():
            rewards_w = self.beta * pi_logr_w
            rewards_l = self.beta * pi_logr_l
            margin = (rewards_w - rewards_l).mean()
            reward_acc = (rewards_w > rewards_l).float().mean()

        return loss, margin, reward_acc

    def train_step(self, micro_batch):
        '''
           One training step for GPO.
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

        # Compute GPO loss
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
