import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import ray
import os
from typing import Any, Dict
from transformers import AutoConfig

from oxrl.utils.setup import load_model_and_ref
from oxrl.algs.base import BaseAlgorithm

# PPO with value head (critic) and GAE for advantage estimation.
# Pro:  fine-grained per-token credit assignment via learned value function.
# Con:  ~2x memory (value head), harder to tune (vf_clip, tau, gamma).
# Use when: reward is sparse or delayed and you need better credit assignment.
# Prefer sgrpo/gspo for math/code tasks where outcome reward is sufficient.
@ray.remote
class PPO(BaseAlgorithm):
    def __init__(self,
                 model_path: str,
                 model_dtype: torch.dtype,
                 trust_remote_code: bool,
                 attn_impl: str,
                 kl_coeff: float,
                 clip_low: float,
                 clip_high: float,
                 vf_clip: float,
                 tau: float,
                 gamma: float,
                 entropy_coeff: float,
                 use_cache: bool,
                 micro_batch_size_per_gpu: int,
                 update_after_full_replay: bool,
                 deepspeed_config: Any,
                 lora_config=None,
                 ref_model_path: str = None,
                 deepspeed_ref_config: Any = None,
                 lr: float = 1e-5,
                 betas: list = None,
                 weight_decay: float = 0.01,
                 adam_epsilon: float = 1e-8,
                 model_class: str = "llm",
                 freeze_vision_encoder: bool = True,
                 # accept but ignore GRPO-specific kwargs so training_engine_setup works unchanged
                 **kwargs,
                 ):

        self.alg_name = "PPO"

        # model related parameters
        self.model_path = model_path
        self.ref_model_path = ref_model_path
        self.use_cache = use_cache
        self.attn_impl = attn_impl
        self.model_dtype = model_dtype
        self.trust_remote_code = trust_remote_code
        self.lora_config = lora_config

        # training related parameters
        self.deepspeed_config = deepspeed_config
        self.deepspeed_ref_config = deepspeed_ref_config
        self.micro_batch_size_per_gpu = micro_batch_size_per_gpu

        # policy related parameters
        self.kl_coeff = float(kl_coeff)
        self.clip_low = float(clip_low)
        self.clip_high = float(clip_high)
        self.ent_coeff = float(entropy_coeff)

        # PPO-specific parameters
        self.tau = float(tau)           # GAE lambda
        self.gamma = float(gamma)       # discount factor
        self.vf_clip = float(vf_clip)   # value function clip range

        # optimizer hyperparameters
        self.lr = float(lr)
        self.betas = betas if betas is not None else [0.9, 0.95]
        self.weight_decay = float(weight_decay)
        self.adam_epsilon = float(adam_epsilon)

        # VLM / multimodal
        self.model_class = model_class
        self.freeze_vision_encoder = freeze_vision_encoder
        self.processor = None

        # use cross entropy loss for policy gradient
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

        # if true, the update is done after seeing all samples in the replay buffer
        # treating the entire buffer as a single batch.
        self.update_only_after_full_replay = update_after_full_replay

        self.ready = False
        self.init_training_engine()
        self.ready = True

    def is_ready(self) -> bool:
        '''
            Barrier method to ensure all Ray actors are initialized before DeepSpeed collective ops.
        '''
        return self.ready

    def init_training_engine(self):
        '''
            Since we are using ray, each ray actor MUST create its own deepspeed engine.
            Each ray actor process is a separate process: 1 actor = 1 gpu = 1 ds rank.
            For PPO we also create a value head and its own optimizer.
        '''
        from oxrl.utils.utils import import_deepspeed_safely
        deepspeed = import_deepspeed_safely()

        # Convert pydantic model to python Dict for DeepSpeed
        ds_config_dict = self.deepspeed_config.model_dump()

        # check to avoid re-initializing distributed backend
        if not torch.distributed.is_initialized():
            deepspeed.init_distributed()

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print(f"[Alg:{self.alg_name}][Rank {rank}] Initializing training engine...")

        # 1. Load model (and optional reference model)
        model, ref_model = self.load_model()
        print(f"[Alg:{self.alg_name}][Rank {rank}] Model loaded: {self.model_path}")

        # Load processor and freeze vision encoder for VLM training
        if self.model_class == "vlm":
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=self.trust_remote_code
            )
            print(f"[Alg:{self.alg_name}][Rank {rank}] AutoProcessor loaded for VLM training")
            if self.freeze_vision_encoder:
                self._freeze_vision_encoder(model)

        # 2. Apply LoRA if enabled
        if self.lora_config and self.lora_config.enabled:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
                model = prepare_model_for_kbit_training(model)

            peft_config = LoraConfig(
                r=self.lora_config.r,
                lora_alpha=self.lora_config.lora_alpha,
                target_modules=self.lora_config.target_modules,
                lora_dropout=self.lora_config.lora_dropout,
                bias=self.lora_config.bias,
                task_type=self.lora_config.task_type,
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        # 3. Initialize policy DeepSpeed engine
        # Remove optimizer from ds_config_dict so deepspeed doesn't build FusedAdam
        if "optimizer" in ds_config_dict:
            del ds_config_dict["optimizer"]

        # Filter for trainable parameters (crucial for LoRA)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            betas=tuple(self.betas),
            weight_decay=self.weight_decay,
            eps=self.adam_epsilon,
        )

        self.policy_engine, self.optimizer, _, _ = deepspeed.initialize(
                                                            model=model,
                                                            model_parameters=trainable_params,
                                                            config=ds_config_dict,
                                                            optimizer=optimizer
                                                            )
        print(f"[Alg:{self.alg_name}][Rank {rank}] DeepSpeed policy engine initialized on device: {self.policy_engine.device}")

        # 4. Initialize reference model engine (optional, for KL penalty)
        self.ref_model_engine = None
        if ref_model is not None:
            ref_model.eval()
            ref_ds_config = self.deepspeed_ref_config.model_dump()
            self.ref_model_engine, _, _, _ = deepspeed.initialize(
                                                        model=ref_model,
                                                        config=ref_ds_config
                                                        )
            print(f"[Alg:{self.alg_name}][Rank {rank}] Reference model initialized with DeepSpeed")

        # 5. Initialize value head
        # Use the policy model's hidden states + a linear value head.
        # This avoids a full second model; we just need a lightweight projection.
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code)
        hidden_size = config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1).to(
            device=self.policy_engine.device,
            dtype=self.policy_engine.dtype if hasattr(self.policy_engine, 'dtype') else torch.float32,
        )
        self.value_optimizer = torch.optim.AdamW(
            self.value_head.parameters(),
            lr=self.lr,
            betas=tuple(self.betas),
            weight_decay=self.weight_decay,
            eps=self.adam_epsilon,
        )
        print(f"[Alg:{self.alg_name}][Rank {rank}] Value head initialized (hidden_size={hidden_size})")

    def load_model(self):
        return load_model_and_ref(
            model_path=self.model_path,
            model_dtype=self.model_dtype,
            trust_remote_code=self.trust_remote_code,
            attn_impl=self.attn_impl,
            ref_model_path=self.ref_model_path if self.kl_coeff > 0.0 else None
        )

    def ref_forward(self, input_ids, att_mask, target_ids, pos_ids, mm_kwargs=None):
        '''
            input_ids and att_mask are [B, T]
            pos_ids is [B, T] or None
            target_ids is [B, T-1]
            mm_kwargs: optional dict of multimodal tensors (pixel_values, etc.)
            Returns:
                ref_logprobs: [B, T-1]
        '''
        with torch.no_grad():
            if pos_ids is not None:
                pos_ids = pos_ids.to(input_ids.device)

            token_type_ids = torch.zeros_like(input_ids)

            forward_kwargs = dict(
                input_ids=input_ids,
                attention_mask=att_mask,
                position_ids=pos_ids,
                token_type_ids=token_type_ids,
                use_cache=self.use_cache,
            )
            if mm_kwargs:
                forward_kwargs.update(mm_kwargs)

            output = self.ref_model_engine(**forward_kwargs)

            # [B, T, V] -> [B, T-1, V]
            logits = output.logits[:, :-1, :].contiguous()
            B, T_minus_1, vocab_size = logits.shape

            neg_logprobs = self.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1))
            ref_logprobs = -neg_logprobs.view(B, T_minus_1)

        return ref_logprobs

    def policy_forward(self, input_ids, att_mask, pos_ids, mm_kwargs=None):
        '''
            input_ids and att_mask are [B, T]
            pos_ids is [B, T] or None
            mm_kwargs: optional dict of multimodal tensors (pixel_values, etc.)
            Returns:
                logprobs: [B, T-1]
                entropies: [B, T-1] or None
                target_ids: [B, T-1]
        '''
        if pos_ids is not None:
            pos_ids = pos_ids.to(input_ids.device)

        token_type_ids = torch.zeros_like(input_ids)

        forward_kwargs = dict(
            input_ids=input_ids,
            attention_mask=att_mask,
            position_ids=pos_ids,
            token_type_ids=token_type_ids,
            use_cache=self.use_cache,
        )
        if mm_kwargs:
            forward_kwargs.update(mm_kwargs)

        output = self.policy_engine(**forward_kwargs)

        # [B, T, V] -> [B, T-1, V]
        logits = output.logits[:, :-1, :].contiguous()
        B, T_minus_1, vocab_size = logits.shape

        # [B, T] -> [B, T-1]
        target_ids = input_ids[:, 1:].contiguous()

        neg_logprobs = self.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1))
        logprobs = -neg_logprobs.view(B, T_minus_1)

        entropies = None
        if self.ent_coeff > 0.0:
            entropies = torch.distributions.Categorical(logits=logits).entropy()

        return logprobs, entropies, target_ids

    def value_forward(self, input_ids, att_mask, pos_ids, mm_kwargs=None):
        '''
            Forward through the policy model to get hidden states, then project
            through the value head to get per-token value estimates.

            Input:
                input_ids/att_mask: [B, T]
                pos_ids: [B, T] or None
                mm_kwargs: optional dict of multimodal tensors (pixel_values, etc.)
            Returns:
                values: [B, T-1] aligned with input state at t, predicting future
                last_value: [B] value of the very last token (for bootstrapping)
        '''
        if pos_ids is not None:
            pos_ids = pos_ids.to(input_ids.device)

        token_type_ids = torch.zeros_like(input_ids)

        forward_kwargs = dict(
            input_ids=input_ids,
            attention_mask=att_mask,
            position_ids=pos_ids,
            token_type_ids=token_type_ids,
            use_cache=self.use_cache,
            output_hidden_states=True,
        )
        if mm_kwargs:
            forward_kwargs.update(mm_kwargs)

        output = self.policy_engine(**forward_kwargs)

        # Last hidden state: [B, T, H]
        # Detach from the policy computation graph so that value loss
        # gradients do NOT leak into the policy engine parameters.
        hidden_states = output.hidden_states[-1].detach()

        # Project through value head: [B, T, H] -> [B, T, 1] -> [B, T]
        every_token_values = self.value_head(hidden_states).squeeze(-1)

        # [B, T] -> [B, T-1] (aligned with logprobs which are shifted by 1)
        values = every_token_values[:, :-1].contiguous()
        # [B] for bootstrapping
        last_value = every_token_values[:, -1].contiguous()

        return values, last_value

    def compute_advantages(self,
                           rewards: torch.Tensor,
                           values: torch.Tensor,
                           done: torch.Tensor,
                           mask: torch.Tensor,
                           last_val: torch.Tensor | None = None,
                          ):
        '''
            rewards, values: [B, T]
            done, mask: [B, T]
            done:    1 if t is EOS (terminal), 0 otherwise.
                     MUST be set at every packed sequence boundary so it
                     shows the boundary of each sequence.
            mask:    1 if valid token, 0 if padding.
            GAE and returns: [B, T]
            last_val: [B]
            return: rets, advs which would be both [B, T]
        '''
        # 1. Device and shape setup
        device = values.device
        dtype  = values.dtype
        B, T   = values.shape
        rets   = torch.zeros_like(values)
        advs   = torch.zeros_like(values)
        last_adv = torch.zeros(B, dtype=dtype, device=device)
        rewards  = rewards.to(dtype=dtype, device=device)

        # 2. Delay casting the mask to the same dtype for indexing and checks.
        mask  = mask.to(device=device)
        done  = done.to(device=device)
        mask  = (mask > 0.5)
        done  = (done > 0.5)

        # 3. Check for nan in rewards or values for valid tokens
        if not torch.isfinite(rewards[mask]).all() or not torch.isfinite(values[mask]).all():
            raise ValueError("rewards or values contain NaN on valid positions")

        if (done & (~mask)).any():
            raise ValueError("done flag set on padding positions")

        # 4. Reject *holes* in the valid region.
        #    Pred-aligned masks have the pattern: [0..0, 1..1, 0..0]
        #    (leading prompt zeros, response ones, trailing padding zeros).
        #    We allow exactly ONE contiguous block of 1s, which means at most
        #    ONE 0->1 transition.  Two or more 0->1 transitions indicate
        #    a gap (e.g., [1,1,0,1,1]) which the backward GAE sweep cannot
        #    handle correctly.
        transitions_01 = mask[:, 1:] & (~mask[:, :-1])   # [B, T-1]
        if transitions_01.sum(dim=1).max().item() > 1:
            raise ValueError(
                "mask has more than one 0->1 transition per row (hole in valid region). "
                "Expected pattern: [0..0, 1..1, 0..0]."
            )

        # prefill val and reward for invalid tokens (i.e., padding) as they can contain nan in padded slot
        rewards = rewards.masked_fill(~mask, 0.0)
        values  = values.detach().masked_fill(~mask, 0.0)

        # 5. empty sequences
        if T == 0:
            empty = rewards.new_zeros((B, 0))
            return empty, empty

        # 6. next value
        if last_val is not None:
            next_val = last_val.to(dtype=dtype, device=device).detach().reshape(B)
        else:
            # biased estimation especially where there is need for bootstrapping, i.e.,
            # no EOS in generation like [x1,x2,x3]
            next_val = torch.zeros(B, dtype=dtype, device=device)

        # 7. Using (tensor > 0.5) is safer than bool() if inputs are already floats
        # especially in case of BF16/FP16 training.
        mask  = mask.to(dtype=dtype, device=device)
        done  = done.to(dtype=dtype, device=device)

        # 8. Compute returns and advantages
        for t in reversed(range(T)): # [T-1, 0]
            # Done is 1 if EOS/Terminal, we do NOT bootstrap from t+1.
            not_done = 1.0 - done[:, t]
            is_valid = mask[:, t]

            # GAE: A[t] = delta[t] + gamma * tau * A[t+1] * (1 - done[t])
            delta = rewards[:, t] + (self.gamma * next_val * not_done) - values[:, t]
            last_adv   = is_valid * (delta + (self.gamma * self.tau * last_adv * not_done))
            advs[:, t] = last_adv

            # to avoid any leaking from padding.
            next_val = values[:, t] * is_valid

        rets = advs + values

        return rets, advs

    def compute_policy_loss(self,
                            logprobs: torch.Tensor,
                            old_logprobs: torch.Tensor,
                            advantages: torch.Tensor,
                            mask: torch.Tensor,
                            entropies: torch.Tensor,
                            ref_logprobs: torch.Tensor = None,
                            ):
        '''
            PPO uses the same token-level clipped surrogate as SGRPO.
            See oxrl/algs/losses/sgrpo.py for the implementation.
        '''
        from oxrl.algs.losses import get_loss_fn
        loss_fn = get_loss_fn("ppo")
        return loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            mask=mask,
            entropies=entropies,
            ref_logprobs=ref_logprobs,
            clip_low=self.clip_low,
            clip_high=self.clip_high,
            ent_coeff=self.ent_coeff,
            kl_coeff=self.kl_coeff,
        )

    def compute_value_loss(self,
                           values: torch.Tensor,
                           v_old: torch.Tensor,
                           returns: torch.Tensor,
                           mask: torch.Tensor,
                           ):
        '''
            Compute clipped value loss:
                1. if v_old:  loss = 0.5 * max((values - rets)^2, (v_clipped - rets)^2)
                2. otherwise: loss = 0.5 * (values - rets)^2
        '''
        rets = returns.detach()
        v_loss = (values - rets).pow(2)
        denom = mask.sum().clamp(min=1.0)

        if self.vf_clip > 0 and v_old is not None:
            v_old = v_old.detach()

            v_clipped = v_old + torch.clamp(values - v_old, -self.vf_clip, self.vf_clip)
            v_loss_clipped = (v_clipped - rets).pow(2)
            vmax = torch.maximum(v_loss, v_loss_clipped)
            loss = 0.5 * (vmax * mask).sum() / denom

            with torch.no_grad():
                vf_clipfrac = (values - v_old).abs() > self.vf_clip
                vf_clipfrac = (vf_clipfrac * mask).sum() / denom

        else:
            loss = 0.5 * (v_loss * mask).sum() / denom
            vf_clipfrac = 0.0

        metrics = {
            'vf_clipfrac': vf_clipfrac if isinstance(vf_clipfrac, float) else vf_clipfrac.item(),
        }

        return loss, metrics

    def train_step(self, engine_id, micro_batches):
        '''
           This function implements a training step per rank/gpu for the local batch.
           The batch size for each gpu/rank should be micro_batch_size_per_gpu.
           micro_batches is a partition of the replay buffer (list of micro-batches) for the current rank/gpu.

           Interface matches GRPO.train_step(self, engine_id, micro_batches).
           Returns a dict of averaged metrics compatible with main_rl.py aggregation.
        '''
        assert self.policy_engine is not None, "DeepSpeed engine not initialized"

        device = self.policy_engine.device

        # 1. Models to train mode
        self.policy_engine.train()
        self.value_head.train()

        # 2. zero grads
        self.policy_engine.zero_grad()
        self.value_optimizer.zero_grad()

        # 3. create progress bar
        num_micro = len(micro_batches)
        if engine_id == 0:
            progress_bar = tqdm(micro_batches, total=num_micro, desc=f"[Alg:{self.alg_name}] Training Step in rank {engine_id}")
        else:
            progress_bar = micro_batches

        ga_pi_attr = getattr(self.policy_engine, 'gradient_accumulation_steps', 1)
        ga_pi = int(ga_pi_attr() if callable(ga_pi_attr) else ga_pi_attr)

        # track metrics across all micro-batches
        all_metrics = []
        for step, micro_batch in enumerate(progress_bar):
            is_last = (step == (num_micro - 1))
            is_boundary = (((step + 1) % ga_pi) == 0) or is_last

            ########
            # 1. Data from buffer
            ########
            # all are [B, T]
            rewards      = micro_batch['rewards'].to(device, non_blocking=True)
            done         = micro_batch['done'].to(device, non_blocking=True)
            mask         = micro_batch['mask'].to(device, non_blocking=True)
            old_logprobs = micro_batch['old_logprobs'].to(device, non_blocking=True)

            # PPO-specific: values and last_val from rollout
            v_olds   = micro_batch.get('v_olds', None)
            if v_olds is not None:
                v_olds = v_olds.to(device, non_blocking=True)
            last_val = micro_batch.get('last_val', None)
            if last_val is not None:
                last_val = last_val.to(device, non_blocking=True)

            input_ids = micro_batch['input_ids'].to(device, non_blocking=True)
            att_mask  = micro_batch['attn_mask'].to(device, non_blocking=True)
            pos_ids   = micro_batch.get('position_ids', None)

            # Prepare multimodal inputs if present
            mm_kwargs = {}
            if self.model_class == "vlm" and micro_batch.get("multi_modal_data") is not None:
                mm_kwargs = self._prepare_mm_inputs(micro_batch["multi_modal_data"], device)

            ########
            # 2. Get current values from value head (for GAE computation)
            ########
            # If old values were provided from rollout, use them.
            # Otherwise compute fresh values (less ideal but functional).
            if v_olds is not None:
                # Use rollout-time values for GAE (standard PPO)
                # Slice to [B, T-1] to align with logprobs
                values_for_gae = v_olds[:, :-1]
                last_val_for_gae = last_val
                # done and mask also need slicing to [B, T-1]
                rewards_gae = rewards[:, :-1]
                done_gae    = done[:, :-1]
                mask_gae    = mask[:, :-1]
            else:
                # Compute values from current policy hidden states
                with torch.no_grad():
                    values_for_gae, last_val_for_gae = self.value_forward(input_ids, att_mask, pos_ids, mm_kwargs=mm_kwargs)
                # values_for_gae is already [B, T-1], slice rewards/done/mask to match
                rewards_gae = rewards[:, :-1]
                done_gae    = done[:, :-1]
                mask_gae    = mask[:, :-1]

            ########
            # 3. Compute advantages with GAE
            ########
            returns, advantages = self.compute_advantages(
                                                    rewards=rewards_gae,
                                                    values=values_for_gae,
                                                    done=done_gae,
                                                    mask=mask_gae,
                                                    last_val=last_val_for_gae)

            # Normalize advantages
            adv_mean = (advantages * mask_gae).sum() / mask_gae.sum().clamp(min=1)
            adv_var  = ((advantages - adv_mean).pow(2) * mask_gae).sum() / mask_gae.sum().clamp(min=1)
            advantages = (advantages - adv_mean) / (adv_var.sqrt() + 1e-8)

            ########
            # 4. Policy update
            ########
            pi_logprobs, pi_entropies, target_ids = self.policy_forward(
                                                                input_ids=input_ids,
                                                                att_mask=att_mask,
                                                                pos_ids=pos_ids,
                                                                mm_kwargs=mm_kwargs)

            # Slice old_logprobs to [B, T-1] to match pi_logprobs
            old_lp = old_logprobs[:, :-1]

            ref_logprobs = None
            if self.kl_coeff > 0.0 and self.ref_model_engine is not None:
                ref_logprobs = self.ref_forward(input_ids=input_ids,
                                                att_mask=att_mask,
                                                target_ids=target_ids,
                                                pos_ids=pos_ids,
                                                mm_kwargs=mm_kwargs)

            pi_loss, pi_metrics = self.compute_policy_loss(
                                                    logprobs=pi_logprobs,
                                                    old_logprobs=old_lp,
                                                    advantages=advantages,
                                                    mask=mask_gae,
                                                    entropies=pi_entropies,
                                                    ref_logprobs=ref_logprobs)

            if self.update_only_after_full_replay:
                self.policy_engine.set_gradient_accumulation_boundary(is_boundary)
            else:
                self.policy_engine.set_gradient_accumulation_boundary(True)

            # backward pass for policy
            self.policy_engine.backward(pi_loss)
            self.policy_engine.step()

            ########
            # 5. Value update
            ########
            values_new, _ = self.value_forward(input_ids, att_mask, pos_ids, mm_kwargs=mm_kwargs)
            vl_loss, vl_metrics = self.compute_value_loss(
                                                    values=values_new,
                                                    v_old=values_for_gae,
                                                    returns=returns,
                                                    mask=mask_gae)

            vl_loss.backward()
            self.value_optimizer.step()
            self.value_optimizer.zero_grad()

            ########
            # 6. Collect metrics
            ########
            step_metrics = dict(pi_metrics)
            step_metrics['vf_clipfrac'] = vl_metrics['vf_clipfrac']
            all_metrics.append(step_metrics)

            if engine_id == 0:
                progress_bar.set_postfix({
                    "loss": f"{pi_loss.item():.4f}",
                    "clip": f"{pi_metrics['clipfrac']:.3f}",
                    "kl_old": f"{pi_metrics['kl_old']:.4f}",
                    "kl_ref": f"{pi_metrics['kl_ref']:.4f}",
                })

        # aggregate metrics across all micro-batches
        aggregated_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                aggregated_metrics[key] = np.mean([m[key] for m in all_metrics])

        return aggregated_metrics

    def offload_to_cpu(self) -> bool:
        """Offload DeepSpeed optimizer states, gradients, and cached memory to CPU.

        Frees GPU memory so the vLLM rollout engine can reload the updated model.
        See GRPO.offload_to_cpu for detailed documentation.
        """
        import gc
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print(f"[Alg:{self.alg_name}][Rank {rank}] Offloading optimizer states to CPU...")

        try:
            self.policy_engine.zero_grad()

            optimizer = self.policy_engine.optimizer
            offloaded = 0
            base_optimizer = getattr(optimizer, 'optimizer', optimizer)
            for state in base_optimizer.state.values():
                for key, val in state.items():
                    if isinstance(val, torch.Tensor) and val.is_cuda:
                        state[key] = val.to('cpu', non_blocking=True)
                        offloaded += 1

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

            if torch.cuda.is_available():
                free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
                total_mem = torch.cuda.mem_get_info()[1] / (1024**3)
                print(
                    f"[Alg:{self.alg_name}][Rank {rank}] Offloaded {offloaded} optimizer tensors to CPU. "
                    f"GPU memory: {free_mem:.2f}/{total_mem:.2f} GiB free"
                )
            return True
        except Exception as e:
            print(f"[Alg:{self.alg_name}][Rank {rank}] WARNING: offload_to_cpu failed: {e}")
            return False

    def reload_to_gpu(self) -> bool:
        """Reload optimizer states back to GPU after vLLM refresh completes."""
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        device = self.policy_engine.device
        print(f"[Alg:{self.alg_name}][Rank {rank}] Reloading optimizer states to {device}...")

        try:
            optimizer = self.policy_engine.optimizer
            base_optimizer = getattr(optimizer, 'optimizer', optimizer)
            reloaded = 0
            for state in base_optimizer.state.values():
                for key, val in state.items():
                    if isinstance(val, torch.Tensor) and val.device.type == 'cpu':
                        state[key] = val.to(device, non_blocking=True)
                        reloaded += 1

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            print(f"[Alg:{self.alg_name}][Rank {rank}] Reloaded {reloaded} optimizer tensors to {device}")
            return True
        except Exception as e:
            print(f"[Alg:{self.alg_name}][Rank {rank}] WARNING: reload_to_gpu failed: {e}")
            return False

    def gather_state_dict(self):
        """Gather ZeRO-3 partitioned weights into a single state dict in memory.

        Collective operation — must be called on ALL ranks.
        Returns the state dict on rank 0 (with value head), None on other ranks.
        """
        from oxrl.tools.lora_merge import strip_lora_and_merge
        from oxrl.tools.checkpoint import get_base_model_config

        rank = torch.distributed.get_rank()
        print(f"[Alg:{self.alg_name}][Rank {rank}] Gathering state dict in memory...")

        state_dict = self.policy_engine._zero3_consolidated_16bit_state_dict()

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if rank == 0 and state_dict is not None:
            if self.lora_config and self.lora_config.enabled:
                print(f"[Alg:{self.alg_name}][Rank {rank}] Merging LoRA weights in memory...")
                state_dict = strip_lora_and_merge(
                    state_dict, self.lora_config.lora_alpha, self.lora_config.r
                )

            config = get_base_model_config(self.policy_engine)
            if config is not None:
                state_dict["__model_config_dict__"] = config.to_dict()

            state_dict["__value_head_state_dict__"] = {
                k: v.cpu() for k, v in self.value_head.state_dict().items()
            }

            print(f"[Alg:{self.alg_name}][Rank {rank}] State dict gathered: {len(state_dict)} keys (including value head)")
            return state_dict
        else:
            return None

    def save_checkpoint(self, output_dir: str, tag: str, state_dict_ref=None):
        '''
            Saves the model in hf compatible format for vllm, etc.

            If state_dict_ref is provided, rank 0 writes the pre-gathered state dict.
            If state_dict_ref is None, falls back to save_16bit_model (original behavior).
            Also saves the value head state dict on rank 0.
        '''
        from oxrl.tools.checkpoint import (
            save_state_dict_to_safetensors,
            save_config_json,
            fix_lora_checkpoint_files,
            get_base_model_config,
        )

        rank = torch.distributed.get_rank()
        print(f"[Alg:{self.alg_name}][Rank {rank}] Saving checkpoint to {output_dir} with tag {tag}...")

        try:
            if state_dict_ref is not None:
                if rank == 0:
                    state_dict = state_dict_ref
                    config_dict = state_dict.pop("__model_config_dict__", None)
                    value_head_sd = state_dict.pop("__value_head_state_dict__", None)

                    save_state_dict_to_safetensors(output_dir, state_dict)
                    print(f"[Alg:{self.alg_name}][Rank {rank}] Weights saved via object store")

                    if config_dict is not None:
                        save_config_json(output_dir, config_dict)
                        print(f"[Alg:{self.alg_name}][Rank {rank}] Config saved")

                    if value_head_sd is not None:
                        torch.save(value_head_sd, os.path.join(output_dir, "value_head.pt"))
                        print(f"[Alg:{self.alg_name}][Rank {rank}] Value head saved")

                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
            else:
                self.policy_engine.save_16bit_model(output_dir)

                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

                if rank == 0 and self.lora_config and self.lora_config.enabled:
                    print(f"[Alg:{self.alg_name}][Rank {rank}] Stripping PEFT prefixes and merging weights...")
                    fix_lora_checkpoint_files(
                        output_dir, self.lora_config.lora_alpha, self.lora_config.r
                    )

                if rank == 0:
                    config = get_base_model_config(self.policy_engine)
                    if config is not None:
                        config.save_pretrained(output_dir)
                        print(f"[Alg:{self.alg_name}][Rank {rank}] Config saved")

                    value_head_path = os.path.join(output_dir, "value_head.pt")
                    torch.save(self.value_head.state_dict(), value_head_path)
                    print(f"[Alg:{self.alg_name}][Rank {rank}] Value head saved to {value_head_path}")

                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

            print(f"[Alg:{self.alg_name}][Rank {rank}] Checkpoint save completed!")

        except Exception as e:
            print(f"[Alg:{self.alg_name}][Rank {rank}] Error saving checkpoint to {output_dir}: {e}")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            raise
