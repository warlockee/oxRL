import torch
import numpy as np
from tqdm import tqdm
import ray
import os
from typing import Any

from oxrl.utils.setup import load_model_and_ref
from oxrl.algs.base import BaseAlgorithm

# Group Relative Policy Optimization — no value head, uses group-normalized
# rewards as advantages. Handles loss variants: sgrpo, gspo, cispo, rlhf, rlaif.
# Pro:  simpler than PPO (no critic), works well with outcome-based rewards.
# Con:  no per-token credit assignment; relies on reward normalization across
#       the sample group. Works best with n_samples >= 4 per prompt.
@ray.remote
class GRPO(BaseAlgorithm):
    def __init__(self,
                 model_path: str,
                 model_dtype: torch.dtype,
                 trust_remote_code: bool,
                 attn_impl: str,
                 kl_coeff: float,
                 clip_low: float,
                 clip_high: float,
                 entropy_coeff: float,
                 use_cache: bool,
                 micro_batch_size_per_gpu: int,
                 update_after_full_replay: bool,
                 deepspeed_config: Any,
                 lora_config = None,
                 ref_model_path: str = None,
                 deepspeed_ref_config: Any = None,
                 loss_variant: str = "sgrpo",
                 lr: float = 1e-5,
                 betas: list = None,
                 weight_decay: float = 0.01,
                 adam_epsilon: float = 1e-8,
                 model_class: str = "llm",
                 freeze_vision_encoder: bool = True,
                 ):

        self.loss_variant = loss_variant
        self.alg_name = loss_variant.upper()
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

        # if true, it means the update is done after seeing all samples in the reply buffer
        # treating the entire buffer as a single batch.
        self.update_only_after_full_replay = update_after_full_replay

        self.ready = False
        self.init_training_engine()
        self.ready = True

    def is_ready(self):
        '''
            Barrier method to ensure all Ray actors are initialized before DeepSpeed collective ops.
        '''
        return self.ready

    def init_training_engine(self):
        '''
            Since, we are using ray, each ray actor MUST create its own deepspeed engine.
            This is because each ray actor process is a separate process as it should be 1 actor = 1 gpu = 1 ds rank.
        '''
        from oxrl.utils.utils import import_deepspeed_safely
        deepspeed = import_deepspeed_safely()
        
        # Convert pydantic model to python Dict for DeepSpeed
        ds_config_dict = self.deepspeed_config.model_dump()

        # check to avoid re-initializing distributed backend
        if not torch.distributed.is_initialized():
            # 1. Initialize distributed training engine
            deepspeed.init_distributed()

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print(f"[Alg:{self.alg_name}][Rank {rank}] Initializing training engine...")

        # 2. Load model
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

        # Apply LoRA if enabled
        if self.lora_config and self.lora_config.enabled:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            
            # Note: prepare_model_for_kbit_training is useful even if not using kbit
            # as it enables gradient checkpointing etc. in a PEFT-friendly way.
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

        # 2. Initialize model engine
        
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
        print(f"[Alg:{self.alg_name}][Rank {rank}] DeepSpeed engine initialized on device: {self.policy_engine.device}")

        self.ref_model_engine = None
        if ref_model is not None:
            ref_model.eval()
            # Use inference-only config (no optimizer needed)
            ref_ds_config = self.deepspeed_ref_config.model_dump()
            self.ref_model_engine, _, _, _ = deepspeed.initialize(
                                                        model=ref_model,
                                                        config=ref_ds_config
                                                        )
            print(f"[Alg:{self.alg_name}][Rank {rank}] Reference model initialized with DeepSpeed")

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
                logits is [B, T-1, vocab_size]
        '''
        # feed data to model
        with torch.no_grad():
            if pos_ids is not None:
                pos_ids = pos_ids.to(input_ids.device)

            # Generate token_type_ids (zeros) as some models like Gemma 3 require them
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

            # cross_entropy return -logprobs but we need logprobs
            # logits is [B, T-1, vocab_size] --> [B * (T-1), vocab_size]
            # target_ids is [B, T-1] --> [B * (T-1)]
            neg_logprobs = self.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1))
            ref_logprobs = -neg_logprobs.view(B, T_minus_1)

        return ref_logprobs

    def policy_forward(self, input_ids, att_mask, pos_ids, mm_kwargs=None):
        '''
            input_ids and att_mask are [B, T]
            pos_ids is [B, T] or None
            mm_kwargs: optional dict of multimodal tensors (pixel_values, etc.)
            Returns:
                logits is [B, T-1, vocab_size]
                entropies is [B, T-1]
        '''
        # if pos_ids is not provided, HF will add that automatically.
        if pos_ids is not None:
            pos_ids = pos_ids.to(input_ids.device)

        # Generate token_type_ids (zeros)
        token_type_ids = torch.zeros_like(input_ids)

        # feed data to model
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

        # cross_entropy return -logprobs but we need logprobs
        # logits is [B, T-1, vocab_size] --> [B * (T-1), vocab_size]
        # target_ids is [B, T-1] --> [B * (T-1)]
        neg_logprobs = self.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1))
        logprobs = -neg_logprobs.view(B, T_minus_1)
        # we can also do this, but it is less efficient I guess
        #   logprobs = logits.log_softmax(dim=-1)
        #   logprobs = torch.gather(logprobs, dim=-1, index=target_ids)

        entropies = None
        if self.ent_coeff > 0.0:
            entropies = torch.distributions.Categorical(logits=logits).entropy()

        return logprobs, entropies, target_ids

    def compute_policy_loss(self,
                            logprobs: torch.Tensor,
                            old_logprobs: torch.Tensor,
                            advantages: torch.Tensor,
                            mask: torch.Tensor,
                            entropies: torch.Tensor,
                            ref_logprobs: torch.Tensor,
                            ):
        '''
            Dispatches to the appropriate loss function based on self.loss_variant.
            See oxrl/algs/losses/ for individual implementations (sgrpo, gspo, cispo).
        '''
        from oxrl.algs.losses import get_loss_fn
        loss_fn = get_loss_fn(self.loss_variant)
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

    def train_step(self, engine_id, micro_batches):
        '''
           This function implements a training step per rank/gpu for local_batch.
           The batch size for each gpu/rank should be micro_batch_size_per_gpu.
           micro_batches is a partition of the replay buffer (list of micro-batches) for the current rank/gpu.
        '''
        assert self.policy_engine is not None, "DeepSpeed engine not initialized"

        device = self.policy_engine.device

        # 1. Models to train mode
        self.policy_engine.train()

        # 2. zero grads
        self.policy_engine.zero_grad()

        # 3. create progress bar
        num_micro = len(micro_batches)
        # torch.distributed.get_rank() would be the same thing as engine_id
        if engine_id == 0:
            progress_bar = tqdm(micro_batches, total=num_micro, desc="[Alg:{}] Training Step in rank {}".format(self.alg_name, engine_id))

        else:
            progress_bar = micro_batches # No tqdm for other ranks

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
            # zscore is normalized rewards using the number of samples for each proompt (X -mu) / (std + eps)
            # this is a simple baseline for policy gradients (PPO in this code) as it reflects relative quality
            # among that prompt's samples.
            advs      = micro_batch['zscore'][:, :-1].to(device, non_blocking=True)
            #done      = micro_batch['done'][:, :-1].to(device, non_blocking=True)
            mask      = micro_batch['mask'][:, :-1].to(device, non_blocking=True)
            old_logprobs = micro_batch['old_logprobs'][:, :-1].to(device, non_blocking=True)

            input_ids = micro_batch['input_ids'].to(device, non_blocking=True)
            att_mask  = micro_batch['attn_mask'].to(device, non_blocking=True)
            pos_ids   = micro_batch.get('position_ids', None)

            # Prepare multimodal inputs if present
            mm_kwargs = {}
            if self.model_class == "vlm" and micro_batch.get("multi_modal_data") is not None:
                mm_kwargs = self._prepare_mm_inputs(micro_batch["multi_modal_data"], device)

            ########
            # 2. Compute loss
            ########
            # Forward pass through the current policy.
            pi_logprobs, pi_entropies, target_ids = self.policy_forward(input_ids=input_ids,
                                                                        att_mask=att_mask,
                                                                        pos_ids=pos_ids,
                                                                        mm_kwargs=mm_kwargs)

            ref_logprobs = None
            if self.kl_coeff > 0.0 and self.ref_model_engine is not None:
                ref_logprobs = self.ref_forward(input_ids=input_ids,
                                                att_mask=att_mask,
                                                target_ids=target_ids,
                                                pos_ids=pos_ids,
                                                mm_kwargs=mm_kwargs,
                                                )

            # Compute policy loss using the current policy.
            pi_loss, pi_metrics = self.compute_policy_loss(logprobs=pi_logprobs,
                                                           old_logprobs=old_logprobs,
                                                           advantages=advs,
                                                           mask=mask,
                                                           entropies=pi_entropies,
                                                           ref_logprobs=ref_logprobs)

            # store metrics
            all_metrics.append(pi_metrics)
            if engine_id == 0:
                progress_bar.set_postfix({
                    "loss": f"{pi_loss.item():.4f}",
                    "clip": f"{pi_metrics['clipfrac']:.3f}",
                    "kl_old": f"{pi_metrics['kl_old']:.4f}",
                    "kl_ref": f"{pi_metrics['kl_ref']:.4f}"
                })

            if self.update_only_after_full_replay:
                # Accumulate gradients across all micro-batches, only update at the end
                self.policy_engine.set_gradient_accumulation_boundary(is_boundary)
            else:
                # Update after every micro-batch (treat each as a boundary)
                self.policy_engine.set_gradient_accumulation_boundary(True)

            # backward pass
            self.policy_engine.backward(pi_loss)
            self.policy_engine.step()

        # aggregate metrics across all micro-batches
        aggregated_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                aggregated_metrics[key] = np.mean([m[key] for m in all_metrics])

        return aggregated_metrics

    def gather_state_dict(self):
        """Gather ZeRO-3 partitioned weights into a single state dict in memory.

        Collective operation — must be called on ALL ranks.
        Returns the state dict on rank 0, None on other ranks.
        """
        from oxrl.tools.lora_merge import strip_lora_and_merge
        from oxrl.tools.checkpoint import get_base_model_config

        rank = torch.distributed.get_rank()
        print(f"[Alg:{self.alg_name}][Rank {rank}] Gathering state dict in memory...")

        # All ranks must participate in this collective operation
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

            print(f"[Alg:{self.alg_name}][Rank {rank}] State dict gathered: {len(state_dict)} keys")
            return state_dict
        else:
            return None

    def save_checkpoint(self, output_dir: str, tag: str, state_dict_ref=None):
        '''
            Saves the model in hf compatible format for vllm, etc.

            If state_dict_ref is provided, rank 0 retrieves the pre-gathered state
            dict from the Ray object store and writes it to disk (no ZeRO-3 gather).
            If state_dict_ref is None, falls back to save_16bit_model (original behavior).

            Note: when state_dict_ref is None, must call on ALL ranks for zero-3 correctness.
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

                    save_state_dict_to_safetensors(output_dir, state_dict)
                    print(f"[Alg:{self.alg_name}][Rank {rank}] Weights saved via object store")

                    if config_dict is not None:
                        save_config_json(output_dir, config_dict)
                        print(f"[Alg:{self.alg_name}][Rank {rank}] Config saved")

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

                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

            print(f"[Alg:{self.alg_name}][Rank {rank}] Checkpoint save completed!")

        except Exception as e:
            print(f"[Alg:{self.alg_name}][Rank {rank}] Error saving checkpoint to {output_dir}: {e}")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            raise
