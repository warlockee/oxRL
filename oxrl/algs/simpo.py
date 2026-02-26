import torch
import torch.nn.functional as F
import os
import glob
from tqdm import tqdm
import ray
import deepspeed
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Any, Dict, List, Optional, Tuple

@ray.remote
class SimPO:
    def __init__(self,
                 model_path: str,
                 model_dtype: torch.dtype,
                 trust_remote_code: bool,
                 attn_impl: str,
                 beta: float,
                 gamma: float,
                 use_cache: bool,
                 micro_batch_size_per_gpu: int,
                 deepspeed_config: deepspeed.DeepSpeedConfig,
                 lora_config = None,
                 ):

        self.alg_name = "SimPO"
        # model related parameters
        self.model_path = model_path
        self.use_cache = use_cache
        self.attn_impl = attn_impl
        self.model_dtype = model_dtype
        self.trust_remote_code = trust_remote_code
        self.lora_config = lora_config

        # training related parameters
        self.deepspeed_config = deepspeed_config
        self.micro_batch_size_per_gpu = micro_batch_size_per_gpu

        # SimPO specific parameters
        self.beta = float(beta)
        self.gamma = float(gamma)

        self.ready = False
        self.init_training_engine()
        self.ready = True

    def is_ready(self):
        return self.ready

    def init_training_engine(self):
        ds_config_dict = self.deepspeed_config.model_dump()
        if not torch.distributed.is_initialized():
            deepspeed.init_distributed()

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        model = self.load_model()

        if self.lora_config and self.lora_config.enabled:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            if getattr(model, "is_loaded_in_4bit", False):
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

        if "optimizer" in ds_config_dict:
            del ds_config_dict["optimizer"]
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-6)

        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=trainable_params,
            config=ds_config_dict,
            optimizer=optimizer
        )

    def load_model(self):
        return AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=self.model_dtype,
            trust_remote_code=self.trust_remote_code,
            attn_implementation=None if self.attn_impl == '' else self.attn_impl
        )

    def get_logps(self, logits, labels):
        # logits: [B, T, V], labels: [B, T]
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(2)).squeeze(2)
        
        # Mask out padding tokens (labels = -100 or 0 depending on setup)
        mask = (shift_labels != -100)
        return (per_token_logps * mask).sum(-1) / mask.sum(-1) # Length-normalized

    def train_step(self, chosen_input_ids, chosen_attention_mask, rejected_input_ids, rejected_attention_mask):
        # chosen_input_ids: [B, T], rejected_input_ids: [B, T]
        
        # Forward pass for chosen
        outputs_c = self.model_engine(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask)
        logps_c = self.get_logps(outputs_c.logits, chosen_input_ids)
        
        # Forward pass for rejected
        outputs_r = self.model_engine(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask)
        logps_r = self.get_logps(outputs_r.logits, rejected_input_ids)
        
        # SimPO Loss: -log(sigmoid(beta * (logps_c - logps_r) - gamma))
        logits = self.beta * (logps_c - logps_r) - self.gamma
        loss = -F.logsigmoid(logits).mean()
        
        self.model_engine.backward(loss)
        self.model_engine.step()
        
        return {"loss": loss.item(), "margin": (logps_c - logps_r).mean().item()}

    def save_checkpoint(self, output_dir: str, tag: str):
        rank = torch.distributed.get_rank()
        self.model_engine.save_16bit_model(output_dir)
        if rank == 0:
            self.model_engine.module.config.save_pretrained(output_dir)
