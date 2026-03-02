"""
DeepSpeed config synchronization logic.

Reads values from Config.train / Config.model and writes them into
Config.deepspeed / Config.deepspeed_ref so the two stay consistent.

Pure functions — no I/O, no YAML parsing.
"""
from oxrl.configs.schema import Config, DeepSpeedRef


def sync_deepspeed_config(config: Config, world_size: int) -> None:
    """Sync DeepSpeed config from train/model settings (mutates config in-place)."""
    _sync_batch_sizes(config, world_size)
    _sync_gradient_clipping(config)
    _sync_dtype(config)
    _sync_optimizer(config)
    _sync_scheduler(config)
    _sync_zero_defaults(config)
    _sync_ref_model_config(config)


def _sync_batch_sizes(config: Config, world_size: int) -> None:
    """1 - Batch Sizes (required for both SL and RL)."""
    config.deepspeed.train_micro_batch_size_per_gpu = config.train.train_batch_size_per_gpu
    config.deepspeed.gradient_accumulation_steps = config.train.gradient_accumulation_steps

    if world_size is not None and config.run.method == "sl":
        config.deepspeed.train_batch_size = (
            config.train.train_batch_size_per_gpu
            * config.train.gradient_accumulation_steps
            * world_size
        )


def _sync_gradient_clipping(config: Config) -> None:
    """2 - Gradient Clipping."""
    config.deepspeed.gradient_clipping = float(config.train.clip_grad_norm)


def _sync_dtype(config: Config) -> None:
    """3 - FP16 / BF16."""
    dtype = config.model.dtype.lower()
    if dtype in ("float16", "fp16"):
        config.deepspeed.fp16["enabled"] = True
        config.deepspeed.bf16["enabled"] = False
    elif dtype in ("bfloat16", "bf16"):
        config.deepspeed.fp16["enabled"] = False
        config.deepspeed.bf16["enabled"] = True
    else:
        config.deepspeed.fp16["enabled"] = False
        config.deepspeed.bf16["enabled"] = False


def _sync_optimizer(config: Config) -> None:
    """4 - Optimizer (Auto-Sync)."""
    if "adamw" in config.train.optimizer_name.lower():
        ds_opt_type = "AdamW"
    elif "adam" in config.train.optimizer_name.lower():
        ds_opt_type = "Adam"
    else:
        raise ValueError(f"Unsupported optimizer: {config.train.optimizer_name}")

    config.deepspeed.optimizer = {
        "type": ds_opt_type,
        "params": {
            "lr": config.train.lr,
            "betas": config.train.betas,
            "weight_decay": config.train.weight_decay,
            "eps": config.train.adam_epsilon,
        },
    }


def _sync_scheduler(config: Config) -> None:
    """5 - Scheduler (Auto-Sync)."""
    if config.train.lr_scheduler == "WarmupCosineLR":
        if config.run.method == "sl":
            if config.train.micro_batches_per_epoch is None:
                raise ValueError("micro_batches_per_epoch must be set for SL training")
            optimizer_steps_per_epoch = (
                config.train.micro_batches_per_epoch // config.train.gradient_accumulation_steps
            )
        else:
            if config.train.train_steps_per_epoch is None:
                raise ValueError("train_steps_per_epoch must be set for RL training")
            optimizer_steps_per_epoch = config.train.train_steps_per_epoch

        total_optimizer_steps = config.train.total_number_of_epochs * optimizer_steps_per_epoch
        warmup_steps = int(total_optimizer_steps * config.train.warmup_steps_ratio)

        config.deepspeed.scheduler = {
            "type": config.train.lr_scheduler,
            "params": {
                "total_num_steps": total_optimizer_steps,
                "warmup_min_ratio": 0.0,
                "cos_min_ratio": 0.1,
                "warmup_num_steps": warmup_steps,
            },
        }
    else:
        raise ValueError(f"Unsupported scheduler: {config.train.lr_scheduler}")


def _sync_zero_defaults(config: Config) -> None:
    """6 - ZeRO Defaults (Ensure robust ZeRO-3 settings)."""
    if config.deepspeed.zero_optimization is None:
        config.deepspeed.zero_optimization = {}

    keys_to_remove = []
    for k, v in config.deepspeed.zero_optimization.items():
        if v is None:
            keys_to_remove.append(k)
        elif isinstance(v, dict) and v.get("device") == "none":
            keys_to_remove.append(k)

    for k in keys_to_remove:
        del config.deepspeed.zero_optimization[k]

    if config.deepspeed.zero_optimization.get("stage") == 3:
        if "stage3_gather_16bit_weights_on_model_save" not in config.deepspeed.zero_optimization:
            config.deepspeed.zero_optimization["stage3_gather_16bit_weights_on_model_save"] = True


def _sync_ref_model_config(config: Config) -> None:
    """7 - Generate ref model config (inference-only, no optimizer/updates)."""
    if config.deepspeed_ref is None and config.model.ref_model:
        ds_dict = config.deepspeed.model_dump()

        ds_dict.pop("optimizer", None)
        ds_dict.pop("scheduler", None)

        if ds_dict.get("zero_optimization"):
            ds_dict["zero_optimization"].pop("offload_optimizer", None)

            if config.model.ref_model_offload_to_cpu:
                ds_dict["zero_optimization"]["offload_param"] = {
                    "device": "cpu",
                    "pin_memory": True,
                }
            else:
                ds_dict["zero_optimization"].pop("offload_param", None)

        config.deepspeed_ref = DeepSpeedRef(
            fp16=ds_dict.get("fp16", {"enabled": False}),
            bf16=ds_dict.get("bf16", {"enabled": False}),
            zero_optimization=ds_dict.get("zero_optimization", {}),
            train_micro_batch_size_per_gpu=ds_dict.get("train_micro_batch_size_per_gpu"),
            activation_checkpointing=ds_dict.get("activation_checkpointing"),
        )
