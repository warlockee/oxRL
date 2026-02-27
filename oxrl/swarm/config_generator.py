"""
Auto-generate oxRL YAML config files from model metadata.

Given a model name (e.g. "Qwen/Qwen2.5-0.5B-Instruct"), a task type, and the
approximate parameter count, this module produces a complete config dict that
passes the Pydantic ``Config(extra='forbid')`` validation defined in
``configs/load.py``.
"""

import uuid
import yaml
from typing import Optional


# ---------------------------------------------------------------------------
# Task -> (reward_func, dataset slug) mapping
# ---------------------------------------------------------------------------
TASK_MAP = {
    "math":      {"reward_func": "gsm8k_reward_func",  "dataset": "gsm8k"},
    "math-hard": {"reward_func": "math_reward_func",   "dataset": "math_hard"},
    "code":      {"reward_func": "code_reward_func",    "dataset": "mbpp"},
    "instruct":  {"reward_func": "format_reward_func",  "dataset": "ultrafeedback"},
    "reasoning": {"reward_func": "reasoning_reward_func", "dataset": "openr1_math"},
    "vision":    {"reward_func": "multimodal_reward_func",   "dataset": "vision_dummy"},
    "audio":     {"reward_func": "multimodal_reward_func",   "dataset": "audio_dummy"},
    "openr1-math": {"reward_func": "reasoning_reward_func", "dataset": "openr1_math"},
}

# Task -> (max_seq_len, max_tokens)
TASK_SEQ = {
    "math":      {"max_seq_len": 512, "max_tokens": 512},
    "math-hard": {"max_seq_len": 512, "max_tokens": 512},
    "code":      {"max_seq_len": 1024, "max_tokens": 512},
    "instruct":  {"max_seq_len": 1024, "max_tokens": 512},
    "reasoning": {"max_seq_len": 1024, "max_tokens": 1024},
    "vision":    {"max_seq_len": 1024, "max_tokens": 512},
    "audio":     {"max_seq_len": 1024, "max_tokens": 512},
    "openr1-math": {"max_seq_len": 1024, "max_tokens": 1024},
}


def make_slug(model_name: str) -> str:
    """Convert 'Qwen/Qwen2.5-0.5B-Instruct' -> 'qwen2.5-0.5b-instruct'."""
    # Take the part after the last '/' (the actual model name), then lowercase.
    return model_name.split("/")[-1].lower()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _batch_params(param_count_b: float) -> dict:
    """Return batch-size related params scaled inversely with model size.

    Rollout batch sizes are set large enough that the full dataset can be
    processed in a reasonable time (targeting <5 min for rollout generation
    per epoch on an A100-40GB).
    """
    if param_count_b <= 1.0:
        return {
            "train_batch_size_per_gpu": 2,
            "rollout_batch_size_per_gpu": 64,
            "n_samples": 4,
            "gradient_accumulation_steps": 8,
        }
    elif param_count_b <= 2.0:
        return {
            "train_batch_size_per_gpu": 1,
            "rollout_batch_size_per_gpu": 32,
            "n_samples": 4,
            "gradient_accumulation_steps": 8,
        }
    elif param_count_b <= 4.0:
        return {
            "train_batch_size_per_gpu": 1,
            "rollout_batch_size_per_gpu": 16,
            "n_samples": 4,
            "gradient_accumulation_steps": 8,
        }
    elif param_count_b <= 7.0:
        return {
            "train_batch_size_per_gpu": 1,
            "rollout_batch_size_per_gpu": 8,
            "n_samples": 2,
            "gradient_accumulation_steps": 4,
        }
    else:
        return {
            "train_batch_size_per_gpu": 1,
            "rollout_batch_size_per_gpu": 4,
            "n_samples": 2,
            "gradient_accumulation_steps": 4,
        }


def _deepspeed_offload(param_count_b: float, use_lora: bool = False) -> dict:
    """Return offload_optimizer and offload_param device settings."""
    if use_lora or param_count_b <= 7.0:
        return {
            "offload_optimizer": {"device": "none", "pin_memory": True},
            "offload_param":     {"device": "none", "pin_memory": True},
        }
    else:
        return {
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "offload_param":     {"device": "cpu", "pin_memory": True},
        }


def _gpu_memory_utilization(param_count_b: float) -> float:
    if param_count_b <= 3.0:
        return 0.4
    elif param_count_b <= 7.0:
        return 0.6
    else:
        return 0.85


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_config(
    model_name: str,
    task: str,
    param_count_b: float,
    data_dir: str = "./data",
    checkpoint_dir: str = "./checkpoints",
    experiment_id: Optional[str] = None,
) -> dict:
    """
    Auto-generate a complete oxRL config dict.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. ``"Qwen/Qwen2.5-0.5B-Instruct"``.
    task : str
        One of ``"math"``, ``"math-hard"``, ``"code"``, ``"instruct"``,
        ``"reasoning"``.
    param_count_b : float
        Approximate parameter count in *billions* (e.g. 0.5, 1.5, 7.0).
    data_dir : str
        Root directory where preprocessed parquet files live.
    checkpoint_dir : str
        Root directory for saving training checkpoints.
    experiment_id : str, optional
        Human-readable experiment identifier.  Auto-generated if not provided.

    Returns
    -------
    dict
        A config dictionary whose structure matches the Pydantic ``Config``
        class in ``configs/load.py`` (with ``extra='forbid'``).
    """
    if task not in TASK_MAP:
        raise ValueError(
            f"Unknown task {task!r}. Choose from: {list(TASK_MAP.keys())}"
        )

    model_slug = make_slug(model_name)
    task_info = TASK_MAP[task]
    seq_info = TASK_SEQ[task]
    dataset = task_info["dataset"]

    if experiment_id is None:
        experiment_id = f"{model_slug}_{dataset}_{uuid.uuid4().hex[:6]}"

    # Enable LoRA for 7B+ models to fit in memory
    _use_lora = True if param_count_b >= 7.0 else False

    batch = _batch_params(param_count_b)
    offload = _deepspeed_offload(param_count_b, use_lora=_use_lora)
    gpu_mem = _gpu_memory_utilization(param_count_b)

    # Build data name following existing convention:
    #   gsm8k_qwen05b_wsp_train  (dataset + model_slug + _wsp_train)
    data_name = f"{dataset}_{model_slug}_wsp_train"

    # For onboarding runs, use a smaller subset (500 rows) for speed.
    # Create the subset file if it doesn't exist.
    import os
    _MAX_ONBOARDING_ROWS = 100
    full_train_path = f"{data_dir}/{dataset}_{model_slug}_wsp_train.parquet"
    subset_train_path = f"{data_dir}/{dataset}_{model_slug}_wsp_train_onboard.parquet"
    if os.path.exists(full_train_path):
        if not os.path.exists(subset_train_path):
            try:
                import pandas as pd
                df = pd.read_parquet(full_train_path)
                if len(df) > _MAX_ONBOARDING_ROWS:
                    df = df.sample(n=_MAX_ONBOARDING_ROWS, random_state=42)
                df.to_parquet(subset_train_path)
            except Exception:
                subset_train_path = full_train_path
        train_data_path = subset_train_path
    else:
        train_data_path = full_train_path

    # Use 2+2 GPU layout for 7B+ models to avoid timeouts
    _training_gpus = 2 if param_count_b >= 6.5 else 1
    _rollout_gpus = 2 if param_count_b >= 6.5 else 1

    config = {
        # ------------------------------------------------------------------
        # run
        # ------------------------------------------------------------------
        "run": {
            "experiment_id": experiment_id,
            "training_gpus": _training_gpus,
            "rollout_gpus": _rollout_gpus,
            "checkpoint_dir": f"{checkpoint_dir}/{experiment_id}",
            "tracking_uri": "",
            "project_name": "oxrl-exp",
            "ray_address": None,
            "ray_master_port": 29600,
            "distributed_training_strategy": "deepspeed-zero3",
            "seed": 42,
        },

        # ------------------------------------------------------------------
        # lora
        # ------------------------------------------------------------------
        "lora": {
            "enabled": _use_lora,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },

        # ------------------------------------------------------------------
        # train
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # train
        # ------------------------------------------------------------------
        "train": {
            "alg_name": "sgrpo",
            "optimizer_name": "adamw",
            "lr": 1e-6,
            "adam_epsilon": 1e-8,
            "betas": [0.9, 0.95],
            "weight_decay": 0.01,
            "warmup_steps_ratio": 0.1,
            "clip_grad_norm": 1.0,
            "lr_scheduler": "WarmupCosineLR",
            "kl_coeff": 0.0,
            "clip_low": -0.2,
            "clip_high": 0.2,
            "entropy_coeff": 0.0,
            "update_after_full_replay": True,
            "total_number_of_epochs": 1,
            "train_steps_per_epoch": 1,
            "dynamic_ratio_every_step": True,
            "train_batch_size_per_gpu": batch["train_batch_size_per_gpu"],
            "gradient_accumulation_steps": batch["gradient_accumulation_steps"],
            "val_batch_size_per_gpu": 16,
            "normalize_loss": True,
        },

        # ------------------------------------------------------------------
        # model
        # ------------------------------------------------------------------
        "model": {
            "name": model_name,
            "dtype": "bfloat16",
            "ref_model": "",
            "ref_model_offload_to_cpu": True,
            "trust_remote_code": True,
            "use_cache": False,
            "model_class": "llm",
            "gradient_checkpointing": True,
            "attn_implementation": "eager",
        },

        # ------------------------------------------------------------------
        # data
        # ------------------------------------------------------------------
        "data": {
            "train_dnames": [data_name],
            "train_ratios": {data_name: 1.0},
            "train_files_path": train_data_path,
            "val_files_path": f"{data_dir}/{dataset}_{model_slug}_wsp_test.parquet",
            "num_workers": 4,
            "max_seq_len": seq_info["max_seq_len"],
            "prompt_key": "prompt",
            "answer_key": "answer",
        },

        # ------------------------------------------------------------------
        # reward
        # ------------------------------------------------------------------
        "reward": {
            "broadcast": False,
            "eps_reward_norm": 1e-8,
            "reward_func": task_info["reward_func"],
        },

        # ------------------------------------------------------------------
        # rollout
        # ------------------------------------------------------------------
        "rollout": {
            "temperature": 1.0,
            "max_tokens": seq_info["max_tokens"],
            "n_samples": batch["n_samples"],
            "top_p": 1.0,
            "top_k": -1,
            "ignore_eos": False,
            "stop": "",
            "gpu_memory_utilization": gpu_mem,
            "stop_token_ids": [],
            "prompt_logprobs": False,
            "force_strict_on_policy": True,
            "tensor_parallel_size": 1,
            "rollout_batch_size_per_gpu": batch["rollout_batch_size_per_gpu"],
        },

        # ------------------------------------------------------------------
        # deepspeed
        # ------------------------------------------------------------------
        "deepspeed": {
            "zero_optimization": {
                "stage": 3,
                "stage3_param_persistence_threshold": 1e5,
                "stage3_prefetch_bucket_size": 5e7,
                "offload_optimizer": offload["offload_optimizer"],
                "offload_param": offload["offload_param"],
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "allgather_bucket_size": 5e8,
                "stage3_gather_16bit_weights_on_model_save": True,
            },
            "activation_checkpointing": {
                "partition_activations": True,
                "contiguous_memory_optimization": True,
            },
            "steps_per_print": 100,
            "wall_clock_breakdown": False,
            "flops_profiler": {
                "enabled": False,
                "profile_step": 10,
                "module_depth": -1,
                "top_modules": 1,
                "detailed": True,
                "output_file": None,
            },
        },

        # ------------------------------------------------------------------
        # inference_engine
        # ------------------------------------------------------------------
        "inference_engine": {
            "name": "vllm",
        },
    }

    return config


def save_config(config: dict, output_path: str) -> str:
    """Save *config* as a YAML file and return *output_path*."""
    with open(output_path, "w") as fh:
        yaml.dump(config, fh, default_flow_style=False, sort_keys=False)
    return output_path


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = generate_config(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        task="math",
        param_count_b=0.5,
        experiment_id="qwen05b_gsm8k_demo",
    )

    print(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
