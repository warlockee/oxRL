"""
Pydantic configuration schemas for oxRL.

Every config section is a standalone Pydantic model with extra='forbid'
to catch typos. Config is the root model that composes all sections.

No logic, no I/O — just shapes and constraints.
"""
from typing import Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class Run(BaseModel):
    '''
      This contains general experiment settings.
    '''
    model_config = ConfigDict(extra='forbid')
    experiment_id: str
    distributed_training_strategy: str = "deepspeed-zero3"
    seed: int = 42
    project_name: str = "oxrl-exp"
    tracking_uri: str = ""
    method: str = None

    # RL-specific fields
    training_gpus: int | None = None
    rollout_gpus: int | None = None
    ray_address: str | None = None
    ray_master_port: int = 29500
    checkpoint_dir: str | None = None

class Train(BaseModel):
    '''
        Everything related to training goes here like optimizer, scheduler, etc.
    '''
    model_config = ConfigDict(extra='forbid')
    ###############
    # optimizer related arguments
    ###############
    optimizer_name: str = "adamw"
    # sgrpo:  token-level clipped surrogate. Default for dense models.
    # gspo:   sequence-level clipped surrogate. Use for MoE models.
    # cispo:  clipped ratio weights log-prob. More conservative than sgrpo.
    # ppo:    full PPO with value head. Higher cost, finer credit assignment.
    # rlhf/rlaif: aliases for sgrpo (readability only).
    alg_name: str
    lr: float = Field(default=1e-5, gt=0)
    adam_epsilon: float = 1e-8
    betas: list[float] = Field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.01
    warmup_steps_ratio: float = 0.1
    clip_grad_norm: float = 1.0
    lr_scheduler: str = "WarmupCosineLR"

    # RL-specific policy arguments
    kl_coeff: float | None = 0.0
    clip_low: float | None = 0.2
    clip_high: float | None = 0.2
    entropy_coeff: float | None = 0.0
    update_after_full_replay: bool | None = True

    ###############
    # general training  loop arguments
    ###############
    # Here, an "epoch" is defined by a fixed number of training steps, not a full sweep of the dataset.
    # Each epoch processes: train_steps_per_epoch * global_batch_size samples.
    # We define epochs this way to control how different datasets are mixed during training and
    # to have more control over the training process.
    total_number_of_epochs: int

    # RL: train_steps_per_epoch = number of optimizer steps per epoch.
    # Setting this to 1 gives strict on-policy training: the policy is
    # resampled before every gradient update, preventing any off-policy drift.
    # Higher values (e.g., 5) reuse the same rollout data for multiple updates,
    # which is more compute-efficient but introduces mild off-policy staleness.
    train_steps_per_epoch: int | None = None

    # SL: micro_batches_per_epoch = number of micro-batch iterations per epoch
    # Optimizer steps = micro_batches_per_epoch // gradient_accumulation_steps
    micro_batches_per_epoch: int | None = None

    # Preference optimization arguments
    beta: float = 0.1
    # Arguments which are common to both deepspeed and standalone training.
    ###############
    # Some of the below arguments also can be set in deepspeed config. However to avoid any confusion and increase code readability,
    # we are setting them here and update deepspeed config accordingly.
    # Note: train_batch_size_per_gpu is same as train_micro_batch_size_per_gpu
    # global batch_size would be train_batch_size_per_gpu * gradient_accumulation_steps * number_of_gpus.
    train_batch_size_per_gpu: int = 2
    gradient_accumulation_steps: int = 1
    val_batch_size_per_gpu: int = 16

    normalize_loss: bool = True

    # Knowledge Distillation
    kd_alpha: float = 0.5
    kd_temperature: float = 2.0

    # Rejection Sampling Fine-Tuning
    reward_threshold: float = 0.5

    # RLHF
    reward_model_path: str = ""

    # SimPO
    simpo_gamma: float = 0.5

    # PPO
    ppo_vf_clip: float = 0.2
    ppo_tau: float = 0.95
    ppo_gamma: float = 0.99

class Data(BaseModel):
    '''
        Everything related to data goes here.
    '''
    model_config = ConfigDict(extra='forbid')
    train_dnames: list[str]
    train_ratios: dict[str, float]
    train_files_path: str
    val_files_path: str
    num_workers: int = 4
    max_seq_len: int = 512
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    chosen_key: str = "chosen"
    rejected_key: str = "rejected"

class Model(BaseModel):
    '''
        Information like model_name, ref_model_path, dtype, etc.
    '''
    model_config = ConfigDict(extra='forbid')
    name: str
    dtype: str = "bfloat16"
    ref_model: str = ""
    ref_model_offload_to_cpu: bool = True
    trust_remote_code: bool = False
    use_cache: bool = False
    model_class: str = "llm"
    attn_implementation: str = "flash_attention_2"
    gradient_checkpointing: bool = True

class DeepSpeed(BaseModel):
    '''
        Everything related to DeepSpeed goes here.
    '''
    model_config = ConfigDict(extra='forbid')
    train_batch_size: int | None = None  # Calculated automatically
    train_micro_batch_size_per_gpu: int | None = None
    gradient_accumulation_steps: int | None = None
    gradient_clipping: float | None = None

    # Optimizer/Scheduler are usually dicts in DS config
    optimizer: Dict[str, Any] | None = None
    scheduler: Dict[str, Any] | None = None

    fp16: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    bf16: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})

    # ZeRO Optimization
    zero_optimization: Dict[str, Any] = Field(default_factory=lambda: {
        "stage": 3,
        "stage3_param_persistence_threshold": 1e5,
        "stage3_prefetch_bucket_size": 5e7,
        "contiguous_gradients": True,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8,
        "stage3_gather_16bit_weights_on_model_save": True,
    })

    # Activation Checkpointing
    activation_checkpointing: Dict[str, Any] = Field(default_factory=lambda: {
        "partition_activations": True,
        "contiguous_memory_optimization": True,
    })

    # Logging
    steps_per_print: int = 100
    wall_clock_breakdown: bool = False

    # Flops profiler
    flops_profiler: Dict[str, Any] | None = None

    # Monitor config
    monitor_config: Dict[str, Any] | None = None

class DeepSpeedRef(BaseModel):
    '''
        Inference-only deepspeed for ref model in rl(no optimizer, no updates).
    '''
    model_config = ConfigDict(extra='forbid')
    fp16: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    bf16: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    zero_optimization: Dict[str, Any] = Field(default_factory=dict)
    train_micro_batch_size_per_gpu: int | None = None
    # Activation checkpointing (can help with memory)
    activation_checkpointing: Dict[str, Any] | None = None

class InferenceEngine(BaseModel):
    '''
        Everything related to inference goes here.
    '''
    model_config = ConfigDict(extra='forbid')
    name: str = "vllm"

class Lora(BaseModel):
    '''
        LoRA (Low-Rank Adaptation) settings.
    '''
    model_config = ConfigDict(extra='forbid')
    enabled: bool = False
    r: int = 16
    lora_alpha: int = 32
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

class Reward(BaseModel):
    '''
        Everything related to rewards (RL-specific).
    '''
    model_config = ConfigDict(extra='forbid')
    broadcast: bool = False
    eps_reward_norm: float = 1e-8
    # default_reward_func:    binary EOS check (sanity / external reward)
    # gsm8k_reward_func:      binary numeric match (GSM8K math)
    # math_reward_func:       binary \boxed{} match (MATH dataset)
    # soft_math_reward_func:  graduated 1.0/0.5/0.2 (partial credit math)
    # code_reward_func:       binary test execution (MBPP code-gen)
    # format_reward_func:     0-1.0 style checklist (instruction-following)
    # mcqa_reward_func:       binary letter match (MMLU-Pro QA)
    # reasoning_reward_func:  0-1.0 tags + correctness (R1-style CoT)
    # multimodal_reward_func: 0-1.0 correctness + modality (vision/audio)
    # rm_reward_func:         continuous via trained reward model (RLHF)
    reward_func: str = "default_reward_func"

class Rollout(BaseModel):
    '''
        Everything related to rollout generation (RL-specific).
    '''
    model_config = ConfigDict(extra='forbid')
    temperature: float = 1.0
    max_tokens: int = 512
    n_samples: int = 8
    top_p: float = 1.0
    top_k: int = -1
    ignore_eos: bool = False
    stop: str = ""
    gpu_memory_utilization: float = 0.5
    stop_token_ids: list[int] = Field(default_factory=list)
    prompt_logprobs: bool = False
    # Enforces unbiased sampling: temperature=1.0, top_p=1.0, top_k=-1, no stop
    # tokens. This ensures the sampling distribution matches the true policy.
    # Note: this controls *sampling purity*, not how many gradient steps are taken
    # per rollout batch. For strict on-policy *training* (resample before every
    # gradient step), set train_steps_per_epoch=1 in the Train config.
    force_strict_on_policy: bool = True
    tensor_parallel_size: int = 1
    rollout_batch_size_per_gpu: int = 2

class Config(BaseModel):
    '''
        This is the main configuration class for the experiment where it puts all the sub-configurations
        together to form a complete configuration for the experiment.
    '''
    model_config = ConfigDict(extra='forbid')
    run: Run
    train: Train
    model: Model
    data: Data
    deepspeed: DeepSpeed = Field(default_factory=DeepSpeed)
    lora: Lora = Field(default_factory=Lora)
    inference_engine: InferenceEngine = Field(default_factory=InferenceEngine)
    # RL-specific sections
    reward: Reward = Field(default_factory=Reward)
    rollout: Rollout = Field(default_factory=Rollout)
    # Reference model DeepSpeed config
    deepspeed_ref: DeepSpeedRef | None = None

    def sync_deepspeed_config(self, world_size: int) -> None:
        """Thin wrapper that delegates to oxrl.configs.sync (backward compat)."""
        from oxrl.configs.sync import sync_deepspeed_config
        sync_deepspeed_config(self, world_size)
