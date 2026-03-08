"""
Factory functions for creating Ray-based training and rollout engine actors.
"""
import os
from oxrl.utils.utils import safe_string_to_torch_dtype
from oxrl.rollouts.vllm_engine import VLLMRolloutEngine
from oxrl.algs.grpo import GRPO
from oxrl.algs.ppo import PPO

# sgrpo:  token-level clipped surrogate — default for dense models
# gspo:   sequence-level clipped surrogate — use for MoE models
# cispo:  clipped ratio as weight on log-prob — more conservative updates
# ppo:    full PPO with value head + GAE
# rlhf/rlaif: sgrpo aliases for readability
RL_ALGORITHMS = {
    "sgrpo": GRPO,
    "cispo": GRPO,
    "gspo": GRPO,
    "rlhf": GRPO,
    "rlaif": GRPO,
    "ppo": PPO,
}


def get_algorithm_class(alg_name: str):
    """Look up algorithm class by name.

    Checks the built-in registry first, then falls back to any custom
    algorithms registered via :func:`oxrl.models.research_adapters.register_algorithm`.
    ``@ray.remote`` is applied automatically if the custom class lacks it.
    Raises ValueError if the name is not found in either registry.
    """
    key = alg_name.lower()
    if key in RL_ALGORITHMS:
        return RL_ALGORITHMS[key]
    from oxrl.models.research_adapters import get_custom_algorithm
    custom = get_custom_algorithm(key)
    if custom is not None:
        if not hasattr(custom, '_ray_class_method_names'):
            import ray
            custom = ray.remote(custom)
        return custom
    raise ValueError(f"Unknown algorithm: {alg_name}. Available: {list(RL_ALGORITHMS)}")


def training_engine_setup(params, alg, world_size, master_addr, master_port):
    """Create Ray training engine actors (one per GPU)."""
    kwargs = {
        # model related arguments
        "model_path": params.model.name,
        "ref_model_path": params.model.ref_model,
        "model_dtype": safe_string_to_torch_dtype(params.model.dtype),
        "trust_remote_code": params.model.trust_remote_code,
        "attn_impl": params.model.attn_implementation,
        "use_cache": params.model.use_cache,
        # training related arguments
        "kl_coeff": params.train.kl_coeff,
        "clip_low": params.train.clip_low,
        "clip_high": params.train.clip_high,
        "entropy_coeff": params.train.entropy_coeff,
        "micro_batch_size_per_gpu": params.train.train_batch_size_per_gpu,
        "update_after_full_replay": params.train.update_after_full_replay,
        # deepspeed related arguments
        "deepspeed_config": params.deepspeed,
        "deepspeed_ref_config": params.deepspeed_ref,
        "lora_config": params.lora,
        # multimodal / VLM arguments
        "model_class": params.model.model_class,
        "freeze_vision_encoder": params.model.freeze_vision_encoder,
        # algorithm related arguments
        "loss_variant": (params.train.loss_variant or params.train.alg_name).lower(),
        # optimizer hyperparameters
        "lr": params.train.lr,
        "betas": params.train.betas,
        "weight_decay": params.train.weight_decay,
        "adam_epsilon": params.train.adam_epsilon,
    }
    # PPO needs extra kwargs
    if params.train.alg_name.lower() == "ppo":
        kwargs["vf_clip"] = params.train.ppo_vf_clip
        kwargs["tau"] = params.train.ppo_tau
        kwargs["gamma"] = params.train.ppo_gamma

    ray_runners = []
    for rank in range(world_size):
        ray_vars = {
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": str(master_port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": "0",
            "DS_SKIP_CUDA_CHECK": "1",
        }
        for _env_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "CUDA_HOME"):
            if os.environ.get(_env_key):
                ray_vars[_env_key] = os.environ[_env_key]
        runner = alg.options(num_gpus=1, runtime_env={"env_vars": ray_vars}).remote(**kwargs)
        ray_runners.append(runner)

    return ray_runners


def rollout_engine_setup(params, reward_fnc, eos_id):
    """Create Ray rollout engine actors (vLLM inference workers)."""
    tp = int(params.rollout.tensor_parallel_size)
    rollout_gpus = int(params.run.rollout_gpus)

    kwargs = {
        "model_path": params.model.name,
        "trust_remote_code": params.model.trust_remote_code,
        "seed": params.run.seed,
        "temperature": params.rollout.temperature,
        "max_tokens": params.rollout.max_tokens,
        "n_samples": params.rollout.n_samples,
        "top_p": params.rollout.top_p,
        "top_k": params.rollout.top_k,
        "ignore_eos": params.rollout.ignore_eos,
        "stop": params.rollout.stop,
        "stop_token_ids": params.rollout.stop_token_ids,
        "prompt_logprobs": params.rollout.prompt_logprobs,
        "gpu_memory_utilization": params.rollout.gpu_memory_utilization,
        "force_strict_on_policy": params.rollout.force_strict_on_policy,
        "eos_id": eos_id,
        "tensor_parallel_size": tp,
        "reward_func": reward_fnc,
        "reward_broadcast": params.reward.broadcast,
        "eps_reward_norm": params.reward.eps_reward_norm,
    }

    num_rollout_engines = max(1, rollout_gpus // tp)
    _rollout_env_vars = {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"}
    for _env_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "CUDA_HOME"):
        if os.environ.get(_env_key):
            _rollout_env_vars[_env_key] = os.environ[_env_key]

    rollout_engines = []
    for i in range(num_rollout_engines):
        kwargs["engine_id"] = i
        _rollout_opts = {"num_gpus": tp}
        if _rollout_env_vars:
            _rollout_opts["runtime_env"] = {"env_vars": _rollout_env_vars}
        rollout_engines.append(VLLMRolloutEngine.options(**_rollout_opts).remote(**kwargs))

    return num_rollout_engines, rollout_engines
