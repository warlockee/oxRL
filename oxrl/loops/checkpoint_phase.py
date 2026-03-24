"""
Checkpoint saving and rollout engine refresh phase.

Handles GPU memory contention between DeepSpeed training engines and vLLM
rollout engines that share the same physical GPUs. Before refreshing the
rollout engine (which destroys and recreates the vLLM LLM instance), we
offload DeepSpeed optimizer states to CPU to free GPU memory for the rollout
engine. Optimizer states are reloaded later (before the next training phase)
by the caller.
"""
import os
import torch
import ray

from oxrl.utils.ray_utils import ray_get_with_timeout


def offload_training_engines(training_engine_runners, logger, epoch, timeout_sec=0):
    """Offload DeepSpeed optimizer states to CPU on all training engines.

    This frees GPU memory (typically ~2x parameter size for Adam optimizer
    states) so the vLLM rollout engine can reload the updated model on
    the shared GPU.

    Public function — called by main_rl.py to manage memory around phases.
    """
    logger.info(f"[Epoch {epoch+1}] Offloading training engine optimizer states to CPU...")
    offload_futures = [engine.offload_to_cpu.remote() for engine in training_engine_runners]
    try:
        offload_results = ray_get_with_timeout(
            offload_futures, timeout_sec=timeout_sec, description="offload_to_cpu"
        )
        success_count = sum(1 for r in offload_results if r)
        logger.info(
            f"[Epoch {epoch+1}] Offloaded {success_count}/{len(training_engine_runners)} "
            f"training engines to CPU"
        )
    except Exception as e:
        logger.warning(f"[Epoch {epoch+1}] offload_to_cpu failed: {e} (continuing anyway)")


def reload_training_engines(training_engine_runners, logger, epoch, timeout_sec=0):
    """Reload DeepSpeed optimizer states back to GPU on all training engines.

    Called before the training phase to restore optimizer states.

    Public function — called by main_rl.py to manage memory around phases.
    """
    logger.info(f"[Epoch {epoch+1}] Reloading training engine optimizer states to GPU...")
    reload_futures = [engine.reload_to_gpu.remote() for engine in training_engine_runners]
    try:
        reload_results = ray_get_with_timeout(
            reload_futures, timeout_sec=timeout_sec, description="reload_to_gpu"
        )
        success_count = sum(1 for r in reload_results if r)
        logger.info(
            f"[Epoch {epoch+1}] Reloaded {success_count}/{len(training_engine_runners)} "
            f"training engines to GPU"
        )
    except Exception as e:
        logger.warning(f"[Epoch {epoch+1}] reload_to_gpu failed: {e} (continuing anyway)")


def save_and_refresh(
    training_engine_runners,
    rollout_engines,
    model_path,
    tag,
    policy_version,
    tokenizer,
    rank,
    logger,
    epoch,
    timeout_sec=0,
):
    """Save checkpoint and refresh rollout engines.

    Tries in-memory gather + parallel save/refresh (fast path).
    Falls back to disk-based sequential flow if gather fails.

    IMPORTANT: This function offloads DeepSpeed optimizer states to CPU before
    refreshing the rollout engines, and LEAVES them offloaded. The caller
    (main_rl.py) must call reload_training_engines() before the next training
    phase. This keeps optimizer states off GPU during the rollout phase,
    maximizing available memory for vLLM generation.
    """
    # 1. Gather state dict in memory (all ranks participate — collective op)
    try:
        gather_futures = [engine.gather_state_dict.remote() for engine in training_engine_runners]
        gather_results = ray_get_with_timeout(
            gather_futures, timeout_sec=timeout_sec, description="gather_state_dict"
        )
        state_dict = next((r for r in gather_results if r is not None), None)
    except Exception as e:
        logger.warning(f"[Epoch {epoch+1}] gather_state_dict failed ({e}), falling back to disk-based flow")
        state_dict = None

    if state_dict is not None:
        # Fast path: in-memory gather + parallel save/refresh
        config_dict = state_dict.pop("__model_config_dict__", None)
        value_head_sd = state_dict.pop("__value_head_state_dict__", None)
        state_dict_ref = ray.put(state_dict)
        del state_dict

        if rank == 0:
            os.makedirs(model_path, exist_ok=True)
            tokenizer.save_pretrained(model_path)
            if config_dict is not None:
                from oxrl.tools.checkpoint import save_config_json
                save_config_json(model_path, config_dict)
                logger.info(f"[Epoch {epoch+1}] Saved config.json to {model_path}")

        # Save checkpoint first (does not need GPU memory freed)
        save_futures = [
            engine.save_checkpoint.remote(output_dir=model_path, tag=tag, state_dict_ref=state_dict_ref)
            for engine in training_engine_runners
        ]
        ray_get_with_timeout(
            save_futures, timeout_sec=timeout_sec,
            description="save_checkpoint"
        )

        # Offload training engines to CPU BEFORE refreshing rollout engines.
        # This frees GPU memory so vLLM can reload the model.
        offload_training_engines(training_engine_runners, logger, epoch, timeout_sec)

        # Now refresh rollout engines with freed GPU memory
        refresh_futures = [
            eng.refresh_model_from_state_dict.remote(state_dict_ref, config_dict, policy_version)
            for eng in rollout_engines
        ]
        ray_get_with_timeout(
            refresh_futures, timeout_sec=timeout_sec,
            description="refresh_model"
        )

        # NOTE: Optimizer states are LEFT offloaded on CPU. The caller
        # (main_rl.py) will call reload_training_engines() before training.
        # This keeps GPU memory free for the rollout phase.

        if rank == 0 and value_head_sd is not None:
            torch.save(value_head_sd, os.path.join(model_path, "value_head.pt"))
            logger.info(f"[Epoch {epoch+1}] Value head saved to {model_path}")
    else:
        # Fallback: sequential disk-based flow
        if rank == 0:
            os.makedirs(model_path, exist_ok=True)
            tokenizer.save_pretrained(model_path)

        save_futures = [
            engine.save_checkpoint.remote(output_dir=model_path, tag=tag)
            for engine in training_engine_runners
        ]
        ray_get_with_timeout(save_futures, timeout_sec=timeout_sec, description="save_checkpoint (fallback)")

        if rank == 0:
            os.sync()

        # Offload training engines to CPU BEFORE refreshing rollout engines
        offload_training_engines(training_engine_runners, logger, epoch, timeout_sec)

        refresh_futures = [
            eng.refresh_model.remote(model_path, policy_version) for eng in rollout_engines
        ]
        ray_get_with_timeout(refresh_futures, timeout_sec=timeout_sec, description="refresh_model (fallback)")

        # NOTE: Optimizer states are LEFT offloaded on CPU (same as fast path).

    logger.info(f"[Epoch {epoch+1}] Checkpoint saved and rollout engines refreshed")
