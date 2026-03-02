"""
Checkpoint saving and rollout engine refresh phase.
"""
import os
import torch
import ray


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
):
    """Save checkpoint and refresh rollout engines.

    Tries in-memory gather + parallel save/refresh (fast path).
    Falls back to disk-based sequential flow if gather fails.
    """
    # 1. Gather state dict in memory (all ranks participate — collective op)
    try:
        gather_futures = [engine.gather_state_dict.remote() for engine in training_engine_runners]
        gather_results = ray.get(gather_futures)
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

        save_futures = [
            engine.save_checkpoint.remote(output_dir=model_path, tag=tag, state_dict_ref=state_dict_ref)
            for engine in training_engine_runners
        ]
        refresh_futures = [
            eng.refresh_model_from_state_dict.remote(state_dict_ref, config_dict, policy_version)
            for eng in rollout_engines
        ]
        ray.get(save_futures + refresh_futures)

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
        ray.get(save_futures)

        if rank == 0:
            os.sync()

        refresh_futures = [
            eng.refresh_model.remote(model_path, policy_version) for eng in rollout_engines
        ]
        ray.get(refresh_futures)

    logger.info(f"[Epoch {epoch+1}] Checkpoint saved and rollout engines refreshed")
