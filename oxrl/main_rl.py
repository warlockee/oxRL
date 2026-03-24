"""
oxRL — RL training orchestrator.

Thin entry point that wires together setup, rollout, training, and checkpoint phases.
Each phase lives in its own module under oxrl/setup/ and oxrl/loops/.
"""
import os
import argparse
import importlib
import time
import numpy as np
import ray

import oxrl.configs.load as cfg
from oxrl.utils.utils import get_experiment_dir_name
from oxrl.utils.logging import setup_logging, setup_mlflow, log_metrics, end_run
from oxrl.utils.setup import set_random_seeds, get_rank_info, load_tokenizer
from oxrl.rollouts.replay_buffer import ReplayBuffer

from oxrl.setup.ray_setup import setup_ray
from oxrl.setup.engine_factory import get_algorithm_class, training_engine_setup, rollout_engine_setup
from oxrl.setup.dataloader_factory import rollout_dataloader_setup
from oxrl.loops.rollout_phase import collect_rollouts
from oxrl.loops.train_phase import run_training_steps
from oxrl.loops.checkpoint_phase import save_and_refresh, reload_training_engines


def main(config_file, experiment_id, log_level="INFO"):
    # 1. Miscellaneous setups
    rank, local_rank = get_rank_info()
    logger = setup_logging(rank=rank, log_level=log_level)
    logger.info("Starting RL training...")

    config = cfg.load_and_verify(method="rl", input_yaml=config_file, experiment_id=experiment_id)
    set_random_seeds(seed=config.run.seed)
    mlflow_run = setup_mlflow(config=config, tracking_uri=config.run.tracking_uri, rank=rank)
    logger.info(f"Config loaded. experiment_id: {config.run.experiment_id}")

    training_gpus = config.run.training_gpus
    rollout_gpus = config.run.rollout_gpus

    # 2. Initialize Ray
    logger.info("Initializing Ray cluster...")
    ray_engine, master_addr = setup_ray(ray_address=config.run.ray_address)
    logger.info(f"Ray initialized. Master address: {master_addr}")

    # Validate GPU budget: warn if training + rollout exceeds available GPUs.
    # The engine_factory handles this via fractional GPU allocation, but log
    # the situation so users understand what is happening.
    total_cluster_gpus = int(ray.cluster_resources().get("GPU", 0))
    total_requested = training_gpus + rollout_gpus
    if total_requested > total_cluster_gpus:
        logger.warning(
            f"GPU over-subscription: training_gpus={training_gpus} + rollout_gpus={rollout_gpus} "
            f"= {total_requested} > {total_cluster_gpus} available GPUs. "
            f"Rollout engines will colocate with training engines via fractional GPU allocation."
        )

    # 3. Initialize training engine
    logger.info(f"Setting up training algorithm: {config.train.alg_name}")
    alg = get_algorithm_class(config.train.alg_name)
    training_engine_runners = training_engine_setup(
        params=config, alg=alg, world_size=training_gpus,
        master_addr=master_addr, master_port=config.run.ray_master_port,
    )
    assert len(training_engine_runners) == training_gpus
    logger.info(f"Created {len(training_engine_runners)} training engine runners")

    # Wait for initialization barrier
    logger.info("Waiting for all training engines to initialize...")
    ready = ray.get([engine.is_ready.remote() for engine in training_engine_runners])
    logger.info("All training engines ready!")

    # 4. Load tokenizer
    logger.info(f"Loading tokenizer from {config.model.name}")
    tokenizer = load_tokenizer(
        model_name=config.model.name, trust_remote_code=config.model.trust_remote_code, rank=rank,
    )
    logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # 5. Initialize rollout engines
    logger.info("Setting up inference/rollout engines...")
    if not config.reward.reward_func:
        raise ValueError("Reward function not specified")
    reward_module = importlib.import_module("oxrl.rewards")
    reward_fnc = getattr(reward_module, config.reward.reward_func)
    logger.info(f"Using reward function: {config.reward.reward_func}")

    num_rollout_engines, rollout_engines = rollout_engine_setup(
        params=config, reward_fnc=reward_fnc, eos_id=tokenizer.eos_token_id,
    )
    logger.info(f"Created {num_rollout_engines} rollout engines")

    # 6. Dataloader and replay buffer
    rollout_dataloader = rollout_dataloader_setup(
        params=config, tokenizer=tokenizer, num_rollout_engines=num_rollout_engines,
    )
    logger.info(f"Rollout dataloader ready. Total batches per epoch: {len(rollout_dataloader)}")

    replay_buffer = ReplayBuffer(pad_token_id=tokenizer.pad_token_id, max_seq_len=config.data.max_seq_len)

    # 7. Training loop
    policy_version = 0
    number_of_epochs = config.train.total_number_of_epochs
    number_of_training_steps_per_epoch = config.train.train_steps_per_epoch
    global_step = 0

    logger.info("=" * 50)
    logger.info(f"Starting training: {number_of_epochs} epochs, {number_of_training_steps_per_epoch} steps/epoch")
    logger.info(f"Training GPUs: {training_gpus}, Rollout GPUs: {rollout_gpus}")
    logger.info("=" * 50)

    for epoch in range(number_of_epochs):
        epoch_start_time = time.time()
        logger.info(f"[Epoch {epoch+1}/{number_of_epochs}] Starting rollout generation...")

        # Phase 1: Rollout
        rollout_stats = collect_rollouts(
            rollout_dataloader=rollout_dataloader,
            num_rollout_engines=num_rollout_engines,
            rollout_engines=rollout_engines,
            epoch=epoch, policy_version=policy_version,
            replay_buffer=replay_buffer, ray_agent=ray,
        )
        logger.info(
            f"[Epoch {epoch+1}] Rollout complete: {rollout_stats['total_samples_generated']} samples, "
            f"avg_reward={rollout_stats['avg_reward']:.4f}, time={rollout_stats['rollout_time']:.2f}s"
        )

        if len(replay_buffer) <= 1:
            raise ValueError("Replay buffer is empty")

        # Phase 2: Training
        # Reload optimizer states to GPU before training (they are offloaded
        # to CPU after the previous epoch's checkpoint+refresh phase to keep
        # GPU memory free during rollout generation).
        if epoch > 0:
            reload_training_engines(
                training_engine_runners, logger, epoch,
                timeout_sec=getattr(config.run, 'ray_task_timeout_sec', 0),
            )

        logger.info(f"[Epoch {epoch+1}] Starting training on {len(replay_buffer)} replay buffer samples...")
        train_start_time = time.time()

        epoch_metrics, global_step = run_training_steps(
            training_engine_runners=training_engine_runners,
            replay_buffer=replay_buffer,
            train_batch_size_per_gpu=config.train.train_batch_size_per_gpu,
            number_of_training_steps=number_of_training_steps_per_epoch,
            epoch=epoch, global_step=global_step,
            logger=logger, rank=rank, log_metrics_fn=log_metrics,
        )

        policy_version += 1
        replay_buffer.reset()

        train_time = time.time() - train_start_time
        epoch_time = time.time() - epoch_start_time
        epoch_avg_loss = np.mean(epoch_metrics["loss_total"])

        logger.info(
            f"[Epoch {epoch+1}] Training complete: time={train_time:.2f}s, avg_loss={epoch_avg_loss:.4f}"
        )

        # Log epoch metrics
        if rank == 0 and mlflow_run:
            log_metrics({
                "epoch/avg_loss": epoch_avg_loss,
                "epoch/avg_kl_old": np.mean(epoch_metrics["kl_old"]),
                "epoch/avg_kl_ref": np.mean(epoch_metrics["kl_ref"]),
                "epoch/avg_clipfrac": np.mean(epoch_metrics["clipfrac"]),
                "epoch/avg_reward": rollout_stats["avg_reward"],
                "epoch/avg_response_len": rollout_stats["avg_response_len"],
                "epoch/total_samples": rollout_stats["total_samples_generated"],
                "epoch/rollout_time_sec": rollout_stats["rollout_time"],
                "epoch/train_time_sec": train_time,
                "epoch/total_time_sec": epoch_time,
            }, step=epoch + 1)

        # Phase 3: Checkpoint + Refresh
        tag = f"iter{epoch+1:06d}_v{policy_version:06d}"
        model_path = get_experiment_dir_name(
            output_dir=config.run.checkpoint_dir, tag=tag, experiment_id=config.run.experiment_id,
        )
        logger.info(f"[Epoch {epoch+1}] Saving checkpoint to {model_path}")

        save_and_refresh(
            training_engine_runners=training_engine_runners,
            rollout_engines=rollout_engines,
            model_path=model_path, tag=tag, policy_version=policy_version,
            tokenizer=tokenizer, rank=rank, logger=logger, epoch=epoch,
        )

        logger.info(f"[Epoch {epoch+1}] Complete! Total epoch time: {epoch_time:.2f}s")
        logger.info("=" * 50)

    if rank == 0 and mlflow_run:
        end_run()

    logger.info("Training completed successfully!")
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="./config/dummy.yaml", help="config file")
    parser.add_argument("--experiment_id", type=str, default="run_1", help="experiment id")
    parser.add_argument("--log-level", type=str, default="INFO", help="logging level")
    args = parser.parse_args()
    main(args.config_file, args.experiment_id, args.log_level)
