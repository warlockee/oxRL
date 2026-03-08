"""
oxRL — RL training orchestrator.

Thin entry point that wires together setup, rollout, training, and checkpoint phases.
Each phase lives in its own module under oxrl/setup/ and oxrl/loops/.
"""
import os
import re
import argparse
import time
import numpy as np
import ray

import oxrl.configs.load as cfg
from oxrl.utils.utils import get_experiment_dir_name
from oxrl.utils.logging import setup_logging, setup_tracker, log_metrics, end_run
from oxrl.utils.gpu_metrics import get_gpu_memory_metrics
from oxrl.utils.setup import set_random_seeds, get_rank_info, load_tokenizer
from oxrl.rollouts.replay_buffer import ReplayBuffer

from oxrl.setup.ray_setup import setup_ray
from oxrl.setup.engine_factory import get_algorithm_class, training_engine_setup, rollout_engine_setup
from oxrl.setup.dataloader_factory import rollout_dataloader_setup
from oxrl.loops.rollout_phase import collect_rollouts
from oxrl.loops.train_phase import run_training_steps
from oxrl.loops.checkpoint_phase import save_and_refresh


def main(config_file, experiment_id, log_level="INFO"):
    # 1. Miscellaneous setups
    rank, local_rank = get_rank_info()
    logger = setup_logging(rank=rank, log_level=log_level)
    logger.info("Starting RL training...")

    config = cfg.load_and_verify(method="rl", input_yaml=config_file, experiment_id=experiment_id)

    # Auto-load researcher extension module (registers custom losses/algorithms)
    if hasattr(config, 'research') and config.research.module:
        from oxrl.models.research_adapters import load_research_module
        load_research_module(config.research.module)

    set_random_seeds(seed=config.run.seed)
    tracker = setup_tracker(config=config, rank=rank)
    logger.info(f"Config loaded. experiment_id: {config.run.experiment_id}")

    # Resume from checkpoint: override model path so engines load from checkpoint
    start_epoch = 0
    start_policy_version = 0
    if config.run.resume_from:
        resume_path = config.run.resume_from
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        tag = os.path.basename(resume_path)
        match = re.match(r"iter(\d+)_v(\d+)", tag)
        if match:
            start_epoch = int(match.group(1))
            start_policy_version = int(match.group(2))
        else:
            match = re.match(r"iter(\d+)", tag)
            if match:
                start_epoch = int(match.group(1))
        config.model.name = resume_path
        logger.info(f"Resuming from epoch {start_epoch}, policy v{start_policy_version}, path={resume_path}")

    training_gpus = config.run.training_gpus
    rollout_gpus = config.run.rollout_gpus

    # 2. Initialize Ray
    logger.info("Initializing Ray cluster...")
    ray_engine, master_addr = setup_ray(ray_address=config.run.ray_address)
    logger.info(f"Ray initialized. Master address: {master_addr}")

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
    from oxrl.rewards.loader import create_reward_backend
    reward_fnc = create_reward_backend(
        reward_func=config.reward.reward_func,
        composite_rewards=config.reward.composite_rewards,
        llm_judge_config=config.reward.llm_judge,
        reward_model_config=config.reward.reward_model,
    )
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
    policy_version = start_policy_version
    number_of_epochs = config.train.total_number_of_epochs
    number_of_training_steps_per_epoch = config.train.train_steps_per_epoch
    global_step = start_epoch * number_of_training_steps_per_epoch
    best_reward = float('-inf')
    timeout_sec = config.run.ray_task_timeout_sec

    logger.info("=" * 50)
    logger.info(f"Starting training: {number_of_epochs} epochs, {number_of_training_steps_per_epoch} steps/epoch")
    logger.info(f"Training GPUs: {training_gpus}, Rollout GPUs: {rollout_gpus}")
    if start_epoch > 0:
        logger.info(f"Resuming from epoch {start_epoch}, global_step {global_step}")
    logger.info("=" * 50)

    for epoch in range(start_epoch, number_of_epochs):
        epoch_retries = 0
        while True:
          try:
            epoch_start_time = time.time()
            logger.info(f"[Epoch {epoch+1}/{number_of_epochs}] Starting rollout generation...")

            # Phase 1: Rollout
            rollout_stats = collect_rollouts(
                rollout_dataloader=rollout_dataloader,
                num_rollout_engines=num_rollout_engines,
                rollout_engines=rollout_engines,
                epoch=epoch, policy_version=policy_version,
                replay_buffer=replay_buffer, ray_agent=ray,
                timeout_sec=timeout_sec,
            )
            logger.info(
                f"[Epoch {epoch+1}] Rollout complete: {rollout_stats['total_samples_generated']} samples, "
                f"avg_reward={rollout_stats['avg_reward']:.4f}, time={rollout_stats['rollout_time']:.2f}s"
            )

            if len(replay_buffer) <= 1:
                raise ValueError("Replay buffer is empty")

            # Phase 2: Training
            logger.info(f"[Epoch {epoch+1}] Starting training on {len(replay_buffer)} replay buffer samples...")
            train_start_time = time.time()

            epoch_metrics, global_step = run_training_steps(
                training_engine_runners=training_engine_runners,
                replay_buffer=replay_buffer,
                train_batch_size_per_gpu=config.train.train_batch_size_per_gpu,
                number_of_training_steps=number_of_training_steps_per_epoch,
                epoch=epoch, global_step=global_step,
                logger=logger, rank=rank, log_metrics_fn=log_metrics,
                timeout_sec=timeout_sec,
            )

            policy_version += 1
            replay_buffer.reset()

            train_time = time.time() - train_start_time
            epoch_time = time.time() - epoch_start_time
            epoch_avg_loss = np.mean(epoch_metrics["loss_total"])

            logger.info(
                f"[Epoch {epoch+1}] Training complete: time={train_time:.2f}s, avg_loss={epoch_avg_loss:.4f}"
            )

            # Log epoch metrics (NoOpTracker handles non-rank-0)
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
                **get_gpu_memory_metrics(),
            }, step=epoch + 1)

            # Best-model checkpoint (based on average reward)
            if config.run.save_best_checkpoint and rollout_stats["avg_reward"] > best_reward:
                best_reward = rollout_stats["avg_reward"]
                best_path = get_experiment_dir_name(
                    output_dir=config.run.checkpoint_dir, tag="best",
                    experiment_id=config.run.experiment_id,
                )
                logger.info(f"[Epoch {epoch+1}] New best reward: {best_reward:.4f}. Saving to {best_path}")
                save_and_refresh(
                    training_engine_runners=training_engine_runners,
                    rollout_engines=[],  # Don't refresh rollout engines for best checkpoint
                    model_path=best_path, tag="best", policy_version=policy_version,
                    tokenizer=tokenizer, rank=rank, logger=logger, epoch=epoch,
                    timeout_sec=timeout_sec,
                )

            # Phase 3: Periodic Checkpoint + Refresh
            is_checkpoint_epoch = (
                (epoch + 1) % config.run.checkpoint_every_n_epochs == 0
                or (epoch + 1) == number_of_epochs
            )
            if is_checkpoint_epoch:
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
                    timeout_sec=timeout_sec,
                )

                # Cleanup old checkpoints
                if config.run.keep_last_n_checkpoints is not None and rank == 0:
                    from oxrl.tools.checkpoint import cleanup_old_checkpoints
                    cleanup_old_checkpoints(
                        checkpoint_dir=config.run.checkpoint_dir,
                        experiment_id=config.run.experiment_id,
                        keep_last_n=config.run.keep_last_n_checkpoints,
                        exclude_tags=["best"],
                    )

            logger.info(f"[Epoch {epoch+1}] Complete! Total epoch time: {epoch_time:.2f}s")
            logger.info("=" * 50)
            break  # Epoch succeeded — exit retry loop

          except Exception as e:
            epoch_retries += 1
            if epoch_retries > config.run.max_epoch_retries:
                logger.error(f"[Epoch {epoch+1}] Failed after {epoch_retries} attempts: {e}")
                raise
            logger.warning(
                f"[Epoch {epoch+1}] Failed (attempt {epoch_retries}/{config.run.max_epoch_retries}): "
                f"{e}. Retrying..."
            )
            replay_buffer.reset()
            time.sleep(5)

    if rank == 0:
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
