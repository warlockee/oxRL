"""
Rollout collection phase: generate samples from rollout engines.
"""
import sys
import time

from oxrl.utils.ray_utils import ray_get_with_timeout

# How often (in batches) to print a progress line from the main process.
# This ensures visible output even when Ray buffers remote actor stdout.
_PROGRESS_INTERVAL = 25


def collect_rollouts(
    rollout_dataloader,
    num_rollout_engines,
    rollout_engines,
    epoch,
    policy_version,
    replay_buffer,
    ray_agent,
    timeout_sec=0,
):
    """Run rollout engines and collect samples into the replay buffer.

    Returns dict with stats: total_samples_generated, avg_reward, avg_response_len, rollout_time.
    """
    assert num_rollout_engines == len(rollout_engines), (
        "Number of rollout engines does not match with the number of rollout engines"
    )

    rollout_start_time = time.time()
    total_samples_generated = 0
    total_reward_sum = 0.0
    total_response_len = 0

    batch_size = rollout_dataloader.batch_size
    dataset_size = len(rollout_dataloader.dataset)
    num_steps_to_generate_all = (dataset_size + batch_size - 1) // batch_size

    print(
        f"[Rollout Stats] Dataset size: {dataset_size} | "
        f"Batch size: {batch_size} "
        f"({num_rollout_engines} engines x {batch_size // num_rollout_engines} per engine), "
        f"Steps to generate all samples: {num_steps_to_generate_all}"
    )

    for batch_idx, rollout_batch in enumerate(rollout_dataloader):
        # 1. split data across rollout engines
        shard_size = (len(rollout_batch) + num_rollout_engines - 1) // num_rollout_engines
        rollout_shards = [
            rollout_batch[i * shard_size : (i + 1) * shard_size]
            for i in range(num_rollout_engines)
        ]
        rollout_shards = [shard for shard in rollout_shards if len(shard) > 0]

        # 2. schedule rollout generation
        rollout_samples = []
        for i, shard in enumerate(rollout_shards):
            rollout_samples.append(
                rollout_engines[i].generate.remote(
                    prompts=shard, current_iter=epoch, policy_version=policy_version
                )
            )

        # 3. gather rollouts
        rollout_lists = ray_get_with_timeout(
            rollout_samples, timeout_sec=timeout_sec, description="rollout generation"
        )

        # 4. merge and collect stats
        rollout_merged = []
        for rl in rollout_lists:
            rollout_merged.extend(rl)
            for sample in rl:
                total_samples_generated += 1
                total_reward_sum += sample["rewards"].sum().item()
                total_response_len += sample["response_len"]

        # 5. add to replay buffer
        replay_buffer.add_batch_seqs(rollout_merged)

        # 6. periodic progress from the main process (visible even when Ray
        #    buffers remote actor stdout, preventing "apparent deadlock" in logs)
        if (batch_idx + 1) % _PROGRESS_INTERVAL == 0 or (batch_idx + 1) == num_steps_to_generate_all:
            elapsed = time.time() - rollout_start_time
            avg_r = total_reward_sum / max(1, total_samples_generated)
            print(
                f"[Rollout] batch {batch_idx + 1}/{num_steps_to_generate_all} | "
                f"samples={total_samples_generated} | "
                f"avg_reward={avg_r:.4f} | "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
            sys.stdout.flush()

    rollout_time = time.time() - rollout_start_time
    avg_reward = total_reward_sum / max(1, total_samples_generated)
    avg_response_len = total_response_len / max(1, total_samples_generated)

    if len(replay_buffer) <= 1:
        raise ValueError("Replay buffer is empty")

    return {
        "total_samples_generated": total_samples_generated,
        "avg_reward": avg_reward,
        "avg_response_len": avg_response_len,
        "rollout_time": rollout_time,
    }
