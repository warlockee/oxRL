"""
Training phase: run optimizer steps on replay buffer data.
"""
import numpy as np
import ray
from torch.utils.data import DataLoader


def run_training_steps(
    training_engine_runners,
    replay_buffer,
    train_batch_size_per_gpu,
    number_of_training_steps,
    epoch,
    global_step,
    logger,
    rank,
    mlflow_run,
    log_metrics_fn,
):
    """Execute training steps for one epoch. Returns (epoch_metrics, new_global_step)."""
    # Create dataloader from replay buffer
    train_batches = list(
        DataLoader(
            dataset=replay_buffer,
            batch_size=train_batch_size_per_gpu,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=replay_buffer.collate_fn,
        )
    )
    logger.info(f"[Epoch {epoch+1}] Created {len(train_batches)} training batches")

    # Pad batches for equal distribution across engines
    num_train_engines = len(training_engine_runners)
    num_batches = len(train_batches)
    batches_per_engine = (num_batches + num_train_engines - 1) // num_train_engines
    total_batches_needed = batches_per_engine * num_train_engines

    if total_batches_needed > num_batches:
        padding = [train_batches[-1]] * (total_batches_needed - num_batches)
        train_batches_padded = train_batches + padding
    else:
        train_batches_padded = train_batches

    # Run training steps
    epoch_metrics = {
        "loss_total": [],
        "loss_pi": [],
        "loss_ent": [],
        "kl_ref": [],
        "kl_old": [],
        "clipfrac": [],
    }

    for tidx in range(number_of_training_steps):
        train_futures = []
        for eid, engine in enumerate(training_engine_runners):
            shard = train_batches_padded[eid::num_train_engines]
            assert len(shard) > 0, f"Engine {eid} has empty shard - this will cause DeepSpeed hang"
            train_futures.append(engine.train_step.remote(engine_id=eid, micro_batches=shard))

        train_metrics = ray.get(train_futures)

        avg_loss = np.mean([m.get("loss_total", 0.0) for m in train_metrics])
        avg_loss_pi = np.mean([m.get("loss_pi", 0.0) for m in train_metrics])
        avg_loss_ent = np.mean([m.get("loss_ent", 0.0) for m in train_metrics])
        avg_kl_ref = np.mean([m.get("kl_ref", 0.0) for m in train_metrics])
        avg_kl_old = np.mean([m.get("kl_old", 0.0) for m in train_metrics])
        avg_clipfrac = np.mean([m.get("clipfrac", 0.0) for m in train_metrics])

        epoch_metrics["loss_total"].append(avg_loss)
        epoch_metrics["loss_pi"].append(avg_loss_pi)
        epoch_metrics["loss_ent"].append(avg_loss_ent)
        epoch_metrics["kl_ref"].append(avg_kl_ref)
        epoch_metrics["kl_old"].append(avg_kl_old)
        epoch_metrics["clipfrac"].append(avg_clipfrac)

        global_step += 1

        if tidx % 10 == 0:
            logger.info(
                f"[Epoch {epoch+1}][Step {tidx+1}/{number_of_training_steps}] "
                f"loss={avg_loss:.4f}, pi_loss={avg_loss_pi:.4f}, ent_loss={avg_loss_ent:.4f}, "
                f"kl_ref={avg_kl_ref:.4f}, kl_old={avg_kl_old:.6f}, clipfrac={avg_clipfrac:.4f}"
            )

        if rank == 0 and mlflow_run:
            log_metrics_fn(
                {
                    "train/loss_total": avg_loss,
                    "train/loss_pi": avg_loss_pi,
                    "train/loss_ent": avg_loss_ent,
                    "train/kl_ref": avg_kl_ref,
                    "train/kl_old": avg_kl_old,
                    "train/clipfrac": avg_clipfrac,
                },
                step=global_step,
            )

    return epoch_metrics, global_step
