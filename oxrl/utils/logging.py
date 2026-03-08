import os
import logging

from oxrl.utils.metrics import MetricsTracker, NoOpTracker, create_tracker

try:
    import mlflow
except ImportError:
    mlflow = None

# Module-level tracker singleton
_tracker: MetricsTracker = NoOpTracker()

def setup_logging(rank: int, log_level: str = "INFO", exp_name: str = "") -> logging.Logger:
    '''
        Setup logging configuration. Only rank 0 logs to console to avoid duplicate messages.
    '''
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger(exp_name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    # Format with timestamp and rank
    formatter = logging.Formatter(
        fmt=f"[%(asctime)s][Rank {rank}][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler (only rank 0)
    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def setup_tracker(config, rank: int) -> MetricsTracker:
    """Create and store a MetricsTracker from config.run.tracker.

    Returns the tracker instance (also accessible via get_tracker()).
    Logs base config params automatically.
    """
    global _tracker
    _tracker = create_tracker(config, rank)

    # Log base config parameters (common to SL and RL)
    _tracker.log_params({
        "alg_name": config.train.alg_name,
        "model_name": config.model.name,
        "learning_rate": config.train.lr,
        "train_batch_size_per_gpu": config.train.train_batch_size_per_gpu,
        "gradient_accumulation_steps": config.train.gradient_accumulation_steps,
        "total_epochs": config.train.total_number_of_epochs,
        "seed": config.run.seed,
        "tracker": getattr(config.run, "tracker", "mlflow"),
    })

    # Log method-specific step config
    if config.run.method == "sl":
        _tracker.log_params({
            "micro_batches_per_epoch": config.train.micro_batches_per_epoch,
        })
    elif config.run.method == "rl":
        _tracker.log_params({
            "train_steps_per_epoch": config.train.train_steps_per_epoch,
            "n_samples": config.rollout.n_samples,
            "max_tokens": config.rollout.max_tokens,
            "kl_coeff": config.train.kl_coeff,
            "clip_low": config.train.clip_low,
            "clip_high": config.train.clip_high,
            "entropy_coeff": config.train.entropy_coeff,
            "training_gpus": config.run.training_gpus,
            "rollout_gpus": config.run.rollout_gpus,
        })

    return _tracker

def get_tracker() -> MetricsTracker:
    """Return the current module-level MetricsTracker."""
    return _tracker

def setup_mlflow(config, tracking_uri: str, rank: int):
    '''
        Setup MLflow tracking. Only rank 0 logs to MLflow.
        Backward-compatible entry point — also sets the module-level tracker.
    '''
    if rank != 0:
        return None

    if mlflow is None:
        logging.getLogger().warning("mlflow not installed — skipping experiment tracking. pip install mlflow to enable.")
        return None

    mlflow.set_tracking_uri(tracking_uri)

    # Set experiment name
    experiment_name = config.run.project_name
    mlflow.set_experiment(experiment_name)

    # Start run
    run = mlflow.start_run(run_name=config.run.experiment_id)

    # Log base config parameters (common to SL and RL)
    mlflow.log_params({
        "alg_name": config.train.alg_name,
        "model_name": config.model.name,
        "learning_rate": config.train.lr,
        "train_batch_size_per_gpu": config.train.train_batch_size_per_gpu,
        "gradient_accumulation_steps": config.train.gradient_accumulation_steps,
        "total_epochs": config.train.total_number_of_epochs,
        "seed": config.run.seed,
    })

    # Log method-specific step config
    if config.run.method == "sl":
        mlflow.log_params({
            "micro_batches_per_epoch": config.train.micro_batches_per_epoch,
        })
    elif config.run.method == "rl":
        mlflow.log_params({
            "train_steps_per_epoch": config.train.train_steps_per_epoch,
            "n_samples": config.rollout.n_samples,
            "max_tokens": config.rollout.max_tokens,
            "kl_coeff": config.train.kl_coeff,
            "clip_low": config.train.clip_low,
            "clip_high": config.train.clip_high,
            "entropy_coeff": config.train.entropy_coeff,
            "training_gpus": config.run.training_gpus,
            "rollout_gpus": config.run.rollout_gpus,
        })

    # Also set the module-level tracker for new-style callers
    global _tracker
    from oxrl.utils.metrics import MLflowTracker
    # Create a lightweight wrapper that reuses the already-started run
    tracker = NoOpTracker.__new__(MLflowTracker)
    tracker._mlflow = mlflow
    tracker._run = run
    _tracker = tracker

    return run

def log_metrics(metrics: dict, step: int = None):
    '''Log metrics via the active tracker. Falls back to direct mlflow for backward compat.'''
    _tracker.log_metrics(metrics, step=step)

def end_run():
    '''End the current tracking run.'''
    _tracker.end_run()
