"""
Pluggable metrics tracking for oxRL.

Provides a MetricsTracker ABC with MLflow, WandB, TensorBoard, and NoOp
backends. Use create_tracker() to instantiate based on config.
"""
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MetricsTracker(ABC):
    """Abstract base for experiment-tracking backends."""

    @abstractmethod
    def log_metrics(self, metrics: dict, step: int = None):
        ...

    @abstractmethod
    def log_params(self, params: dict):
        ...

    @abstractmethod
    def end_run(self):
        ...


class NoOpTracker(MetricsTracker):
    """Silent tracker for non-rank-0 workers or disabled tracking."""

    def log_metrics(self, metrics: dict, step: int = None):
        pass

    def log_params(self, params: dict):
        pass

    def end_run(self):
        pass


class MLflowTracker(MetricsTracker):
    """Wraps mlflow tracking calls."""

    def __init__(self, tracking_uri: str, experiment_name: str, run_name: str):
        try:
            import mlflow as _mlflow
        except ImportError:
            raise ImportError("mlflow not installed. pip install mlflow")
        self._mlflow = _mlflow
        _mlflow.set_tracking_uri(tracking_uri)
        _mlflow.set_experiment(experiment_name)
        self._run = _mlflow.start_run(run_name=run_name)

    def log_metrics(self, metrics: dict, step: int = None):
        self._mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: dict):
        self._mlflow.log_params(params)

    def end_run(self):
        self._mlflow.end_run()


class WandbTracker(MetricsTracker):
    """Wraps Weights & Biases tracking calls."""

    def __init__(self, project: str, name: str, config: dict = None):
        try:
            import wandb as _wandb
        except ImportError:
            raise ImportError("wandb not installed. pip install wandb")
        self._wandb = _wandb
        self._run = _wandb.init(project=project, name=name, config=config or {})

    def log_metrics(self, metrics: dict, step: int = None):
        if step is not None:
            metrics = {**metrics, "global_step": step}
        self._wandb.log(metrics)

    def log_params(self, params: dict):
        self._run.config.update(params)

    def end_run(self):
        self._wandb.finish()


class TensorBoardTracker(MetricsTracker):
    """Wraps torch.utils.tensorboard.SummaryWriter."""

    def __init__(self, log_dir: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError(
                "tensorboard not installed. pip install tensorboard"
            )
        self._writer = SummaryWriter(log_dir=log_dir)

    def log_metrics(self, metrics: dict, step: int = None):
        for key, value in metrics.items():
            self._writer.add_scalar(key, value, global_step=step)
        self._writer.flush()

    def log_params(self, params: dict):
        # TensorBoard has no native params — log as text
        text = "\n".join(f"**{k}**: {v}" for k, v in params.items())
        self._writer.add_text("params", text)
        self._writer.flush()

    def end_run(self):
        self._writer.close()


def create_tracker(config, rank: int) -> MetricsTracker:
    """Factory: build a MetricsTracker from config.run.tracker.

    Returns NoOpTracker for non-rank-0 workers so callers never need
    to guard with ``if rank == 0``.
    """
    if rank != 0:
        return NoOpTracker()

    tracker_name = getattr(config.run, "tracker", "mlflow")

    if tracker_name == "none":
        return NoOpTracker()

    if tracker_name == "mlflow":
        tracking_uri = getattr(config.run, "tracking_uri", "")
        try:
            return MLflowTracker(
                tracking_uri=tracking_uri,
                experiment_name=config.run.project_name,
                run_name=config.run.experiment_id,
            )
        except ImportError:
            logger.warning("mlflow not installed — falling back to NoOpTracker")
            return NoOpTracker()

    if tracker_name == "wandb":
        return WandbTracker(
            project=config.run.project_name,
            name=config.run.experiment_id,
            config=config.model_dump() if hasattr(config, "model_dump") else {},
        )

    if tracker_name == "tensorboard":
        import os
        log_dir = os.path.join(
            getattr(config.run, "checkpoint_dir", None) or "runs",
            config.run.experiment_id,
        )
        return TensorBoardTracker(log_dir=log_dir)

    raise ValueError(
        f"Unknown tracker '{tracker_name}'. "
        f"Supported: mlflow, wandb, tensorboard, none"
    )
