"""
Tests for MetricsTracker abstraction in oxrl/utils/metrics.py.
All tests run on CPU without external services.
Run with: pytest tests/test_metrics.py -v
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oxrl.utils.metrics import (
    MetricsTracker,
    NoOpTracker,
    MLflowTracker,
    WandbTracker,
    TensorBoardTracker,
    create_tracker,
)


# ============================================================
# NoOpTracker
# ============================================================
class TestNoOpTracker:
    def test_is_metrics_tracker(self):
        assert isinstance(NoOpTracker(), MetricsTracker)

    def test_log_metrics_no_crash(self):
        t = NoOpTracker()
        t.log_metrics({"loss": 0.5}, step=1)

    def test_log_params_no_crash(self):
        t = NoOpTracker()
        t.log_params({"lr": 1e-5})

    def test_end_run_no_crash(self):
        t = NoOpTracker()
        t.end_run()

    def test_log_metrics_with_none_step(self):
        t = NoOpTracker()
        t.log_metrics({"x": 1.0}, step=None)

    def test_log_empty_metrics(self):
        t = NoOpTracker()
        t.log_metrics({})

    def test_log_empty_params(self):
        t = NoOpTracker()
        t.log_params({})

    def test_multiple_calls(self):
        t = NoOpTracker()
        for i in range(100):
            t.log_metrics({"step": i}, step=i)
        t.end_run()


# ============================================================
# MLflowTracker
# ============================================================
class TestMLflowTracker:
    @patch("oxrl.utils.metrics.MLflowTracker.__init__", return_value=None)
    def test_is_metrics_tracker(self, mock_init):
        t = MLflowTracker.__new__(MLflowTracker)
        assert isinstance(t, MetricsTracker)

    @patch.dict("sys.modules", {"mlflow": MagicMock()})
    def test_log_metrics_delegates(self):
        mock_mlflow = MagicMock()
        t = MLflowTracker.__new__(MLflowTracker)
        t._mlflow = mock_mlflow
        t._run = MagicMock()
        t.log_metrics({"loss": 0.5}, step=10)
        mock_mlflow.log_metrics.assert_called_once_with({"loss": 0.5}, step=10)

    @patch.dict("sys.modules", {"mlflow": MagicMock()})
    def test_log_params_delegates(self):
        mock_mlflow = MagicMock()
        t = MLflowTracker.__new__(MLflowTracker)
        t._mlflow = mock_mlflow
        t._run = MagicMock()
        t.log_params({"lr": 1e-5})
        mock_mlflow.log_params.assert_called_once_with({"lr": 1e-5})

    @patch.dict("sys.modules", {"mlflow": MagicMock()})
    def test_end_run_delegates(self):
        mock_mlflow = MagicMock()
        t = MLflowTracker.__new__(MLflowTracker)
        t._mlflow = mock_mlflow
        t._run = MagicMock()
        t.end_run()
        mock_mlflow.end_run.assert_called_once()

    @patch.dict("sys.modules", {"mlflow": MagicMock()})
    def test_init_calls_mlflow(self):
        import sys as _sys
        mock_mlflow = _sys.modules["mlflow"]
        mock_mlflow.start_run.return_value = MagicMock()
        t = MLflowTracker(tracking_uri="http://localhost:5000",
                          experiment_name="test-exp",
                          run_name="run-1")
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
        mock_mlflow.set_experiment.assert_called_once_with("test-exp")
        mock_mlflow.start_run.assert_called_once_with(run_name="run-1")

    def test_import_error_when_no_mlflow(self):
        with patch.dict("sys.modules", {"mlflow": None}):
            with pytest.raises(ImportError, match="mlflow not installed"):
                MLflowTracker(tracking_uri="", experiment_name="e", run_name="r")


# ============================================================
# WandbTracker
# ============================================================
class TestWandbTracker:
    @patch.dict("sys.modules", {"wandb": MagicMock()})
    def test_is_metrics_tracker(self):
        t = WandbTracker.__new__(WandbTracker)
        t._wandb = MagicMock()
        t._run = MagicMock()
        assert isinstance(t, MetricsTracker)

    @patch.dict("sys.modules", {"wandb": MagicMock()})
    def test_log_metrics_delegates(self):
        mock_wandb = MagicMock()
        t = WandbTracker.__new__(WandbTracker)
        t._wandb = mock_wandb
        t._run = MagicMock()
        t.log_metrics({"loss": 0.5}, step=10)
        mock_wandb.log.assert_called_once_with({"loss": 0.5, "global_step": 10})

    @patch.dict("sys.modules", {"wandb": MagicMock()})
    def test_log_metrics_no_step(self):
        mock_wandb = MagicMock()
        t = WandbTracker.__new__(WandbTracker)
        t._wandb = mock_wandb
        t._run = MagicMock()
        t.log_metrics({"loss": 0.5})
        mock_wandb.log.assert_called_once_with({"loss": 0.5})

    @patch.dict("sys.modules", {"wandb": MagicMock()})
    def test_log_params_updates_config(self):
        mock_run = MagicMock()
        t = WandbTracker.__new__(WandbTracker)
        t._wandb = MagicMock()
        t._run = mock_run
        t.log_params({"lr": 1e-5})
        mock_run.config.update.assert_called_once_with({"lr": 1e-5})

    @patch.dict("sys.modules", {"wandb": MagicMock()})
    def test_end_run_calls_finish(self):
        mock_wandb = MagicMock()
        t = WandbTracker.__new__(WandbTracker)
        t._wandb = mock_wandb
        t._run = MagicMock()
        t.end_run()
        mock_wandb.finish.assert_called_once()

    @patch.dict("sys.modules", {"wandb": MagicMock()})
    def test_init_calls_wandb_init(self):
        import sys as _sys
        mock_wandb = _sys.modules["wandb"]
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        t = WandbTracker(project="proj", name="run-1", config={"lr": 1e-5})
        mock_wandb.init.assert_called_once_with(
            project="proj", name="run-1", config={"lr": 1e-5}
        )

    def test_import_error_when_no_wandb(self):
        with patch.dict("sys.modules", {"wandb": None}):
            with pytest.raises(ImportError, match="wandb not installed"):
                WandbTracker(project="p", name="n")


# ============================================================
# TensorBoardTracker
# ============================================================
class TestTensorBoardTracker:
    @patch("torch.utils.tensorboard.SummaryWriter")
    def test_is_metrics_tracker(self, mock_sw):
        t = TensorBoardTracker(log_dir="/tmp/tb")
        assert isinstance(t, MetricsTracker)

    @patch("torch.utils.tensorboard.SummaryWriter")
    def test_log_metrics_calls_add_scalar(self, mock_sw_cls):
        mock_writer = MagicMock()
        mock_sw_cls.return_value = mock_writer
        t = TensorBoardTracker(log_dir="/tmp/tb")
        t.log_metrics({"train/loss": 0.5, "train/acc": 0.9}, step=5)
        assert mock_writer.add_scalar.call_count == 2
        mock_writer.add_scalar.assert_any_call("train/loss", 0.5, global_step=5)
        mock_writer.add_scalar.assert_any_call("train/acc", 0.9, global_step=5)
        mock_writer.flush.assert_called()

    @patch("torch.utils.tensorboard.SummaryWriter")
    def test_log_params_calls_add_text(self, mock_sw_cls):
        mock_writer = MagicMock()
        mock_sw_cls.return_value = mock_writer
        t = TensorBoardTracker(log_dir="/tmp/tb")
        t.log_params({"lr": 1e-5})
        mock_writer.add_text.assert_called_once()
        call_args = mock_writer.add_text.call_args
        assert call_args[0][0] == "params"
        assert "lr" in call_args[0][1]

    @patch("torch.utils.tensorboard.SummaryWriter")
    def test_end_run_closes_writer(self, mock_sw_cls):
        mock_writer = MagicMock()
        mock_sw_cls.return_value = mock_writer
        t = TensorBoardTracker(log_dir="/tmp/tb")
        t.end_run()
        mock_writer.close.assert_called_once()

    @patch("torch.utils.tensorboard.SummaryWriter")
    def test_flush_on_log_metrics(self, mock_sw_cls):
        mock_writer = MagicMock()
        mock_sw_cls.return_value = mock_writer
        t = TensorBoardTracker(log_dir="/tmp/tb")
        t.log_metrics({"x": 1.0})
        mock_writer.flush.assert_called()

    @patch("torch.utils.tensorboard.SummaryWriter")
    def test_empty_metrics(self, mock_sw_cls):
        mock_writer = MagicMock()
        mock_sw_cls.return_value = mock_writer
        t = TensorBoardTracker(log_dir="/tmp/tb")
        t.log_metrics({}, step=1)
        mock_writer.add_scalar.assert_not_called()
        mock_writer.flush.assert_called()


# ============================================================
# create_tracker factory
# ============================================================
class TestCreateTracker:
    def _make_config(self, tracker="mlflow"):
        config = MagicMock()
        config.run.tracker = tracker
        config.run.project_name = "test-project"
        config.run.experiment_id = "exp-1"
        config.run.tracking_uri = ""
        config.run.checkpoint_dir = "/tmp/checkpoints"
        config.model_dump.return_value = {}
        return config

    def test_non_rank_zero_returns_noop(self):
        config = self._make_config()
        tracker = create_tracker(config, rank=1)
        assert isinstance(tracker, NoOpTracker)

    def test_non_rank_zero_any_tracker_returns_noop(self):
        for name in ["mlflow", "wandb", "tensorboard", "none"]:
            config = self._make_config(tracker=name)
            tracker = create_tracker(config, rank=1)
            assert isinstance(tracker, NoOpTracker)

    def test_none_tracker_returns_noop(self):
        config = self._make_config(tracker="none")
        tracker = create_tracker(config, rank=0)
        assert isinstance(tracker, NoOpTracker)

    @patch.dict("sys.modules", {"mlflow": MagicMock()})
    def test_mlflow_tracker(self):
        import sys as _sys
        _sys.modules["mlflow"].start_run.return_value = MagicMock()
        config = self._make_config(tracker="mlflow")
        tracker = create_tracker(config, rank=0)
        assert isinstance(tracker, MLflowTracker)

    def test_mlflow_fallback_to_noop(self):
        """When mlflow is not installed, fall back to NoOpTracker."""
        with patch.dict("sys.modules", {"mlflow": None}):
            config = self._make_config(tracker="mlflow")
            tracker = create_tracker(config, rank=0)
            assert isinstance(tracker, NoOpTracker)

    @patch.dict("sys.modules", {"wandb": MagicMock()})
    def test_wandb_tracker(self):
        import sys as _sys
        mock_wandb = _sys.modules["wandb"]
        mock_wandb.init.return_value = MagicMock()
        config = self._make_config(tracker="wandb")
        tracker = create_tracker(config, rank=0)
        assert isinstance(tracker, WandbTracker)

    @patch("torch.utils.tensorboard.SummaryWriter")
    def test_tensorboard_tracker(self, mock_sw):
        config = self._make_config(tracker="tensorboard")
        tracker = create_tracker(config, rank=0)
        assert isinstance(tracker, TensorBoardTracker)

    def test_unknown_tracker_raises(self):
        config = self._make_config(tracker="prometheus")
        with pytest.raises(ValueError, match="Unknown tracker 'prometheus'"):
            create_tracker(config, rank=0)

    def test_tensorboard_log_dir_uses_checkpoint_dir(self):
        with patch("torch.utils.tensorboard.SummaryWriter") as mock_sw:
            config = self._make_config(tracker="tensorboard")
            config.run.checkpoint_dir = "/my/checkpoints"
            create_tracker(config, rank=0)
            call_args = mock_sw.call_args
            assert "/my/checkpoints" in call_args[1]["log_dir"]

    def test_tensorboard_log_dir_fallback_runs(self):
        with patch("torch.utils.tensorboard.SummaryWriter") as mock_sw:
            config = self._make_config(tracker="tensorboard")
            config.run.checkpoint_dir = None
            create_tracker(config, rank=0)
            call_args = mock_sw.call_args
            assert "runs" in call_args[1]["log_dir"]


# ============================================================
# Backward compatibility: logging.py integration
# ============================================================
class TestLoggingBackwardCompat:
    def test_setup_tracker_returns_tracker(self):
        from oxrl.utils.logging import setup_tracker, get_tracker
        config = MagicMock()
        config.run.tracker = "none"
        config.run.project_name = "test"
        config.run.experiment_id = "e1"
        config.run.tracking_uri = ""
        config.run.method = "sl"
        config.train.alg_name = "sft"
        config.model.name = "m"
        config.train.lr = 1e-5
        config.train.train_batch_size_per_gpu = 2
        config.train.gradient_accumulation_steps = 1
        config.train.total_number_of_epochs = 1
        config.run.seed = 42
        config.train.micro_batches_per_epoch = 100
        tracker = setup_tracker(config=config, rank=0)
        assert isinstance(tracker, NoOpTracker)
        assert isinstance(get_tracker(), NoOpTracker)

    def test_setup_tracker_non_rank_zero(self):
        from oxrl.utils.logging import setup_tracker
        config = MagicMock()
        config.run.tracker = "mlflow"
        config.run.project_name = "p"
        config.run.experiment_id = "e"
        config.run.tracking_uri = ""
        config.run.method = "sl"
        config.train.alg_name = "sft"
        config.model.name = "m"
        config.train.lr = 1e-5
        config.train.train_batch_size_per_gpu = 2
        config.train.gradient_accumulation_steps = 1
        config.train.total_number_of_epochs = 1
        config.run.seed = 42
        config.train.micro_batches_per_epoch = 100
        tracker = setup_tracker(config=config, rank=1)
        assert isinstance(tracker, NoOpTracker)

    def test_log_metrics_delegates_to_tracker(self):
        from oxrl.utils import logging as log_mod
        mock_tracker = MagicMock(spec=MetricsTracker)
        original = log_mod._tracker
        try:
            log_mod._tracker = mock_tracker
            log_mod.log_metrics({"loss": 0.5}, step=1)
            mock_tracker.log_metrics.assert_called_once_with({"loss": 0.5}, step=1)
        finally:
            log_mod._tracker = original

    def test_end_run_delegates_to_tracker(self):
        from oxrl.utils import logging as log_mod
        mock_tracker = MagicMock(spec=MetricsTracker)
        original = log_mod._tracker
        try:
            log_mod._tracker = mock_tracker
            log_mod.end_run()
            mock_tracker.end_run.assert_called_once()
        finally:
            log_mod._tracker = original

    def test_setup_logging_returns_logger(self):
        from oxrl.utils.logging import setup_logging
        import logging
        logger = setup_logging(rank=0, log_level="WARNING", exp_name="test_compat")
        assert isinstance(logger, logging.Logger)

    def test_get_tracker_default_is_noop(self):
        """Before any setup, the default tracker should be NoOp."""
        from oxrl.utils.metrics import NoOpTracker
        from oxrl.utils import logging as log_mod
        # Reset to default
        original = log_mod._tracker
        log_mod._tracker = NoOpTracker()
        try:
            assert isinstance(log_mod.get_tracker(), NoOpTracker)
        finally:
            log_mod._tracker = original
