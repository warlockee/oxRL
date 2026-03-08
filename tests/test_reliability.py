"""
Tests for distributed reliability features:
- cleanup_old_checkpoints
- ray_get_with_timeout
- Epoch tag parsing (resume logic)
- Config schema reliability fields

Run with: pytest tests/test_reliability.py -v
"""
import os
import re
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oxrl.configs.schema import Run
from oxrl.tools.checkpoint import cleanup_old_checkpoints
from oxrl.utils.ray_utils import ray_get_with_timeout


# ============================================================
# Config schema — reliability fields
# ============================================================
class TestReliabilityConfigDefaults:
    """All new fields must default to backward-compatible values."""

    def test_resume_from_default_none(self):
        r = Run(experiment_id="exp1")
        assert r.resume_from is None

    def test_checkpoint_every_n_epochs_default_1(self):
        r = Run(experiment_id="exp1")
        assert r.checkpoint_every_n_epochs == 1

    def test_keep_last_n_checkpoints_default_none(self):
        r = Run(experiment_id="exp1")
        assert r.keep_last_n_checkpoints is None

    def test_save_best_checkpoint_default_false(self):
        r = Run(experiment_id="exp1")
        assert r.save_best_checkpoint is False

    def test_ray_task_timeout_sec_default_1800(self):
        r = Run(experiment_id="exp1")
        assert r.ray_task_timeout_sec == 1800

    def test_max_epoch_retries_default_0(self):
        r = Run(experiment_id="exp1")
        assert r.max_epoch_retries == 0


class TestReliabilityConfigCustom:
    """New fields accept custom values."""

    def test_resume_from_custom(self):
        r = Run(experiment_id="exp1", resume_from="/path/to/ckpt")
        assert r.resume_from == "/path/to/ckpt"

    def test_checkpoint_every_n_epochs_custom(self):
        r = Run(experiment_id="exp1", checkpoint_every_n_epochs=5)
        assert r.checkpoint_every_n_epochs == 5

    def test_keep_last_n_checkpoints_custom(self):
        r = Run(experiment_id="exp1", keep_last_n_checkpoints=3)
        assert r.keep_last_n_checkpoints == 3

    def test_save_best_checkpoint_custom(self):
        r = Run(experiment_id="exp1", save_best_checkpoint=True)
        assert r.save_best_checkpoint is True

    def test_ray_task_timeout_sec_custom(self):
        r = Run(experiment_id="exp1", ray_task_timeout_sec=600)
        assert r.ray_task_timeout_sec == 600

    def test_ray_task_timeout_sec_zero(self):
        r = Run(experiment_id="exp1", ray_task_timeout_sec=0)
        assert r.ray_task_timeout_sec == 0

    def test_max_epoch_retries_custom(self):
        r = Run(experiment_id="exp1", max_epoch_retries=3)
        assert r.max_epoch_retries == 3

    def test_all_reliability_fields_together(self):
        r = Run(
            experiment_id="exp1",
            resume_from="/ckpts/iter000010_v000010",
            checkpoint_every_n_epochs=2,
            keep_last_n_checkpoints=5,
            save_best_checkpoint=True,
            ray_task_timeout_sec=3600,
            max_epoch_retries=2,
        )
        assert r.resume_from == "/ckpts/iter000010_v000010"
        assert r.checkpoint_every_n_epochs == 2
        assert r.keep_last_n_checkpoints == 5
        assert r.save_best_checkpoint is True
        assert r.ray_task_timeout_sec == 3600
        assert r.max_epoch_retries == 2


class TestReliabilityConfigBackwardCompat:
    """Old configs without new fields still work."""

    def test_old_style_config_works(self):
        r = Run(experiment_id="exp1", seed=123, project_name="my-proj")
        # All reliability fields should be at defaults
        assert r.resume_from is None
        assert r.checkpoint_every_n_epochs == 1
        assert r.keep_last_n_checkpoints is None
        assert r.save_best_checkpoint is False
        assert r.ray_task_timeout_sec == 1800
        assert r.max_epoch_retries == 0
        # Original fields untouched
        assert r.seed == 123
        assert r.project_name == "my-proj"


# ============================================================
# cleanup_old_checkpoints
# ============================================================
class TestCleanupOldCheckpoints:
    """Test checkpoint directory cleanup logic."""

    def test_keeps_last_n(self, tmp_path):
        exp_dir = tmp_path / "checkpoints" / "exp1"
        exp_dir.mkdir(parents=True)
        # Create 5 checkpoint dirs
        for i in range(1, 6):
            (exp_dir / f"iter{i:06d}_v{i:06d}").mkdir()

        cleanup_old_checkpoints(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            experiment_id="exp1",
            keep_last_n=2,
        )

        remaining = sorted(os.listdir(exp_dir))
        assert remaining == ["iter000004_v000004", "iter000005_v000005"]

    def test_keeps_all_when_fewer_than_n(self, tmp_path):
        exp_dir = tmp_path / "checkpoints" / "exp1"
        exp_dir.mkdir(parents=True)
        (exp_dir / "iter000001").mkdir()
        (exp_dir / "iter000002").mkdir()

        cleanup_old_checkpoints(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            experiment_id="exp1",
            keep_last_n=5,
        )

        remaining = sorted(os.listdir(exp_dir))
        assert remaining == ["iter000001", "iter000002"]

    def test_excludes_tags(self, tmp_path):
        exp_dir = tmp_path / "checkpoints" / "exp1"
        exp_dir.mkdir(parents=True)
        for i in range(1, 6):
            (exp_dir / f"iter{i:06d}").mkdir()
        (exp_dir / "best").mkdir()

        cleanup_old_checkpoints(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            experiment_id="exp1",
            keep_last_n=2,
            exclude_tags=["best"],
        )

        remaining = sorted(os.listdir(exp_dir))
        assert "best" in remaining
        assert "iter000004" in remaining
        assert "iter000005" in remaining
        assert len(remaining) == 3

    def test_handles_empty_dir(self, tmp_path):
        exp_dir = tmp_path / "checkpoints" / "exp1"
        exp_dir.mkdir(parents=True)

        # Should not raise
        cleanup_old_checkpoints(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            experiment_id="exp1",
            keep_last_n=3,
        )
        assert os.listdir(exp_dir) == []

    def test_handles_nonexistent_dir(self, tmp_path):
        # Should not raise
        cleanup_old_checkpoints(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            experiment_id="nonexistent",
            keep_last_n=3,
        )

    def test_keep_last_n_zero_removes_all(self, tmp_path):
        exp_dir = tmp_path / "checkpoints" / "exp1"
        exp_dir.mkdir(parents=True)
        for i in range(1, 4):
            (exp_dir / f"iter{i:06d}").mkdir()

        cleanup_old_checkpoints(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            experiment_id="exp1",
            keep_last_n=0,
        )

        remaining = os.listdir(exp_dir)
        assert remaining == []

    def test_ignores_non_iter_dirs(self, tmp_path):
        exp_dir = tmp_path / "checkpoints" / "exp1"
        exp_dir.mkdir(parents=True)
        (exp_dir / "iter000001").mkdir()
        (exp_dir / "iter000002").mkdir()
        (exp_dir / "iter000003").mkdir()
        (exp_dir / "some_other_dir").mkdir()
        (exp_dir / "logs").mkdir()

        cleanup_old_checkpoints(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            experiment_id="exp1",
            keep_last_n=1,
        )

        remaining = sorted(os.listdir(exp_dir))
        assert "iter000003" in remaining
        assert "some_other_dir" in remaining
        assert "logs" in remaining
        assert "iter000001" not in remaining
        assert "iter000002" not in remaining

    def test_sorts_by_epoch_number_not_lexically(self, tmp_path):
        """Epoch 10 > epoch 9, even though '10' < '9' lexicographically."""
        exp_dir = tmp_path / "checkpoints" / "exp1"
        exp_dir.mkdir(parents=True)
        for i in [1, 2, 9, 10, 11]:
            (exp_dir / f"iter{i:06d}").mkdir()

        cleanup_old_checkpoints(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            experiment_id="exp1",
            keep_last_n=2,
        )

        remaining = sorted(os.listdir(exp_dir))
        assert remaining == ["iter000010", "iter000011"]

    def test_files_in_checkpoint_dirs_are_removed(self, tmp_path):
        """Checkpoint dirs may contain files — rmtree should handle them."""
        exp_dir = tmp_path / "checkpoints" / "exp1"
        exp_dir.mkdir(parents=True)
        for i in range(1, 4):
            d = exp_dir / f"iter{i:06d}"
            d.mkdir()
            (d / "model.safetensors").write_text("dummy")
            (d / "config.json").write_text("{}")

        cleanup_old_checkpoints(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            experiment_id="exp1",
            keep_last_n=1,
        )

        remaining = os.listdir(exp_dir)
        assert remaining == ["iter000003"]
        assert (exp_dir / "iter000003" / "model.safetensors").exists()


# ============================================================
# ray_get_with_timeout
# ============================================================
class TestRayGetWithTimeout:
    """Test the ray_get_with_timeout wrapper.

    ray is imported lazily inside the function, so we patch sys.modules["ray"].
    """

    def _make_mock_ray(self):
        mock_ray = MagicMock()
        mock_ray.exceptions.GetTimeoutError = type("GetTimeoutError", (Exception,), {})
        return mock_ray

    def test_timeout_zero_passes_through(self):
        """timeout_sec=0 should call ray.get() without timeout."""
        mock_ray = self._make_mock_ray()
        mock_ray.get.return_value = [{"loss": 0.5}]

        with patch.dict("sys.modules", {"ray": mock_ray}):
            result = ray_get_with_timeout(["future1"], timeout_sec=0)

        mock_ray.get.assert_called_once_with(["future1"])
        assert result == [{"loss": 0.5}]

    def test_timeout_positive_passes_timeout_to_ray(self):
        """timeout_sec>0 should pass timeout= to ray.get()."""
        mock_ray = self._make_mock_ray()
        mock_ray.get.return_value = [{"loss": 0.5}]

        with patch.dict("sys.modules", {"ray": mock_ray}):
            result = ray_get_with_timeout(["future1"], timeout_sec=300)

        mock_ray.get.assert_called_once_with(["future1"], timeout=300)
        assert result == [{"loss": 0.5}]

    def test_timeout_raises_timeout_error(self):
        """GetTimeoutError should be converted to TimeoutError."""
        mock_ray = self._make_mock_ray()
        mock_ray.get.side_effect = mock_ray.exceptions.GetTimeoutError("timed out")

        with patch.dict("sys.modules", {"ray": mock_ray}):
            with pytest.raises(TimeoutError, match="timed out after 60s"):
                ray_get_with_timeout(["f1", "f2"], timeout_sec=60, description="test op")

    def test_timeout_error_includes_description(self):
        """Error message should include the description."""
        mock_ray = self._make_mock_ray()
        mock_ray.get.side_effect = mock_ray.exceptions.GetTimeoutError("timed out")

        with patch.dict("sys.modules", {"ray": mock_ray}):
            with pytest.raises(TimeoutError, match="rollout generation"):
                ray_get_with_timeout(["f1"], timeout_sec=30, description="rollout generation")

    def test_timeout_error_includes_pending_count(self):
        """Error message should include the number of pending futures."""
        mock_ray = self._make_mock_ray()
        mock_ray.get.side_effect = mock_ray.exceptions.GetTimeoutError("timed out")

        with patch.dict("sys.modules", {"ray": mock_ray}):
            with pytest.raises(TimeoutError, match="Pending futures: 3"):
                ray_get_with_timeout(["f1", "f2", "f3"], timeout_sec=10)

    def test_single_future(self):
        """Should work with a single future (not a list)."""
        mock_ray = self._make_mock_ray()
        mock_ray.get.return_value = {"loss": 0.1}

        with patch.dict("sys.modules", {"ray": mock_ray}):
            result = ray_get_with_timeout("single_future", timeout_sec=0)

        mock_ray.get.assert_called_once_with("single_future")
        assert result == {"loss": 0.1}

    def test_single_future_timeout_pending_count(self):
        """Non-list future should report 1 pending future."""
        mock_ray = self._make_mock_ray()
        mock_ray.get.side_effect = mock_ray.exceptions.GetTimeoutError("timed out")

        with patch.dict("sys.modules", {"ray": mock_ray}):
            with pytest.raises(TimeoutError, match="Pending futures: 1"):
                ray_get_with_timeout("single_future", timeout_sec=10)

    def test_non_timeout_exception_propagates(self):
        """Non-timeout exceptions should propagate unchanged."""
        mock_ray = self._make_mock_ray()
        mock_ray.get.side_effect = RuntimeError("worker crashed")

        with patch.dict("sys.modules", {"ray": mock_ray}):
            with pytest.raises(RuntimeError, match="worker crashed"):
                ray_get_with_timeout(["f1"], timeout_sec=60)


# ============================================================
# Epoch tag parsing (regex used in main_rl.py and main_sl.py)
# ============================================================
class TestEpochTagParsing:
    """Test the regex patterns used for checkpoint resume."""

    def test_rl_tag_with_version(self):
        """RL tags: iter000003_v000003 -> epoch=3, version=3."""
        tag = "iter000003_v000003"
        match = re.match(r"iter(\d+)_v(\d+)", tag)
        assert match is not None
        assert int(match.group(1)) == 3
        assert int(match.group(2)) == 3

    def test_rl_tag_large_numbers(self):
        tag = "iter000100_v000200"
        match = re.match(r"iter(\d+)_v(\d+)", tag)
        assert match is not None
        assert int(match.group(1)) == 100
        assert int(match.group(2)) == 200

    def test_sl_tag_iter_only(self):
        """SL tags: iter000005 -> epoch=5."""
        tag = "iter000005"
        match = re.match(r"iter(\d+)", tag)
        assert match is not None
        assert int(match.group(1)) == 5

    def test_sl_tag_from_path(self):
        """os.path.basename extracts tag from full path."""
        path = "/checkpoints/exp1/iter000010"
        tag = os.path.basename(path)
        match = re.match(r"iter(\d+)", tag)
        assert match is not None
        assert int(match.group(1)) == 10

    def test_rl_tag_from_path(self):
        path = "/checkpoints/exp1/iter000010_v000015"
        tag = os.path.basename(path)
        match = re.match(r"iter(\d+)_v(\d+)", tag)
        assert match is not None
        assert int(match.group(1)) == 10
        assert int(match.group(2)) == 15

    def test_best_tag_no_match(self):
        """'best' tag should not match iter pattern."""
        tag = "best"
        match = re.match(r"iter(\d+)", tag)
        assert match is None

    def test_malformed_tag_no_match(self):
        tag = "checkpoint_latest"
        match = re.match(r"iter(\d+)", tag)
        assert match is None

    def test_rl_fallback_to_iter_only(self):
        """RL resume: if iter+version fails, fall back to iter-only."""
        tag = "iter000005"
        match = re.match(r"iter(\d+)_v(\d+)", tag)
        if match:
            epoch = int(match.group(1))
            version = int(match.group(2))
        else:
            match = re.match(r"iter(\d+)", tag)
            assert match is not None
            epoch = int(match.group(1))
            version = 0

        assert epoch == 5
        assert version == 0


# ============================================================
# Checkpoint frequency logic
# ============================================================
class TestCheckpointFrequency:
    """Test the checkpoint frequency condition used in both main files."""

    @pytest.mark.parametrize("epoch,freq,total,expected", [
        (0, 1, 10, True),    # Every epoch
        (1, 1, 10, True),
        (0, 5, 10, False),   # Every 5 — epoch 1 (0-based) is not 5th
        (4, 5, 10, True),    # epoch+1=5, divisible by 5
        (9, 5, 10, True),    # Last epoch — always save
        (2, 3, 10, True),    # epoch+1=3, divisible by 3
        (5, 3, 10, True),    # epoch+1=6, divisible by 3
        (3, 3, 10, False),   # epoch+1=4, not divisible by 3
    ])
    def test_is_checkpoint_epoch(self, epoch, freq, total, expected):
        is_checkpoint = (epoch + 1) % freq == 0 or (epoch + 1) == total
        assert is_checkpoint == expected, (
            f"epoch={epoch}, freq={freq}, total={total}: expected {expected}, got {is_checkpoint}"
        )
