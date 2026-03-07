"""
Tests for config loading via load_and_verify() in oxrl/configs/loader.py.
All tests run on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_config_load.py -v
"""
import pytest
import sys
import os
import tempfile
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oxrl.configs.loader import load_and_verify


def _write_yaml(data, tmpdir):
    path = os.path.join(tmpdir, "config.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


def _minimal_config():
    return {
        "run": {"experiment_id": "placeholder"},
        "train": {"alg_name": "sft", "total_number_of_epochs": 1,
                  "micro_batches_per_epoch": 100},
        "model": {"name": "m"},
        "data": {"train_dnames": ["d"], "train_ratios": {"d": 1.0},
                 "train_files_path": "/tmp/d", "val_files_path": "/tmp/v"},
    }


class TestLoadAndVerify:
    def test_valid_sl_config(self, tmp_path):
        path = _write_yaml(_minimal_config(), str(tmp_path))
        config = load_and_verify(method="sl", input_yaml=path,
                                 experiment_id="exp1", world_size=4)
        assert config.run.experiment_id == "exp1"
        assert config.run.method == "sl"

    def test_valid_rl_config(self, tmp_path):
        raw = _minimal_config()
        raw["train"]["train_steps_per_epoch"] = 20
        raw["run"]["training_gpus"] = 2
        path = _write_yaml(raw, str(tmp_path))
        config = load_and_verify(method="rl", input_yaml=path,
                                 experiment_id="exp2")
        assert config.run.experiment_id == "exp2"
        assert config.run.method == "rl"

    def test_experiment_id_override(self, tmp_path):
        path = _write_yaml(_minimal_config(), str(tmp_path))
        config = load_and_verify(method="sl", input_yaml=path,
                                 experiment_id="override_id", world_size=1)
        assert config.run.experiment_id == "override_id"

    def test_method_override(self, tmp_path):
        path = _write_yaml(_minimal_config(), str(tmp_path))
        config = load_and_verify(method="sl", input_yaml=path,
                                 experiment_id="e", world_size=1)
        assert config.run.method == "sl"

    def test_file_not_found(self):
        with pytest.raises(SystemExit):
            load_and_verify(method="sl", input_yaml="/nonexistent/path.yaml",
                            experiment_id="e", world_size=1)

    def test_invalid_yaml(self, tmp_path):
        path = os.path.join(str(tmp_path), "bad.yaml")
        with open(path, "w") as f:
            f.write(":\n  - :\n  bad: [")
        with pytest.raises(SystemExit):
            load_and_verify(method="sl", input_yaml=path,
                            experiment_id="e", world_size=1)

    def test_validation_error_extra_field(self, tmp_path):
        raw = _minimal_config()
        raw["run"]["not_a_real_field"] = "oops"
        path = _write_yaml(raw, str(tmp_path))
        with pytest.raises(SystemExit):
            load_and_verify(method="sl", input_yaml=path,
                            experiment_id="e", world_size=1)

    def test_validation_error_missing_required(self, tmp_path):
        raw = _minimal_config()
        del raw["model"]["name"]
        path = _write_yaml(raw, str(tmp_path))
        with pytest.raises(SystemExit):
            load_and_verify(method="sl", input_yaml=path,
                            experiment_id="e", world_size=1)

    def test_deepspeed_sync_after_load(self, tmp_path):
        path = _write_yaml(_minimal_config(), str(tmp_path))
        config = load_and_verify(method="sl", input_yaml=path,
                                 experiment_id="e", world_size=2)
        assert config.deepspeed.optimizer is not None
        assert config.deepspeed.scheduler is not None
        assert config.deepspeed.bf16["enabled"] is True

    def test_sl_missing_world_size_raises(self, tmp_path):
        path = _write_yaml(_minimal_config(), str(tmp_path))
        with pytest.raises(ValueError, match="world_size must be specified"):
            load_and_verify(method="sl", input_yaml=path,
                            experiment_id="e", world_size=None)
