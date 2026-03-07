"""
Comprehensive tests for DeepSpeed config sync functions in oxrl/configs/sync.py.
All tests run on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_config_sync.py -v
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oxrl.configs.schema import Config, DeepSpeedRef
from oxrl.configs.sync import (
    sync_deepspeed_config,
    _sync_batch_sizes,
    _sync_gradient_clipping,
    _sync_dtype,
    _sync_optimizer,
    _sync_scheduler,
    _sync_zero_defaults,
    _sync_ref_model_config,
)


def _make_config(**overrides):
    """Build a minimal valid Config for testing sync functions."""
    raw = {
        "run": {"experiment_id": "test"},
        "train": {"alg_name": "sft", "total_number_of_epochs": 1,
                  "micro_batches_per_epoch": 100,
                  "train_batch_size_per_gpu": 4,
                  "gradient_accumulation_steps": 2},
        "model": {"name": "m"},
        "data": {"train_dnames": ["d"], "train_ratios": {"d": 1.0},
                 "train_files_path": "/tmp/d", "val_files_path": "/tmp/v"},
    }
    for key, val in overrides.items():
        section, field = key.split(".", 1) if "." in key else (key, None)
        if field:
            raw[section][field] = val
        else:
            raw[key] = val
    return Config(**raw)


# ============================================================
# _sync_batch_sizes
# ============================================================
class TestSyncBatchSizes:
    def test_micro_batch_copied(self):
        cfg = _make_config()
        cfg.run.method = "sl"
        _sync_batch_sizes(cfg, world_size=4)
        assert cfg.deepspeed.train_micro_batch_size_per_gpu == 4

    def test_grad_accum_copied(self):
        cfg = _make_config()
        cfg.run.method = "sl"
        _sync_batch_sizes(cfg, world_size=4)
        assert cfg.deepspeed.gradient_accumulation_steps == 2

    def test_sl_train_batch_size(self):
        cfg = _make_config()
        cfg.run.method = "sl"
        _sync_batch_sizes(cfg, world_size=4)
        # train_batch_size = micro_batch * grad_accum * world_size = 4*2*4 = 32
        assert cfg.deepspeed.train_batch_size == 32

    def test_rl_no_train_batch_size(self):
        cfg = _make_config()
        cfg.run.method = "rl"
        _sync_batch_sizes(cfg, world_size=4)
        # RL does not set train_batch_size
        assert cfg.deepspeed.train_batch_size is None

    def test_sl_with_different_world_sizes(self):
        for ws in [1, 2, 8]:
            cfg = _make_config()
            cfg.run.method = "sl"
            _sync_batch_sizes(cfg, world_size=ws)
            expected = 4 * 2 * ws
            assert cfg.deepspeed.train_batch_size == expected

    def test_none_world_size_no_crash(self):
        cfg = _make_config()
        cfg.run.method = "rl"
        _sync_batch_sizes(cfg, world_size=None)
        assert cfg.deepspeed.train_batch_size is None


# ============================================================
# _sync_gradient_clipping
# ============================================================
class TestSyncGradientClipping:
    def test_default_clipping(self):
        cfg = _make_config()
        _sync_gradient_clipping(cfg)
        assert cfg.deepspeed.gradient_clipping == 1.0

    def test_custom_clipping(self):
        cfg = _make_config(**{"train.clip_grad_norm": 0.5})
        _sync_gradient_clipping(cfg)
        assert cfg.deepspeed.gradient_clipping == 0.5

    def test_clipping_is_float(self):
        cfg = _make_config()
        _sync_gradient_clipping(cfg)
        assert isinstance(cfg.deepspeed.gradient_clipping, float)


# ============================================================
# _sync_dtype
# ============================================================
class TestSyncDtype:
    def test_bfloat16(self):
        cfg = _make_config(**{"model.dtype": "bfloat16"})
        _sync_dtype(cfg)
        assert cfg.deepspeed.bf16["enabled"] is True
        assert cfg.deepspeed.fp16["enabled"] is False

    def test_bf16_short(self):
        cfg = _make_config(**{"model.dtype": "bf16"})
        _sync_dtype(cfg)
        assert cfg.deepspeed.bf16["enabled"] is True

    def test_float16(self):
        cfg = _make_config(**{"model.dtype": "float16"})
        _sync_dtype(cfg)
        assert cfg.deepspeed.fp16["enabled"] is True
        assert cfg.deepspeed.bf16["enabled"] is False

    def test_fp16_short(self):
        cfg = _make_config(**{"model.dtype": "fp16"})
        _sync_dtype(cfg)
        assert cfg.deepspeed.fp16["enabled"] is True

    def test_float32(self):
        cfg = _make_config(**{"model.dtype": "float32"})
        _sync_dtype(cfg)
        assert cfg.deepspeed.fp16["enabled"] is False
        assert cfg.deepspeed.bf16["enabled"] is False

    def test_case_insensitive(self):
        cfg = _make_config(**{"model.dtype": "BFloat16"})
        _sync_dtype(cfg)
        assert cfg.deepspeed.bf16["enabled"] is True


# ============================================================
# _sync_optimizer
# ============================================================
class TestSyncOptimizer:
    def test_adamw(self):
        cfg = _make_config(**{"train.optimizer_name": "adamw"})
        _sync_optimizer(cfg)
        assert cfg.deepspeed.optimizer["type"] == "AdamW"

    def test_adam(self):
        cfg = _make_config(**{"train.optimizer_name": "adam"})
        _sync_optimizer(cfg)
        assert cfg.deepspeed.optimizer["type"] == "Adam"

    def test_AdamW_mixed_case(self):
        cfg = _make_config(**{"train.optimizer_name": "AdamW"})
        _sync_optimizer(cfg)
        assert cfg.deepspeed.optimizer["type"] == "AdamW"

    def test_unsupported_optimizer_raises(self):
        cfg = _make_config(**{"train.optimizer_name": "sgd"})
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            _sync_optimizer(cfg)

    def test_optimizer_params(self):
        cfg = _make_config(**{"train.lr": 3e-4, "train.weight_decay": 0.05})
        _sync_optimizer(cfg)
        params = cfg.deepspeed.optimizer["params"]
        assert params["lr"] == 3e-4
        assert params["weight_decay"] == 0.05
        assert params["betas"] == [0.9, 0.95]
        assert params["eps"] == 1e-8


# ============================================================
# _sync_scheduler
# ============================================================
class TestSyncScheduler:
    def test_sl_scheduler_total_steps(self):
        cfg = _make_config()
        cfg.run.method = "sl"
        cfg.train.micro_batches_per_epoch = 100
        cfg.train.gradient_accumulation_steps = 2
        cfg.train.total_number_of_epochs = 5
        _sync_scheduler(cfg)
        # optimizer_steps_per_epoch = 100 // 2 = 50
        # total = 5 * 50 = 250
        assert cfg.deepspeed.scheduler["params"]["total_num_steps"] == 250

    def test_rl_scheduler_total_steps(self):
        cfg = _make_config()
        cfg.run.method = "rl"
        cfg.train.train_steps_per_epoch = 20
        cfg.train.total_number_of_epochs = 10
        _sync_scheduler(cfg)
        # total = 10 * 20 = 200
        assert cfg.deepspeed.scheduler["params"]["total_num_steps"] == 200

    def test_warmup_steps(self):
        cfg = _make_config()
        cfg.run.method = "rl"
        cfg.train.train_steps_per_epoch = 100
        cfg.train.total_number_of_epochs = 10
        cfg.train.warmup_steps_ratio = 0.1
        _sync_scheduler(cfg)
        # total = 1000, warmup = 100
        assert cfg.deepspeed.scheduler["params"]["warmup_num_steps"] == 100

    def test_sl_missing_micro_batches_raises(self):
        cfg = _make_config()
        cfg.run.method = "sl"
        cfg.train.micro_batches_per_epoch = None
        with pytest.raises(ValueError, match="micro_batches_per_epoch"):
            _sync_scheduler(cfg)

    def test_rl_missing_train_steps_raises(self):
        cfg = _make_config()
        cfg.run.method = "rl"
        cfg.train.train_steps_per_epoch = None
        with pytest.raises(ValueError, match="train_steps_per_epoch"):
            _sync_scheduler(cfg)

    def test_unsupported_scheduler_raises(self):
        cfg = _make_config()
        cfg.run.method = "sl"
        cfg.train.lr_scheduler = "StepLR"
        with pytest.raises(ValueError, match="Unsupported scheduler"):
            _sync_scheduler(cfg)

    def test_scheduler_type(self):
        cfg = _make_config()
        cfg.run.method = "sl"
        cfg.train.micro_batches_per_epoch = 100
        _sync_scheduler(cfg)
        assert cfg.deepspeed.scheduler["type"] == "WarmupCosineLR"

    def test_warmup_ratio_zero(self):
        cfg = _make_config()
        cfg.run.method = "rl"
        cfg.train.train_steps_per_epoch = 50
        cfg.train.total_number_of_epochs = 4
        cfg.train.warmup_steps_ratio = 0.0
        _sync_scheduler(cfg)
        assert cfg.deepspeed.scheduler["params"]["warmup_num_steps"] == 0


# ============================================================
# _sync_zero_defaults
# ============================================================
class TestSyncZeroDefaults:
    def test_removes_none_values(self):
        cfg = _make_config()
        cfg.deepspeed.zero_optimization["offload_param"] = None
        _sync_zero_defaults(cfg)
        assert "offload_param" not in cfg.deepspeed.zero_optimization

    def test_removes_device_none_offload(self):
        cfg = _make_config()
        cfg.deepspeed.zero_optimization["offload_optimizer"] = {"device": "none"}
        _sync_zero_defaults(cfg)
        assert "offload_optimizer" not in cfg.deepspeed.zero_optimization

    def test_stage3_gather_flag_added(self):
        cfg = _make_config()
        cfg.deepspeed.zero_optimization = {"stage": 3}
        _sync_zero_defaults(cfg)
        assert cfg.deepspeed.zero_optimization["stage3_gather_16bit_weights_on_model_save"] is True

    def test_stage3_gather_flag_not_overwritten(self):
        cfg = _make_config()
        cfg.deepspeed.zero_optimization = {
            "stage": 3,
            "stage3_gather_16bit_weights_on_model_save": False,
        }
        _sync_zero_defaults(cfg)
        assert cfg.deepspeed.zero_optimization["stage3_gather_16bit_weights_on_model_save"] is False

    def test_non_stage3_no_gather_flag(self):
        cfg = _make_config()
        cfg.deepspeed.zero_optimization = {"stage": 2}
        _sync_zero_defaults(cfg)
        assert "stage3_gather_16bit_weights_on_model_save" not in cfg.deepspeed.zero_optimization

    def test_none_zero_optimization_handled(self):
        cfg = _make_config()
        cfg.deepspeed.zero_optimization = None
        _sync_zero_defaults(cfg)
        assert cfg.deepspeed.zero_optimization == {}


# ============================================================
# _sync_ref_model_config
# ============================================================
class TestSyncRefModelConfig:
    def test_no_ref_model_no_deepspeed_ref(self):
        cfg = _make_config(**{"model.ref_model": ""})
        _sync_ref_model_config(cfg)
        assert cfg.deepspeed_ref is None

    def test_ref_model_creates_deepspeed_ref(self):
        cfg = _make_config(**{"model.ref_model": "some-model"})
        cfg.run.method = "sl"
        cfg.train.micro_batches_per_epoch = 100
        # First sync dtype and batch sizes so deepspeed has values
        _sync_batch_sizes(cfg, world_size=1)
        _sync_dtype(cfg)
        _sync_optimizer(cfg)
        _sync_scheduler(cfg)
        _sync_ref_model_config(cfg)
        assert cfg.deepspeed_ref is not None
        assert isinstance(cfg.deepspeed_ref, DeepSpeedRef)

    def test_ref_model_inherits_dtype(self):
        cfg = _make_config(**{"model.ref_model": "ref", "model.dtype": "bfloat16"})
        cfg.run.method = "sl"
        cfg.train.micro_batches_per_epoch = 100
        _sync_batch_sizes(cfg, world_size=1)
        _sync_dtype(cfg)
        _sync_optimizer(cfg)
        _sync_scheduler(cfg)
        _sync_ref_model_config(cfg)
        assert cfg.deepspeed_ref.bf16["enabled"] is True

    def test_ref_model_no_optimizer(self):
        cfg = _make_config(**{"model.ref_model": "ref"})
        cfg.run.method = "sl"
        cfg.train.micro_batches_per_epoch = 100
        _sync_batch_sizes(cfg, world_size=1)
        _sync_dtype(cfg)
        _sync_optimizer(cfg)
        _sync_scheduler(cfg)
        _sync_ref_model_config(cfg)
        # DeepSpeedRef has no optimizer field
        assert not hasattr(cfg.deepspeed_ref, "optimizer")

    def test_ref_model_offload_to_cpu(self):
        cfg = _make_config(**{"model.ref_model": "ref", "model.ref_model_offload_to_cpu": True})
        cfg.run.method = "sl"
        cfg.train.micro_batches_per_epoch = 100
        _sync_batch_sizes(cfg, world_size=1)
        _sync_dtype(cfg)
        _sync_optimizer(cfg)
        _sync_scheduler(cfg)
        _sync_ref_model_config(cfg)
        zo = cfg.deepspeed_ref.zero_optimization
        assert zo.get("offload_param", {}).get("device") == "cpu"

    def test_ref_model_no_offload(self):
        cfg = _make_config(**{"model.ref_model": "ref", "model.ref_model_offload_to_cpu": False})
        cfg.run.method = "sl"
        cfg.train.micro_batches_per_epoch = 100
        _sync_batch_sizes(cfg, world_size=1)
        _sync_dtype(cfg)
        _sync_optimizer(cfg)
        _sync_scheduler(cfg)
        _sync_ref_model_config(cfg)
        zo = cfg.deepspeed_ref.zero_optimization
        assert "offload_param" not in zo

    def test_existing_deepspeed_ref_not_overwritten(self):
        cfg = _make_config(**{"model.ref_model": "ref"})
        cfg.deepspeed_ref = DeepSpeedRef(bf16={"enabled": True})
        _sync_ref_model_config(cfg)
        # Should not be overwritten since deepspeed_ref already exists
        assert cfg.deepspeed_ref.bf16 == {"enabled": True}


# ============================================================
# Full sync_deepspeed_config integration
# ============================================================
class TestSyncIntegration:
    def test_full_sl_sync(self):
        cfg = _make_config()
        cfg.run.method = "sl"
        cfg.train.micro_batches_per_epoch = 100
        sync_deepspeed_config(cfg, world_size=4)
        assert cfg.deepspeed.train_batch_size == 32
        assert cfg.deepspeed.gradient_clipping == 1.0
        assert cfg.deepspeed.bf16["enabled"] is True
        assert cfg.deepspeed.optimizer is not None
        assert cfg.deepspeed.scheduler is not None

    def test_full_rl_sync(self):
        cfg = _make_config()
        cfg.run.method = "rl"
        cfg.train.train_steps_per_epoch = 20
        sync_deepspeed_config(cfg, world_size=2)
        assert cfg.deepspeed.train_micro_batch_size_per_gpu == 4
        assert cfg.deepspeed.optimizer["type"] == "AdamW"

    def test_idempotent(self):
        cfg = _make_config()
        cfg.run.method = "rl"
        cfg.train.train_steps_per_epoch = 10
        sync_deepspeed_config(cfg, world_size=2)
        ds1 = cfg.deepspeed.model_dump()
        sync_deepspeed_config(cfg, world_size=2)
        ds2 = cfg.deepspeed.model_dump()
        assert ds1 == ds2

    def test_ref_model_integration(self):
        cfg = _make_config(**{"model.ref_model": "ref-model"})
        cfg.run.method = "rl"
        cfg.train.train_steps_per_epoch = 10
        sync_deepspeed_config(cfg, world_size=2)
        assert cfg.deepspeed_ref is not None
        assert cfg.deepspeed_ref.bf16["enabled"] is True
