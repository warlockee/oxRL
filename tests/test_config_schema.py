"""
Comprehensive tests for all Pydantic config models in oxrl/configs/schema.py.
All tests run on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_config_schema.py -v
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pydantic import ValidationError
from oxrl.configs.schema import (
    Run, Train, Data, Model, DeepSpeed, DeepSpeedRef,
    InferenceEngine, Lora, Reward, Rollout, Config,
)


# ============================================================
# Run model
# ============================================================
class TestRun:
    def test_valid_construction(self):
        r = Run(experiment_id="exp1")
        assert r.experiment_id == "exp1"
        assert r.seed == 42
        assert r.distributed_training_strategy == "deepspeed-zero3"
        assert r.project_name == "oxrl-exp"

    def test_missing_experiment_id(self):
        with pytest.raises(ValidationError):
            Run()

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            Run(experiment_id="exp1", unknown_field="x")

    def test_custom_values(self):
        r = Run(experiment_id="e2", seed=99, project_name="my-project",
                training_gpus=4, rollout_gpus=2, ray_address="auto")
        assert r.seed == 99
        assert r.project_name == "my-project"
        assert r.training_gpus == 4
        assert r.rollout_gpus == 2
        assert r.ray_address == "auto"

    def test_method_default_none(self):
        r = Run(experiment_id="exp1")
        assert r.method is None

    def test_checkpoint_dir_default_none(self):
        r = Run(experiment_id="exp1")
        assert r.checkpoint_dir is None

    def test_tracker_default_mlflow(self):
        r = Run(experiment_id="exp1")
        assert r.tracker == "mlflow"

    def test_tracker_custom_values(self):
        for val in ["wandb", "tensorboard", "none"]:
            r = Run(experiment_id="exp1", tracker=val)
            assert r.tracker == val

    # ---- Reliability / fault-tolerance fields ----
    def test_resume_from_default(self):
        r = Run(experiment_id="exp1")
        assert r.resume_from is None

    def test_resume_from_custom(self):
        r = Run(experiment_id="exp1", resume_from="/ckpts/iter000005")
        assert r.resume_from == "/ckpts/iter000005"

    def test_checkpoint_every_n_epochs_default(self):
        r = Run(experiment_id="exp1")
        assert r.checkpoint_every_n_epochs == 1

    def test_keep_last_n_checkpoints_default(self):
        r = Run(experiment_id="exp1")
        assert r.keep_last_n_checkpoints is None

    def test_save_best_checkpoint_default(self):
        r = Run(experiment_id="exp1")
        assert r.save_best_checkpoint is False

    def test_ray_task_timeout_sec_default(self):
        r = Run(experiment_id="exp1")
        assert r.ray_task_timeout_sec == 1800

    def test_max_epoch_retries_default(self):
        r = Run(experiment_id="exp1")
        assert r.max_epoch_retries == 0


# ============================================================
# Train model
# ============================================================
class TestTrain:
    def _make(self, **kw):
        defaults = dict(alg_name="sgrpo", total_number_of_epochs=1, train_steps_per_epoch=10)
        defaults.update(kw)
        return Train(**defaults)

    def test_valid_construction(self):
        t = self._make()
        assert t.alg_name == "sgrpo"
        assert t.lr == 1e-5
        assert t.beta == 0.1

    def test_missing_alg_name(self):
        with pytest.raises(ValidationError):
            Train(total_number_of_epochs=1)

    def test_missing_total_epochs(self):
        with pytest.raises(ValidationError):
            Train(alg_name="sft")

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            self._make(unknown_thing=42)

    def test_lr_must_be_positive(self):
        with pytest.raises(ValidationError):
            self._make(lr=0)

    def test_lr_negative_rejected(self):
        with pytest.raises(ValidationError):
            self._make(lr=-0.001)

    def test_wrong_type_lr(self):
        with pytest.raises(ValidationError):
            self._make(lr="not_a_float")

    def test_default_optimizer(self):
        t = self._make()
        assert t.optimizer_name == "adamw"

    def test_default_betas(self):
        t = self._make()
        assert t.betas == [0.9, 0.95]

    def test_default_clip_grad_norm(self):
        t = self._make()
        assert t.clip_grad_norm == 1.0

    def test_default_scheduler(self):
        t = self._make()
        assert t.lr_scheduler == "WarmupCosineLR"

    # ---- Algorithm hyperparameter defaults ----
    def test_default_beta(self):
        assert self._make().beta == 0.1

    def test_default_simpo_gamma(self):
        assert self._make().simpo_gamma == 0.5

    def test_default_cpo_alpha(self):
        assert self._make().cpo_alpha == 1.0

    def test_default_rdpo_alpha(self):
        assert self._make().rdpo_alpha == 0.01

    def test_default_betadpo_alpha(self):
        assert self._make().betadpo_alpha == 0.5

    def test_default_caldpo_lambda(self):
        assert self._make().caldpo_lambda == 1.0

    def test_default_exo_epsilon(self):
        assert self._make().exo_epsilon == 1e-3

    def test_default_dpop_lambda(self):
        assert self._make().dpop_lambda == 5.0

    def test_default_gpo_loss_type(self):
        assert self._make().gpo_loss_type == "exponential"

    def test_default_fdpo_divergence(self):
        assert self._make().fdpo_divergence == "reverse_kl"

    def test_default_bpo_balance_factor(self):
        assert self._make().bpo_balance_factor == 0.3

    def test_default_discopop_tau(self):
        assert self._make().discopop_tau == 0.05

    def test_default_odpo_delta(self):
        assert self._make().odpo_delta == 1.0

    def test_default_focalpo_gamma(self):
        assert self._make().focalpo_gamma == 1.0

    def test_default_hdpo_alpha(self):
        assert self._make().hdpo_alpha == 1.0

    def test_default_dposhift_lambda(self):
        assert self._make().dposhift_lambda == 0.5

    def test_default_cposimpo_alpha(self):
        assert self._make().cposimpo_alpha == 1.0

    def test_default_drdpo_beta_prime(self):
        assert self._make().drdpo_beta_prime == 1.0

    def test_default_dpnll_alpha(self):
        assert self._make().dpnll_alpha == 1.0

    def test_default_c2dpo_lambda(self):
        assert self._make().c2dpo_lambda == 2e-4

    def test_default_alpha_dpo_alpha(self):
        assert self._make().alpha_dpo_alpha == 0.1

    def test_default_ppo_vf_clip(self):
        assert self._make().ppo_vf_clip == 0.2

    def test_default_ppo_gamma(self):
        assert self._make().ppo_gamma == 0.99

    def test_custom_values_roundtrip(self):
        t = self._make(beta=0.5, simpo_gamma=1.0, cpo_alpha=2.0, lr=3e-4)
        assert t.beta == 0.5
        assert t.simpo_gamma == 1.0
        assert t.cpo_alpha == 2.0
        assert t.lr == 3e-4

    def test_kl_coeff_default(self):
        assert self._make().kl_coeff == 0.0

    def test_clip_low_default(self):
        assert self._make().clip_low == 0.2

    def test_clip_high_default(self):
        assert self._make().clip_high == 0.2


# ============================================================
# Data model
# ============================================================
class TestData:
    def _make(self, **kw):
        defaults = dict(
            train_dnames=["d1"], train_ratios={"d1": 1.0},
            train_files_path="/tmp/d", val_files_path="/tmp/v",
        )
        defaults.update(kw)
        return Data(**defaults)

    def test_valid_construction(self):
        d = self._make()
        assert d.max_seq_len == 512
        assert d.prompt_key == "prompt"

    def test_missing_train_dnames(self):
        with pytest.raises(ValidationError):
            Data(train_ratios={"d": 1.0}, train_files_path="/tmp", val_files_path="/tmp")

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            self._make(foo="bar")

    def test_custom_keys(self):
        d = self._make(prompt_key="q", answer_key="a", chosen_key="c", rejected_key="r")
        assert d.prompt_key == "q"
        assert d.answer_key == "a"
        assert d.chosen_key == "c"
        assert d.rejected_key == "r"


# ============================================================
# Model model
# ============================================================
class TestModel:
    def test_valid_construction(self):
        m = Model(name="test-model")
        assert m.name == "test-model"
        assert m.dtype == "bfloat16"
        assert m.ref_model == ""

    def test_missing_name(self):
        with pytest.raises(ValidationError):
            Model()

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            Model(name="m", xxx=1)

    def test_custom_values(self):
        m = Model(name="m", dtype="float16", ref_model="ref-m",
                  trust_remote_code=True, model_class="vlm")
        assert m.dtype == "float16"
        assert m.ref_model == "ref-m"
        assert m.trust_remote_code is True
        assert m.model_class == "vlm"


# ============================================================
# DeepSpeed model
# ============================================================
class TestDeepSpeed:
    def test_valid_defaults(self):
        ds = DeepSpeed()
        assert ds.train_batch_size is None
        assert ds.fp16 == {"enabled": False}
        assert ds.bf16 == {"enabled": False}
        assert ds.zero_optimization["stage"] == 3

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            DeepSpeed(foo=1)

    def test_custom_optimizer(self):
        ds = DeepSpeed(optimizer={"type": "Adam", "params": {"lr": 1e-4}})
        assert ds.optimizer["type"] == "Adam"


# ============================================================
# DeepSpeedRef model
# ============================================================
class TestDeepSpeedRef:
    def test_valid_defaults(self):
        dsr = DeepSpeedRef()
        assert dsr.fp16 == {"enabled": False}
        assert dsr.bf16 == {"enabled": False}
        assert dsr.zero_optimization == {}

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            DeepSpeedRef(bar=2)


# ============================================================
# InferenceEngine model
# ============================================================
class TestInferenceEngine:
    def test_valid_defaults(self):
        ie = InferenceEngine()
        assert ie.name == "vllm"

    def test_custom_name(self):
        ie = InferenceEngine(name="sglang")
        assert ie.name == "sglang"

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            InferenceEngine(gpu=0)


# ============================================================
# Lora model
# ============================================================
class TestLora:
    def test_valid_defaults(self):
        l = Lora()
        assert l.enabled is False
        assert l.r == 16
        assert l.lora_alpha == 32
        assert "q_proj" in l.target_modules

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            Lora(xyz=1)

    def test_custom_values(self):
        l = Lora(enabled=True, r=8, lora_alpha=16, lora_dropout=0.1)
        assert l.enabled is True
        assert l.r == 8
        assert l.lora_alpha == 16
        assert l.lora_dropout == 0.1


# ============================================================
# Reward model
# ============================================================
class TestReward:
    def test_valid_defaults(self):
        r = Reward()
        assert r.reward_func == "default_reward_func"
        assert r.broadcast is False

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            Reward(nope=True)

    def test_custom_reward_func(self):
        r = Reward(reward_func="gsm8k_reward_func")
        assert r.reward_func == "gsm8k_reward_func"


# ============================================================
# Rollout model
# ============================================================
class TestRollout:
    def test_valid_defaults(self):
        r = Rollout()
        assert r.temperature == 1.0
        assert r.max_tokens == 512
        assert r.n_samples == 8
        assert r.force_strict_on_policy is True

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            Rollout(bad=1)

    def test_custom_values(self):
        r = Rollout(temperature=0.7, max_tokens=1024, n_samples=4, top_p=0.9)
        assert r.temperature == 0.7
        assert r.max_tokens == 1024
        assert r.n_samples == 4
        assert r.top_p == 0.9


# ============================================================
# Config (root) model
# ============================================================
class TestConfig:
    def _minimal_raw(self, **overrides):
        raw = {
            "run": {"experiment_id": "test"},
            "train": {"alg_name": "sft", "total_number_of_epochs": 1,
                      "micro_batches_per_epoch": 100},
            "model": {"name": "m"},
            "data": {"train_dnames": ["d"], "train_ratios": {"d": 1.0},
                     "train_files_path": "/tmp/d", "val_files_path": "/tmp/v"},
        }
        raw.update(overrides)
        return raw

    def test_valid_construction(self):
        c = Config(**self._minimal_raw())
        assert c.run.experiment_id == "test"
        assert isinstance(c.deepspeed, DeepSpeed)
        assert isinstance(c.lora, Lora)

    def test_extra_field_forbidden(self):
        raw = self._minimal_raw()
        raw["unknown_section"] = {"a": 1}
        with pytest.raises(ValidationError):
            Config(**raw)

    def test_missing_run(self):
        raw = self._minimal_raw()
        del raw["run"]
        with pytest.raises(ValidationError):
            Config(**raw)

    def test_missing_train(self):
        raw = self._minimal_raw()
        del raw["train"]
        with pytest.raises(ValidationError):
            Config(**raw)

    def test_missing_model(self):
        raw = self._minimal_raw()
        del raw["model"]
        with pytest.raises(ValidationError):
            Config(**raw)

    def test_missing_data(self):
        raw = self._minimal_raw()
        del raw["data"]
        with pytest.raises(ValidationError):
            Config(**raw)

    def test_deepspeed_defaults(self):
        c = Config(**self._minimal_raw())
        assert c.deepspeed.zero_optimization["stage"] == 3

    def test_reward_defaults(self):
        c = Config(**self._minimal_raw())
        assert c.reward.reward_func == "default_reward_func"

    def test_rollout_defaults(self):
        c = Config(**self._minimal_raw())
        assert c.rollout.n_samples == 8

    def test_deepspeed_ref_default_none(self):
        c = Config(**self._minimal_raw())
        assert c.deepspeed_ref is None

    def test_sync_deepspeed_method_exists(self):
        c = Config(**self._minimal_raw())
        assert hasattr(c, "sync_deepspeed_config")
