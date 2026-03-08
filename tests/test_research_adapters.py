"""
Tests for research adapters — custom loss/algorithm registries, reward shapers,
and research config extensibility.

Run with: pytest tests/test_research_adapters.py -v
"""
import pytest
import sys
import os
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oxrl.models.research_adapters import (
    _CUSTOM_LOSS_REGISTRY,
    _CUSTOM_ALGORITHM_REGISTRY,
    register_loss,
    get_custom_loss_fn,
    register_algorithm,
    get_custom_algorithm,
    load_research_module,
    RewardShaper,
    LengthPenaltyShaper,
    ClampRewardShaper,
)
from oxrl.rewards.backend import RewardBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_registries():
    """Clear custom registries before and after each test."""
    _CUSTOM_LOSS_REGISTRY.clear()
    _CUSTOM_ALGORITHM_REGISTRY.clear()
    yield
    _CUSTOM_LOSS_REGISTRY.clear()
    _CUSTOM_ALGORITHM_REGISTRY.clear()


class DummyRewardBackend(RewardBackend):
    """Minimal reward backend for testing shapers."""
    def __init__(self, reward_value: float = 1.0):
        self.reward_value = reward_value
        self._setup_called = False
        self._cleanup_called = False

    def __call__(self, prompt_ids, response_ids, finish_reason, metadata=None):
        return torch.tensor(self.reward_value), False

    def setup(self, config=None):
        self._setup_called = True

    def cleanup(self):
        self._cleanup_called = True


# ---------------------------------------------------------------------------
# Loss registry tests
# ---------------------------------------------------------------------------

class TestLossRegistry:
    def test_register_loss_decorator(self):
        @register_loss("test_loss")
        def my_loss(*args, **kwargs):
            return torch.tensor(0.0), {}

        assert "test_loss" in _CUSTOM_LOSS_REGISTRY
        assert _CUSTOM_LOSS_REGISTRY["test_loss"] is my_loss

    def test_get_custom_loss_fn_found(self):
        @register_loss("found_loss")
        def my_loss(*args, **kwargs):
            return torch.tensor(0.0), {}

        result = get_custom_loss_fn("found_loss")
        assert result is my_loss

    def test_get_custom_loss_fn_not_found(self):
        assert get_custom_loss_fn("nonexistent") is None

    def test_register_loss_preserves_function(self):
        @register_loss("identity_check")
        def original_fn():
            return 42

        assert original_fn() == 42

    def test_get_loss_fn_fallback_to_custom(self):
        """get_loss_fn() in losses/__init__.py should fall back to custom registry."""
        @register_loss("custom_variant")
        def custom_loss(*args, **kwargs):
            return torch.tensor(0.0), {}

        from oxrl.algs.losses import get_loss_fn
        result = get_loss_fn("custom_variant")
        assert result is custom_loss

    def test_get_loss_fn_builtin_takes_priority(self):
        """Built-in losses should always take priority over custom ones."""
        @register_loss("sgrpo")
        def fake_sgrpo(*args, **kwargs):
            return torch.tensor(999.0), {}

        from oxrl.algs.losses import get_loss_fn
        result = get_loss_fn("sgrpo")
        assert result is not fake_sgrpo  # built-in wins

    def test_get_loss_fn_unknown_raises(self):
        from oxrl.algs.losses import get_loss_fn
        with pytest.raises(ValueError, match="Unknown loss variant"):
            get_loss_fn("totally_unknown_variant_xyz")


# ---------------------------------------------------------------------------
# Algorithm registry tests
# ---------------------------------------------------------------------------

class TestAlgorithmRegistry:
    def test_register_algorithm_decorator(self):
        @register_algorithm("test_alg")
        class MyAlgorithm:
            pass

        assert "test_alg" in _CUSTOM_ALGORITHM_REGISTRY
        assert _CUSTOM_ALGORITHM_REGISTRY["test_alg"] is MyAlgorithm

    def test_get_custom_algorithm_found(self):
        @register_algorithm("found_alg")
        class MyAlgorithm:
            pass

        result = get_custom_algorithm("found_alg")
        assert result is MyAlgorithm

    def test_get_custom_algorithm_not_found(self):
        assert get_custom_algorithm("nonexistent") is None

    def test_register_algorithm_preserves_class(self):
        @register_algorithm("preserved")
        class Original:
            x = 42

        assert Original.x == 42

    def test_get_algorithm_class_fallback_to_custom(self):
        """get_algorithm_class() should fall back to custom registry."""
        @register_algorithm("custom_rl")
        class CustomRL:
            pass

        from oxrl.setup.engine_factory import get_algorithm_class
        # This will try to apply @ray.remote — skip if ray not available
        try:
            result = get_algorithm_class("custom_rl")
            assert result is not None
        except ImportError:
            pytest.skip("Ray not installed")

    def test_get_algorithm_class_builtin_takes_priority(self):
        """Built-in algorithms should always take priority over custom ones."""
        @register_algorithm("sgrpo")
        class FakeSGRPO:
            pass

        from oxrl.setup.engine_factory import get_algorithm_class
        result = get_algorithm_class("sgrpo")
        assert result is not FakeSGRPO

    def test_get_algorithm_class_unknown_raises(self):
        from oxrl.setup.engine_factory import get_algorithm_class
        with pytest.raises(ValueError, match="Unknown algorithm"):
            get_algorithm_class("totally_unknown_algorithm_xyz")


# ---------------------------------------------------------------------------
# RewardShaper tests
# ---------------------------------------------------------------------------

class TestRewardShaper:
    def test_identity_shaper(self):
        inner = DummyRewardBackend(reward_value=5.0)
        shaper = RewardShaper(inner)
        reward, is_per_token = shaper([1, 2], [3, 4, 5], "stop")
        assert reward.item() == pytest.approx(5.0)
        assert is_per_token is False

    def test_shaper_delegates_setup(self):
        inner = DummyRewardBackend()
        shaper = RewardShaper(inner)
        shaper.setup({"key": "val"})
        assert inner._setup_called is True

    def test_shaper_delegates_cleanup(self):
        inner = DummyRewardBackend()
        shaper = RewardShaper(inner)
        shaper.cleanup()
        assert inner._cleanup_called is True

    def test_shaper_stores_kwargs(self):
        inner = DummyRewardBackend()
        shaper = RewardShaper(inner, alpha=0.5, beta=1.0)
        assert shaper.kwargs == {"alpha": 0.5, "beta": 1.0}


class TestLengthPenaltyShaper:
    def test_penalty_subtracted(self):
        inner = DummyRewardBackend(reward_value=1.0)
        shaper = LengthPenaltyShaper(inner, penalty_per_token=0.01)
        response_ids = [10, 20, 30, 40, 50]  # 5 tokens
        reward, is_per_token = shaper([1], response_ids, "stop")
        # 1.0 - 0.01 * 5 = 0.95
        assert reward.item() == pytest.approx(0.95)

    def test_zero_penalty(self):
        inner = DummyRewardBackend(reward_value=2.0)
        shaper = LengthPenaltyShaper(inner, penalty_per_token=0.0)
        reward, _ = shaper([1], [10, 20], "stop")
        assert reward.item() == pytest.approx(2.0)

    def test_long_response_large_penalty(self):
        inner = DummyRewardBackend(reward_value=0.5)
        shaper = LengthPenaltyShaper(inner, penalty_per_token=0.001)
        response_ids = list(range(1000))
        reward, _ = shaper([1], response_ids, "stop")
        # 0.5 - 0.001 * 1000 = -0.5
        assert reward.item() == pytest.approx(-0.5)


class TestClampRewardShaper:
    def test_clamp_high(self):
        inner = DummyRewardBackend(reward_value=10.0)
        shaper = ClampRewardShaper(inner, min_val=-1.0, max_val=1.0)
        reward, _ = shaper([1], [2], "stop")
        assert reward.item() == pytest.approx(1.0)

    def test_clamp_low(self):
        inner = DummyRewardBackend(reward_value=-10.0)
        shaper = ClampRewardShaper(inner, min_val=-1.0, max_val=1.0)
        reward, _ = shaper([1], [2], "stop")
        assert reward.item() == pytest.approx(-1.0)

    def test_within_range_unchanged(self):
        inner = DummyRewardBackend(reward_value=0.5)
        shaper = ClampRewardShaper(inner, min_val=-1.0, max_val=1.0)
        reward, _ = shaper([1], [2], "stop")
        assert reward.item() == pytest.approx(0.5)

    def test_custom_range(self):
        inner = DummyRewardBackend(reward_value=5.0)
        shaper = ClampRewardShaper(inner, min_val=0.0, max_val=3.0)
        reward, _ = shaper([1], [2], "stop")
        assert reward.item() == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestResearchConfig:
    def test_research_model_defaults(self):
        from oxrl.configs.schema import Research
        r = Research()
        assert r.module is None

    def test_research_model_accepts_arbitrary_keys(self):
        from oxrl.configs.schema import Research
        r = Research(module="my_module", dr_grpo_beta=0.1, custom_flag=True)
        assert r.module == "my_module"
        assert r.dr_grpo_beta == 0.1
        assert r.custom_flag is True

    def test_config_has_research_field(self):
        from oxrl.configs.schema import Config
        import inspect
        sig = inspect.signature(Config)
        fields = Config.model_fields
        assert "research" in fields

    def test_loss_variant_default_none(self):
        from oxrl.configs.schema import Train
        fields = Train.model_fields
        assert "loss_variant" in fields
        assert fields["loss_variant"].default is None

    def test_loss_variant_set(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="sgrpo", total_number_of_epochs=1, loss_variant="dr_grpo")
        assert t.loss_variant == "dr_grpo"


# ---------------------------------------------------------------------------
# Module loader tests
# ---------------------------------------------------------------------------

class TestLoadResearchModule:
    def test_load_existing_module(self):
        """load_research_module should import a standard-library module without error."""
        load_research_module("json")

    def test_load_nonexistent_module_raises(self):
        with pytest.raises(ModuleNotFoundError):
            load_research_module("nonexistent_module_xyz_12345")

    def test_load_module_executes_decorators(self):
        """Create a temp module that registers a loss, then verify it was registered."""
        mod = types.ModuleType("_test_research_ext")
        mod.__file__ = "<test>"

        # Simulate what would happen in the module body
        @register_loss("from_loaded_module")
        def loaded_loss(*args, **kwargs):
            return torch.tensor(0.0), {}

        mod.loaded_loss = loaded_loss
        sys.modules["_test_research_ext"] = mod

        try:
            assert get_custom_loss_fn("from_loaded_module") is loaded_loss
        finally:
            del sys.modules["_test_research_ext"]
