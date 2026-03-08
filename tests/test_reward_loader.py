"""Tests for pluggable reward loading, composite rewards, LLM judge, and schema."""
import pickle
from unittest.mock import patch, MagicMock

import pytest
import torch

from oxrl.rewards.backend import RewardBackend, FunctionRewardBackend
from oxrl.rewards.loader import (
    resolve_reward_func, create_reward_backend,
    CompositeReward, CompositeRewardBackend,
    _resolve_single,
)
from oxrl.rewards.llm_judge import LLMJudgeReward
from oxrl.rewards.rm_reward import RewardModelBackend
from oxrl.configs.schema import (
    Reward,
    CompositeRewardEntry,
    LLMJudgeConfig,
    RewardModelConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_reward(prompt_ids, response_ids, finish_reason, metadata=None):
    """Simple reward: 1.0 at last token."""
    r = torch.zeros(len(response_ids), dtype=torch.float32)
    if len(response_ids) > 0:
        r[-1] = 1.0
    return r, False


def _dummy_reward_half(prompt_ids, response_ids, finish_reason, metadata=None):
    """Simple reward: 0.5 at last token."""
    r = torch.zeros(len(response_ids), dtype=torch.float32)
    if len(response_ids) > 0:
        r[-1] = 0.5
    return r, False


def _per_token_reward(prompt_ids, response_ids, finish_reason, metadata=None):
    """Per-token reward (not supported in composite)."""
    r = torch.ones(len(response_ids), dtype=torch.float32) * 0.1
    return r, True


def _make_llm_judge_config(**overrides):
    defaults = dict(
        api_base="http://localhost:8000/v1",
        model="test-model",
        prompt_template="Rate: {prompt} {response}",
        max_tokens=64,
        temperature=0.0,
        max_retries=2,
        retry_delay=0.0,
        timeout=5.0,
        fallback_score=0.0,
        normalize_to_01=True,
        api_key="test-key",
        api_key_env="OPENAI_API_KEY",
    )
    defaults.update(overrides)
    return LLMJudgeConfig(**defaults)


# ===========================================================================
# RewardBackend ABC tests
# ===========================================================================

class TestRewardBackend:
    def test_function_backend_is_reward_backend(self):
        fb = FunctionRewardBackend(_dummy_reward)
        assert isinstance(fb, RewardBackend)

    def test_function_backend_delegates(self):
        fb = FunctionRewardBackend(_dummy_reward)
        r, is_per_token = fb([1, 2], [10, 20, 30], "stop")
        assert not is_per_token
        assert r.shape == (3,)
        assert abs(r[-1].item() - 1.0) < 1e-6
        assert r[0].item() == 0.0

    def test_function_backend_setup_cleanup_noop(self):
        fb = FunctionRewardBackend(_dummy_reward)
        # Should not raise
        fb.setup()
        fb.cleanup()

    def test_function_backend_picklable(self):
        fb = FunctionRewardBackend(_dummy_reward)
        restored = pickle.loads(pickle.dumps(fb))
        r1, _ = fb([1], [10], "stop")
        r2, _ = restored([1], [10], "stop")
        assert torch.allclose(r1, r2)

    def test_function_backend_repr(self):
        fb = FunctionRewardBackend(_dummy_reward)
        assert "_dummy_reward" in repr(fb)


# ===========================================================================
# Loader tests
# ===========================================================================

class TestResolveRewardFunc:
    def test_builtin_name(self):
        fn = resolve_reward_func("default_reward_func")
        assert isinstance(fn, RewardBackend)

    def test_dotted_path(self):
        fn = resolve_reward_func("oxrl.rewards.base.default_reward_func")
        assert isinstance(fn, RewardBackend)

    def test_all_builtins_resolve(self):
        builtins = [
            "default_reward_func",
            "gsm8k_reward_func",
            "math_reward_func",
            "soft_math_reward_func",
            "code_reward_func",
            "format_reward_func",
            "mcqa_reward_func",
            "reasoning_reward_func",
            "multimodal_reward_func",
            "rm_reward_func",
        ]
        for name in builtins:
            fn = resolve_reward_func(name)
            assert isinstance(fn, RewardBackend), f"{name} is not RewardBackend"
            assert callable(fn), f"{name} did not resolve to callable"

    def test_unknown_builtin_raises(self):
        with pytest.raises(AttributeError):
            resolve_reward_func("nonexistent_reward_xyz")

    def test_unknown_module_raises(self):
        with pytest.raises(ModuleNotFoundError):
            resolve_reward_func("nonexistent.module.func")

    def test_composite_without_config_raises(self):
        with pytest.raises(ValueError, match="composite_rewards"):
            resolve_reward_func("composite")

    def test_llm_judge_without_config_raises(self):
        with pytest.raises(ValueError, match="llm_judge"):
            resolve_reward_func("llm_judge_reward_func")

    def test_composite_resolves(self):
        entries = [CompositeRewardEntry(func="default_reward_func", weight=1.0)]
        fn = resolve_reward_func("composite", composite_rewards=entries)
        assert isinstance(fn, CompositeRewardBackend)
        assert isinstance(fn, RewardBackend)

    def test_llm_judge_resolves(self):
        config = _make_llm_judge_config()
        fn = resolve_reward_func("llm_judge_reward_func", llm_judge_config=config)
        assert isinstance(fn, LLMJudgeReward)
        assert isinstance(fn, RewardBackend)


class TestCreateRewardBackend:
    def test_returns_reward_backend_for_builtin(self):
        backend = create_reward_backend("gsm8k_reward_func")
        assert isinstance(backend, RewardBackend)

    def test_returns_reward_backend_for_llm_judge(self):
        config = _make_llm_judge_config()
        backend = create_reward_backend(
            "llm_judge_reward_func", llm_judge_config=config,
        )
        assert isinstance(backend, RewardBackend)
        assert isinstance(backend, LLMJudgeReward)

    def test_backward_compat_alias(self):
        """resolve_reward_func still works as backward-compat alias."""
        r1 = resolve_reward_func("default_reward_func")
        r2 = create_reward_backend("default_reward_func")
        assert type(r1) == type(r2)

    def test_composite_reward_alias_still_works(self):
        """CompositeReward alias still resolves."""
        assert CompositeReward is CompositeRewardBackend


# ===========================================================================
# Composite reward tests
# ===========================================================================

class TestCompositeReward:
    def test_equal_weights_average(self):
        composite = CompositeRewardBackend([
            (FunctionRewardBackend(_dummy_reward), 1.0),
            (FunctionRewardBackend(_dummy_reward_half), 1.0),
        ])
        r, is_per_token = composite([1, 2], [10, 20, 30], "stop")
        assert not is_per_token
        assert r.shape == (3,)
        # (1.0 * 0.5 + 0.5 * 0.5) = 0.75 at last position
        assert abs(r[-1].item() - 0.75) < 1e-6
        assert r[0].item() == 0.0

    def test_unequal_weights(self):
        composite = CompositeRewardBackend([
            (FunctionRewardBackend(_dummy_reward), 3.0),
            (FunctionRewardBackend(_dummy_reward_half), 1.0),
        ])
        r, _ = composite([1], [10, 20], "stop")
        # normalized: 3/4=0.75, 1/4=0.25
        expected = 0.75 * 1.0 + 0.25 * 0.5
        assert abs(r[-1].item() - expected) < 1e-6

    def test_empty_composite_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            CompositeRewardBackend([])

    def test_empty_response_returns_zero(self):
        composite = CompositeRewardBackend([(FunctionRewardBackend(_dummy_reward), 1.0)])
        r, is_per_token = composite([1, 2], [], "stop")
        assert r.shape == (0,)
        assert not is_per_token

    def test_per_token_sub_reward_raises(self):
        composite = CompositeRewardBackend([(FunctionRewardBackend(_per_token_reward), 1.0)])
        with pytest.raises(ValueError, match="per-token"):
            composite([1], [10, 20], "stop")

    def test_picklable(self):
        composite = CompositeRewardBackend([(FunctionRewardBackend(_dummy_reward), 1.0)])
        restored = pickle.loads(pickle.dumps(composite))
        r1, _ = composite([1], [10], "stop")
        r2, _ = restored([1], [10], "stop")
        assert torch.allclose(r1, r2)

    def test_dotted_path_in_entry(self):
        entries = [CompositeRewardEntry(func="oxrl.rewards.base.default_reward_func", weight=1.0)]
        fn = create_reward_backend("composite", composite_rewards=entries)
        r, is_per_token = fn([1], [10, 20], "stop")
        assert r.shape == (2,)
        assert not is_per_token

    def test_reward_contract(self):
        """Output matches (Tensor[len(response_ids)], bool) contract."""
        composite = CompositeRewardBackend([(FunctionRewardBackend(_dummy_reward), 1.0)])
        response_ids = [10, 20, 30, 40]
        result = composite([1, 2], response_ids, "stop")
        assert isinstance(result, tuple)
        assert len(result) == 2
        r, is_per_token = result
        assert isinstance(r, torch.Tensor)
        assert r.shape == (len(response_ids),)
        assert isinstance(is_per_token, bool)

    def test_is_reward_backend(self):
        composite = CompositeRewardBackend([(FunctionRewardBackend(_dummy_reward), 1.0)])
        assert isinstance(composite, RewardBackend)

    def test_setup_cleanup_delegate(self):
        """setup/cleanup delegate to sub-backends."""
        mock_backend = MagicMock(spec=RewardBackend)
        mock_backend.__call__ = MagicMock(
            return_value=(torch.zeros(1), False)
        )
        composite = CompositeRewardBackend([(mock_backend, 1.0)])
        composite.setup(config="test")
        mock_backend.setup.assert_called_once_with("test")
        composite.cleanup()
        mock_backend.cleanup.assert_called_once()


# ===========================================================================
# RewardModelBackend tests
# ===========================================================================

class TestRewardModelBackend:
    def test_is_reward_backend(self):
        rm = RewardModelBackend(model_path="/fake/path", device="cpu")
        assert isinstance(rm, RewardBackend)

    def test_picklable_excludes_model(self):
        rm = RewardModelBackend(model_path="/fake/path", device="cpu")
        state = rm.__getstate__()
        assert "_model" not in state
        assert "_value_head" not in state
        assert state["_model_path"] == "/fake/path"
        assert state["_device"] == "cpu"

        restored = pickle.loads(pickle.dumps(rm))
        assert restored._model_path == "/fake/path"
        assert restored._model is None
        assert restored._value_head is None

    def test_repr(self):
        rm = RewardModelBackend(model_path="/fake/path")
        assert "/fake/path" in repr(rm)


# ===========================================================================
# LLM Judge tests
# ===========================================================================

class TestLLMJudgeIsBackend:
    def test_is_reward_backend(self):
        judge = LLMJudgeReward(_make_llm_judge_config())
        assert isinstance(judge, RewardBackend)

    def test_cleanup_closes_client(self):
        judge = LLMJudgeReward(_make_llm_judge_config())
        mock_client = MagicMock()
        judge._client = mock_client
        judge.cleanup()
        mock_client.close.assert_called_once()
        assert judge._client is None

    def test_cleanup_noop_when_no_client(self):
        judge = LLMJudgeReward(_make_llm_judge_config())
        # Should not raise
        judge.cleanup()


class TestLLMJudgeExtractScore:
    def setup_method(self):
        self.judge = LLMJudgeReward(_make_llm_judge_config())

    def test_integer(self):
        assert self.judge._extract_score("7") == 0.7

    def test_decimal(self):
        assert abs(self.judge._extract_score("8.5") - 0.85) < 1e-6

    def test_text_with_number(self):
        assert abs(self.judge._extract_score("I would rate this a 6 out of 10") - 0.6) < 1e-6

    def test_no_number_returns_fallback(self):
        assert self.judge._extract_score("no score here") == 0.0

    def test_clamped_above_10(self):
        # Score > 10 gets clamped to 10, then normalized to 1.0
        assert abs(self.judge._extract_score("15") - 1.0) < 1e-6

    def test_normalize_disabled(self):
        judge = LLMJudgeReward(_make_llm_judge_config(normalize_to_01=False))
        assert abs(judge._extract_score("7") - 7.0) < 1e-6

    def test_normalize_disabled_clamp(self):
        judge = LLMJudgeReward(_make_llm_judge_config(normalize_to_01=False))
        assert abs(judge._extract_score("15") - 10.0) < 1e-6


class TestLLMJudgeReward:
    def test_empty_response_no_api_call(self):
        judge = LLMJudgeReward(_make_llm_judge_config())
        r, is_per_token = judge([], [], "stop", metadata={})
        assert r.shape == (0,)
        assert not is_per_token

    def test_mocked_api_call(self):
        judge = LLMJudgeReward(_make_llm_judge_config())

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Score: 8"}}]
        }

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        judge._client = mock_client

        r, is_per_token = judge(
            [1, 2], [10, 20, 30], "stop",
            metadata={"prompt_text": "What is 2+2?", "response_text": "4"}
        )
        assert not is_per_token
        assert r.shape == (3,)
        assert abs(r[-1].item() - 0.8) < 1e-6
        assert r[0].item() == 0.0
        mock_client.post.assert_called_once()

    def test_api_failure_returns_fallback(self):
        judge = LLMJudgeReward(_make_llm_judge_config(fallback_score=0.25))

        mock_client = MagicMock()
        mock_client.post.side_effect = Exception("Connection refused")
        judge._client = mock_client

        r, _ = judge(
            [1], [10, 20], "stop",
            metadata={"prompt_text": "test", "response_text": "test"}
        )
        assert abs(r[-1].item() - 0.25) < 1e-6
        assert mock_client.post.call_count == 2  # max_retries=2

    def test_picklable(self):
        judge = LLMJudgeReward(_make_llm_judge_config())
        restored = pickle.loads(pickle.dumps(judge))
        assert restored.api_base == judge.api_base
        assert restored.model == judge.model
        assert restored._client is None  # client excluded from pickle

    def test_reward_contract(self):
        """Output matches (Tensor[len(response_ids)], bool) contract."""
        judge = LLMJudgeReward(_make_llm_judge_config())
        # Empty response — no API call needed
        response_ids = []
        result = judge([1, 2], response_ids, "stop", metadata={})
        assert isinstance(result, tuple)
        assert len(result) == 2
        r, is_per_token = result
        assert isinstance(r, torch.Tensor)
        assert r.shape == (len(response_ids),)
        assert isinstance(is_per_token, bool)

    def test_missing_metadata_keys(self):
        """Works even without prompt_text/response_text in metadata."""
        judge = LLMJudgeReward(_make_llm_judge_config())

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "5"}}]
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        judge._client = mock_client

        r, _ = judge([1], [10], "stop", metadata=None)
        assert abs(r[-1].item() - 0.5) < 1e-6


# ===========================================================================
# Schema tests
# ===========================================================================

class TestSchemaBackwardCompat:
    def test_reward_defaults(self):
        """Existing configs work unchanged — new fields default to None."""
        r = Reward()
        assert r.composite_rewards is None
        assert r.llm_judge is None
        assert r.reward_model is None
        assert r.reward_func == "default_reward_func"

    def test_composite_config_parses(self):
        r = Reward(
            reward_func="composite",
            composite_rewards=[
                CompositeRewardEntry(func="gsm8k_reward_func", weight=2.0),
                CompositeRewardEntry(func="default_reward_func"),
            ],
        )
        assert len(r.composite_rewards) == 2
        assert r.composite_rewards[0].weight == 2.0
        assert r.composite_rewards[1].weight == 1.0

    def test_llm_judge_config_parses(self):
        r = Reward(
            reward_func="llm_judge_reward_func",
            llm_judge=LLMJudgeConfig(
                api_base="http://localhost:8000/v1",
                model="test-model",
            ),
        )
        assert r.llm_judge.api_base == "http://localhost:8000/v1"
        assert r.llm_judge.normalize_to_01 is True
        assert r.llm_judge.max_retries == 3

    def test_composite_entry_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            CompositeRewardEntry(func="test", weight=1.0, unknown_field="bad")

    def test_llm_judge_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            LLMJudgeConfig(
                api_base="http://localhost:8000/v1",
                model="test",
                unknown_field="bad",
            )


class TestRewardModelConfigSchema:
    def test_parses(self):
        cfg = RewardModelConfig(model_path="/path/to/rm")
        assert cfg.model_path == "/path/to/rm"
        assert cfg.device == "cuda"

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            RewardModelConfig(model_path="/path", unknown="bad")

    def test_reward_model_defaults_to_none(self):
        r = Reward()
        assert r.reward_model is None

    def test_reward_model_config_in_reward(self):
        r = Reward(
            reward_func="rm_reward_func",
            reward_model=RewardModelConfig(model_path="/path/to/rm"),
        )
        assert r.reward_model.model_path == "/path/to/rm"
        assert r.reward_model.device == "cuda"
