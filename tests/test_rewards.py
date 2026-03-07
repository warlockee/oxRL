"""
Comprehensive tests for all reward functions and extraction helpers.
All tests run on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_rewards.py -v
"""
import pytest
import torch
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oxrl.rewards.base import (
    default_reward_func, extract_answer, extract_math_answer,
    extract_mcqa_answer, _normalize_math,
)
from oxrl.rewards.math import gsm8k_reward_func, math_reward_func, soft_math_reward_func
from oxrl.rewards.code import code_reward_func
from oxrl.rewards.format import format_reward_func
from oxrl.rewards.qa import mcqa_reward_func
from oxrl.rewards.reasoning import reasoning_reward_func
from oxrl.rewards.multimodal import multimodal_reward_func


# ============================================================
# extract_answer
# ============================================================
class TestExtractAnswer:
    def test_hash_delimited(self):
        assert extract_answer("The answer is #### 42") == "42"

    def test_negative_number(self):
        assert extract_answer("#### -7") == "-7"

    def test_decimal_number(self):
        assert extract_answer("#### 3.14") == "3.14"

    def test_comma_in_number(self):
        assert extract_answer("#### 1,000") == "1000"

    def test_fallback_last_number(self):
        assert extract_answer("I got 10 then 20 then 30") == "30"

    def test_no_match(self):
        assert extract_answer("no numbers here") is None

    def test_empty_string(self):
        assert extract_answer("") is None

    def test_hash_with_spaces(self):
        assert extract_answer("####   99") == "99"


# ============================================================
# extract_math_answer
# ============================================================
class TestExtractMathAnswer:
    def test_boxed(self):
        # Source uses "\boxed{" (non-raw), where \b = backspace char
        assert extract_math_answer("The answer is \boxed{42}") == "42"

    def test_nested_braces(self):
        # \f is a form-feed char in Python strings, so use the actual string the source produces
        text = "\boxed{\\frac{1}{2}}"
        result = extract_math_answer(text)
        assert result == "\\frac{1}{2}"

    def test_fallback_to_hash(self):
        assert extract_math_answer("#### hello world") == "hello world"

    def test_unclosed_brace(self):
        result = extract_math_answer("\boxed{unclosed")
        # depth > 0 so falls through to hash fallback
        assert result is None

    def test_no_match(self):
        assert extract_math_answer("just text") is None

    def test_empty(self):
        assert extract_math_answer("") is None

    def test_multiple_boxed_uses_last(self):
        text = "\boxed{first} ... \boxed{second}"
        assert extract_math_answer(text) == "second"

    def test_whitespace_stripped(self):
        assert extract_math_answer("\boxed{  x  }") == "x"


# ============================================================
# extract_mcqa_answer
# ============================================================
class TestExtractMcqaAnswer:
    def test_the_answer_is(self):
        assert extract_mcqa_answer("The answer is A") == "A"

    def test_the_answer_is_parenthesized(self):
        assert extract_mcqa_answer("The answer is (B)") == "B"

    def test_answer_colon(self):
        assert extract_mcqa_answer("Answer: C") == "C"

    def test_fallback_last_letter(self):
        assert extract_mcqa_answer("I think D is correct") == "D"

    def test_no_match(self):
        assert extract_mcqa_answer("no letters") is None

    def test_case_sensitivity(self):
        assert extract_mcqa_answer("the answer is A") == "A"


# ============================================================
# _normalize_math
# ============================================================
class TestNormalizeMath:
    def test_strip(self):
        assert _normalize_math("  hello  ") == "hello"

    def test_lowercase(self):
        assert _normalize_math("HELLO") == "hello"

    def test_trailing_dot_removal(self):
        assert _normalize_math("42.") == "42"

    def test_combined(self):
        assert _normalize_math("  ANSWER.  ") == "answer"


# ============================================================
# default_reward_func
# ============================================================
class TestDefaultRewardFunc:
    def test_stop_gives_one(self):
        r, is_per_token = default_reward_func([1, 2], [3, 4, 5], "stop")
        assert is_per_token is False
        assert r[-1].item() == 1.0

    def test_length_gives_zero(self):
        r, is_per_token = default_reward_func([1, 2], [3, 4, 5], "length")
        assert r[-1].item() == 0.0

    def test_empty_response(self):
        r, is_per_token = default_reward_func([1, 2], [], "stop")
        assert len(r) == 0

    def test_shape(self):
        r, _ = default_reward_func([1], [10, 20, 30], "stop")
        assert r.shape == (3,)

    def test_dtype(self):
        r, _ = default_reward_func([1], [10], "stop")
        assert r.dtype == torch.float32

    def test_all_zeros_except_last(self):
        r, _ = default_reward_func([1], [10, 20, 30], "stop")
        assert r[0].item() == 0.0
        assert r[1].item() == 0.0
        assert r[2].item() == 1.0


# ============================================================
# gsm8k_reward_func
# ============================================================
class TestGsm8kRewardFunc:
    def test_exact_match(self):
        meta = {"answer": "42", "response_text": "The answer is #### 42"}
        r, _ = gsm8k_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == 1.0

    def test_wrong_answer(self):
        meta = {"answer": "42", "response_text": "#### 99"}
        r, _ = gsm8k_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == 0.0

    def test_tolerance(self):
        meta = {"answer": "42", "response_text": "#### 42.000001"}
        r, _ = gsm8k_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == 1.0

    def test_comma_in_number(self):
        meta = {"answer": "1000", "response_text": "#### 1,000"}
        r, _ = gsm8k_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == 1.0

    def test_no_metadata(self):
        r, _ = gsm8k_reward_func([1], [2, 3], "stop", None)
        assert r[-1].item() == 0.0

    def test_missing_answer_key(self):
        meta = {"response_text": "#### 42"}
        r, _ = gsm8k_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == 0.0

    def test_empty_response_ids(self):
        r, _ = gsm8k_reward_func([1], [], "stop", {"answer": "42"})
        assert len(r) == 0

    def test_returns_tuple(self):
        result = gsm8k_reward_func([1], [2], "stop", {"answer": "1", "response_text": "1"})
        assert isinstance(result, tuple)
        assert len(result) == 2


# ============================================================
# math_reward_func
# ============================================================
class TestMathRewardFunc:
    def test_boxed_match(self):
        meta = {"answer": "42", "response_text": "The answer is \boxed{42}"}
        r, _ = math_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == 1.0

    def test_case_insensitive_via_normalize(self):
        meta = {"answer": "X", "response_text": "\boxed{x}"}
        r, _ = math_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == 1.0

    def test_hash_fallback(self):
        meta = {"answer": "hello", "response_text": "#### hello"}
        r, _ = math_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == 1.0

    def test_wrong_answer(self):
        meta = {"answer": "42", "response_text": "\boxed{99}"}
        r, _ = math_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == 0.0

    def test_no_metadata(self):
        r, _ = math_reward_func([1], [2], "stop", None)
        assert r[-1].item() == 0.0

    def test_empty_response(self):
        r, _ = math_reward_func([1], [], "stop", {"answer": "1"})
        assert len(r) == 0


# ============================================================
# soft_math_reward_func
# ============================================================
class TestSoftMathRewardFunc:
    def test_exact_match(self):
        meta = {"answer": "100", "response_text": "#### 100"}
        r, _ = soft_math_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == 1.0

    def test_within_10_percent(self):
        meta = {"answer": "100", "response_text": "#### 95"}
        r, _ = soft_math_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == 0.5

    def test_beyond_10_percent(self):
        meta = {"answer": "100", "response_text": "#### 50"}
        r, _ = soft_math_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == pytest.approx(0.2)

    def test_no_extraction(self):
        meta = {"answer": "100", "response_text": "no numbers at all"}
        r, _ = soft_math_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == 0.0

    def test_zero_denominator_guard(self):
        # answer=0, within 10% uses (abs(gt) + 1e-8) so no div by zero
        meta = {"answer": "0", "response_text": "#### 0.000001"}
        r, _ = soft_math_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == 1.0  # exact match within tolerance

    def test_no_metadata(self):
        r, _ = soft_math_reward_func([1], [2], "stop", None)
        assert r[-1].item() == 0.0


# ============================================================
# code_reward_func
# ============================================================
class TestCodeRewardFunc:
    def test_passing_code(self):
        code = '```python\ndef add(a, b): return a + b\n```'
        tests = "assert add(1, 2) == 3\nassert add(0, 0) == 0"
        meta = {"response_text": code, "test_cases": tests}
        r, _ = code_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == 1.0

    def test_failing_code(self):
        code = '```python\ndef add(a, b): return a - b\n```'
        tests = "assert add(1, 2) == 3"
        meta = {"response_text": code, "test_cases": tests}
        r, _ = code_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == 0.0

    def test_syntax_error(self):
        code = '```python\ndef add(a, b:\n```'
        tests = "assert True"
        meta = {"response_text": code, "test_cases": tests}
        r, _ = code_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == 0.0

    def test_timeout(self):
        code = '```python\nimport time; time.sleep(100)\n```'
        tests = "assert True"
        meta = {"response_text": code, "test_cases": tests}
        r, _ = code_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == 0.0

    def test_no_code_block(self):
        meta = {"response_text": "def add(a, b): return a + b",
                "test_cases": "assert add(1, 2) == 3"}
        r, _ = code_reward_func([1], [2, 3], "stop", meta)
        # Falls back to full response text as code
        assert r[-1].item() == 1.0

    def test_no_metadata(self):
        r, _ = code_reward_func([1], [2], "stop", None)
        assert r[-1].item() == 0.0

    def test_empty_response_ids(self):
        r, _ = code_reward_func([1], [], "stop", {"response_text": "x", "test_cases": "y"})
        assert len(r) == 0


# ============================================================
# format_reward_func
# ============================================================
class TestFormatRewardFunc:
    def test_all_criteria_met(self):
        text = "A" * 51 + "\n\nSome content here"  # length>50, has \n\n, not starting with I
        meta = {"response_text": text}
        r, _ = format_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == 1.0

    def test_partial_scores(self):
        # Only length > 50
        text = "A" * 51
        meta = {"response_text": text}
        r, _ = format_reward_func([1], [2, 3], "length", meta)
        # length>50: 0.25, no \n\n: 0, not starting with I: 0.25, not stop: 0
        assert r[-1].item() == pytest.approx(0.5)

    def test_starts_with_I(self):
        text = "I think this is good " * 5  # length > 50, starts with I
        meta = {"response_text": text}
        r, _ = format_reward_func([1], [2, 3], "stop", meta)
        # length>50: 0.25, no \n\n: 0, starts with I: 0, stop: 0.25
        assert r[-1].item() == pytest.approx(0.5)

    def test_empty_response_ids(self):
        r, _ = format_reward_func([1], [], "stop", {"response_text": "x"})
        assert len(r) == 0

    def test_no_metadata(self):
        r, _ = format_reward_func([1], [2], "stop", None)
        # No response_text, only checks finish_reason for 0.25
        # Plus empty string: len<=50, no \n\n, no leading I but empty doesn't match
        assert r[-1].item() == pytest.approx(0.25)  # just stop


# ============================================================
# mcqa_reward_func
# ============================================================
class TestMcqaRewardFunc:
    def test_correct_letter(self):
        meta = {"answer": "B", "response_text": "The answer is B"}
        r, _ = mcqa_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == 1.0

    def test_wrong_letter(self):
        meta = {"answer": "B", "response_text": "The answer is C"}
        r, _ = mcqa_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == 0.0

    def test_case_insensitive(self):
        meta = {"answer": "a", "response_text": "The answer is A"}
        r, _ = mcqa_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == 1.0

    def test_no_metadata(self):
        r, _ = mcqa_reward_func([1], [2], "stop", None)
        assert r[-1].item() == 0.0

    def test_empty_response_ids(self):
        r, _ = mcqa_reward_func([1], [], "stop", {"answer": "A"})
        assert len(r) == 0


# ============================================================
# reasoning_reward_func
# ============================================================
class TestReasoningRewardFunc:
    def test_think_tags_score(self):
        meta = {"answer": "wrong", "response_text": "<think>thinking</think>no answer tags"}
        r, _ = reasoning_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == pytest.approx(0.2)

    def test_answer_tags_score(self):
        meta = {"answer": "wrong", "response_text": "<answer>wrong_val</answer>"}
        r, _ = reasoning_reward_func([1], [2, 3], "stop", meta)
        # has answer tags (0.2), answer doesn't match (0)
        assert r[-1].item() == pytest.approx(0.2)

    def test_full_correct(self):
        meta = {"answer": "42", "response_text": "<think>step by step</think><answer>42</answer>"}
        r, _ = reasoning_reward_func([1], [2, 3], "stop", meta)
        # think: 0.2, answer tags: 0.2, correct answer: 0.6
        assert r[-1].item() == pytest.approx(1.0)

    def test_boxed_fallback_with_answer_tags(self):
        meta = {"answer": "42", "response_text": "<answer>\boxed{42}</answer>"}
        r, _ = reasoning_reward_func([1], [2, 3], "stop", meta)
        # answer tags: 0.2, regex extracts "\boxed{42}" as answer text
        # _normalize_math("\boxed{42}") != "42", no else branch for inner if
        # So only 0.2 from answer tags
        assert r[-1].item() == pytest.approx(0.2)

    def test_boxed_fallback_no_answer_tags(self):
        meta = {"answer": "42", "response_text": "\boxed{42}"}
        r, _ = reasoning_reward_func([1], [2, 3], "stop", meta)
        # no think, no answer tags, extract_math_answer finds 42 -> 0.4
        assert r[-1].item() == pytest.approx(0.4)

    def test_no_metadata(self):
        r, _ = reasoning_reward_func([1], [2], "stop", None)
        assert r[-1].item() == 0.0

    def test_thought_tags_also_work(self):
        meta = {"answer": "wrong", "response_text": "<thought>hmm</thought>nothing"}
        r, _ = reasoning_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == pytest.approx(0.2)


# ============================================================
# multimodal_reward_func
# ============================================================
class TestMultimodalRewardFunc:
    def test_correct_numeric(self):
        meta = {"answer": "42", "response_text": "The result is #### 42"}
        r, _ = multimodal_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == 1.0

    def test_correct_string_match(self):
        meta = {"answer": "cat", "response_text": "I see a cat in the image"}
        r, _ = multimodal_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == 1.0

    def test_modality_awareness_fallback(self):
        meta = {"answer": "cat", "response_text": "I see an image but can't tell"}
        r, _ = multimodal_reward_func([1], [2, 3], "stop", meta)
        assert r[-1].item() == pytest.approx(0.2)

    def test_no_metadata(self):
        r, _ = multimodal_reward_func([1], [2], "stop", None)
        assert r[-1].item() == 0.0


# ============================================================
# rm_reward_func
# ============================================================
class TestRmRewardFunc:
    def test_error_when_model_not_loaded(self):
        import oxrl.rewards.rm_reward as rm_mod
        # Reset global state
        rm_mod._rm_model = None
        rm_mod._rm_value_head = None
        with pytest.raises(RuntimeError, match="Reward model not loaded"):
            rm_mod.rm_reward_func([1, 2], [3, 4], "stop")

    def test_empty_response(self):
        import oxrl.rewards.rm_reward as rm_mod
        rm_mod._rm_model = None
        rm_mod._rm_value_head = None
        r, is_per_token = rm_mod.rm_reward_func([1, 2], [], "stop")
        assert len(r) == 0
        assert is_per_token is False

    def test_mock_model(self):
        import oxrl.rewards.rm_reward as rm_mod
        # Create mock model and value head
        mock_model = MagicMock()
        mock_param = torch.zeros(1)
        mock_model.parameters.return_value = iter([mock_param])
        hidden = torch.randn(1, 5, 64)
        mock_output = MagicMock()
        mock_output.hidden_states = [hidden]
        mock_model.return_value = mock_output

        mock_vh = MagicMock()
        mock_vh.return_value = torch.tensor([[0.5, 0.3, 0.1, 0.8, 0.9]])

        rm_mod._rm_model = mock_model
        rm_mod._rm_value_head = mock_vh

        r, is_per_token = rm_mod.rm_reward_func([1, 2], [3, 4, 5], "stop")
        assert r.shape == (3,)
        assert is_per_token is False
        assert r[-1].item() == pytest.approx(0.9, abs=0.01)

        # Cleanup
        rm_mod._rm_model = None
        rm_mod._rm_value_head = None


# ============================================================
# Cross-cutting contract tests
# ============================================================
class TestRewardContracts:
    """All reward functions should return (Tensor, bool) with correct shape."""

    FUNCS = [
        (default_reward_func, {}),
        (gsm8k_reward_func, {"answer": "1", "response_text": "1"}),
        (math_reward_func, {"answer": "1", "response_text": "\boxed{1}"}),
        (soft_math_reward_func, {"answer": "1", "response_text": "1"}),
        (format_reward_func, {"response_text": "hello"}),
        (mcqa_reward_func, {"answer": "A", "response_text": "A"}),
        (reasoning_reward_func, {"answer": "1", "response_text": "1"}),
        (multimodal_reward_func, {"answer": "1", "response_text": "1"}),
    ]

    @pytest.mark.parametrize("func,meta", FUNCS)
    def test_returns_tuple(self, func, meta):
        result = func([1], [2, 3, 4], "stop", meta)
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.parametrize("func,meta", FUNCS)
    def test_tensor_shape(self, func, meta):
        r, _ = func([1], [2, 3, 4], "stop", meta)
        assert r.shape == (3,)

    @pytest.mark.parametrize("func,meta", FUNCS)
    def test_empty_response(self, func, meta):
        r, _ = func([1], [], "stop", meta)
        assert len(r) == 0

    @pytest.mark.parametrize("func,meta", FUNCS)
    def test_is_per_token_is_bool(self, func, meta):
        _, is_per_token = func([1], [2], "stop", meta)
        assert isinstance(is_per_token, bool)
