"""
Comprehensive tests for dataset classes in oxrl/datasets/.
Uses MockTokenizer to avoid downloading real models.
All tests run on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_datasets.py -v
"""
import pytest
import torch
import sys
import os
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd


# ============================================================
# MockTokenizer — no model download needed
# ============================================================
class MockTokenizer:
    """Minimal tokenizer mock that encodes each character as its ordinal."""
    def __init__(self, pad_id=0, eos_id=1):
        self.pad_token_id = pad_id
        self.eos_token_id = eos_id
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def apply_chat_template(self, conversation, add_generation_prompt=True,
                            tokenize=True, return_tensors=None):
        # Reconstruct text from conversation
        text = ""
        for msg in conversation:
            text += msg.get("content", "") + " "
        text = text.strip()

        if not tokenize:
            return text

        ids = [ord(c) % 100 + 2 for c in text]  # ids >= 2 to avoid pad/eos
        if return_tensors == 'pt':
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = [ord(c) % 100 + 2 for c in text]
        attn = [1] * len(ids)
        if return_tensors == 'pt':
            return {
                'input_ids': torch.tensor([ids], dtype=torch.long),
                'attention_mask': torch.tensor([attn], dtype=torch.long),
            }
        return {'input_ids': ids, 'attention_mask': attn}

    def encode(self, text, add_special_tokens=True):
        return [ord(c) % 100 + 2 for c in text]

    def add_special_tokens(self, special_tokens_dict):
        if 'pad_token' in special_tokens_dict:
            self.pad_token = special_tokens_dict['pad_token']


def _make_parquet(data, tmpdir):
    """Write data to a parquet file and return the path."""
    path = os.path.join(tmpdir, "data.parquet")
    df = pd.DataFrame(data)
    df.to_parquet(path, index=False)
    return path


# ============================================================
# PromptOnlyDataset
# ============================================================
class TestPromptOnlyDataset:
    def _make_dataset(self, tmpdir, data=None, **kwargs):
        from oxrl.datasets.prompt_only import PromptOnlyDataset
        if data is None:
            data = [
                {"prompt": [{"role": "user", "content": "Hello"}]},
                {"prompt": [{"role": "user", "content": "World"}]},
            ]
        path = _make_parquet(data, tmpdir)
        defaults = dict(
            prompt_key="prompt", tokenizer=MockTokenizer(),
            max_seq_len=128, data_path=path,
        )
        defaults.update(kwargs)
        return PromptOnlyDataset(**defaults)

    def test_construction(self, tmp_path):
        ds = self._make_dataset(str(tmp_path))
        assert len(ds) == 2

    def test_getitem_returns_prompt_token_ids(self, tmp_path):
        ds = self._make_dataset(str(tmp_path))
        item = ds[0]
        assert "prompt_token_ids" in item
        assert isinstance(item["prompt_token_ids"], list)
        assert len(item["prompt_token_ids"]) > 0

    def test_truncation(self, tmp_path):
        data = [{"prompt": [{"role": "user", "content": "A" * 200}]}]
        ds = self._make_dataset(str(tmp_path), data=data, max_seq_len=10)
        item = ds[0]
        assert len(item["prompt_token_ids"]) <= 9  # max_seq_len - 1

    def test_metadata_passthrough(self, tmp_path):
        data = [{"prompt": [{"role": "user", "content": "Hi"}], "answer": "42"}]
        ds = self._make_dataset(str(tmp_path), data=data, answer_key="answer")
        item = ds[0]
        assert "metadata" in item
        assert item["metadata"]["answer"] == "42"

    def test_len(self, tmp_path):
        data = [{"prompt": [{"role": "user", "content": f"Q{i}"}]} for i in range(5)]
        ds = self._make_dataset(str(tmp_path), data=data)
        assert len(ds) == 5

    def test_empty_prompt_key_raises(self, tmp_path):
        data = [{"prompt": [{"role": "user", "content": "Hi"}]}]
        path = _make_parquet(data, str(tmp_path))
        from oxrl.datasets.prompt_only import PromptOnlyDataset
        with pytest.raises(AssertionError, match="prompt_key cannot be empty"):
            PromptOnlyDataset(prompt_key="", tokenizer=MockTokenizer(),
                              max_seq_len=128, data_path=path)

    def test_none_tokenizer_raises(self, tmp_path):
        data = [{"prompt": [{"role": "user", "content": "Hi"}]}]
        path = _make_parquet(data, str(tmp_path))
        from oxrl.datasets.prompt_only import PromptOnlyDataset
        with pytest.raises(AssertionError, match="tokenizer cannot be None"):
            PromptOnlyDataset(prompt_key="prompt", tokenizer=None,
                              max_seq_len=128, data_path=path)

    def test_zero_seq_len_raises(self, tmp_path):
        data = [{"prompt": [{"role": "user", "content": "Hi"}]}]
        path = _make_parquet(data, str(tmp_path))
        from oxrl.datasets.prompt_only import PromptOnlyDataset
        with pytest.raises(AssertionError, match="max_seq_len must be > 0"):
            PromptOnlyDataset(prompt_key="prompt", tokenizer=MockTokenizer(),
                              max_seq_len=0, data_path=path)

    def test_prompt_structured_in_result(self, tmp_path):
        ds = self._make_dataset(str(tmp_path))
        item = ds[0]
        assert "prompt_structured" in item


# ============================================================
# PromptResponseDataset
# ============================================================
class TestPromptResponseDataset:
    def _make_dataset(self, tmpdir, data=None, **kwargs):
        from oxrl.datasets.prompt_response import PromptResponseDataset
        if data is None:
            data = [
                {"prompt": [{"role": "user", "content": "Hello"}], "answer": "Hi there"},
                {"prompt": [{"role": "user", "content": "Bye"}], "answer": "Goodbye"},
            ]
        path = _make_parquet(data, tmpdir)
        defaults = dict(
            prompt_key="prompt", answer_key="answer",
            tokenizer=MockTokenizer(), max_seq_len=128, data_path=path,
        )
        defaults.update(kwargs)
        return PromptResponseDataset(**defaults)

    def test_construction(self, tmp_path):
        ds = self._make_dataset(str(tmp_path))
        assert len(ds) == 2

    def test_getitem_keys(self, tmp_path):
        ds = self._make_dataset(str(tmp_path))
        item = ds[0]
        assert "input_ids" in item
        assert "attn_mask" in item
        assert "loss_mask" in item

    def test_output_length(self, tmp_path):
        ds = self._make_dataset(str(tmp_path), max_seq_len=64)
        item = ds[0]
        assert item["input_ids"].shape[0] == 64
        assert item["attn_mask"].shape[0] == 64
        assert item["loss_mask"].shape[0] == 63  # T-1

    def test_loss_mask_zeros_for_prompt(self, tmp_path):
        ds = self._make_dataset(str(tmp_path), max_seq_len=128)
        item = ds[0]
        # prompt_ids would be from apply_chat_template on "Hello"
        # The first several positions of loss_mask should be 0
        # At least position 0 should be masked
        assert item["loss_mask"][0].item() == 0

    def test_loss_mask_ones_for_answer(self, tmp_path):
        ds = self._make_dataset(str(tmp_path), max_seq_len=128)
        item = ds[0]
        # Some answer tokens should be unmasked
        assert item["loss_mask"].sum().item() > 0

    def test_padding_correct(self, tmp_path):
        ds = self._make_dataset(str(tmp_path), max_seq_len=256)
        item = ds[0]
        # With large max_seq_len, there should be padding
        # Padding tokens have attn_mask = 0
        assert (item["attn_mask"] == 0).any()

    def test_truncation(self, tmp_path):
        data = [{"prompt": [{"role": "user", "content": "A" * 10}], "answer": "B" * 10}]
        ds = self._make_dataset(str(tmp_path), data=data, max_seq_len=15)
        item = ds[0]
        assert item["input_ids"].shape[0] == 15

    def test_empty_answer_key_raises(self, tmp_path):
        data = [{"prompt": [{"role": "user", "content": "Hi"}], "answer": "ok"}]
        path = _make_parquet(data, str(tmp_path))
        from oxrl.datasets.prompt_response import PromptResponseDataset
        with pytest.raises(AssertionError, match="answer_key cannot be empty"):
            PromptResponseDataset(prompt_key="prompt", answer_key="",
                                  tokenizer=MockTokenizer(), max_seq_len=128, data_path=path)

    def test_none_tokenizer_raises(self, tmp_path):
        data = [{"prompt": [{"role": "user", "content": "Hi"}], "answer": "ok"}]
        path = _make_parquet(data, str(tmp_path))
        from oxrl.datasets.prompt_response import PromptResponseDataset
        with pytest.raises(AssertionError, match="tokenizer cannot be None"):
            PromptResponseDataset(prompt_key="prompt", answer_key="answer",
                                  tokenizer=None, max_seq_len=128, data_path=path)

    def test_input_ids_dtype(self, tmp_path):
        ds = self._make_dataset(str(tmp_path))
        item = ds[0]
        assert item["input_ids"].dtype == torch.long

    def test_eos_appended(self, tmp_path):
        ds = self._make_dataset(str(tmp_path), max_seq_len=256)
        item = ds[0]
        # EOS token (id=1) should appear somewhere in the sequence
        assert (item["input_ids"] == 1).any()


# ============================================================
# PromptPreferenceDataset
# ============================================================
class TestPromptPreferenceDataset:
    def _make_dataset(self, tmpdir, data=None, **kwargs):
        from oxrl.datasets.prompt_preference import PromptPreferenceDataset
        if data is None:
            data = [
                {"prompt": [{"role": "user", "content": "Hello"}],
                 "chosen": "Good answer", "rejected": "Bad answer"},
                {"prompt": [{"role": "user", "content": "Bye"}],
                 "chosen": "See you", "rejected": "Whatever"},
            ]
        path = _make_parquet(data, tmpdir)
        defaults = dict(
            prompt_key="prompt", chosen_key="chosen", rejected_key="rejected",
            tokenizer=MockTokenizer(), max_seq_len=128, data_path=path,
        )
        defaults.update(kwargs)
        return PromptPreferenceDataset(**defaults)

    def test_construction(self, tmp_path):
        ds = self._make_dataset(str(tmp_path))
        assert len(ds) == 2

    def test_getitem_has_6_keys(self, tmp_path):
        ds = self._make_dataset(str(tmp_path))
        item = ds[0]
        expected_keys = {"chosen_input_ids", "chosen_attn_mask", "chosen_loss_mask",
                         "rejected_input_ids", "rejected_attn_mask", "rejected_loss_mask"}
        assert set(item.keys()) == expected_keys

    def test_consistent_lengths(self, tmp_path):
        ds = self._make_dataset(str(tmp_path), max_seq_len=64)
        item = ds[0]
        assert item["chosen_input_ids"].shape[0] == 64
        assert item["rejected_input_ids"].shape[0] == 64
        assert item["chosen_attn_mask"].shape[0] == 64
        assert item["rejected_attn_mask"].shape[0] == 64
        assert item["chosen_loss_mask"].shape[0] == 63
        assert item["rejected_loss_mask"].shape[0] == 63

    def test_eos_in_both(self, tmp_path):
        ds = self._make_dataset(str(tmp_path), max_seq_len=256)
        item = ds[0]
        assert (item["chosen_input_ids"] == 1).any()  # EOS in chosen
        assert (item["rejected_input_ids"] == 1).any()  # EOS in rejected

    def test_prompt_masked_in_both(self, tmp_path):
        ds = self._make_dataset(str(tmp_path), max_seq_len=128)
        item = ds[0]
        assert item["chosen_loss_mask"][0].item() == 0
        assert item["rejected_loss_mask"][0].item() == 0

    def test_empty_chosen_key_raises(self, tmp_path):
        data = [{"prompt": [{"role": "user", "content": "Hi"}],
                 "chosen": "ok", "rejected": "no"}]
        path = _make_parquet(data, str(tmp_path))
        from oxrl.datasets.prompt_preference import PromptPreferenceDataset
        with pytest.raises(AssertionError, match="chosen_key cannot be empty"):
            PromptPreferenceDataset(
                prompt_key="prompt", chosen_key="", rejected_key="rejected",
                tokenizer=MockTokenizer(), max_seq_len=128, data_path=path)

    def test_none_tokenizer_raises(self, tmp_path):
        data = [{"prompt": [{"role": "user", "content": "Hi"}],
                 "chosen": "ok", "rejected": "no"}]
        path = _make_parquet(data, str(tmp_path))
        from oxrl.datasets.prompt_preference import PromptPreferenceDataset
        with pytest.raises(AssertionError, match="tokenizer cannot be None"):
            PromptPreferenceDataset(
                prompt_key="prompt", chosen_key="chosen", rejected_key="rejected",
                tokenizer=None, max_seq_len=128, data_path=path)


# ============================================================
# MixedRatioSampler
# ============================================================
class TestMixedRatioSampler:
    def _make_sampler(self, **kwargs):
        from oxrl.datasets.mixed_ratio_sampler import MixedRatioSampler
        defaults = dict(
            seed=42, dnames=["d1", "d2"],
            ratios={"d1": 0.7, "d2": 0.3},
            local_batch_size=10, steps_per_epoch=5,
            len_datasets={"d1": 100, "d2": 50},
            shuffle_within_batch=False,
        )
        defaults.update(kwargs)
        return MixedRatioSampler(**defaults)

    def test_construction(self):
        sampler = self._make_sampler()
        assert len(sampler) == 5

    def test_construction_validation_mismatch(self):
        from oxrl.datasets.mixed_ratio_sampler import MixedRatioSampler
        with pytest.raises(AssertionError):
            MixedRatioSampler(
                seed=42, dnames=["d1"], ratios={"d1": 0.5, "d2": 0.5},
                local_batch_size=10, steps_per_epoch=5,
                len_datasets={"d1": 100, "d2": 50},
                shuffle_within_batch=False,
            )

    def test_fixed_ratio_allocation(self):
        sampler = self._make_sampler(dynamic_ratio_every_step=False)
        # With dynamic_ratio_every_step=False and 2 datasets < batch_size 10,
        # uses fixed ratio: d1 gets 7, d2 gets 3 (largest remainder)
        assert sampler.sample_per_dataset["d1"] == 7
        assert sampler.sample_per_dataset["d2"] == 3

    def test_batch_count(self):
        sampler = self._make_sampler(steps_per_epoch=10)
        batches = list(sampler)
        assert len(batches) == 10

    def test_batch_size(self):
        sampler = self._make_sampler(local_batch_size=8)
        batches = list(sampler)
        for batch in batches:
            assert len(batch) == 8

    def test_index_bounds(self):
        sampler = self._make_sampler()
        for batch in sampler:
            for idx in batch:
                assert 0 <= idx < 150  # total = 100 + 50

    def test_shuffle_control(self):
        sampler_no_shuffle = self._make_sampler(shuffle_within_batch=False, seed=123)
        sampler_shuffle = self._make_sampler(shuffle_within_batch=True, seed=123)
        # Both produce batches of same size
        b1 = list(sampler_no_shuffle)
        b2 = list(sampler_shuffle)
        assert len(b1) == len(b2)

    def test_reproducibility_same_seed(self):
        s1 = self._make_sampler(seed=42)
        s2 = self._make_sampler(seed=42)
        b1 = list(s1)
        b2 = list(s2)
        assert b1 == b2

    def test_different_ranks(self):
        s0 = self._make_sampler(rank=0, world_size=2)
        s1 = self._make_sampler(rank=1, world_size=2)
        b0 = list(s0)
        b1 = list(s1)
        # Different ranks should produce different batches
        assert b0 != b1

    def test_epoch_reshuffling(self):
        sampler = self._make_sampler()
        b_epoch0 = list(sampler)
        sampler.set_epoch(1)
        b_epoch1 = list(sampler)
        # Different epochs should (with high probability) produce different batches
        assert b_epoch0 != b_epoch1

    def test_dynamic_ratio_fallback(self):
        # When num_datasets > local_batch_size, force dynamic
        sampler = self._make_sampler(
            dnames=["d1", "d2", "d3"],
            ratios={"d1": 0.5, "d2": 0.3, "d3": 0.2},
            len_datasets={"d1": 100, "d2": 50, "d3": 30},
            local_batch_size=2,
            dynamic_ratio_every_step=False,
        )
        # redo_ratio_every_step should be True because num_datasets(3) > batch_size(2)
        assert sampler.redo_ratio_every_step is True

    def test_invalid_rank(self):
        from oxrl.datasets.mixed_ratio_sampler import MixedRatioSampler
        with pytest.raises(AssertionError):
            MixedRatioSampler(
                seed=42, dnames=["d1"], ratios={"d1": 1.0},
                local_batch_size=4, steps_per_epoch=1,
                len_datasets={"d1": 10}, shuffle_within_batch=False,
                rank=2, world_size=2,  # rank must be < world_size
            )

    def test_single_dataset(self):
        sampler = self._make_sampler(
            dnames=["d1"], ratios={"d1": 1.0},
            len_datasets={"d1": 100}, local_batch_size=4,
        )
        batches = list(sampler)
        assert len(batches) == 5
        for batch in batches:
            assert len(batch) == 4
            for idx in batch:
                assert 0 <= idx < 100
