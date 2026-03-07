"""
Comprehensive tests for utility functions in oxrl/utils/.
All tests run on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_utils.py -v
"""
import pytest
import torch
import numpy as np
import random
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# set_random_seeds
# ============================================================
class TestSetRandomSeeds:
    def test_reproducibility_random(self):
        from oxrl.utils.setup import set_random_seeds
        set_random_seeds(42)
        a = random.random()
        set_random_seeds(42)
        b = random.random()
        assert a == b

    def test_reproducibility_numpy(self):
        from oxrl.utils.setup import set_random_seeds
        set_random_seeds(42)
        a = np.random.rand()
        set_random_seeds(42)
        b = np.random.rand()
        assert a == b

    def test_reproducibility_torch(self):
        from oxrl.utils.setup import set_random_seeds
        set_random_seeds(42)
        a = torch.rand(1).item()
        set_random_seeds(42)
        b = torch.rand(1).item()
        assert a == b

    def test_different_seeds_differ(self):
        from oxrl.utils.setup import set_random_seeds
        set_random_seeds(42)
        a = random.random()
        set_random_seeds(99)
        b = random.random()
        assert a != b


# ============================================================
# get_rank_info
# ============================================================
class TestGetRankInfo:
    @patch.dict(os.environ, {"RANK": "0", "LOCAL_RANK": "0"})
    @patch("torch.cuda.is_available", return_value=False)
    def test_cpu_fallback(self, mock_cuda):
        from oxrl.utils.setup import get_rank_info
        rank, local_rank = get_rank_info()
        assert rank == 0
        assert local_rank == 0

    @patch.dict(os.environ, {"RANK": "3", "LOCAL_RANK": "1"})
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=4)
    @patch("torch.cuda.set_device")
    def test_reads_env_vars(self, mock_set, mock_count, mock_avail):
        from oxrl.utils.setup import get_rank_info
        rank, local_rank = get_rank_info()
        assert rank == 3
        assert local_rank == 1
        mock_set.assert_called_once_with(1)

    @patch.dict(os.environ, {"RANK": "0", "LOCAL_RANK": "4"})
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=4)
    def test_local_rank_exceeds_gpus(self, mock_count, mock_avail):
        from oxrl.utils.setup import get_rank_info
        with pytest.raises(RuntimeError, match="LOCAL_RANK 4 >= available GPUs 4"):
            get_rank_info()

    @patch.dict(os.environ, {}, clear=True)
    @patch("torch.cuda.is_available", return_value=False)
    def test_defaults(self, mock_cuda):
        # Remove RANK and LOCAL_RANK from env
        os.environ.pop("RANK", None)
        os.environ.pop("LOCAL_RANK", None)
        from oxrl.utils.setup import get_rank_info
        rank, local_rank = get_rank_info()
        assert rank == 0
        assert local_rank == 0


# ============================================================
# get_distributed_info
# ============================================================
class TestGetDistributedInfo:
    @patch.dict(os.environ, {"WORLD_SIZE": "8", "RANK": "3", "LOCAL_RANK": "1"})
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=4)
    @patch("torch.cuda.set_device")
    def test_reads_env_vars(self, mock_set, mock_count, mock_avail):
        from oxrl.utils.setup import get_distributed_info
        rank, world_size, local_rank = get_distributed_info()
        assert rank == 3
        assert world_size == 8
        assert local_rank == 1

    @patch("torch.cuda.is_available", return_value=False)
    def test_defaults(self, mock_cuda):
        env = os.environ.copy()
        for k in ["WORLD_SIZE", "RANK", "LOCAL_RANK"]:
            env.pop(k, None)
        with patch.dict(os.environ, env, clear=True):
            from oxrl.utils.setup import get_distributed_info
            rank, world_size, local_rank = get_distributed_info()
            assert rank == 0
            assert world_size == 1
            assert local_rank == 0


# ============================================================
# load_tokenizer
# ============================================================
class TestLoadTokenizer:
    @patch("oxrl.utils.setup.AutoTokenizer")
    def test_basic_loading(self, mock_auto):
        mock_tok = MagicMock()
        mock_tok.pad_token_id = 0
        mock_auto.from_pretrained.return_value = mock_tok
        from oxrl.utils.setup import load_tokenizer
        tok = load_tokenizer("test-model")
        mock_auto.from_pretrained.assert_called_once_with("test-model", trust_remote_code=False)
        assert tok == mock_tok

    @patch("oxrl.utils.setup.AutoTokenizer")
    def test_patches_missing_pad_token(self, mock_auto):
        mock_tok = MagicMock()
        mock_tok.pad_token_id = None
        mock_tok.eos_token = "<eos>"
        mock_auto.from_pretrained.return_value = mock_tok
        from oxrl.utils.setup import load_tokenizer
        tok = load_tokenizer("test-model", rank=0)
        mock_tok.add_special_tokens.assert_called_once_with({'pad_token': '<eos>'})

    @patch("oxrl.utils.setup.AutoTokenizer")
    def test_pad_token_fallback_eos_id(self, mock_auto):
        mock_tok = MagicMock()
        mock_tok.pad_token_id = None
        mock_tok.eos_token = None
        mock_tok.eos_token_id = 2
        mock_auto.from_pretrained.return_value = mock_tok
        from oxrl.utils.setup import load_tokenizer
        tok = load_tokenizer("test-model")
        assert tok.pad_token_id == 2

    @patch("oxrl.utils.setup.AutoTokenizer")
    def test_rank_zero_warning(self, mock_auto, capsys):
        mock_tok = MagicMock()
        mock_tok.pad_token_id = None
        mock_tok.eos_token = "<eos>"
        mock_auto.from_pretrained.return_value = mock_tok
        from oxrl.utils.setup import load_tokenizer
        load_tokenizer("test-model", rank=0)
        captured = capsys.readouterr()
        assert "Warning" in captured.out

    @patch("oxrl.utils.setup.AutoTokenizer")
    def test_non_rank_zero_no_warning(self, mock_auto, capsys):
        mock_tok = MagicMock()
        mock_tok.pad_token_id = None
        mock_tok.eos_token = "<eos>"
        mock_auto.from_pretrained.return_value = mock_tok
        from oxrl.utils.setup import load_tokenizer
        load_tokenizer("test-model", rank=1)
        captured = capsys.readouterr()
        assert "Warning" not in captured.out


# ============================================================
# load_model_and_ref
# ============================================================
class TestLoadModelAndRef:
    @patch("transformers.AutoConfig")
    @patch("transformers.AutoModelForCausalLM")
    def test_basic_loading(self, mock_causal, mock_config):
        mock_config.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_causal.from_pretrained.return_value = mock_model
        from oxrl.utils.setup import load_model_and_ref
        model, ref = load_model_and_ref("path", torch.bfloat16, False, "flash_attention_2")
        assert model == mock_model
        assert ref is None

    @patch("transformers.AutoConfig")
    @patch("transformers.AutoModelForCausalLM")
    def test_ref_model_loading(self, mock_causal, mock_config):
        mock_config.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_ref = MagicMock()
        mock_causal.from_pretrained.side_effect = [mock_model, mock_ref]
        from oxrl.utils.setup import load_model_and_ref
        model, ref = load_model_and_ref("path", torch.bfloat16, False, "",
                                        ref_model_path="ref-path")
        assert model == mock_model
        assert ref == mock_ref

    def test_fallback_chain(self):
        """Verify the fallback chain is structured correctly (source check)."""
        import inspect
        from oxrl.utils.setup import load_model_and_ref
        source = inspect.getsource(load_model_and_ref)
        assert "AutoModelForCausalLM" in source
        assert "AutoModelForImageTextToText" in source
        assert "AutoModel" in source
        # Ultimate fallback must use AutoModel.from_pretrained
        assert "AutoModel.from_pretrained" in source


# ============================================================
# safe_string_to_torch_dtype
# ============================================================
class TestSafeStringToTorchDtype:
    def test_fp16(self):
        from oxrl.utils.utils import safe_string_to_torch_dtype
        assert safe_string_to_torch_dtype("fp16") == torch.float16

    def test_float16(self):
        from oxrl.utils.utils import safe_string_to_torch_dtype
        assert safe_string_to_torch_dtype("float16") == torch.float16

    def test_bf16(self):
        from oxrl.utils.utils import safe_string_to_torch_dtype
        assert safe_string_to_torch_dtype("bf16") == torch.bfloat16

    def test_bfloat16(self):
        from oxrl.utils.utils import safe_string_to_torch_dtype
        assert safe_string_to_torch_dtype("bfloat16") == torch.bfloat16

    def test_fp32(self):
        from oxrl.utils.utils import safe_string_to_torch_dtype
        assert safe_string_to_torch_dtype("fp32") == torch.float32

    def test_fp64(self):
        from oxrl.utils.utils import safe_string_to_torch_dtype
        assert safe_string_to_torch_dtype("fp64") == torch.float64

    def test_case_insensitive(self):
        from oxrl.utils.utils import safe_string_to_torch_dtype
        assert safe_string_to_torch_dtype("FP16") == torch.float16
        assert safe_string_to_torch_dtype("BFloat16") == torch.bfloat16

    def test_passthrough_torch_dtype(self):
        from oxrl.utils.utils import safe_string_to_torch_dtype
        assert safe_string_to_torch_dtype(torch.float16) == torch.float16

    def test_none_returns_none(self):
        from oxrl.utils.utils import safe_string_to_torch_dtype
        assert safe_string_to_torch_dtype(None) is None

    def test_unsupported_raises(self):
        from oxrl.utils.utils import safe_string_to_torch_dtype
        with pytest.raises(ValueError, match="Unsupported"):
            safe_string_to_torch_dtype("int8")


# ============================================================
# ensure_* monkey-patches (idempotency)
# ============================================================
class TestEnsurePatches:
    def test_ensure_sliding_window_cache_idempotent(self):
        from oxrl.utils.setup import ensure_sliding_window_cache
        ensure_sliding_window_cache()
        ensure_sliding_window_cache()  # No crash

    def test_ensure_loss_kwargs_idempotent(self):
        from oxrl.utils.setup import ensure_loss_kwargs
        ensure_loss_kwargs()
        ensure_loss_kwargs()

    def test_ensure_pytorch_gelu_tanh_idempotent(self):
        from oxrl.utils.setup import ensure_pytorch_gelu_tanh
        ensure_pytorch_gelu_tanh()
        ensure_pytorch_gelu_tanh()

    def test_ensure_cache_usable_length_idempotent(self):
        from oxrl.utils.setup import ensure_cache_usable_length
        ensure_cache_usable_length()
        ensure_cache_usable_length()

    def test_ensure_auto_docstring_idempotent(self):
        from oxrl.utils.setup import ensure_auto_docstring_union_type
        ensure_auto_docstring_union_type()
        ensure_auto_docstring_union_type()


# ============================================================
# Tensor utils: ensure_1d, pad_1d_to_length
# ============================================================
class TestEnsure1d:
    def test_valid_1d(self):
        from oxrl.tools.tensor_utils import ensure_1d
        x = torch.tensor([1, 2, 3])
        result = ensure_1d(x, "test")
        assert torch.equal(result, x)

    def test_2d_raises(self):
        from oxrl.tools.tensor_utils import ensure_1d
        x = torch.tensor([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="Expected test to be 1D"):
            ensure_1d(x, "test")

    def test_0d_raises(self):
        from oxrl.tools.tensor_utils import ensure_1d
        x = torch.tensor(5)
        with pytest.raises(ValueError, match="Expected test to be 1D"):
            ensure_1d(x, "test")


class TestPad1dToLength:
    def test_pad(self):
        from oxrl.tools.tensor_utils import pad_1d_to_length
        x = torch.tensor([1, 2, 3])
        result = pad_1d_to_length(x, pad_value=0, target_len=5)
        assert result.shape == (5,)
        assert result[3].item() == 0
        assert result[4].item() == 0

    def test_truncate(self):
        from oxrl.tools.tensor_utils import pad_1d_to_length
        x = torch.tensor([1, 2, 3, 4, 5])
        result = pad_1d_to_length(x, pad_value=0, target_len=3)
        assert result.shape == (3,)
        assert torch.equal(result, torch.tensor([1, 2, 3]))

    def test_exact(self):
        from oxrl.tools.tensor_utils import pad_1d_to_length
        x = torch.tensor([1, 2, 3])
        result = pad_1d_to_length(x, pad_value=0, target_len=3)
        assert torch.equal(result, x)

    def test_pad_value(self):
        from oxrl.tools.tensor_utils import pad_1d_to_length
        x = torch.tensor([1.0])
        result = pad_1d_to_length(x, pad_value=-1.0, target_len=3)
        assert result[1].item() == -1.0
        assert result[2].item() == -1.0

    def test_preserves_dtype(self):
        from oxrl.tools.tensor_utils import pad_1d_to_length
        x = torch.tensor([1, 2], dtype=torch.long)
        result = pad_1d_to_length(x, pad_value=0, target_len=4)
        assert result.dtype == torch.long


# ============================================================
# get_experiment_dir_name
# ============================================================
class TestGetExperimentDirName:
    def test_basic(self):
        from oxrl.utils.utils import get_experiment_dir_name
        result = get_experiment_dir_name("/output", "v1", "exp1")
        assert result == "/output/exp1/v1"

    def test_nested_output_dir(self):
        from oxrl.utils.utils import get_experiment_dir_name
        result = get_experiment_dir_name("/a/b/c", "tag", "eid")
        assert result == "/a/b/c/eid/tag"
