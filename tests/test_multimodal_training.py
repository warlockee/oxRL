"""
Tests for multimodal VLM training pipeline.
Covers: ReplayBuffer multimodal metadata, _prepare_mm_inputs, _freeze_vision_encoder,
schema VLM fields, and rollout sample propagation.
All tests run on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_multimodal_training.py -v
"""
import pytest
import sys
import os
import base64
import io
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from typing import Dict, Any, Optional, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# ReplayBuffer — multimodal metadata storage and collation
# ============================================================
from oxrl.rollouts.replay_buffer import ReplayBuffer


def _make_tensors(seq_len=10):
    """Create a consistent set of tensors for replay buffer add()."""
    return {
        "input_ids": torch.randint(0, 1000, (seq_len,)),
        "rewards": torch.randn(seq_len),
        "zscores": torch.randn(seq_len),
        "masks": torch.ones(seq_len),
        "dones": torch.zeros(seq_len),
        "old_logprobs": torch.randn(seq_len),
    }


class TestReplayBufferMultimodal:
    def test_add_without_multimodal(self):
        """Buffer works normally when multi_modal_data is None (default)."""
        buf = ReplayBuffer(pad_token_id=0, max_seq_len=128)
        buf.add(**_make_tensors())
        assert len(buf) == 1
        assert buf.items[0]["multi_modal_data"] is None

    def test_add_with_multimodal(self):
        """Buffer stores base64 image metadata."""
        buf = ReplayBuffer(pad_token_id=0, max_seq_len=128)
        mm_data = {"image": base64.b64encode(b"fake_png_data").decode()}
        buf.add(**_make_tensors(), multi_modal_data=mm_data)
        assert len(buf) == 1
        assert buf.items[0]["multi_modal_data"] is mm_data

    def test_collate_no_multimodal(self):
        """Collated batch has multi_modal_data=None when no samples have it."""
        buf = ReplayBuffer(pad_token_id=0, max_seq_len=128)
        buf.add(**_make_tensors(8))
        buf.add(**_make_tensors(8))
        batch = buf.collate_fn([buf[0], buf[1]])
        assert batch["multi_modal_data"] is None

    def test_collate_with_multimodal(self):
        """Collated batch includes multimodal metadata list when present."""
        buf = ReplayBuffer(pad_token_id=0, max_seq_len=128)
        mm1 = {"image": "abc123"}
        mm2 = {"image": "def456"}
        buf.add(**_make_tensors(8), multi_modal_data=mm1)
        buf.add(**_make_tensors(8), multi_modal_data=mm2)
        batch = buf.collate_fn([buf[0], buf[1]])
        assert batch["multi_modal_data"] is not None
        assert len(batch["multi_modal_data"]) == 2
        assert batch["multi_modal_data"][0] is mm1
        assert batch["multi_modal_data"][1] is mm2

    def test_collate_mixed_multimodal(self):
        """Mixed batch (some None, some with data) activates mm list."""
        buf = ReplayBuffer(pad_token_id=0, max_seq_len=128)
        mm = {"image": "abc123"}
        buf.add(**_make_tensors(8), multi_modal_data=None)
        buf.add(**_make_tensors(8), multi_modal_data=mm)
        batch = buf.collate_fn([buf[0], buf[1]])
        assert batch["multi_modal_data"] is not None
        assert batch["multi_modal_data"][0] is None
        assert batch["multi_modal_data"][1] is mm

    def test_add_batch_seqs_with_multimodal(self):
        """add_batch_seqs correctly passes multi_modal_data through."""
        buf = ReplayBuffer(pad_token_id=0, max_seq_len=128)
        mm_data = {"image": "test_b64"}
        t = _make_tensors(8)
        sample = {
            "input_ids": t["input_ids"],
            "rewards": t["rewards"],
            "pred_zscores": t["zscores"],
            "pred_masks": t["masks"],
            "pred_dones": t["dones"],
            "pred_old_logprobs": t["old_logprobs"],
            "response_len": 8,
            "multi_modal_data": mm_data,
        }
        buf.add_batch_seqs([sample])
        assert len(buf) == 1
        assert buf.items[0]["multi_modal_data"] is mm_data


# ============================================================
# _freeze_vision_encoder — vision encoder freezing
# ============================================================
from oxrl.algs.base import BaseAlgorithm


class ConcreteAlgorithm(BaseAlgorithm):
    """Minimal concrete subclass for testing base class helpers."""

    def __init__(self):
        self.processor = None

    def is_ready(self):
        return True

    def train_step(self, *args, **kwargs):
        return {}

    def save_checkpoint(self, output_dir, tag, state_dict_ref=None):
        pass


class TestFreezeVisionEncoder:
    def test_freeze_vision_tower(self):
        """Freezes params when model has a vision_tower attribute."""
        alg = ConcreteAlgorithm()
        model = MagicMock()
        vision_tower = nn.Linear(10, 10)
        # Verify params are trainable before freeze
        assert all(p.requires_grad for p in vision_tower.parameters())
        model.vision_tower = vision_tower
        model.visual = None
        model.vision_model = None
        model.vision_encoder = None
        model.model = None

        alg._freeze_vision_encoder(model)

        # All params should be frozen
        for p in vision_tower.parameters():
            assert not p.requires_grad

    def test_freeze_nested_vision_model(self):
        """Freezes params when vision encoder is on model.model (HF wrapper pattern)."""
        alg = ConcreteAlgorithm()
        model = MagicMock()
        vision_model = nn.Linear(10, 10)
        model.vision_tower = None
        model.visual = None
        model.vision_model = None
        model.vision_encoder = None
        inner = MagicMock()
        inner.vision_tower = None
        inner.visual = None
        inner.vision_model = vision_model
        inner.vision_encoder = None
        model.model = inner

        alg._freeze_vision_encoder(model)

        for p in vision_model.parameters():
            assert not p.requires_grad

    def test_warns_when_no_vision_encoder(self, capsys):
        """Prints warning when no vision encoder attribute is found."""
        alg = ConcreteAlgorithm()
        model = MagicMock()
        model.vision_tower = None
        model.visual = None
        model.vision_model = None
        model.vision_encoder = None
        model.model = MagicMock()
        model.model.vision_tower = None
        model.model.visual = None
        model.model.vision_model = None
        model.model.vision_encoder = None

        alg._freeze_vision_encoder(model)

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "Could not find vision encoder" in captured.out


# ============================================================
# _prepare_mm_inputs — base64 to model-ready tensors
# ============================================================


def _create_test_image_b64(width=4, height=4):
    """Create a small PNG image and return its base64 encoding."""
    from PIL import Image
    img = Image.new("RGB", (width, height), (128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class TestPrepareMMInputs:
    def test_all_none_returns_empty(self):
        """Returns empty dict when all multimodal entries are None."""
        alg = ConcreteAlgorithm()
        result = alg._prepare_mm_inputs([None, None, None], "cpu")
        assert result == {}

    def test_none_input_returns_empty(self):
        """Returns empty dict when multi_modal_data_list is None."""
        alg = ConcreteAlgorithm()
        result = alg._prepare_mm_inputs(None, "cpu")
        assert result == {}

    def test_image_processing(self):
        """Base64 image is decoded and processed into pixel_values."""
        alg = ConcreteAlgorithm()

        # Mock processor with image_processor
        mock_processor = MagicMock()
        pixel_values = torch.randn(1, 3, 224, 224)
        mock_processor.image_processor.return_value = {"pixel_values": pixel_values}
        alg.processor = mock_processor

        img_b64 = _create_test_image_b64()
        mm_list = [{"image": img_b64}]
        result = alg._prepare_mm_inputs(mm_list, "cpu")

        assert "pixel_values" in result
        mock_processor.image_processor.assert_called_once()
        # Verify PIL images were passed
        call_args = mock_processor.image_processor.call_args
        assert "images" in call_args.kwargs
        assert len(call_args.kwargs["images"]) == 1

    def test_mixed_batch_with_dummy_fill(self):
        """Mixed batch (some None, some image) fills None with dummy black image."""
        alg = ConcreteAlgorithm()

        mock_processor = MagicMock()
        pixel_values = torch.randn(2, 3, 224, 224)
        mock_processor.image_processor.return_value = {"pixel_values": pixel_values}
        alg.processor = mock_processor

        img_b64 = _create_test_image_b64()
        mm_list = [None, {"image": img_b64}]
        result = alg._prepare_mm_inputs(mm_list, "cpu")

        assert "pixel_values" in result
        # Both entries should be passed (one real, one dummy)
        call_args = mock_processor.image_processor.call_args
        images = call_args.kwargs["images"]
        assert len(images) == 2
        # First should be dummy black image, second should be the real one
        from PIL import Image
        assert isinstance(images[0], Image.Image)
        assert isinstance(images[1], Image.Image)

    def test_tensors_moved_to_device(self):
        """Output tensors are moved to the specified device."""
        alg = ConcreteAlgorithm()

        mock_processor = MagicMock()
        pixel_values = torch.randn(1, 3, 4, 4)
        mock_processor.image_processor.return_value = {"pixel_values": pixel_values}
        alg.processor = mock_processor

        img_b64 = _create_test_image_b64()
        result = alg._prepare_mm_inputs([{"image": img_b64}], "cpu")

        assert result["pixel_values"].device == torch.device("cpu")


# ============================================================
# Schema — VLM fields
# ============================================================
from pydantic import ValidationError
from oxrl.configs.schema import Model


class TestSchemaVLM:
    def test_model_class_default(self):
        """model_class defaults to 'llm'."""
        m = Model(name="test/model")
        assert m.model_class == "llm"

    def test_model_class_vlm(self):
        """model_class can be set to 'vlm'."""
        m = Model(name="test/model", model_class="vlm")
        assert m.model_class == "vlm"

    def test_freeze_vision_encoder_default(self):
        """freeze_vision_encoder defaults to True."""
        m = Model(name="test/model")
        assert m.freeze_vision_encoder is True

    def test_freeze_vision_encoder_false(self):
        """freeze_vision_encoder can be set to False."""
        m = Model(name="test/model", freeze_vision_encoder=False)
        assert m.freeze_vision_encoder is False

    def test_extra_field_forbidden(self):
        """Extra fields are forbidden on Model."""
        with pytest.raises(ValidationError):
            Model(name="test/model", nonexistent_field="x")
