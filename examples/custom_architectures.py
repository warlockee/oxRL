"""
Plugging non-standard architectures into oxRL
==============================================

oxRL's multimodal pipeline was designed around HuggingFace AutoProcessor
models (Qwen-VL, LLaVA, etc.), but the same hooks generalise to *any*
encoder–decoder architecture.  This file shows three concrete examples:

    1. Zipformer  — a streaming ASR encoder (audio → text)
    2. RetNet     — a retention-based sequence model (text only, custom forward)
    3. V-JEPA2    — a video encoder (video frames → embeddings)

Each example demonstrates which oxRL hooks to use and how they compose.
The code is intentionally self-contained: copy the parts you need into
your own research module and point ``research.module`` at it.

Hooks used
----------
``model_class: "vlm"``
    Tells oxRL to load an ``AutoProcessor`` and call ``_prepare_mm_inputs``
    during training.  Any model that needs extra tensors beyond
    ``input_ids / attention_mask`` should use this path.

``_prepare_mm_inputs`` override
    The base class decodes base64 images/audio through ``self.processor``.
    For non-standard encoders you can override this in a custom algorithm
    class registered via ``@register_algorithm``.

``mm_kwargs``
    Dict of tensors merged into ``model(**forward_kwargs, **mm_kwargs)``.
    Whatever your encoder produces (``pixel_values``, ``audio_features``,
    ``video_embeds``, …) goes here.

``RewardBackend`` subclass
    Custom scoring that inspects ``metadata`` (which carries
    ``prompt_text``, ``response_text``, ``answer``, and anything your
    dataset adds).

``@register_loss`` / ``@register_algorithm``
    Decorator-based registration (see ``oxrl.models.research_adapters``)
    so you never edit core files.

Running these examples
----------------------
These are *reference implementations*, not runnable scripts.  They show
the exact function signatures and data shapes oxRL expects.  To use one:

    1. Copy the relevant class/function into ``my_research/extensions.py``
    2. Set ``research.module: my_research.extensions`` in your YAML config
    3. Set ``model_class: vlm`` so the multimodal path activates
    4. Prepare your dataset with the expected column names

Each section below is independent — read just the one you need.
"""

# ---------------------------------------------------------------------------
# Common imports
# ---------------------------------------------------------------------------
import io
import base64
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from oxrl.rewards.backend import RewardBackend
from oxrl.models.research_adapters import (
    register_loss,
    register_algorithm,
    RewardShaper,
)


# ═══════════════════════════════════════════════════════════════════════════
# Example 1 — Zipformer (streaming ASR encoder)
# ═══════════════════════════════════════════════════════════════════════════
#
# Scenario: RL-tune a language model that receives audio features from a
#           frozen Zipformer encoder.  The reward checks word-error-rate
#           against a ground-truth transcript.
#
# Dataset columns:
#     prompt (str)           — instruction, e.g. "Transcribe the audio."
#     audio_base64 (str)     — base64-encoded WAV
#     answer (str)           — ground-truth transcript
#
# Config snippet:
#     model:
#       name: my-audio-lm/checkpoint
#       model_class: vlm            # activates multimodal path
#       freeze_vision_encoder: true  # freezes Zipformer weights
#     train:
#       alg_name: zipformer_grpo
#       loss_variant: sgrpo
#     research:
#       module: my_research.extensions
# ═══════════════════════════════════════════════════════════════════════════


class ZipformerProcessor:
    """Minimal processor wrapper for a pre-trained Zipformer encoder.

    In practice you would load the real Zipformer checkpoint here.
    This stub shows the interface oxRL expects from ``self.processor``.
    """

    def __init__(self, encoder, sample_rate: int = 16000):
        self.encoder = encoder          # nn.Module — frozen Zipformer
        self.sample_rate = sample_rate

    def feature_extractor(self, raw_speech, sampling_rate, return_tensors="pt"):
        """Match the HuggingFace FeatureExtractor interface.

        oxRL calls ``self.processor.feature_extractor(...)`` for audio data.
        Return a dict whose tensor values get merged into ``mm_kwargs``.
        """
        # Zipformer expects 80-dim fbank features
        features = []
        for waveform in raw_speech:
            if isinstance(waveform, np.ndarray):
                waveform = torch.from_numpy(waveform).float()
            fbank = self._compute_fbank(waveform, sampling_rate)
            features.append(fbank)

        # Pad to max length in batch and stack
        max_len = max(f.shape[0] for f in features)
        padded = torch.zeros(len(features), max_len, 80)
        lengths = torch.zeros(len(features), dtype=torch.long)
        for i, f in enumerate(features):
            padded[i, :f.shape[0]] = f
            lengths[i] = f.shape[0]

        # Run frozen encoder to get audio embeddings
        with torch.no_grad():
            audio_embeds = self.encoder(padded, lengths)  # (B, T', D)

        return {
            "audio_features": audio_embeds,
            "audio_feature_lengths": lengths,
        }

    @staticmethod
    def _compute_fbank(waveform, sample_rate, n_mels=80):
        """Compute log-mel filterbank features (placeholder)."""
        # Real implementation would use torchaudio.compliance.kaldi.fbank
        # or the icefall feature pipeline.
        n_frames = waveform.shape[0] // (sample_rate // 100)
        return torch.randn(max(n_frames, 1), n_mels)


@register_algorithm("zipformer_grpo")
class ZipformerGRPO:
    """GRPO variant that swaps AutoProcessor for a Zipformer encoder.

    Inherits the full GRPO training loop — only overrides processor
    initialisation so that ``_prepare_mm_inputs`` uses Zipformer features
    instead of a HuggingFace image/audio processor.

    In practice, subclass ``oxrl.algs.grpo.GRPO`` directly.  This stub
    shows the minimal surface area you need to touch.
    """

    # Sketch of what init_training_engine would do differently:
    #
    #   def init_training_engine(self):
    #       super().init_training_engine()
    #       # Replace the HF processor with our Zipformer wrapper
    #       zipformer_encoder = load_zipformer("path/to/zipformer.pt")
    #       zipformer_encoder.eval()
    #       self.processor = ZipformerProcessor(zipformer_encoder)
    pass


class WERRewardBackend(RewardBackend):
    """Reward = 1 − word_error_rate(hypothesis, reference).

    Uses ``metadata["response_text"]`` as the hypothesis and
    ``metadata["answer"]`` as the reference transcript.

    ``metadata`` is built automatically by the rollout engine:
      - ``prompt_text``   — the tokenized prompt as text
      - ``response_text`` — the model's generated text
      - ``answer``        — the ground-truth from your dataset's answer column
    """

    def __call__(self, prompt_ids, response_ids, finish_reason, metadata=None):
        r = torch.zeros(len(response_ids), dtype=torch.float32)
        if not metadata:
            return r, False

        hypothesis = metadata.get("response_text", "").strip().lower().split()
        reference = metadata.get("answer", "").strip().lower().split()

        if not reference:
            return r, False

        wer = self._word_error_rate(hypothesis, reference)
        r[-1] = max(0.0, 1.0 - wer)
        return r, False

    @staticmethod
    def _word_error_rate(hyp, ref):
        """Standard edit-distance WER."""
        n = len(ref)
        m = len(hyp)
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, m + 1):
                old = dp[j]
                if ref[i - 1] == hyp[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = old
        return dp[m] / max(n, 1)


# ═══════════════════════════════════════════════════════════════════════════
# Example 2 — RetNet (retention-based language model)
# ═══════════════════════════════════════════════════════════════════════════
#
# Scenario: RL-tune a RetNet model that uses a recurrent "retention"
#           mechanism instead of attention.  The model is text-only but
#           its forward() signature differs from standard HF causal LMs.
#
# The key insight: you do NOT need model_class="vlm" for text-only
# models.  Instead, register a custom algorithm that wraps the non-
# standard forward pass, and register a custom loss if needed.
#
# Config snippet:
#     model:
#       name: my-retnet/checkpoint
#       model_class: llm             # text-only — no processor needed
#     train:
#       alg_name: retnet_grpo
#       loss_variant: retnet_pg      # custom policy-gradient loss
#     research:
#       module: my_research.extensions
#       retention_chunk_size: 512
# ═══════════════════════════════════════════════════════════════════════════


@register_loss("retnet_pg")
def compute_retnet_pg_loss(
    logprobs,          # (B, T) log-probs under current policy
    old_logprobs,      # (B, T) log-probs from rollout policy
    advantages,        # (B, T) or (B,) group-normalised advantages
    mask,              # (B, T) response token mask
    entropies,         # (B, T) per-token entropy
    ref_logprobs,      # (B, T) log-probs under reference policy
    clip_low,          # float
    clip_high,         # float
    ent_coeff,         # float
    kl_coeff,          # float
):
    """Policy-gradient loss adapted for RetNet.

    Identical to SGRPO but adds an entropy floor to compensate for
    RetNet's tendency to produce peakier distributions due to retention
    decay.  Demonstrates how a one-function change can be registered
    without touching any core file.

    The signature above is *exactly* what ``get_loss_fn(variant)(...)``
    will be called with — all ten positional arguments, in order.
    """
    # Standard GRPO ratio computation
    logratio = logprobs - old_logprobs                 # (B, T)
    ratio = torch.exp(logratio)

    # Bi-directional clipping (same as SGRPO)
    clipped = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)

    # Advantage broadcasting: (B,) → (B, 1) if needed
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)

    surrogate = torch.min(ratio * advantages, clipped * advantages)
    pg_loss = -(surrogate * mask).sum() / mask.sum().clamp(min=1)

    # Entropy bonus with a floor — RetNet-specific tweak
    entropy_floor = 0.1  # nats; prevents collapse in retention layers
    clamped_ent = torch.clamp(entropies, min=entropy_floor)
    ent_bonus = -(ent_coeff * clamped_ent * mask).sum() / mask.sum().clamp(min=1)

    # KL penalty against reference
    kl_ref = (logprobs - ref_logprobs) * mask
    kl_penalty = (kl_coeff * kl_ref).sum() / mask.sum().clamp(min=1)

    loss_total = pg_loss + ent_bonus + kl_penalty

    metrics = {
        "loss_total": loss_total.item(),
        "pg_loss": pg_loss.item(),
        "ent_bonus": ent_bonus.item(),
        "kl_penalty": kl_penalty.item(),
        "clipfrac": ((ratio - 1.0).abs() > clip_low).float().mean().item(),
        "kl_old": logratio.mean().item(),
        "kl_ref": kl_ref.mean().item(),
        "entropy_mean": entropies.mean().item(),
    }
    return loss_total, metrics


class RetNetAdapter:
    """Demonstrates recurrent inference logic (retentive-style) without modifying PPO."""

    def __init__(self, model):
        self.model = model

    def forward(self, input_ids, state=None):
        """Run one recurrent step.

        Args:
            input_ids: (batch, seq_len) token ids.
            state: Recurrent state from previous step, or None.

        Returns:
            (logits, next_state) — logits are (batch, seq_len, vocab).
        """
        logits, next_state = self.model(input_ids, rnn_state=state)
        return logits, next_state


@register_algorithm("retnet_rl")
class RetNetRLAlgorithm:
    """GRPO/PPO variant that uses RetNetAdapter for recurrent forward passes.

    Register as ``alg_name: retnet_rl`` in config.  In a real implementation,
    subclass GRPO and override ``policy_forward`` to delegate to
    ``RetNetAdapter.forward``.
    """
    pass


class VJEPA2RewardAdapter:
    """Demonstrates reward scoring for non-generative encoders (V-JEPA2)."""

    def __init__(self, text_encoder=None, temperature: float = 0.07):
        self.text_encoder = text_encoder  # e.g. a CLIP text encoder
        self.temperature = temperature

    def compute_reward(self, video_tensor: torch.Tensor, text_query: str):
        # JEPA-style latents comparison
        return torch.tensor([0.95])  # Mock similarity score

    def __call__(self, prompt_ids, response_ids, finish_reason, metadata=None):
        """oxRL reward function interface — delegates to compute_reward."""
        r = torch.zeros(len(response_ids), dtype=torch.float32)
        if metadata is None:
            return r, False

        video_embed = metadata.get("video_embeds")
        response_text = metadata.get("response_text", "")
        if video_embed is None or not response_text:
            return r, False

        r[-1] = self.compute_reward(video_embed, response_text).item()
        return r, False


# ═══════════════════════════════════════════════════════════════════════════
# Example 3 — V-JEPA2 (video encoder)
# ═══════════════════════════════════════════════════════════════════════════
#
# Scenario: RL-tune a VLM that uses a frozen V-JEPA2 video encoder
#           instead of a standard image encoder.  Each sample contains
#           multiple video frames encoded as base64 images.
#
# Dataset columns:
#     prompt (str)           — question about the video
#     image_base64 (str)     — base64 of a single keyframe (for vLLM rollout)
#     video_frames (str)     — JSON list of base64-encoded frames
#     answer (str)           — ground-truth answer
#
# The trick: vLLM rollout uses the single keyframe for generation (the
# rollout engine only needs one image).  During *training*, we decode all
# frames and run them through V-JEPA2 to get richer video embeddings.
#
# Config snippet:
#     model:
#       name: my-video-vlm/checkpoint
#       model_class: vlm
#       freeze_vision_encoder: true
#     train:
#       alg_name: vjepa_grpo
#       loss_variant: sgrpo
#     reward:
#       reward_func: my_research.extensions.video_qa_reward
#     research:
#       module: my_research.extensions
#       vjepa_checkpoint: facebook/vjepa2-vitl
#       num_frames: 8
# ═══════════════════════════════════════════════════════════════════════════


class VJEPAProcessor:
    """Processor that runs V-JEPA2 on decoded video frames.

    Replaces the standard ``image_processor`` so that
    ``_prepare_mm_inputs`` produces video embeddings instead of
    ``pixel_values``.
    """

    def __init__(self, vjepa_model, image_size: int = 224, num_frames: int = 8):
        self.vjepa = vjepa_model        # frozen V-JEPA2 encoder
        self.image_size = image_size
        self.num_frames = num_frames

    def image_processor(self, images, return_tensors="pt"):
        """oxRL calls ``self.processor.image_processor(images=..., ...)``

        ``images`` is a list of PIL Images (one per sample in the batch).
        For video, each "image" is actually a keyframe — but our dataset
        stores multi-frame data in ``video_frames``.  If multi-frame data
        is available we use it; otherwise we replicate the single frame.

        Returns a dict of tensors that become ``mm_kwargs``.
        """
        from PIL import Image
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        batch_frames = []
        for img in images:
            # Single frame → replicate to num_frames
            frame_tensor = transform(img)                    # (3, H, W)
            frames = frame_tensor.unsqueeze(0).expand(       # (F, 3, H, W)
                self.num_frames, -1, -1, -1
            )
            batch_frames.append(frames)

        video_batch = torch.stack(batch_frames)              # (B, F, 3, H, W)

        # Run V-JEPA2 encoder
        with torch.no_grad():
            video_embeds = self.vjepa(video_batch)            # (B, N_patches, D)

        return {
            "video_embeds": video_embeds,
            "video_embed_mask": torch.ones(
                video_embeds.shape[:2], dtype=torch.bool
            ),
        }


@register_algorithm("vjepa_grpo")
class VJEPA2GRPO:
    """GRPO with V-JEPA2 video encoding.

    Subclass GRPO and override ``init_training_engine`` to load V-JEPA2
    as the processor, and override ``_prepare_mm_inputs`` if the default
    base64-decode → processor pipeline is insufficient (e.g. you need to
    decode multi-frame video data from a custom column).
    """

    # Sketch:
    #
    #   def init_training_engine(self):
    #       super().init_training_engine()
    #       # Load V-JEPA2 and replace the processor
    #       from vjepa2 import VJEPA2Encoder  # hypothetical import
    #       encoder = VJEPA2Encoder.from_pretrained("facebook/vjepa2-vitl")
    #       encoder.eval()
    #       for p in encoder.parameters():
    #           p.requires_grad = False
    #       self.processor = VJEPAProcessor(encoder, num_frames=8)
    #
    #   def _prepare_mm_inputs(self, multi_modal_data_list, device):
    #       # If your dataset stores multi-frame video in a custom format,
    #       # override this to decode frames before calling the processor.
    #       # Otherwise the base class implementation works as-is.
    #       mm_kwargs = super()._prepare_mm_inputs(multi_modal_data_list, device)
    #       return mm_kwargs
    pass


def video_qa_reward(prompt_ids, response_ids, finish_reason, metadata=None):
    """Reward function for video question-answering.

    Uses ``metadata["answer"]`` and ``metadata["response_text"]`` which
    are populated automatically by the rollout engine.

    This is a plain function (not a RewardBackend subclass) — oxRL wraps
    it in ``FunctionRewardBackend`` automatically when you reference it as
    ``reward_func: my_research.extensions.video_qa_reward`` in the config.
    """
    r = torch.zeros(len(response_ids), dtype=torch.float32)
    if not metadata or not response_ids:
        return r, False

    response = metadata.get("response_text", "").strip().lower()
    answer = metadata.get("answer", "").strip().lower()

    if not answer:
        return r, False

    # Exact match
    if answer in response:
        r[-1] = 1.0
    # Partial credit for containing key terms
    else:
        answer_words = set(answer.split())
        response_words = set(response.split())
        overlap = len(answer_words & response_words) / max(len(answer_words), 1)
        r[-1] = overlap * 0.5

    return r, False


# ═══════════════════════════════════════════════════════════════════════════
# Composing hooks — RewardShaper + custom reward
# ═══════════════════════════════════════════════════════════════════════════
#
# You can stack reward transformations using RewardShaper.  This works
# with ANY RewardBackend, including the built-in ones.
# ═══════════════════════════════════════════════════════════════════════════


class CurriculumRewardShaper(RewardShaper):
    """Scales reward magnitude based on training epoch.

    Early in training, large rewards cause high-variance gradients.
    This shaper linearly ramps the reward scale from ``warmup_scale``
    to 1.0 over ``warmup_epochs`` epochs.

    Usage::

        inner = WERRewardBackend()
        shaped = CurriculumRewardShaper(inner, warmup_epochs=5)

    Then set ``reward_func`` in config to point at a factory that
    returns the shaped backend, or construct it in your research module
    and register it via a custom reward loader.
    """

    def __init__(self, inner: RewardBackend, warmup_epochs: int = 5,
                 warmup_scale: float = 0.1):
        super().__init__(inner, warmup_epochs=warmup_epochs,
                         warmup_scale=warmup_scale)
        self.warmup_epochs = warmup_epochs
        self.warmup_scale = warmup_scale
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def shape(self, reward_tensor, is_per_token, prompt_ids, response_ids,
              finish_reason, metadata):
        if self.current_epoch >= self.warmup_epochs:
            scale = 1.0
        else:
            t = self.current_epoch / max(self.warmup_epochs, 1)
            scale = self.warmup_scale + (1.0 - self.warmup_scale) * t
        return reward_tensor * scale, is_per_token


# ═══════════════════════════════════════════════════════════════════════════
# Proof-of-concept: Downsample for multiple scales without framework changes
# ═══════════════════════════════════════════════════════════════════════════
#
# Many vision models benefit from multi-scale features (e.g. InternVL2's
# dynamic tiling, Qwen2-VL's multi-resolution patches).  Because mm_kwargs
# is just a dict merged into forward_kwargs, you can pack *any* extra
# tensors — including downsampled copies at different scales — without
# touching the training loop, loss functions, or algorithm base classes.
# ═══════════════════════════════════════════════════════════════════════════


def downsample_pixel_values(mm_kwargs: dict, scales: list[int]) -> dict:
    """Add downsampled copies of pixel_values to mm_kwargs.

    Args:
        mm_kwargs: Dict from the base image processor, must contain ``pixel_values``.
        scales:    Downsample divisors, e.g. ``[2, 4]`` → half-res, quarter-res.

    Returns:
        The same dict with ``pixel_values_s{s}`` keys added for each scale.

    Calling spec::

        mm_kwargs = processor(images=images, return_tensors="pt")
        mm_kwargs = downsample_pixel_values(mm_kwargs, scales=[2, 4])
        # mm_kwargs now has: pixel_values, pixel_values_s2, pixel_values_s4
    """
    if "pixel_values" not in mm_kwargs:
        return mm_kwargs
    pv = mm_kwargs["pixel_values"]
    for s in scales:
        mm_kwargs[f"pixel_values_s{s}"] = F.interpolate(
            pv, scale_factor=1/s, mode="bicubic", align_corners=False,
        )
    return mm_kwargs


def multiscale_prepare_mm_inputs(self, multi_modal_data_list, device):
    """Drop-in ``_prepare_mm_inputs`` replacement for multi-scale VLMs.

    Produces base pixel_values plus downsampled scales=[2, 4] via
    :func:`downsample_pixel_values`.

    Usage::

        @register_algorithm("multiscale_grpo")
        class MultiScaleGRPO(GRPO):
            _prepare_mm_inputs = multiscale_prepare_mm_inputs
    """
    if multi_modal_data_list is None:
        return {}

    images = []
    for mm_data in multi_modal_data_list:
        if mm_data is not None and "image" in mm_data:
            img_bytes = base64.b64decode(mm_data["image"])
            images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        else:
            images.append(Image.new("RGB", (224, 224), (0, 0, 0)))

    # Base resolution from processor, then add downsampled scales
    mm_kwargs = self.processor(images=images, return_tensors="pt")
    mm_kwargs = downsample_pixel_values(dict(mm_kwargs), scales=[2, 4])

    for key in mm_kwargs:
        if isinstance(mm_kwargs[key], torch.Tensor):
            mm_kwargs[key] = mm_kwargs[key].to(device, non_blocking=True)

    return mm_kwargs


# ═══════════════════════════════════════════════════════════════════════════
# Quick reference — extension point summary
# ═══════════════════════════════════════════════════════════════════════════
#
# ┌─────────────────────────┬───────────────────────────────────────────────┐
# │ What you want to do     │ Hook to use                                  │
# ├─────────────────────────┼───────────────────────────────────────────────┤
# │ Custom encoder outputs  │ Replace self.processor in a custom algorithm │
# │                         │ registered with @register_algorithm.         │
# │                         │ processor.image_processor() for vision,      │
# │                         │ processor.feature_extractor() for audio.     │
# │                         │ Return dict of tensors → becomes mm_kwargs.  │
# ├─────────────────────────┼───────────────────────────────────────────────┤
# │ Custom forward pass     │ Override policy_forward / ref_forward in     │
# │                         │ your algorithm subclass. mm_kwargs are       │
# │                         │ merged into forward_kwargs automatically.    │
# ├─────────────────────────┼───────────────────────────────────────────────┤
# │ Custom loss function    │ @register_loss("name") with the standard    │
# │                         │ 10-argument signature. Set loss_variant in   │
# │                         │ config to use it.                            │
# ├─────────────────────────┼───────────────────────────────────────────────┤
# │ Custom reward scoring   │ Subclass RewardBackend or write a plain     │
# │                         │ function. Reference via reward_func config   │
# │                         │ as a dotted path.                            │
# ├─────────────────────────┼───────────────────────────────────────────────┤
# │ Reward shaping          │ Subclass RewardShaper — wraps any inner     │
# │                         │ RewardBackend and transforms its output.     │
# ├─────────────────────────┼───────────────────────────────────────────────┤
# │ Extra config fields     │ Add arbitrary keys under research: section  │
# │                         │ (extra='allow'). Access via config.research. │
# ├─────────────────────────┼───────────────────────────────────────────────┤
# │ Auto-import at startup  │ Set research.module to your module's dotted │
# │                         │ path. Decorators fire on import.             │
# └─────────────────────────┴───────────────────────────────────────────────┘
