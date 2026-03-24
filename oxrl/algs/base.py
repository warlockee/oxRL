import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseAlgorithm(ABC):
    """
    Abstract Base Class for all oxRL algorithms.
    Provides a standardized interface for LLMs to implement new methods (like DPO).
    """

    @abstractmethod
    def is_ready(self) -> bool:
        """Barrier method to ensure all Ray actors are initialized."""
        pass

    @abstractmethod
    def train_step(self, *args, **kwargs) -> Dict[str, float]:
        """Perform a single optimization step and return metrics."""
        pass

    @abstractmethod
    def save_checkpoint(self, output_dir: str, tag: str, state_dict_ref=None):
        """Save model weights and configuration.

        If state_dict_ref is provided (a Ray object ref), rank 0 retrieves and
        writes the pre-gathered state dict instead of doing ZeRO-3 gather.
        """
        pass

    def gather_state_dict(self) -> Optional[dict]:
        """Gather ZeRO-3 partitioned weights into a single state dict in memory.

        Collective operation — must be called on ALL ranks.
        Returns the state dict on rank 0, None on other ranks.
        Includes '__model_config_dict__' key with model config on rank 0.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement gather_state_dict. "
            "Falling back to disk-based checkpoint flow."
        )

    def offload_to_cpu(self) -> bool:
        """Offload optimizer states, gradients, and parameters to CPU.

        Called before vLLM model refresh to free GPU memory for the rollout
        engine. Returns True if offload succeeded.

        Default implementation does nothing. Subclasses (GRPO, PPO) override
        with DeepSpeed-aware logic.
        """
        return False

    def reload_to_gpu(self) -> bool:
        """Reload optimizer states, gradients, and parameters back to GPU.

        Called after vLLM model refresh completes, before the next training
        epoch begins. Returns True if reload succeeded.

        Default implementation does nothing. Subclasses (GRPO, PPO) override
        with DeepSpeed-aware logic.
        """
        return False

    # ------------------------------------------------------------------
    # VLM / multimodal helpers (shared by GRPO and PPO)
    # ------------------------------------------------------------------

    def _freeze_vision_encoder(self, model) -> None:
        """Freeze the vision encoder weights for VLM training.

        Searches for common vision encoder attribute names across VLM
        architectures and freezes all their parameters.
        """
        vision_attr_names = [
            "vision_tower",    # LLaVA
            "visual",          # Qwen2-VL
            "vision_model",    # Llama 3.2 Vision, generic
            "vision_encoder",  # InternVL
        ]

        frozen_count = 0
        for attr_name in vision_attr_names:
            vision_module = getattr(model, attr_name, None)
            if vision_module is None:
                # Check nested model.model (common with HF wrappers)
                inner = getattr(model, "model", None)
                if inner is not None:
                    vision_module = getattr(inner, attr_name, None)

            if vision_module is not None:
                for param in vision_module.parameters():
                    param.requires_grad = False
                    frozen_count += 1
                break

        if frozen_count == 0:
            print(
                f"[{self.__class__.__name__}] WARNING: Could not find vision "
                f"encoder to freeze. Checked: {vision_attr_names}"
            )
        else:
            print(f"[{self.__class__.__name__}] Froze {frozen_count} vision encoder parameters")

    def _prepare_mm_inputs(
        self, multi_modal_data_list: List[Optional[Dict[str, Any]]], device
    ) -> Dict[str, Any]:
        """Decode base64 multimodal data and process into model-ready tensors.

        Args:
            multi_modal_data_list: One entry per sample in the batch.
                Each entry is {"image": "<base64>"} or {"audio": "<base64>"} or None.
            device: Target torch device for output tensors.

        Returns:
            Dict of tensors (e.g. {"pixel_values": tensor}) or empty dict.
        """
        if multi_modal_data_list is None:
            return {}

        import base64
        import io
        from PIL import Image

        images = []
        audios = []
        has_images = False
        has_audio = False

        for mm_data in multi_modal_data_list:
            if mm_data is not None and "image" in mm_data:
                img_bytes = base64.b64decode(mm_data["image"])
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append(image)
                has_images = True
            else:
                images.append(None)

            if mm_data is not None and "audio" in mm_data:
                import soundfile as sf
                audio_bytes = base64.b64decode(mm_data["audio"])
                audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
                audios.append((audio_data, sample_rate))
                has_audio = True
            else:
                audios.append(None)

        mm_kwargs = {}

        if has_images:
            # Fill None entries with dummy black images (loss mask zeros out gradients)
            reference_img = next(img for img in images if img is not None)
            dummy = Image.new("RGB", reference_img.size, (0, 0, 0))
            filled_images = [img if img is not None else dummy for img in images]
            processed = self.processor.image_processor(
                images=filled_images, return_tensors="pt"
            )
            for key, val in processed.items():
                if isinstance(val, torch.Tensor):
                    mm_kwargs[key] = val.to(device, non_blocking=True)
                else:
                    mm_kwargs[key] = val

        if has_audio:
            import numpy as np
            # Fill None entries with silent audio
            reference_audio = next(a for a in audios if a is not None)
            ref_sr = reference_audio[1]
            dummy_audio = (np.zeros(ref_sr, dtype=np.float32), ref_sr)  # 1s silence
            filled_audios = [a if a is not None else dummy_audio for a in audios]
            raw_speech = [a[0] for a in filled_audios]
            sampling_rate = filled_audios[0][1]
            processed = self.processor.feature_extractor(
                raw_speech=raw_speech,
                sampling_rate=sampling_rate,
                return_tensors="pt",
            )
            for key, val in processed.items():
                if isinstance(val, torch.Tensor):
                    mm_kwargs[key] = val.to(device, non_blocking=True)
                else:
                    mm_kwargs[key] = val

        return mm_kwargs
