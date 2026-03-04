import os
import random
import numpy as np
import torch
from transformers import AutoTokenizer

_sliding_window_patched = False
_loss_kwargs_patched = False
_pytorch_gelu_tanh_patched = False
_cache_usable_length_patched = False
_auto_docstring_patched = False

def ensure_sliding_window_cache():
    """
    Monkey-patch missing SlidingWindowCache for Phi-4-mini compatibility.
    Safe to call multiple times; only patches once.
    """
    global _sliding_window_patched
    if _sliding_window_patched:
        return
    try:
        from transformers.cache_utils import SlidingWindowCache  # noqa: F401
    except ImportError:
        from transformers.cache_utils import DynamicCache as _DynCache
        import transformers.cache_utils as _cu
        class SlidingWindowCache(_DynCache):
            """Stub for models that import SlidingWindowCache (e.g. Phi-4-mini)."""
            pass
        _cu.SlidingWindowCache = SlidingWindowCache
    _sliding_window_patched = True


def ensure_loss_kwargs():
    """
    Monkey-patch missing LossKwargs for Phi-4-mini remote code compatibility.
    In transformers>=4.54, LossKwargs was removed from transformers.utils.
    Models with stale remote code (e.g. Phi-4-mini-instruct) still try to
    import it.  We inject a minimal stub so the import succeeds.
    Safe to call multiple times; only patches once.
    """
    global _loss_kwargs_patched
    if _loss_kwargs_patched:
        return
    try:
        from transformers.utils import LossKwargs  # noqa: F401
    except ImportError:
        import transformers.utils as _tu
        from typing import Optional
        try:
            from typing import TypedDict
        except ImportError:
            from typing_extensions import TypedDict

        class LossKwargs(TypedDict, total=False):
            """Stub for models that import LossKwargs (removed in transformers>=4.54)."""
            num_items_in_batch: Optional[int]

        _tu.LossKwargs = LossKwargs
    _loss_kwargs_patched = True


def ensure_pytorch_gelu_tanh():
    """
    Monkey-patch missing PytorchGELUTanh for Kimi-VL compatibility.
    Some remote model code (e.g. moonshotai/Kimi-VL) imports PytorchGELUTanh
    from transformers.activations, but it was removed/renamed in newer versions.
    Safe to call multiple times; only patches once.
    """
    global _pytorch_gelu_tanh_patched
    if _pytorch_gelu_tanh_patched:
        return
    try:
        from transformers.activations import PytorchGELUTanh  # noqa: F401
    except ImportError:
        import transformers.activations as _acts
        import torch.nn as nn
        class PytorchGELUTanh(nn.Module):
            """Stub: GELU with tanh approximation for models that import PytorchGELUTanh."""
            def __init__(self):
                super().__init__()
                self.act = nn.GELU(approximate="tanh")
            def forward(self, input):
                return self.act(input)
        _acts.PytorchGELUTanh = PytorchGELUTanh
    _pytorch_gelu_tanh_patched = True


def ensure_cache_usable_length():
    """
    Monkey-patch DynamicCache for backward compatibility with old transformers API.
    In transformers>=4.57, DynamicCache was refactored:
      - key_cache/value_cache lists replaced by layers[i].keys/values
      - get_usable_length removed (use get_seq_length instead)
    Models with stale remote code (e.g. moonshotai/Kimi-VL) still use the old API.
    Safe to call multiple times; only patches once.
    """
    global _cache_usable_length_patched
    if _cache_usable_length_patched:
        return
    from transformers.cache_utils import DynamicCache

    # Add get_usable_length if missing
    if not hasattr(DynamicCache, 'get_usable_length'):
        def _get_usable_length(self, new_seq_length, layer_idx=None):
            """Return the length of the already-cached key/value pairs."""
            if layer_idx is not None:
                return self.get_seq_length(layer_idx)
            return self.get_seq_length()
        DynamicCache.get_usable_length = _get_usable_length

    # Add key_cache property if missing (delegates to layers[i].keys)
    if not hasattr(DynamicCache, 'key_cache'):
        class _KeyCacheProxy:
            """Proxy list that maps index access to layers[i].keys."""
            def __init__(self, cache):
                self._cache = cache
            def __len__(self):
                return len(self._cache.layers)
            def __getitem__(self, idx):
                if idx < len(self._cache.layers):
                    return self._cache.layers[idx].keys
                raise IndexError(f"layer {idx} not in cache (have {len(self._cache.layers)} layers)")
            def __setitem__(self, idx, value):
                if idx < len(self._cache.layers):
                    self._cache.layers[idx].keys = value
                else:
                    raise IndexError(f"layer {idx} not in cache")
            def append(self, value):
                pass  # layers managed by update()

        @property
        def _key_cache_prop(self):
            return _KeyCacheProxy(self)
        DynamicCache.key_cache = _key_cache_prop

    # Add value_cache property if missing
    if not hasattr(DynamicCache, 'value_cache'):
        class _ValueCacheProxy:
            """Proxy list that maps index access to layers[i].values."""
            def __init__(self, cache):
                self._cache = cache
            def __len__(self):
                return len(self._cache.layers)
            def __getitem__(self, idx):
                if idx < len(self._cache.layers):
                    return self._cache.layers[idx].values
                raise IndexError(f"layer {idx} not in cache (have {len(self._cache.layers)} layers)")
            def __setitem__(self, idx, value):
                if idx < len(self._cache.layers):
                    self._cache.layers[idx].values = value
                else:
                    raise IndexError(f"layer {idx} not in cache")
            def append(self, value):
                pass  # layers managed by update()

        @property
        def _value_cache_prop(self):
            return _ValueCacheProxy(self)
        DynamicCache.value_cache = _value_cache_prop

    _cache_usable_length_patched = True


def ensure_auto_docstring_union_type():
    """
    Monkey-patch transformers auto_docstring to handle Python 3.10+ union types.
    In Python 3.10, `X | Y` creates a `types.UnionType` which doesn't have `__name__`.
    The transformers auto_docstring code tries to access `param.annotation.__name__`
    which fails for union types. We patch _process_parameter_type to handle this.
    Safe to call multiple times; only patches once.
    """
    global _auto_docstring_patched
    if _auto_docstring_patched:
        return
    import types as _types
    import sys as _sys
    try:
        import importlib
        _ad = importlib.import_module('transformers.utils.auto_docstring')
        if hasattr(_ad, '_process_parameter_type'):
            _orig_fn = _ad._process_parameter_type
            import inspect as _inspect

            def _patched_process_parameter_type(param, param_name, func):
                # Handle types.UnionType (e.g., int | None) which lacks __name__
                if param.annotation != _inspect.Parameter.empty:
                    if isinstance(param.annotation, _types.UnionType):
                        return str(param.annotation), True
                return _orig_fn(param, param_name, func)

            _ad._process_parameter_type = _patched_process_parameter_type
    except (ImportError, AttributeError):
        pass  # Old transformers version without auto_docstring
    _auto_docstring_patched = True


# Apply at import time for modules that import utils/setup.py
ensure_sliding_window_cache()
ensure_loss_kwargs()
ensure_pytorch_gelu_tanh()
ensure_cache_usable_length()
ensure_auto_docstring_union_type()

def set_random_seeds(seed):
    '''
        Set random seeds to make runs more reproducible (still not guaranteed). With distributed training,
        floating-point math and non-deterministic ops (e.g., torch.Tensor.index_add_) can still cause differences,
        seeding just reduces the variance a bit.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_rank_info():
    '''
        Detect rank from environment variables.
    '''
    # Unique id of gpu in the ENTIRE WORLD. It ranges from 0 to world_size - 1
    rank = int(os.environ.get('RANK', 0))

    # Unique id of gpu in the LOCAL node (or simply one node). It ranges from 0 to local_node_size - 1
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # add some checks to make sure number of gpus and local rank are correct.
    if not torch.cuda.is_available():
        if rank == 0:
            print("Warning: CUDA is not available, running on CPU. Sorry!")
    else:
        num_local_gpus = torch.cuda.device_count()
        if local_rank >= num_local_gpus:
            raise RuntimeError(f"LOCAL_RANK {local_rank} >= available GPUs {num_local_gpus}")

        torch.cuda.set_device(local_rank)

    return rank, local_rank

def get_distributed_info():
    '''
        Detect rank and world size from environment variables.
        we way to run is to use torchrun (torchrun --nnodes=2 --nproc_per_node=4 main_sl.py) where we can specify
        nnodes=2 -> world_size
        nproc_per_node=4 -> local_world_size/num_local_gpus
    '''
    # total number of gpus (e.g, 2 nodes x 4 gpus = 8 gpus in total). world size need to be at least 1
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # Unique id of gpu in the ENTIRE WORLD. It ranges from 0 to world_size - 1
    rank = int(os.environ.get('RANK', 0))

    # Unique id of gpu in the LOCAL node (or simply one node). It ranges from 0 to local_node_size - 1
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # add some checks to make sure number of gpus and local rank are correct.
    if not torch.cuda.is_available():
        if rank == 0:
            print("Warning: CUDA is not available, running on CPU. Sorry!")
    else:
        num_local_gpus = torch.cuda.device_count()
        if local_rank >= num_local_gpus:
            raise RuntimeError(f"LOCAL_RANK {local_rank} >= available GPUs {num_local_gpus}")

        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank

def load_tokenizer(model_name, trust_remote_code=False, rank=0):
    '''
       Load tokenizer from huggingface.
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=trust_remote_code)

    # if pad token is not present, we use eos token as pad token
    # log warning if pad token is not present.
    if tokenizer.pad_token_id is None:
        if rank == 0:
            print("Warning: Pad token is not present, using eos token as pad token")
        if getattr(tokenizer, 'eos_token', None) is not None:
            # prefer explicit token if available
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        else:
            # fallback to eos token id
            tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer

def load_model_and_ref(model_path, model_dtype, trust_remote_code, attn_impl, ref_model_path=None, device_map=None):
    '''
        Unified loader for CausalLM, Multimodal, and ImageText models.
        Handles architecture fallbacks automatically.

        Args:
            device_map: Optional device_map passed to from_pretrained (e.g. 'cpu', 'auto').
                        Use 'cpu' for very large models that will be sharded by DeepSpeed.
    '''
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoConfig, AutoModel
    try:
        from transformers import AutoModelForMultimodalLM
    except ImportError:
        AutoModelForMultimodalLM = None

    def _load_single(path, cfg):
        load_classes = [AutoModelForCausalLM, AutoModelForImageTextToText]
        if AutoModelForMultimodalLM:
            load_classes.append(AutoModelForMultimodalLM)

        # Architecture-specific fallbacks
        try:
            from transformers import Qwen2VLForConditionalGeneration
            load_classes.append(Qwen2VLForConditionalGeneration)
        except ImportError: pass
        try:
            from transformers import Qwen2AudioForConditionalGeneration
            load_classes.append(Qwen2AudioForConditionalGeneration)
        except ImportError: pass

        extra_kwargs = {}
        if device_map is not None:
            extra_kwargs['device_map'] = device_map

        for cls in load_classes:
            try:
                return cls.from_pretrained(path,
                                          dtype=model_dtype,
                                          trust_remote_code=trust_remote_code,
                                          config=cfg,
                                          attn_implementation=None if attn_impl == '' else attn_impl,
                                          **extra_kwargs)
            except (ValueError, TypeError):
                continue

        # Ultimate fallback
        return AutoModel.from_pretrained(path,
                                        dtype=model_dtype,
                                        trust_remote_code=trust_remote_code,
                                        config=cfg,
                                        attn_implementation=None if attn_impl == '' else attn_impl,
                                        **extra_kwargs)

    # Load Main Model
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = _load_single(model_path, config)

    # Load Ref Model (Optional)
    ref_model = None
    if ref_model_path:
        ref_cfg = AutoConfig.from_pretrained(ref_model_path, trust_remote_code=trust_remote_code)
        ref_model = _load_single(ref_model_path, ref_cfg)

    return model, ref_model
