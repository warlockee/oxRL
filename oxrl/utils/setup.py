import os
import random
import numpy as np
import torch
from transformers import AutoTokenizer

# Monkey-patch missing SlidingWindowCache for Phi-4-mini compatibility
try:
    from transformers.cache_utils import SlidingWindowCache  # noqa: F401
except ImportError:
    from transformers.cache_utils import DynamicCache as _DynCache
    import transformers.cache_utils as _cu
    class SlidingWindowCache(_DynCache):
        """Stub for models that import SlidingWindowCache (e.g. Phi-4-mini)."""
        pass
    _cu.SlidingWindowCache = SlidingWindowCache

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

def load_model_and_ref(model_path, model_dtype, trust_remote_code, attn_impl, ref_model_path=None):
    '''
        Unified loader for CausalLM, Multimodal, and ImageText models.
        Handles architecture fallbacks automatically.
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

        for cls in load_classes:
            try:
                return cls.from_pretrained(path,
                                          dtype=model_dtype,
                                          trust_remote_code=trust_remote_code,
                                          config=cfg,
                                          attn_implementation=None if attn_impl == '' else attn_impl)
            except (ValueError, TypeError):
                continue
        
        # Ultimate fallback
        return AutoModel.from_pretrained(path,
                                        dtype=model_dtype,
                                        trust_remote_code=trust_remote_code,
                                        config=cfg,
                                        attn_implementation=None if attn_impl == '' else attn_impl)

    # Load Main Model
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = _load_single(model_path, config)

    # Load Ref Model (Optional)
    ref_model = None
    if ref_model_path:
        ref_cfg = AutoConfig.from_pretrained(ref_model_path, trust_remote_code=trust_remote_code)
        ref_model = _load_single(ref_model_path, ref_cfg)

    return model, ref_model
