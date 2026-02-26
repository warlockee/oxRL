import torch
import os

def safe_string_to_torch_dtype(dtype_in):
    '''
        dtype_in might be a string in config (e.g., "fp16", "float16"). transformers expects torch.float16 or torch.bfloat16 etc.,
        when passed as torch_dtype. We must convert strings safely.
    '''

    if isinstance(dtype_in, torch.dtype):
        return dtype_in

    if dtype_in is None:
        return None

    if isinstance(dtype_in, str):
        s = dtype_in.lower()
        if s in ("fp16", "float16"):
            return torch.float16

        if s in ("bf16", "bfloat16"):
            return torch.bfloat16

        if s in ("fp32", "float32"):
            return torch.float32

        if s in ("fp64", "float64"):
            return torch.float64

    raise ValueError(f"Unsupported model_dtype: {dtype_in}")

def ensure_1d(x: torch.Tensor, name: str) -> torch.Tensor:
    '''
        Sanity check to make sure the input is a 1D tensor.
    '''
    if x.dim() != 1:
        raise ValueError(f"Expected {name} to be 1D, got {x.dim()}D")

    return x

def pad_1d_to_length(x: torch.Tensor, pad_value: float, target_len: int) -> torch.Tensor:
    '''
        Pad/truncate 1D sequence x[T] to target_len.
        Always returns length == target_len.
    '''
    seq_len = x.numel()

    if seq_len > target_len:
        return x[:target_len]

    if seq_len < target_len:
        pad = torch.full((target_len - seq_len,),
                            pad_value,
                            dtype=x.dtype,
                            device=x.device)
        return torch.cat([x, pad], dim=0)

    return x

def get_experiment_dir_name(output_dir: str, tag: str, experiment_id: str):
    '''
       It creates output_dir/experiment_id/tag
    '''
    experiment_dir = os.path.join(output_dir, experiment_id, tag)
    return experiment_dir

def import_deepspeed_safely():
    """
    Attempts to import deepspeed and handle common initialization errors
    (like MissingCUDAException) with clear warnings instead of fatal crashes.
    """
    import os
    # Ensure bypass is enabled if not already set
    if "DS_SKIP_CUDA_CHECK" not in os.environ:
        os.environ["DS_SKIP_CUDA_CHECK"] = "1"
    
    try:
        import deepspeed
        return deepspeed
    except Exception as e:
        print(f"\n[WARNING] DeepSpeed initialization encountered an issue: {e}")
        print("[WARNING] oxRL will attempt to continue, but performance might be degraded.")
        print("[WARNING] If you encounter further errors, ensure CUDA Toolkit (nvcc) is installed.")
        
        # Try to import again with even more aggressive bypasses if possible
        # or just re-raise if it's a fundamental ImportError
        try:
            import deepspeed
            return deepspeed
        except:
            raise ImportError("DeepSpeed could not be imported. Please install it with: pip install deepspeed")