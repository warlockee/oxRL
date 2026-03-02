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

# Re-export from new canonical locations for backward compatibility
from oxrl.tools.tensor_utils import ensure_1d, pad_1d_to_length  # noqa: F401

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
    import shutil

    # Ensure bypass is enabled if not already set
    if "DS_SKIP_CUDA_CHECK" not in os.environ:
        os.environ["DS_SKIP_CUDA_CHECK"] = "1"

    # Auto-detect CUDA_HOME if not set (DeepSpeed needs it for op compatibility checks)
    if not os.environ.get("CUDA_HOME"):
        # 1. Check common locations
        for candidate in [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "cuda_env"),
            "/usr/local/cuda",
            "/usr/local/cuda-12",
            "/usr/local/cuda-11",
        ]:
            if os.path.isfile(os.path.join(candidate, "bin", "nvcc")):
                os.environ["CUDA_HOME"] = candidate
                break
        else:
            # 2. Try to find nvcc on PATH
            nvcc_path = shutil.which("nvcc")
            if nvcc_path:
                os.environ["CUDA_HOME"] = os.path.dirname(os.path.dirname(nvcc_path))

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