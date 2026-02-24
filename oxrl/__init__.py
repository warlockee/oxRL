import os
import subprocess

def _ensure_cuda_home():
    """Auto-detect CUDA_HOME if not set."""
    if not os.environ.get("CUDA_HOME"):
        # 1. Try standard paths
        possible_paths = [
            "/usr/local/cuda",
            "/opt/cuda",
            "/usr/lib/cuda",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                os.environ["CUDA_HOME"] = path
                os.environ["PATH"] = f"{path}/bin:{os.environ.get('PATH', '')}"
                return

        # 2. Try 'which nvcc'
        try:
            nvcc_path = subprocess.check_output(["which", "nvcc"], text=True).strip()
            if nvcc_path:
                cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
                os.environ["CUDA_HOME"] = cuda_home
                return
        except Exception:
            pass

_ensure_cuda_home()

from .trainer import Trainer

__version__ = "0.1.0"
