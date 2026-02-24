import os

def _ensure_cuda_home():
    """Auto-detect CUDA_HOME if not set."""
    if not os.environ.get("CUDA_HOME"):
        possible_paths = [
            "/usr/local/cuda",
            "/opt/cuda",
            "/usr/lib/cuda",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                os.environ["CUDA_HOME"] = path
                os.environ["PATH"] = f"{path}/bin:{os.environ.get('PATH', '')}"
                # Use print cautiously or log it
                break

_ensure_cuda_home()

from .trainer import Trainer

__version__ = "0.1.0"
