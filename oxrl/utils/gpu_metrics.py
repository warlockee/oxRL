"""
GPU memory, throughput, and timing utilities for observability.

All functions are safe on CPU — they return empty dicts or 0.0 when
CUDA is unavailable.
"""
import time

import torch


def get_gpu_memory_metrics() -> dict:
    """Return GPU memory stats in GB. Empty dict on CPU."""
    if not torch.cuda.is_available():
        return {}

    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)

    return {
        "gpu/mem_allocated_gb": allocated / (1024 ** 3),
        "gpu/mem_reserved_gb": reserved / (1024 ** 3),
        "gpu/mem_free_gb": free / (1024 ** 3),
    }


def compute_throughput(num_tokens: int, elapsed_sec: float) -> float:
    """Return tokens/sec, guarding against zero elapsed time."""
    if elapsed_sec <= 0:
        return 0.0
    return num_tokens / elapsed_sec


class StepTimer:
    """Context manager that records wall-clock elapsed time.

    Usage::

        with StepTimer() as t:
            do_work()
        print(t.elapsed_sec)
    """

    def __init__(self):
        self.elapsed_sec = 0.0
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_sec = time.perf_counter() - self._start
        return False
