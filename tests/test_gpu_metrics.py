"""
Tests for GPU metrics utilities in oxrl/utils/gpu_metrics.py.
All tests run on CPU without CUDA.
Run with: pytest tests/test_gpu_metrics.py -v
"""
import pytest
import sys
import os
import time
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oxrl.utils.gpu_metrics import get_gpu_memory_metrics, compute_throughput, StepTimer


# ============================================================
# get_gpu_memory_metrics
# ============================================================
class TestGetGpuMemoryMetrics:
    @patch("torch.cuda.is_available", return_value=False)
    def test_cpu_returns_empty_dict(self, mock_avail):
        result = get_gpu_memory_metrics()
        assert result == {}

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.current_device", return_value=0)
    @patch("torch.cuda.mem_get_info", return_value=(4 * 1024**3, 8 * 1024**3))
    @patch("torch.cuda.memory_allocated", return_value=2 * 1024**3)
    @patch("torch.cuda.memory_reserved", return_value=3 * 1024**3)
    def test_gpu_returns_correct_keys(self, mock_reserved, mock_alloc, mock_mem, mock_dev, mock_avail):
        result = get_gpu_memory_metrics()
        assert "gpu/mem_allocated_gb" in result
        assert "gpu/mem_reserved_gb" in result
        assert "gpu/mem_free_gb" in result

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.current_device", return_value=0)
    @patch("torch.cuda.mem_get_info", return_value=(4 * 1024**3, 8 * 1024**3))
    @patch("torch.cuda.memory_allocated", return_value=2 * 1024**3)
    @patch("torch.cuda.memory_reserved", return_value=3 * 1024**3)
    def test_gpu_values_in_gb(self, mock_reserved, mock_alloc, mock_mem, mock_dev, mock_avail):
        result = get_gpu_memory_metrics()
        assert abs(result["gpu/mem_allocated_gb"] - 2.0) < 0.01
        assert abs(result["gpu/mem_reserved_gb"] - 3.0) < 0.01
        assert abs(result["gpu/mem_free_gb"] - 4.0) < 0.01

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.current_device", return_value=0)
    @patch("torch.cuda.mem_get_info", return_value=(0, 0))
    @patch("torch.cuda.memory_allocated", return_value=0)
    @patch("torch.cuda.memory_reserved", return_value=0)
    def test_zero_memory(self, mock_reserved, mock_alloc, mock_mem, mock_dev, mock_avail):
        result = get_gpu_memory_metrics()
        assert result["gpu/mem_allocated_gb"] == 0.0
        assert result["gpu/mem_reserved_gb"] == 0.0
        assert result["gpu/mem_free_gb"] == 0.0


# ============================================================
# compute_throughput
# ============================================================
class TestComputeThroughput:
    def test_normal_case(self):
        result = compute_throughput(1000, 2.0)
        assert result == 500.0

    def test_zero_elapsed(self):
        result = compute_throughput(1000, 0.0)
        assert result == 0.0

    def test_negative_elapsed(self):
        result = compute_throughput(1000, -1.0)
        assert result == 0.0

    def test_zero_tokens(self):
        result = compute_throughput(0, 1.0)
        assert result == 0.0

    def test_large_values(self):
        result = compute_throughput(1_000_000, 0.001)
        assert result == pytest.approx(1e9)

    def test_fractional(self):
        result = compute_throughput(100, 3.0)
        assert result == pytest.approx(100 / 3.0)


# ============================================================
# StepTimer
# ============================================================
class TestStepTimer:
    def test_records_elapsed_time(self):
        with StepTimer() as t:
            time.sleep(0.05)
        assert t.elapsed_sec >= 0.04  # Allow some tolerance

    def test_initial_elapsed_is_zero(self):
        t = StepTimer()
        assert t.elapsed_sec == 0.0

    def test_context_manager_returns_self(self):
        with StepTimer() as t:
            assert isinstance(t, StepTimer)

    def test_elapsed_available_after_exit(self):
        timer = StepTimer()
        with timer:
            time.sleep(0.01)
        assert timer.elapsed_sec > 0

    def test_does_not_suppress_exceptions(self):
        with pytest.raises(ValueError):
            with StepTimer() as t:
                raise ValueError("test")
        # Timer should still have recorded some time
        assert t.elapsed_sec >= 0
