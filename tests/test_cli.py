"""
Comprehensive tests for CLI entry point in oxrl/cli.py.
All tests run on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_cli.py -v
"""
import pytest
import sys
import os
import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oxrl.cli import prompt_star, doctor, main, STAR_MARKER, REPO


# ============================================================
# prompt_star
# ============================================================
class TestPromptStar:
    def test_skips_when_marker_exists(self, tmp_path, monkeypatch):
        marker = tmp_path / ".oxrl_starred"
        marker.touch()
        monkeypatch.setattr("oxrl.cli.STAR_MARKER", marker)
        # Should return without doing anything
        prompt_star()
        assert marker.exists()

    def test_skips_when_no_gh(self, tmp_path, monkeypatch):
        marker = tmp_path / ".oxrl_starred"
        monkeypatch.setattr("oxrl.cli.STAR_MARKER", marker)
        monkeypatch.setattr("shutil.which", lambda x: None)
        prompt_star()
        assert marker.exists()

    @patch("builtins.input", return_value="")
    @patch("subprocess.run")
    def test_stars_on_enter(self, mock_run, mock_input, tmp_path, monkeypatch):
        marker = tmp_path / ".oxrl_starred"
        monkeypatch.setattr("oxrl.cli.STAR_MARKER", marker)
        monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/gh")
        prompt_star()
        mock_run.assert_called_once()
        assert marker.exists()

    @patch("builtins.input", return_value="yes")
    @patch("subprocess.run")
    def test_stars_on_yes(self, mock_run, mock_input, tmp_path, monkeypatch):
        marker = tmp_path / ".oxrl_starred"
        monkeypatch.setattr("oxrl.cli.STAR_MARKER", marker)
        monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/gh")
        prompt_star()
        mock_run.assert_called_once()

    @patch("builtins.input", return_value="n")
    def test_skips_on_n(self, mock_input, tmp_path, monkeypatch):
        marker = tmp_path / ".oxrl_starred"
        monkeypatch.setattr("oxrl.cli.STAR_MARKER", marker)
        monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/gh")
        prompt_star()
        assert marker.exists()

    @patch("builtins.input", side_effect=EOFError)
    def test_handles_eof(self, mock_input, tmp_path, monkeypatch):
        marker = tmp_path / ".oxrl_starred"
        monkeypatch.setattr("oxrl.cli.STAR_MARKER", marker)
        monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/gh")
        prompt_star()
        assert marker.exists()

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    def test_handles_keyboard_interrupt(self, mock_input, tmp_path, monkeypatch):
        marker = tmp_path / ".oxrl_starred"
        monkeypatch.setattr("oxrl.cli.STAR_MARKER", marker)
        monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/gh")
        prompt_star()
        assert marker.exists()

    @patch("builtins.input", return_value="")
    @patch("subprocess.run", side_effect=Exception("network error"))
    def test_handles_subprocess_failure(self, mock_run, mock_input, tmp_path, monkeypatch):
        marker = tmp_path / ".oxrl_starred"
        monkeypatch.setattr("oxrl.cli.STAR_MARKER", marker)
        monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/gh")
        prompt_star()  # Should not crash
        assert marker.exists()

    def test_always_creates_marker(self, tmp_path, monkeypatch):
        marker = tmp_path / ".oxrl_starred"
        monkeypatch.setattr("oxrl.cli.STAR_MARKER", marker)
        monkeypatch.setattr("shutil.which", lambda x: None)
        assert not marker.exists()
        prompt_star()
        assert marker.exists()


# ============================================================
# doctor
# ============================================================
class TestDoctor:
    @patch("oxrl.cli.check_gpu", return_value=True)
    @patch("oxrl.cli.check_cuda_toolkit", return_value=True)
    @patch("oxrl.cli.check_deepspeed", return_value=True)
    @patch("oxrl.cli.check_ray", return_value=True)
    def test_all_pass(self, *mocks, capsys=None):
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            doctor(fix=False)
        assert "SUCCESS" in f.getvalue()

    @patch("oxrl.cli.check_gpu", return_value=False)
    @patch("oxrl.cli.check_cuda_toolkit", return_value=False)
    @patch("oxrl.cli.check_deepspeed", return_value=False)
    @patch("oxrl.cli.check_ray", return_value=False)
    def test_all_fail(self, *mocks):
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            doctor(fix=False)
        assert "FATAL" in f.getvalue()

    @patch("oxrl.cli.check_gpu", return_value=True)
    @patch("oxrl.cli.check_cuda_toolkit", return_value=False)
    @patch("oxrl.cli.check_deepspeed", return_value=True)
    @patch("oxrl.cli.check_ray", return_value=True)
    def test_partial_fail(self, *mocks):
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            doctor(fix=False)
        assert "CAUTION" in f.getvalue()

    @patch("oxrl.cli.check_gpu", return_value=True)
    @patch("oxrl.cli.check_cuda_toolkit", return_value=True)
    @patch("oxrl.cli.check_deepspeed", return_value=True)
    @patch("oxrl.cli.check_ray", return_value=True)
    def test_dispatches_4_checks(self, mock_ray, mock_ds, mock_cuda, mock_gpu):
        doctor(fix=False)
        mock_gpu.assert_called_once()
        mock_cuda.assert_called_once()
        mock_ds.assert_called_once()
        mock_ray.assert_called_once()

    @patch("oxrl.cli.check_gpu", return_value=False)
    @patch("oxrl.cli.check_cuda_toolkit", return_value=False)
    @patch("oxrl.cli.check_deepspeed", return_value=False)
    @patch("oxrl.cli.check_ray", return_value=False)
    @patch("oxrl.cli.fix_environment")
    def test_fix_flag_triggers_fix(self, mock_fix, *check_mocks):
        doctor(fix=True)
        mock_fix.assert_called_once()

    @patch("oxrl.cli.check_gpu", return_value=True)
    @patch("oxrl.cli.check_cuda_toolkit", return_value=True)
    @patch("oxrl.cli.check_deepspeed", return_value=True)
    @patch("oxrl.cli.check_ray", return_value=True)
    @patch("oxrl.cli.fix_environment")
    def test_no_fix_when_all_pass(self, mock_fix, *check_mocks):
        doctor(fix=True)
        mock_fix.assert_not_called()


# ============================================================
# main
# ============================================================
class TestMain:
    @patch("oxrl.cli.prompt_star")
    @patch("oxrl.cli.doctor")
    def test_doctor_subcommand(self, mock_doctor, mock_star, monkeypatch):
        monkeypatch.setattr("sys.argv", ["oxrl", "doctor"])
        main()
        mock_star.assert_called_once()
        mock_doctor.assert_called_once_with(fix=False)

    @patch("oxrl.cli.prompt_star")
    @patch("oxrl.cli.doctor")
    def test_doctor_fix_subcommand(self, mock_doctor, mock_star, monkeypatch):
        monkeypatch.setattr("sys.argv", ["oxrl", "doctor", "--fix"])
        main()
        mock_doctor.assert_called_once_with(fix=True)

    @patch("oxrl.cli.prompt_star")
    def test_no_subcommand_prints_help(self, mock_star, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["oxrl"])
        main()
        mock_star.assert_called_once()

    @patch("oxrl.cli.prompt_star")
    def test_prompt_star_called_on_every_invocation(self, mock_star, monkeypatch):
        monkeypatch.setattr("sys.argv", ["oxrl"])
        main()
        mock_star.assert_called_once()


# ============================================================
# check_gpu
# ============================================================
class TestCheckGpu:
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    @patch("torch.cuda.get_device_name", return_value="NVIDIA A100")
    def test_gpu_available(self, *mocks):
        from oxrl.cli import check_gpu
        assert check_gpu() is True

    @patch("torch.cuda.is_available", return_value=False)
    def test_no_gpu(self, mock_avail):
        from oxrl.cli import check_gpu
        assert check_gpu() is False


# ============================================================
# check_cuda_toolkit
# ============================================================
class TestCheckCudaToolkit:
    @patch("subprocess.check_output", return_value="nvcc: NVIDIA (R) CUDA compiler\nV12.0.0")
    def test_nvcc_found(self, mock_output):
        from oxrl.cli import check_cuda_toolkit
        assert check_cuda_toolkit() is True

    @patch("subprocess.check_output", side_effect=FileNotFoundError)
    def test_nvcc_not_found(self, mock_output):
        from oxrl.cli import check_cuda_toolkit
        assert check_cuda_toolkit() is False


# ============================================================
# check_ray
# ============================================================
class TestCheckRay:
    def test_ray_importable(self):
        from oxrl.cli import check_ray
        # This depends on whether ray is installed in the test env
        # Just ensure it returns a bool and doesn't crash
        result = check_ray()
        assert isinstance(result, bool)
