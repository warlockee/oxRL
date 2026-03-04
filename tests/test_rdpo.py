"""
Tests for R-DPO (Robust DPO with Length Regularization) algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_rdpo.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestRDPOImport:
    """Test R-DPO is properly importable and registered."""

    def test_rdpo_importable(self):
        from oxrl.algs.rdpo import RDPO
        assert RDPO is not None

    def test_rdpo_in_algs_init(self):
        from oxrl.algs import RDPO
        assert RDPO is not None

    def test_rdpo_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"rdpo"' in source, "RDPO should be registered in SL_ALGORITHMS"


class TestRDPOInterface:
    """Test R-DPO has the expected interface."""

    def test_rdpo_has_required_methods(self):
        from oxrl.algs.rdpo import RDPO
        assert hasattr(RDPO, 'compute_logps')
        assert hasattr(RDPO, 'forward')
        assert hasattr(RDPO, 'compute_loss')
        assert hasattr(RDPO, 'train_step')
        assert hasattr(RDPO, 'eval_step')

    def test_rdpo_init_params(self):
        from oxrl.algs.rdpo import RDPO
        sig = inspect.signature(RDPO.__init__)
        params = list(sig.parameters.keys())
        assert 'model_engine' in params
        assert 'ref_model_engine' in params
        assert 'optimizer' in params
        assert 'beta' in params
        assert 'alpha' in params

    def test_rdpo_requires_ref_model(self):
        """R-DPO requires a reference model (like standard DPO)."""
        from oxrl.algs.rdpo import RDPO
        sig = inspect.signature(RDPO.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params, "RDPO should require ref_model_engine"


class TestRDPOLossMath:
    """Test R-DPO loss computation with known values."""

    def _make_rdpo(self, beta=0.1, alpha=0.01):
        from oxrl.algs.rdpo import RDPO
        rdpo = RDPO.__new__(RDPO)
        rdpo.beta = beta
        rdpo.alpha = alpha
        return rdpo

    def test_alpha_zero_matches_dpo(self):
        """With alpha=0 (no length regularization), R-DPO should match DPO."""
        from oxrl.algs.dpo import DPO
        rdpo = self._make_rdpo(beta=0.1, alpha=0.0)
        dpo = DPO.__new__(DPO)
        dpo.beta = 0.1

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])
        len_w = torch.tensor([10.0, 8.0])
        len_l = torch.tensor([15.0, 12.0])

        rdpo_loss, _, _ = rdpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l, len_w, len_l)
        dpo_loss, _ = dpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        assert abs(rdpo_loss.item() - dpo_loss.item()) < 1e-5, \
            f"alpha=0 R-DPO should match DPO: {rdpo_loss} vs {dpo_loss}"

    def test_loss_formula_verification(self):
        """Verify: loss = -logsigmoid(beta * (logr_w - logr_l) - alpha * (len_w - len_l))."""
        rdpo = self._make_rdpo(beta=0.1, alpha=0.05)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])
        len_w = torch.tensor([10.0, 8.0])
        len_l = torch.tensor([15.0, 12.0])

        loss, margin, avg_len_diff = rdpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l, len_w, len_l)

        # Manual computation
        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        dpo_logits = 0.1 * (logr_w - logr_l)
        length_penalty = 0.05 * (len_w - len_l)
        expected_logits = dpo_logits - length_penalty
        expected_loss = -F.logsigmoid(expected_logits).mean()

        assert abs(loss.item() - expected_loss.item()) < 1e-5, \
            f"R-DPO loss mismatch: {loss} vs {expected_loss}"

    def test_length_regularization_effect(self):
        """Length regularization should penalize when chosen is longer than rejected."""
        rdpo = self._make_rdpo(beta=0.1, alpha=0.1)

        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])

        # Case 1: chosen is longer (should increase loss via penalty)
        len_w_long = torch.tensor([20.0])
        len_l_short = torch.tensor([5.0])
        loss_chosen_longer, _, _ = rdpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l,
            len_w_long, len_l_short)

        # Case 2: chosen is shorter (should decrease loss via penalty)
        len_w_short = torch.tensor([5.0])
        len_l_long = torch.tensor([20.0])
        loss_chosen_shorter, _, _ = rdpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l,
            len_w_short, len_l_long)

        assert loss_chosen_longer.item() > loss_chosen_shorter.item(), \
            f"Chosen-longer should have higher loss: {loss_chosen_longer} vs {loss_chosen_shorter}"

    def test_equal_lengths_no_regularization(self):
        """When lengths are equal, R-DPO should match DPO regardless of alpha."""
        from oxrl.algs.dpo import DPO
        rdpo = self._make_rdpo(beta=0.1, alpha=0.5)
        dpo = DPO.__new__(DPO)
        dpo.beta = 0.1

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])
        same_len = torch.tensor([10.0, 10.0])

        rdpo_loss, _, _ = rdpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l, same_len, same_len)
        dpo_loss, _ = dpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        assert abs(rdpo_loss.item() - dpo_loss.item()) < 1e-5, \
            f"Equal lengths should match DPO: {rdpo_loss} vs {dpo_loss}"

    def test_loss_is_finite(self):
        """R-DPO loss should always be finite."""
        rdpo = self._make_rdpo(beta=0.1, alpha=0.01)

        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])
        len_w = torch.tensor([10.0, 20.0])
        len_l = torch.tensor([15.0, 25.0])

        loss, margin, avg_len_diff = rdpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l, len_w, len_l)

        assert torch.isfinite(loss), f"Loss should be finite, got {loss}"
        assert torch.isfinite(margin), f"Margin should be finite, got {margin}"

    def test_avg_length_diff_metric(self):
        """Test that avg_length_diff is computed correctly."""
        rdpo = self._make_rdpo(beta=0.1, alpha=0.01)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])
        len_w = torch.tensor([10.0, 20.0])
        len_l = torch.tensor([15.0, 10.0])

        _, _, avg_len_diff = rdpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l, len_w, len_l)

        expected = ((10.0 - 15.0) + (20.0 - 10.0)) / 2.0  # = 2.5
        assert abs(avg_len_diff.item() - expected) < 1e-5, \
            f"avg_length_diff should be {expected}, got {avg_len_diff}"

    def test_correct_preference_lower_loss(self):
        """When log-ratio favors chosen, loss should be lower than when it favors rejected."""
        rdpo = self._make_rdpo(beta=0.5, alpha=0.0)

        # Policy strongly prefers chosen (log-ratio much higher for chosen)
        pi_logps_w = torch.tensor([-0.5, -0.3])
        pi_logps_l = torch.tensor([-3.0, -2.5])
        ref_logps_w = torch.tensor([-1.0, -0.8])
        ref_logps_l = torch.tensor([-1.0, -0.8])
        len_w = torch.tensor([10.0, 10.0])
        len_l = torch.tensor([10.0, 10.0])

        loss_good, _, _ = rdpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l, len_w, len_l)
        # Swap chosen/rejected in policy logps only (ref stays same)
        loss_bad, _, _ = rdpo.compute_loss(
            pi_logps_l, pi_logps_w, ref_logps_w, ref_logps_l, len_w, len_l)

        assert loss_good.item() < loss_bad.item(), \
            f"Correct preference should have lower loss: {loss_good} vs {loss_bad}"


class TestRDPOConfig:
    """Test R-DPO config fields."""

    def test_config_has_rdpo_field(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="rdpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert hasattr(t, 'rdpo_alpha'), "Train should have rdpo_alpha"

    def test_config_default_value(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="rdpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.rdpo_alpha == 0.01

    def test_config_custom_value(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="rdpo", total_number_of_epochs=1,
                  micro_batches_per_epoch=10, rdpo_alpha=0.05)
        assert t.rdpo_alpha == 0.05
