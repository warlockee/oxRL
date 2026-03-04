"""
Tests for Cal-DPO (Calibrated Direct Preference Optimization) algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_caldpo.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestCalDPOImport:
    def test_caldpo_importable(self):
        from oxrl.algs.caldpo import CalDPO
        assert CalDPO is not None

    def test_caldpo_in_algs_init(self):
        from oxrl.algs import CalDPO
        assert CalDPO is not None

    def test_caldpo_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"caldpo"' in source


class TestCalDPOInterface:
    def test_caldpo_has_required_methods(self):
        from oxrl.algs.caldpo import CalDPO
        assert hasattr(CalDPO, 'compute_logps')
        assert hasattr(CalDPO, 'forward')
        assert hasattr(CalDPO, 'compute_loss')
        assert hasattr(CalDPO, 'train_step')
        assert hasattr(CalDPO, 'eval_step')

    def test_caldpo_init_params(self):
        from oxrl.algs.caldpo import CalDPO
        sig = inspect.signature(CalDPO.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params
        assert 'caldpo_lambda' in params


class TestCalDPOLossMath:
    def _make_caldpo(self, beta=0.1, caldpo_lambda=1.0):
        from oxrl.algs.caldpo import CalDPO
        cal = CalDPO.__new__(CalDPO)
        cal.beta = beta
        cal.caldpo_lambda = caldpo_lambda
        return cal

    def test_zero_lambda_is_bt_loss(self):
        """With lambda=0, Cal-DPO should be the BT preference loss only."""
        cal = self._make_caldpo(beta=0.1, caldpo_lambda=0.0)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _, loss_bt, loss_cal = cal.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # With lambda=0, total loss should equal BT loss
        assert abs(loss.item() - loss_bt.item()) < 1e-5

    def test_bt_loss_formula(self):
        """Verify: L_BT = -logsigmoid(log_ratio_w - log_ratio_l), no beta scaling."""
        cal = self._make_caldpo(beta=0.2, caldpo_lambda=0.0)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _, _, _ = cal.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        # Note: no beta scaling in BT loss
        expected = -F.logsigmoid(logr_w - logr_l).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_calibration_loss_formula(self):
        """Verify: L_Cal = (logr_w - 1/(2*beta))^2 + (logr_l + 1/(2*beta))^2."""
        cal = self._make_caldpo(beta=0.2, caldpo_lambda=1.0)

        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])

        _, _, _, _, loss_cal = cal.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        logr_w = pi_logps_w - ref_logps_w  # 0.1
        logr_l = pi_logps_l - ref_logps_l  # 0.1
        target_w = 1.0 / (2.0 * 0.2)  # 2.5
        target_l = -1.0 / (2.0 * 0.2)  # -2.5
        expected_cal = ((logr_w - target_w).pow(2) + (logr_l - target_l).pow(2)).mean()

        assert abs(loss_cal.item() - expected_cal.item()) < 1e-5

    def test_combined_loss_formula(self):
        """Verify: L = L_BT + lambda * L_Cal."""
        lam = 0.5
        cal = self._make_caldpo(beta=0.1, caldpo_lambda=lam)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _, loss_bt, loss_cal = cal.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        expected = loss_bt + lam * loss_cal
        assert abs(loss.item() - expected.item()) < 1e-5

    def test_calibration_increases_loss(self):
        """Calibration term should increase total loss (it's always >= 0)."""
        cal_no = self._make_caldpo(beta=0.1, caldpo_lambda=0.0)
        cal_yes = self._make_caldpo(beta=0.1, caldpo_lambda=1.0)

        pi_logps_w = torch.tensor([-0.5, -0.3])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.0, -0.8])
        ref_logps_l = torch.tensor([-1.0, -0.8])

        loss_no, _, _, _, _ = cal_no.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss_yes, _, _, _, _ = cal_yes.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        assert loss_yes.item() >= loss_no.item(), \
            f"Calibration should increase loss: {loss_yes} vs {loss_no}"

    def test_calibration_loss_zero_at_targets(self):
        """Calibration loss should be zero when rewards are at target values."""
        beta = 0.1
        cal = self._make_caldpo(beta=beta, caldpo_lambda=1.0)
        target_w = 1.0 / (2.0 * beta)  # 5.0
        target_l = -1.0 / (2.0 * beta)  # -5.0

        # Set log ratios to exactly the target values
        pi_logps_w = torch.tensor([target_w + 1.0])
        ref_logps_w = torch.tensor([1.0])
        pi_logps_l = torch.tensor([target_l + 2.0])
        ref_logps_l = torch.tensor([2.0])

        _, _, _, _, loss_cal = cal.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert abs(loss_cal.item()) < 1e-5, \
            f"Calibration loss should be 0 at targets: {loss_cal}"

    def test_loss_is_finite(self):
        cal = self._make_caldpo(beta=0.1, caldpo_lambda=1.0)
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])
        loss, margin, reward_acc, _, _ = cal.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert torch.isfinite(loss)
        assert torch.isfinite(margin)

    def test_reward_accuracy(self):
        cal = self._make_caldpo(beta=0.1, caldpo_lambda=1.0)
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])
        _, _, reward_acc, _, _ = cal.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert reward_acc.item() == 1.0, "All chosen should have higher reward"

    def test_returns_five_values(self):
        cal = self._make_caldpo(beta=0.1, caldpo_lambda=1.0)
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        result = cal.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert len(result) == 5, "compute_loss should return (loss, margin, reward_acc, loss_bt, loss_cal)"


class TestCalDPOConfig:
    def test_config_has_caldpo_field(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="caldpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert hasattr(t, 'caldpo_lambda')

    def test_config_default_value(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="caldpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.caldpo_lambda == 1.0

    def test_config_custom_value(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="caldpo", total_number_of_epochs=1,
                  micro_batches_per_epoch=10, caldpo_lambda=0.5)
        assert t.caldpo_lambda == 0.5


class TestCalDPOComputeLogps:
    def _make_caldpo(self):
        from oxrl.algs.caldpo import CalDPO
        cal = CalDPO.__new__(CalDPO)
        cal.beta = 0.1
        cal.caldpo_lambda = 1.0
        return cal

    def test_compute_logps_shape(self):
        cal = self._make_caldpo()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = cal.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_logps_mask(self):
        cal = self._make_caldpo()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        mask_full = torch.ones(1, 5)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        logps_full = cal.compute_logps(logits, target_ids, mask_full)
        logps_half = cal.compute_logps(logits, target_ids, mask_half)
        assert logps_full.item() != logps_half.item()
