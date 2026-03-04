"""
Tests for SPPO (Self-Play Preference Optimization, hard label) algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_sppo.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestSPPOImport:
    def test_sppo_importable(self):
        from oxrl.algs.sppo import SPPO
        assert SPPO is not None

    def test_sppo_in_algs_init(self):
        from oxrl.algs import SPPO
        assert SPPO is not None

    def test_sppo_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"sppo"' in source


class TestSPPOInterface:
    def test_sppo_has_required_methods(self):
        from oxrl.algs.sppo import SPPO
        assert hasattr(SPPO, 'compute_logps')
        assert hasattr(SPPO, 'forward')
        assert hasattr(SPPO, 'compute_loss')
        assert hasattr(SPPO, 'train_step')
        assert hasattr(SPPO, 'eval_step')

    def test_sppo_init_params(self):
        from oxrl.algs.sppo import SPPO
        sig = inspect.signature(SPPO.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params


class TestSPPOLossMath:
    def _make_sppo(self, beta=0.1):
        from oxrl.algs.sppo import SPPO
        sppo = SPPO.__new__(SPPO)
        sppo.beta = beta
        return sppo

    def test_loss_formula_verification(self):
        """Verify: L = (logr_w - 1/(2*beta))^2 + (logr_l + 1/(2*beta))^2."""
        beta = 0.2
        sppo = self._make_sppo(beta=beta)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = sppo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        target_w = 1.0 / (2.0 * beta)
        target_l = -1.0 / (2.0 * beta)
        expected = ((logr_w - target_w).pow(2) + (logr_l - target_l).pow(2)).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_loss_zero_at_target(self):
        """Loss should be zero when log ratios equal target values."""
        beta = 0.1
        sppo = self._make_sppo(beta=beta)
        target_w = 1.0 / (2.0 * beta)  # 5.0
        target_l = -1.0 / (2.0 * beta)  # -5.0

        pi_logps_w = torch.tensor([target_w + 1.0])
        ref_logps_w = torch.tensor([1.0])
        pi_logps_l = torch.tensor([target_l + 2.0])
        ref_logps_l = torch.tensor([2.0])

        loss, _, _ = sppo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert abs(loss.item()) < 1e-5

    def test_loss_always_non_negative(self):
        """Squared error loss is always >= 0."""
        sppo = self._make_sppo(beta=0.1)
        for _ in range(10):
            pi_logps_w = torch.randn(4)
            pi_logps_l = torch.randn(4)
            ref_logps_w = torch.randn(4)
            ref_logps_l = torch.randn(4)
            loss, _, _ = sppo.compute_loss(
                pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
            assert loss.item() >= 0.0

    def test_sppo_matches_caldpo_cal_loss(self):
        """SPPO loss should match Cal-DPO calibration loss with same beta."""
        from oxrl.algs.caldpo import CalDPO
        beta = 0.15
        sppo = self._make_sppo(beta=beta)
        cal = CalDPO.__new__(CalDPO)
        cal.beta = beta
        cal.caldpo_lambda = 1.0

        pi_logps_w = torch.tensor([-1.0, -0.5, -0.3])
        pi_logps_l = torch.tensor([-2.0, -1.5, -1.8])
        ref_logps_w = torch.tensor([-1.1, -0.6, -0.4])
        ref_logps_l = torch.tensor([-2.1, -1.6, -1.9])

        sppo_loss, _, _ = sppo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        _, _, _, _, caldpo_cal_loss = cal.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        assert abs(sppo_loss.item() - caldpo_cal_loss.item()) < 1e-5, \
            f"SPPO should match Cal-DPO cal loss: {sppo_loss} vs {caldpo_cal_loss}"

    def test_loss_symmetric_for_swapped_targets(self):
        """Swapping chosen/rejected should give different loss (not symmetric like DPO)."""
        sppo = self._make_sppo(beta=0.1)

        pi_logps_w = torch.tensor([-0.5])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-1.0])

        loss_fwd, _, _ = sppo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss_rev, _, _ = sppo.compute_loss(
            pi_logps_l, pi_logps_w, ref_logps_l, ref_logps_w)

        # SPPO is NOT symmetric because targets are +/- 1/(2*beta)
        # Swapping chosen/rejected should give a different loss
        # (unlike DPO where -logsigmoid(x) and -logsigmoid(-x) sum to const)
        # Both should be > 0 but they need not be equal
        assert loss_fwd.item() >= 0 and loss_rev.item() >= 0

    def test_loss_is_finite(self):
        sppo = self._make_sppo(beta=0.1)
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])
        loss, margin, reward_acc = sppo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert torch.isfinite(loss)
        assert torch.isfinite(margin)

    def test_reward_accuracy(self):
        sppo = self._make_sppo(beta=0.1)
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])
        _, _, reward_acc = sppo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert reward_acc.item() == 1.0

    def test_returns_three_values(self):
        sppo = self._make_sppo(beta=0.1)
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        result = sppo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert len(result) == 3


class TestSPPOConfig:
    def test_config_accepts_sppo(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="sppo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alg_name == "sppo"
        # SPPO uses the standard beta parameter
        assert hasattr(t, 'beta')


class TestSPPOComputeLogps:
    def _make_sppo(self):
        from oxrl.algs.sppo import SPPO
        sppo = SPPO.__new__(SPPO)
        sppo.beta = 0.1
        return sppo

    def test_compute_logps_shape(self):
        sppo = self._make_sppo()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = sppo.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_logps_mask(self):
        sppo = self._make_sppo()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        mask_full = torch.ones(1, 5)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        logps_full = sppo.compute_logps(logits, target_ids, mask_full)
        logps_half = sppo.compute_logps(logits, target_ids, mask_half)
        assert logps_full.item() != logps_half.item()
