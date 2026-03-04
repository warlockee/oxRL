"""
Tests for cDPO (Conservative DPO with Label Smoothing) algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_cdpo.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestCDPOImport:
    def test_cdpo_importable(self):
        from oxrl.algs.cdpo import CDPO
        assert CDPO is not None

    def test_cdpo_in_algs_init(self):
        from oxrl.algs import CDPO
        assert CDPO is not None

    def test_cdpo_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"cdpo"' in source


class TestCDPOInterface:
    def test_cdpo_has_required_methods(self):
        from oxrl.algs.cdpo import CDPO
        assert hasattr(CDPO, 'compute_logps')
        assert hasattr(CDPO, 'forward')
        assert hasattr(CDPO, 'compute_loss')
        assert hasattr(CDPO, 'train_step')
        assert hasattr(CDPO, 'eval_step')

    def test_cdpo_init_params(self):
        from oxrl.algs.cdpo import CDPO
        sig = inspect.signature(CDPO.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params
        assert 'label_smoothing' in params


class TestCDPOLossMath:
    def _make_cdpo(self, beta=0.1, label_smoothing=0.1):
        from oxrl.algs.cdpo import CDPO
        cdpo = CDPO.__new__(CDPO)
        cdpo.beta = beta
        cdpo.label_smoothing = label_smoothing
        return cdpo

    def test_zero_smoothing_matches_dpo(self):
        """With label_smoothing=0, cDPO should match DPO exactly."""
        from oxrl.algs.dpo import DPO
        cdpo = self._make_cdpo(beta=0.1, label_smoothing=0.0)
        dpo = DPO.__new__(DPO)
        dpo.beta = 0.1

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        cdpo_loss, _, _ = cdpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        dpo_loss, _ = dpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        assert abs(cdpo_loss.item() - dpo_loss.item()) < 1e-5, \
            f"eps=0 cDPO should match DPO: {cdpo_loss} vs {dpo_loss}"

    def test_loss_formula_verification(self):
        """Verify: loss = -[(1-eps)*logsigmoid(beta*h) + eps*logsigmoid(-beta*h)]."""
        cdpo = self._make_cdpo(beta=0.2, label_smoothing=0.15)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = cdpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        h = 0.2 * (logr_w - logr_l)
        expected = -(0.85 * F.logsigmoid(h) + 0.15 * F.logsigmoid(-h)).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_smoothing_increases_loss(self):
        """Label smoothing should increase loss when model has correct preference."""
        cdpo_clean = self._make_cdpo(label_smoothing=0.0)
        cdpo_smooth = self._make_cdpo(label_smoothing=0.2)

        pi_logps_w = torch.tensor([-0.5, -0.3])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.0, -0.8])
        ref_logps_l = torch.tensor([-1.0, -0.8])

        loss_clean, _, _ = cdpo_clean.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss_smooth, _, _ = cdpo_smooth.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # When model is correct, smoothing should increase loss (makes it harder)
        assert loss_smooth.item() > loss_clean.item(), \
            f"Smoothing should increase loss: {loss_smooth} vs {loss_clean}"

    def test_max_smoothing_flips(self):
        """At label_smoothing=0.5, the loss should be symmetric (no preference direction)."""
        cdpo = self._make_cdpo(beta=0.1, label_smoothing=0.5)

        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])

        loss_fwd, _, _ = cdpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss_rev, _, _ = cdpo.compute_loss(
            pi_logps_l, pi_logps_w, ref_logps_l, ref_logps_w)

        # At eps=0.5, logsigmoid(x) and logsigmoid(-x) are equally weighted
        # so flipping should give the same loss
        assert abs(loss_fwd.item() - loss_rev.item()) < 1e-5, \
            "eps=0.5 should be symmetric"

    def test_loss_is_finite(self):
        cdpo = self._make_cdpo(beta=0.1, label_smoothing=0.1)
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])
        loss, margin, reward_acc = cdpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert torch.isfinite(loss)
        assert torch.isfinite(margin)

    def test_reward_accuracy(self):
        cdpo = self._make_cdpo(beta=0.1, label_smoothing=0.1)
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])
        _, _, reward_acc = cdpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert reward_acc.item() == 1.0, "All chosen should have higher reward"


class TestCDPOConfig:
    def test_config_has_cdpo_field(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="cdpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert hasattr(t, 'cdpo_label_smoothing')

    def test_config_default_value(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="cdpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.cdpo_label_smoothing == 0.1

    def test_config_custom_value(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="cdpo", total_number_of_epochs=1,
                  micro_batches_per_epoch=10, cdpo_label_smoothing=0.2)
        assert t.cdpo_label_smoothing == 0.2
