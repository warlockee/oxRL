"""
Tests for FocalPO (Focal Preference Optimization) algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_focalpo.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestFocalPOImport:
    def test_focalpo_importable(self):
        from oxrl.algs.focalpo import FocalPO
        assert FocalPO is not None

    def test_focalpo_in_algs_init(self):
        from oxrl.algs import FocalPO
        assert FocalPO is not None

    def test_focalpo_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"focalpo"' in source


class TestFocalPOInterface:
    def test_has_required_methods(self):
        from oxrl.algs.focalpo import FocalPO
        assert hasattr(FocalPO, 'compute_logps')
        assert hasattr(FocalPO, 'forward')
        assert hasattr(FocalPO, 'compute_loss')
        assert hasattr(FocalPO, 'train_step')
        assert hasattr(FocalPO, 'eval_step')

    def test_init_params(self):
        from oxrl.algs.focalpo import FocalPO
        sig = inspect.signature(FocalPO.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params
        assert 'focalpo_gamma' in params


class TestFocalPOLossMath:
    def _make_focalpo(self, beta=0.1, focalpo_gamma=1.0):
        from oxrl.algs.focalpo import FocalPO
        f = FocalPO.__new__(FocalPO)
        f.beta = beta
        f.focalpo_gamma = focalpo_gamma
        return f

    def test_loss_formula_verification(self):
        """Verify: L = -p^gamma * logsigmoid(beta * h)."""
        beta = 0.2
        gamma = 2.0
        f = self._make_focalpo(beta=beta, focalpo_gamma=gamma)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = f.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        h = beta * (logr_w - logr_l)
        p = torch.sigmoid(h)
        expected = -(p.pow(gamma) * F.logsigmoid(h)).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_reduces_to_dpo_when_gamma_zero(self):
        """When gamma=0, p^0 = 1, so FocalPO = DPO."""
        beta = 0.1
        f = self._make_focalpo(beta=beta, focalpo_gamma=0.0)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = f.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        h = beta * ((pi_logps_w - ref_logps_w) - (pi_logps_l - ref_logps_l))
        dpo_loss = -F.logsigmoid(h).mean()

        assert abs(loss.item() - dpo_loss.item()) < 1e-5

    def test_loss_lower_than_dpo_initially(self):
        """At initialization (h~0, p~0.5), FocalPO loss < DPO loss for gamma>0."""
        pi_logps_w = torch.tensor([0.0])
        pi_logps_l = torch.tensor([0.0])
        ref_logps_w = torch.tensor([0.0])
        ref_logps_l = torch.tensor([0.0])

        f_focal = self._make_focalpo(beta=0.1, focalpo_gamma=2.0)
        f_dpo = self._make_focalpo(beta=0.1, focalpo_gamma=0.0)

        loss_focal, _, _ = f_focal.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss_dpo, _, _ = f_dpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # At h=0: p=0.5, focal_weight = 0.5^2 = 0.25, so focal_loss = 0.25 * dpo_loss
        assert loss_focal.item() < loss_dpo.item()

    def test_focal_weight_effect(self):
        """Higher gamma should reduce loss for uncertain samples more."""
        pi_logps_w = torch.tensor([0.0])
        pi_logps_l = torch.tensor([0.0])
        ref_logps_w = torch.tensor([0.0])
        ref_logps_l = torch.tensor([0.0])

        f_low = self._make_focalpo(beta=0.1, focalpo_gamma=1.0)
        f_high = self._make_focalpo(beta=0.1, focalpo_gamma=3.0)

        loss_low, _, _ = f_low.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss_high, _, _ = f_high.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # Higher gamma gives more down-weighting at p=0.5
        assert loss_high.item() < loss_low.item()

    def test_loss_always_positive(self):
        """FocalPO loss should always be positive."""
        f = self._make_focalpo(beta=0.1, focalpo_gamma=1.0)
        for _ in range(10):
            pi_logps_w = torch.randn(4)
            pi_logps_l = torch.randn(4)
            ref_logps_w = torch.randn(4)
            ref_logps_l = torch.randn(4)
            loss, _, _ = f.compute_loss(
                pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
            assert loss.item() > 0

    def test_loss_is_finite(self):
        f = self._make_focalpo(beta=0.1, focalpo_gamma=1.0)
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])
        loss, margin, reward_acc = f.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert torch.isfinite(loss)
        assert torch.isfinite(margin)

    def test_reward_accuracy(self):
        f = self._make_focalpo(beta=0.1, focalpo_gamma=1.0)
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])
        _, _, reward_acc = f.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert reward_acc.item() == 1.0

    def test_returns_three_values(self):
        f = self._make_focalpo(beta=0.1, focalpo_gamma=1.0)
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        result = f.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert len(result) == 3


class TestFocalPOConfig:
    def test_config_accepts_focalpo(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="focalpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alg_name == "focalpo"
        assert hasattr(t, 'focalpo_gamma')

    def test_config_default_value(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="focalpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.focalpo_gamma == 1.0


class TestFocalPOComputeLogps:
    def _make_focalpo(self):
        from oxrl.algs.focalpo import FocalPO
        f = FocalPO.__new__(FocalPO)
        f.beta = 0.1
        f.focalpo_gamma = 1.0
        return f

    def test_compute_logps_shape(self):
        f = self._make_focalpo()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = f.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_logps_mask(self):
        f = self._make_focalpo()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        mask_full = torch.ones(1, 5)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        logps_full = f.compute_logps(logits, target_ids, mask_full)
        logps_half = f.compute_logps(logits, target_ids, mask_half)
        assert logps_full.item() != logps_half.item()
