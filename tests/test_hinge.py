"""
Tests for Hinge (SLiC-HF) preference optimization algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_hinge.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestHingeImport:
    def test_hinge_importable(self):
        from oxrl.algs.hinge import Hinge
        assert Hinge is not None

    def test_hinge_in_algs_init(self):
        from oxrl.algs import Hinge
        assert Hinge is not None

    def test_hinge_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"hinge"' in source


class TestHingeInterface:
    def test_hinge_has_required_methods(self):
        from oxrl.algs.hinge import Hinge
        assert hasattr(Hinge, 'compute_logps')
        assert hasattr(Hinge, 'forward')
        assert hasattr(Hinge, 'compute_loss')
        assert hasattr(Hinge, 'train_step')
        assert hasattr(Hinge, 'eval_step')

    def test_hinge_init_params(self):
        from oxrl.algs.hinge import Hinge
        sig = inspect.signature(Hinge.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params


class TestHingeLossMath:
    def _make_hinge(self, beta=0.1):
        from oxrl.algs.hinge import Hinge
        h = Hinge.__new__(Hinge)
        h.beta = beta
        return h

    def test_loss_formula_verification(self):
        """Verify: L = max(0, 1 - beta * (logr_w - logr_l))."""
        beta = 0.2
        h = self._make_hinge(beta=beta)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = h.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        expected = torch.relu(1.0 - beta * (logr_w - logr_l)).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_loss_zero_when_margin_large(self):
        """Loss should be zero when reward margin exceeds 1/beta."""
        beta = 0.1
        h = self._make_hinge(beta=beta)
        # Need margin > 1/beta = 10

        pi_logps_w = torch.tensor([5.0])
        pi_logps_l = torch.tensor([-10.0])
        ref_logps_w = torch.tensor([0.0])
        ref_logps_l = torch.tensor([0.0])

        loss, _, _ = h.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # logr_w - logr_l = 5 - (-10) = 15
        # beta * 15 = 1.5 > 1, so relu(1 - 1.5) = relu(-0.5) = 0
        assert abs(loss.item()) < 1e-5

    def test_loss_positive_when_margin_small(self):
        """Loss should be positive when margin is below threshold."""
        h = self._make_hinge(beta=0.1)

        # Tiny margin
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-1.1])
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-1.0])

        loss, _, _ = h.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # logr_w - logr_l = 0 - (-0.1) = 0.1
        # beta * 0.1 = 0.01 < 1, so relu(1 - 0.01) = 0.99
        assert loss.item() > 0

    def test_loss_always_non_negative(self):
        """Hinge loss is always >= 0."""
        h = self._make_hinge(beta=0.1)
        for _ in range(10):
            pi_logps_w = torch.randn(4)
            pi_logps_l = torch.randn(4)
            ref_logps_w = torch.randn(4)
            ref_logps_l = torch.randn(4)
            loss, _, _ = h.compute_loss(
                pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
            assert loss.item() >= 0

    def test_loss_is_finite(self):
        h = self._make_hinge(beta=0.1)
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])
        loss, margin, reward_acc = h.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert torch.isfinite(loss)
        assert torch.isfinite(margin)

    def test_reward_accuracy(self):
        h = self._make_hinge(beta=0.1)
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])
        _, _, reward_acc = h.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert reward_acc.item() == 1.0

    def test_returns_three_values(self):
        h = self._make_hinge(beta=0.1)
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        result = h.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert len(result) == 3


class TestHingeConfig:
    def test_config_accepts_hinge(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="hinge", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alg_name == "hinge"
        assert hasattr(t, 'beta')


class TestHingeComputeLogps:
    def _make_hinge(self):
        from oxrl.algs.hinge import Hinge
        h = Hinge.__new__(Hinge)
        h.beta = 0.1
        return h

    def test_compute_logps_shape(self):
        h = self._make_hinge()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = h.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_logps_mask(self):
        h = self._make_hinge()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        mask_full = torch.ones(1, 5)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        logps_full = h.compute_logps(logits, target_ids, mask_full)
        logps_half = h.compute_logps(logits, target_ids, mask_half)
        assert logps_full.item() != logps_half.item()
