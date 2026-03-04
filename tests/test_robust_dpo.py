"""
Tests for Robust DPO preference optimization algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_robust_dpo.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestRobustDPOImport:
    def test_robust_dpo_importable(self):
        from oxrl.algs.robust_dpo import RobustDPO
        assert RobustDPO is not None

    def test_robust_dpo_in_algs_init(self):
        from oxrl.algs import RobustDPO
        assert RobustDPO is not None

    def test_robust_dpo_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"robust_dpo"' in source


class TestRobustDPOInterface:
    def test_has_required_methods(self):
        from oxrl.algs.robust_dpo import RobustDPO
        assert hasattr(RobustDPO, 'compute_logps')
        assert hasattr(RobustDPO, 'forward')
        assert hasattr(RobustDPO, 'compute_loss')
        assert hasattr(RobustDPO, 'train_step')
        assert hasattr(RobustDPO, 'eval_step')

    def test_init_params(self):
        from oxrl.algs.robust_dpo import RobustDPO
        sig = inspect.signature(RobustDPO.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params
        assert 'label_smoothing' in params


class TestRobustDPOLossMath:
    def _make_robust_dpo(self, beta=0.1, label_smoothing=0.1):
        from oxrl.algs.robust_dpo import RobustDPO
        r = RobustDPO.__new__(RobustDPO)
        r.beta = beta
        r.label_smoothing = label_smoothing
        return r

    def test_loss_formula_verification(self):
        """Verify: L = -[(1-eps)*logsigmoid(beta*h) + eps*logsigmoid(-beta*h)] / (1-2*eps)."""
        beta = 0.2
        eps = 0.15
        r = self._make_robust_dpo(beta=beta, label_smoothing=eps)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = r.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        h = beta * (logr_w - logr_l)
        numerator = (1 - eps) * F.logsigmoid(h) + eps * F.logsigmoid(-h)
        expected = -(numerator / (1 - 2 * eps)).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_reduces_to_dpo_when_eps_zero(self):
        """When eps=0, Robust DPO should equal standard DPO loss."""
        beta = 0.1
        r = self._make_robust_dpo(beta=beta, label_smoothing=0.0)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = r.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        h = beta * ((pi_logps_w - ref_logps_w) - (pi_logps_l - ref_logps_l))
        dpo_loss = -F.logsigmoid(h).mean()

        assert abs(loss.item() - dpo_loss.item()) < 1e-5

    def test_differs_from_cdpo(self):
        """Robust DPO differs from cDPO by the 1/(1-2*eps) normalization."""
        beta = 0.1
        eps = 0.2
        r = self._make_robust_dpo(beta=beta, label_smoothing=eps)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        robust_loss, _, _ = r.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # cDPO loss (without normalization)
        h = beta * ((pi_logps_w - ref_logps_w) - (pi_logps_l - ref_logps_l))
        cdpo_loss = -((1 - eps) * F.logsigmoid(h) + eps * F.logsigmoid(-h)).mean()

        # Robust = cDPO / (1 - 2*eps)
        expected_ratio = 1.0 / (1.0 - 2.0 * eps)
        assert abs(robust_loss.item() / cdpo_loss.item() - expected_ratio) < 1e-4

    def test_loss_always_positive(self):
        """Robust DPO loss should always be positive."""
        r = self._make_robust_dpo(beta=0.1, label_smoothing=0.1)
        for _ in range(10):
            pi_logps_w = torch.randn(4)
            pi_logps_l = torch.randn(4)
            ref_logps_w = torch.randn(4)
            ref_logps_l = torch.randn(4)
            loss, _, _ = r.compute_loss(
                pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
            assert loss.item() > 0

    def test_loss_is_finite(self):
        r = self._make_robust_dpo(beta=0.1, label_smoothing=0.1)
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])
        loss, margin, reward_acc = r.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert torch.isfinite(loss)
        assert torch.isfinite(margin)

    def test_loss_increases_with_larger_eps(self):
        """Higher noise rate should increase the loss magnitude."""
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])

        r_low = self._make_robust_dpo(beta=0.1, label_smoothing=0.05)
        r_high = self._make_robust_dpo(beta=0.1, label_smoothing=0.3)

        loss_low, _, _ = r_low.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss_high, _, _ = r_high.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # With higher eps, the normalization 1/(1-2*eps) amplifies the loss
        assert loss_high.item() > loss_low.item()

    def test_reward_accuracy(self):
        r = self._make_robust_dpo(beta=0.1, label_smoothing=0.1)
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])
        _, _, reward_acc = r.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert reward_acc.item() == 1.0

    def test_returns_three_values(self):
        r = self._make_robust_dpo(beta=0.1, label_smoothing=0.1)
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        result = r.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert len(result) == 3


class TestRobustDPOConfig:
    def test_config_accepts_robust_dpo(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="robust_dpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alg_name == "robust_dpo"
        assert hasattr(t, 'robust_dpo_label_smoothing')

    def test_config_default_value(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="robust_dpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.robust_dpo_label_smoothing == 0.1


class TestRobustDPOComputeLogps:
    def _make_robust_dpo(self):
        from oxrl.algs.robust_dpo import RobustDPO
        r = RobustDPO.__new__(RobustDPO)
        r.beta = 0.1
        r.label_smoothing = 0.1
        return r

    def test_compute_logps_shape(self):
        r = self._make_robust_dpo()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = r.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_logps_mask(self):
        r = self._make_robust_dpo()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        mask_full = torch.ones(1, 5)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        logps_full = r.compute_logps(logits, target_ids, mask_full)
        logps_half = r.compute_logps(logits, target_ids, mask_half)
        assert logps_full.item() != logps_half.item()
