"""
Tests for Chi-PO (Chi-Squared Preference Optimization) algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_chipo.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestChiPOImport:
    def test_chipo_importable(self):
        from oxrl.algs.chipo import ChiPO
        assert ChiPO is not None

    def test_chipo_in_algs_init(self):
        from oxrl.algs import ChiPO
        assert ChiPO is not None

    def test_chipo_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"chipo"' in source


class TestChiPOInterface:
    def test_has_required_methods(self):
        from oxrl.algs.chipo import ChiPO
        assert hasattr(ChiPO, 'compute_logps')
        assert hasattr(ChiPO, 'forward')
        assert hasattr(ChiPO, 'chi_link')
        assert hasattr(ChiPO, 'compute_loss')
        assert hasattr(ChiPO, 'train_step')
        assert hasattr(ChiPO, 'eval_step')

    def test_init_params(self):
        from oxrl.algs.chipo import ChiPO
        sig = inspect.signature(ChiPO.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params

    def test_no_extra_hyperparameters(self):
        """Chi-PO should use the same hyperparameters as DPO (no new ones)."""
        from oxrl.algs.chipo import ChiPO
        from oxrl.algs.dpo import DPO
        chipo_params = set(inspect.signature(ChiPO.__init__).parameters.keys())
        dpo_params = set(inspect.signature(DPO.__init__).parameters.keys())
        assert chipo_params == dpo_params


class TestChiLinkFunction:
    def _make_chipo(self, beta=0.1):
        from oxrl.algs.chipo import ChiPO
        c = ChiPO.__new__(ChiPO)
        c.beta = beta
        return c

    def test_chi_link_at_zero(self):
        """phi(exp(0)) = exp(0) + 0 = 1 + 0 = 1."""
        c = self._make_chipo()
        result = c.chi_link(torch.tensor([0.0]))
        assert abs(result.item() - 1.0) < 1e-5

    def test_chi_link_positive_logr(self):
        """phi(exp(logr)) = exp(logr) + logr for logr > 0."""
        c = self._make_chipo()
        logr = torch.tensor([1.0])
        result = c.chi_link(logr)
        expected = math.exp(1.0) + 1.0
        assert abs(result.item() - expected) < 1e-5

    def test_chi_link_negative_logr(self):
        """phi(exp(logr)) = exp(logr) + logr for logr < 0."""
        c = self._make_chipo()
        logr = torch.tensor([-1.0])
        result = c.chi_link(logr)
        expected = math.exp(-1.0) + (-1.0)
        assert abs(result.item() - expected) < 1e-5

    def test_chi_link_is_monotonically_increasing(self):
        """The chi link function should be monotonically increasing."""
        c = self._make_chipo()
        logr = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = c.chi_link(logr)
        for i in range(len(result) - 1):
            assert result[i].item() < result[i + 1].item()

    def test_chi_link_batch(self):
        """Chi link should work on batches."""
        c = self._make_chipo()
        logr = torch.tensor([-1.0, 0.0, 1.0])
        result = c.chi_link(logr)
        assert result.shape == (3,)

    def test_chi_link_clamping(self):
        """Very large logr values should be clamped to prevent overflow."""
        c = self._make_chipo()
        logr = torch.tensor([100.0])
        result = c.chi_link(logr)
        assert torch.isfinite(result)


class TestChiPOLossMath:
    def _make_chipo(self, beta=0.1):
        from oxrl.algs.chipo import ChiPO
        c = ChiPO.__new__(ChiPO)
        c.beta = beta
        return c

    def test_loss_formula(self):
        """Verify Chi-PO loss: -logsigmoid(beta * (phi(r_w) - phi(r_l)))."""
        beta = 0.2
        c = self._make_chipo(beta=beta)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = c.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # Manual computation
        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        phi_w = torch.exp(logr_w) + logr_w
        phi_l = torch.exp(logr_l) + logr_l
        logits = beta * (phi_w - phi_l)
        expected = -F.logsigmoid(logits).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_differs_from_dpo(self):
        """Chi-PO should give a different loss than standard DPO."""
        beta = 0.1
        c = self._make_chipo(beta=beta)

        # Use asymmetric log-ratios so phi(r_w) - phi(r_l) != logr_w - logr_l
        pi_logps_w = torch.tensor([-0.5, -0.3])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0])

        chipo_loss, _, _ = c.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # Standard DPO loss
        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        dpo_logits = beta * (logr_w - logr_l)
        dpo_loss = -F.logsigmoid(dpo_logits).mean()

        assert abs(chipo_loss.item() - dpo_loss.item()) > 1e-4

    def test_approx_dpo_when_close_to_ref(self):
        """When pi ~= ref (logr ~= 0), Chi-PO should be close to DPO + constant offset."""
        beta = 0.1
        c = self._make_chipo(beta=beta)

        # Very small log-ratios (pi close to ref)
        pi_logps_w = torch.tensor([-1.000, -0.500])
        pi_logps_l = torch.tensor([-2.000, -1.500])
        ref_logps_w = torch.tensor([-1.001, -0.501])
        ref_logps_l = torch.tensor([-2.001, -1.501])

        chipo_loss, _, _ = c.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # When logr ~= 0, phi(exp(logr)) = exp(logr) + logr ~= 1 + logr
        # So phi_w - phi_l ~= logr_w - logr_l (same as DPO up to constant cancellation)
        # The losses should be very close
        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        dpo_logits = beta * (logr_w - logr_l)
        dpo_loss = -F.logsigmoid(dpo_logits).mean()

        # They should be quite close when log-ratios are near zero
        assert abs(chipo_loss.item() - dpo_loss.item()) < 0.05

    def test_loss_is_finite(self):
        c = self._make_chipo(beta=0.1)
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])
        loss, margin, reward_acc = c.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert torch.isfinite(loss)
        assert torch.isfinite(margin)

    def test_reward_accuracy(self):
        c = self._make_chipo(beta=0.1)
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])
        _, _, reward_acc = c.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert reward_acc.item() == 1.0

    def test_returns_three_values(self):
        c = self._make_chipo(beta=0.1)
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        result = c.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert len(result) == 3

    def test_loss_positive(self):
        c = self._make_chipo(beta=0.1)
        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])
        loss, _, _ = c.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert loss.item() > 0


class TestChiPOGradientFlow:
    def _make_chipo(self, beta=0.1):
        from oxrl.algs.chipo import ChiPO
        c = ChiPO.__new__(ChiPO)
        c.beta = beta
        return c

    def test_gradient_flow(self):
        """Verify gradients flow through the Chi-PO loss."""
        c = self._make_chipo(beta=0.1)
        pi_logps_w = torch.tensor([-1.0, -0.5], requires_grad=True)
        pi_logps_l = torch.tensor([-2.0, -1.5], requires_grad=True)
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = c.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss.backward()

        assert pi_logps_w.grad is not None
        assert pi_logps_l.grad is not None
        assert torch.all(torch.isfinite(pi_logps_w.grad))
        assert torch.all(torch.isfinite(pi_logps_l.grad))


class TestChiPOComputeLogps:
    def _make_chipo(self):
        from oxrl.algs.chipo import ChiPO
        c = ChiPO.__new__(ChiPO)
        c.beta = 0.1
        return c

    def test_compute_logps_shape(self):
        c = self._make_chipo()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = c.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_logps_mask(self):
        c = self._make_chipo()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        mask_full = torch.ones(1, 5)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        logps_full = c.compute_logps(logits, target_ids, mask_full)
        logps_half = c.compute_logps(logits, target_ids, mask_half)
        assert logps_full.item() != logps_half.item()


class TestChiPOConfig:
    def test_config_accepts_chipo(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="chipo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alg_name == "chipo"
