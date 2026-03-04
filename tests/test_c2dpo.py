"""
Tests for C2-DPO (Constrained Controlled DPO) algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_c2dpo.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestC2DPOImport:
    def test_c2dpo_importable(self):
        from oxrl.algs.c2dpo import C2DPO
        assert C2DPO is not None

    def test_c2dpo_in_algs_init(self):
        from oxrl.algs import C2DPO
        assert C2DPO is not None

    def test_c2dpo_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"c2dpo"' in source


class TestC2DPOInterface:
    def test_has_required_methods(self):
        from oxrl.algs.c2dpo import C2DPO
        assert hasattr(C2DPO, 'compute_logps')
        assert hasattr(C2DPO, 'forward')
        assert hasattr(C2DPO, 'compute_loss')
        assert hasattr(C2DPO, 'train_step')
        assert hasattr(C2DPO, 'eval_step')

    def test_init_params(self):
        from oxrl.algs.c2dpo import C2DPO
        sig = inspect.signature(C2DPO.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params
        assert 'c2_lambda' in params

    def test_default_lambda(self):
        """Default c2_lambda should be 2e-4 as in the paper."""
        from oxrl.algs.c2dpo import C2DPO
        sig = inspect.signature(C2DPO.__init__)
        assert sig.parameters['c2_lambda'].default == 2e-4


class TestC2DPOLossMath:
    def _make_c2dpo(self, beta=0.1, c2_lambda=2e-4):
        from oxrl.algs.c2dpo import C2DPO
        s = C2DPO.__new__(C2DPO)
        s.beta = beta
        s.c2_lambda = c2_lambda
        return s

    def test_loss_formula(self):
        """Verify C2-DPO loss: L_DPO + lambda * (logr_w + logr_l)^2."""
        beta = 0.2
        c2_lambda = 0.01
        s = self._make_c2dpo(beta=beta, c2_lambda=c2_lambda)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, dpo_loss, constraint, _, _ = s.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # Manual computation
        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        expected_dpo = -F.logsigmoid(beta * (logr_w - logr_l)).mean()
        expected_constraint = ((logr_w + logr_l) ** 2).mean()
        expected_total = expected_dpo + c2_lambda * expected_constraint

        assert abs(loss.item() - expected_total.item()) < 1e-5
        assert abs(dpo_loss.item() - expected_dpo.item()) < 1e-5
        assert abs(constraint.item() - expected_constraint.item()) < 1e-5

    def test_lambda_zero_equals_dpo(self):
        """With lambda=0, C2-DPO should be identical to DPO."""
        beta = 0.1
        s = self._make_c2dpo(beta=beta, c2_lambda=0.0)

        pi_logps_w = torch.tensor([-0.5, -0.3])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0])

        loss, dpo_loss, _, _, _ = s.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # With lambda=0, total should equal DPO
        assert abs(loss.item() - dpo_loss.item()) < 1e-5

        # Also verify the DPO component matches standard DPO
        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        expected_dpo = -F.logsigmoid(beta * (logr_w - logr_l)).mean()
        assert abs(dpo_loss.item() - expected_dpo.item()) < 1e-5

    def test_constraint_is_zero_when_logrs_cancel(self):
        """When logr_w + logr_l = 0, constraint should be zero."""
        s = self._make_c2dpo(beta=0.1, c2_lambda=1.0)

        # Make logr_w = -logr_l -> logr_w + logr_l = 0
        pi_logps_w = torch.tensor([-0.5])
        pi_logps_l = torch.tensor([-1.5])
        ref_logps_w = torch.tensor([-1.0])  # logr_w = 0.5
        ref_logps_l = torch.tensor([-1.0])  # logr_l = -0.5
        # logr_w + logr_l = 0.5 + (-0.5) = 0

        _, _, constraint, _, _ = s.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert abs(constraint.item()) < 1e-5

    def test_constraint_positive(self):
        """Constraint should always be non-negative (it's a squared term)."""
        s = self._make_c2dpo(beta=0.1, c2_lambda=1.0)
        pi_logps_w = torch.tensor([-0.5, -0.3])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0])

        _, _, constraint, _, _ = s.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert constraint.item() >= 0

    def test_larger_lambda_increases_loss(self):
        """Larger lambda should increase total loss (constraint is non-negative)."""
        beta = 0.1

        pi_logps_w = torch.tensor([-0.5, -0.3])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0])

        s_low = self._make_c2dpo(beta=beta, c2_lambda=1e-4)
        s_high = self._make_c2dpo(beta=beta, c2_lambda=1.0)

        loss_low, _, _, _, _ = s_low.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss_high, _, _, _, _ = s_high.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        assert loss_high.item() > loss_low.item()

    def test_returns_five_values(self):
        """compute_loss should return (loss, dpo_loss, constraint, margin, reward_acc)."""
        s = self._make_c2dpo()
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        result = s.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert len(result) == 5

    def test_reward_accuracy(self):
        s = self._make_c2dpo(beta=0.1)
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])
        _, _, _, _, reward_acc = s.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert reward_acc.item() == 1.0

    def test_loss_is_finite(self):
        s = self._make_c2dpo(beta=0.1)
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])
        loss, dpo_loss, constraint, margin, reward_acc = s.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert torch.isfinite(loss)
        assert torch.isfinite(dpo_loss)
        assert torch.isfinite(constraint)
        assert torch.isfinite(margin)


class TestC2DPOGradientFlow:
    def _make_c2dpo(self, beta=0.1, c2_lambda=0.01):
        from oxrl.algs.c2dpo import C2DPO
        s = C2DPO.__new__(C2DPO)
        s.beta = beta
        s.c2_lambda = c2_lambda
        return s

    def test_gradient_flow(self):
        """Verify gradients flow through the C2-DPO loss."""
        s = self._make_c2dpo(beta=0.1)
        pi_logps_w = torch.tensor([-1.0, -0.5], requires_grad=True)
        pi_logps_l = torch.tensor([-2.0, -1.5], requires_grad=True)
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _, _, _ = s.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss.backward()

        assert pi_logps_w.grad is not None
        assert pi_logps_l.grad is not None
        assert torch.all(torch.isfinite(pi_logps_w.grad))
        assert torch.all(torch.isfinite(pi_logps_l.grad))

    def test_constraint_adds_gradient_to_both(self):
        """The constraint term should add gradient contributions to both chosen and rejected."""
        s = self._make_c2dpo(beta=0.1, c2_lambda=1.0)

        pi_logps_w = torch.tensor([-0.5], requires_grad=True)
        pi_logps_l = torch.tensor([-2.0], requires_grad=True)
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-1.0])

        loss, _, _, _, _ = s.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss.backward()

        # Both should have non-zero gradients
        assert abs(pi_logps_w.grad.item()) > 0
        assert abs(pi_logps_l.grad.item()) > 0


class TestC2DPOComputeLogps:
    def _make_c2dpo(self):
        from oxrl.algs.c2dpo import C2DPO
        s = C2DPO.__new__(C2DPO)
        s.beta = 0.1
        s.c2_lambda = 2e-4
        return s

    def test_compute_logps_shape(self):
        s = self._make_c2dpo()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = s.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_logps_mask(self):
        s = self._make_c2dpo()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        mask_full = torch.ones(1, 5)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        logps_full = s.compute_logps(logits, target_ids, mask_full)
        logps_half = s.compute_logps(logits, target_ids, mask_half)
        assert logps_full.item() != logps_half.item()


class TestC2DPOConfig:
    def test_config_accepts_c2dpo(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="c2dpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alg_name == "c2dpo"

    def test_config_default_lambda(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="c2dpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.c2dpo_lambda == 2e-4

    def test_config_custom_lambda(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="c2dpo", total_number_of_epochs=1, micro_batches_per_epoch=10,
                  c2dpo_lambda=0.01)
        assert t.c2dpo_lambda == 0.01
