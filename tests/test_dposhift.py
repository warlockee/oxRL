"""
Tests for DPO-Shift (Shifting the Distribution of DPO) algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_dposhift.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDPOShiftImport:
    def test_dposhift_importable(self):
        from oxrl.algs.dposhift import DPOShift
        assert DPOShift is not None

    def test_dposhift_in_algs_init(self):
        from oxrl.algs import DPOShift
        assert DPOShift is not None

    def test_dposhift_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"dposhift"' in source


class TestDPOShiftInterface:
    def test_has_required_methods(self):
        from oxrl.algs.dposhift import DPOShift
        assert hasattr(DPOShift, 'compute_logps')
        assert hasattr(DPOShift, 'forward')
        assert hasattr(DPOShift, 'compute_loss')
        assert hasattr(DPOShift, 'train_step')
        assert hasattr(DPOShift, 'eval_step')

    def test_init_params(self):
        from oxrl.algs.dposhift import DPOShift
        sig = inspect.signature(DPOShift.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params
        assert 'shift_lambda' in params


class TestDPOShiftLossMath:
    def _make_dposhift(self, beta=0.1, shift_lambda=0.5):
        from oxrl.algs.dposhift import DPOShift
        d = DPOShift.__new__(DPOShift)
        d.beta = beta
        d.shift_lambda = shift_lambda
        return d

    def test_loss_formula_verification(self):
        """Verify: L = -logsigmoid(beta*logr_w - shift_lambda*beta*logr_l)."""
        beta = 0.2
        shift_lambda = 0.5
        d = self._make_dposhift(beta=beta, shift_lambda=shift_lambda)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        logits = beta * logr_w - shift_lambda * beta * logr_l
        expected = -F.logsigmoid(logits).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_reduces_to_dpo_when_lambda_one(self):
        """When shift_lambda=1, DPO-Shift should equal standard DPO."""
        beta = 0.1
        d = self._make_dposhift(beta=beta, shift_lambda=1.0)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # Standard DPO: -logsigmoid(beta * (logr_w - logr_l))
        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        dpo_loss = -F.logsigmoid(beta * (logr_w - logr_l)).mean()

        assert abs(loss.item() - dpo_loss.item()) < 1e-5

    def test_lambda_less_than_one_differs_from_dpo(self):
        """With shift_lambda<1, loss should differ from standard DPO."""
        beta = 0.1
        d_dpo = self._make_dposhift(beta=beta, shift_lambda=1.0)
        d_shift = self._make_dposhift(beta=beta, shift_lambda=0.5)

        pi_logps_w = torch.tensor([-0.5])
        pi_logps_l = torch.tensor([-3.0])
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-2.0])

        loss_dpo, _, _ = d_dpo.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss_shift, _, _ = d_shift.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        assert abs(loss_dpo.item() - loss_shift.item()) > 1e-3

    def test_lower_lambda_reduces_rejected_influence(self):
        """Lower shift_lambda should make loss more dependent on chosen and
        less dependent on rejected. With lambda=0, only chosen matters."""
        beta = 0.1
        d_zero = self._make_dposhift(beta=beta, shift_lambda=0.0)

        pi_logps_w = torch.tensor([-1.0])
        ref_logps_w = torch.tensor([-1.1])

        # Two different rejected log-probs -- with lambda=0, loss should be same
        pi_logps_l_a = torch.tensor([-2.0])
        ref_logps_l_a = torch.tensor([-2.1])
        pi_logps_l_b = torch.tensor([-5.0])
        ref_logps_l_b = torch.tensor([-0.5])

        loss_a, _, _ = d_zero.compute_loss(pi_logps_w, pi_logps_l_a, ref_logps_w, ref_logps_l_a)
        loss_b, _, _ = d_zero.compute_loss(pi_logps_w, pi_logps_l_b, ref_logps_w, ref_logps_l_b)

        # Both should be equal since rejected term is zeroed out
        assert abs(loss_a.item() - loss_b.item()) < 1e-5

    def test_different_lambda_values_monotonicity(self):
        """Test several lambda values produce different losses."""
        beta = 0.1
        pi_logps_w = torch.tensor([-0.5])
        pi_logps_l = torch.tensor([-3.0])
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-2.0])

        losses = []
        for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
            d = self._make_dposhift(beta=beta, shift_lambda=lam)
            loss, _, _ = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
            losses.append(loss.item())

        # All should be different
        for i in range(len(losses)):
            for j in range(i + 1, len(losses)):
                assert abs(losses[i] - losses[j]) > 1e-6

    def test_loss_is_finite(self):
        d = self._make_dposhift(beta=0.1, shift_lambda=0.5)
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])
        loss, margin, reward_acc = d.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert torch.isfinite(loss)
        assert torch.isfinite(margin)

    def test_reward_accuracy(self):
        d = self._make_dposhift(beta=0.1, shift_lambda=0.5)
        # chosen has higher logps than rejected -> reward_acc should be 1.0
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])
        _, _, reward_acc = d.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert reward_acc.item() == 1.0

    def test_returns_three_values(self):
        d = self._make_dposhift(beta=0.1, shift_lambda=0.5)
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        result = d.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert len(result) == 3

    def test_loss_non_negative(self):
        """DPO-Shift loss is -logsigmoid(...) which is always >= 0."""
        d = self._make_dposhift(beta=0.5, shift_lambda=0.3)
        pi_logps_w = torch.tensor([-0.1, -0.5, -2.0])
        pi_logps_l = torch.tensor([-5.0, -3.0, -0.1])
        ref_logps_w = torch.tensor([-0.5, -1.0, -1.5])
        ref_logps_l = torch.tensor([-1.0, -0.5, -0.2])
        loss, _, _ = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert loss.item() >= 0.0

    def test_gradient_flows(self):
        """Verify gradients flow through the loss."""
        d = self._make_dposhift(beta=0.1, shift_lambda=0.5)
        pi_logps_w = torch.tensor([-1.0], requires_grad=True)
        pi_logps_l = torch.tensor([-2.0], requires_grad=True)
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        loss, _, _ = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss.backward()
        assert pi_logps_w.grad is not None
        assert pi_logps_l.grad is not None


class TestDPOShiftConfig:
    def test_config_accepts_dposhift(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="dposhift", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alg_name == "dposhift"
        assert hasattr(t, 'dposhift_lambda')

    def test_config_default_value(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="dposhift", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.dposhift_lambda == 0.5

    def test_config_custom_value(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="dposhift", total_number_of_epochs=1,
                  micro_batches_per_epoch=10, dposhift_lambda=0.3)
        assert t.dposhift_lambda == 0.3


class TestDPOShiftComputeLogps:
    def _make_dposhift(self):
        from oxrl.algs.dposhift import DPOShift
        d = DPOShift.__new__(DPOShift)
        d.beta = 0.1
        d.shift_lambda = 0.5
        return d

    def test_compute_logps_shape(self):
        d = self._make_dposhift()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = d.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_logps_mask(self):
        d = self._make_dposhift()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        mask_full = torch.ones(1, 5)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        logps_full = d.compute_logps(logits, target_ids, mask_full)
        logps_half = d.compute_logps(logits, target_ids, mask_half)
        assert logps_full.item() != logps_half.item()
