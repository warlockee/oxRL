"""
Tests for DPOP (DPO-Positive) preference optimization algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_dpop.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDPOPImport:
    def test_dpop_importable(self):
        from oxrl.algs.dpop import DPOP
        assert DPOP is not None

    def test_dpop_in_algs_init(self):
        from oxrl.algs import DPOP
        assert DPOP is not None

    def test_dpop_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"dpop"' in source


class TestDPOPInterface:
    def test_has_required_methods(self):
        from oxrl.algs.dpop import DPOP
        assert hasattr(DPOP, 'compute_logps')
        assert hasattr(DPOP, 'forward')
        assert hasattr(DPOP, 'compute_loss')
        assert hasattr(DPOP, 'train_step')
        assert hasattr(DPOP, 'eval_step')

    def test_init_params(self):
        from oxrl.algs.dpop import DPOP
        sig = inspect.signature(DPOP.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params
        assert 'dpop_lambda' in params


class TestDPOPLossMath:
    def _make_dpop(self, beta=0.1, dpop_lambda=5.0):
        from oxrl.algs.dpop import DPOP
        d = DPOP.__new__(DPOP)
        d.beta = beta
        d.dpop_lambda = dpop_lambda
        return d

    def test_loss_formula_verification(self):
        """Verify: L = -logsigmoid(beta*(logr_w - logr_l - lambda*max(0,-logr_w)))."""
        beta = 0.2
        lam = 3.0
        d = self._make_dpop(beta=beta, dpop_lambda=lam)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = d.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        penalty = torch.relu(-logr_w)
        expected = -F.logsigmoid(beta * (logr_w - logr_l - lam * penalty)).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_reduces_to_dpo_when_lambda_zero(self):
        """When lambda=0, DPOP should equal standard DPO loss."""
        beta = 0.1
        d = self._make_dpop(beta=beta, dpop_lambda=0.0)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = d.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        h = beta * ((pi_logps_w - ref_logps_w) - (pi_logps_l - ref_logps_l))
        dpo_loss = -F.logsigmoid(h).mean()

        assert abs(loss.item() - dpo_loss.item()) < 1e-5

    def test_penalty_inactive_when_chosen_above_ref(self):
        """Penalty should be zero when pi(y_w) >= pi_ref(y_w)."""
        d = self._make_dpop(beta=0.1, dpop_lambda=5.0)

        # logr_w > 0 means pi > pi_ref
        pi_logps_w = torch.tensor([0.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.0])  # logr_w = 1.0 > 0
        ref_logps_l = torch.tensor([-2.0])

        loss_dpop, _, _ = d.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # Should equal DPO since penalty is zero
        d_dpo = self._make_dpop(beta=0.1, dpop_lambda=0.0)
        loss_dpo, _, _ = d_dpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        assert abs(loss_dpop.item() - loss_dpo.item()) < 1e-5

    def test_penalty_active_when_chosen_below_ref(self):
        """Penalty should increase loss when pi(y_w) < pi_ref(y_w)."""
        d_dpop = self._make_dpop(beta=0.1, dpop_lambda=5.0)
        d_dpo = self._make_dpop(beta=0.1, dpop_lambda=0.0)

        # logr_w < 0 means pi < pi_ref
        pi_logps_w = torch.tensor([-2.0])
        pi_logps_l = torch.tensor([-3.0])
        ref_logps_w = torch.tensor([-1.0])  # logr_w = -1.0 < 0
        ref_logps_l = torch.tensor([-3.0])

        loss_dpop, _, _ = d_dpop.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss_dpo, _, _ = d_dpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        assert loss_dpop.item() > loss_dpo.item()

    def test_loss_is_finite(self):
        d = self._make_dpop(beta=0.1, dpop_lambda=5.0)
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])
        loss, margin, reward_acc = d.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert torch.isfinite(loss)
        assert torch.isfinite(margin)

    def test_reward_accuracy(self):
        d = self._make_dpop(beta=0.1, dpop_lambda=5.0)
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])
        _, _, reward_acc = d.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert reward_acc.item() == 1.0

    def test_returns_three_values(self):
        d = self._make_dpop(beta=0.1, dpop_lambda=5.0)
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        result = d.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert len(result) == 3


class TestDPOPConfig:
    def test_config_accepts_dpop(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="dpop", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alg_name == "dpop"
        assert hasattr(t, 'dpop_lambda')

    def test_config_default_value(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="dpop", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.dpop_lambda == 5.0


class TestDPOPComputeLogps:
    def _make_dpop(self):
        from oxrl.algs.dpop import DPOP
        d = DPOP.__new__(DPOP)
        d.beta = 0.1
        d.dpop_lambda = 5.0
        return d

    def test_compute_logps_shape(self):
        d = self._make_dpop()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = d.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_logps_mask(self):
        d = self._make_dpop()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        mask_full = torch.ones(1, 5)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        logps_full = d.compute_logps(logits, target_ids, mask_full)
        logps_half = d.compute_logps(logits, target_ids, mask_half)
        assert logps_full.item() != logps_half.item()
