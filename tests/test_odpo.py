"""
Tests for ODPO (DPO with Offset) preference optimization algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_odpo.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestODPOImport:
    def test_odpo_importable(self):
        from oxrl.algs.odpo import ODPO
        assert ODPO is not None

    def test_odpo_in_algs_init(self):
        from oxrl.algs import ODPO
        assert ODPO is not None

    def test_odpo_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"odpo"' in source


class TestODPOInterface:
    def test_has_required_methods(self):
        from oxrl.algs.odpo import ODPO
        assert hasattr(ODPO, 'compute_logps')
        assert hasattr(ODPO, 'forward')
        assert hasattr(ODPO, 'compute_loss')
        assert hasattr(ODPO, 'train_step')
        assert hasattr(ODPO, 'eval_step')

    def test_init_params(self):
        from oxrl.algs.odpo import ODPO
        sig = inspect.signature(ODPO.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params
        assert 'odpo_delta' in params


class TestODPOLossMath:
    def _make_odpo(self, beta=0.1, odpo_delta=1.0):
        from oxrl.algs.odpo import ODPO
        o = ODPO.__new__(ODPO)
        o.beta = beta
        o.odpo_delta = odpo_delta
        return o

    def test_loss_formula_verification(self):
        """Verify: L = -logsigmoid(beta*(logr_w - logr_l) - delta)."""
        beta = 0.2
        delta = 0.5
        o = self._make_odpo(beta=beta, odpo_delta=delta)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = o.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        expected = -F.logsigmoid(beta * (logr_w - logr_l) - delta).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_reduces_to_dpo_when_delta_zero(self):
        """When delta=0, ODPO should equal standard DPO loss."""
        beta = 0.1
        o = self._make_odpo(beta=beta, odpo_delta=0.0)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = o.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        h = beta * ((pi_logps_w - ref_logps_w) - (pi_logps_l - ref_logps_l))
        dpo_loss = -F.logsigmoid(h).mean()

        assert abs(loss.item() - dpo_loss.item()) < 1e-5

    def test_delta_increases_loss(self):
        """Positive delta should increase loss (harder to satisfy)."""
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])

        o_no_offset = self._make_odpo(beta=0.1, odpo_delta=0.0)
        o_with_offset = self._make_odpo(beta=0.1, odpo_delta=1.0)

        loss_no, _, _ = o_no_offset.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss_with, _, _ = o_with_offset.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        assert loss_with.item() > loss_no.item()

    def test_loss_always_positive(self):
        """ODPO loss should always be positive."""
        o = self._make_odpo(beta=0.1, odpo_delta=1.0)
        for _ in range(10):
            pi_logps_w = torch.randn(4)
            pi_logps_l = torch.randn(4)
            ref_logps_w = torch.randn(4)
            ref_logps_l = torch.randn(4)
            loss, _, _ = o.compute_loss(
                pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
            assert loss.item() > 0

    def test_loss_is_finite(self):
        o = self._make_odpo(beta=0.1, odpo_delta=1.0)
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])
        loss, margin, reward_acc = o.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert torch.isfinite(loss)
        assert torch.isfinite(margin)

    def test_reward_accuracy(self):
        o = self._make_odpo(beta=0.1, odpo_delta=1.0)
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])
        _, _, reward_acc = o.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert reward_acc.item() == 1.0

    def test_returns_three_values(self):
        o = self._make_odpo(beta=0.1, odpo_delta=1.0)
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        result = o.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert len(result) == 3


class TestODPOConfig:
    def test_config_accepts_odpo(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="odpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alg_name == "odpo"
        assert hasattr(t, 'odpo_delta')

    def test_config_default_value(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="odpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.odpo_delta == 1.0


class TestODPOComputeLogps:
    def _make_odpo(self):
        from oxrl.algs.odpo import ODPO
        o = ODPO.__new__(ODPO)
        o.beta = 0.1
        o.odpo_delta = 1.0
        return o

    def test_compute_logps_shape(self):
        o = self._make_odpo()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = o.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_logps_mask(self):
        o = self._make_odpo()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        mask_full = torch.ones(1, 5)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        logps_full = o.compute_logps(logits, target_ids, mask_full)
        logps_half = o.compute_logps(logits, target_ids, mask_half)
        assert logps_full.item() != logps_half.item()
