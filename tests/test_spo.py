"""
Tests for SPO (Self Preference Optimization) algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_spo.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestSPOImport:
    def test_spo_importable(self):
        from oxrl.algs.spo import SPO
        assert SPO is not None

    def test_spo_in_algs_init(self):
        from oxrl.algs import SPO
        assert SPO is not None

    def test_spo_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"spo"' in source


class TestSPOInterface:
    def test_has_required_methods(self):
        from oxrl.algs.spo import SPO
        assert hasattr(SPO, 'compute_logps')
        assert hasattr(SPO, 'forward')
        assert hasattr(SPO, 'compute_loss')
        assert hasattr(SPO, 'train_step')
        assert hasattr(SPO, 'eval_step')

    def test_init_params(self):
        from oxrl.algs.spo import SPO
        sig = inspect.signature(SPO.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params

    def test_no_extra_hyperparameters(self):
        """SPO should use the same hyperparameters as DPO (no new ones)."""
        from oxrl.algs.spo import SPO
        from oxrl.algs.dpo import DPO
        spo_params = set(inspect.signature(SPO.__init__).parameters.keys())
        dpo_params = set(inspect.signature(DPO.__init__).parameters.keys())
        assert spo_params == dpo_params


class TestSPOLossMath:
    def _make_spo(self, beta=0.1):
        from oxrl.algs.spo import SPO
        s = SPO.__new__(SPO)
        s.beta = beta
        return s

    def test_loss_formula(self):
        """Verify SPO loss: -SiLU(beta * (logr_w - logr_l)).mean()."""
        beta = 0.2
        s = self._make_spo(beta=beta)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = s.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # Manual computation
        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        logits = beta * (logr_w - logr_l)
        expected = -F.silu(logits).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_differs_from_dpo(self):
        """SPO should give a different loss than standard DPO for non-zero logits."""
        beta = 0.1
        s = self._make_spo(beta=beta)

        pi_logps_w = torch.tensor([-0.5, -0.3])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0])

        spo_loss, _, _ = s.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # Standard DPO loss
        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        dpo_logits = beta * (logr_w - logr_l)
        dpo_loss = -F.logsigmoid(dpo_logits).mean()

        assert abs(spo_loss.item() - dpo_loss.item()) > 1e-4

    def test_silu_ge_logsigmoid(self):
        """SiLU(x) >= logsigmoid(x) for all x, so -SiLU(x) <= -logsigmoid(x)."""
        # Verify the core property: SiLU(x) >= logsigmoid(x) for all x
        x = torch.linspace(-10, 10, 1000)
        silu_vals = F.silu(x)
        logsig_vals = F.logsigmoid(x)
        # SiLU should be >= logsigmoid elementwise
        assert (silu_vals >= logsig_vals - 1e-5).all()

    def test_silu_has_bounded_minimum(self):
        """The SiLU function should have a finite minimum, unlike logsigmoid."""
        # SiLU(x) has min ~ -0.278 at x ~ -1.28
        x = torch.linspace(-10, 10, 1000)
        silu_values = F.silu(x)
        min_silu = silu_values.min().item()

        # Verify the minimum is finite and around -0.278
        assert min_silu > -0.3
        assert min_silu < -0.2

    def test_loss_at_zero_logits(self):
        """When logits are zero, SiLU(0) = 0, so loss should be 0."""
        s = self._make_spo(beta=0.1)
        # Equal log-ratios -> logits = 0
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-2.0])

        loss, _, _ = s.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert abs(loss.item()) < 1e-5

    def test_loss_is_finite(self):
        s = self._make_spo(beta=0.1)
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])
        loss, margin, reward_acc = s.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert torch.isfinite(loss)
        assert torch.isfinite(margin)

    def test_reward_accuracy(self):
        s = self._make_spo(beta=0.1)
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])
        _, _, reward_acc = s.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert reward_acc.item() == 1.0

    def test_returns_three_values(self):
        s = self._make_spo(beta=0.1)
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        result = s.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert len(result) == 3


class TestSPOGradientFlow:
    def _make_spo(self, beta=0.1):
        from oxrl.algs.spo import SPO
        s = SPO.__new__(SPO)
        s.beta = beta
        return s

    def test_gradient_flow(self):
        """Verify gradients flow through the SPO loss."""
        s = self._make_spo(beta=0.1)
        pi_logps_w = torch.tensor([-1.0, -0.5], requires_grad=True)
        pi_logps_l = torch.tensor([-2.0, -1.5], requires_grad=True)
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = s.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss.backward()

        assert pi_logps_w.grad is not None
        assert pi_logps_l.grad is not None
        assert torch.all(torch.isfinite(pi_logps_w.grad))
        assert torch.all(torch.isfinite(pi_logps_l.grad))


class TestSPOComputeLogps:
    def _make_spo(self):
        from oxrl.algs.spo import SPO
        s = SPO.__new__(SPO)
        s.beta = 0.1
        return s

    def test_compute_logps_shape(self):
        s = self._make_spo()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = s.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_logps_mask(self):
        s = self._make_spo()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        mask_full = torch.ones(1, 5)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        logps_full = s.compute_logps(logits, target_ids, mask_full)
        logps_half = s.compute_logps(logits, target_ids, mask_half)
        assert logps_full.item() != logps_half.item()


class TestSPOConfig:
    def test_config_accepts_spo(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="spo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alg_name == "spo"
