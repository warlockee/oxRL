"""
Tests for Dr. DPO (Distributionally Robust DPO) algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_drdpo.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDrDPOImport:
    def test_drdpo_importable(self):
        from oxrl.algs.drdpo import DrDPO
        assert DrDPO is not None

    def test_drdpo_in_algs_init(self):
        from oxrl.algs import DrDPO
        assert DrDPO is not None

    def test_drdpo_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"drdpo"' in source


class TestDrDPOInterface:
    def test_has_required_methods(self):
        from oxrl.algs.drdpo import DrDPO
        assert hasattr(DrDPO, 'compute_logps')
        assert hasattr(DrDPO, 'forward')
        assert hasattr(DrDPO, 'compute_loss')
        assert hasattr(DrDPO, 'train_step')
        assert hasattr(DrDPO, 'eval_step')

    def test_init_params(self):
        from oxrl.algs.drdpo import DrDPO
        sig = inspect.signature(DrDPO.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params
        assert 'beta_prime' in params

    def test_default_beta_prime(self):
        from oxrl.algs.drdpo import DrDPO
        sig = inspect.signature(DrDPO.__init__)
        assert sig.parameters['beta_prime'].default == 1.0


class TestDrDPOLossMath:
    def _make_drdpo(self, beta=0.1, beta_prime=1.0):
        from oxrl.algs.drdpo import DrDPO
        d = DrDPO.__new__(DrDPO)
        d.beta = beta
        d.beta_prime = beta_prime
        return d

    def test_loss_formula(self):
        """Verify Dr. DPO loss: -beta' * log(mean(exp(-per_sample_loss / beta')))."""
        beta = 0.2
        beta_prime = 0.5
        d = self._make_drdpo(beta=beta, beta_prime=beta_prime)

        pi_logps_w = torch.tensor([-1.0, -0.5, -0.3])
        pi_logps_l = torch.tensor([-2.0, -1.5, -1.0])
        ref_logps_w = torch.tensor([-1.1, -0.6, -0.4])
        ref_logps_l = torch.tensor([-2.1, -1.6, -1.1])

        loss, _, _ = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # Manual computation
        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        logits = beta * (logr_w - logr_l)
        per_sample = -F.logsigmoid(logits)
        expected = -beta_prime * torch.log(torch.mean(torch.exp(-per_sample / beta_prime)))

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_reduces_to_dpo_at_large_beta_prime(self):
        """With very large beta_prime, Dr. DPO should approximate standard DPO."""
        beta = 0.1
        d_large = self._make_drdpo(beta=beta, beta_prime=100.0)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss_drdpo, _, _ = d_large.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # Standard DPO loss
        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        logits = beta * (logr_w - logr_l)
        dpo_loss = -F.logsigmoid(logits).mean()

        assert abs(loss_drdpo.item() - dpo_loss.item()) < 0.01

    def test_single_sample_equals_dpo(self):
        """With a single sample, Dr. DPO should equal standard DPO regardless of beta_prime."""
        beta = 0.1
        d = self._make_drdpo(beta=beta, beta_prime=0.5)

        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])

        loss, _, _ = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # With single sample, mean(exp(-L/bp)) = exp(-L/bp), so
        # -bp * log(exp(-L/bp)) = -bp * (-L/bp) = L
        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        logits = beta * (logr_w - logr_l)
        dpo_loss = -F.logsigmoid(logits).mean()

        assert abs(loss.item() - dpo_loss.item()) < 1e-5

    def test_small_beta_prime_focuses_on_easy_samples(self):
        """Smaller beta_prime focuses on easier samples (lower per-sample loss)."""
        beta = 0.1
        d_small = self._make_drdpo(beta=beta, beta_prime=0.01)
        d_large = self._make_drdpo(beta=beta, beta_prime=10.0)

        # One easy sample (large margin) and one hard sample (small margin)
        pi_logps_w = torch.tensor([-0.5, -1.5])
        pi_logps_l = torch.tensor([-3.0, -1.6])
        ref_logps_w = torch.tensor([-1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0])

        loss_small, _, _ = d_small.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss_large, _, _ = d_large.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # With small beta_prime, Dr. DPO focuses on the minimum loss (easy sample)
        # so the overall loss should be lower
        assert loss_small.item() < loss_large.item()

    def test_loss_is_finite(self):
        d = self._make_drdpo(beta=0.1, beta_prime=1.0)
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])
        loss, margin, reward_acc = d.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert torch.isfinite(loss)
        assert torch.isfinite(margin)

    def test_reward_accuracy(self):
        d = self._make_drdpo(beta=0.1, beta_prime=1.0)
        # All samples have positive margin (chosen > rejected)
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])
        _, _, reward_acc = d.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert reward_acc.item() == 1.0

    def test_returns_three_values(self):
        d = self._make_drdpo(beta=0.1, beta_prime=1.0)
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        result = d.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert len(result) == 3

    def test_loss_positive(self):
        """Dr. DPO loss should be positive (it's a negative log-likelihood)."""
        d = self._make_drdpo(beta=0.1, beta_prime=1.0)
        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])
        loss, _, _ = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert loss.item() > 0

    def test_beta_prime_monotonicity(self):
        """As beta_prime decreases, Dr. DPO loss decreases (focuses on min loss)."""
        beta = 0.1
        # Use samples with varied losses
        pi_logps_w = torch.tensor([-0.5, -1.5, -0.3])
        pi_logps_l = torch.tensor([-3.0, -1.6, -2.5])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])

        losses = []
        for bp in [0.01, 0.1, 1.0, 10.0, 100.0]:
            d = self._make_drdpo(beta=beta, beta_prime=bp)
            loss, _, _ = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
            losses.append(loss.item())

        # Losses should generally increase as beta_prime increases (approaching mean)
        # because log-sum-exp lower bound is min and upper bound is mean
        assert losses[0] < losses[-1]


class TestDrDPOGradientFlow:
    def _make_drdpo(self, beta=0.1, beta_prime=1.0):
        from oxrl.algs.drdpo import DrDPO
        d = DrDPO.__new__(DrDPO)
        d.beta = beta
        d.beta_prime = beta_prime
        return d

    def test_gradient_flow(self):
        """Verify gradients flow through the Dr. DPO loss."""
        d = self._make_drdpo(beta=0.1, beta_prime=1.0)
        pi_logps_w = torch.tensor([-1.0, -0.5], requires_grad=True)
        pi_logps_l = torch.tensor([-2.0, -1.5], requires_grad=True)
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss.backward()

        assert pi_logps_w.grad is not None
        assert pi_logps_l.grad is not None
        assert torch.all(torch.isfinite(pi_logps_w.grad))
        assert torch.all(torch.isfinite(pi_logps_l.grad))


class TestDrDPOComputeLogps:
    def _make_drdpo(self):
        from oxrl.algs.drdpo import DrDPO
        d = DrDPO.__new__(DrDPO)
        d.beta = 0.1
        d.beta_prime = 1.0
        return d

    def test_compute_logps_shape(self):
        d = self._make_drdpo()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = d.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_logps_mask(self):
        d = self._make_drdpo()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        mask_full = torch.ones(1, 5)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        logps_full = d.compute_logps(logits, target_ids, mask_full)
        logps_half = d.compute_logps(logits, target_ids, mask_half)
        assert logps_full.item() != logps_half.item()


class TestDrDPOConfig:
    def test_config_accepts_drdpo(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="drdpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alg_name == "drdpo"

    def test_config_has_beta_prime(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="drdpo", total_number_of_epochs=1, micro_batches_per_epoch=10,
                  drdpo_beta_prime=0.5)
        assert t.drdpo_beta_prime == 0.5

    def test_config_default_beta_prime(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="drdpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.drdpo_beta_prime == 1.0
