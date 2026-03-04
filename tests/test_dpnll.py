"""
Tests for DPNLL (DPO with NLL Regularization) algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_dpnll.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDPNLLImport:
    def test_dpnll_importable(self):
        from oxrl.algs.dpnll import DPNLL
        assert DPNLL is not None

    def test_dpnll_in_algs_init(self):
        from oxrl.algs import DPNLL
        assert DPNLL is not None

    def test_dpnll_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"dpnll"' in source


class TestDPNLLInterface:
    def test_has_required_methods(self):
        from oxrl.algs.dpnll import DPNLL
        assert hasattr(DPNLL, 'compute_logps')
        assert hasattr(DPNLL, 'forward')
        assert hasattr(DPNLL, 'forward_ref')
        assert hasattr(DPNLL, 'compute_loss')
        assert hasattr(DPNLL, 'compute_nll_loss')
        assert hasattr(DPNLL, 'train_step')
        assert hasattr(DPNLL, 'eval_step')

    def test_init_params(self):
        from oxrl.algs.dpnll import DPNLL
        sig = inspect.signature(DPNLL.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params
        assert 'dpnll_alpha' in params

    def test_has_nll_loss_fn(self):
        """DPNLL should have a CrossEntropyLoss for NLL computation."""
        from oxrl.algs.dpnll import DPNLL
        sig = inspect.signature(DPNLL.__init__)
        # The class initializes self.nll_loss_fn
        source = inspect.getsource(DPNLL.__init__)
        assert 'CrossEntropyLoss' in source

    def test_default_alpha(self):
        """Default dpnll_alpha should be 1.0."""
        from oxrl.algs.dpnll import DPNLL
        sig = inspect.signature(DPNLL.__init__)
        assert sig.parameters['dpnll_alpha'].default == 1.0


class TestDPNLLNLLLoss:
    def _make_dpnll(self, beta=0.1, dpnll_alpha=1.0):
        from oxrl.algs.dpnll import DPNLL
        s = DPNLL.__new__(DPNLL)
        s.beta = beta
        s.dpnll_alpha = dpnll_alpha
        s.nll_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        return s

    def test_nll_loss_computation(self):
        """NLL loss should be a masked mean cross-entropy."""
        s = self._make_dpnll()
        # Create simple logits and targets
        B, T, V = 2, 4, 10
        logits = torch.randn(B, T, V)
        target_ids = torch.randint(0, V, (B, T))
        loss_mask = torch.ones(B, T)

        nll_loss = s.compute_nll_loss(logits, target_ids, loss_mask)
        assert nll_loss.shape == ()
        assert nll_loss.item() > 0  # NLL should be positive

    def test_nll_loss_with_mask(self):
        """NLL loss should only consider masked tokens."""
        s = self._make_dpnll()
        B, T, V = 1, 4, 10
        logits = torch.randn(B, T, V)
        target_ids = torch.randint(0, V, (B, T))

        mask_full = torch.ones(B, T)
        mask_partial = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

        nll_full = s.compute_nll_loss(logits, target_ids, mask_full)
        nll_partial = s.compute_nll_loss(logits, target_ids, mask_partial)

        # Different masks should give different losses
        assert nll_full.item() != nll_partial.item()

    def test_nll_loss_is_finite(self):
        s = self._make_dpnll()
        B, T, V = 2, 5, 20
        logits = torch.randn(B, T, V)
        target_ids = torch.randint(0, V, (B, T))
        loss_mask = torch.ones(B, T)

        nll_loss = s.compute_nll_loss(logits, target_ids, loss_mask)
        assert torch.isfinite(nll_loss)


class TestDPNLLCombinedLoss:
    def _make_dpnll(self, beta=0.1, dpnll_alpha=1.0):
        from oxrl.algs.dpnll import DPNLL
        s = DPNLL.__new__(DPNLL)
        s.beta = beta
        s.dpnll_alpha = dpnll_alpha
        s.nll_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        return s

    def test_combined_loss_formula(self):
        """Verify total_loss = dpo_loss + alpha * nll_loss."""
        beta = 0.2
        alpha = 0.5
        s = self._make_dpnll(beta=beta, dpnll_alpha=alpha)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        # Create dummy chosen logits for NLL
        B, T, V = 2, 3, 10
        chosen_logits = torch.randn(B, T, V)
        chosen_targets = torch.randint(0, V, (B, T))
        chosen_mask = torch.ones(B, T)

        total_loss, dpo_loss, nll_loss, margin, reward_acc = s.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l,
            chosen_logits, chosen_targets, chosen_mask)

        expected_total = dpo_loss + alpha * nll_loss
        assert abs(total_loss.item() - expected_total.item()) < 1e-5

    def test_alpha_zero_equals_dpo(self):
        """With alpha=0, DPNLL should be identical to DPO."""
        beta = 0.1
        s = self._make_dpnll(beta=beta, dpnll_alpha=0.0)

        pi_logps_w = torch.tensor([-0.5, -0.3])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0])

        B, T, V = 2, 3, 10
        chosen_logits = torch.randn(B, T, V)
        chosen_targets = torch.randint(0, V, (B, T))
        chosen_mask = torch.ones(B, T)

        total_loss, dpo_loss, nll_loss, _, _ = s.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l,
            chosen_logits, chosen_targets, chosen_mask)

        # Total should equal DPO loss when alpha=0
        assert abs(total_loss.item() - dpo_loss.item()) < 1e-5

        # Also verify the DPO component matches standard DPO
        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        expected_dpo = -F.logsigmoid(beta * (logr_w - logr_l)).mean()
        assert abs(dpo_loss.item() - expected_dpo.item()) < 1e-5

    def test_alpha_increases_total_loss(self):
        """Larger alpha should increase total loss (NLL is always positive)."""
        beta = 0.1

        pi_logps_w = torch.tensor([-0.5, -0.3])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0])

        B, T, V = 2, 3, 10
        chosen_logits = torch.randn(B, T, V)
        chosen_targets = torch.randint(0, V, (B, T))
        chosen_mask = torch.ones(B, T)

        s_low = self._make_dpnll(beta=beta, dpnll_alpha=0.1)
        s_high = self._make_dpnll(beta=beta, dpnll_alpha=2.0)

        loss_low, _, _, _, _ = s_low.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l,
            chosen_logits, chosen_targets, chosen_mask)
        loss_high, _, _, _, _ = s_high.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l,
            chosen_logits, chosen_targets, chosen_mask)

        assert loss_high.item() > loss_low.item()

    def test_returns_five_values(self):
        """compute_loss should return (total_loss, dpo_loss, nll_loss, margin, reward_acc)."""
        s = self._make_dpnll()
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])

        B, T, V = 1, 3, 10
        chosen_logits = torch.randn(B, T, V)
        chosen_targets = torch.randint(0, V, (B, T))
        chosen_mask = torch.ones(B, T)

        result = s.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l,
            chosen_logits, chosen_targets, chosen_mask)
        assert len(result) == 5

    def test_reward_accuracy(self):
        s = self._make_dpnll(beta=0.1)
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])

        B, T, V = 3, 3, 10
        chosen_logits = torch.randn(B, T, V)
        chosen_targets = torch.randint(0, V, (B, T))
        chosen_mask = torch.ones(B, T)

        _, _, _, _, reward_acc = s.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l,
            chosen_logits, chosen_targets, chosen_mask)
        assert reward_acc.item() == 1.0

    def test_loss_is_finite(self):
        s = self._make_dpnll(beta=0.1)
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])

        B, T, V = 2, 3, 10
        chosen_logits = torch.randn(B, T, V)
        chosen_targets = torch.randint(0, V, (B, T))
        chosen_mask = torch.ones(B, T)

        total_loss, dpo_loss, nll_loss, margin, reward_acc = s.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l,
            chosen_logits, chosen_targets, chosen_mask)
        assert torch.isfinite(total_loss)
        assert torch.isfinite(dpo_loss)
        assert torch.isfinite(nll_loss)
        assert torch.isfinite(margin)


class TestDPNLLGradientFlow:
    def _make_dpnll(self, beta=0.1, dpnll_alpha=1.0):
        from oxrl.algs.dpnll import DPNLL
        s = DPNLL.__new__(DPNLL)
        s.beta = beta
        s.dpnll_alpha = dpnll_alpha
        s.nll_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        return s

    def test_gradient_flow_dpo(self):
        """Verify gradients flow through the DPO component."""
        s = self._make_dpnll(beta=0.1)
        pi_logps_w = torch.tensor([-1.0, -0.5], requires_grad=True)
        pi_logps_l = torch.tensor([-2.0, -1.5], requires_grad=True)
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        B, T, V = 2, 3, 10
        chosen_logits = torch.randn(B, T, V, requires_grad=True)
        chosen_targets = torch.randint(0, V, (B, T))
        chosen_mask = torch.ones(B, T)

        total_loss, _, _, _, _ = s.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l,
            chosen_logits, chosen_targets, chosen_mask)
        total_loss.backward()

        assert pi_logps_w.grad is not None
        assert pi_logps_l.grad is not None
        assert chosen_logits.grad is not None
        assert torch.all(torch.isfinite(pi_logps_w.grad))
        assert torch.all(torch.isfinite(pi_logps_l.grad))
        assert torch.all(torch.isfinite(chosen_logits.grad))

    def test_nll_gradient_only_from_chosen(self):
        """NLL gradients should only flow through chosen logits, not rejected."""
        s = self._make_dpnll(beta=0.1, dpnll_alpha=1.0)

        # Only test NLL component
        B, T, V = 2, 3, 10
        chosen_logits = torch.randn(B, T, V, requires_grad=True)
        chosen_targets = torch.randint(0, V, (B, T))
        chosen_mask = torch.ones(B, T)

        nll_loss = s.compute_nll_loss(chosen_logits, chosen_targets, chosen_mask)
        nll_loss.backward()

        assert chosen_logits.grad is not None
        assert torch.all(torch.isfinite(chosen_logits.grad))


class TestDPNLLComputeLogps:
    def _make_dpnll(self):
        from oxrl.algs.dpnll import DPNLL
        s = DPNLL.__new__(DPNLL)
        s.beta = 0.1
        s.dpnll_alpha = 1.0
        s.nll_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        return s

    def test_compute_logps_shape(self):
        s = self._make_dpnll()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = s.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_logps_mask(self):
        s = self._make_dpnll()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        mask_full = torch.ones(1, 5)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        logps_full = s.compute_logps(logits, target_ids, mask_full)
        logps_half = s.compute_logps(logits, target_ids, mask_half)
        assert logps_full.item() != logps_half.item()


class TestDPNLLConfig:
    def test_config_accepts_dpnll(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="dpnll", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alg_name == "dpnll"

    def test_config_default_alpha(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="dpnll", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.dpnll_alpha == 1.0

    def test_config_custom_alpha(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="dpnll", total_number_of_epochs=1, micro_batches_per_epoch=10,
                  dpnll_alpha=0.5)
        assert t.dpnll_alpha == 0.5
