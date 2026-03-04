"""
Tests for BPO (Balanced Preference Optimization) algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_bpo.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestBPOImport:
    def test_bpo_importable(self):
        from oxrl.algs.bpo import BPO
        assert BPO is not None

    def test_bpo_in_algs_init(self):
        from oxrl.algs import BPO
        assert BPO is not None

    def test_bpo_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"bpo"' in source


class TestBPOInterface:
    def test_has_required_methods(self):
        from oxrl.algs.bpo import BPO
        assert hasattr(BPO, 'compute_logps')
        assert hasattr(BPO, 'forward')
        assert hasattr(BPO, 'compute_loss')
        assert hasattr(BPO, 'train_step')
        assert hasattr(BPO, 'eval_step')

    def test_init_params(self):
        from oxrl.algs.bpo import BPO
        sig = inspect.signature(BPO.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params
        assert 'balance_factor' in params


class TestBPOLossMath:
    def _make_bpo(self, beta=0.1, balance_factor=0.3):
        from oxrl.algs.bpo import BPO
        d = BPO.__new__(BPO)
        d.beta = beta
        d.balance_factor = balance_factor
        return d

    def test_loss_formula_verification(self):
        """Verify: L = -logsigmoid(beta * min(logr_w, balance_factor * (-logr_l)))."""
        beta = 0.2
        balance_factor = 0.3
        d = self._make_bpo(beta=beta, balance_factor=balance_factor)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        balanced_logits = torch.min(logr_w, balance_factor * (-logr_l))
        logits = beta * balanced_logits
        expected = -F.logsigmoid(logits).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_symmetric_balance_factor_1(self):
        """With balance_factor=1.0, both terms weighted equally in min."""
        beta = 0.1
        d = self._make_bpo(beta=beta, balance_factor=1.0)

        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])

        loss, _, _ = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        logr_w = pi_logps_w - ref_logps_w  # 0.1
        logr_l = pi_logps_l - ref_logps_l  # 0.1
        balanced = torch.min(logr_w, -logr_l)  # min(0.1, -0.1) = -0.1
        expected = -F.logsigmoid(beta * balanced).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_differs_from_standard_dpo(self):
        """BPO loss should differ from standard DPO."""
        beta = 0.1
        d = self._make_bpo(beta=beta, balance_factor=0.3)

        pi_logps_w = torch.tensor([-0.5])
        pi_logps_l = torch.tensor([-3.0])
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-2.0])

        loss_bpo, _, _ = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # Standard DPO: -logsigmoid(beta * (logr_w - logr_l))
        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        dpo_loss = -F.logsigmoid(beta * (logr_w - logr_l)).mean()

        assert abs(loss_bpo.item() - dpo_loss.item()) > 1e-3

    def test_chosen_bottleneck(self):
        """When chosen improvement is small, min should select it."""
        beta = 0.1
        d = self._make_bpo(beta=beta, balance_factor=1.0)

        # logr_w = 0.01 (small chosen improvement)
        # -logr_l = 5.0 (large rejected suppression)
        pi_logps_w = torch.tensor([-0.99])
        ref_logps_w = torch.tensor([-1.0])  # logr_w = 0.01
        pi_logps_l = torch.tensor([-7.0])
        ref_logps_l = torch.tensor([-2.0])  # logr_l = -5.0, -logr_l = 5.0

        loss, _, _ = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # min(0.01, 5.0) = 0.01, so loss depends on chosen improvement
        logr_w = pi_logps_w - ref_logps_w
        expected = -F.logsigmoid(beta * logr_w).mean()
        assert abs(loss.item() - expected.item()) < 1e-5

    def test_rejected_bottleneck(self):
        """When rejected suppression is small, min should select it."""
        beta = 0.1
        d = self._make_bpo(beta=beta, balance_factor=1.0)

        # logr_w = 5.0 (large chosen improvement)
        # -logr_l = 0.01 (small rejected suppression)
        pi_logps_w = torch.tensor([4.0])
        ref_logps_w = torch.tensor([-1.0])  # logr_w = 5.0
        pi_logps_l = torch.tensor([-2.01])
        ref_logps_l = torch.tensor([-2.0])  # logr_l = -0.01, -logr_l = 0.01

        loss, _, _ = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # min(5.0, 0.01) = 0.01, so loss depends on rejected suppression
        logr_l = pi_logps_l - ref_logps_l
        expected = -F.logsigmoid(beta * (-logr_l)).mean()
        assert abs(loss.item() - expected.item()) < 1e-5

    def test_balance_factor_scales_rejected(self):
        """Lower balance_factor should reduce the rejected term's influence in min."""
        beta = 0.1
        pi_logps_w = torch.tensor([-0.5])
        pi_logps_l = torch.tensor([-3.0])
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-2.0])

        d_low = self._make_bpo(beta=beta, balance_factor=0.1)
        d_high = self._make_bpo(beta=beta, balance_factor=1.0)

        loss_low, _, _ = d_low.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss_high, _, _ = d_high.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        assert abs(loss_low.item() - loss_high.item()) > 1e-4

    def test_loss_is_finite(self):
        d = self._make_bpo(beta=0.1, balance_factor=0.3)
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])
        loss, margin, reward_acc = d.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert torch.isfinite(loss)
        assert torch.isfinite(margin)

    def test_loss_non_negative(self):
        """BPO loss is -logsigmoid(...) which is always >= 0."""
        d = self._make_bpo(beta=0.5, balance_factor=0.3)
        pi_logps_w = torch.tensor([-0.1, -0.5, -2.0])
        pi_logps_l = torch.tensor([-5.0, -3.0, -0.1])
        ref_logps_w = torch.tensor([-0.5, -1.0, -1.5])
        ref_logps_l = torch.tensor([-1.0, -0.5, -0.2])
        loss, _, _ = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert loss.item() >= 0.0

    def test_reward_accuracy(self):
        d = self._make_bpo(beta=0.1, balance_factor=0.3)
        # chosen has higher logps than rejected -> reward_acc should be 1.0
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])
        _, _, reward_acc = d.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert reward_acc.item() == 1.0

    def test_returns_three_values(self):
        d = self._make_bpo(beta=0.1, balance_factor=0.3)
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        result = d.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert len(result) == 3

    def test_gradient_flows(self):
        """Verify gradients flow through the loss."""
        d = self._make_bpo(beta=0.1, balance_factor=0.3)
        pi_logps_w = torch.tensor([-1.0], requires_grad=True)
        pi_logps_l = torch.tensor([-2.0], requires_grad=True)
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        loss, _, _ = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss.backward()
        assert pi_logps_w.grad is not None or pi_logps_l.grad is not None

    def test_different_balance_factors_produce_different_losses(self):
        """Different balance_factor values should produce different losses."""
        beta = 0.1
        pi_logps_w = torch.tensor([-0.5, -1.0, -0.3])
        pi_logps_l = torch.tensor([-2.0, -3.0, -2.5])
        ref_logps_w = torch.tensor([-1.0, -1.2, -0.8])
        ref_logps_l = torch.tensor([-1.5, -2.0, -1.8])

        losses = []
        for bf in [0.1, 0.3, 0.5, 0.8, 1.0]:
            d = self._make_bpo(beta=beta, balance_factor=bf)
            loss, _, _ = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
            losses.append(loss.item())

        # At least some should differ
        unique = set(round(l, 6) for l in losses)
        assert len(unique) > 1


class TestBPOConfig:
    def test_config_accepts_bpo(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="bpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alg_name == "bpo"
        assert hasattr(t, 'bpo_balance_factor')

    def test_config_default_value(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="bpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.bpo_balance_factor == 0.3

    def test_config_custom_value(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="bpo", total_number_of_epochs=1,
                  micro_batches_per_epoch=10, bpo_balance_factor=0.5)
        assert t.bpo_balance_factor == 0.5


class TestBPOComputeLogps:
    def _make_bpo(self):
        from oxrl.algs.bpo import BPO
        d = BPO.__new__(BPO)
        d.beta = 0.1
        d.balance_factor = 0.3
        return d

    def test_compute_logps_shape(self):
        d = self._make_bpo()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = d.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_logps_mask(self):
        d = self._make_bpo()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        mask_full = torch.ones(1, 5)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        logps_full = d.compute_logps(logits, target_ids, mask_full)
        logps_half = d.compute_logps(logits, target_ids, mask_half)
        assert logps_full.item() != logps_half.item()
