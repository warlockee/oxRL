"""
Tests for BCO (Binary Classifier Optimization) pairwise algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_bco.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestBCOImport:
    def test_bco_importable(self):
        from oxrl.algs.bco import BCO
        assert BCO is not None

    def test_bco_in_algs_init(self):
        from oxrl.algs import BCO
        assert BCO is not None

    def test_bco_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"bco"' in source


class TestBCOInterface:
    def test_has_required_methods(self):
        from oxrl.algs.bco import BCO
        assert hasattr(BCO, 'compute_logps')
        assert hasattr(BCO, 'forward')
        assert hasattr(BCO, 'compute_loss')
        assert hasattr(BCO, 'train_step')
        assert hasattr(BCO, 'eval_step')

    def test_init_params(self):
        from oxrl.algs.bco import BCO
        sig = inspect.signature(BCO.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params


class TestBCOLossMath:
    def _make_bco(self, beta=0.1):
        from oxrl.algs.bco import BCO
        b = BCO.__new__(BCO)
        b.beta = beta
        return b

    def test_loss_formula_verification(self):
        """Verify: L = -logsigmoid(beta*logr_w) - logsigmoid(-beta*logr_l)."""
        beta = 0.2
        b = self._make_bco(beta=beta)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = b.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        expected = (-F.logsigmoid(beta * logr_w) - F.logsigmoid(-beta * logr_l)).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_bco_differs_from_dpo(self):
        """BCO uses absolute rewards, not differences like DPO."""
        beta = 0.1
        b = self._make_bco(beta=beta)

        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])

        bco_loss, _, _ = b.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # DPO loss
        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        dpo_loss = -F.logsigmoid(beta * (logr_w - logr_l)).mean()

        assert abs(bco_loss.item() - dpo_loss.item()) > 1e-4

    def test_loss_independent_terms(self):
        """BCO loss separates into chosen and rejected terms independently."""
        b = self._make_bco(beta=0.1)

        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])

        loss, _, _ = b.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # Chosen term
        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        chosen_term = -F.logsigmoid(0.1 * logr_w).mean()
        rejected_term = -F.logsigmoid(-0.1 * logr_l).mean()

        assert abs(loss.item() - (chosen_term + rejected_term).item()) < 1e-5

    def test_loss_always_positive(self):
        """BCO loss should always be positive."""
        b = self._make_bco(beta=0.1)
        for _ in range(10):
            pi_logps_w = torch.randn(4)
            pi_logps_l = torch.randn(4)
            ref_logps_w = torch.randn(4)
            ref_logps_l = torch.randn(4)
            loss, _, _ = b.compute_loss(
                pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
            assert loss.item() > 0

    def test_loss_is_finite(self):
        b = self._make_bco(beta=0.1)
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])
        loss, margin, reward_acc = b.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert torch.isfinite(loss)
        assert torch.isfinite(margin)

    def test_loss_at_init(self):
        """At init (logr~0), loss ~ 2*log(2) ~ 1.386."""
        b = self._make_bco(beta=0.1)
        pi_logps_w = torch.tensor([0.0])
        pi_logps_l = torch.tensor([0.0])
        ref_logps_w = torch.tensor([0.0])
        ref_logps_l = torch.tensor([0.0])
        loss, _, _ = b.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        # -logsigmoid(0) = -log(0.5) = log(2) for each term
        assert abs(loss.item() - 2 * torch.log(torch.tensor(2.0)).item()) < 1e-5

    def test_reward_accuracy(self):
        b = self._make_bco(beta=0.1)
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])
        _, _, reward_acc = b.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert reward_acc.item() == 1.0

    def test_returns_three_values(self):
        b = self._make_bco(beta=0.1)
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        result = b.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert len(result) == 3


class TestBCOConfig:
    def test_config_accepts_bco(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="bco", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alg_name == "bco"
        assert hasattr(t, 'beta')


class TestBCOComputeLogps:
    def _make_bco(self):
        from oxrl.algs.bco import BCO
        b = BCO.__new__(BCO)
        b.beta = 0.1
        return b

    def test_compute_logps_shape(self):
        b = self._make_bco()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = b.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_logps_mask(self):
        b = self._make_bco()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        mask_full = torch.ones(1, 5)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        logps_full = b.compute_logps(logits, target_ids, mask_full)
        logps_half = b.compute_logps(logits, target_ids, mask_half)
        assert logps_full.item() != logps_half.item()
