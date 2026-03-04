"""
Tests for NCA (Noise Contrastive Alignment, pairwise variant) algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_nca.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestNCAImport:
    def test_nca_importable(self):
        from oxrl.algs.nca import NCA
        assert NCA is not None

    def test_nca_in_algs_init(self):
        from oxrl.algs import NCA
        assert NCA is not None

    def test_nca_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"nca"' in source


class TestNCAInterface:
    def test_nca_has_required_methods(self):
        from oxrl.algs.nca import NCA
        assert hasattr(NCA, 'compute_logps')
        assert hasattr(NCA, 'forward')
        assert hasattr(NCA, 'compute_loss')
        assert hasattr(NCA, 'train_step')
        assert hasattr(NCA, 'eval_step')

    def test_nca_init_params(self):
        from oxrl.algs.nca import NCA
        sig = inspect.signature(NCA.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params


class TestNCALossMath:
    def _make_nca(self, beta=0.1):
        from oxrl.algs.nca import NCA
        nca = NCA.__new__(NCA)
        nca.beta = beta
        return nca

    def test_loss_formula_verification(self):
        """Verify: L = -logsigmoid(r_w) - 0.5*(logsigmoid(-r_w) + logsigmoid(-r_l))."""
        beta = 0.2
        nca = self._make_nca(beta=beta)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = nca.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        r_w = beta * (pi_logps_w - ref_logps_w)
        r_l = beta * (pi_logps_l - ref_logps_l)
        expected = (
            -F.logsigmoid(r_w)
            - 0.5 * (F.logsigmoid(-r_w) + F.logsigmoid(-r_l))
        ).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_nca_differs_from_dpo(self):
        """NCA should generally differ from DPO (they optimize different objectives)."""
        from oxrl.algs.dpo import DPO
        beta = 0.1
        nca = self._make_nca(beta=beta)
        dpo = DPO.__new__(DPO)
        dpo.beta = beta

        pi_logps_w = torch.tensor([-0.5, -0.3])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.0, -0.8])
        ref_logps_l = torch.tensor([-1.0, -0.8])

        nca_loss, _, _ = nca.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        dpo_loss, _ = dpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # They should differ since NCA has the extra regularization term
        assert nca_loss.item() != dpo_loss.item()

    def test_loss_at_init(self):
        """At init (log ratios ~ 0), NCA loss should be finite and positive."""
        nca = self._make_nca(beta=0.1)
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-2.0])

        loss, _, _ = nca.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # r_w = r_l = 0
        # -logsigmoid(0) - 0.5*(logsigmoid(0) + logsigmoid(0))
        # = -(-log2) - 0.5*(-log2 + -log2) = log2 + log2 = 2*log2 ~ 1.386
        expected = 2 * torch.log(torch.tensor(2.0))
        assert abs(loss.item() - expected.item()) < 1e-4

    def test_loss_is_finite(self):
        nca = self._make_nca(beta=0.1)
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])
        loss, margin, reward_acc = nca.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert torch.isfinite(loss)
        assert torch.isfinite(margin)

    def test_loss_always_positive(self):
        """NCA loss should always be positive (sum of negative logsigmoids)."""
        nca = self._make_nca(beta=0.1)
        for _ in range(10):
            pi_logps_w = torch.randn(4)
            pi_logps_l = torch.randn(4)
            ref_logps_w = torch.randn(4)
            ref_logps_l = torch.randn(4)
            loss, _, _ = nca.compute_loss(
                pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
            assert loss.item() > 0

    def test_reward_accuracy(self):
        nca = self._make_nca(beta=0.1)
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])
        _, _, reward_acc = nca.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert reward_acc.item() == 1.0

    def test_returns_three_values(self):
        nca = self._make_nca(beta=0.1)
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        result = nca.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert len(result) == 3


class TestNCAConfig:
    def test_config_accepts_nca(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="nca", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alg_name == "nca"
        assert hasattr(t, 'beta')


class TestNCAComputeLogps:
    def _make_nca(self):
        from oxrl.algs.nca import NCA
        nca = NCA.__new__(NCA)
        nca.beta = 0.1
        return nca

    def test_compute_logps_shape(self):
        nca = self._make_nca()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = nca.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_logps_mask(self):
        nca = self._make_nca()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        mask_full = torch.ones(1, 5)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        logps_full = nca.compute_logps(logits, target_ids, mask_full)
        logps_half = nca.compute_logps(logits, target_ids, mask_half)
        assert logps_full.item() != logps_half.item()
