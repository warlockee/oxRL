"""
Tests for f-DPO (f-Divergence Preference Optimization) algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_fdpo.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestFDPOImport:
    def test_fdpo_importable(self):
        from oxrl.algs.fdpo import FDPO
        assert FDPO is not None

    def test_fdpo_in_algs_init(self):
        from oxrl.algs import FDPO
        assert FDPO is not None

    def test_fdpo_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"fdpo"' in source


class TestFDPOInterface:
    def test_has_required_methods(self):
        from oxrl.algs.fdpo import FDPO
        assert hasattr(FDPO, 'compute_logps')
        assert hasattr(FDPO, 'forward')
        assert hasattr(FDPO, 'compute_loss')
        assert hasattr(FDPO, 'train_step')
        assert hasattr(FDPO, 'eval_step')

    def test_init_params(self):
        from oxrl.algs.fdpo import FDPO
        sig = inspect.signature(FDPO.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params
        assert 'fdpo_divergence' in params
        assert 'fdpo_alpha' in params

    def test_invalid_divergence_raises(self):
        from oxrl.algs.fdpo import FDPO
        with pytest.raises(ValueError, match="Invalid fdpo_divergence"):
            FDPO(model_engine=None, ref_model_engine=None,
                 optimizer=None, fdpo_divergence="invalid")


class TestFDPOReverseKL:
    """Reverse KL should be equivalent to standard DPO."""
    def _make_fdpo(self, beta=0.1):
        from oxrl.algs.fdpo import FDPO
        f = FDPO.__new__(FDPO)
        f.beta = beta
        f.fdpo_divergence = "reverse_kl"
        f.fdpo_alpha = 0.5
        return f

    def test_reverse_kl_equals_dpo(self):
        beta = 0.1
        f = self._make_fdpo(beta=beta)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = f.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        h = beta * ((pi_logps_w - ref_logps_w) - (pi_logps_l - ref_logps_l))
        dpo_loss = -F.logsigmoid(h).mean()

        assert abs(loss.item() - dpo_loss.item()) < 1e-5


class TestFDPOForwardKL:
    """Test forward KL divergence type."""
    def _make_fdpo(self, beta=0.1):
        from oxrl.algs.fdpo import FDPO
        f = FDPO.__new__(FDPO)
        f.beta = beta
        f.fdpo_divergence = "forward_kl"
        f.fdpo_alpha = 0.5
        return f

    def test_forward_kl_formula(self):
        beta = 0.2
        f = self._make_fdpo(beta=beta)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = f.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        h = beta * (logr_w - logr_l) + beta * (torch.exp(-logr_w) - torch.exp(-logr_l))
        expected = -F.logsigmoid(h).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_forward_kl_differs_from_reverse_kl(self):
        from oxrl.algs.fdpo import FDPO
        f_rev = FDPO.__new__(FDPO)
        f_rev.beta = 0.1
        f_rev.fdpo_divergence = "reverse_kl"
        f_rev.fdpo_alpha = 0.5

        f_fwd = self._make_fdpo(beta=0.1)

        # Use inputs that produce non-zero logratios
        pi_logps_w = torch.tensor([-0.5])
        pi_logps_l = torch.tensor([-3.0])
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-2.0])

        loss_rev, _, _ = f_rev.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss_fwd, _, _ = f_fwd.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        assert abs(loss_rev.item() - loss_fwd.item()) > 1e-3


class TestFDPOJSDivergence:
    """Test Jensen-Shannon divergence type."""
    def _make_fdpo(self, beta=0.1):
        from oxrl.algs.fdpo import FDPO
        f = FDPO.__new__(FDPO)
        f.beta = beta
        f.fdpo_divergence = "js_divergence"
        f.fdpo_alpha = 0.5
        return f

    def test_js_formula(self):
        beta = 0.2
        f = self._make_fdpo(beta=beta)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = f.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        h = beta * (logr_w - logr_l) - (F.softplus(logr_w) - F.softplus(logr_l))
        expected = -F.logsigmoid(h).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_js_differs_from_reverse_kl(self):
        from oxrl.algs.fdpo import FDPO
        f_rev = FDPO.__new__(FDPO)
        f_rev.beta = 0.1
        f_rev.fdpo_divergence = "reverse_kl"
        f_rev.fdpo_alpha = 0.5

        f_js = self._make_fdpo(beta=0.1)

        pi_logps_w = torch.tensor([-0.5])
        pi_logps_l = torch.tensor([-3.0])
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-2.0])

        loss_rev, _, _ = f_rev.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss_js, _, _ = f_js.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        assert abs(loss_rev.item() - loss_js.item()) > 1e-3


class TestFDPOAlphaDivergence:
    """Test alpha divergence type."""
    def _make_fdpo(self, beta=0.1, alpha=0.5):
        from oxrl.algs.fdpo import FDPO
        f = FDPO.__new__(FDPO)
        f.beta = beta
        f.fdpo_divergence = "alpha_divergence"
        f.fdpo_alpha = alpha
        return f

    def test_alpha_formula(self):
        beta = 0.2
        alpha = 0.3
        f = self._make_fdpo(beta=beta, alpha=alpha)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = f.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        h = (torch.exp(logr_l * -alpha) - torch.exp(logr_w * -alpha)) / alpha
        expected = -F.logsigmoid(h).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_alpha_differs_from_reverse_kl(self):
        from oxrl.algs.fdpo import FDPO
        f_rev = FDPO.__new__(FDPO)
        f_rev.beta = 0.1
        f_rev.fdpo_divergence = "reverse_kl"
        f_rev.fdpo_alpha = 0.5

        f_alpha = self._make_fdpo(beta=0.1, alpha=0.3)

        pi_logps_w = torch.tensor([-0.5])
        pi_logps_l = torch.tensor([-3.0])
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-2.0])

        loss_rev, _, _ = f_rev.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss_alpha, _, _ = f_alpha.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        assert abs(loss_rev.item() - loss_alpha.item()) > 1e-3


class TestFDPOConfig:
    def test_config_accepts_fdpo(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="fdpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alg_name == "fdpo"
        assert hasattr(t, 'fdpo_divergence')
        assert hasattr(t, 'fdpo_alpha')

    def test_config_default_values(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="fdpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.fdpo_divergence == "reverse_kl"
        assert t.fdpo_alpha == 0.5


class TestFDPOComputeLogps:
    def _make_fdpo(self):
        from oxrl.algs.fdpo import FDPO
        f = FDPO.__new__(FDPO)
        f.beta = 0.1
        f.fdpo_divergence = "reverse_kl"
        f.fdpo_alpha = 0.5
        return f

    def test_compute_logps_shape(self):
        f = self._make_fdpo()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = f.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_logps_mask(self):
        f = self._make_fdpo()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        mask_full = torch.ones(1, 5)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        logps_full = f.compute_logps(logits, target_ids, mask_full)
        logps_half = f.compute_logps(logits, target_ids, mask_half)
        assert logps_full.item() != logps_half.item()


class TestFDPOGeneral:
    """Cross-divergence general tests."""
    def _make_fdpo(self, divergence="reverse_kl", beta=0.1):
        from oxrl.algs.fdpo import FDPO
        f = FDPO.__new__(FDPO)
        f.beta = beta
        f.fdpo_divergence = divergence
        f.fdpo_alpha = 0.5
        return f

    def test_all_divergences_finite(self):
        for div in ["reverse_kl", "forward_kl", "js_divergence", "alpha_divergence"]:
            f = self._make_fdpo(divergence=div)
            pi_logps_w = torch.tensor([-2.5, -3.0])
            pi_logps_l = torch.tensor([-4.0, -5.0])
            ref_logps_w = torch.tensor([-2.6, -3.1])
            ref_logps_l = torch.tensor([-4.1, -5.1])
            loss, margin, reward_acc = f.compute_loss(
                pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
            assert torch.isfinite(loss), f"Non-finite loss for {div}"

    def test_all_divergences_return_three_values(self):
        for div in ["reverse_kl", "forward_kl", "js_divergence", "alpha_divergence"]:
            f = self._make_fdpo(divergence=div)
            pi_logps_w = torch.tensor([-1.0])
            pi_logps_l = torch.tensor([-2.0])
            ref_logps_w = torch.tensor([-1.1])
            ref_logps_l = torch.tensor([-2.1])
            result = f.compute_loss(
                pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
            assert len(result) == 3

    def test_reward_accuracy_all_divergences(self):
        for div in ["reverse_kl", "forward_kl", "js_divergence", "alpha_divergence"]:
            f = self._make_fdpo(divergence=div)
            pi_logps_w = torch.tensor([-0.5, -0.3])
            pi_logps_l = torch.tensor([-2.0, -1.5])
            ref_logps_w = torch.tensor([-1.0, -1.0])
            ref_logps_l = torch.tensor([-1.0, -1.0])
            _, _, reward_acc = f.compute_loss(
                pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
            assert reward_acc.item() == 1.0
