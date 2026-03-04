"""
Tests for H-DPO (Entropy Controllable DPO) algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_hdpo.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestHDPOImport:
    def test_hdpo_importable(self):
        from oxrl.algs.hdpo import HDPO
        assert HDPO is not None

    def test_hdpo_in_algs_init(self):
        from oxrl.algs import HDPO
        assert HDPO is not None

    def test_hdpo_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"hdpo"' in source


class TestHDPOInterface:
    def test_has_required_methods(self):
        from oxrl.algs.hdpo import HDPO
        assert hasattr(HDPO, 'compute_logps')
        assert hasattr(HDPO, 'forward')
        assert hasattr(HDPO, 'compute_loss')
        assert hasattr(HDPO, 'train_step')
        assert hasattr(HDPO, 'eval_step')

    def test_init_params(self):
        from oxrl.algs.hdpo import HDPO
        sig = inspect.signature(HDPO.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params
        assert 'hdpo_alpha' in params


class TestHDPOLossMath:
    def _make_hdpo(self, beta=0.1, hdpo_alpha=1.0):
        from oxrl.algs.hdpo import HDPO
        h = HDPO.__new__(HDPO)
        h.beta = beta
        h.hdpo_alpha = hdpo_alpha
        return h

    def test_loss_formula_verification(self):
        """Verify: L = -logsigmoid(alpha*beta*log(pi_w/pi_l) - beta*log(ref_w/ref_l))."""
        beta = 0.2
        alpha = 0.8
        h = self._make_hdpo(beta=beta, hdpo_alpha=alpha)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = h.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        pi_log_ratio = pi_logps_w - pi_logps_l
        ref_log_ratio = ref_logps_w - ref_logps_l
        logits = alpha * beta * pi_log_ratio - beta * ref_log_ratio
        expected = -F.logsigmoid(logits).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_reduces_to_dpo_when_alpha_one(self):
        """When alpha=1, H-DPO should equal standard DPO."""
        beta = 0.1
        h = self._make_hdpo(beta=beta, hdpo_alpha=1.0)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = h.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # Standard DPO: -logsigmoid(beta * (logr_w - logr_l))
        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        dpo_loss = -F.logsigmoid(beta * (logr_w - logr_l)).mean()

        assert abs(loss.item() - dpo_loss.item()) < 1e-5

    def test_alpha_less_than_one_differs(self):
        """With alpha<1, loss should differ from standard DPO."""
        beta = 0.1
        h_dpo = self._make_hdpo(beta=beta, hdpo_alpha=1.0)
        h_low = self._make_hdpo(beta=beta, hdpo_alpha=0.5)

        pi_logps_w = torch.tensor([-0.5])
        pi_logps_l = torch.tensor([-3.0])
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-2.0])

        loss_dpo, _, _ = h_dpo.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss_low, _, _ = h_low.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        assert abs(loss_dpo.item() - loss_low.item()) > 1e-3

    def test_alpha_greater_than_one_differs(self):
        """With alpha>1, loss should differ from standard DPO."""
        beta = 0.1
        h_dpo = self._make_hdpo(beta=beta, hdpo_alpha=1.0)
        h_high = self._make_hdpo(beta=beta, hdpo_alpha=2.0)

        pi_logps_w = torch.tensor([-0.5])
        pi_logps_l = torch.tensor([-3.0])
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-2.0])

        loss_dpo, _, _ = h_dpo.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss_high, _, _ = h_high.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        assert abs(loss_dpo.item() - loss_high.item()) > 1e-3

    def test_loss_is_finite(self):
        h = self._make_hdpo(beta=0.1, hdpo_alpha=0.5)
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])
        loss, margin, reward_acc = h.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert torch.isfinite(loss)
        assert torch.isfinite(margin)

    def test_reward_accuracy(self):
        h = self._make_hdpo(beta=0.1, hdpo_alpha=1.0)
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])
        _, _, reward_acc = h.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert reward_acc.item() == 1.0

    def test_returns_three_values(self):
        h = self._make_hdpo(beta=0.1, hdpo_alpha=1.0)
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        result = h.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert len(result) == 3


class TestHDPOConfig:
    def test_config_accepts_hdpo(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="hdpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alg_name == "hdpo"
        assert hasattr(t, 'hdpo_alpha')

    def test_config_default_value(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="hdpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.hdpo_alpha == 1.0


class TestHDPOComputeLogps:
    def _make_hdpo(self):
        from oxrl.algs.hdpo import HDPO
        h = HDPO.__new__(HDPO)
        h.beta = 0.1
        h.hdpo_alpha = 1.0
        return h

    def test_compute_logps_shape(self):
        h = self._make_hdpo()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = h.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_logps_mask(self):
        h = self._make_hdpo()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        mask_full = torch.ones(1, 5)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        logps_full = h.compute_logps(logits, target_ids, mask_full)
        logps_half = h.compute_logps(logits, target_ids, mask_half)
        assert logps_full.item() != logps_half.item()
