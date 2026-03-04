"""
Tests for DiscoPOP (Discovered Preference Optimization) algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_discopop.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDiscoPOPImport:
    def test_discopop_importable(self):
        from oxrl.algs.discopop import DiscoPOP
        assert DiscoPOP is not None

    def test_discopop_in_algs_init(self):
        from oxrl.algs import DiscoPOP
        assert DiscoPOP is not None

    def test_discopop_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"discopop"' in source


class TestDiscoPOPInterface:
    def test_has_required_methods(self):
        from oxrl.algs.discopop import DiscoPOP
        assert hasattr(DiscoPOP, 'compute_logps')
        assert hasattr(DiscoPOP, 'forward')
        assert hasattr(DiscoPOP, 'compute_loss')
        assert hasattr(DiscoPOP, 'train_step')
        assert hasattr(DiscoPOP, 'eval_step')

    def test_init_params(self):
        from oxrl.algs.discopop import DiscoPOP
        sig = inspect.signature(DiscoPOP.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params
        assert 'discopop_tau' in params


class TestDiscoPOPLossMath:
    def _make_discopop(self, beta=0.1, discopop_tau=0.05):
        from oxrl.algs.discopop import DiscoPOP
        d = DiscoPOP.__new__(DiscoPOP)
        d.beta = beta
        d.discopop_tau = discopop_tau
        return d

    def test_loss_formula_verification(self):
        """Verify: L = (1-sigma(x/tau))*softplus(-x) + sigma(x/tau)*exp(-x)."""
        beta = 0.2
        tau = 0.1
        d = self._make_discopop(beta=beta, discopop_tau=tau)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = d.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        x = beta * (logr_w - logr_l)

        w_exp = torch.sigmoid(x / tau)
        w_dpo = 1.0 - w_exp
        f_dpo = F.softplus(-x)
        f_exp = torch.exp(-x)
        expected = (w_dpo * f_dpo + w_exp * f_exp).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_loss_approaches_dpo_for_negative_x(self):
        """When x << 0, weight shifts to logistic/DPO loss."""
        d = self._make_discopop(beta=0.1, discopop_tau=0.05)
        # Large negative x: loser has much higher log-ratio than winner
        pi_logps_w = torch.tensor([-10.0])
        pi_logps_l = torch.tensor([0.0])
        ref_logps_w = torch.tensor([0.0])
        ref_logps_l = torch.tensor([0.0])

        loss, _, _ = d.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        x = torch.tensor(0.1 * (-10.0 - 0.0))  # = -1.0
        # For x=-1.0 and tau=0.05: sigmoid(-1.0/0.05) = sigmoid(-20) ~ 0
        # So loss ~ softplus(1.0) = log(1+e) ~ 1.313
        dpo_loss = F.softplus(-x)
        assert abs(loss.item() - dpo_loss.item()) < 0.01

    def test_loss_approaches_exp_for_positive_x(self):
        """When x >> 0, weight shifts to exponential loss."""
        d = self._make_discopop(beta=0.1, discopop_tau=0.05)
        # Large positive x
        pi_logps_w = torch.tensor([5.0])
        pi_logps_l = torch.tensor([-5.0])
        ref_logps_w = torch.tensor([0.0])
        ref_logps_l = torch.tensor([0.0])

        loss, _, _ = d.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        x = torch.tensor(0.1 * (5.0 - (-5.0)))  # = 1.0
        # For x=1.0 and tau=0.05: sigmoid(1.0/0.05) = sigmoid(20) ~ 1
        # So loss ~ exp(-1.0) ~ 0.368
        exp_loss = torch.exp(-x)
        assert abs(loss.item() - exp_loss.item()) < 0.01

    def test_loss_always_positive(self):
        """DiscoPOP loss should always be positive."""
        d = self._make_discopop(beta=0.1, discopop_tau=0.05)
        for _ in range(10):
            pi_logps_w = torch.randn(4)
            pi_logps_l = torch.randn(4)
            ref_logps_w = torch.randn(4)
            ref_logps_l = torch.randn(4)
            loss, _, _ = d.compute_loss(
                pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
            assert loss.item() > 0

    def test_loss_is_finite(self):
        d = self._make_discopop(beta=0.1, discopop_tau=0.05)
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])
        loss, margin, reward_acc = d.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert torch.isfinite(loss)
        assert torch.isfinite(margin)

    def test_tau_controls_blending(self):
        """Larger tau should blend more evenly; smaller tau gives sharper transition."""
        pi_logps_w = torch.tensor([0.5])
        pi_logps_l = torch.tensor([-0.5])
        ref_logps_w = torch.tensor([0.0])
        ref_logps_l = torch.tensor([0.0])

        d_sharp = self._make_discopop(beta=0.1, discopop_tau=0.01)
        d_smooth = self._make_discopop(beta=0.1, discopop_tau=1.0)

        loss_sharp, _, _ = d_sharp.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss_smooth, _, _ = d_smooth.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # Both are valid losses but differ due to blending
        assert abs(loss_sharp.item() - loss_smooth.item()) > 1e-4

    def test_reward_accuracy(self):
        d = self._make_discopop(beta=0.1, discopop_tau=0.05)
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])
        _, _, reward_acc = d.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert reward_acc.item() == 1.0

    def test_returns_three_values(self):
        d = self._make_discopop(beta=0.1, discopop_tau=0.05)
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        result = d.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert len(result) == 3


class TestDiscoPOPConfig:
    def test_config_accepts_discopop(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="discopop", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alg_name == "discopop"
        assert hasattr(t, 'discopop_tau')

    def test_config_default_value(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="discopop", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.discopop_tau == 0.05


class TestDiscoPOPComputeLogps:
    def _make_discopop(self):
        from oxrl.algs.discopop import DiscoPOP
        d = DiscoPOP.__new__(DiscoPOP)
        d.beta = 0.1
        d.discopop_tau = 0.05
        return d

    def test_compute_logps_shape(self):
        d = self._make_discopop()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = d.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_logps_mask(self):
        d = self._make_discopop()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        mask_full = torch.ones(1, 5)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        logps_full = d.compute_logps(logits, target_ids, mask_full)
        logps_half = d.compute_logps(logits, target_ids, mask_half)
        assert logps_full.item() != logps_half.item()
