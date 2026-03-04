"""
Tests for Minor DPO (DPO with Clamped Reject Penalty) algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_minor_dpo.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestMinorDPOImport:
    def test_minor_dpo_importable(self):
        from oxrl.algs.minor_dpo import MinorDPO
        assert MinorDPO is not None

    def test_minor_dpo_in_algs_init(self):
        from oxrl.algs import MinorDPO
        assert MinorDPO is not None

    def test_minor_dpo_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"minor_dpo"' in source


class TestMinorDPOInterface:
    def test_has_required_methods(self):
        from oxrl.algs.minor_dpo import MinorDPO
        assert hasattr(MinorDPO, 'compute_logps')
        assert hasattr(MinorDPO, 'forward')
        assert hasattr(MinorDPO, 'compute_loss')
        assert hasattr(MinorDPO, 'train_step')
        assert hasattr(MinorDPO, 'eval_step')

    def test_init_params(self):
        from oxrl.algs.minor_dpo import MinorDPO
        sig = inspect.signature(MinorDPO.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params

    def test_no_extra_hyperparameters(self):
        """Minor DPO should use the same hyperparameters as DPO (no new ones)."""
        from oxrl.algs.minor_dpo import MinorDPO
        from oxrl.algs.dpo import DPO
        minor_params = set(inspect.signature(MinorDPO.__init__).parameters.keys())
        dpo_params = set(inspect.signature(DPO.__init__).parameters.keys())
        assert minor_params == dpo_params


class TestMinorDPOLossMath:
    def _make_minor(self, beta=0.1):
        from oxrl.algs.minor_dpo import MinorDPO
        s = MinorDPO.__new__(MinorDPO)
        s.beta = beta
        return s

    def test_loss_formula(self):
        """Verify Minor DPO loss: -logsigmoid(beta * (logr_w - max(0, logr_l))).mean()."""
        beta = 0.2
        s = self._make_minor(beta=beta)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = s.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # Manual computation
        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        clamped_logr_l = torch.clamp(logr_l, min=0.0)
        logits = beta * (logr_w - clamped_logr_l)
        expected = -F.logsigmoid(logits).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_equals_dpo_when_logr_l_positive(self):
        """When logr_l > 0, Minor DPO equals standard DPO (clamp has no effect)."""
        beta = 0.1
        s = self._make_minor(beta=beta)

        # Ensure logr_l > 0 by making pi > ref for rejected
        pi_logps_w = torch.tensor([-0.5])
        pi_logps_l = torch.tensor([-1.0])  # pi_l
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-2.0])  # ref_l < pi_l, so logr_l > 0

        loss, _, _ = s.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # Standard DPO loss
        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        dpo_loss = -F.logsigmoid(beta * (logr_w - logr_l)).mean()

        assert abs(loss.item() - dpo_loss.item()) < 1e-5

    def test_differs_from_dpo_when_logr_l_negative(self):
        """When logr_l < 0, Minor DPO differs from DPO (clamp activates)."""
        beta = 0.1
        s = self._make_minor(beta=beta)

        # Ensure logr_l < 0 by making pi < ref for rejected
        pi_logps_w = torch.tensor([-0.5])
        pi_logps_l = torch.tensor([-3.0])  # pi_l much lower
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-1.0])  # ref_l > pi_l, so logr_l < 0

        loss, _, _ = s.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # Standard DPO loss
        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        dpo_loss = -F.logsigmoid(beta * (logr_w - logr_l)).mean()

        # Minor DPO should give a DIFFERENT (larger) loss since clamping
        # removes the negative logr_l contribution
        assert abs(loss.item() - dpo_loss.item()) > 1e-4

    def test_clamp_effect_on_loss(self):
        """When logr_l < 0, clamping to 0 should make loss larger than DPO.

        This is because DPO would have: logr_w - logr_l = logr_w - (negative)
        = logr_w + |logr_l|, which gives a larger logit -> smaller loss.
        Minor DPO clamps: logr_w - 0 = logr_w, giving smaller logit -> larger loss.
        """
        beta = 0.1
        s = self._make_minor(beta=beta)

        pi_logps_w = torch.tensor([-0.5])
        pi_logps_l = torch.tensor([-3.0])
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-1.0])  # logr_l = -2.0 < 0

        loss, _, _ = s.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # DPO loss
        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        dpo_loss = -F.logsigmoid(beta * (logr_w - logr_l)).mean()

        # Minor DPO loss >= DPO loss when logr_l < 0
        assert loss.item() >= dpo_loss.item() - 1e-5

    def test_loss_at_zero_logits(self):
        """When logr_w = logr_l = 0, both DPO and Minor DPO give log(2)."""
        s = self._make_minor(beta=0.1)
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-2.0])

        loss, _, _ = s.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        expected = -F.logsigmoid(torch.tensor(0.0)).item()  # log(2)
        assert abs(loss.item() - expected) < 1e-5

    def test_loss_is_finite(self):
        s = self._make_minor(beta=0.1)
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])
        loss, margin, reward_acc = s.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert torch.isfinite(loss)
        assert torch.isfinite(margin)

    def test_reward_accuracy(self):
        s = self._make_minor(beta=0.1)
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])
        _, _, reward_acc = s.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert reward_acc.item() == 1.0

    def test_returns_three_values(self):
        s = self._make_minor(beta=0.1)
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        result = s.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert len(result) == 3

    def test_loss_always_positive(self):
        """Minor DPO loss should always be positive (unlike SPO)."""
        s = self._make_minor(beta=0.1)
        # Even with very favorable logits, -logsigmoid(x) > 0 for all x
        pi_logps_w = torch.tensor([-0.1, -0.05])
        pi_logps_l = torch.tensor([-10.0, -20.0])
        ref_logps_w = torch.tensor([-1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0])
        loss, _, _ = s.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert loss.item() > 0


class TestMinorDPOGradientFlow:
    def _make_minor(self, beta=0.1):
        from oxrl.algs.minor_dpo import MinorDPO
        s = MinorDPO.__new__(MinorDPO)
        s.beta = beta
        return s

    def test_gradient_flow(self):
        """Verify gradients flow through the Minor DPO loss."""
        s = self._make_minor(beta=0.1)
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

    def test_no_gradient_when_logr_l_negative(self):
        """When logr_l < 0 and clamped to 0, gradient wrt pi_logps_l should be 0."""
        s = self._make_minor(beta=0.1)
        pi_logps_w = torch.tensor([-0.5], requires_grad=True)
        pi_logps_l = torch.tensor([-3.0], requires_grad=True)  # logr_l = -2.0 < 0
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-1.0])

        loss, _, _ = s.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss.backward()

        # When logr_l < 0, clamped to 0, so gradient wrt pi_logps_l is 0
        assert abs(pi_logps_l.grad.item()) < 1e-5


class TestMinorDPOComputeLogps:
    def _make_minor(self):
        from oxrl.algs.minor_dpo import MinorDPO
        s = MinorDPO.__new__(MinorDPO)
        s.beta = 0.1
        return s

    def test_compute_logps_shape(self):
        s = self._make_minor()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = s.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_logps_mask(self):
        s = self._make_minor()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        mask_full = torch.ones(1, 5)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        logps_full = s.compute_logps(logits, target_ids, mask_full)
        logps_half = s.compute_logps(logits, target_ids, mask_half)
        assert logps_full.item() != logps_half.item()


class TestMinorDPOConfig:
    def test_config_accepts_minor_dpo(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="minor_dpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alg_name == "minor_dpo"
