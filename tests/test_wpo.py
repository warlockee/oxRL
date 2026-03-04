"""
Tests for WPO (Weighted Preference Optimization) algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_wpo.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestWPOImport:
    def test_wpo_importable(self):
        from oxrl.algs.wpo import WPO
        assert WPO is not None

    def test_wpo_in_algs_init(self):
        from oxrl.algs import WPO
        assert WPO is not None

    def test_wpo_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"wpo"' in source


class TestWPOInterface:
    def test_has_required_methods(self):
        from oxrl.algs.wpo import WPO
        assert hasattr(WPO, 'compute_logps')
        assert hasattr(WPO, 'compute_avg_logps')
        assert hasattr(WPO, 'forward')
        assert hasattr(WPO, 'compute_loss')
        assert hasattr(WPO, 'train_step')
        assert hasattr(WPO, 'eval_step')

    def test_init_params(self):
        from oxrl.algs.wpo import WPO
        sig = inspect.signature(WPO.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params
        assert 'label_smoothing' in params


class TestWPOLossMath:
    def _make_wpo(self, beta=0.1, label_smoothing=0.0):
        from oxrl.algs.wpo import WPO
        w = WPO.__new__(WPO)
        w.beta = beta
        w.label_smoothing = label_smoothing
        return w

    def test_loss_formula_verification(self):
        """Verify WPO loss with importance weighting."""
        beta = 0.2
        w = self._make_wpo(beta=beta)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])
        pi_avg_logps_w = torch.tensor([-0.5, -0.3])
        pi_avg_logps_l = torch.tensor([-1.0, -0.8])

        loss, _, _, _ = w.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l,
            pi_avg_logps_w, pi_avg_logps_l)

        # Manual computation
        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        logits = beta * (logr_w - logr_l)
        per_sample = -F.logsigmoid(logits)
        weights = torch.clamp(torch.exp(pi_avg_logps_w + pi_avg_logps_l), max=1.0)
        expected = (weights * per_sample).sum() / weights.sum()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_weight_clamping(self):
        """Weights should be clamped to max=1."""
        w = self._make_wpo(beta=0.1)

        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-2.0])
        # avg_logps sum > 0 would give weight > 1, but clamped to 1
        pi_avg_logps_w = torch.tensor([1.0])
        pi_avg_logps_l = torch.tensor([1.0])

        _, _, _, avg_weight = w.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l,
            pi_avg_logps_w, pi_avg_logps_l)

        assert avg_weight.item() <= 1.0 + 1e-5

    def test_reduces_to_dpo_with_unit_weights(self):
        """When all weights=1, WPO should equal DPO loss."""
        beta = 0.1
        w = self._make_wpo(beta=beta)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])
        # avg_logps sum = 0 means weight = exp(0) = 1
        pi_avg_logps_w = torch.tensor([0.0, 0.0])
        pi_avg_logps_l = torch.tensor([0.0, 0.0])

        loss, _, _, _ = w.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l,
            pi_avg_logps_w, pi_avg_logps_l)

        h = beta * ((pi_logps_w - ref_logps_w) - (pi_logps_l - ref_logps_l))
        dpo_loss = -F.logsigmoid(h).mean()

        assert abs(loss.item() - dpo_loss.item()) < 1e-5

    def test_low_probability_samples_downweighted(self):
        """Samples with low avg log prob should get lower weight."""
        w = self._make_wpo(beta=0.1)

        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-2.0])

        # High probability sample
        _, _, _, weight_high = w.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l,
            torch.tensor([-0.1]), torch.tensor([-0.1]))

        # Low probability sample
        _, _, _, weight_low = w.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l,
            torch.tensor([-5.0]), torch.tensor([-5.0]))

        assert weight_high.item() > weight_low.item()

    def test_label_smoothing(self):
        """With label_smoothing > 0, loss should differ from vanilla."""
        w_vanilla = self._make_wpo(beta=0.1, label_smoothing=0.0)
        w_smooth = self._make_wpo(beta=0.1, label_smoothing=0.1)

        # Use inputs that produce non-zero h so label smoothing matters
        pi_logps_w = torch.tensor([-0.5])
        pi_logps_l = torch.tensor([-3.0])
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-2.0])
        avg_w = torch.tensor([0.0])
        avg_l = torch.tensor([0.0])

        loss_vanilla, _, _, _ = w_vanilla.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l, avg_w, avg_l)
        loss_smooth, _, _, _ = w_smooth.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l, avg_w, avg_l)

        assert abs(loss_vanilla.item() - loss_smooth.item()) > 1e-5

    def test_loss_is_finite(self):
        w = self._make_wpo(beta=0.1)
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])
        avg_w = torch.tensor([-1.0, -2.0])
        avg_l = torch.tensor([-1.5, -2.5])
        loss, margin, reward_acc, avg_weight = w.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l, avg_w, avg_l)
        assert torch.isfinite(loss)
        assert torch.isfinite(margin)

    def test_reward_accuracy(self):
        w = self._make_wpo(beta=0.1)
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])
        avg_w = torch.tensor([0.0, 0.0, 0.0])
        avg_l = torch.tensor([0.0, 0.0, 0.0])
        _, _, reward_acc, _ = w.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l, avg_w, avg_l)
        assert reward_acc.item() == 1.0

    def test_returns_four_values(self):
        w = self._make_wpo(beta=0.1)
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        avg_w = torch.tensor([0.0])
        avg_l = torch.tensor([0.0])
        result = w.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l, avg_w, avg_l)
        assert len(result) == 4


class TestWPOConfig:
    def test_config_accepts_wpo(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="wpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alg_name == "wpo"
        assert hasattr(t, 'wpo_label_smoothing')

    def test_config_default_value(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="wpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.wpo_label_smoothing == 0.0


class TestWPOComputeLogps:
    def _make_wpo(self):
        from oxrl.algs.wpo import WPO
        w = WPO.__new__(WPO)
        w.beta = 0.1
        w.label_smoothing = 0.0
        return w

    def test_compute_logps_shape(self):
        w = self._make_wpo()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = w.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_avg_logps_shape(self):
        w = self._make_wpo()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        avg_logps = w.compute_avg_logps(logits, target_ids, loss_mask)
        assert avg_logps.shape == (2,)

    def test_avg_logps_vs_sum_logps(self):
        """avg_logps should be sum_logps / num_tokens."""
        w = self._make_wpo()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        loss_mask = torch.ones(1, 5)
        sum_logps = w.compute_logps(logits, target_ids, loss_mask)
        avg_logps = w.compute_avg_logps(logits, target_ids, loss_mask)
        assert abs(sum_logps.item() / 5.0 - avg_logps.item()) < 1e-5

    def test_compute_logps_mask(self):
        w = self._make_wpo()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        mask_full = torch.ones(1, 5)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        logps_full = w.compute_logps(logits, target_ids, mask_full)
        logps_half = w.compute_logps(logits, target_ids, mask_half)
        assert logps_full.item() != logps_half.item()
