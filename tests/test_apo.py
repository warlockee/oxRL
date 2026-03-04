"""
Tests for APO (Anchored Preference Optimization) algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_apo.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestAPOImport:
    def test_apo_importable(self):
        from oxrl.algs.apo import APO
        assert APO is not None

    def test_apo_in_algs_init(self):
        from oxrl.algs import APO
        assert APO is not None

    def test_apo_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"apo"' in source


class TestAPOInterface:
    def test_apo_has_required_methods(self):
        from oxrl.algs.apo import APO
        assert hasattr(APO, 'compute_logps')
        assert hasattr(APO, 'forward')
        assert hasattr(APO, 'compute_loss')
        assert hasattr(APO, 'train_step')
        assert hasattr(APO, 'eval_step')

    def test_apo_init_params(self):
        from oxrl.algs.apo import APO
        sig = inspect.signature(APO.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params
        assert 'apo_mode' in params

    def test_invalid_mode_raises(self):
        from oxrl.algs.apo import APO
        with pytest.raises(ValueError, match="Unknown apo_mode"):
            APO(model_engine=None, ref_model_engine=None, optimizer=None,
                apo_mode="invalid")


class TestAPOZeroLossMath:
    def _make_apo(self, beta=0.1, mode="zero"):
        from oxrl.algs.apo import APO
        apo = APO.__new__(APO)
        apo.beta = beta
        apo.apo_mode = mode
        return apo

    def test_zero_formula_verification(self):
        """Verify: L_zero = -sigmoid(r_w) + sigmoid(r_l)."""
        beta = 0.2
        apo = self._make_apo(beta=beta, mode="zero")

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = apo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        r_w = beta * (pi_logps_w - ref_logps_w)
        r_l = beta * (pi_logps_l - ref_logps_l)
        expected = (-torch.sigmoid(r_w) + torch.sigmoid(r_l)).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_zero_loss_at_init(self):
        """At init (log ratios ~ 0), APO-zero loss should be near 0."""
        apo = self._make_apo(beta=0.1, mode="zero")

        # Equal policy and reference (log ratios = 0)
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-2.0])

        loss, _, _ = apo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # sigmoid(0) = 0.5, so -0.5 + 0.5 = 0
        assert abs(loss.item()) < 1e-5, \
            f"APO-zero should be 0 at init: {loss}"

    def test_zero_negative_when_correct(self):
        """APO-zero loss is negative when model has correct preference."""
        apo = self._make_apo(beta=0.5, mode="zero")

        # Strong preference for chosen
        pi_logps_w = torch.tensor([0.0])
        pi_logps_l = torch.tensor([-5.0])
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-1.0])

        loss, _, _ = apo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # sigmoid(positive) > 0.5 and sigmoid(negative) < 0.5
        # so -sigmoid(positive) + sigmoid(negative) < 0
        assert loss.item() < 0


class TestAPODownLossMath:
    def _make_apo(self, beta=0.1, mode="down"):
        from oxrl.algs.apo import APO
        apo = APO.__new__(APO)
        apo.beta = beta
        apo.apo_mode = mode
        return apo

    def test_down_formula_verification(self):
        """Verify: L_down = sigmoid(r_w) - sigmoid(r_w - r_l)."""
        beta = 0.2
        apo = self._make_apo(beta=beta, mode="down")

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = apo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        r_w = beta * (pi_logps_w - ref_logps_w)
        r_l = beta * (pi_logps_l - ref_logps_l)
        expected = (torch.sigmoid(r_w) - torch.sigmoid(r_w - r_l)).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_down_loss_at_init(self):
        """At init (log ratios ~ 0), APO-down loss should be near 0."""
        apo = self._make_apo(beta=0.1, mode="down")

        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-2.0])

        loss, _, _ = apo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # sigmoid(0) - sigmoid(0-0) = 0.5 - 0.5 = 0
        assert abs(loss.item()) < 1e-5

    def test_down_modes_differ(self):
        """APO-zero and APO-down should give different losses."""
        from oxrl.algs.apo import APO
        apo_zero = APO.__new__(APO)
        apo_zero.beta = 0.1
        apo_zero.apo_mode = "zero"

        apo_down = APO.__new__(APO)
        apo_down.beta = 0.1
        apo_down.apo_mode = "down"

        pi_logps_w = torch.tensor([-0.5, -0.3])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.0, -0.8])
        ref_logps_l = torch.tensor([-1.0, -0.8])

        loss_zero, _, _ = apo_zero.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss_down, _, _ = apo_down.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # They should generally differ
        assert loss_zero.item() != loss_down.item()


class TestAPOGeneral:
    def _make_apo(self, beta=0.1, mode="zero"):
        from oxrl.algs.apo import APO
        apo = APO.__new__(APO)
        apo.beta = beta
        apo.apo_mode = mode
        return apo

    def test_loss_is_finite(self):
        apo = self._make_apo(beta=0.1, mode="zero")
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])
        loss, margin, reward_acc = apo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert torch.isfinite(loss)
        assert torch.isfinite(margin)

    def test_reward_accuracy(self):
        apo = self._make_apo(beta=0.1, mode="zero")
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])
        _, _, reward_acc = apo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert reward_acc.item() == 1.0


class TestAPOConfig:
    def test_config_has_apo_field(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="apo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert hasattr(t, 'apo_mode')

    def test_config_default_mode(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="apo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.apo_mode == "zero"

    def test_config_custom_mode(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="apo", total_number_of_epochs=1,
                  micro_batches_per_epoch=10, apo_mode="down")
        assert t.apo_mode == "down"


class TestAPOComputeLogps:
    def _make_apo(self):
        from oxrl.algs.apo import APO
        apo = APO.__new__(APO)
        apo.beta = 0.1
        apo.apo_mode = "zero"
        return apo

    def test_compute_logps_shape(self):
        apo = self._make_apo()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = apo.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)
