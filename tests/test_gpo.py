"""
Tests for GPO (Generalized Preference Optimization) algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_gpo.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestGPOImport:
    def test_gpo_importable(self):
        from oxrl.algs.gpo import GPO
        assert GPO is not None

    def test_gpo_in_algs_init(self):
        from oxrl.algs import GPO
        assert GPO is not None

    def test_gpo_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"gpo"' in source


class TestGPOInterface:
    def test_has_required_methods(self):
        from oxrl.algs.gpo import GPO
        assert hasattr(GPO, 'compute_logps')
        assert hasattr(GPO, 'forward')
        assert hasattr(GPO, 'compute_loss')
        assert hasattr(GPO, 'train_step')
        assert hasattr(GPO, 'eval_step')

    def test_init_params(self):
        from oxrl.algs.gpo import GPO
        sig = inspect.signature(GPO.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params
        assert 'gpo_loss_type' in params

    def test_invalid_loss_type_raises(self):
        from oxrl.algs.gpo import GPO
        with pytest.raises(ValueError, match="Invalid gpo_loss_type"):
            GPO(model_engine=None, ref_model_engine=None,
                optimizer=None, gpo_loss_type="invalid")


class TestGPOLogistic:
    """Test logistic loss type (should match DPO)."""
    def _make_gpo(self, beta=0.1):
        from oxrl.algs.gpo import GPO
        g = GPO.__new__(GPO)
        g.beta = beta
        g.gpo_loss_type = "logistic"
        return g

    def test_logistic_equals_dpo(self):
        """Logistic GPO: f(h) = log(1+exp(-h)) = softplus(-h) = -logsigmoid(h)."""
        beta = 0.1
        g = self._make_gpo(beta=beta)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = g.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        h = beta * ((pi_logps_w - ref_logps_w) - (pi_logps_l - ref_logps_l))
        expected = -F.logsigmoid(h).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_logistic_at_zero(self):
        """At h=0, logistic loss = log(2) = 0.6931."""
        g = self._make_gpo(beta=1.0)
        z = torch.tensor([0.0])
        loss, _, _ = g.compute_loss(z, z, z, z)
        assert abs(loss.item() - 0.6931) < 1e-3


class TestGPOExponential:
    """Test exponential loss type: f(h) = exp(-h)."""
    def _make_gpo(self, beta=0.1):
        from oxrl.algs.gpo import GPO
        g = GPO.__new__(GPO)
        g.beta = beta
        g.gpo_loss_type = "exponential"
        return g

    def test_exponential_formula(self):
        beta = 0.2
        g = self._make_gpo(beta=beta)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = g.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        h = beta * ((pi_logps_w - ref_logps_w) - (pi_logps_l - ref_logps_l))
        expected = torch.exp(-h).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_exponential_at_zero(self):
        """At h=0, exp(0)=1."""
        g = self._make_gpo(beta=1.0)
        z = torch.tensor([0.0])
        loss, _, _ = g.compute_loss(z, z, z, z)
        assert abs(loss.item() - 1.0) < 1e-5

    def test_exponential_decreases_with_positive_h(self):
        """exp(-h) decreases as h increases."""
        g = self._make_gpo(beta=1.0)

        # h > 0 (chosen better than rejected)
        pi_logps_w = torch.tensor([0.0])
        pi_logps_l = torch.tensor([-5.0])
        ref_logps_w = torch.tensor([0.0])
        ref_logps_l = torch.tensor([0.0])

        loss, _, _ = g.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert loss.item() < 1.0  # less than exp(0)=1

    def test_exponential_increases_with_negative_h(self):
        """exp(-h) increases as h decreases below 0."""
        g = self._make_gpo(beta=1.0)

        # h < 0 (rejected better than chosen)
        pi_logps_w = torch.tensor([-5.0])
        pi_logps_l = torch.tensor([0.0])
        ref_logps_w = torch.tensor([0.0])
        ref_logps_l = torch.tensor([0.0])

        loss, _, _ = g.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert loss.item() > 1.0  # greater than exp(0)=1


class TestGPOTruncatedQuadratic:
    """Test truncated quadratic loss: f(h) = max(0, 1-h)^2."""
    def _make_gpo(self, beta=0.1):
        from oxrl.algs.gpo import GPO
        g = GPO.__new__(GPO)
        g.beta = beta
        g.gpo_loss_type = "truncated_quadratic"
        return g

    def test_truncated_quadratic_formula(self):
        beta = 0.2
        g = self._make_gpo(beta=beta)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = g.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        h = beta * ((pi_logps_w - ref_logps_w) - (pi_logps_l - ref_logps_l))
        expected = (torch.relu(1.0 - h) ** 2).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_truncated_quadratic_zero_loss_for_large_margin(self):
        """When h > 1, loss should be zero (truncation)."""
        g = self._make_gpo(beta=1.0)

        # Make h > 1
        pi_logps_w = torch.tensor([0.0])
        pi_logps_l = torch.tensor([-5.0])
        ref_logps_w = torch.tensor([0.0])
        ref_logps_l = torch.tensor([0.0])

        loss, _, _ = g.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert loss.item() == 0.0

    def test_truncated_quadratic_at_boundary(self):
        """At h=1, loss should be 0."""
        g = self._make_gpo(beta=1.0)
        # h = 1.0
        pi_logps_w = torch.tensor([1.0])
        pi_logps_l = torch.tensor([0.0])
        ref_logps_w = torch.tensor([0.0])
        ref_logps_l = torch.tensor([0.0])
        loss, _, _ = g.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert abs(loss.item()) < 1e-5

    def test_truncated_quadratic_at_zero(self):
        """At h=0, loss = max(0,1-0)^2 = 1."""
        g = self._make_gpo(beta=1.0)
        z = torch.tensor([0.0])
        loss, _, _ = g.compute_loss(z, z, z, z)
        assert abs(loss.item() - 1.0) < 1e-5


class TestGPOSavage:
    """Test savage loss: f(h) = 1/(1+exp(h))^2 = sigmoid(-h)^2."""
    def _make_gpo(self, beta=0.1):
        from oxrl.algs.gpo import GPO
        g = GPO.__new__(GPO)
        g.beta = beta
        g.gpo_loss_type = "savage"
        return g

    def test_savage_formula(self):
        beta = 0.2
        g = self._make_gpo(beta=beta)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = g.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        h = beta * ((pi_logps_w - ref_logps_w) - (pi_logps_l - ref_logps_l))
        expected = (torch.sigmoid(-h) ** 2).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_savage_at_zero(self):
        """At h=0, savage = sigmoid(0)^2 = 0.5^2 = 0.25."""
        g = self._make_gpo(beta=1.0)
        z = torch.tensor([0.0])
        loss, _, _ = g.compute_loss(z, z, z, z)
        assert abs(loss.item() - 0.25) < 1e-5

    def test_savage_decreases_with_positive_h(self):
        """sigmoid(-h)^2 decreases as h increases."""
        g = self._make_gpo(beta=1.0)

        pi_logps_w = torch.tensor([0.0])
        pi_logps_l = torch.tensor([-5.0])
        ref_logps_w = torch.tensor([0.0])
        ref_logps_l = torch.tensor([0.0])

        loss, _, _ = g.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert loss.item() < 0.25  # less than at h=0


class TestGPOConfig:
    def test_config_accepts_gpo(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="gpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alg_name == "gpo"
        assert hasattr(t, 'gpo_loss_type')

    def test_config_default_value(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="gpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.gpo_loss_type == "exponential"


class TestGPOComputeLogps:
    def _make_gpo(self):
        from oxrl.algs.gpo import GPO
        g = GPO.__new__(GPO)
        g.beta = 0.1
        g.gpo_loss_type = "exponential"
        return g

    def test_compute_logps_shape(self):
        g = self._make_gpo()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = g.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_logps_mask(self):
        g = self._make_gpo()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        mask_full = torch.ones(1, 5)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        logps_full = g.compute_logps(logits, target_ids, mask_full)
        logps_half = g.compute_logps(logits, target_ids, mask_half)
        assert logps_full.item() != logps_half.item()


class TestGPOGeneral:
    """Cross-loss-type general tests."""
    def _make_gpo(self, loss_type="exponential", beta=0.1):
        from oxrl.algs.gpo import GPO
        g = GPO.__new__(GPO)
        g.beta = beta
        g.gpo_loss_type = loss_type
        return g

    def test_all_loss_types_finite(self):
        """All loss types should return finite values."""
        for lt in ["logistic", "exponential", "truncated_quadratic", "savage"]:
            g = self._make_gpo(loss_type=lt)
            pi_logps_w = torch.tensor([-2.5, -3.0])
            pi_logps_l = torch.tensor([-4.0, -5.0])
            ref_logps_w = torch.tensor([-2.6, -3.1])
            ref_logps_l = torch.tensor([-4.1, -5.1])
            loss, margin, reward_acc = g.compute_loss(
                pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
            assert torch.isfinite(loss), f"Non-finite loss for {lt}"
            assert torch.isfinite(margin), f"Non-finite margin for {lt}"

    def test_all_loss_types_return_three_values(self):
        for lt in ["logistic", "exponential", "truncated_quadratic", "savage"]:
            g = self._make_gpo(loss_type=lt)
            pi_logps_w = torch.tensor([-1.0])
            pi_logps_l = torch.tensor([-2.0])
            ref_logps_w = torch.tensor([-1.1])
            ref_logps_l = torch.tensor([-2.1])
            result = g.compute_loss(
                pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
            assert len(result) == 3, f"{lt} did not return 3 values"

    def test_all_loss_types_nonneg(self):
        """All GPO losses are convex functions >= 0 for the given inputs."""
        for lt in ["logistic", "exponential", "truncated_quadratic", "savage"]:
            g = self._make_gpo(loss_type=lt)
            # Test various inputs
            for w, l in [(-1.0, -2.0), (-3.0, -1.0), (0.0, 0.0)]:
                pi_logps_w = torch.tensor([w])
                pi_logps_l = torch.tensor([l])
                ref_logps_w = torch.tensor([0.0])
                ref_logps_l = torch.tensor([0.0])
                loss, _, _ = g.compute_loss(
                    pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
                assert loss.item() >= 0.0, f"Negative loss for {lt}"

    def test_reward_accuracy_all_types(self):
        for lt in ["logistic", "exponential", "truncated_quadratic", "savage"]:
            g = self._make_gpo(loss_type=lt)
            pi_logps_w = torch.tensor([-0.5, -0.3])
            pi_logps_l = torch.tensor([-2.0, -1.5])
            ref_logps_w = torch.tensor([-1.0, -1.0])
            ref_logps_l = torch.tensor([-1.0, -1.0])
            _, _, reward_acc = g.compute_loss(
                pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
            assert reward_acc.item() == 1.0, f"Wrong reward_acc for {lt}"
