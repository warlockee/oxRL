"""
Tests for AlphaPO (Reward Shape Matters for LLM Alignment) algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_alphapo.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestAlphaPOImport:
    """Test AlphaPO is properly importable and registered."""

    def test_alphapo_importable(self):
        from oxrl.algs.alphapo import AlphaPO
        assert AlphaPO is not None

    def test_alphapo_in_algs_init(self):
        from oxrl.algs import AlphaPO
        assert AlphaPO is not None

    def test_alphapo_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"alphapo"' in source, "AlphaPO should be registered in SL_ALGORITHMS"

    def test_alphapo_uses_preference_dataset(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"alphapo"' in source


class TestAlphaPOInterface:
    """Test AlphaPO has the expected interface."""

    def test_alphapo_has_required_methods(self):
        from oxrl.algs.alphapo import AlphaPO
        assert hasattr(AlphaPO, 'compute_logps')
        assert hasattr(AlphaPO, 'forward')
        assert hasattr(AlphaPO, 'logps_to_rewards')
        assert hasattr(AlphaPO, 'compute_loss')
        assert hasattr(AlphaPO, 'train_step')
        assert hasattr(AlphaPO, 'eval_step')

    def test_alphapo_init_params(self):
        from oxrl.algs.alphapo import AlphaPO
        sig = inspect.signature(AlphaPO.__init__)
        params = list(sig.parameters.keys())
        assert 'model_engine' in params
        assert 'optimizer' in params
        assert 'beta' in params
        assert 'gamma' in params
        assert 'alpha' in params

    def test_alphapo_no_ref_model(self):
        """AlphaPO should NOT require a reference model."""
        from oxrl.algs.alphapo import AlphaPO
        sig = inspect.signature(AlphaPO.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' not in params


class TestAlphaPORewardTransformation:
    """Test the core alpha reward transformation."""

    def _make_alphapo(self, alpha=1.0, beta=0.1, gamma=0.5):
        from oxrl.algs.alphapo import AlphaPO
        apo = AlphaPO.__new__(AlphaPO)
        apo.alpha = alpha
        apo.beta = beta
        apo.gamma = gamma
        return apo

    def test_alpha_zero_reduces_to_logps(self):
        """When alpha=0, logps_to_rewards should return logps unchanged."""
        apo = self._make_alphapo(alpha=0.0)
        logps = torch.tensor([-0.5, -1.0, -2.0])
        rewards = apo.logps_to_rewards(logps)
        assert torch.allclose(rewards, logps), \
            f"alpha=0 should give rewards=logps, got {rewards} vs {logps}"

    def test_alpha_nonzero_transforms(self):
        """When alpha!=0, rewards should differ from logps."""
        apo = self._make_alphapo(alpha=1.0)
        logps = torch.tensor([-0.5, -1.0, -2.0])
        rewards = apo.logps_to_rewards(logps)

        # Verify: r = (1 - p^{-alpha}) / alpha = (1 - exp(-alpha * logps)) / alpha
        p_neg_alpha = torch.exp(-1.0 * logps)
        expected = (1.0 - p_neg_alpha) / 1.0
        assert torch.allclose(rewards, expected, atol=1e-5), \
            f"alpha=1 transformation incorrect: {rewards} vs {expected}"

    def test_alpha_transformation_formula(self):
        """Verify exact formula: r = (1 - p^{-alpha}) / alpha."""
        for alpha_val in [0.5, 1.0, 2.0, 5.0]:
            apo = self._make_alphapo(alpha=alpha_val)
            logps = torch.tensor([-0.3, -0.7, -1.5])
            rewards = apo.logps_to_rewards(logps)

            neg_alpha_logps = (-alpha_val * logps).clamp(max=50.0)
            p_neg_alpha = torch.exp(neg_alpha_logps)
            expected = (1.0 - p_neg_alpha) / alpha_val

            assert torch.allclose(rewards, expected, atol=1e-5), \
                f"alpha={alpha_val}: {rewards} vs {expected}"

    def test_rewards_are_finite(self):
        """Rewards should be finite even for extreme log-probabilities."""
        apo = self._make_alphapo(alpha=1.0)

        # Normal range
        logps_normal = torch.tensor([-0.5, -1.0, -3.0])
        r_normal = apo.logps_to_rewards(logps_normal)
        assert torch.all(torch.isfinite(r_normal)), f"Normal rewards should be finite: {r_normal}"

        # Very negative logps (near-zero probability)
        logps_extreme = torch.tensor([-10.0, -50.0, -100.0])
        r_extreme = apo.logps_to_rewards(logps_extreme)
        assert torch.all(torch.isfinite(r_extreme)), f"Extreme rewards should be finite: {r_extreme}"

    def test_different_alpha_gives_different_rewards(self):
        """Different alpha values should give different reward values."""
        logps = torch.tensor([-0.5, -1.0, -2.0])

        rewards = {}
        for alpha_val in [0.0, 0.5, 1.0, 2.0]:
            apo = self._make_alphapo(alpha=alpha_val)
            rewards[alpha_val] = apo.logps_to_rewards(logps)

        # Pairwise comparison: rewards should differ
        for a1, r1 in rewards.items():
            for a2, r2 in rewards.items():
                if a1 != a2:
                    assert not torch.allclose(r1, r2, atol=1e-3), \
                        f"alpha={a1} and alpha={a2} should give different rewards"


class TestAlphaPOLossMath:
    """Test AlphaPO loss computation."""

    def _make_alphapo(self, alpha=1.0, beta=0.1, gamma=0.5):
        from oxrl.algs.alphapo import AlphaPO
        apo = AlphaPO.__new__(AlphaPO)
        apo.alpha = alpha
        apo.beta = beta
        apo.gamma = gamma
        return apo

    def test_alpha_zero_matches_simpo(self):
        """With alpha=0, AlphaPO should produce same loss as SimPO."""
        from oxrl.algs.simpo import SimPO
        apo = self._make_alphapo(alpha=0.0, beta=0.1, gamma=0.5)
        simpo = SimPO.__new__(SimPO)
        simpo.beta = 0.1
        simpo.gamma = 0.5

        logps_w = torch.tensor([-0.5, -0.3])
        logps_l = torch.tensor([-1.5, -1.0])

        apo_loss, _, _ = apo.compute_loss(logps_w, logps_l)
        simpo_loss, _ = simpo.compute_loss(logps_w, logps_l)

        assert abs(apo_loss.item() - simpo_loss.item()) < 1e-5, \
            f"alpha=0 AlphaPO should match SimPO: {apo_loss} vs {simpo_loss}"

    def test_loss_formula_verification(self):
        """Verify: loss = -logsigmoid(beta * (r_w - r_l) - gamma)."""
        apo = self._make_alphapo(alpha=1.0, beta=0.2, gamma=0.3)
        logps_w = torch.tensor([-0.5, -0.3])
        logps_l = torch.tensor([-1.5, -1.0])

        loss, margin, _ = apo.compute_loss(logps_w, logps_l)

        # Manual computation
        r_w = apo.logps_to_rewards(logps_w)
        r_l = apo.logps_to_rewards(logps_l)
        expected_logits = 0.2 * (r_w - r_l) - 0.3
        expected_loss = -F.logsigmoid(expected_logits).mean()

        assert abs(loss.item() - expected_loss.item()) < 1e-5

    def test_loss_is_finite(self):
        """Loss should always be finite."""
        for alpha_val in [0.0, 0.5, 1.0, 2.0]:
            apo = self._make_alphapo(alpha=alpha_val)
            logps_w = torch.tensor([-0.5, -1.0, -2.0])
            logps_l = torch.tensor([-1.5, -2.0, -3.0])
            loss, margin, logps_margin = apo.compute_loss(logps_w, logps_l)
            assert torch.isfinite(loss), f"Loss should be finite for alpha={alpha_val}"
            assert torch.isfinite(margin), f"Margin should be finite for alpha={alpha_val}"

    def test_correct_preference_lower_loss(self):
        """When chosen has higher log-probs, loss should be lower."""
        apo = self._make_alphapo(alpha=1.0)

        logps_w = torch.tensor([-0.3, -0.5])
        logps_l = torch.tensor([-2.0, -3.0])

        loss_good, _, _ = apo.compute_loss(logps_w, logps_l)
        loss_bad, _, _ = apo.compute_loss(logps_l, logps_w)

        assert loss_good.item() < loss_bad.item(), \
            f"Correct preference should have lower loss: {loss_good} vs {loss_bad}"

    def test_margin_is_positive_for_correct_preference(self):
        """Margin should be positive when chosen > rejected."""
        apo = self._make_alphapo(alpha=1.0)
        logps_w = torch.tensor([-0.3, -0.5])
        logps_l = torch.tensor([-2.0, -3.0])
        _, margin, logps_margin = apo.compute_loss(logps_w, logps_l)
        assert margin.item() > 0, "Margin should be positive"
        assert logps_margin.item() > 0, "Logps margin should be positive"


class TestAlphaPOConfig:
    """Test AlphaPO config fields."""

    def test_config_has_alphapo_field(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="alphapo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert hasattr(t, 'alphapo_alpha'), "Train should have alphapo_alpha"

    def test_config_default_value(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="alphapo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alphapo_alpha == 1.0, f"Default alphapo_alpha should be 1.0, got {t.alphapo_alpha}"

    def test_config_custom_value(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="alphapo", total_number_of_epochs=1,
                  micro_batches_per_epoch=10, alphapo_alpha=0.5)
        assert t.alphapo_alpha == 0.5


class TestAlphaPOLogps:
    """Test length-normalized log probability computation."""

    def _make_alphapo(self):
        from oxrl.algs.alphapo import AlphaPO
        apo = AlphaPO.__new__(AlphaPO)
        return apo

    def test_logps_length_normalized(self):
        """Verify log-probs are length-normalized."""
        apo = self._make_alphapo()
        B, T, V = 2, 10, 50
        logits = torch.randn(B, T, V)
        target_ids = torch.randint(0, V, (B, T))

        # Full mask
        mask_full = torch.ones(B, T)
        # Half mask
        mask_half = torch.ones(B, T)
        mask_half[:, 5:] = 0

        logps_full = apo.compute_logps(logits, target_ids, mask_full)
        logps_half = apo.compute_logps(logits, target_ids, mask_half)

        # Length-normalized means dividing by number of valid tokens
        # So results should generally differ
        assert not torch.allclose(logps_full, logps_half), \
            "Different masks should give different length-normalized logps"

    def test_logps_shape(self):
        """Output should be [B]."""
        apo = self._make_alphapo()
        B, T, V = 3, 8, 50
        logits = torch.randn(B, T, V)
        target_ids = torch.randint(0, V, (B, T))
        mask = torch.ones(B, T)
        logps = apo.compute_logps(logits, target_ids, mask)
        assert logps.shape == (B,), f"Expected shape ({B},), got {logps.shape}"
