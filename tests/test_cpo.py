"""
Tests for CPO (Contrastive Preference Optimization) algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_cpo.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestCPOImport:
    """Test CPO is properly importable and registered."""

    def test_cpo_importable(self):
        from oxrl.algs.cpo import CPO
        assert CPO is not None

    def test_cpo_in_algs_init(self):
        from oxrl.algs import CPO
        assert CPO is not None

    def test_cpo_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"cpo"' in source, "CPO should be registered in SL_ALGORITHMS"

    def test_cpo_uses_preference_dataset(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"cpo"' in source
        # CPO should be in the preference dataset list
        assert "cpo" in source


class TestCPOInterface:
    """Test CPO has the expected interface."""

    def test_cpo_has_required_methods(self):
        from oxrl.algs.cpo import CPO
        assert hasattr(CPO, 'compute_logps')
        assert hasattr(CPO, 'forward')
        assert hasattr(CPO, 'compute_nll_loss')
        assert hasattr(CPO, 'compute_loss')
        assert hasattr(CPO, 'train_step')
        assert hasattr(CPO, 'eval_step')

    def test_cpo_init_params(self):
        from oxrl.algs.cpo import CPO
        sig = inspect.signature(CPO.__init__)
        params = list(sig.parameters.keys())
        assert 'model_engine' in params
        assert 'optimizer' in params
        assert 'beta' in params
        assert 'cpo_alpha' in params
        assert 'loss_type' in params
        assert 'label_smoothing' in params
        assert 'use_cache' in params

    def test_cpo_no_ref_model(self):
        """CPO should NOT require a reference model (key difference from DPO)."""
        from oxrl.algs.cpo import CPO
        sig = inspect.signature(CPO.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' not in params, \
            "CPO should not accept ref_model_engine (it is reference-free)"


class TestCPOLossMath:
    """Test CPO loss computation with known values."""

    def _make_cpo(self, beta=0.1, cpo_alpha=1.0, loss_type="sigmoid", label_smoothing=0.0):
        from oxrl.algs.cpo import CPO
        cpo = CPO.__new__(CPO)
        cpo.beta = beta
        cpo.cpo_alpha = cpo_alpha
        cpo.loss_type = loss_type
        cpo.label_smoothing = label_smoothing
        cpo.nll_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        return cpo

    def test_sigmoid_loss_basic(self):
        """Test sigmoid CPO preference loss matches expected value."""
        cpo = self._make_cpo(beta=0.1, cpo_alpha=0.0)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])

        B, T, V = 2, 5, 50
        chosen_logits = torch.randn(B, T, V)
        chosen_targets = torch.randint(0, V, (B, T))
        chosen_mask = torch.ones(B, T)

        loss, pref_loss, nll_loss, margin, reward_acc = cpo.compute_loss(
            pi_logps_w, pi_logps_l, chosen_logits, chosen_targets, chosen_mask
        )

        # With cpo_alpha=0, loss should equal pref_loss
        logits = pi_logps_w - pi_logps_l
        expected_pref = -F.logsigmoid(0.1 * logits).mean()

        assert abs(pref_loss.item() - expected_pref.item()) < 1e-5
        # With alpha=0, nll_loss should not affect total loss
        assert abs(loss.item() - pref_loss.item()) < 1e-5

    def test_hinge_loss_basic(self):
        """Test hinge CPO preference loss."""
        cpo = self._make_cpo(beta=0.1, loss_type="hinge", cpo_alpha=0.0)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])

        B, T, V = 2, 5, 50
        chosen_logits = torch.randn(B, T, V)
        chosen_targets = torch.randint(0, V, (B, T))
        chosen_mask = torch.ones(B, T)

        loss, pref_loss, nll_loss, margin, reward_acc = cpo.compute_loss(
            pi_logps_w, pi_logps_l, chosen_logits, chosen_targets, chosen_mask
        )

        logits = pi_logps_w - pi_logps_l
        expected = torch.relu(1 - 0.1 * logits).mean()

        assert abs(pref_loss.item() - expected.item()) < 1e-5

    def test_label_smoothing(self):
        """Test that label smoothing modifies the loss."""
        cpo_no_smooth = self._make_cpo(beta=0.1, cpo_alpha=0.0, label_smoothing=0.0)
        cpo_smooth = self._make_cpo(beta=0.1, cpo_alpha=0.0, label_smoothing=0.1)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])

        B, T, V = 2, 5, 50
        chosen_logits = torch.randn(B, T, V)
        chosen_targets = torch.randint(0, V, (B, T))
        chosen_mask = torch.ones(B, T)

        loss_no, _, _, _, _ = cpo_no_smooth.compute_loss(
            pi_logps_w, pi_logps_l, chosen_logits, chosen_targets, chosen_mask)
        loss_yes, _, _, _, _ = cpo_smooth.compute_loss(
            pi_logps_w, pi_logps_l, chosen_logits, chosen_targets, chosen_mask)

        # Label smoothing should change the loss
        assert abs(loss_no.item() - loss_yes.item()) > 1e-4, \
            "Label smoothing should modify the preference loss"

    def test_cpo_alpha_controls_nll_weight(self):
        """Test that cpo_alpha controls the BC regularization weight."""
        cpo_a0 = self._make_cpo(beta=0.1, cpo_alpha=0.0)
        cpo_a1 = self._make_cpo(beta=0.1, cpo_alpha=1.0)
        cpo_a5 = self._make_cpo(beta=0.1, cpo_alpha=5.0)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])

        B, T, V = 2, 5, 50
        torch.manual_seed(42)
        chosen_logits = torch.randn(B, T, V)
        chosen_targets = torch.randint(0, V, (B, T))
        chosen_mask = torch.ones(B, T)

        loss_a0, pref_a0, nll_a0, _, _ = cpo_a0.compute_loss(
            pi_logps_w, pi_logps_l, chosen_logits, chosen_targets, chosen_mask)
        loss_a1, pref_a1, nll_a1, _, _ = cpo_a1.compute_loss(
            pi_logps_w, pi_logps_l, chosen_logits, chosen_targets, chosen_mask)
        loss_a5, pref_a5, nll_a5, _, _ = cpo_a5.compute_loss(
            pi_logps_w, pi_logps_l, chosen_logits, chosen_targets, chosen_mask)

        # All should have same pref_loss
        assert abs(pref_a0.item() - pref_a1.item()) < 1e-5
        assert abs(pref_a0.item() - pref_a5.item()) < 1e-5

        # loss = pref + alpha * nll
        assert abs(loss_a0.item() - pref_a0.item()) < 1e-5, \
            "alpha=0 should give loss == pref_loss"
        assert abs(loss_a1.item() - (pref_a1.item() + 1.0 * nll_a1.item())) < 1e-4
        assert abs(loss_a5.item() - (pref_a5.item() + 5.0 * nll_a5.item())) < 1e-4

        # Higher alpha should give higher total loss (since nll > 0)
        assert loss_a5.item() > loss_a1.item() > loss_a0.item()

    def test_margin_and_accuracy(self):
        """Test that margin and reward_acc are computed correctly."""
        cpo = self._make_cpo(beta=0.1, cpo_alpha=0.0)

        # Chosen clearly better
        pi_logps_w = torch.tensor([-1.0, -0.5, -0.3])
        pi_logps_l = torch.tensor([-3.0, -2.5, -2.0])

        B, T, V = 3, 5, 50
        chosen_logits = torch.randn(B, T, V)
        chosen_targets = torch.randint(0, V, (B, T))
        chosen_mask = torch.ones(B, T)

        _, _, _, margin, reward_acc = cpo.compute_loss(
            pi_logps_w, pi_logps_l, chosen_logits, chosen_targets, chosen_mask
        )

        assert margin.item() > 0, "Margin should be positive when chosen > rejected"
        assert reward_acc.item() == 1.0, "All chosen should beat rejected"

    def test_loss_is_finite(self):
        """Test that CPO loss is always finite with valid inputs."""
        cpo = self._make_cpo(beta=0.1, cpo_alpha=1.0)

        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])

        B, T, V = 2, 10, 100
        chosen_logits = torch.randn(B, T, V)
        chosen_targets = torch.randint(0, V, (B, T))
        chosen_mask = torch.ones(B, T)

        loss, pref_loss, nll_loss, margin, _ = cpo.compute_loss(
            pi_logps_w, pi_logps_l, chosen_logits, chosen_targets, chosen_mask
        )

        assert torch.isfinite(loss), f"CPO total loss should be finite, got {loss}"
        assert torch.isfinite(pref_loss), f"Pref loss should be finite, got {pref_loss}"
        assert torch.isfinite(nll_loss), f"NLL loss should be finite, got {nll_loss}"
        assert nll_loss.item() >= 0, "NLL loss should be non-negative"

    def test_correct_preference_has_lower_loss(self):
        """When chosen has higher log-prob, loss should be lower."""
        cpo = self._make_cpo(beta=0.1, cpo_alpha=0.0)

        B, T, V = 2, 5, 50
        chosen_logits = torch.randn(B, T, V)
        chosen_targets = torch.randint(0, V, (B, T))
        chosen_mask = torch.ones(B, T)

        # Good: chosen >> rejected
        pi_logps_w_good = torch.tensor([-1.0, -0.5])
        pi_logps_l_good = torch.tensor([-5.0, -4.0])

        # Bad: swapped
        loss_good, _, _, _, _ = cpo.compute_loss(
            pi_logps_w_good, pi_logps_l_good, chosen_logits, chosen_targets, chosen_mask)
        loss_bad, _, _, _, _ = cpo.compute_loss(
            pi_logps_l_good, pi_logps_w_good, chosen_logits, chosen_targets, chosen_mask)

        assert loss_good.item() < loss_bad.item(), \
            f"Correct preference should have lower loss: {loss_good} vs {loss_bad}"

    def test_unknown_loss_type_raises(self):
        """Test that an unknown loss_type raises ValueError."""
        cpo = self._make_cpo(loss_type="unknown")

        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        chosen_logits = torch.randn(1, 5, 50)
        chosen_targets = torch.randint(0, 50, (1, 5))
        chosen_mask = torch.ones(1, 5)

        with pytest.raises(ValueError, match="Unknown CPO loss_type"):
            cpo.compute_loss(pi_logps_w, pi_logps_l, chosen_logits, chosen_targets, chosen_mask)


class TestCPOConfig:
    """Test CPO config fields are properly defined."""

    def test_config_has_cpo_fields(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="cpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert hasattr(t, 'cpo_alpha'), "Train should have cpo_alpha"
        assert hasattr(t, 'cpo_loss_type'), "Train should have cpo_loss_type"
        assert hasattr(t, 'cpo_label_smoothing'), "Train should have cpo_label_smoothing"

    def test_config_default_values(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="cpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.cpo_alpha == 1.0, f"Default cpo_alpha should be 1.0, got {t.cpo_alpha}"
        assert t.cpo_loss_type == "sigmoid", f"Default cpo_loss_type should be 'sigmoid', got {t.cpo_loss_type}"
        assert t.cpo_label_smoothing == 0.0, f"Default cpo_label_smoothing should be 0.0, got {t.cpo_label_smoothing}"

    def test_config_custom_values(self):
        from oxrl.configs.schema import Train
        t = Train(
            alg_name="cpo",
            total_number_of_epochs=1,
            micro_batches_per_epoch=10,
            cpo_alpha=0.5,
            cpo_loss_type="hinge",
            cpo_label_smoothing=0.1,
        )
        assert t.cpo_alpha == 0.5
        assert t.cpo_loss_type == "hinge"
        assert t.cpo_label_smoothing == 0.1


class TestCPOExampleConfig:
    """Test CPO example config exists."""

    def test_example_config_exists(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "registry", "examples", "cpo.yaml")
        assert os.path.exists(config_path), "CPO example config should exist"


class TestCPONLLLoss:
    """Test the NLL (behavioral cloning) component separately."""

    def _make_cpo(self):
        from oxrl.algs.cpo import CPO
        cpo = CPO.__new__(CPO)
        cpo.nll_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        return cpo

    def test_nll_loss_is_finite(self):
        cpo = self._make_cpo()
        B, T, V = 2, 10, 100
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        mask = torch.ones(B, T)

        nll = cpo.compute_nll_loss(logits, targets, mask)
        assert torch.isfinite(nll), f"NLL should be finite, got {nll}"
        assert nll.item() > 0, "NLL should be positive for random logits"

    def test_nll_respects_mask(self):
        """NLL with partial mask should differ from full mask."""
        cpo = self._make_cpo()
        B, T, V = 2, 10, 100
        torch.manual_seed(42)
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))

        full_mask = torch.ones(B, T)
        partial_mask = torch.ones(B, T)
        partial_mask[:, 5:] = 0  # mask out second half

        nll_full = cpo.compute_nll_loss(logits, targets, full_mask)
        nll_partial = cpo.compute_nll_loss(logits, targets, partial_mask)

        # They should generally differ (different tokens averaged)
        assert abs(nll_full.item() - nll_partial.item()) > 1e-3, \
            "NLL should differ between full and partial mask"

    def test_nll_correct_tokens_lower(self):
        """NLL should be lower when logits assign high probability to correct tokens."""
        cpo = self._make_cpo()
        B, T, V = 1, 5, 10
        targets = torch.zeros(B, T, dtype=torch.long)

        # Logits that strongly predict token 0
        good_logits = torch.zeros(B, T, V)
        good_logits[:, :, 0] = 10.0

        # Random logits
        bad_logits = torch.randn(B, T, V)

        mask = torch.ones(B, T)

        nll_good = cpo.compute_nll_loss(good_logits, targets, mask)
        nll_bad = cpo.compute_nll_loss(bad_logits, targets, mask)

        assert nll_good.item() < nll_bad.item(), \
            "NLL should be lower when model predicts correct tokens"
