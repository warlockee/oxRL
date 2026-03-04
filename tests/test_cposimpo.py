"""
Tests for CPO-SimPO (Contrastive Preference Optimization + SimPO) algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_cposimpo.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestCPOSimPOImport:
    def test_cposimpo_importable(self):
        from oxrl.algs.cposimpo import CPOSimPO
        assert CPOSimPO is not None

    def test_cposimpo_in_algs_init(self):
        from oxrl.algs import CPOSimPO
        assert CPOSimPO is not None

    def test_cposimpo_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"cposimpo"' in source


class TestCPOSimPOInterface:
    def test_has_required_methods(self):
        from oxrl.algs.cposimpo import CPOSimPO
        assert hasattr(CPOSimPO, 'compute_logps')
        assert hasattr(CPOSimPO, 'forward')
        assert hasattr(CPOSimPO, 'compute_loss')
        assert hasattr(CPOSimPO, 'compute_nll_loss')
        assert hasattr(CPOSimPO, 'train_step')
        assert hasattr(CPOSimPO, 'eval_step')

    def test_init_params(self):
        from oxrl.algs.cposimpo import CPOSimPO
        sig = inspect.signature(CPOSimPO.__init__)
        params = list(sig.parameters.keys())
        assert 'model_engine' in params
        assert 'optimizer' in params
        assert 'beta' in params
        assert 'gamma' in params
        assert 'cposimpo_alpha' in params

    def test_no_ref_model_needed(self):
        """CPO-SimPO is reference-free."""
        from oxrl.algs.cposimpo import CPOSimPO
        sig = inspect.signature(CPOSimPO.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' not in params


class TestCPOSimPOLossMath:
    def _make_cposimpo(self, beta=2.0, gamma=0.5, alpha=1.0):
        from oxrl.algs.cposimpo import CPOSimPO
        c = CPOSimPO.__new__(CPOSimPO)
        c.beta = beta
        c.gamma = gamma
        c.cposimpo_alpha = alpha
        c.nll_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        return c

    def test_pref_loss_formula(self):
        """Verify: L_pref = -logsigmoid(beta*(avg_logp_w - avg_logp_l) - gamma)."""
        beta = 2.0
        gamma = 0.5
        c = self._make_cposimpo(beta=beta, gamma=gamma, alpha=0.0)

        # Length-normalized logps (already avg logps)
        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])

        # Create dummy logits/targets/mask for NLL (alpha=0 so NLL won't matter)
        dummy_logits = torch.randn(2, 5, 100)
        dummy_targets = torch.randint(0, 100, (2, 5))
        dummy_mask = torch.ones(2, 5)

        total_loss, pref_loss, nll_loss, _, _ = c.compute_loss(
            pi_logps_w, pi_logps_l, dummy_logits, dummy_targets, dummy_mask)

        logits = beta * (pi_logps_w - pi_logps_l) - gamma
        expected_pref = -F.logsigmoid(logits).mean()

        assert abs(pref_loss.item() - expected_pref.item()) < 1e-5
        # With alpha=0, total_loss == pref_loss
        assert abs(total_loss.item() - pref_loss.item()) < 1e-5

    def test_reduces_to_simpo_when_alpha_zero(self):
        """When alpha=0, CPO-SimPO is equivalent to SimPO."""
        beta = 2.0
        gamma = 0.5
        c = self._make_cposimpo(beta=beta, gamma=gamma, alpha=0.0)

        pi_logps_w = torch.tensor([-0.8, -0.3])
        pi_logps_l = torch.tensor([-2.5, -1.8])

        dummy_logits = torch.randn(2, 5, 100)
        dummy_targets = torch.randint(0, 100, (2, 5))
        dummy_mask = torch.ones(2, 5)

        total_loss, pref_loss, nll_loss, _, _ = c.compute_loss(
            pi_logps_w, pi_logps_l, dummy_logits, dummy_targets, dummy_mask)

        # SimPO loss
        simpo_logits = beta * (pi_logps_w - pi_logps_l) - gamma
        simpo_loss = -F.logsigmoid(simpo_logits).mean()

        assert abs(total_loss.item() - simpo_loss.item()) < 1e-5

    def test_nll_added_when_alpha_positive(self):
        """With alpha>0, NLL component should increase total loss."""
        beta = 2.0
        gamma = 0.5
        c_no_nll = self._make_cposimpo(beta=beta, gamma=gamma, alpha=0.0)
        c_with_nll = self._make_cposimpo(beta=beta, gamma=gamma, alpha=1.0)

        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])

        # Use meaningful logits for NLL
        chosen_logits = torch.randn(1, 5, 100)
        chosen_targets = torch.randint(0, 100, (1, 5))
        chosen_mask = torch.ones(1, 5)

        loss_no_nll, _, _, _, _ = c_no_nll.compute_loss(
            pi_logps_w, pi_logps_l, chosen_logits, chosen_targets, chosen_mask)
        loss_with_nll, _, nll_loss, _, _ = c_with_nll.compute_loss(
            pi_logps_w, pi_logps_l, chosen_logits, chosen_targets, chosen_mask)

        # Total loss should be higher when alpha>0 (NLL is always positive)
        assert loss_with_nll.item() > loss_no_nll.item()
        assert nll_loss.item() > 0.0

    def test_alpha_scales_nll(self):
        """Different alpha values should scale the NLL contribution."""
        beta = 2.0
        gamma = 0.5

        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        chosen_logits = torch.randn(1, 5, 100)
        chosen_targets = torch.randint(0, 100, (1, 5))
        chosen_mask = torch.ones(1, 5)

        losses = []
        for alpha_val in [0.0, 0.5, 1.0, 2.0]:
            c = self._make_cposimpo(beta=beta, gamma=gamma, alpha=alpha_val)
            loss, _, _, _, _ = c.compute_loss(
                pi_logps_w, pi_logps_l, chosen_logits, chosen_targets, chosen_mask)
            losses.append(loss.item())

        # Losses should increase with alpha
        for i in range(len(losses) - 1):
            assert losses[i] < losses[i + 1]

    def test_nll_loss_computation(self):
        """Verify NLL loss matches manual cross-entropy calculation."""
        c = self._make_cposimpo()
        V = 10
        logits = torch.randn(1, 3, V)
        targets = torch.randint(0, V, (1, 3))
        mask = torch.ones(1, 3)

        nll = c.compute_nll_loss(logits, targets, mask)

        # Manual: cross-entropy per token, then mean
        expected = F.cross_entropy(logits.view(-1, V), targets.view(-1))
        assert abs(nll.item() - expected.item()) < 1e-5

    def test_nll_loss_respects_mask(self):
        """NLL loss should only compute over masked tokens."""
        c = self._make_cposimpo()
        V = 10
        logits = torch.randn(1, 4, V)
        targets = torch.randint(0, V, (1, 4))
        mask_full = torch.ones(1, 4)
        mask_partial = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

        nll_full = c.compute_nll_loss(logits, targets, mask_full)
        nll_partial = c.compute_nll_loss(logits, targets, mask_partial)

        # Different masks should give different NLL
        assert abs(nll_full.item() - nll_partial.item()) > 1e-5

    def test_loss_is_finite(self):
        c = self._make_cposimpo(beta=2.0, gamma=0.5, alpha=1.0)
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        chosen_logits = torch.randn(2, 5, 50)
        chosen_targets = torch.randint(0, 50, (2, 5))
        chosen_mask = torch.ones(2, 5)
        total, pref, nll, margin, racc = c.compute_loss(
            pi_logps_w, pi_logps_l, chosen_logits, chosen_targets, chosen_mask)
        assert torch.isfinite(total)
        assert torch.isfinite(pref)
        assert torch.isfinite(nll)
        assert torch.isfinite(margin)

    def test_reward_accuracy(self):
        c = self._make_cposimpo(beta=2.0, gamma=0.5, alpha=0.0)
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        dummy_logits = torch.randn(3, 5, 50)
        dummy_targets = torch.randint(0, 50, (3, 5))
        dummy_mask = torch.ones(3, 5)
        _, _, _, _, reward_acc = c.compute_loss(
            pi_logps_w, pi_logps_l, dummy_logits, dummy_targets, dummy_mask)
        assert reward_acc.item() == 1.0

    def test_returns_five_values(self):
        c = self._make_cposimpo()
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        chosen_logits = torch.randn(1, 5, 50)
        chosen_targets = torch.randint(0, 50, (1, 5))
        chosen_mask = torch.ones(1, 5)
        result = c.compute_loss(
            pi_logps_w, pi_logps_l, chosen_logits, chosen_targets, chosen_mask)
        assert len(result) == 5

    def test_gradient_flows(self):
        """Verify gradients flow through both loss components."""
        c = self._make_cposimpo(beta=2.0, gamma=0.5, alpha=1.0)
        pi_logps_w = torch.tensor([-1.0], requires_grad=True)
        pi_logps_l = torch.tensor([-2.0], requires_grad=True)
        chosen_logits = torch.randn(1, 5, 50, requires_grad=True)
        chosen_targets = torch.randint(0, 50, (1, 5))
        chosen_mask = torch.ones(1, 5)
        total, _, _, _, _ = c.compute_loss(
            pi_logps_w, pi_logps_l, chosen_logits, chosen_targets, chosen_mask)
        total.backward()
        assert pi_logps_w.grad is not None
        assert pi_logps_l.grad is not None
        assert chosen_logits.grad is not None


class TestCPOSimPOConfig:
    def test_config_accepts_cposimpo(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="cposimpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alg_name == "cposimpo"
        assert hasattr(t, 'cposimpo_alpha')

    def test_config_default_value(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="cposimpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.cposimpo_alpha == 1.0

    def test_config_custom_value(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="cposimpo", total_number_of_epochs=1,
                  micro_batches_per_epoch=10, cposimpo_alpha=0.5)
        assert t.cposimpo_alpha == 0.5


class TestCPOSimPOComputeLogps:
    def _make_cposimpo(self):
        from oxrl.algs.cposimpo import CPOSimPO
        c = CPOSimPO.__new__(CPOSimPO)
        c.beta = 2.0
        c.gamma = 0.5
        c.cposimpo_alpha = 1.0
        c.nll_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        return c

    def test_compute_logps_shape(self):
        c = self._make_cposimpo()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = c.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_logps_length_normalized(self):
        """Verify logps are length-normalized (divided by sequence length)."""
        c = self._make_cposimpo()
        logits = torch.randn(1, 10, 100)
        target_ids = torch.randint(0, 100, (1, 10))

        mask_5 = torch.tensor([[1.0] * 5 + [0.0] * 5])
        mask_10 = torch.ones(1, 10)

        logps_5 = c.compute_logps(logits, target_ids, mask_5)
        logps_10 = c.compute_logps(logits, target_ids, mask_10)

        # Should be different due to both different tokens and normalization
        assert logps_5.item() != logps_10.item()

    def test_compute_logps_mask(self):
        c = self._make_cposimpo()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        mask_full = torch.ones(1, 5)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        logps_full = c.compute_logps(logits, target_ids, mask_full)
        logps_half = c.compute_logps(logits, target_ids, mask_half)
        assert logps_full.item() != logps_half.item()
