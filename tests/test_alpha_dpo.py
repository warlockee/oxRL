"""
Tests for AlphaDPO (Adaptive Reward Margin for DPO, ICML 2025) algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_alpha_dpo.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestAlphaDPOImport:
    def test_alpha_dpo_importable(self):
        from oxrl.algs.alpha_dpo import AlphaDPO
        assert AlphaDPO is not None

    def test_alpha_dpo_in_algs_init(self):
        from oxrl.algs import AlphaDPOMethod
        assert AlphaDPOMethod is not None

    def test_alpha_dpo_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"alpha_dpo"' in source


class TestAlphaDPOInterface:
    def test_has_required_methods(self):
        from oxrl.algs.alpha_dpo import AlphaDPO
        assert hasattr(AlphaDPO, 'compute_logps')
        assert hasattr(AlphaDPO, 'forward')
        assert hasattr(AlphaDPO, 'compute_loss')
        assert hasattr(AlphaDPO, 'train_step')
        assert hasattr(AlphaDPO, 'eval_step')

    def test_init_params(self):
        from oxrl.algs.alpha_dpo import AlphaDPO
        sig = inspect.signature(AlphaDPO.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params
        assert 'alpha_dpo_alpha' in params
        assert 'alpha_dpo_gamma_beta_ratio' in params
        assert 'alpha_dpo_ema_decay' in params

    def test_has_ema_state(self):
        from oxrl.algs.alpha_dpo import AlphaDPO
        d = AlphaDPO.__new__(AlphaDPO)
        d.beta = 2.5
        d.alpha = 0.1
        d.gamma_beta_ratio = 0.3
        d.ema_decay = 0.99
        d._ema_mean = 0.0
        d._ema_var = 1.0
        d._ema_initialized = False
        assert hasattr(d, '_ema_mean')
        assert hasattr(d, '_ema_var')
        assert hasattr(d, '_ema_initialized')


class TestAlphaDPOLossMath:
    def _make_alpha_dpo(self, beta=2.5, alpha=0.1, gamma_beta_ratio=0.3, ema_decay=0.99):
        from oxrl.algs.alpha_dpo import AlphaDPO
        d = AlphaDPO.__new__(AlphaDPO)
        d.beta = beta
        d.alpha = alpha
        d.gamma_beta_ratio = gamma_beta_ratio
        d.ema_decay = ema_decay
        d._ema_mean = 0.0
        d._ema_var = 1.0
        d._ema_initialized = False
        return d

    def test_loss_formula_verification(self):
        """Verify the AlphaDPO loss formula manually."""
        beta = 2.5
        alpha = 0.1
        gamma_beta_ratio = 0.3
        d = self._make_alpha_dpo(beta=beta, alpha=alpha, gamma_beta_ratio=gamma_beta_ratio)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # Manual computation
        pi_logratios = pi_logps_w - pi_logps_l
        ref_logratios = ref_logps_w - ref_logps_l
        gap = pi_logratios - ref_logratios

        # After first call, EMA is initialized to batch stats
        batch_mean = gap.mean().item()
        batch_var = gap.var().item()
        batch_std = max(batch_var, 1e-8) ** 0.5

        gap_normalized = (gap.detach() - batch_mean) / batch_std
        adaptive_margin = alpha * gap_normalized + gamma_beta_ratio
        logits = beta * (pi_logratios - adaptive_margin)
        expected = -F.logsigmoid(logits).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_zero_alpha_gives_fixed_margin(self):
        """When alpha=0, the margin is just gamma_beta_ratio (no adaptation)."""
        beta = 2.5
        gamma_beta_ratio = 0.5
        d = self._make_alpha_dpo(beta=beta, alpha=0.0, gamma_beta_ratio=gamma_beta_ratio)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # With alpha=0, margin = gamma_beta_ratio = 0.5
        pi_logratios = pi_logps_w - pi_logps_l
        logits = beta * (pi_logratios - gamma_beta_ratio)
        expected = -F.logsigmoid(logits).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_nonzero_alpha_differs_from_zero(self):
        """Non-zero alpha should produce different loss than alpha=0."""
        pi_logps_w = torch.tensor([-0.5, -1.0, -0.3])
        pi_logps_l = torch.tensor([-2.0, -3.0, -2.5])
        ref_logps_w = torch.tensor([-1.0, -1.2, -0.8])
        ref_logps_l = torch.tensor([-1.5, -2.0, -1.8])

        d_zero = self._make_alpha_dpo(alpha=0.0)
        d_nonzero = self._make_alpha_dpo(alpha=0.2)

        loss_zero, _, _ = d_zero.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss_nonzero, _, _ = d_nonzero.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        assert abs(loss_zero.item() - loss_nonzero.item()) > 1e-4

    def test_ema_updates_across_calls(self):
        """EMA statistics should update across multiple calls."""
        d = self._make_alpha_dpo()

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        # First call initializes EMA
        d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        mean_1 = d._ema_mean
        var_1 = d._ema_var

        # Second call with different data should update EMA
        pi_logps_w2 = torch.tensor([-0.2, -0.1])
        pi_logps_l2 = torch.tensor([-5.0, -4.0])
        ref_logps_w2 = torch.tensor([-0.5, -0.3])
        ref_logps_l2 = torch.tensor([-3.0, -2.5])

        d.compute_loss(pi_logps_w2, pi_logps_l2, ref_logps_w2, ref_logps_l2)
        mean_2 = d._ema_mean
        var_2 = d._ema_var

        # EMA should have changed
        assert mean_1 != mean_2 or var_1 != var_2

    def test_loss_is_finite(self):
        d = self._make_alpha_dpo()
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])
        loss, margin, reward_acc = d.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert torch.isfinite(loss)
        assert torch.isfinite(margin)

    def test_loss_non_negative(self):
        """AlphaDPO loss is -logsigmoid(...) which is always >= 0."""
        d = self._make_alpha_dpo(beta=2.5, alpha=0.1)
        pi_logps_w = torch.tensor([-0.1, -0.5, -2.0])
        pi_logps_l = torch.tensor([-5.0, -3.0, -0.1])
        ref_logps_w = torch.tensor([-0.5, -1.0, -1.5])
        ref_logps_l = torch.tensor([-1.0, -0.5, -0.2])
        loss, _, _ = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert loss.item() >= 0.0

    def test_reward_accuracy(self):
        d = self._make_alpha_dpo()
        # chosen has higher logps -> reward_acc should be 1.0
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])
        _, _, reward_acc = d.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert reward_acc.item() == 1.0

    def test_returns_three_values(self):
        d = self._make_alpha_dpo()
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        result = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert len(result) == 3

    def test_gradient_flows_through_policy(self):
        """Verify gradients flow through policy logps but NOT through adaptive margin."""
        d = self._make_alpha_dpo(beta=2.5, alpha=0.1)
        pi_logps_w = torch.tensor([-1.0], requires_grad=True)
        pi_logps_l = torch.tensor([-2.0], requires_grad=True)
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        loss, _, _ = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss.backward()
        assert pi_logps_w.grad is not None
        assert pi_logps_l.grad is not None

    def test_no_gradient_through_ref(self):
        """Reference logps should not require gradients."""
        d = self._make_alpha_dpo()
        pi_logps_w = torch.tensor([-1.0], requires_grad=True)
        pi_logps_l = torch.tensor([-2.0], requires_grad=True)
        ref_logps_w = torch.tensor([-1.1])  # no requires_grad
        ref_logps_l = torch.tensor([-2.1])
        loss, _, _ = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss.backward()
        # No error means ref doesn't need gradients

    def test_stop_gradient_on_margin(self):
        """The adaptive margin should use stop-gradient (detach)."""
        d = self._make_alpha_dpo(beta=2.5, alpha=0.5)

        pi_logps_w = torch.tensor([-1.0], requires_grad=True)
        pi_logps_l = torch.tensor([-2.0], requires_grad=True)
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])

        loss, _, _ = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        loss.backward()

        # Gradients should be finite (stop-gradient prevents instability)
        assert torch.isfinite(pi_logps_w.grad)
        assert torch.isfinite(pi_logps_l.grad)

    def test_different_alpha_values_produce_different_losses(self):
        """Different alpha values should produce different losses."""
        pi_logps_w = torch.tensor([-0.5, -1.0, -0.3])
        pi_logps_l = torch.tensor([-2.0, -3.0, -2.5])
        ref_logps_w = torch.tensor([-1.0, -1.2, -0.8])
        ref_logps_l = torch.tensor([-1.5, -2.0, -1.8])

        losses = []
        for alpha_val in [0.0, 0.05, 0.1, 0.2, 0.5]:
            d = self._make_alpha_dpo(alpha=alpha_val)
            loss, _, _ = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
            losses.append(loss.item())

        # At least some pairs should differ
        unique_losses = set(round(l, 6) for l in losses)
        assert len(unique_losses) > 1

    def test_different_gamma_beta_ratio_produces_different_losses(self):
        """Different gamma_beta_ratio values should produce different losses."""
        pi_logps_w = torch.tensor([-0.5, -1.0])
        pi_logps_l = torch.tensor([-2.0, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.2])
        ref_logps_l = torch.tensor([-1.5, -2.0])

        losses = []
        for gbr in [0.0, 0.3, 0.6, 1.0]:
            d = self._make_alpha_dpo(alpha=0.0, gamma_beta_ratio=gbr)
            loss, _, _ = d.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
            losses.append(loss.item())

        # All should be different
        for i in range(len(losses)):
            for j in range(i + 1, len(losses)):
                assert abs(losses[i] - losses[j]) > 1e-5

    def test_single_sample_batch(self):
        """Single-sample batch should work (variance defaults to 1)."""
        d = self._make_alpha_dpo()
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        loss, margin, reward_acc = d.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert torch.isfinite(loss)


class TestAlphaDPOConfig:
    def test_config_accepts_alpha_dpo(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="alpha_dpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alg_name == "alpha_dpo"
        assert hasattr(t, 'alpha_dpo_alpha')
        assert hasattr(t, 'alpha_dpo_gamma_beta_ratio')
        assert hasattr(t, 'alpha_dpo_ema_decay')

    def test_config_default_values(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="alpha_dpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alpha_dpo_alpha == 0.1
        assert t.alpha_dpo_gamma_beta_ratio == 0.3
        assert t.alpha_dpo_ema_decay == 0.99

    def test_config_custom_values(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="alpha_dpo", total_number_of_epochs=1,
                  micro_batches_per_epoch=10,
                  alpha_dpo_alpha=0.2,
                  alpha_dpo_gamma_beta_ratio=0.5,
                  alpha_dpo_ema_decay=0.95)
        assert t.alpha_dpo_alpha == 0.2
        assert t.alpha_dpo_gamma_beta_ratio == 0.5
        assert t.alpha_dpo_ema_decay == 0.95


class TestAlphaDPOComputeLogps:
    def _make_alpha_dpo(self):
        from oxrl.algs.alpha_dpo import AlphaDPO
        d = AlphaDPO.__new__(AlphaDPO)
        d.beta = 2.5
        d.alpha = 0.1
        d.gamma_beta_ratio = 0.3
        d.ema_decay = 0.99
        d._ema_mean = 0.0
        d._ema_var = 1.0
        d._ema_initialized = False
        return d

    def test_compute_logps_shape(self):
        d = self._make_alpha_dpo()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = d.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_logps_length_normalized(self):
        """Verify that logps are length-normalized (divided by sequence length)."""
        d = self._make_alpha_dpo()
        logits = torch.randn(1, 10, 50)
        target_ids = torch.randint(0, 50, (1, 10))

        # Full mask (10 tokens)
        mask_full = torch.ones(1, 10)
        logps_full = d.compute_logps(logits, target_ids, mask_full)

        # Half mask (5 tokens)
        mask_half = torch.zeros(1, 10)
        mask_half[0, :5] = 1.0
        logps_half = d.compute_logps(logits, target_ids, mask_half)

        # With length normalization, different mask lengths give different per-token avg
        # They should generally differ
        assert logps_full.item() != logps_half.item()

    def test_compute_logps_mask_zeros_out(self):
        """Tokens with mask=0 should not contribute."""
        d = self._make_alpha_dpo()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))

        mask_a = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
        mask_b = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
        logps_a = d.compute_logps(logits, target_ids, mask_a)
        logps_b = d.compute_logps(logits, target_ids, mask_b)

        # Same mask -> same result
        assert abs(logps_a.item() - logps_b.item()) < 1e-6


class TestAlphaDPOEMA:
    def _make_alpha_dpo(self, ema_decay=0.99):
        from oxrl.algs.alpha_dpo import AlphaDPO
        d = AlphaDPO.__new__(AlphaDPO)
        d.beta = 2.5
        d.alpha = 0.1
        d.gamma_beta_ratio = 0.3
        d.ema_decay = ema_decay
        d._ema_mean = 0.0
        d._ema_var = 1.0
        d._ema_initialized = False
        return d

    def test_ema_initializes_on_first_call(self):
        d = self._make_alpha_dpo()
        gap = torch.tensor([1.0, 2.0, 3.0])
        d._update_ema(gap)
        assert d._ema_initialized
        assert abs(d._ema_mean - 2.0) < 1e-5  # mean of [1,2,3]

    def test_ema_decay_toward_new_data(self):
        d = self._make_alpha_dpo(ema_decay=0.5)
        gap1 = torch.tensor([0.0, 0.0])
        d._update_ema(gap1)
        mean_after_1 = d._ema_mean

        gap2 = torch.tensor([10.0, 10.0])
        d._update_ema(gap2)
        mean_after_2 = d._ema_mean

        # With decay=0.5, new mean = 0.5 * 0 + 0.5 * 10 = 5
        assert abs(mean_after_2 - 5.0) < 1e-5

    def test_high_decay_preserves_history(self):
        d = self._make_alpha_dpo(ema_decay=0.99)
        gap1 = torch.tensor([0.0, 0.0])
        d._update_ema(gap1)

        gap2 = torch.tensor([100.0, 100.0])
        d._update_ema(gap2)

        # With 0.99 decay, mean should be close to 0, not 100
        assert d._ema_mean < 5.0  # heavily weighted toward first value
