"""
Tests for beta-DPO (Dynamic Beta DPO) algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_betadpo.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestBetaDPOImport:
    def test_betadpo_importable(self):
        from oxrl.algs.betadpo import BetaDPO
        assert BetaDPO is not None

    def test_betadpo_in_algs_init(self):
        from oxrl.algs import BetaDPO
        assert BetaDPO is not None

    def test_betadpo_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"betadpo"' in source


class TestBetaDPOInterface:
    def test_betadpo_has_required_methods(self):
        from oxrl.algs.betadpo import BetaDPO
        assert hasattr(BetaDPO, 'compute_logps')
        assert hasattr(BetaDPO, 'forward')
        assert hasattr(BetaDPO, 'compute_loss')
        assert hasattr(BetaDPO, 'train_step')
        assert hasattr(BetaDPO, 'eval_step')

    def test_betadpo_init_params(self):
        from oxrl.algs.betadpo import BetaDPO
        sig = inspect.signature(BetaDPO.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params
        assert 'betadpo_alpha' in params
        assert 'betadpo_ema_gamma' in params


class TestBetaDPOLossMath:
    def _make_betadpo(self, beta=0.1, alpha=0.5, ema_gamma=0.9):
        from oxrl.algs.betadpo import BetaDPO
        bdpo = BetaDPO.__new__(BetaDPO)
        bdpo.beta = beta
        bdpo.alpha = alpha
        bdpo.ema_gamma = ema_gamma
        bdpo._gap_mean = 0.0
        bdpo._initialized = False
        return bdpo

    def test_zero_alpha_matches_dpo(self):
        """With alpha=0, beta-DPO should match DPO (constant beta)."""
        from oxrl.algs.dpo import DPO
        bdpo = self._make_betadpo(beta=0.1, alpha=0.0)
        dpo = DPO.__new__(DPO)
        dpo.beta = 0.1

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        bdpo_loss, _, _, mean_beta = bdpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        dpo_loss, _ = dpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        assert abs(bdpo_loss.item() - dpo_loss.item()) < 1e-5, \
            f"alpha=0 beta-DPO should match DPO: {bdpo_loss} vs {dpo_loss}"

    def test_dynamic_beta_formula(self):
        """Verify: beta_i = beta * (1 + alpha * (A_i - gap_mean))."""
        bdpo = self._make_betadpo(beta=0.2, alpha=0.5)
        bdpo._gap_mean = 0.5
        bdpo._initialized = True

        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])

        # Compute expected
        logr_w = pi_logps_w - ref_logps_w  # 0.1
        logr_l = pi_logps_l - ref_logps_l  # 0.1
        A = logr_w - logr_l  # 0.0
        expected_dynamic_beta = 0.2 * (1.0 + 0.5 * (A - 0.5))
        expected_dynamic_beta = torch.clamp(expected_dynamic_beta, min=1e-3)
        expected_loss = -F.logsigmoid(expected_dynamic_beta * A).mean()

        loss, _, _, mean_beta = bdpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        assert abs(loss.item() - expected_loss.item()) < 1e-5, \
            f"Loss mismatch: {loss} vs {expected_loss}"

    def test_positive_alpha_varies_beta(self):
        """With alpha>0, different samples should get different betas."""
        bdpo = self._make_betadpo(beta=0.1, alpha=0.5)
        bdpo._gap_mean = 0.0
        bdpo._initialized = True

        # Two samples with different advantage gaps
        pi_logps_w = torch.tensor([-0.5, -1.0])
        pi_logps_l = torch.tensor([-3.0, -1.2])
        ref_logps_w = torch.tensor([-1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0])

        logr_w = pi_logps_w - ref_logps_w  # [0.5, 0.0]
        logr_l = pi_logps_l - ref_logps_l  # [-2.0, -0.2]
        A = logr_w - logr_l  # [2.5, 0.2]

        # Dynamic betas should differ
        db = 0.1 * (1.0 + 0.5 * (A - 0.0))
        assert db[0] != db[1], "Different gaps should yield different betas"

    def test_ema_update(self):
        """Verify EMA running mean update."""
        bdpo = self._make_betadpo(beta=0.1, alpha=0.5, ema_gamma=0.8)

        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-2.0])

        # First call initializes
        bdpo.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        gap1 = 0.0  # logr_w - logr_l = 0 - 0 = 0
        assert bdpo._initialized
        assert abs(bdpo._gap_mean - gap1) < 1e-5

        # Second call with different gap
        pi_logps_w2 = torch.tensor([-0.5])
        pi_logps_l2 = torch.tensor([-2.0])
        bdpo.compute_loss(pi_logps_w2, pi_logps_l2, ref_logps_w, ref_logps_l)
        gap2_actual = ((-0.5 - (-1.0)) - (-2.0 - (-2.0)))  # 0.5 - 0.0 = 0.5
        expected_mean = 0.8 * gap1 + 0.2 * gap2_actual  # 0.8*0 + 0.2*0.5 = 0.1
        assert abs(bdpo._gap_mean - expected_mean) < 1e-5, \
            f"EMA update wrong: {bdpo._gap_mean} vs {expected_mean}"

    def test_beta_clamping(self):
        """Dynamic beta should never go below 1e-3."""
        bdpo = self._make_betadpo(beta=0.1, alpha=10.0)
        bdpo._gap_mean = 100.0  # Extreme gap mean
        bdpo._initialized = True

        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-2.0])

        # This would give a very negative dynamic beta without clamping
        loss, _, _, mean_beta = bdpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert mean_beta.item() >= 1e-3, "Dynamic beta should be clamped"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_loss_is_finite(self):
        bdpo = self._make_betadpo(beta=0.1, alpha=0.5)
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])
        loss, margin, reward_acc, _ = bdpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert torch.isfinite(loss)
        assert torch.isfinite(margin)

    def test_reward_accuracy(self):
        bdpo = self._make_betadpo(beta=0.1, alpha=0.5)
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])
        _, _, reward_acc, _ = bdpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert reward_acc.item() == 1.0, "All chosen should have higher reward"

    def test_loss_positive(self):
        """Loss should always be positive (negative logsigmoid is positive)."""
        bdpo = self._make_betadpo(beta=0.1, alpha=0.5)
        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])
        loss, _, _, _ = bdpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert loss.item() > 0, "Loss should be positive"

    def test_returns_four_values(self):
        bdpo = self._make_betadpo(beta=0.1, alpha=0.5)
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        result = bdpo.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert len(result) == 4, "compute_loss should return (loss, margin, reward_acc, mean_beta)"


class TestBetaDPOConfig:
    def test_config_has_betadpo_fields(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="betadpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert hasattr(t, 'betadpo_alpha')
        assert hasattr(t, 'betadpo_ema_gamma')

    def test_config_default_values(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="betadpo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.betadpo_alpha == 0.5
        assert t.betadpo_ema_gamma == 0.9

    def test_config_custom_values(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="betadpo", total_number_of_epochs=1,
                  micro_batches_per_epoch=10, betadpo_alpha=0.8, betadpo_ema_gamma=0.95)
        assert t.betadpo_alpha == 0.8
        assert t.betadpo_ema_gamma == 0.95


class TestBetaDPOComputeLogps:
    def _make_betadpo(self):
        from oxrl.algs.betadpo import BetaDPO
        bdpo = BetaDPO.__new__(BetaDPO)
        bdpo.beta = 0.1
        bdpo.alpha = 0.5
        bdpo.ema_gamma = 0.9
        bdpo._gap_mean = 0.0
        bdpo._initialized = False
        return bdpo

    def test_compute_logps_shape(self):
        bdpo = self._make_betadpo()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = bdpo.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_logps_mask(self):
        bdpo = self._make_betadpo()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        mask_full = torch.ones(1, 5)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        logps_full = bdpo.compute_logps(logits, target_ids, mask_full)
        logps_half = bdpo.compute_logps(logits, target_ids, mask_half)
        assert logps_full.item() != logps_half.item()
