"""
Tests for EXO (Efficient Exact Optimization) preference optimization algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_exo.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestEXOImport:
    def test_exo_importable(self):
        from oxrl.algs.exo import EXO
        assert EXO is not None

    def test_exo_in_algs_init(self):
        from oxrl.algs import EXO
        assert EXO is not None

    def test_exo_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"exo"' in source


class TestEXOInterface:
    def test_has_required_methods(self):
        from oxrl.algs.exo import EXO
        assert hasattr(EXO, 'compute_logps')
        assert hasattr(EXO, 'forward')
        assert hasattr(EXO, 'compute_loss')
        assert hasattr(EXO, 'train_step')
        assert hasattr(EXO, 'eval_step')

    def test_init_params(self):
        from oxrl.algs.exo import EXO
        sig = inspect.signature(EXO.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params
        assert 'exo_epsilon' in params


class TestEXOLossMath:
    def _make_exo(self, beta=0.1, exo_epsilon=1e-3):
        from oxrl.algs.exo import EXO
        e = EXO.__new__(EXO)
        e.beta = beta
        e.exo_epsilon = exo_epsilon
        return e

    def test_loss_formula_verification(self):
        """Verify: L = q_w*(log(q_w)-log(p_w)) + q_l*(log(q_l)-log(p_l))."""
        beta = 0.2
        eps = 0.01
        e = self._make_exo(beta=beta, exo_epsilon=eps)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = e.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        delta = beta * (logr_w - logr_l)

        q_w = torch.sigmoid(delta)
        log_q_w = F.logsigmoid(delta)
        q_l = torch.sigmoid(-delta)
        log_q_l = F.logsigmoid(-delta)
        log_p_w = torch.log(torch.tensor(1.0 - eps))
        log_p_l = torch.log(torch.tensor(eps))

        expected = (q_w * (log_q_w - log_p_w) + q_l * (log_q_l - log_p_l)).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_loss_always_non_negative(self):
        """Reverse KL is always >= 0."""
        e = self._make_exo(beta=0.1, exo_epsilon=1e-3)
        for _ in range(10):
            pi_logps_w = torch.randn(4)
            pi_logps_l = torch.randn(4)
            ref_logps_w = torch.randn(4)
            ref_logps_l = torch.randn(4)
            loss, _, _ = e.compute_loss(
                pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
            assert loss.item() >= -1e-6  # KL >= 0

    def test_loss_is_finite(self):
        e = self._make_exo(beta=0.1, exo_epsilon=1e-3)
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])
        loss, margin, reward_acc = e.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert torch.isfinite(loss)
        assert torch.isfinite(margin)

    def test_loss_zero_when_model_matches_labels(self):
        """Loss should be near zero when q matches p."""
        e = self._make_exo(beta=0.1, exo_epsilon=1e-3)
        # When delta is very large positive, q_w -> 1 and p_w = 1-eps ~ 1
        # So KL(q||p) -> 1*(log(1)-log(1-eps)) + 0*(log(0)-log(eps)) ~ eps
        pi_logps_w = torch.tensor([100.0])
        pi_logps_l = torch.tensor([-100.0])
        ref_logps_w = torch.tensor([0.0])
        ref_logps_l = torch.tensor([0.0])
        loss, _, _ = e.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        # Should be very small (close to -log(1-eps) which is ~eps for small eps)
        assert loss.item() < 0.01

    def test_loss_increases_with_wrong_ordering(self):
        """Loss should be larger when winner/loser are reversed."""
        e = self._make_exo(beta=0.1, exo_epsilon=1e-3)
        # Correct ordering: winner has higher log-ratio
        pi_logps_w = torch.tensor([-0.5])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.0])
        ref_logps_l = torch.tensor([-1.0])

        loss_correct, _, _ = e.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        # Swapped ordering
        loss_wrong, _, _ = e.compute_loss(
            pi_logps_l, pi_logps_w, ref_logps_l, ref_logps_w)

        assert loss_wrong.item() > loss_correct.item()

    def test_reward_accuracy(self):
        e = self._make_exo(beta=0.1, exo_epsilon=1e-3)
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])
        _, _, reward_acc = e.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert reward_acc.item() == 1.0

    def test_returns_three_values(self):
        e = self._make_exo(beta=0.1, exo_epsilon=1e-3)
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        result = e.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert len(result) == 3


class TestEXOConfig:
    def test_config_accepts_exo(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="exo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alg_name == "exo"
        assert hasattr(t, 'exo_epsilon')

    def test_config_default_value(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="exo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.exo_epsilon == 1e-3


class TestEXOComputeLogps:
    def _make_exo(self):
        from oxrl.algs.exo import EXO
        e = EXO.__new__(EXO)
        e.beta = 0.1
        e.exo_epsilon = 1e-3
        return e

    def test_compute_logps_shape(self):
        e = self._make_exo()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = e.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_logps_mask(self):
        e = self._make_exo()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        mask_full = torch.ones(1, 5)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        logps_full = e.compute_logps(logits, target_ids, mask_full)
        logps_half = e.compute_logps(logits, target_ids, mask_half)
        assert logps_full.item() != logps_half.item()
