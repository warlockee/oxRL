"""
Tests for SamPO (Down-Sampled KL Divergence DPO) algorithm.
Runs on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_sampo.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestSamPOImport:
    def test_sampo_importable(self):
        from oxrl.algs.sampo import SamPO
        assert SamPO is not None

    def test_sampo_in_algs_init(self):
        from oxrl.algs import SamPO
        assert SamPO is not None

    def test_sampo_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"sampo"' in source


class TestSamPOInterface:
    def test_has_required_methods(self):
        from oxrl.algs.sampo import SamPO
        assert hasattr(SamPO, 'compute_logps')
        assert hasattr(SamPO, 'compute_per_token_logps')
        assert hasattr(SamPO, 'forward')
        assert hasattr(SamPO, 'forward_per_token')
        assert hasattr(SamPO, 'downsample_logps')
        assert hasattr(SamPO, 'compute_loss')
        assert hasattr(SamPO, 'train_step')
        assert hasattr(SamPO, 'eval_step')

    def test_init_params(self):
        from oxrl.algs.sampo import SamPO
        sig = inspect.signature(SamPO.__init__)
        params = list(sig.parameters.keys())
        assert 'ref_model_engine' in params
        assert 'beta' in params


class TestSamPODownsampling:
    def _make_sampo(self, beta=0.1):
        from oxrl.algs.sampo import SamPO
        s = SamPO.__new__(SamPO)
        s.beta = beta
        return s

    def test_equal_length_no_change(self):
        """When both sequences have equal length, downsampling should use all tokens."""
        s = self._make_sampo()
        # Both sequences have 3 valid tokens
        per_token_w = torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0]])
        per_token_l = torch.tensor([[0.4, 0.5, 0.6, 0.0, 0.0]])
        mask_w = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
        mask_l = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])

        sampled_w, sampled_l = s.downsample_logps(per_token_w, per_token_l, mask_w, mask_l)

        # Both should use all 3 tokens
        assert abs(sampled_w.item() - 0.6) < 1e-5  # 0.1 + 0.2 + 0.3
        assert abs(sampled_l.item() - 1.5) < 1e-5  # 0.4 + 0.5 + 0.6

    def test_unequal_length_samples_fewer(self):
        """When sequences differ in length, should sample min(len_w, len_l) tokens."""
        s = self._make_sampo()
        # Chosen has 5 tokens, rejected has 2 tokens
        per_token_w = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
        per_token_l = torch.tensor([[0.6, 0.7, 0.0, 0.0, 0.0]])
        mask_w = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])
        mask_l = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])

        # Run multiple times to verify stochastic sampling
        sums_w = []
        for _ in range(100):
            sampled_w, sampled_l = s.downsample_logps(per_token_w, per_token_l, mask_w, mask_l)
            sums_w.append(sampled_w.item())

        # Rejected always uses all 2 tokens: 0.6 + 0.7 = 1.3
        _, sampled_l = s.downsample_logps(per_token_w, per_token_l, mask_w, mask_l)
        assert abs(sampled_l.item() - 1.3) < 1e-5

        # Chosen should sample 2 out of 5 tokens (varies each time)
        # Check that we get different sums (stochastic)
        unique_sums = set(round(x, 4) for x in sums_w)
        # With 5 tokens pick 2: C(5,2) = 10 possible combinations
        assert len(unique_sums) > 1

    def test_zero_length_returns_zero(self):
        """If min length is 0, should return 0."""
        s = self._make_sampo()
        per_token_w = torch.tensor([[0.1, 0.2, 0.3]])
        per_token_l = torch.tensor([[0.0, 0.0, 0.0]])
        mask_w = torch.tensor([[1.0, 1.0, 1.0]])
        mask_l = torch.tensor([[0.0, 0.0, 0.0]])

        sampled_w, sampled_l = s.downsample_logps(per_token_w, per_token_l, mask_w, mask_l)
        assert sampled_w.item() == 0.0
        assert sampled_l.item() == 0.0

    def test_batch_processing(self):
        """Verify batch processing handles multiple samples independently."""
        s = self._make_sampo()
        per_token_w = torch.tensor([
            [0.1, 0.2, 0.0],
            [0.3, 0.4, 0.5],
        ])
        per_token_l = torch.tensor([
            [0.6, 0.0, 0.0],
            [0.7, 0.8, 0.0],
        ])
        mask_w = torch.tensor([
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ])
        mask_l = torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ])

        sampled_w, sampled_l = s.downsample_logps(per_token_w, per_token_l, mask_w, mask_l)

        assert sampled_w.shape == (2,)
        assert sampled_l.shape == (2,)

        # Sample 0: T_m = min(2, 1) = 1, rejected uses its 1 token
        assert abs(sampled_l[0].item() - 0.6) < 1e-5

        # Sample 1: T_m = min(3, 2) = 2, rejected uses both its tokens
        assert abs(sampled_l[1].item() - 1.5) < 1e-5  # 0.7 + 0.8


class TestSamPOLossMath:
    def _make_sampo(self, beta=0.1):
        from oxrl.algs.sampo import SamPO
        s = SamPO.__new__(SamPO)
        s.beta = beta
        return s

    def test_loss_formula_standard_dpo(self):
        """Verify compute_loss implements standard DPO: -logsigmoid(beta*(logr_w - logr_l))."""
        beta = 0.2
        s = self._make_sampo(beta=beta)

        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.1, -0.6])
        ref_logps_l = torch.tensor([-2.1, -1.6])

        loss, _, _ = s.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        expected = -F.logsigmoid(beta * (logr_w - logr_l)).mean()

        assert abs(loss.item() - expected.item()) < 1e-5

    def test_loss_is_finite(self):
        s = self._make_sampo(beta=0.1)
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        ref_logps_w = torch.tensor([-2.6, -3.1])
        ref_logps_l = torch.tensor([-4.1, -5.1])
        loss, margin, reward_acc = s.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert torch.isfinite(loss)
        assert torch.isfinite(margin)

    def test_reward_accuracy(self):
        s = self._make_sampo(beta=0.1)
        pi_logps_w = torch.tensor([-0.5, -0.3, -0.2])
        pi_logps_l = torch.tensor([-2.0, -1.5, -3.0])
        ref_logps_w = torch.tensor([-1.0, -1.0, -1.0])
        ref_logps_l = torch.tensor([-1.0, -1.0, -1.0])
        _, _, reward_acc = s.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert reward_acc.item() == 1.0

    def test_returns_three_values(self):
        s = self._make_sampo(beta=0.1)
        pi_logps_w = torch.tensor([-1.0])
        pi_logps_l = torch.tensor([-2.0])
        ref_logps_w = torch.tensor([-1.1])
        ref_logps_l = torch.tensor([-2.1])
        result = s.compute_loss(
            pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)
        assert len(result) == 3


class TestSamPOComputeLogps:
    def _make_sampo(self):
        from oxrl.algs.sampo import SamPO
        s = SamPO.__new__(SamPO)
        s.beta = 0.1
        return s

    def test_compute_logps_shape(self):
        s = self._make_sampo()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        logps = s.compute_logps(logits, target_ids, loss_mask)
        assert logps.shape == (2,)

    def test_compute_per_token_logps_shape(self):
        s = self._make_sampo()
        logits = torch.randn(2, 9, 100)
        target_ids = torch.randint(0, 100, (2, 9))
        loss_mask = torch.ones(2, 9)
        per_token = s.compute_per_token_logps(logits, target_ids, loss_mask)
        assert per_token.shape == (2, 9)

    def test_per_token_sum_equals_logps(self):
        """Per-token logps summed should equal compute_logps result."""
        s = self._make_sampo()
        logits = torch.randn(2, 5, 50)
        target_ids = torch.randint(0, 50, (2, 5))
        loss_mask = torch.ones(2, 5)
        logps = s.compute_logps(logits, target_ids, loss_mask)
        per_token = s.compute_per_token_logps(logits, target_ids, loss_mask)
        per_token_sum = per_token.sum(-1)
        assert torch.allclose(logps, per_token_sum, atol=1e-5)

    def test_compute_logps_mask(self):
        s = self._make_sampo()
        logits = torch.randn(1, 5, 50)
        target_ids = torch.randint(0, 50, (1, 5))
        mask_full = torch.ones(1, 5)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        logps_full = s.compute_logps(logits, target_ids, mask_full)
        logps_half = s.compute_logps(logits, target_ids, mask_half)
        assert logps_full.item() != logps_half.item()


class TestSamPOConfig:
    def test_config_accepts_sampo(self):
        from oxrl.configs.schema import Train
        t = Train(alg_name="sampo", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert t.alg_name == "sampo"
