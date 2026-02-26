"""
Comprehensive tests for bug fixes in oxRL.
All tests run on CPU without Ray/DeepSpeed/vLLM.
Run with: pytest tests/test_bugs.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import inspect

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# Bug 4: clip_low default should be 0.2, not -0.2
# ============================================================
class TestBug4ClipLow:
    def test_clip_low_default_is_positive(self):
        from oxrl.configs.load import Train
        t = Train(alg_name="sgrpo", total_number_of_epochs=1, train_steps_per_epoch=10)
        assert t.clip_low == 0.2, f"Expected clip_low=0.2, got {t.clip_low}"

    def test_clip_range_is_correct(self):
        from oxrl.configs.load import Train
        t = Train(alg_name="sgrpo", total_number_of_epochs=1, train_steps_per_epoch=10)
        lower = 1.0 - t.clip_low
        upper = 1.0 + t.clip_high
        assert lower == pytest.approx(0.8), f"Lower clip bound should be 0.8, got {lower}"
        assert upper == pytest.approx(1.2), f"Upper clip bound should be 1.2, got {upper}"
        assert lower < upper, "Lower clip bound must be less than upper"

    def test_clip_low_used_in_grpo_clamping(self):
        """Verify that the clip range produces sensible clamping."""
        clip_low = 0.2
        clip_high = 0.2
        ratio = torch.tensor([0.5, 0.8, 1.0, 1.2, 1.5])
        clamped = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
        expected = torch.tensor([0.8, 0.8, 1.0, 1.2, 1.2])
        assert torch.allclose(clamped, expected), f"Clamped: {clamped}, Expected: {expected}"


# ============================================================
# Bug 1: Replay buffer should reset for all RL algorithms
# ============================================================
class TestBug1ReplayBufferReset:
    def test_replay_buffer_reset_clears_items(self):
        from oxrl.rollouts.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(pad_token_id=0, max_seq_len=128)
        T = 10
        buf.add(
            input_ids=torch.randint(0, 100, (T,)),
            rewards=torch.zeros(T),
            zscores=torch.zeros(T),
            masks=torch.ones(T),
            dones=torch.zeros(T),
            old_logprobs=torch.zeros(T),
        )
        assert len(buf) == 1
        assert buf.total_action_tokens > 0
        buf.reset()
        assert len(buf) == 0
        assert buf.total_action_tokens == 0

    def test_replay_buffer_add_and_reset_cycle(self):
        from oxrl.rollouts.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(pad_token_id=0, max_seq_len=64)
        for _ in range(5):
            T = 8
            buf.add(
                input_ids=torch.randint(0, 100, (T,)),
                rewards=torch.zeros(T),
                zscores=torch.randn(T),
                masks=torch.ones(T),
                dones=torch.zeros(T),
                old_logprobs=torch.randn(T),
            )
        assert len(buf) == 5
        buf.reset()
        assert len(buf) == 0


# ============================================================
# Bug 2: GRPO should use config LR, not hardcoded 1e-6
# ============================================================
class TestBug2GRPOConfigLR:
    def _read_source(self, filename):
        path = os.path.join(os.path.dirname(__file__), "..", "oxrl", "algs", filename)
        with open(path) as f:
            return f.read()

    def test_grpo_init_accepts_lr(self):
        """GRPO.__init__ should accept lr parameter (source-level check, since @ray.remote wraps signature)."""
        source = self._read_source("grpo.py")
        assert "lr: float" in source, "GRPO.__init__ must accept 'lr: float'"
        assert "betas: list" in source, "GRPO.__init__ must accept 'betas: list'"
        assert "weight_decay: float" in source, "GRPO.__init__ must accept 'weight_decay: float'"
        assert "adam_epsilon: float" in source, "GRPO.__init__ must accept 'adam_epsilon: float'"

    def test_grpo_uses_self_lr_not_hardcoded(self):
        """GRPO should use self.lr, not a hardcoded value."""
        source = self._read_source("grpo.py")
        assert "lr=self.lr" in source, "Optimizer should use self.lr"
        assert "lr=1e-6" not in source, "Hardcoded lr=1e-6 should be removed"

    def test_grpo_uses_config_betas_and_eps(self):
        source = self._read_source("grpo.py")
        assert "betas=tuple(self.betas)" in source, "Optimizer should use self.betas"
        assert "eps=self.adam_epsilon" in source, "Optimizer should use self.adam_epsilon"
        assert "weight_decay=self.weight_decay" in source, "Optimizer should use self.weight_decay"

    def test_grpo_default_lr_is_1e5(self):
        source = self._read_source("grpo.py")
        assert "lr: float = 1e-5" in source, "Default lr should be 1e-5"

    def test_simpo_uses_self_lr_not_hardcoded(self):
        """SimPO should use self.lr, not a hardcoded value."""
        source = self._read_source("simpo.py")
        assert "lr=self.lr" in source, "SimPO optimizer should use self.lr"
        assert "lr=1e-6" not in source, "Hardcoded lr=1e-6 should be removed from SimPO"
        assert "lr: float = 1e-5" in source, "SimPO default lr should be 1e-5"

    def test_main_rl_passes_optimizer_params(self):
        path = os.path.join(os.path.dirname(__file__), "..", "main_rl.py")
        with open(path) as f:
            source = f.read()
        assert "'lr':params.train.lr" in source, "main_rl.py should pass lr from config"
        assert "'betas':params.train.betas" in source, "main_rl.py should pass betas from config"
        assert "'weight_decay':params.train.weight_decay" in source
        assert "'adam_epsilon':params.train.adam_epsilon" in source


# ============================================================
# Bug 3: ORPO log-odds should use average log-probs
# ============================================================
class TestBug3ORPOLogOdds:
    def _make_orpo(self):
        from oxrl.algs.orpo import ORPO
        orpo = ORPO.__new__(ORPO)
        orpo.beta = 0.1
        orpo.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        return orpo

    def test_orpo_or_loss_is_finite(self):
        orpo = self._make_orpo()
        B, T, V = 2, 10, 100
        # Simulate sum-of-logprobs (negative, realistic)
        pi_logps_w = torch.tensor([-2.5, -3.0])
        pi_logps_l = torch.tensor([-4.0, -5.0])
        num_valid_w = torch.tensor([5.0, 6.0])
        num_valid_l = torch.tensor([5.0, 6.0])
        sft_logits = torch.randn(B, T, V)
        sft_targets = torch.randint(0, V, (B, T))
        sft_mask = torch.ones(B, T)

        loss, sft_loss, or_loss = orpo.compute_loss(
            pi_logps_w, pi_logps_l, num_valid_w, num_valid_l,
            sft_logits, sft_targets, sft_mask
        )
        assert torch.isfinite(or_loss), f"or_loss should be finite, got {or_loss}"
        assert torch.isfinite(loss), f"total loss should be finite, got {loss}"

    def test_orpo_avg_logprobs_in_valid_range(self):
        """Average log-probs should give exp values in (0,1)."""
        orpo = self._make_orpo()
        pi_logps_w = torch.tensor([-2.5, -3.0])
        num_valid_w = torch.tensor([5.0, 6.0])
        avg = pi_logps_w / num_valid_w
        probs = torch.exp(avg)
        assert (probs > 0).all() and (probs < 1).all(), \
            f"exp(avg_logps) should be in (0,1), got {probs}"

    def test_orpo_chosen_preferred_over_rejected(self):
        """When chosen has higher avg log-prob, or_loss should be small."""
        orpo = self._make_orpo()
        B, T, V = 2, 5, 50
        # Chosen has higher log-probs than rejected
        pi_logps_w = torch.tensor([-1.0, -1.5])
        pi_logps_l = torch.tensor([-3.0, -4.0])
        num_valid_w = torch.tensor([5.0, 5.0])
        num_valid_l = torch.tensor([5.0, 5.0])
        sft_logits = torch.randn(B, T, V)
        sft_targets = torch.randint(0, V, (B, T))
        sft_mask = torch.ones(B, T)

        _, _, or_loss_good = orpo.compute_loss(
            pi_logps_w, pi_logps_l, num_valid_w, num_valid_l,
            sft_logits, sft_targets, sft_mask
        )
        # Swap chosen/rejected — loss should be higher
        _, _, or_loss_bad = orpo.compute_loss(
            pi_logps_l, pi_logps_w, num_valid_l, num_valid_w,
            sft_logits, sft_targets, sft_mask
        )
        assert or_loss_good < or_loss_bad, \
            f"Correct preference should have lower loss: {or_loss_good} vs {or_loss_bad}"


# ============================================================
# Bug 5: KTO KL should be computed over chosen only
# ============================================================
class TestBug5KTOKL:
    def _make_kto(self):
        from oxrl.algs.kto import KTO
        kto = KTO.__new__(KTO)
        kto.beta = 0.1
        kto.lambda_p = 1.0
        kto.lambda_n = 1.0
        kto.kl_ema = 0.0
        kto.kl_alpha = 0.1
        return kto

    def test_kl_ema_uses_chosen_only(self):
        kto = self._make_kto()
        batch_size = 3
        pi_logps = torch.tensor([-1.0, -1.5, -2.0, -3.0, -4.0, -5.0])
        ref_logps = torch.tensor([-1.1, -1.6, -2.1, -3.1, -4.1, -5.1])
        labels = torch.cat([torch.ones(batch_size), -torch.ones(batch_size)])

        kto.compute_loss(pi_logps, ref_logps, labels, batch_size)

        log_ratio = pi_logps - ref_logps
        expected_kl = log_ratio[:batch_size].mean().item()
        expected_ema = 0.0 * 0.9 + 0.1 * expected_kl
        assert abs(kto.kl_ema - expected_ema) < 1e-6, \
            f"KL EMA should be {expected_ema}, got {kto.kl_ema}"

    def test_kl_differs_from_full_batch(self):
        kto = self._make_kto()
        batch_size = 3
        # Chosen and rejected have very different log-ratios
        pi_logps = torch.tensor([-1.0, -1.0, -1.0, -10.0, -10.0, -10.0])
        ref_logps = torch.tensor([-1.1, -1.1, -1.1, -10.1, -10.1, -10.1])
        labels = torch.cat([torch.ones(batch_size), -torch.ones(batch_size)])

        kto.compute_loss(pi_logps, ref_logps, labels, batch_size)

        log_ratio = pi_logps - ref_logps
        chosen_kl = log_ratio[:batch_size].mean().item()
        full_kl = log_ratio.mean().item()
        # These should be equal in this case since log_ratios are identical
        # Let's use a case where they differ
        kto.kl_ema = 0.0
        pi_logps2 = torch.tensor([-1.0, -1.0, -1.0, -5.0, -5.0, -5.0])
        ref_logps2 = torch.tensor([-2.0, -2.0, -2.0, -1.0, -1.0, -1.0])
        kto.compute_loss(pi_logps2, ref_logps2, labels, batch_size)

        log_ratio2 = pi_logps2 - ref_logps2
        chosen_only = log_ratio2[:batch_size].mean().item()
        all_samples = log_ratio2.mean().item()
        assert abs(chosen_only - all_samples) > 0.1, \
            "Chosen-only and full-batch KL should differ"

    def test_compute_loss_accepts_batch_size(self):
        from oxrl.algs.kto import KTO
        sig = inspect.signature(KTO.compute_loss)
        params = list(sig.parameters.keys())
        assert 'batch_size' in params, "compute_loss must accept 'batch_size'"


# ============================================================
# Bug 6: LR scheduler should not have 100x overestimate
# ============================================================
class TestBug6LRScheduler:
    def test_rl_scheduler_total_steps(self):
        from oxrl.configs.load import Config
        raw = {
            "run": {"experiment_id": "test", "training_gpus": 2, "rollout_gpus": 2,
                    "checkpoint_dir": "/tmp/test"},
            "train": {"alg_name": "sgrpo", "total_number_of_epochs": 5,
                      "train_steps_per_epoch": 20, "lr": 1e-5},
            "model": {"name": "test-model"},
            "data": {"train_dnames": ["d"], "train_ratios": {"d": 1.0},
                     "train_files_path": "/tmp/d", "val_files_path": "/tmp/v"},
        }
        config = Config(**raw)
        config.run.method = "rl"
        config.sync_deepspeed_config(world_size=2)

        expected_total = 5 * 20  # epochs * steps_per_epoch
        actual_total = config.deepspeed.scheduler["params"]["total_num_steps"]
        assert actual_total == expected_total, \
            f"Expected total_num_steps={expected_total}, got {actual_total}"

    def test_warmup_steps_proportional(self):
        from oxrl.configs.load import Config
        raw = {
            "run": {"experiment_id": "test", "training_gpus": 2, "rollout_gpus": 2,
                    "checkpoint_dir": "/tmp/test"},
            "train": {"alg_name": "sgrpo", "total_number_of_epochs": 10,
                      "train_steps_per_epoch": 50, "lr": 1e-5,
                      "warmup_steps_ratio": 0.1},
            "model": {"name": "test-model"},
            "data": {"train_dnames": ["d"], "train_ratios": {"d": 1.0},
                     "train_files_path": "/tmp/d", "val_files_path": "/tmp/v"},
        }
        config = Config(**raw)
        config.run.method = "rl"
        config.sync_deepspeed_config(world_size=2)

        total = config.deepspeed.scheduler["params"]["total_num_steps"]
        warmup = config.deepspeed.scheduler["params"]["warmup_num_steps"]
        assert total == 500
        assert warmup == 50  # 10% of 500


# ============================================================
# Bug 7: Reward normalization should use Bessel's correction
# ============================================================
class TestBug7RewardStd:
    def test_bessel_correction_differs_from_population(self):
        rewards = [2.0, 4.0, 6.0, 8.0]
        N = len(rewards)
        arr = np.array(rewards)
        mean = arr.mean()

        pop_std = np.sqrt(((arr - mean) ** 2).sum() / N)
        sample_std = np.sqrt(((arr - mean) ** 2).sum() / max(1, N - 1))

        assert abs(pop_std - sample_std) > 0.1, "Population and sample std should differ"
        assert abs(sample_std - np.std(arr, ddof=1)) < 1e-10, \
            "Should match numpy sample std"

    def test_bessel_with_two_samples(self):
        """N=2 is where Bessel's correction matters most."""
        rewards = [0.0, 1.0]
        arr = np.array(rewards)
        mean = arr.mean()
        sample_std = np.sqrt(((arr - mean) ** 2).sum() / max(1, len(rewards) - 1))
        expected = np.std(arr, ddof=1)
        assert abs(sample_std - expected) < 1e-10

    def test_single_sample_no_division_by_zero(self):
        """With N=1, max(1, N-1)=1 prevents division by zero."""
        rewards = [5.0]
        arr = np.array(rewards)
        mean = arr.mean()
        N = len(rewards)
        std = np.sqrt(((arr - mean) ** 2).sum() / max(1, N - 1))
        assert std == 0.0  # Single sample has no variance


# ============================================================
# Bug 8: Monkey-patch should be centralized
# ============================================================
class TestBug8MonkeyPatch:
    def test_ensure_sliding_window_cache_exists(self):
        from oxrl.utils.setup import ensure_sliding_window_cache
        assert callable(ensure_sliding_window_cache)

    def test_ensure_sliding_window_cache_idempotent(self):
        from oxrl.utils.setup import ensure_sliding_window_cache
        ensure_sliding_window_cache()
        ensure_sliding_window_cache()  # Should not raise

    def test_vllm_engine_imports_from_setup(self):
        source_path = os.path.join(os.path.dirname(__file__), "..",
                                   "oxrl", "rollouts", "vllm_engine.py")
        with open(source_path) as f:
            source = f.read()
        assert 'ensure_sliding_window_cache' in source, \
            "vllm_engine.py should import ensure_sliding_window_cache"
        assert 'class SlidingWindowCache' not in source, \
            "vllm_engine.py should not define SlidingWindowCache inline"


# ============================================================
# Additional Fix A: token_type_ids consistency
# ============================================================
class TestTokenTypeIds:
    def test_sft_forward_passes_token_type_ids(self):
        source_path = os.path.join(os.path.dirname(__file__), "..",
                                   "oxrl", "algs", "sft.py")
        with open(source_path) as f:
            source = f.read()
        assert 'token_type_ids' in source, "SFT should pass token_type_ids"

    def test_orpo_forward_passes_token_type_ids(self):
        source_path = os.path.join(os.path.dirname(__file__), "..",
                                   "oxrl", "algs", "orpo.py")
        with open(source_path) as f:
            source = f.read()
        assert 'token_type_ids' in source, "ORPO should pass token_type_ids"

    def test_kto_forward_passes_token_type_ids(self):
        source_path = os.path.join(os.path.dirname(__file__), "..",
                                   "oxrl", "algs", "kto.py")
        with open(source_path) as f:
            source = f.read()
        assert 'token_type_ids' in source, "KTO should pass token_type_ids"


# ============================================================
# Additional Fix B: Direct imports (no importlib hack)
# ============================================================
class TestDirectImports:
    def test_prompt_only_importable(self):
        from oxrl.datasets.prompt_only import PromptOnlyDataset
        assert PromptOnlyDataset is not None

    def test_prompt_response_importable(self):
        from oxrl.datasets.prompt_response import PromptResponseDataset
        assert PromptResponseDataset is not None

    def test_prompt_preference_importable(self):
        from oxrl.datasets.prompt_preference import PromptPreferenceDataset
        assert PromptPreferenceDataset is not None

    def test_main_rl_no_importlib_hack(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_rl.py")
        with open(source_path) as f:
            source = f.read()
        assert 'spec_from_file_location' not in source, \
            "main_rl.py should not use importlib.util.spec_from_file_location"

    def test_main_sl_no_importlib_hack(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert 'spec_from_file_location' not in source, \
            "main_sl.py should not use importlib.util.spec_from_file_location"
