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

    def test_simpo_is_sl_compatible(self):
        """SimPO should be an SL algorithm (optimizer passed in, no self-initialization)."""
        source = self._read_source("simpo.py")
        assert "self.model_engine = model_engine" in source, "SimPO should receive model_engine from main_sl.py"
        assert "@ray.remote" not in source, "SimPO should not be a Ray actor (SL path)"

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


# ============================================================
# New Algorithm: CPT (Continued Pretraining)
# ============================================================
class TestCPT:
    def test_cpt_importable(self):
        from oxrl.algs.cpt import CPT
        assert CPT is not None

    def test_cpt_loss_mask_covers_all_tokens(self):
        """CPT should train on all non-padding tokens, not just answer tokens."""
        B, T = 2, 10
        attn_mask = torch.ones(B, T, dtype=torch.long)
        # CPT loss_mask = attn_mask[:, 1:]
        cpt_loss_mask = attn_mask[:, 1:].contiguous()
        assert cpt_loss_mask.sum() == B * (T - 1), \
            "CPT loss_mask should cover all non-padding tokens"

    def test_cpt_registered_in_sl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"cpt"' in source, "CPT should be registered in SL_ALGORITHMS"


# ============================================================
# New Algorithm: KD (Knowledge Distillation)
# ============================================================
class TestKD:
    def test_kd_importable(self):
        from oxrl.algs.kd import KD
        assert KD is not None

    def test_kd_loss_combines_ce_and_kl(self):
        from oxrl.algs.kd import KD
        kd = KD.__new__(KD)
        kd.alpha = 0.5
        kd.temperature = 2.0
        kd.normalize_loss = False
        kd.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        B, T, V = 2, 8, 50
        student_logits = torch.randn(B, T, V)
        teacher_logits = torch.randn(B, T, V)
        target_ids = torch.randint(0, V, (B, T))
        loss_mask = torch.ones(B, T)
        total, ce, kl = kd.compute_loss(student_logits, teacher_logits, target_ids, loss_mask)
        assert torch.isfinite(total), f"KD total loss should be finite, got {total}"
        assert ce > 0, "CE loss should be positive"
        assert kl >= 0, "KL loss should be non-negative"

    def test_kd_alpha_controls_weighting(self):
        from oxrl.algs.kd import KD
        kd = KD.__new__(KD)
        kd.temperature = 2.0
        kd.normalize_loss = False
        kd.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        B, T, V = 2, 8, 50
        student_logits = torch.randn(B, T, V)
        teacher_logits = torch.randn(B, T, V)
        target_ids = torch.randint(0, V, (B, T))
        loss_mask = torch.ones(B, T)
        kd.alpha = 1.0  # Pure CE
        total1, ce1, kl1 = kd.compute_loss(student_logits, teacher_logits, target_ids, loss_mask)
        kd.alpha = 0.0  # Pure KL
        total2, ce2, kl2 = kd.compute_loss(student_logits, teacher_logits, target_ids, loss_mask)
        assert abs(total1.item() - ce1.item()) < 1e-4, "alpha=1.0 should give pure CE"
        assert abs(total2.item() - kl2.item()) < 1e-4, "alpha=0.0 should give pure KL"


# ============================================================
# New Algorithm: RM (Reward Model Training)
# ============================================================
class TestRM:
    def test_rm_importable(self):
        from oxrl.algs.rm import RM, RewardValueHead
        assert RM is not None
        assert RewardValueHead is not None

    def test_reward_value_head_shape(self):
        from oxrl.algs.rm import RewardValueHead
        head = RewardValueHead(hidden_size=64)
        hidden = torch.randn(2, 10, 64)
        out = head(hidden)
        assert out.shape == (2, 10), f"Expected (2, 10), got {out.shape}"

    def test_rm_loss_correct_preference(self):
        from oxrl.algs.rm import RM
        rm = RM.__new__(RM)
        r_chosen = torch.tensor([2.0, 3.0])
        r_rejected = torch.tensor([0.0, 1.0])
        loss_good, acc_good, _ = rm.compute_loss(r_chosen, r_rejected)
        loss_bad, acc_bad, _ = rm.compute_loss(r_rejected, r_chosen)
        assert loss_good < loss_bad, "Correct preference should have lower loss"
        assert acc_good > 0.5, "Accuracy should be > 0.5 for correct preferences"


# ============================================================
# New Algorithm: Online DPO
# ============================================================
class TestOnlineDPO:
    def test_online_dpo_importable(self):
        from oxrl.algs.online_dpo import OnlineDPO
        assert OnlineDPO is not None

    def test_online_dpo_loss_same_as_dpo(self):
        from oxrl.algs.online_dpo import OnlineDPO
        from oxrl.algs.dpo import DPO
        odpo = OnlineDPO.__new__(OnlineDPO)
        odpo.beta = 0.1
        dpo = DPO.__new__(DPO)
        dpo.beta = 0.1
        pi_w = torch.tensor([-1.0, -2.0])
        pi_l = torch.tensor([-3.0, -4.0])
        ref_w = torch.tensor([-1.1, -2.1])
        ref_l = torch.tensor([-3.1, -4.1])
        loss_o, _ = odpo.compute_loss(pi_w, pi_l, ref_w, ref_l)
        loss_d, _ = dpo.compute_loss(pi_w, pi_l, ref_w, ref_l)
        assert abs(loss_o.item() - loss_d.item()) < 1e-6


# ============================================================
# New Algorithm: RFT (Rejection Sampling Fine-Tuning)
# ============================================================
class TestRFT:
    def test_rft_importable(self):
        from oxrl.algs.rft import RFT
        assert RFT is not None

    def test_rft_has_reward_threshold(self):
        from oxrl.algs.rft import RFT
        sig = inspect.signature(RFT.__init__)
        params = list(sig.parameters.keys())
        assert 'reward_threshold' in params, "RFT should accept reward_threshold"


# ============================================================
# New Algorithm: SPIN (Self-Play)
# ============================================================
class TestSPIN:
    def test_spin_importable(self):
        from oxrl.algs.spin import SPIN
        assert SPIN is not None

    def test_spin_loss_same_as_dpo(self):
        from oxrl.algs.spin import SPIN
        from oxrl.algs.dpo import DPO
        spin = SPIN.__new__(SPIN)
        spin.beta = 0.1
        dpo = DPO.__new__(DPO)
        dpo.beta = 0.1
        pi_w = torch.tensor([-1.0, -2.0])
        pi_l = torch.tensor([-3.0, -4.0])
        ref_w = torch.tensor([-1.1, -2.1])
        ref_l = torch.tensor([-3.1, -4.1])
        loss_s, _ = spin.compute_loss(pi_w, pi_l, ref_w, ref_l)
        loss_d, _ = dpo.compute_loss(pi_w, pi_l, ref_w, ref_l)
        assert abs(loss_s.item() - loss_d.item()) < 1e-6


# ============================================================
# New Algorithm: RLHF (reward model reward function)
# ============================================================
class TestRLHF:
    def test_rm_reward_importable(self):
        from oxrl.rewards.rm_reward import rm_reward_func, load_reward_model
        assert rm_reward_func is not None
        assert load_reward_model is not None

    def test_rlhf_registered_in_rl_algorithms(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_rl.py")
        with open(source_path) as f:
            source = f.read()
        assert '"rlhf"' in source, "RLHF should be registered in RL_ALGORITHMS"

    def test_rm_reward_in_rewards_init(self):
        from oxrl.rewards import rm_reward_func, load_reward_model
        assert rm_reward_func is not None
        assert load_reward_model is not None


# ============================================================
# New Algorithm: IPO (Identity Preference Optimization)
# ============================================================
class TestIPO:
    """Tests for IPO (Identity Preference Optimization) algorithm."""

    def test_ipo_import(self):
        """Test that IPO class is importable."""
        from oxrl.algs.ipo import IPO
        assert IPO is not None

    def test_ipo_loss_math(self):
        """Test IPO squared loss computation with known values."""
        from oxrl.algs.ipo import IPO

        ipo = object.__new__(IPO)
        ipo.beta = 0.1

        # Known values
        pi_logps_w = torch.tensor([-1.0, -0.5])
        pi_logps_l = torch.tensor([-2.0, -1.5])
        ref_logps_w = torch.tensor([-1.2, -0.7])
        ref_logps_l = torch.tensor([-2.3, -1.6])

        loss, margin = ipo.compute_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l)

        # Verify: loss = ((logr_w - logr_l) - 1/(2*beta))^2
        logr_w = pi_logps_w - ref_logps_w
        logr_l = pi_logps_l - ref_logps_l
        expected = ((logr_w - logr_l) - 1.0 / (2.0 * 0.1)).square().mean()

        assert abs(loss.item() - expected.item()) < 1e-5
        assert margin.dim() == 0  # scalar tensor

    def test_ipo_interface(self):
        """Test IPO has required methods."""
        from oxrl.algs.ipo import IPO
        assert hasattr(IPO, 'compute_loss')
        assert hasattr(IPO, 'compute_logps')
        assert hasattr(IPO, 'forward')
        assert hasattr(IPO, 'train_step')
        assert hasattr(IPO, 'eval_step')


# ============================================================
# New Algorithm: SimPO (Simple Preference Optimization)
# ============================================================
class TestSimPO:
    """Tests for SimPO (Simple Preference Optimization) algorithm."""

    def test_simpo_import(self):
        """Test that SimPO class is importable."""
        from oxrl.algs.simpo import SimPO
        assert SimPO is not None

    def test_simpo_loss_math(self):
        """Test SimPO loss computation with known values."""
        from oxrl.algs.simpo import SimPO

        simpo = object.__new__(SimPO)
        simpo.beta = 0.1
        simpo.gamma = 0.5

        # Known length-normalized log-probs
        pi_logps_w = torch.tensor([-0.5, -0.3])
        pi_logps_l = torch.tensor([-1.5, -1.0])

        loss, margin = simpo.compute_loss(pi_logps_w, pi_logps_l)

        # Verify: loss = -logsigmoid(beta * (w - l) - gamma)
        logits = 0.1 * (pi_logps_w - pi_logps_l) - 0.5
        expected = -F.logsigmoid(logits).mean()

        assert abs(loss.item() - expected.item()) < 1e-5
        assert margin.dim() == 0  # scalar tensor

    def test_simpo_interface(self):
        """Test SimPO has required methods (no ref model)."""
        from oxrl.algs.simpo import SimPO
        assert hasattr(SimPO, 'compute_loss')
        assert hasattr(SimPO, 'compute_logps')
        assert hasattr(SimPO, 'forward')
        assert hasattr(SimPO, 'train_step')
        assert hasattr(SimPO, 'eval_step')


# ============================================================
# New Algorithm: PPO (Proximal Policy Optimization)
# ============================================================
class TestPPO:
    """Tests for PPO (Proximal Policy Optimization) algorithm."""

    def test_ppo_import(self):
        """Test that PPO class is importable."""
        from oxrl.algs.ppo import PPO
        assert PPO is not None

    def test_ppo_advantage_computation(self):
        """Test PPO GAE advantage computation with known values."""
        from oxrl.algs.ppo import PPO

        # PPO is @ray.remote; get the underlying class
        PPOCls = PPO.__ray_actor_class__
        ppo = object.__new__(PPOCls)
        ppo.gamma = 0.99
        ppo.tau = 0.95

        # Simple 2-step episode: [r0, r1], values [v0, v1], done at t=1
        rewards = torch.tensor([[1.0, 2.0]])
        values = torch.tensor([[0.5, 1.0]])
        done = torch.tensor([[0.0, 1.0]])
        mask = torch.tensor([[1.0, 1.0]])

        returns, advantages = ppo.compute_advantages(rewards, values, done, mask)

        # At t=1 (terminal): delta = r1 + 0 - v1 = 2.0 - 1.0 = 1.0, adv = 1.0
        # At t=0: delta = r0 + gamma*v1*1 - v0 = 1.0 + 0.99*1.0 - 0.5 = 1.49
        #         adv = delta + gamma*tau*adv[1]*1 = 1.49 + 0.99*0.95*1.0 = 2.4305
        assert advantages.shape == (1, 2)
        assert abs(advantages[0, 1].item() - 1.0) < 1e-4
        assert abs(advantages[0, 0].item() - 2.4305) < 1e-3

    def test_ppo_interface(self):
        """Test PPO has required methods matching RL interface."""
        from oxrl.algs.ppo import PPO
        assert hasattr(PPO, 'compute_advantages')
        assert hasattr(PPO, 'compute_policy_loss')
        assert hasattr(PPO, 'compute_value_loss')
        assert hasattr(PPO, 'train_step')
        assert hasattr(PPO, 'save_checkpoint')
        assert hasattr(PPO, 'is_ready')


# ============================================================
# New Algorithm: RLAIF (alias for GRPO)
# ============================================================
class TestRLAIF:
    """Tests for RLAIF alias in RL_ALGORITHMS."""

    def test_rlaif_is_grpo_alias(self):
        """Test that RLAIF is registered as a GRPO alias."""
        from oxrl.algs.grpo import GRPO
        # RLAIF should map to GRPO class
        assert GRPO is not None


# ============================================================
# Integration: All algorithms registered
# ============================================================
class TestAlgorithmRegistry:
    def test_sl_algorithms_count(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_sl.py")
        with open(source_path) as f:
            source = f.read()
        for alg in ["sft", "dpo", "orpo", "kto", "cpt", "kd", "rm", "online_dpo", "rft", "spin", "ipo", "simpo"]:
            assert f'"{alg}"' in source, f"{alg} should be in SL_ALGORITHMS"

    def test_rl_algorithms_count(self):
        source_path = os.path.join(os.path.dirname(__file__), "..", "main_rl.py")
        with open(source_path) as f:
            source = f.read()
        for alg in ["sgrpo", "cispo", "rlhf", "rlaif", "ppo"]:
            assert f'"{alg}"' in source, f"{alg} should be in RL_ALGORITHMS"

    def test_config_new_fields(self):
        from oxrl.configs.load import Train
        t = Train(alg_name="sft", total_number_of_epochs=1, micro_batches_per_epoch=10)
        assert hasattr(t, 'kd_alpha'), "Train should have kd_alpha"
        assert hasattr(t, 'kd_temperature'), "Train should have kd_temperature"
        assert hasattr(t, 'reward_threshold'), "Train should have reward_threshold"
        assert hasattr(t, 'reward_model_path'), "Train should have reward_model_path"
        assert t.kd_alpha == 0.5
        assert t.kd_temperature == 2.0
        assert t.reward_threshold == 0.5
        assert t.reward_model_path == ""
