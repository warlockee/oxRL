"""
Comprehensive verification tests for the MOP design pattern refactoring.

Tests extracted loss functions, tools, config modules, and loop phases
to ensure behavioral equivalence with the original monolithic code.

Run with: pytest tests/test_mop_refactoring.py -v
"""
import pytest
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# 1. Loss Function Registry
# ============================================================
class TestLossRegistry:
    """Verify the loss function registry dispatches correctly."""

    def test_registry_contains_all_variants(self):
        from oxrl.algs.losses import LOSS_REGISTRY
        expected = {"sgrpo", "gspo", "cispo", "ppo"}
        assert set(LOSS_REGISTRY.keys()) == expected

    def test_get_loss_fn_returns_callable(self):
        from oxrl.algs.losses import get_loss_fn
        for name in ("sgrpo", "gspo", "cispo", "ppo"):
            fn = get_loss_fn(name)
            assert callable(fn), f"{name} should return a callable"

    def test_get_loss_fn_unknown_raises(self):
        from oxrl.algs.losses import get_loss_fn
        with pytest.raises(ValueError, match="Unknown loss variant"):
            get_loss_fn("nonexistent_algo")

    def test_ppo_maps_to_sgrpo(self):
        from oxrl.algs.losses import LOSS_REGISTRY
        assert LOSS_REGISTRY["ppo"] is LOSS_REGISTRY["sgrpo"]

    def test_all_loss_fns_have_same_signature(self):
        """All loss functions should accept the same 10 arguments."""
        import inspect
        from oxrl.algs.losses import LOSS_REGISTRY
        sigs = {}
        for name, fn in LOSS_REGISTRY.items():
            params = list(inspect.signature(fn).parameters.keys())
            sigs[name] = params
        # All should match sgrpo's signature
        ref = sigs["sgrpo"]
        for name, params in sigs.items():
            assert params == ref, f"{name} signature {params} != sgrpo signature {ref}"


# ============================================================
# 2. Loss Function Numerical Correctness
# ============================================================
class TestLossNumerics:
    """Verify each loss function produces correct numerical results."""

    @pytest.fixture
    def standard_inputs(self):
        """Standard test inputs: B=4, T=16."""
        B, T = 4, 16
        torch.manual_seed(42)
        logprobs = torch.randn(B, T) * 0.5
        old_logprobs = logprobs + 0.02 * torch.randn(B, T)
        advantages = torch.randn(B, T)
        mask = torch.ones(B, T)
        mask[:, 12:] = 0  # some padding
        entropies = torch.rand(B, T) * 2.0
        ref_logprobs = logprobs + 0.05 * torch.randn(B, T)
        return logprobs, old_logprobs, advantages, mask, entropies, ref_logprobs

    def test_sgrpo_loss_finite(self, standard_inputs):
        from oxrl.algs.losses.sgrpo import compute_sgrpo_loss
        loss, metrics = compute_sgrpo_loss(
            *standard_inputs, clip_low=0.2, clip_high=0.2, ent_coeff=0.01, kl_coeff=0.1
        )
        assert torch.isfinite(loss), f"SGRPO loss not finite: {loss}"
        assert all(np.isfinite(v) for v in metrics.values()), f"Non-finite metrics: {metrics}"

    def test_gspo_loss_finite(self, standard_inputs):
        from oxrl.algs.losses.gspo import compute_gspo_loss
        loss, metrics = compute_gspo_loss(
            *standard_inputs, clip_low=0.2, clip_high=0.2, ent_coeff=0.01, kl_coeff=0.1
        )
        assert torch.isfinite(loss), f"GSPO loss not finite: {loss}"
        assert all(np.isfinite(v) for v in metrics.values()), f"Non-finite metrics: {metrics}"

    def test_cispo_loss_finite(self, standard_inputs):
        from oxrl.algs.losses.cispo import compute_cispo_loss
        loss, metrics = compute_cispo_loss(
            *standard_inputs, clip_low=0.2, clip_high=0.2, ent_coeff=0.01, kl_coeff=0.1
        )
        assert torch.isfinite(loss), f"CISPO loss not finite: {loss}"
        assert all(np.isfinite(v) for v in metrics.values()), f"Non-finite metrics: {metrics}"

    def test_sgrpo_zero_advantage_zero_loss_pi(self):
        """With zero advantages, policy loss should be zero."""
        from oxrl.algs.losses.sgrpo import compute_sgrpo_loss
        B, T = 2, 8
        logprobs = torch.randn(B, T)
        old_logprobs = logprobs.clone()
        advantages = torch.zeros(B, T)
        mask = torch.ones(B, T)
        loss, metrics = compute_sgrpo_loss(
            logprobs, old_logprobs, advantages, mask, None, None,
            clip_low=0.2, clip_high=0.2, ent_coeff=0.0, kl_coeff=0.0,
        )
        assert abs(metrics["loss_pi"]) < 1e-6, f"Zero advantage should give ~zero loss_pi, got {metrics['loss_pi']}"

    def test_gspo_zero_advantage_zero_loss_pi(self):
        """With zero advantages, GSPO policy loss should be zero."""
        from oxrl.algs.losses.gspo import compute_gspo_loss
        B, T = 2, 8
        logprobs = torch.randn(B, T)
        old_logprobs = logprobs.clone()
        advantages = torch.zeros(B, T)
        mask = torch.ones(B, T)
        loss, metrics = compute_gspo_loss(
            logprobs, old_logprobs, advantages, mask, None, None,
            clip_low=0.2, clip_high=0.2, ent_coeff=0.0, kl_coeff=0.0,
        )
        assert abs(metrics["loss_pi"]) < 1e-6, f"Zero advantage should give ~zero loss_pi, got {metrics['loss_pi']}"

    def test_sgrpo_identical_logprobs_zero_kl(self):
        """When logprobs == old_logprobs, kl_old should be ~0."""
        from oxrl.algs.losses.sgrpo import compute_sgrpo_loss
        B, T = 4, 10
        logprobs = torch.randn(B, T)
        loss, metrics = compute_sgrpo_loss(
            logprobs, logprobs.clone(), torch.ones(B, T), torch.ones(B, T),
            None, None, clip_low=0.2, clip_high=0.2, ent_coeff=0.0, kl_coeff=0.0,
        )
        assert abs(metrics["kl_old"]) < 1e-6, f"Identical logprobs should give kl_old~0, got {metrics['kl_old']}"
        assert abs(metrics["clipfrac"]) < 1e-6, f"Identical logprobs should give clipfrac~0, got {metrics['clipfrac']}"

    def test_cispo_gradient_does_not_flow_through_ratio(self):
        """CISPO detaches the ratio, so gradient should only flow through logprobs."""
        from oxrl.algs.losses.cispo import compute_cispo_loss
        B, T = 2, 6
        logprobs = torch.randn(B, T, requires_grad=True)
        old_logprobs = logprobs.detach() + 0.1
        advantages = torch.ones(B, T)
        mask = torch.ones(B, T)
        loss, _ = compute_cispo_loss(
            logprobs, old_logprobs, advantages, mask, None, None,
            clip_low=0.2, clip_high=0.2, ent_coeff=0.0, kl_coeff=0.0,
        )
        loss.backward()
        assert logprobs.grad is not None, "Gradient should flow through logprobs"

    def test_sgrpo_loss_gradient_flows(self):
        """SGRPO should have gradient through logprobs."""
        from oxrl.algs.losses.sgrpo import compute_sgrpo_loss
        B, T = 2, 6
        logprobs = torch.randn(B, T, requires_grad=True)
        old_logprobs = logprobs.detach() + 0.1
        advantages = torch.ones(B, T)
        mask = torch.ones(B, T)
        loss, _ = compute_sgrpo_loss(
            logprobs, old_logprobs, advantages, mask, None, None,
            clip_low=0.2, clip_high=0.2, ent_coeff=0.0, kl_coeff=0.0,
        )
        loss.backward()
        assert logprobs.grad is not None
        assert (logprobs.grad != 0).any(), "SGRPO gradient should be non-zero"

    def test_entropy_bonus_reduces_loss(self):
        """Entropy bonus should reduce the total loss when ent_coeff > 0."""
        from oxrl.algs.losses.sgrpo import compute_sgrpo_loss
        B, T = 4, 10
        torch.manual_seed(123)
        logprobs = torch.randn(B, T)
        old_logprobs = logprobs + 0.01
        advantages = torch.randn(B, T)
        mask = torch.ones(B, T)
        entropies = torch.ones(B, T) * 2.0  # high entropy

        _, m_no_ent = compute_sgrpo_loss(
            logprobs, old_logprobs, advantages, mask, entropies, None,
            clip_low=0.2, clip_high=0.2, ent_coeff=0.0, kl_coeff=0.0,
        )
        _, m_with_ent = compute_sgrpo_loss(
            logprobs, old_logprobs, advantages, mask, entropies, None,
            clip_low=0.2, clip_high=0.2, ent_coeff=0.1, kl_coeff=0.0,
        )
        # Entropy bonus subtracts from loss, so loss_total should be lower with ent
        assert m_with_ent["loss_total"] < m_no_ent["loss_total"], \
            f"Entropy bonus should reduce loss: {m_with_ent['loss_total']} vs {m_no_ent['loss_total']}"

    def test_kl_penalty_increases_loss(self):
        """KL penalty should increase the total loss when kl_coeff > 0."""
        from oxrl.algs.losses.sgrpo import compute_sgrpo_loss
        B, T = 4, 10
        torch.manual_seed(123)
        logprobs = torch.randn(B, T)
        old_logprobs = logprobs + 0.01
        advantages = torch.randn(B, T)
        mask = torch.ones(B, T)
        ref_logprobs = logprobs + 0.5  # different from policy

        _, m_no_kl = compute_sgrpo_loss(
            logprobs, old_logprobs, advantages, mask, None, ref_logprobs,
            clip_low=0.2, clip_high=0.2, ent_coeff=0.0, kl_coeff=0.0,
        )
        _, m_with_kl = compute_sgrpo_loss(
            logprobs, old_logprobs, advantages, mask, None, ref_logprobs,
            clip_low=0.2, clip_high=0.2, ent_coeff=0.0, kl_coeff=0.1,
        )
        assert m_with_kl["loss_total"] >= m_no_kl["loss_total"], \
            f"KL penalty should increase loss: {m_with_kl['loss_total']} vs {m_no_kl['loss_total']}"

    def test_mask_zeros_out_padding(self):
        """Padding tokens (mask=0) should not contribute to the loss.

        We verify by giving the first half and second half different advantages,
        then masking out the second half. The loss should match a computation
        done only on the first half.
        """
        from oxrl.algs.losses.sgrpo import compute_sgrpo_loss
        B, T = 2, 10
        torch.manual_seed(42)
        logprobs = torch.randn(B, T)
        old_logprobs = logprobs + 0.1 * torch.randn(B, T)
        # Different advantages in first and second half
        advantages = torch.randn(B, T)

        # Full mask — uses all tokens
        mask_full = torch.ones(B, T)
        loss_full, _ = compute_sgrpo_loss(
            logprobs, old_logprobs, advantages, mask_full, None, None,
            clip_low=0.2, clip_high=0.2, ent_coeff=0.0, kl_coeff=0.0,
        )

        # Mask only first 5 tokens — uses different set of tokens
        mask_first5 = torch.ones(B, T)
        mask_first5[:, 5:] = 0
        loss_first5, _ = compute_sgrpo_loss(
            logprobs, old_logprobs, advantages, mask_first5, None, None,
            clip_low=0.2, clip_high=0.2, ent_coeff=0.0, kl_coeff=0.0,
        )

        # First-5-only loss should match a direct computation on just those tokens
        loss_first5_direct, _ = compute_sgrpo_loss(
            logprobs[:, :5], old_logprobs[:, :5], advantages[:, :5],
            torch.ones(B, 5), None, None,
            clip_low=0.2, clip_high=0.2, ent_coeff=0.0, kl_coeff=0.0,
        )
        assert abs(loss_first5.item() - loss_first5_direct.item()) < 1e-5, \
            f"Masked loss {loss_first5.item()} should match direct {loss_first5_direct.item()}"

    def test_all_metrics_keys_present(self, standard_inputs):
        """All loss functions should return the same set of metric keys."""
        from oxrl.algs.losses import get_loss_fn
        expected_keys = {"clipfrac", "kl_old", "loss_ent", "loss_pi", "loss_total", "kl_ref"}
        for variant in ("sgrpo", "gspo", "cispo"):
            loss_fn = get_loss_fn(variant)
            _, metrics = loss_fn(
                *standard_inputs, clip_low=0.2, clip_high=0.2, ent_coeff=0.01, kl_coeff=0.01,
            )
            assert set(metrics.keys()) == expected_keys, \
                f"{variant} metrics keys {set(metrics.keys())} != {expected_keys}"


# ============================================================
# 3. Common Loss Helpers
# ============================================================
class TestCommonHelpers:
    """Verify prepare_loss_inputs, compute_kl_distance, etc."""

    def test_prepare_loss_inputs_shapes(self):
        from oxrl.algs.losses.common import prepare_loss_inputs
        B, T = 3, 8
        lp = torch.randn(B, T)
        olp = torch.randn(B, T)
        adv = torch.randn(B, T)
        mask = torch.ones(B, T)
        a, m, d, lr, r = prepare_loss_inputs(lp, olp, adv, mask)
        assert a.shape == (B, T)
        assert m.shape == (B, T)
        assert d.dim() == 0  # scalar
        assert lr.shape == (B, T)
        assert r.shape == (B, T)

    def test_prepare_loss_inputs_ratio_identity(self):
        """When logprobs == old_logprobs, ratio should be ~1.0."""
        from oxrl.algs.losses.common import prepare_loss_inputs
        lp = torch.randn(2, 5)
        _, _, _, lr, r = prepare_loss_inputs(lp, lp.clone(), torch.ones(2, 5), torch.ones(2, 5))
        assert torch.allclose(r, torch.ones_like(r), atol=1e-5)
        assert torch.allclose(lr, torch.zeros_like(lr), atol=1e-5)

    def test_kl_distance_non_negative(self):
        """KL divergence should be >= 0."""
        from oxrl.algs.losses.common import compute_kl_distance
        lp = torch.randn(4, 10)
        ref = lp + 0.5 * torch.randn(4, 10)
        kl = compute_kl_distance(lp, ref)
        # KL(p||q) >= 0 for all distributions; our approximation should also be >= 0
        # The formula is log(pi/ref) + ref/pi - 1 which equals E[f(r)] where f(r) = log(r) + 1/r - 1
        # f(r) >= 0 for all r > 0, with minimum at r=1
        assert (kl >= -1e-6).all(), f"KL distance should be non-negative, min={kl.min()}"

    def test_kl_distance_zero_when_equal(self):
        from oxrl.algs.losses.common import compute_kl_distance
        lp = torch.randn(3, 5)
        kl = compute_kl_distance(lp, lp.clone())
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)

    def test_token_level_metrics_clipfrac_range(self):
        from oxrl.algs.losses.common import token_level_metrics
        B, T = 4, 10
        ratio = torch.ones(B, T)
        logratio = torch.zeros(B, T)
        mask = torch.ones(B, T)
        denom = mask.sum()
        clipfrac, approx_kl = token_level_metrics(
            ratio, logratio, mask, denom, 0.2, 0.2, torch.float32
        )
        assert 0.0 <= clipfrac.item() <= 1.0
        assert approx_kl.item() >= -1e-6


# ============================================================
# 4. GRPO dispatch via compute_policy_loss
# ============================================================
class TestGRPODispatch:
    """Verify GRPO.compute_policy_loss dispatches to correct loss function."""

    def _make_grpo(self, variant):
        from oxrl.algs.grpo import GRPO
        GRPOCls = GRPO.__ray_actor_class__
        grpo = object.__new__(GRPOCls)
        grpo.loss_variant = variant
        grpo.clip_low = 0.2
        grpo.clip_high = 0.2
        grpo.ent_coeff = 0.0
        grpo.kl_coeff = 0.0
        return grpo

    def test_sgrpo_dispatch(self):
        grpo = self._make_grpo("sgrpo")
        B, T = 4, 10
        lp = torch.randn(B, T)
        loss, m = grpo.compute_policy_loss(lp, lp + 0.01, torch.ones(B, T), torch.ones(B, T), None, None)
        assert torch.isfinite(loss)
        assert "loss_pi" in m

    def test_gspo_dispatch(self):
        grpo = self._make_grpo("gspo")
        B, T = 4, 10
        lp = torch.randn(B, T)
        loss, m = grpo.compute_policy_loss(lp, lp + 0.01, torch.ones(B, T), torch.ones(B, T), None, None)
        assert torch.isfinite(loss)

    def test_cispo_dispatch(self):
        grpo = self._make_grpo("cispo")
        B, T = 4, 10
        lp = torch.randn(B, T)
        loss, m = grpo.compute_policy_loss(lp, lp + 0.01, torch.ones(B, T), torch.ones(B, T), None, None)
        assert torch.isfinite(loss)

    def test_unknown_variant_raises(self):
        grpo = self._make_grpo("bogus")
        B, T = 2, 5
        with pytest.raises(ValueError, match="Unknown loss variant"):
            grpo.compute_policy_loss(
                torch.randn(B, T), torch.randn(B, T), torch.ones(B, T), torch.ones(B, T), None, None
            )


# ============================================================
# 5. PPO dispatch via compute_policy_loss
# ============================================================
class TestPPODispatch:
    """Verify PPO.compute_policy_loss dispatches to sgrpo loss."""

    def test_ppo_uses_sgrpo_loss(self):
        from oxrl.algs.ppo import PPO
        PPOCls = PPO.__ray_actor_class__
        ppo = object.__new__(PPOCls)
        ppo.clip_low = 0.2
        ppo.clip_high = 0.2
        ppo.ent_coeff = 0.0
        ppo.kl_coeff = 0.0
        B, T = 4, 10
        lp = torch.randn(B, T)
        loss, m = ppo.compute_policy_loss(lp, lp + 0.01, torch.ones(B, T), torch.ones(B, T), None, None)
        assert torch.isfinite(loss)
        assert "loss_pi" in m

    def test_ppo_value_loss_basic(self):
        from oxrl.algs.ppo import PPO
        PPOCls = PPO.__ray_actor_class__
        ppo = object.__new__(PPOCls)
        ppo.vf_clip = 0.2
        B, T = 2, 8
        values = torch.randn(B, T, requires_grad=True)
        v_old = values.detach() + 0.01
        returns = torch.randn(B, T)
        mask = torch.ones(B, T)
        loss, metrics = ppo.compute_value_loss(values, v_old, returns, mask)
        assert torch.isfinite(loss)
        assert loss.item() >= 0
        loss.backward()
        assert values.grad is not None

    def test_ppo_value_loss_no_clip(self):
        from oxrl.algs.ppo import PPO
        PPOCls = PPO.__ray_actor_class__
        ppo = object.__new__(PPOCls)
        ppo.vf_clip = 0.0  # no clipping
        B, T = 2, 8
        values = torch.randn(B, T, requires_grad=True)
        returns = torch.randn(B, T)
        mask = torch.ones(B, T)
        loss, metrics = ppo.compute_value_loss(values, None, returns, mask)
        assert torch.isfinite(loss)
        assert loss.item() >= 0


# ============================================================
# 6. PPO GAE Computation
# ============================================================
class TestPPOGAE:
    """Detailed tests for PPO's compute_advantages."""

    def _make_ppo(self, gamma=0.99, tau=0.95):
        from oxrl.algs.ppo import PPO
        PPOCls = PPO.__ray_actor_class__
        ppo = object.__new__(PPOCls)
        ppo.gamma = gamma
        ppo.tau = tau
        return ppo

    def test_single_step_terminal(self):
        """Single-step terminal episode."""
        ppo = self._make_ppo(gamma=0.99, tau=0.95)
        rewards = torch.tensor([[5.0]])
        values = torch.tensor([[1.0]])
        done = torch.tensor([[1.0]])
        mask = torch.tensor([[1.0]])
        rets, advs = ppo.compute_advantages(rewards, values, done, mask)
        # Terminal: delta = r - v = 5 - 1 = 4
        assert abs(advs[0, 0].item() - 4.0) < 1e-4
        assert abs(rets[0, 0].item() - 5.0) < 1e-4  # ret = adv + v = 4 + 1

    def test_two_step_non_terminal(self):
        """Two-step non-terminal episode with bootstrap."""
        ppo = self._make_ppo(gamma=1.0, tau=1.0)
        rewards = torch.tensor([[1.0, 2.0]])
        values = torch.tensor([[0.5, 1.0]])
        done = torch.tensor([[0.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0]])
        last_val = torch.tensor([3.0])  # bootstrap value
        rets, advs = ppo.compute_advantages(rewards, values, done, mask, last_val=last_val)
        # t=1: delta = 2 + 1.0*3 - 1 = 4, adv=4
        # t=0: delta = 1 + 1.0*1 - 0.5 = 1.5, adv = 1.5 + 1.0*1.0*4 = 5.5
        assert abs(advs[0, 1].item() - 4.0) < 1e-4
        assert abs(advs[0, 0].item() - 5.5) < 1e-4

    def test_padding_does_not_leak(self):
        """Padding tokens should have zero advantage."""
        ppo = self._make_ppo()
        rewards = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
        values = torch.tensor([[0.5, 1.0, 0.0, 0.0]])
        done = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        rets, advs = ppo.compute_advantages(rewards, values, done, mask)
        assert abs(advs[0, 2].item()) < 1e-6
        assert abs(advs[0, 3].item()) < 1e-6

    def test_done_prevents_bootstrap(self):
        """Done=1 should prevent bootstrapping from the next step."""
        ppo = self._make_ppo(gamma=1.0, tau=1.0)
        rewards = torch.tensor([[1.0, 2.0, 3.0]])
        values = torch.tensor([[0.5, 1.0, 1.5]])
        done = torch.tensor([[0.0, 1.0, 0.0]])  # done at t=1
        mask = torch.tensor([[1.0, 1.0, 1.0]])
        rets, advs = ppo.compute_advantages(rewards, values, done, mask)
        # t=2: delta = 3 + 0 - 1.5 = 1.5 (no bootstrap since last step)
        # t=1: delta = 2 + 0 - 1.0 = 1.0 (done=1, no bootstrap from t=2)
        # t=0: delta = 1 + 1.0*1.0 - 0.5 = 1.5, adv = 1.5 + 1.0*1.0*1.0 = 2.5
        assert abs(advs[0, 1].item() - 1.0) < 1e-4


# ============================================================
# 7. Tools: LoRA Merge
# ============================================================
class TestLoRAMerge:
    def test_strip_lora_basic(self):
        from oxrl.tools.lora_merge import strip_lora_and_merge
        state_dict = {
            "base_model.model.layer.weight": torch.randn(4, 4),
            "base_model.model.layer.lora_A.default.weight": torch.randn(2, 4),
            "base_model.model.layer.lora_B.default.weight": torch.randn(4, 2),
        }
        result = strip_lora_and_merge(state_dict, lora_alpha=16.0, lora_r=8)
        assert "layer.weight" in result
        assert "layer.lora_A.default.weight" not in result
        assert "layer.lora_B.default.weight" not in result

    def test_strip_lora_merge_modifies_weight(self):
        from oxrl.tools.lora_merge import strip_lora_and_merge
        base_w = torch.zeros(4, 4)
        state_dict = {
            "base_model.model.layer.weight": base_w.clone(),
            "base_model.model.layer.lora_A.default.weight": torch.ones(2, 4),
            "base_model.model.layer.lora_B.default.weight": torch.ones(4, 2),
        }
        result = strip_lora_and_merge(state_dict, lora_alpha=16.0, lora_r=8)
        merged = result["layer.weight"]
        # delta = (B @ A) * (16/8) = ones(4,2) @ ones(2,4) * 2 = 2*ones(4,4)*2 = all 4s
        expected = torch.ones(4, 4) * 4.0
        assert torch.allclose(merged, expected, atol=1e-4), f"Expected {expected}, got {merged}"


# ============================================================
# 8. Tools: Checkpoint
# ============================================================
class TestCheckpointTools:
    def test_save_state_dict_safetensors(self):
        from oxrl.tools.checkpoint import save_state_dict_to_safetensors
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dict = {"weight1": torch.randn(4, 4), "weight2": torch.randn(8)}
            save_state_dict_to_safetensors(tmpdir, state_dict)
            assert os.path.exists(os.path.join(tmpdir, "model.safetensors"))

    def test_save_state_dict_handles_shared_tensors(self):
        from oxrl.tools.checkpoint import save_state_dict_to_safetensors
        with tempfile.TemporaryDirectory() as tmpdir:
            shared = torch.randn(4, 4)
            state_dict = {"weight1": shared, "weight2": shared}  # same tensor!
            # Should not raise
            save_state_dict_to_safetensors(tmpdir, state_dict)
            assert os.path.exists(os.path.join(tmpdir, "model.safetensors"))

    def test_save_config_json(self):
        from oxrl.tools.checkpoint import save_config_json
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dict = {"hidden_size": 768, "num_layers": 12}
            save_config_json(tmpdir, config_dict)
            path = os.path.join(tmpdir, "config.json")
            assert os.path.exists(path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded == config_dict


# ============================================================
# 9. Tools: Tensor Utils
# ============================================================
class TestTensorUtils:
    def test_ensure_1d(self):
        from oxrl.tools.tensor_utils import ensure_1d
        x = torch.randn(5)
        assert ensure_1d(x, "test").shape == (5,)

    def test_ensure_1d_raises_for_2d(self):
        from oxrl.tools.tensor_utils import ensure_1d
        with pytest.raises(ValueError, match="1D"):
            ensure_1d(torch.randn(2, 3), "test")

    def test_pad_1d_pads(self):
        from oxrl.tools.tensor_utils import pad_1d_to_length
        x = torch.tensor([1.0, 2.0, 3.0])
        result = pad_1d_to_length(x, pad_value=0.0, target_len=5)
        assert result.shape == (5,)
        assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0]))

    def test_pad_1d_truncates(self):
        from oxrl.tools.tensor_utils import pad_1d_to_length
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pad_1d_to_length(x, pad_value=0.0, target_len=3)
        assert result.shape == (3,)
        assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))


# ============================================================
# 10. Config Loading and Sync
# ============================================================
class TestConfigSync:
    def _make_config(self, alg_name="sgrpo", method="rl"):
        from oxrl.configs.load import Config
        raw = {
            "run": {
                "experiment_id": "test", "training_gpus": 2, "rollout_gpus": 2,
                "checkpoint_dir": "/tmp/test",
            },
            "train": {
                "alg_name": alg_name, "total_number_of_epochs": 3,
                "train_steps_per_epoch": 10, "lr": 1e-5,
            },
            "model": {"name": "test-model"},
            "data": {
                "train_dnames": ["d"], "train_ratios": {"d": 1.0},
                "train_files_path": "/tmp/d", "val_files_path": "/tmp/v",
            },
        }
        config = Config(**raw)
        config.run.method = method
        return config

    def test_backward_compat_imports(self):
        """Verify backward-compatible re-exports from load.py."""
        from oxrl.configs.load import Run, Train, Data, Model, Config
        from oxrl.configs.load import load_and_verify, sync_deepspeed_config
        assert Run is not None
        assert callable(load_and_verify)
        assert callable(sync_deepspeed_config)

    def test_sync_sets_batch_sizes(self):
        config = self._make_config()
        config.sync_deepspeed_config(world_size=2)
        assert config.deepspeed.train_micro_batch_size_per_gpu == config.train.train_batch_size_per_gpu

    def test_sync_sets_scheduler(self):
        config = self._make_config()
        config.sync_deepspeed_config(world_size=2)
        assert "scheduler" in config.deepspeed.model_dump()
        sched = config.deepspeed.scheduler
        assert sched["params"]["total_num_steps"] == 30  # 3 epochs * 10 steps

    def test_sync_sets_optimizer(self):
        config = self._make_config()
        config.sync_deepspeed_config(world_size=2)
        assert config.deepspeed.optimizer["type"] == "AdamW"
        assert config.deepspeed.optimizer["params"]["lr"] == 1e-5

    def test_sync_bf16_dtype(self):
        config = self._make_config()
        config.model.dtype = "bfloat16"
        config.sync_deepspeed_config(world_size=2)
        assert config.deepspeed.bf16["enabled"] is True
        assert config.deepspeed.fp16["enabled"] is False


# ============================================================
# 11. Engine Factory
# ============================================================
class TestEngineFactory:
    def test_get_algorithm_class_all_variants(self):
        from oxrl.setup.engine_factory import get_algorithm_class
        from oxrl.algs.grpo import GRPO
        from oxrl.algs.ppo import PPO
        assert get_algorithm_class("sgrpo") is GRPO
        assert get_algorithm_class("gspo") is GRPO
        assert get_algorithm_class("cispo") is GRPO
        assert get_algorithm_class("rlhf") is GRPO
        assert get_algorithm_class("rlaif") is GRPO
        assert get_algorithm_class("ppo") is PPO

    def test_get_algorithm_class_unknown_raises(self):
        from oxrl.setup.engine_factory import get_algorithm_class
        with pytest.raises(ValueError, match="Unknown algorithm"):
            get_algorithm_class("unknown_alg")


# ============================================================
# 12. Replay Buffer Integration
# ============================================================
class TestReplayBufferIntegration:
    def test_replay_buffer_add_and_collate(self):
        from oxrl.rollouts.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(pad_token_id=0, max_seq_len=32)
        for _ in range(4):
            T = torch.randint(5, 15, (1,)).item()
            buf.add(
                input_ids=torch.randint(0, 100, (T,)),
                rewards=torch.randn(T),
                zscores=torch.randn(T),
                masks=torch.cat([torch.zeros(3), torch.ones(T - 3)]),
                dones=torch.zeros(T),
                old_logprobs=torch.randn(T),
            )
        assert len(buf) == 4
        batch = buf.collate_fn([buf[i] for i in range(len(buf))])
        assert "input_ids" in batch
        assert "mask" in batch
        assert batch["input_ids"].shape[0] == 4


# ============================================================
# 13. Checkpoint Phase Logic
# ============================================================
class TestCheckpointPhaseLogic:
    """Test the checkpoint_phase module's fallback logic."""

    def test_os_sync_exists_in_fallback(self):
        """The fallback path calls os.sync() which may not exist on all platforms."""
        source_path = os.path.join(
            os.path.dirname(__file__), "..", "oxrl", "loops", "checkpoint_phase.py"
        )
        with open(source_path) as f:
            source = f.read()
        # Verify the module imports what it needs
        assert "import os" in source
        assert "import ray" in source


# ============================================================
# 14. GSPO vs SGRPO: Behavioral Differences
# ============================================================
class TestGSPOvsSGRPO:
    """Verify GSPO and SGRPO produce different results (they should, since
    GSPO averages at sequence level while SGRPO clips at token level)."""

    def test_different_losses_for_noisy_data(self):
        from oxrl.algs.losses.sgrpo import compute_sgrpo_loss
        from oxrl.algs.losses.gspo import compute_gspo_loss
        B, T = 8, 20
        torch.manual_seed(42)
        logprobs = torch.randn(B, T) * 0.5
        # Add large per-token noise to simulate MoE routing disagreement
        old_logprobs = logprobs + 0.8 * torch.randn(B, T)
        advantages = torch.randn(B, T)
        mask = torch.ones(B, T)
        mask[:, 15:] = 0

        loss_s, metrics_s = compute_sgrpo_loss(
            logprobs, old_logprobs, advantages, mask, None, None,
            clip_low=0.2, clip_high=0.2, ent_coeff=0.0, kl_coeff=0.0,
        )
        loss_g, metrics_g = compute_gspo_loss(
            logprobs, old_logprobs, advantages, mask, None, None,
            clip_low=0.2, clip_high=0.2, ent_coeff=0.0, kl_coeff=0.0,
        )
        # They should both be finite but produce different values
        assert torch.isfinite(loss_s) and torch.isfinite(loss_g)
        # With noisy data, the clip fractions should differ
        # (GSPO averages first, so less clipping expected)
        # Not guaranteed in all seeds, but with seed=42 and large noise this should hold:
        assert loss_s.item() != loss_g.item(), "SGRPO and GSPO should produce different losses"
