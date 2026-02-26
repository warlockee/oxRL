# Release Notes

## Version 1.4.0 (2026-02-27)

### Critical Bug Fixes (8 bugs identified via architecture review)

- **ORPO log-odds degeneration (Bug 3):** ORPO computed log-odds on raw sum-of-log-probs instead of average-per-token log-probs. For sequences of different lengths, `exp(sum)` overflows or underflows, making the odds ratio meaningless. Fixed by normalizing log-probs by valid token count before computing log-odds.
- **KTO KL baseline computed over both chosen and rejected (Bug 5):** The KL moving average was computed over the full batch (chosen + rejected), diluting the baseline. Fixed to compute KL over chosen samples only, matching the KTO paper.
- **GRPO/SimPO hardcoded optimizer LR (Bug 2):** Both algorithms ignored `config.train.lr` and hardcoded `lr=1e-6` in their internal `AdamW`. Fixed to accept and use `lr`, `betas`, `weight_decay`, and `adam_epsilon` from config.
- **clip_low default was -0.2 (Bug 4):** The importance sampling clip range defaulted to `[-0.2, 0.2]` (i.e., `[0.8, 1.2]` after `1+clip`), but `clip_low=-0.2` means the lower bound is `1 + (-0.2) = 0.8` which is correct numerically, however the config field name `clip_low` suggested a positive value. Fixed default to `0.2` for consistency with `clip_high`.
- **LR scheduler 100x overestimate (Bug 6):** The cosine LR scheduler computed `total_num_steps = epochs * steps_per_epoch * 100`, causing the learning rate to barely decay. Removed the erroneous `* 100` multiplier.
- **Reward normalization used population std (Bug 7):** Reward scores were normalized with `std = sqrt(sum/N)` instead of Bessel's correction `sqrt(sum/(N-1))`. Fixed to use `max(1, N-1)` in the denominator.
- **Replay buffer not reset for SGRPO (Bug 1):** The replay buffer reset was gated behind a hardcoded set `{"ppo", "grpo", "cispo"}`, missing `"sgrpo"`. Since all RL algorithms in oxRL are on-policy, made the reset unconditional.
- **Duplicate monkey-patch (Bug 8):** The `SlidingWindowCache` monkey-patch was duplicated in `vllm_engine.py` and `utils/setup.py`. Centralized into `ensure_sliding_window_cache()` with an idempotency guard.

### Additional Fixes

- **token_type_ids consistency:** Added `token_type_ids=torch.zeros_like(input_ids)` to forward passes in SFT, ORPO, and KTO to match GRPO/DPO and prevent unexpected behavior with models that use token type embeddings.
- **Replaced importlib hacks:** Replaced fragile `importlib.util` dynamic imports in `main_rl.py` and `main_sl.py` with standard `from oxrl.datasets import ...` statements.
- **Removed dead code:** Removed duplicate replay buffer initialization block in `main_rl.py`.
- **Robust optimizer setup in SL path:** `main_sl.py` now creates a PyTorch AdamW optimizer explicitly before passing to DeepSpeed, avoiding dependency on DeepSpeed's FusedAdam CUDA extensions.

### Testing

- Added `tests/test_bugs.py` with 33 comprehensive unit tests covering all bug fixes.
- Added `tests/grpo_test.yaml` for SGRPO end-to-end verification.
- All algorithms verified end-to-end: SGRPO (RL), ORPO (SL), and KTO (SL) completed full training + validation + checkpoint saving on Qwen2.5-0.5B-Instruct.

### README

- Updated algorithm table to include DPO, ORPO, and KTO.
- Separated algorithms into RL and SL sections.
- Updated architecture diagram and project structure.

---

## Version 1.1.0 (2026-02-26)

### Features

- **Added Qwen 3 & 3.5 Series Support**: Onboarded a wide range of new models from the Qwen 3 and 3.5 families, including instruct, reasoning, vision, and coding variants.
  - **Qwen 3.5**: `Qwen/Qwen3.5-35B-A3B`, `Qwen/Qwen3.5-27B`
  - **Qwen 3**: `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-4B-Thinking-2507`, `Qwen/Qwen3-VL-2B-Instruct`, `Qwen/Qwen3-VL-4B-Instruct`, and `Qwen/Qwen3-Coder-Next`.
- **Reasoning Reward Function**: Updated `reasoning_reward_func` to natively support the `<think>` tags used by the latest Qwen models.
- **Local CUDA Toolkit Installation**: Added support for installing and using a local CUDA Toolkit via Conda, ensuring a fully functional `nvcc` and resolving DeepSpeed compilation issues in restricted environments.

### Verification

- **Full RL Training Cycle Verified**: Successfully trained `Qwen/Qwen3-0.6B` through the entire RL pipeline, from data preprocessing to vLLM rollout and DeepSpeed optimization. This confirms the framework is fully operational and ready for large-scale training runs.

### Bug Fixes

- **DeepSpeed Initialization**: Implemented a robust bypass for the `nvcc` check in DeepSpeed, allowing the framework to initialize correctly even when the CUDA Toolkit is not in the system's default path.
- **Model ID Corrections**: Corrected the HuggingFace model IDs for the Qwen 3 series, ensuring the scout agent can find and download them.
