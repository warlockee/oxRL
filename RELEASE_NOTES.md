# Release Notes

## Version 1.1.0 (2026-02-26)

### 🚀 Features

- **Added Qwen 3 & 3.5 Series Support**: Onboarded a wide range of new models from the Qwen 3 and 3.5 families, including instruct, reasoning, vision, and coding variants.
  - **Qwen 3.5**: `Qwen/Qwen3.5-35B-A3B`, `Qwen/Qwen3.5-27B`
  - **Qwen 3**: `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-4B-Thinking-2507`, `Qwen/Qwen3-VL-2B-Instruct`, `Qwen/Qwen3-VL-4B-Instruct`, and `Qwen/Qwen3-Coder-Next`.
- **Reasoning Reward Function**: Updated `reasoning_reward_func` to natively support the `<think>` tags used by the latest Qwen models.
- **Local CUDA Toolkit Installation**: Added support for installing and using a local CUDA Toolkit via Conda, ensuring a fully functional `nvcc` and resolving DeepSpeed compilation issues in restricted environments.

### ✅ Verification

- **Full RL Training Cycle Verified**: Successfully trained `Qwen/Qwen3-0.6B` through the entire RL pipeline, from data preprocessing to vLLM rollout and DeepSpeed optimization. This confirms the framework is fully operational and ready for large-scale training runs.

### 🐛 Bug Fixes

- **DeepSpeed Initialization**: Implemented a robust bypass for the `nvcc` check in DeepSpeed, allowing the framework to initialize correctly even when the CUDA Toolkit is not in the system's default path.
- **Model ID Corrections**: Corrected the HuggingFace model IDs for the Qwen 3 series, ensuring the scout agent can find and download them.
