# oxRL Post-Training Engineer Memory

## Project Structure
- Main training loop: `/ceph/workspace/erik/oxRL/main_rl.py`
- Algorithms: `oxrl/algs/grpo.py` (SGRPO, GSPO, CISPO), `oxrl/algs/ppo.py` (PPO with value head)
- Base class: `oxrl/algs/base.py`
- Rollout engine: `oxrl/rollouts/vllm_engine.py` (vLLM-based)
- Replay buffer: `oxrl/rollouts/replay_buffer.py` (pred-aligned data)
- Tests: `tests/test_bugs.py` (68 tests), `tests/test_mop_refactoring.py` (52 tests), `tests/fast_*.yaml` (quick verification configs)
- Test data: `data/gsm8k_mini_train.parquet` (16 prompts), `data/gsm8k_mini_test.parquet` (4 prompts)
- MOP refactoring: Loss functions in `oxrl/algs/losses/`, tools in `oxrl/tools/`, config split in `oxrl/configs/`, setup in `oxrl/setup/`, loops in `oxrl/loops/`

## Verified Algorithms (all PASS with Qwen2.5-0.5B-Instruct)
- SGRPO: token-level clipped surrogate (default for dense models)
- GSPO: sequence-level clipped surrogate (for MoE models)
- CISPO: conservative clipped ratio as weight on log-prob
- PPO: full PPO with value head + GAE advantage estimation

## Known Bugs Found and Fixed (see bugs-fixed.md for details)
1. PPO gradient leakage: hidden_states not detached in value_forward
2. Ray auto-resolves ObjectRef args: save_checkpoint/refresh used ray.get() on already-resolved dict
3. Safetensors shared tensors: Qwen2.5 tied embeddings need clone() before save_file
4. PPO mask validation too strict: pred-aligned masks have legitimate 0->1 transitions
5. Missing config.json in checkpoints: popped from state_dict but not saved to disk
6. CUDA_HOME not propagated to Ray workers: DeepSpeed import failed
7. VLLM_USE_V1=0 not propagated to rollout workers
8. vLLM 0.15.1 VLLM_USE_V1=0 obsolete: V1 is the only engine now. Must use VLLM_ENABLE_V1_MULTIPROCESSING=0 instead to prevent subprocess GPU isolation issues with Ray. Fixed in vllm_engine.py and engine_factory.py.

## Important Patterns
- Replay buffer stores pred-aligned data: masks[i]=1 means position i predicts a valid response token
- GRPO/PPO slice with [:, :-1] to go from token-aligned to pred-aligned logprobs
- Ray auto-resolves ObjectRef args passed to .remote() calls -- never call ray.get() inside remote methods on args that were ObjectRefs
- Use CUDA_VISIBLE_DEVICES=4,5 (or 6,7) -- GPUs 0-3 often occupied
- CUDA_HOME is at `/ceph/workspace/erik/oxRL/cuda_env` -- auto-detected by import_deepspeed_safely()
- For fast verification, use mini dataset (16 prompts) with fast_*.yaml configs (~1 min per algorithm)
