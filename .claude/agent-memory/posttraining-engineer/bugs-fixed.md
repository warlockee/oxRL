# Bugs Found and Fixed in oxRL

## Bug 1: PPO Gradient Leakage from Value Loss to Policy Parameters
**File**: `oxrl/algs/ppo.py` - `value_forward()`
**Root cause**: `hidden_states = output.hidden_states[-1]` was not detached before passing to value_head. When `vl_loss.backward()` ran, gradients leaked into policy engine parameters.
**Fix**: Added `.detach()`: `hidden_states = output.hidden_states[-1].detach()`
**Impact**: Without fix, policy gradients get corrupted by value loss gradients on subsequent micro-batches within the same gradient accumulation boundary.

## Bug 2: ray.get() on Already-Resolved ObjectRef Arguments
**Files**: `oxrl/algs/grpo.py`, `oxrl/algs/ppo.py`, `oxrl/rollouts/vllm_engine.py`
**Root cause**: Ray automatically resolves ObjectRef arguments when calling `.remote()`. Inside `save_checkpoint` and `refresh_model_from_state_dict`, the code called `ray.get(state_dict_ref)` but `state_dict_ref` was already an OrderedDict (not an ObjectRef).
**Fix**: Changed `state_dict = ray.get(state_dict_ref)` to `state_dict = state_dict_ref` in all 3 files.
**Key pattern**: Never call `ray.get()` on arguments received by a remote method -- they're already resolved.

## Bug 3: Safetensors Shared Tensor Error on Checkpoint Save
**Files**: `oxrl/algs/grpo.py`, `oxrl/algs/ppo.py`, `oxrl/rollouts/vllm_engine.py`
**Root cause**: `safetensors.save_file()` raises RuntimeError when tensors share memory (e.g., Qwen2.5's tied `lm_head.weight` and `model.embed_tokens.weight`).
**Fix**: Added shared-tensor detection loop that clones tensors sharing the same data_ptr before saving:
```python
seen_ptrs = {}
for k, v in state_dict.items():
    ptr = v.data_ptr()
    if ptr in seen_ptrs:
        state_dict[k] = v.clone()
    else:
        seen_ptrs[ptr] = k
```

## Bug 4: PPO compute_advantages Mask Validation Too Strict
**File**: `oxrl/algs/ppo.py` - `compute_advantages()`
**Root cause**: Check `if (mask[:, 1:] & (~mask[:, :-1])).any()` rejected ANY 0->1 transition. But pred-aligned masks have a legitimate 0->1 transition at the prompt-to-response boundary: `[0,0,...,0,1,1,...,1,0,0,...,0]`.
**Fix**: Changed to allow exactly one 0->1 transition per row: `if transitions_01.sum(dim=1).max().item() > 1`.

## Bug 5: Missing config.json in Checkpoint Directory
**File**: `main_rl.py`
**Root cause**: `__model_config_dict__` was popped from state_dict before `ray.put()`, then inside `save_checkpoint` the pop returned None so config was never saved.
**Fix**: Added config.json save in main_rl.py driver after tokenizer save, before launching parallel save/refresh.

## Bug 6: CUDA_HOME Not Propagated to Ray Workers
**Files**: `oxrl/utils/utils.py`, `main_rl.py`
**Root cause**: DeepSpeed import failed in Ray workers because CUDA_HOME wasn't set. The toolkit is at `/ceph/workspace/erik/oxRL/cuda_env/`.
**Fix**: Updated `import_deepspeed_safely()` to auto-detect CUDA_HOME. Also forward CUDA_HOME env var to both training and rollout Ray workers.

## Bug 7: VLLM_USE_V1=0 Not Propagated to Rollout Workers
**File**: `main_rl.py`
**Root cause**: vLLM V1 engine was used instead of V0 in rollout workers, causing OOM.
**Fix**: Added `VLLM_USE_V1=0` to rollout worker env vars.

## Bug 8: grpo_test.yaml Data File Path Incorrect
**File**: `tests/grpo_test.yaml`
**Root cause**: Referenced `gsm8k_qwen2.5-0.5b-instruct_wsp_train_onboard.parquet` which doesn't exist.
**Fix**: Changed to `gsm8k_qwen2.5-0.5b-instruct_wsp_train.parquet`.

## Bug 9: VLLM_USE_V1=0 Obsolete in vLLM 0.15.1
**Files**: `oxrl/rollouts/vllm_engine.py` (line 147), `oxrl/setup/engine_factory.py` (line 115)
**Root cause**: vLLM 0.15.1 removed V0 engine entirely. The env var `VLLM_USE_V1` is no longer recognized. V1 is now the only engine, and it spawns a subprocess by default (`VLLM_ENABLE_V1_MULTIPROCESSING=True`). This subprocess does not respect Ray's per-actor GPU assignment, causing it to land on the same GPU as the training engine and trigger OOM.
**Fix**: Replaced `VLLM_USE_V1=0` with `VLLM_ENABLE_V1_MULTIPROCESSING=0` in both files. This disables the subprocess spawning, keeping vLLM in the same process as the Ray actor (which has the correct CUDA_VISIBLE_DEVICES).
**Verification**: All 4 RL algorithms (SGRPO, GSPO, CISPO, PPO) trained successfully end-to-end with this fix.
