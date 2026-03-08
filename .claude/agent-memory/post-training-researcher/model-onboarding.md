# Model Onboarding Notes

## Verified Models
| Model | Params | Status | Loss | Time | Notes |
|-------|--------|--------|------|------|-------|
| moonshotai/Kimi-VL-A3B-Instruct | 16.41B | PASSED | 3.3327 | 205s | MoE VL model, ZeRO-3+CPU offload on 1 GPU |
| moonshotai/Kimi-VL-A3B-Thinking | 16.41B | PASSED | 3.0861 | 270s | Same arch as above, thinking variant |
| moonshotai/Kimi-Linear-48B-A3B-Instruct | 49.12B | PARTIAL | N/A | N/A | Loads to CPU OK, OOM on 4 GPUs, needs 8+ |
| MiniMaxAI/MiniMax-M2 | 230B | SKIP | N/A | N/A | Hardware insufficient (needs 8x A100-80GB) |

## Key Issues Encountered

### Python 3.10 UnionType Compatibility
- Kimi-Linear uses `X | Y` union type annotations in remote model code
- transformers `auto_docstring._process_parameter_type` calls `.annotation.__name__` which fails on `types.UnionType`
- Fix: `ensure_auto_docstring_union_type()` monkey-patch in `oxrl/utils/setup.py`

### Flash Attention Forced by Remote Code
- Kimi-Linear's `modeling_kimi.py` line 913 forces `flash_attention_2`
- Patched cached remote code to accept "eager" attention: `if config._attn_implementation not in ("flash_attention_2", "eager"):`
- File: `/fsx/workspace/erik/hf_cache/modules/transformers_modules/moonshotai/Kimi_hyphen_Linear_hyphen_48B_hyphen_A3B_hyphen_Instruct/.../modeling_kimi.py`

### fla-core Package Required
- Kimi-Linear uses Flash Linear Attention, requires `pip install -U fla-core`
- Installed v0.4.1

### DeepSpeed ZeRO-3 OOM for 49B Models
- 49B params / 4 GPUs = ~24.5GB per partition + overhead exceeds 40GB A100 capacity
- `deepspeed.zero.Init()` + `from_pretrained()` incompatible ("Cannot copy out of meta tensor")
- Need 8+ free A100-40GB GPUs for ZeRO-3 training of 49B models
- `device_map='cpu'` works for loading (0 GPU memory) but DeepSpeed `initialize()` moves partitions to GPU

### MiniMax Models
- No small MiniMax text models exist on HuggingFace
- Smallest is MiniMax-M2: 230B total / 10B active (MoE with 256 experts)
- Requires int8 quantization minimum (~230GB) + 8x A100-80GB GPUs
- Tokenizer and config load OK

## Hardware Constraints
- 8x A100-40GB GPUs (320GB total)
- GPUs 0-3 often occupied by other processes
- GPUs 4-7 typically free
- For models >20B params, need ZeRO-3 + CPU offloading
- For models >40B params, need 8+ GPUs with ZeRO-3

## verify_model.py Key Settings
- Large model threshold: 3B params (skip naive forward/backward, use DeepSpeed)
- Very large model threshold: 20B params (aggressive ZeRO-3 settings)
- Always loads to CPU first (`device_map='cpu'`), then DeepSpeed manages GPU placement
- Uses BF16 for all models
