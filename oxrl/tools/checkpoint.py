"""
Checkpoint I/O tools: save weights, configs, and fix LoRA files on disk.

Pure functions — no model references, no distributed state.
Caller is responsible for rank gating and barriers.
"""
import os
import json
import torch


def save_state_dict_to_safetensors(output_dir: str, state_dict: dict) -> None:
    """Save a state dict as model.safetensors, breaking shared-memory tensors.

    safetensors raises RuntimeError on tensors that share the same data pointer
    (e.g., tied embeddings). We clone duplicates before saving.
    """
    from safetensors.torch import save_file

    os.makedirs(output_dir, exist_ok=True)

    seen_ptrs = {}
    for k, v in state_dict.items():
        ptr = v.data_ptr()
        if ptr in seen_ptrs:
            state_dict[k] = v.clone()
        else:
            seen_ptrs[ptr] = k

    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))


def save_config_json(output_dir: str, config_dict: dict) -> None:
    """Save a model config dict as config.json."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)


def fix_lora_checkpoint_files(output_dir: str, lora_alpha: float, lora_r: int) -> None:
    """Load checkpoint files from disk, strip PEFT prefixes, merge LoRA, re-save.

    Processes both .bin and .safetensors files in output_dir.
    """
    import glob
    from oxrl.tools.lora_merge import strip_lora_and_merge

    checkpoint_files = (
        glob.glob(os.path.join(output_dir, "*.bin"))
        + glob.glob(os.path.join(output_dir, "*.safetensors"))
    )

    if not checkpoint_files:
        return

    print(f"  Fixing LoRA in {len(checkpoint_files)} checkpoint files...")
    for ckpt_path in checkpoint_files:
        is_safetensors = ckpt_path.endswith(".safetensors")
        if is_safetensors:
            from safetensors.torch import load_file, save_file
            sd = load_file(ckpt_path)
        else:
            sd = torch.load(ckpt_path, map_location="cpu")

        sd = strip_lora_and_merge(sd, lora_alpha, lora_r)

        if is_safetensors:
            from safetensors.torch import save_file
            save_file(sd, ckpt_path)
        else:
            torch.save(sd, ckpt_path)


def get_base_model_config(policy_engine):
    """Extract the base HF model config, unwrapping PEFT/DeepSpeed wrappers."""
    model_to_save = policy_engine.module
    if hasattr(model_to_save, "get_base_model"):
        model_to_save = model_to_save.get_base_model()
    if hasattr(model_to_save, "config"):
        return model_to_save.config
    if hasattr(policy_engine.module, "module"):
        if hasattr(policy_engine.module.module, "config"):
            return policy_engine.module.module.config
    return None
