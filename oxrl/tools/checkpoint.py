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


def cleanup_old_checkpoints(
    checkpoint_dir: str, experiment_id: str, keep_last_n: int,
    exclude_tags: list[str] | None = None,
) -> None:
    """Delete old checkpoint directories, keeping only the last N.

    Args:
        checkpoint_dir: Base checkpoint directory.
        experiment_id: Experiment ID (subdirectory name).
        keep_last_n: Number of most recent checkpoints to keep.
        exclude_tags: Directory names to never delete (e.g., ["best"]).
    """
    import re
    import shutil

    experiment_dir = os.path.join(checkpoint_dir, experiment_id)
    if not os.path.isdir(experiment_dir):
        return

    exclude = set(exclude_tags or [])

    # Find all iter* checkpoint directories with their epoch number
    iter_dirs = []
    for name in os.listdir(experiment_dir):
        if name in exclude:
            continue
        full_path = os.path.join(experiment_dir, name)
        if os.path.isdir(full_path) and name.startswith("iter"):
            match = re.match(r"iter(\d+)", name)
            if match:
                iter_dirs.append((int(match.group(1)), full_path))

    # Sort by epoch number ascending, remove the oldest
    iter_dirs.sort(key=lambda x: x[0])
    to_remove = iter_dirs[:-keep_last_n] if keep_last_n > 0 else iter_dirs

    for _, path in to_remove:
        shutil.rmtree(path, ignore_errors=True)


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
