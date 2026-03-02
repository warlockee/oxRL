"""
LoRA weight merging: strip PEFT key prefixes and fold LoRA deltas into base weights.

Pure function — no class state, no model references, no side effects.
Input:  raw state dict (with PEFT prefixes + LoRA A/B matrices), LoRA hyperparams.
Output: clean state dict with merged weights, ready for HF-format saving.
"""


def strip_lora_and_merge(state_dict: dict, lora_alpha: float, lora_r: int) -> dict:
    """Strip PEFT prefixes and merge LoRA weights into base weights in-memory.

    Args:
        state_dict: Raw state dict, possibly with 'base_model.model.' prefixes
                    and '.lora_A.'/'.lora_B.' adapter weights.
        lora_alpha: LoRA scaling numerator.
        lora_r:     LoRA rank (scaling denominator).

    Returns:
        Clean state dict with LoRA deltas merged: base_weight += (B @ A) * (alpha / r).
    """
    scaling = lora_alpha / lora_r

    new_state_dict = {}
    lora_weights = {}

    # Pass 1: separate base weights, base_layer weights, and LoRA adapters
    for k, v in state_dict.items():
        clean_k = k
        if clean_k.startswith("base_model.model."):
            clean_k = clean_k[len("base_model.model."):]

        if ".lora_A." in clean_k or ".lora_B." in clean_k:
            lora_weights[clean_k] = v
        elif ".base_layer." in clean_k:
            new_k = clean_k.replace(".base_layer.", ".")
            new_state_dict[new_k] = v
        else:
            new_state_dict[clean_k] = v

    # Pass 2: merge LoRA deltas into corresponding base weights
    for k in list(new_state_dict.keys()):
        prefix = k.rsplit(".", 1)[0]
        la = f"{prefix}.lora_A.default.weight"
        lb = f"{prefix}.lora_B.default.weight"
        if la in lora_weights and lb in lora_weights:
            la_w = lora_weights[la]
            lb_w = lora_weights[lb]
            base_w = new_state_dict[k]

            try:
                delta = (lb_w @ la_w) * scaling
                if delta.shape == base_w.shape:
                    print(f"  Merging LoRA for {k} (shape {base_w.shape})")
                    new_state_dict[k] = base_w + delta.to(base_w.dtype)
                else:
                    print(f"  WARNING: Shape mismatch for {k}: delta {delta.shape} vs base {base_w.shape}")
            except Exception as e:
                print(f"  WARNING: Failed to merge LoRA for {k}: {e}")

    return new_state_dict
