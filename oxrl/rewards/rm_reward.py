import torch
import os
from typing import Any, Dict, List, Optional, Tuple

_rm_model = None
_rm_value_head = None

def load_reward_model(rm_path: str, device: str = "cuda"):
    """Load a trained reward model from checkpoint."""
    global _rm_model, _rm_value_head
    if _rm_model is not None:
        return _rm_model, _rm_value_head

    from transformers import AutoModelForCausalLM, AutoConfig
    from oxrl.algs.rm import RewardValueHead

    config = AutoConfig.from_pretrained(rm_path)
    model = AutoModelForCausalLM.from_pretrained(rm_path, torch_dtype=torch.bfloat16)
    model.eval()
    model.to(device)

    value_head = RewardValueHead(config.hidden_size)
    vh_path = os.path.join(rm_path, "value_head.pt")
    if os.path.exists(vh_path):
        value_head.load_state_dict(torch.load(vh_path, map_location=device))
    value_head.eval()
    value_head.to(device)

    _rm_model = model
    _rm_value_head = value_head
    return model, value_head

def rm_reward_func(prompt_ids, response_ids, finish_reason, metadata=None):
    """
    Learned reward model: continuous score from a trained value head.
    Use for: RLHF when no verifiable ground truth exists (open-ended generation, chat).
    Pro:  captures nuanced quality beyond binary right/wrong.
    Con:  reward hacking risk (model exploits RM weaknesses), requires a trained RM checkpoint.
    Requires: call load_reward_model(path) before use. Set reward_model_path in config.
    """
    global _rm_model, _rm_value_head
    is_per_token = False
    r = torch.zeros((len(response_ids),), dtype=torch.float32)

    if len(response_ids) == 0:
        return r, is_per_token

    if _rm_model is None:
        raise RuntimeError("Reward model not loaded. Call load_reward_model() first.")

    device = next(_rm_model.parameters()).device
    input_ids = torch.tensor(prompt_ids + response_ids, dtype=torch.long, device=device).unsqueeze(0)
    attn_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = _rm_model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        hidden = output.hidden_states[-1]
        rewards = _rm_value_head(hidden)
        r[-1] = rewards[0, -1].cpu().item()

    return r, is_per_token
