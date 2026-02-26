import torch
from typing import Any, Dict, List, Optional, Tuple

def format_reward_func(prompt_ids: List[int], response_ids: List[int], finish_reason: Any, metadata: Optional[Dict] = None):
    '''Instruction-following format reward.'''
    is_per_token = False
    r = torch.zeros((len(response_ids),), dtype=torch.float32)
    if len(response_ids) == 0:
        return r, is_per_token
    response_text = metadata.get("response_text", "") if metadata else ""
    score = 0.0
    if len(response_text) > 50: score += 0.25
    if "\n\n" in response_text: score += 0.25
    stripped = response_text.lstrip()
    if stripped and not stripped.startswith("I"): score += 0.25
    if str(finish_reason) == "stop": score += 0.25
    r[-1] = score
    return r, is_per_token
