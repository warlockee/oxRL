import re
import torch
from typing import Any, Dict, List, Optional, Tuple
from oxrl.rewards.base import extract_math_answer, _normalize_math

def reasoning_reward_func(prompt_ids: List[int], response_ids: List[int], finish_reason: Any, metadata: Optional[Dict] = None):
    '''Reward function for reasoning models (DeepSeek-R1 style).'''
    is_per_token = False
    r = torch.zeros((len(response_ids),), dtype=torch.float32)
    if len(response_ids) == 0 or not metadata:
        return r, is_per_token
    response_text = metadata.get("response_text", "")
    ground_truth = metadata.get("answer", "")
    score = 0.0
    
    # Check for thought process (flexible tags)
    has_thought = ("<thought>" in response_text and "</thought>" in response_text) or \
                  ("<think>" in response_text and "</think>" in response_text)
    if has_thought:
        score += 0.2
        
    # Check for answer formatting (flexible tags)
    has_answer = ("<answer>" in response_text and "</answer>" in response_text)
    if has_answer:
        score += 0.2
        match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
        if match:
            predicted = match.group(1).strip()
            if _normalize_math(predicted) == _normalize_math(ground_truth):
                score += 0.6
        else:
            predicted = extract_math_answer(response_text)
            if predicted and _normalize_math(predicted) == _normalize_math(ground_truth):
                score += 0.3
    else:
        predicted = extract_math_answer(response_text)
        if predicted and _normalize_math(predicted) == _normalize_math(ground_truth):
            score += 0.4
    r[-1] = score
    return r, is_per_token
