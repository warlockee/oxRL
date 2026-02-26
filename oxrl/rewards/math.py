import torch
from typing import Any, Dict, List, Optional, Tuple
from oxrl.rewards.base import extract_answer, extract_math_answer, _normalize_math

def gsm8k_reward_func(prompt_ids: List[int], response_ids: List[int], finish_reason: Any, metadata: Optional[Dict] = None):
    '''GSM8K math reward: 1.0 if extracted answer matches ground truth.'''
    is_per_token = False
    r = torch.zeros((len(response_ids),), dtype=torch.float32)
    if len(response_ids) == 0 or not metadata:
        return r, is_per_token
    ground_truth = metadata.get("answer", "")
    predicted = extract_answer(metadata.get("response_text", ""))
    if not ground_truth or predicted is None:
        return r, is_per_token
    try:
        if abs(float(predicted) - float(ground_truth)) < 1e-5:
            r[-1] = 1.0
    except ValueError:
        pass
    return r, is_per_token

def math_reward_func(prompt_ids: List[int], response_ids: List[int], finish_reason: Any, metadata: Optional[Dict] = None):
    '''MATH dataset reward: 1.0 if exact string match after normalization.'''
    is_per_token = False
    r = torch.zeros((len(response_ids),), dtype=torch.float32)
    if len(response_ids) == 0 or not metadata:
        return r, is_per_token
    ground_truth = metadata.get("answer", "")
    predicted = extract_math_answer(metadata.get("response_text", ""))
    if not ground_truth or predicted is None:
        return r, is_per_token
    if _normalize_math(predicted) == _normalize_math(ground_truth):
        r[-1] = 1.0
    return r, is_per_token

def soft_math_reward_func(prompt_ids: List[int], response_ids: List[int], finish_reason: Any, metadata: Optional[Dict] = None):
    '''
    Soft math reward: 
    - 1.0 for exact match
    - 0.5 for partial numeric closeness (within 10%)
    - 0.2 for having a number at all if incorrect.
    '''
    is_per_token = False
    r = torch.zeros((len(response_ids),), dtype=torch.float32)
    if len(response_ids) == 0 or not metadata:
        return r, is_per_token
    ground_truth = metadata.get("answer", "")
    predicted = extract_answer(metadata.get("response_text", ""))
    if not ground_truth or predicted is None:
        return r, is_per_token
    
    try:
        gt_val = float(ground_truth)
        pred_val = float(predicted)
        if abs(pred_val - gt_val) < 1e-5:
            r[-1] = 1.0
        elif abs(pred_val - gt_val) / (abs(gt_val) + 1e-8) < 0.1:
            r[-1] = 0.5
        else:
            r[-1] = 0.2
    except ValueError:
        if _normalize_math(predicted) == _normalize_math(ground_truth):
            r[-1] = 1.0
    return r, is_per_token
