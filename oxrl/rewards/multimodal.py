import torch
import re
from typing import Any, Dict, List, Optional, Tuple
from oxrl.rewards.base import extract_answer, _normalize_math

def multimodal_reward_func(prompt_ids: List[int], response_ids: List[int], finish_reason: Any, metadata: Optional[Dict] = None):
    '''
    Generic multimodal reward for Vision/Audio tasks.
    Checks for:
    - 1.0 Correctness (math or string match)
    - 0.5 If it described the modality (keywords like "image", "audio", "video")
    '''
    is_per_token = False
    r = torch.zeros((len(response_ids),), dtype=torch.float32)
    if len(response_ids) == 0 or not metadata:
        return r, is_per_token
    
    response_text = metadata.get("response_text", "").lower()
    ground_truth = metadata.get("answer", "")
    
    score = 0.0
    # Correctness check
    predicted = extract_answer(response_text)
    if predicted and ground_truth:
        try:
            if abs(float(predicted) - float(ground_truth)) < 1e-5:
                score = 1.0
        except ValueError:
            if _normalize_math(predicted) == _normalize_math(ground_truth):
                score = 1.0
    
    # Fallback for non-numeric ground truth
    if score < 1.0 and ground_truth.lower() in response_text:
        score = 1.0
                
    # Modality awareness (soft reward)
    if score < 1.0:
        keywords = ["image", "picture", "audio", "sound", "video", "clip", "see", "hear"]
        if any(k in response_text for k in keywords):
            score = max(score, 0.2)
            
    r[-1] = score
    return r, is_per_token
