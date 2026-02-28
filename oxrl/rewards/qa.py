import torch
from typing import Any, Dict, List, Optional, Tuple
from oxrl.rewards.base import extract_mcqa_answer

def mcqa_reward_func(prompt_ids: List[int], response_ids: List[int], finish_reason: Any, metadata: Optional[Dict] = None):
    '''
    Multiple-choice QA reward: 1.0 if extracted letter matches ground truth.
    Use for: MMLU-Pro, ARC, or any A/B/C/D multiple-choice benchmark.
    Pro:  clean binary signal, robust letter extraction with multiple fallbacks.
    Con:  only checks final letter — doesn't reward reasoning quality.
    Expects metadata["answer"] (correct letter) and metadata["response_text"].
    '''
    is_per_token = False
    r = torch.zeros((len(response_ids),), dtype=torch.float32)
    if len(response_ids) == 0 or not metadata:
        return r, is_per_token
    ground_truth = metadata.get("answer", "")
    predicted = extract_mcqa_answer(metadata.get("response_text", ""))
    if not ground_truth or predicted is None:
        return r, is_per_token
    if predicted.strip().upper() == ground_truth.strip().upper():
        r[-1] = 1.0
    return r, is_per_token
