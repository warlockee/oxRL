import re
import torch
from typing import List, Any, Dict, Optional


def default_reward_func(prompt_ids: List[int], response_ids: List[int], finish_reason: Any, metadata: Optional[Dict] = None):
    '''
      Default reward: 1.0 if the model stopped naturally (EOS), 0.0 otherwise.
      metadata is accepted but unused — keeps the interface consistent.
    '''
    is_per_token = False
    r = torch.zeros((len(response_ids),), dtype=torch.float32)

    if len(response_ids) == 0:
        return r, is_per_token

    r[-1] = 1.0 if str(finish_reason) == "stop" else 0.0
    return r, is_per_token


def extract_answer(text: str) -> Optional[str]:
    '''
      Extract a numerical answer from a GSM8K-style response.
      Looks for "#### <number>" first, falls back to last number in text.
    '''
    # Primary: look for the #### marker the system prompt asks for
    match = re.search(r'####\s*([\-]?[0-9][0-9\,\.]*)', text)
    if match:
        return match.group(1).replace(",", "")

    # Fallback: last number in the response (may false-positive on chain-of-thought)
    numbers = re.findall(r'[\-]?[0-9][0-9\,\.]*', text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def gsm8k_reward_func(prompt_ids: List[int], response_ids: List[int], finish_reason: Any, metadata: Optional[Dict] = None):
    '''
      GSM8K math reward: 1.0 if extracted answer matches ground truth, 0.0 otherwise.
      Expects metadata["answer"] (ground truth) and metadata["response_text"].
    '''
    is_per_token = False
    r = torch.zeros((len(response_ids),), dtype=torch.float32)

    if len(response_ids) == 0 or not metadata:
        return r, is_per_token

    ground_truth = metadata.get("answer", "")
    predicted    = extract_answer(metadata.get("response_text", ""))

    if not ground_truth or predicted is None:
        return r, is_per_token

    try:
        if abs(float(predicted) - float(ground_truth)) < 1e-5:
            r[-1] = 1.0
    except ValueError:
        pass

    return r, is_per_token
