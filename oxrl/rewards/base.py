import re
import torch
from typing import Any, Dict, List, Optional, Tuple

def default_reward_func(prompt_ids: List[int], response_ids: List[int], finish_reason: Any, metadata: Optional[Dict] = None):
    '''
      Default reward: 1.0 if the model stopped naturally (EOS), 0.0 otherwise.
    '''
    is_per_token = False
    r = torch.zeros((len(response_ids),), dtype=torch.float32)
    if len(response_ids) == 0:
        return r, is_per_token
    r[-1] = 1.0 if str(finish_reason) == "stop" else 0.0
    return r, is_per_token

def extract_answer(text: str) -> Optional[str]:
    '''Extract a numerical answer from a GSM8K-style response.'''
    match = re.search(r'####\s*([\-]?[0-9][0-9\,\.]*)', text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r'[\-]?[0-9][0-9\,\.]*', text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None

def extract_math_answer(text: str) -> Optional[str]:
    '''Extract an answer from a MATH-style response (\boxed{...}).'''
    idx = text.rfind("\boxed{")
    if idx != -1:
        start = idx + len("\boxed{")
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            return text[start:i - 1].strip()
    match = re.search(r"####\s*(.+)", text)
    if match:
        return match.group(1).strip()
    return None

def extract_mcqa_answer(text: str) -> Optional[str]:
    '''Extract a multiple-choice answer letter (A-Z).'''
    match = re.search(r"[Tt]he\s+answer\s+is\s*\(?([A-Z])\)?", text)
    if match: return match.group(1)
    match = re.search(r"[Aa]nswer\s*:\s*([A-Z])", text)
    if match: return match.group(1)
    matches = re.findall(r"(?<![a-zA-Z])([A-Z])(?![a-zA-Z])", text)
    if matches: return matches[-1]
    return None

def _normalize_math(s: str) -> str:
    '''Normalize a math answer string for comparison.'''
    return s.strip().lower().rstrip(".")
