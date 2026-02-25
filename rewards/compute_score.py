import re
import subprocess
import tempfile
import torch
from typing import Any, Dict, List, Optional, Tuple


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


# ---------------------------------------------------------------------------
# MATH dataset reward
# ---------------------------------------------------------------------------

def extract_math_answer(text: str) -> Optional[str]:
    '''
      Extract an answer from a MATH-style response.
      Primary: look for \\boxed{...} (standard MATH format).
      Fallback: look for "#### <answer>" (GSM8K-style marker).
    '''
    # Primary: \boxed{...}  — handle nested braces by counting depth
    idx = text.rfind("\\boxed{")
    if idx != -1:
        start = idx + len("\\boxed{")
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

    # Fallback: #### <answer>
    match = re.search(r"####\s*(.+)", text)
    if match:
        return match.group(1).strip()

    return None


def _normalize_math(s: str) -> str:
    '''Normalize a math answer string for comparison.'''
    s = s.strip().lower().rstrip(".")
    return s


def math_reward_func(prompt_ids: List[int], response_ids: List[int], finish_reason: Any, metadata: Optional[Dict] = None) -> Tuple[torch.Tensor, bool]:
    '''
      MATH dataset reward: 1.0 if the extracted answer matches the ground truth
      (exact string match after normalization), 0.0 otherwise.
      Expects metadata["answer"] and metadata["response_text"].
    '''
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


# ---------------------------------------------------------------------------
# Code generation (MBPP) reward
# ---------------------------------------------------------------------------

def code_reward_func(prompt_ids: List[int], response_ids: List[int], finish_reason: Any, metadata: Optional[Dict] = None) -> Tuple[torch.Tensor, bool]:
    '''
      MBPP code-generation reward: 1.0 if the generated code passes all
      provided test cases, 0.0 otherwise.
      Expects metadata["response_text"] and metadata["test_cases"] (a string
      of Python assert statements).
    '''
    is_per_token = False
    r = torch.zeros((len(response_ids),), dtype=torch.float32)

    if len(response_ids) == 0 or not metadata:
        return r, is_per_token

    response_text = metadata.get("response_text", "")
    test_cases = metadata.get("test_cases", "")

    if not response_text or not test_cases:
        return r, is_per_token

    # Extract Python code from markdown code block, or use the whole response
    code_match = re.search(r"```python\s*\n(.*?)```", response_text, re.DOTALL)
    if code_match:
        code = code_match.group(1)
    else:
        code = response_text

    # Combine the generated code with the test cases
    full_script = code + "\n\n" + test_cases

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as f:
            f.write(full_script)
            f.flush()
            result = subprocess.run(
                ["python", f.name],
                capture_output=True,
                timeout=5,
            )
        if result.returncode == 0:
            r[-1] = 1.0
    except Exception:
        # Timeout, OSError, or any other failure — reward stays 0.0
        pass

    return r, is_per_token


# ---------------------------------------------------------------------------
# Format / instruction-following reward (UltraFeedback)
# ---------------------------------------------------------------------------

def format_reward_func(prompt_ids: List[int], response_ids: List[int], finish_reason: Any, metadata: Optional[Dict] = None) -> Tuple[torch.Tensor, bool]:
    '''
      Instruction-following format reward (UltraFeedback).
      Checks structural quality of the response and awards up to 1.0 points:
        0.25 — response length > 50 characters
        0.25 — contains paragraph breaks (double newline)
        0.25 — does not start with "I" (less self-referential)
        0.25 — finished naturally (finish_reason == "stop")
    '''
    is_per_token = False
    r = torch.zeros((len(response_ids),), dtype=torch.float32)

    if len(response_ids) == 0:
        return r, is_per_token

    response_text = metadata.get("response_text", "") if metadata else ""

    score = 0.0

    # Length check
    if len(response_text) > 50:
        score += 0.25

    # Paragraph breaks
    if "\n\n" in response_text:
        score += 0.25

    # Does not start with "I"
    stripped = response_text.lstrip()
    if stripped and not stripped.startswith("I"):
        score += 0.25

    # Completed naturally
    if str(finish_reason) == "stop":
        score += 0.25

    r[-1] = score
    return r, is_per_token


# ---------------------------------------------------------------------------
# Multiple-choice QA reward (MMLU-Pro)
# ---------------------------------------------------------------------------

def extract_mcqa_answer(text: str) -> Optional[str]:
    '''
      Extract a multiple-choice answer letter from a response.
      Tries (in order):
        1. "The answer is (X)" / "The answer is X"
        2. "Answer: X"
        3. Last standalone capital letter (A-Z) in the text
    '''
    # Pattern 1: "The answer is (X)" or "The answer is X"
    match = re.search(r"[Tt]he\s+answer\s+is\s*\(?([A-Z])\)?", text)
    if match:
        return match.group(1)

    # Pattern 2: "Answer: X"
    match = re.search(r"[Aa]nswer\s*:\s*([A-Z])", text)
    if match:
        return match.group(1)

    # Pattern 3: last standalone capital letter
    matches = re.findall(r"(?<![a-zA-Z])([A-Z])(?![a-zA-Z])", text)
    if matches:
        return matches[-1]

    return None


def mcqa_reward_func(prompt_ids: List[int], response_ids: List[int], finish_reason: Any, metadata: Optional[Dict] = None) -> Tuple[torch.Tensor, bool]:
    '''
      MMLU-Pro multiple-choice reward: 1.0 if the extracted answer letter
      matches the ground truth, 0.0 otherwise.
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


# ---------------------------------------------------------------------------
# Reasoning (DeepSeek-R1 style) reward
# ---------------------------------------------------------------------------

def reasoning_reward_func(prompt_ids: List[int], response_ids: List[int], finish_reason: Any, metadata: Optional[Dict] = None) -> Tuple[torch.Tensor, bool]:
    '''
      Reward function for reasoning models (e.g. DeepSeek-R1, Open-R1).
      Awards up to 1.0 points based on:
        0.2 — contains <thought>...</thought> tags
        0.2 — contains <answer>...</answer> tags
        0.6 — extracted answer matches ground truth
    '''
    is_per_token = False
    r = torch.zeros((len(response_ids),), dtype=torch.float32)

    if len(response_ids) == 0 or not metadata:
        return r, is_per_token

    response_text = metadata.get("response_text", "")
    ground_truth = metadata.get("answer", "")

    score = 0.0

    # 1. Format: Thought tags
    if "<thought>" in response_text and "</thought>" in response_text:
        score += 0.2
    
    # 2. Format: Answer tags
    if "<answer>" in response_text and "</answer>" in response_text:
        score += 0.2
        
        # 3. Correctness: Extract answer from within tags if possible
        match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
        if match:
            predicted = match.group(1).strip()
            if _normalize_math(predicted) == _normalize_math(ground_truth):
                score += 0.6
        else:
            # Fallback extraction if tags are present but match fails
            predicted = extract_math_answer(response_text)
            if predicted and _normalize_math(predicted) == _normalize_math(ground_truth):
                score += 0.3 # Reduced reward for missing/misformatted content in tags
    
    else:
        # No answer tags — try standard extraction but with penalty
        predicted = extract_math_answer(response_text)
        if predicted and _normalize_math(predicted) == _normalize_math(ground_truth):
            score += 0.4

    r[-1] = score
    return r, is_per_token
