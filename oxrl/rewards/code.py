import re
import subprocess
import tempfile
import torch
from typing import Any, Dict, List, Optional, Tuple

def code_reward_func(prompt_ids: List[int], response_ids: List[int], finish_reason: Any, metadata: Optional[Dict] = None):
    '''
    Code execution reward: 1.0 if generated code passes test cases, 0.0 otherwise.
    Use for: MBPP, HumanEval, or any code-gen task with test cases.
    Pro:  ground-truth verification via execution — no false positives.
    Con:  slow (subprocess per sample), security risk (runs untrusted code),
          binary signal only (no partial credit for almost-correct code).
    Requires metadata["test_cases"] (Python assert statements) and metadata["response_text"].
    '''
    is_per_token = False
    r = torch.zeros((len(response_ids),), dtype=torch.float32)
    if len(response_ids) == 0 or not metadata:
        return r, is_per_token
    response_text = metadata.get("response_text", "")
    test_cases = metadata.get("test_cases", "")
    if not response_text or not test_cases:
        return r, is_per_token
    code_match = re.search(r"```python\s*\n(.*?)```", response_text, re.DOTALL)
    code = code_match.group(1) if code_match else response_text
    full_script = code + "\n\n" + test_cases
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as f:
            f.write(full_script)
            f.flush()
            result = subprocess.run(["python", f.name], capture_output=True, timeout=5)
        if result.returncode == 0:
            r[-1] = 1.0
    except Exception:
        pass
    return r, is_per_token
