import os
import sys
import platform
import json
import torch
from pathlib import Path

def summarize_failure(model_id, error_msg, log_path=None):
    """
    Aggregates environment info and error logs into a structured Markdown report.
    """
    # 1. Gather Environment Info (Mini-Doctor)
    env_info = {
        "OS": f"{platform.system()} {platform.release()}",
        "Python": sys.version.split()[0],
        "PyTorch": torch.__version__,
        "GPUS": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    # 2. Get last few lines of log if available
    log_tail = ""
    if log_path and os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()
                log_tail = "".join(lines[-50:])
        except Exception as e:
            log_tail = f"Could not read log: {e}"

    # 3. Build Markdown
    report = f"""# 🤖 oxRL Framework Failure Report

## 📋 Context
- **Model:** `{model_id}`
- **Timestamp:** `{os.popen('date -u').read().strip()}`
- **Error Type:** `FRAMEWORK_CRASH`

## 💻 Environment
- **OS:** {env_info['OS']}
- **Python:** {env_info['Python']}
- **PyTorch:** {env_info['PyTorch']}
- **Available GPUs:** {env_info['GPUS']}

## 🚨 Error Message
```python
{error_msg}
```

## 📜 Log Tail (Last 50 lines)
```text
{log_tail}
```

---
**Note to AI Agent:** This report was automatically generated. If this looks like a bug in oxRL (not an OOM or Data error), please open an issue at: 
https://github.com/warlockee/oxRL/issues/new?title=[BUG]+Framework+crash+on+{model_id}&body=PASTE_REPORT_HERE
"""
    return report

def is_framework_error(error_msg):
    """
    Heuristics to determine if an error is likely a framework bug.
    """
    framework_indicators = [
        "RayTaskError",
        "ActorDiedError",
        "AttributeError",
        "ImportError",
        "TypeError",
        "ValueError", # Note: can be user error, but often framework
        "ZeroDivisionError",
    ]
    user_indicators = [
        "out of memory",
        "CUDA OOM",
        "File Not Found",
        "Permission denied",
        "MISSING_PREPROCESSOR",
    ]
    
    msg_lower = str(error_msg).lower()
    
    # User errors take precedence
    for indicator in user_indicators:
        if indicator.lower() in msg_lower:
            return False
            
    for indicator in framework_indicators:
        if indicator.lower() in msg_lower:
            return True
            
    return False
