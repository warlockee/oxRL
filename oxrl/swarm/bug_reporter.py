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

def submit_github_issue(report_md, model_id):
    """
    Submits the generated report as a GitHub issue.
    Requires GITHUB_TOKEN environment variable.
    """
    import requests
    
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return {"success": False, "error": "GITHUB_TOKEN not found in environment."}
    
    repo = "warlockee/oxRL"
    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    
    title = f"[AUTO-BUG] Framework crash on {model_id}"
    
    # 1. Deduplication check: See if an open issue with this title already exists
    try:
        search_url = f"https://api.github.com/repos/{repo}/issues?state=open"
        existing = requests.get(search_url, headers=headers).json()
        if any(issue['title'] == title for issue in existing):
            return {"success": True, "info": "Issue already exists and is open. Skipping."}
    except:
        pass # Continue to submission if check fails

    # 2. Submit
    data = {
        "title": title,
        "body": report_md,
        "labels": ["bug", "automated-report"]
    }
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 201:
        issue_url = response.json().get("html_url")
        return {"success": True, "url": issue_url}
    else:
        return {"success": False, "error": f"GitHub API failed ({response.status_code}): {response.text}"}
