import os
import sys
import subprocess
import torch
import platform
import argparse
from pathlib import Path

def check_gpu():
    print("Checking GPU availability...")
    if not torch.cuda.is_available():
        print("  [ERROR] CUDA is not available to PyTorch.")
        return False
    
    device_count = torch.cuda.device_count()
    print(f"  [OK] Found {device_count} GPU(s).")
    for i in range(device_count):
        print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
    return True

def check_cuda_toolkit():
    print("Checking CUDA Toolkit (nvcc)...")
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        print(f"  [INFO] CUDA_HOME is set to: {cuda_home}")
    else:
        print("  [WARNING] CUDA_HOME is not set.")

    try:
        nvcc_version = subprocess.check_output(["nvcc", "--version"], text=True)
        print("  [OK] nvcc found:")
        for line in nvcc_version.strip().split("\n")[-1:]:
            print(f"    {line}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  [ERROR] nvcc not found in PATH. DeepSpeed may fail to compile custom ops.")
        print("          Try installing CUDA Toolkit or setting DS_SKIP_CUDA_CHECK=1.")
        return False

def check_deepspeed():
    print("Checking DeepSpeed...")
    try:
        from oxrl.utils.utils import import_deepspeed_safely
        deepspeed = import_deepspeed_safely()
        print(f"  [OK] DeepSpeed version {deepspeed.__version__} installed.")
        return True
    except ImportError as e:
        print(f"  [ERROR] DeepSpeed not found or failed to initialize: {e}")
        return False
    except Exception as e:
        print(f"  [ERROR] Unexpected error checking DeepSpeed: {e}")
        return False

def check_ray():
    print("Checking Ray...")
    try:
        import ray
        print(f"  [OK] Ray version {ray.__version__} installed.")
        return True
    except ImportError:
        print("  [ERROR] Ray not found. Install with: pip install ray")
        return False

def fix_environment(results):
    print("\n" + "="*40)
    print(" Attempting to fix environment issues...")
    print("="*40)
    
    gpu_ok, cuda_ok, ds_ok, ray_ok = results
    
    if not cuda_ok:
        print("[FIX] nvcc missing. Setting DS_SKIP_CUDA_CHECK=1 in environment...")
        os.environ["DS_SKIP_CUDA_CHECK"] = "1"
        # Try to persist it to .bashrc if on Linux
        bashrc = os.path.expanduser("~/.bashrc")
        if os.path.exists(bashrc):
            try:
                with open(bashrc, "a") as f:
                    f.write("\n# oxRL self-healing: skip deepspeed cuda check\n")
                    f.write("export DS_SKIP_CUDA_CHECK=1\n")
                print("  [OK] Added export DS_SKIP_CUDA_CHECK=1 to ~/.bashrc")
            except Exception as e:
                print(f"  [ERROR] Could not write to ~/.bashrc: {e}")

    if not ds_ok:
        print("[FIX] Installing DeepSpeed...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "deepspeed"])
            print("  [OK] DeepSpeed installed.")
        except Exception as e:
            print(f"  [ERROR] Failed to install DeepSpeed: {e}")

    if not ray_ok:
        print("[FIX] Installing Ray...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ray[default]"])
            print("  [OK] Ray installed.")
        except Exception as e:
            print(f"  [ERROR] Failed to install Ray: {e}")

    print("\n[INFO] Fixes attempted. Please restart your terminal or re-run 'oxrl doctor'.")
    print("="*40 + "\n")

def doctor(fix=False):
    print("="*40)
    print(" oxRL Doctor - Environment Diagnostics")
    print("="*40)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print("-" * 40)
    
    results = [
        check_gpu(),
        check_cuda_toolkit(),
        check_deepspeed(),
        check_ray()
    ]
    
    print("-" * 40)
    if all(results):
        print("[SUCCESS] Your environment looks healthy for oxRL!")
    else:
        if any(results):
            print("[CAUTION] Some issues were found. oxRL might still work with workarounds.")
        else:
            print("[FATAL] Environment check failed completely.")
        
        if fix:
            fix_environment(results)
        else:
            print("[INFO] Run 'oxrl doctor --fix' to attempt automated fixes.")
    print("="*40)

def main():
    parser = argparse.ArgumentParser(description="oxRL CLI tools")
    subparsers = parser.add_subparsers(dest="command")
    
    doctor_parser = subparsers.add_parser("doctor", help="Check environment for common issues")
    doctor_parser.add_argument("--fix", action="store_true", help="Attempt to automatically fix issues")

    report_parser = subparsers.add_parser("report", help="Generate a GitHub issue report for a failure")
    report_parser.add_argument("--model", type=str, help="Model ID that failed")
    report_parser.add_argument("--log", type=str, help="Path to the failure log")
    report_parser.add_argument("--submit", action="store_true", help="Automatically submit to GitHub (requires GITHUB_TOKEN)")
    
    args = parser.parse_args()
    
    if args.command == "doctor":
        doctor(fix=args.fix)
    elif args.command == "report":
        from oxrl.swarm.bug_reporter import summarize_failure, submit_github_issue
        # Look for the last failure log
        latest_log = None
        registry_dir = Path("registry")
        if registry_dir.exists():
            logs = list(registry_dir.glob("**/train.log"))
            if logs:
                latest_log = str(sorted(logs, key=os.path.getmtime)[-1])
        
        report = summarize_failure(
            model_id=args.model or "unknown",
            error_msg="Manual report requested via CLI",
            log_path=args.log or latest_log
        )
        print(report)

        if args.submit:
            print("\nSubmitting to GitHub...")
            result = submit_github_issue(report, args.model or "unknown")
            if result["success"]:
                print(f"[OK] Issue submitted: {result.get('url', result.get('info'))}")
            else:
                print(f"[ERROR] Submission failed: {result['error']}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
