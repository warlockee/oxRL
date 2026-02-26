import os
import sys
import subprocess
import torch
import platform
import argparse

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

def doctor():
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
    elif any(results):
        print("[CAUTION] Some issues were found. oxRL might still work with workarounds.")
    else:
        print("[FATAL] Environment check failed completely.")
    print("="*40)

def main():
    parser = argparse.ArgumentParser(description="oxRL CLI tools")
    subparsers = parser.add_subparsers(dest="command")
    
    subparsers.add_parser("doctor", help="Check environment for common issues")
    
    args = parser.parse_args()
    
    if args.command == "doctor":
        doctor()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
