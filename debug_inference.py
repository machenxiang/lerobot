"""
Debug script for evaluating SmolVLA on LIBERO
Usage: python debug_inference.py
Then use debugger to set breakpoints and step through.
"""

import os
import sys

# Set environment variables BEFORE importing anything else
os.environ["HF_TOKEN"] = "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"
os.environ["PYTHONPATH"] = "/home/mcx/LIBERO:" + os.environ.get("PYTHONPATH", "")
os.environ["MUJOCO_GL"] = "egl"

# Use hf-mirror.com for China
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("HF_DATASETS_OFFLINE", None)

# Print debug info
print("Debug mode: Starting evaluation...")
print(f"Python: {sys.executable}")
print(f"Working directory: {os.getcwd()}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")

if __name__ == "__main__":
    # 直接运行 eval_local_libero.py
    import subprocess
    result = subprocess.run([
        sys.executable,
        "/home/mcx/lerobot/scripts/eval_local_libero.py",
        "--policy_path=/home/mcx/lerobot/outputs/smolvla_libero_finetune/checkpoints/020000/pretrained_model",
        "--task_suite_name=libero_spatial",
        "--num_trials_per_task=10",
        "--video_out_path=data/libero/videos",
        "--output_csv_path=data/libero/eval_results.csv",
        "--device=cuda",
        "--seed=7",
    ], cwd="/home/mcx/lerobot")
    sys.exit(result.returncode)
