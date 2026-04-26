"""
Debug script for SmolVLA training on LIBERO
Usage: F5 (VSCode) or python debug_train.py
Then use debugger to set breakpoints and step through.
"""

import os
import sys

# ============================================================
# 训练设置 - 修改这里的 RESUME 选择训练模式
# ============================================================
RESUME = True  # True=从本地模型继续训练, False=从头开始训练

# 继续训练时使用的 checkpoint 路径
RESUME_CHECKPOINT_PATH = (
    "outputs/smolvla_libero_finetune/checkpoints/020000/pretrained_model"
)
# ============================================================

# Set environment variables BEFORE importing anything else
os.environ["HF_TOKEN"] = ""  # Set your HF_TOKEN here
os.environ["PYTHONPATH"] = "/home/mcx/LIBERO:" + os.environ.get("PYTHONPATH", "")
os.environ["MUJOCO_GL"] = "egl"

# 使用 hf-mirror.com 镜像（国内加速）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 禁用离线模式，允许联网下载
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("HF_DATASETS_OFFLINE", None)

# Print debug info
print("Debug mode: Starting training config...")
print(f"Python: {sys.executable}")
print(f"Working directory: {os.getcwd()}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
print(
    f"RESUME={RESUME}, checkpoint={RESUME_CHECKPOINT_PATH if RESUME else 'None (从头训练)'}"
)

# Import and run - uses @parser.wrap() to parse args
if __name__ == "__main__":
    from lerobot.scripts.lerobot_train import main

    if RESUME:
        # 从本地 checkpoint 继续训练
        sys.argv = [
            "debug_train.py",
            "--resume=true",
            f"--config_path={RESUME_CHECKPOINT_PATH}/train_config.json",
            "--steps=200000",
            "--tensorboard.enable=true",
            "--log_freq=1000",
            "--save_freq=20000",
            "--policy.push_to_hub=false",
        ]
    else:
        # 从头开始训练
        sys.argv = [
            "debug_train.py",
            "--policy.path=lerobot/smolvla_base",
            "--policy.repo_id=mcx/smolvla_libero",
            "--policy.use_amp=true",
            "--dataset.repo_id=HuggingFaceVLA/libero",
            "--batch_size=8",
            "--steps=200000",
            "--eval_freq=500",
            "--eval.n_episodes=10",
            "--eval.batch_size=10",
            "--output_dir=outputs/smolvla_libero_finetune",
            "--tensorboard.enable=true",
            "--log_freq=100",
            "--save_freq=2000",
            "--policy.push_to_hub=false",
            """--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}""",
        ]

    print(f"Arguments: {sys.argv[1:]}")
    main()
