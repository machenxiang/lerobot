"""评测本地 SmolVLA checkpoint"""

import numpy as np
import torch
from pathlib import Path
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# 本地 checkpoint 路径
CHECKPOINT_PATH = "/home/mcx/lerobot/outputs/smolvla_libero_finetune/checkpoints/002000/pretrained_model"

policy = SmolVLAPolicy.from_pretrained(CHECKPOINT_PATH)
policy.to("cuda")
policy.eval()

# 测试推理
test_obs = {
    "observation.images.image": torch.randn(1, 3, 224, 224).cuda(),
    "observation.images.wrist_image": torch.randn(1, 3, 224, 224).cuda(),
    "observation.state": torch.randn(1, 7).cuda(),
    "task": "pick the red block",
}

with torch.no_grad():
    action = policy.select_action(test_obs)
    print(f"Action shape: {action.shape}")
    print(f"Action: {action}")
