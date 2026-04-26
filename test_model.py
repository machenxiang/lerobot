"""Quick test to verify model inference"""
import os
import sys
import numpy as np
import torch

os.environ["HF_TOKEN"] = ""  # Set your HF_TOKEN here
os.environ["PYTHONPATH"] = "/home/mcx/LIBERO:" + os.environ.get("PYTHONPATH", "")
os.environ["MUJOCO_GL"] = "egl"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("HF_DATASETS_OFFLINE", None)

from transformers import AutoTokenizer
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

print("Loading model...")
policy = SmolVLAPolicy.from_pretrained('/home/mcx/lerobot/outputs/smolvla_libero_finetune/checkpoints/020000/pretrained_model')
policy.to('cuda')
policy.eval()
print("Model loaded!")

print("Loading postprocessor...")
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path='/home/mcx/lerobot/outputs/smolvla_libero_finetune/checkpoints/020000/pretrained_model',
)
print("Postprocessor loaded!")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# Create test observation
agentview_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
wrist_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
state = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0])  # near robot

observation = {
    'observation.images.camera1': torch.from_numpy(agentview_image / 255.0).permute(2, 0, 1).float().unsqueeze(0).cuda(),
    'observation.images.camera2': torch.from_numpy(wrist_img / 255.0).permute(2, 0, 1).float().unsqueeze(0).cuda(),
    'observation.state': torch.from_numpy(state).float().unsqueeze(0).cuda(),
}

lang_tokens = tokenizer('pick up the black bowl', return_tensors='pt', max_length=48, padding='max_length', truncation=True)
observation['observation.language.tokens'] = lang_tokens['input_ids'].cuda()
observation['observation.language.attention_mask'] = lang_tokens['attention_mask'].bool().cuda()

# Run inference 3 times to see if actions differ
print("\n=== Test 1 ===")
with torch.no_grad():
    action = policy.select_action(observation)
    unnorm = postprocessor(action)
print(f"Raw action: {action.cpu().numpy()[0]}")
print(f"Unnorm action: {unnorm.cpu().numpy()[0]}")

print("\n=== Test 2 (same obs) ===")
with torch.no_grad():
    action = policy.select_action(observation)
    unnorm = postprocessor(action)
print(f"Raw action: {action.cpu().numpy()[0]}")
print(f"Unnorm action: {unnorm.cpu().numpy()[0]}")

print("\n=== Test 3 (same obs) ===")
with torch.no_grad():
    action = policy.select_action(observation)
    unnorm = postprocessor(action)
print(f"Raw action: {action.cpu().numpy()[0]}")
print(f"Unnorm action: {unnorm.cpu().numpy()[0]}")

# Now test with different state
print("\n=== Test 4 (different state) ===")
state2 = np.array([0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0])  # different position
observation['observation.state'] = torch.from_numpy(state2).float().unsqueeze(0).cuda()
with torch.no_grad():
    action = policy.select_action(observation)
    unnorm = postprocessor(action)
print(f"Raw action: {action.cpu().numpy()[0]}")
print(f"Unnorm action: {unnorm.cpu().numpy()[0]}")

print("\nDone!")
