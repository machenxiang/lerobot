"""Test evaluation loop with model actions"""
import os
import numpy as np
import torch

os.environ["HF_TOKEN"] = ""  # Set your HF_TOKEN here
os.environ["PYTHONPATH"] = "/home/mcx/LIBERO:" + os.environ.get("PYTHONPATH", "")
os.environ["MUJOCO_GL"] = "egl"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("HF_DATASETS_OFFLINE", None)

from transformers import AutoTokenizer
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from pathlib import Path
import imageio

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

def quat2axisangle(quat):
    import math
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

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

# Setup LIBERO
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()
task = task_suite.get_task(0)
task_description = task.language
task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

env_args = {
    "bddl_file_name": str(task_bddl_file),
    "camera_heights": 256,
    "camera_widths": 256,
}
env = OffScreenRenderEnv(**env_args)
env.seed(7)

print(f"\nTask: {task_description}")

# Reset and wait
obs = env.reset()
policy.reset()
for _ in range(10):
    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

print(f"\n=== Initial eef_pos after wait ===")
print(f"robot0_eef_pos: {obs['robot0_eef_pos']}")

# Run 5 steps with model
frames = []
max_steps = 5
t = 0

while t < max_steps:
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    frames.append(agentview_image)

    state = np.concatenate((
        obs["robot0_eef_pos"],
        quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ))

    observation = {
        "observation.images.camera1": torch.from_numpy(agentview_image / 255.0)
            .permute(2, 0, 1).to(torch.float32).to('cuda').unsqueeze(0),
        "observation.images.camera2": torch.from_numpy(wrist_img / 255.0)
            .permute(2, 0, 1).to(torch.float32).to('cuda').unsqueeze(0),
        "observation.state": torch.from_numpy(state).to(torch.float32).to('cuda').unsqueeze(0),
    }

    lang_tokens = tokenizer(task_description, max_length=48, truncation=True, padding="max_length", return_tensors="pt")
    observation["observation.language.tokens"] = lang_tokens["input_ids"].to('cuda')
    observation["observation.language.attention_mask"] = lang_tokens["attention_mask"].to(torch.bool).to('cuda')

    with torch.inference_mode():
        action_tensor = policy.select_action(observation)

    unnorm_action_tensor = postprocessor(action_tensor)
    action = unnorm_action_tensor.cpu().numpy()[0]

    print(f"\n=== Step {t} ===")
    print(f"eef_pos before: {obs['robot0_eef_pos']}")
    print(f"action: {action}")

    obs, _, done, _ = env.step(action)

    print(f"eef_pos after:  {obs['robot0_eef_pos']}")
    print(f"eef_pos diff:  {obs['robot0_eef_pos'] - action[:3]}")

    t += 1
    if done:
        break

# Save video
writer = imageio.get_writer('/tmp/test_eval.mp4', fps=30)
for image in frames:
    writer.append_data(image)
writer.close()

print(f"\nSaved {len(frames)} frames to /tmp/test_eval.mp4")
env.close()
