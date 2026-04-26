"""Test LIBERO environment with simple actions"""
import os
import numpy as np

os.environ["HF_TOKEN"] = ""  # Set your HF_TOKEN here
os.environ["PYTHONPATH"] = "/home/mcx/LIBERO:" + os.environ.get("PYTHONPATH", "")
os.environ["MUJOCO_GL"] = "egl"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("HF_DATASETS_OFFLINE", None)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from pathlib import Path

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

# Initialize LIBERO task suite
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

print(f"Task: {task_description}")
print("\n=== Initial state ===")
obs = env.reset()
print(f"robot0_eef_pos: {obs['robot0_eef_pos']}")

# Step with dummy action
print("\n=== After 1 step with zero action ===")
obs, _, done, _ = env.step([0.0] * 6 + [-1.0])
print(f"robot0_eef_pos: {obs['robot0_eef_pos']}")

print("\n=== After another step ===")
obs, _, done, _ = env.step([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
print(f"robot0_eef_pos: {obs['robot0_eef_pos']}")

print("\n=== After another step ===")
obs, _, done, _ = env.step([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
print(f"robot0_eef_pos: {obs['robot0_eef_pos']}")

print("\n=== After another step with bigger action ===")
obs, _, done, _ = env.step([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
print(f"robot0_eef_pos: {obs['robot0_eef_pos']}")

env.close()
print("\nDone!")
