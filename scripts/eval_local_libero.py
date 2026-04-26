"""
评测本地 SmolVLA checkpoint on LIBERO benchmark
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import collections
import csv
import dataclasses
import logging
import math
import pathlib
import time

import draccus
import imageio
import numpy as np
import torch

# Patch torch.load for PyTorch 2.6+ compatibility with LIBERO
_original_torch_load = torch.load
def safe_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = safe_load

from transformers import AutoTokenizer
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs import parser

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    policy_path: str = "/home/mcx/lerobot/outputs/smolvla_libero_finetune/checkpoints/020000/pretrained_model"
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    video_out_path: str = "data/libero/videos"
    output_csv_path: str = "data/libero/eval_results.csv"
    device: str = "cuda"
    seed: int = 7


@draccus.wrap()
def eval_libero(args: Args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load Policy from local checkpoint
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.to(args.device)
    policy.eval()

    # Load preprocessor and postprocessor for proper normalization
    # The preprocessor will normalize state using dataset MEAN_STD stats
    # Stats are loaded from the checkpoint's saved processor state
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.policy_path,
    )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # 创建 CSV 文件并写入表头
    csv_path = pathlib.Path(args.output_csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp", "task_id", "task_description", "episode_index",
        "success", "total_steps", "max_steps", "task_success_rate",
        "cumulative_success", "cumulative_episodes", "cumulative_success_rate",
        "video_path"
    ])

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        max_steps = 520

    # Evaluation Loop
    total_episodes, total_successes = 0, 0
    for task_id in tqdm(range(num_tasks_in_suite), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm(
            range(min(args.num_trials_per_task, len(initial_states))),
            desc=f"Task {task_id}",
            leave=False,
        ):
            logging.info(f"\nTask: {task_description}")

            env.reset()
            policy.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

            t = 0
            frames = []
            done = False

            while t < max_steps:
                try:
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    frames.append(agentview_image)

                    # 构建 LIBERO 原始 state (8维)
                    state = np.concatenate((
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],  # 2 values (gripper has 2 fingers)
                    ))

                    # 图像从 [0, 255] 归一化到 [0, 1]，模型内部会转换为 [-1, 1]
                    img_camera1 = agentview_image / 255.0
                    img_camera2 = wrist_img / 255.0

                    # 构建 observation，使用与训练时相同的键名
                    # 重要：这里使用 observation.images.image 和 observation.images.image2
                    # preprocessor 会通过 rename_map 将它们映射到 camera1 和 camera2
                    # 转换为 torch tensor 并添加 batch 维度 (与 build_inference_frame 类似)
                    observation = {
                        "observation.state": torch.from_numpy(state).float().unsqueeze(0),
                        "observation.images.image": torch.from_numpy(img_camera1).float().permute(2, 0, 1).unsqueeze(0),
                        "observation.images.image2": torch.from_numpy(img_camera2).float().permute(2, 0, 1).unsqueeze(0),
                        "task": task_description,
                    }

                    # 使用 preprocessor 规范化所有数据（包括 state 的 MEAN_STD 归一化）
                    # 这确保推理时 state 的分布与训练时一致
                    processed_obs = preprocessor(observation)

                    with torch.inference_mode():
                        action_tensor = policy.select_action(processed_obs)

                    # Unnormalize action using postprocessor
                    unnorm_action_tensor = postprocessor(action_tensor)

                    action = unnorm_action_tensor.cpu().numpy()[0]

                    # 检查action是否有效
                    if np.any(np.isnan(action)) or np.any(np.isinf(action)):
                        print(f"[ERROR] Action contains NaN or Inf!")
                        break

                    obs, _, done, _ = env.step(action)
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    import traceback
                    traceback.print_exc()
                    break

            task_episodes += 1
            total_episodes += 1

            # Save video
            suffix = "success" if done else "failure"
            video_path = pathlib.Path(args.video_out_path) / f"task_{task_id}_ep_{episode_idx}_{suffix}.mp4"
            writer = imageio.get_writer(video_path, fps=30)
            for image in frames:
                writer.append_data(image)
            writer.close()

            # 写入 CSV
            csv_writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                task_id,
                task_description,
                episode_idx,
                int(done),
                t,
                max_steps,
                task_successes / task_episodes if task_episodes > 0 else 0,
                total_successes,
                total_episodes,
                total_successes / total_episodes if total_episodes > 0 else 0,
                str(video_path),
            ])
            csv_file.flush()

            logging.info(f"Success: {done}")
            if total_episodes > 0:
                logging.info(f"Cumulative: {total_successes}/{total_episodes} ({total_successes/total_episodes*100:.1f}%)")

        if task_episodes > 0:
            logging.info(f"Task {task_id} success rate: {task_successes/task_episodes:.2f}")

    logging.info("--- Evaluation finished ---")
    logging.info(f"Total success rate: {total_successes/total_episodes:.2f}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Total successes: {total_successes}")

    # 关闭 CSV 文件
    csv_file.close()


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero()
