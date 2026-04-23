# LeRobot + SmolVLA 在 LIBERO 机器人仿真基准上的安装与评测指南

> **本文记录了 LeRobot 框架结合 SmolVLA 模型在 LIBERO 机器人学习基准上的完整安装踩坑过程**，包括依赖冲突解决、模型下载、训练与评测全流程。

---

## 1. 环境准备

### 1.1 创建 Conda 环境

```bash
conda create -n lerobot_env python=3.10
conda activate lerobot_env
```

### 1.2 安装 PyTorch

建议使用 CUDA 11.8 或 12.1 对应的 PyTorch 版本：

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
```

---

## 2. 安装 LeRobot

```bash
cd ~/lerobot
pip install -e .
```

> ⚠️ **踩坑点**：如果提示 `lerobot-train: No such file`，需要改用以下方式运行：
> ```bash
> python -m lerobot.scripts.lerobot_train <参数>
> ```

---

## 3. 安装 LIBERO 仿真环境

### 3.1 安装 LIBERO 本体

```bash
git clone https://github.com/facebookresearch/LIBERO.git ~/LIBERO
export PYTHONPATH=/home/mcx/LIBERO:$PYTHONPATH  # 永久生效可写入 ~/.bashrc
```

### 3.2 安装 hffmpeg 和 egl_probe（渲染依赖）

```bash
pip install hfenv
# 如果 robosuite 编译失败，使用预编译 wheel
pip install hf-egl-probe
```

> ⚠️ **踩坑点**：`robosuite` 在某些环境下编译 `egl_probe` 会失败，推荐使用预编译的 `hf-egl-probe`。

### 3.3 安装依赖版本兼容包

```bash
pip install robosuite==1.4.0
pip install bddl
pip install numpy==2.2.6  # 注意：numpy>=2.0 会导致某些库兼容性问题
```

---

## 4. 配置 HuggingFace 权限

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx  # 替换为你的 HF token
```

> ⚠️ **踩坑点**：未设置 `HF_TOKEN` 会导致 401 Unauthorized 错误，无法下载模型和数据。

---

## 5. 评测 SmolVLA 模型（LIBERO）

### 5.1 评测脚本

官方提供了评测脚本 `scripts/eval_LIBERO.py`：

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
export PYTHONPATH=/home/mcx/LIBERO:$PYTHONPATH
export MUJOCO_GL=egl

cd ~/lerobot && python scripts/eval_LIBERO.py \
  --policy_path=lerobot/smolvla_base \
  --task_suite_name=libero_spatial \
  --num_steps_wait=10 \
  --num_trials_per_task=50 \
  --video_out_path=data/libero/videos \
  --device=cuda
```

运行效果：
- 评测视频保存在 `data/libero/videos/`
- 视频命名格式：`task_{task_id}_ep_{episode_idx}_{success|failure}.mp4`
- 基础模型成功率约为 0%（预期结果，未微调）

### 5.2 评测脚本关键逻辑

```python
# 加载预训练模型
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

# 构造输入观测
observation = {
    "observation.images.image": torch.from_numpy(agentview_image / 255.0),
    "observation.images.wrist_image": torch.from_numpy(wrist_img / 255.0),
    "observation.state": torch.from_numpy(state),
    "task": task_description,
}

# 推理获取动作
with torch.inference_mode():
    action_tensor = policy.select_action(observation)
```

---

## 6. 训练 SmolVLA 模型

### 6.1 完整训练命令

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
export PYTHONPATH=/home/mcx/LIBERO:$PYTHONPATH
export MUJOCO_GL=egl

cd ~/lerobot && proxychains4 -q python -m lerobot.scripts.lerobot_train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=mcx/smolvla_libero \
  --dataset.repo_id=lerobot/libero_sim_pickplace \
  --batch_size=2 \
  --steps=200 \
  --eval_freq=100 \
  --eval.n_episodes=2 \
  --eval.batch_size=2 \
  --output_dir=outputs/eval_libero
```

### 6.2 参数说明

| 参数 | 说明 |
|------|------|
| `--policy.path` | HuggingFace 上的预训练模型标识符 |
| `--policy.repo_id` | 用于推送模型到 Hub 的仓库地址（必填） |
| `--dataset.repo_id` | 数据集标识符 |
| `--batch_size` | 训练批次大小 |
| `--steps` | 训练步数 |
| `--eval_freq` | 每隔多少步进行一次评测 |
| `--eval.n_episodes` | 每次评测运行的 episode 数量 |
| `--output_dir` | 输出目录 |

> ⚠️ **踩坑点**：`policy.repo_id` 参数必填，否则报错：
> ```
> ValueError: 'policy.repo_id' argument missing. Please specify it to push the model to the hub.
> ```
> 如果不需要推送模型到 Hub，可设置 `--policy.push_to_hub=false`。

### 6.3 训练数据流

1. **Dataset** 加载 383 个 episodes（16GB，包含图像 + 状态 + 动作）
2. **EpisodeAwareSampler** 从 episodes 中采样 frames
3. **DataLoader** 组装成 batches
4. **Preprocessor** 预处理（归一化、图像变换）
5. **Policy.forward()** 计算损失并更新梯度

### 6.4 模型架构简述

SmolVLA 采用 **Flow Matching** 范式：

```python
# 前向损失计算（modeling_smolvla.py）
x_t = time_expanded * noise + (1 - time_expanded) * actions  # 插值
u_t = noise - actions  # 目标速度
v_t = self.action_out_proj(suffix_out)  # 预测速度
loss = F.mse_loss(u_t, v_t)  # MSE 损失
```

---

## 7. 常见问题汇总

| 问题 | 解决方案 |
|------|----------|
| `lerobot-train: No such file` | 使用 `python -m lerobot.scripts.lerobot_train` |
| HF 401 Unauthorized | 设置 `export HF_TOKEN=hf_xxx` |
| 渲染失败/EGL error | 安装 `hf-egl-probe`，设置 `MUJOCO_GL=egl` |
| `robosuite` 编译失败 | 使用 `pip install robosuite==1.4.0` |
| numpy 版本冲突 | `pip install numpy==2.2.6` |
| camera 名称不匹配 | 使用 `--rename_map` 参数重映射 |
| `policy.repo_id` 缺失 | 提供 `--policy.repo_id=用户名/仓库名` |

---

## 8. 下一步学习路径

1. **PEFT 微调**：尝试 LoRA/QLoRA 对 SmolVLA 进行参数高效微调
2. **迁移到其他基准**：RLBench、RoboMimic
3. **实时系统集成**：ROS2 机械臂控制 + 延迟分析

---

> **参考链接**
> - [LeRobot GitHub](https://github.com/huggingface/lerobot)
> - [LIBERO GitHub](https://github.com/facebookresearch/LIBERO)
> - [SmolVLA 论文](https://arxiv.org/abs/XXXX)
