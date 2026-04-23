# LeRobot 训练与评测流程 & SmolVLA 架构分析

> 本文档基于 LeRobot v0.5.2 代码库分析，介绍训练/评测流程、SmolVLA 的输入输出结构，以及如何在项目中启动训练。

---

## 1. 项目结构概览

```
lerobot/
├── src/lerobot/
│   ├── scripts/
│   │   ├── lerobot_train.py       # 训练入口
│   │   └── lerobot_eval.py        # 评测入口
│   ├── policies/
│   │   └── smolvla/
│   │       ├── modeling_smolvla.py  # SmolVLA 模型 + forward/loss
│   │       ├── configuration_smolvla.py
│   │       └── processor_smolvla.py
│   ├── configs/
│   │   ├── train.py               # TrainPipelineConfig
│   │   └── eval.py                # EvalPipelineConfig
│   ├── datasets/
│   │   └── factory.py             # make_dataset()
│   └── envs/
│       └── factory.py             # make_env()
├── outputs/                        # 训练输出目录
└── README/
    └── VLA_Training_Eval_Guide.md  # 本文档
```

---

## 2. 配置系统

LeRobot 使用 **draccus** 库进行 CLI 参数解析。所有 `--batch_size`、`--steps` 等参数都是 `TrainPipelineConfig` 数据类的字段。

### 关键训练配置 (`src/lerobot/configs/train.py`)

```python
@dataclass
class TrainPipelineConfig(HubMixin):
    dataset: DatasetConfig              # 数据集配置
    policy: PreTrainedConfig | None     # 策略配置
    output_dir: Path | None             # 输出目录
    batch_size: int = 8                # 每进程 batch size
    steps: int = 100_000                # 总优化步数
    eval_freq: int = 20_000             # 每 N 步评测一次
    log_freq: int = 200                 # 每 N 步打印一次日志
    save_freq: int = 20_000             # 每 N 步保存一次 checkpoint
    num_workers: int = 4                # DataLoader workers
    optimizer: OptimizerConfig | None
    scheduler: LRSchedulerConfig | None
    eval: EvalConfig                    # 评测配置
    wandb: WandBConfig                  # wandb 日志配置
    peft: PeftConfig | None             # PEFT (LoRA) 配置
```

### 关键评测配置 (`src/lerobot/configs/eval.py`)

```python
@dataclass
class EvalPipelineConfig:
    policy: PreTrainedConfig | None
    env: EnvConfig
    eval: EvalConfig = field(default_factory=EvalConfig)
    # eval.batch_size: 并行环境数
    # eval.n_episodes: 每个任务的 episode 数
```

---

## 3. 训练流程 (`lerobot_train.py`)

### 3.1 整体流程

```
train(cfg)
  │
  ├── 1. 创建 Accelerator (分布式训练 + 混合精度)
  │
  ├── 2. make_dataset(cfg) → DataLoader
  │
  ├── 3. make_policy(cfg, ds_meta) → SmolVLAPolicy
  │
  ├── 4. PEFT 包装 (可选, LoRA)
  │
  ├── 5. make_optimizer_and_scheduler(cfg, policy)
  │
  └── 6. 训练循环 for step in range(cfg.steps):
          │
          ├── batch = next(dataloader)     # 取数据
          ├── batch = preprocessor(batch) # 预处理
          │
          ├── update_policy()             # 核心！
          │     │
          │     ├── policy.forward(batch) → (loss, output_dict)
          │     │       │
          │     │       └── policy 内部计算 loss (Flow Matching MSE)
          │     │
          │     ├── accelerator.backward(loss)
          │     ├── grad_norm = accelerator.clip_grad_norm_()
          │     └── optimizer.step()
          │
          ├── if is_log_step:  log metrics
          ├── if is_eval_step: eval_policy_all()
          └── if is_save_step: save_checkpoint()
```

### 3.2 `update_policy()` — 前向/反向/更新

```python
# lerobot_train.py lines 59-150
def update_policy(policy, batch, optimizer, grad_clip_norm, accelerator, ...):
    policy.train()
    with accelerator.autocast():  # 混合精度
        if rabc_batch_weights is not None:
            # RA-BC: 根据样本质量加权
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")
            loss = (per_sample_loss * rabc_batch_weights).sum() / (rabc_batch_weights.sum() + 1e-6)
        else:
            loss, output_dict = policy.forward(batch)  # 默认返回标量 loss

    accelerator.backward(loss)  # loss.backward()
    if grad_clip_norm > 0:
        accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    optimizer.step()
    optimizer.zero_grad()
```

**关键点**: `policy.forward(batch)` 在策略内部完成全部 loss 计算，返回 `(loss, output_dict)`。训练脚本只负责 `backward()` 和优化器更新。

---

## 4. 评测流程 (`lerobot_eval.py`)

### 4.1 整体流程

```
eval_main(cfg)
  │
  ├── 1. make_env(cfg.env, n_envs=cfg.eval.batch_size) → VectorEnv
  │
  ├── 2. make_policy(cfg.policy) → SmolVLAPolicy
  │
  ├── 3. eval_policy_all()
  │     │
  │     └── for each task:
  │         └── rollout() → 多个 episode 的 rollout
  │
  └── 4. 输出成功率 / 奖励等指标
```

### 4.2 `rollout()` — 单次轨迹收集

```python
# lerobot_eval.py lines 96-248
def rollout(env, policy, ...):
    policy.reset()
    observation, info = env.reset(seed=seed)

    while not done and step < max_steps:
        observation = preprocess_observation(observation)
        action = policy.select_action(observation)  # 推理模式，无 loss
        action = postprocessor(action)
        observation, reward, terminated, truncated, info = env.step(action)
        # 累计奖励、成功标志
```

**关键点**: 评测时调用 `policy.select_action()`，这是纯推理模式，不涉及任何 loss 计算。

---

## 5. SmolVLA 详解

### 5.1 模型架构

SmolVLA = **SmolVLM2-500M** (VLM backbone) + **Action Expert** (动作预测头)

```
输入:
  ├── 图像 (多相机): observation.images.camera1/2/3
  ├── 机器人状态: observation.state
  └── 语言指令: "pick the red block"

VLM (SmolVLM2-500M):
  └── 提取视觉 + 语言特征 → prefix_embs

Action Expert:
  └── 接收 (prefix_embs + time embedding + noise)
      → Transformer blocks
      → action_out_proj
      → 预测 velocity

输出:
  └── actions (chunk_size=50, 动作维度=7)
```

### 5.2 训练输入输出

| 字段 | Shape | 说明 |
|------|-------|------|
| **输入** | | |
| `observation.images.camera{1,2,3}` | `(B, 3, 256, 256)` | 三路相机图像 |
| `observation.state` | `(B, 14)` | 机械臂关节状态 (aloha) |
| `language_tokens` | `(B, 48)` | 语言指令 token |
| `language_attention_mask` | `(B, 48)` | 语言 attention mask |
| `action` | `(B, 50, 8)` | 动作序列 (50步×8维度) |
| **输出** | | |
| `loss` | `Scalar` | Flow Matching MSE loss |
| `action` (inference) | `(B, 50, 8)` | 预测的动作 chunk |

### 5.3 Forward Loss 计算 (Flow Matching)

**文件**: `src/lerobot/policies/smolvla/modeling_smolvla.py:763-799`

```python
def VLAFlowMatching.forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
    """完整的训练前向传播 + loss 计算"""

    # 1. 采样噪声和时间步
    noise = self.sample_noise(actions.shape, actions.device)  # (B, 50, 8)
    time = self.sample_time(actions.shape[0], actions.device)  # (B,)

    # 2. Flow 插值: x_t = t * noise + (1-t) * actions
    time_expanded = time[:, None, None]  # (B, 1, 1)
    x_t = time_expanded * noise + (1 - time_expanded) * actions

    # 3. 目标速度 (flow matching target)
    u_t = noise - actions  # 从 actions 流向 noise 的速度

    # 4. 提取视觉+语言特征
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )

    # 5. 拼接 time embedding + noise 作为 suffix
    suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

    # 6. 通过 VLM + Action Expert
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    (_, suffix_out), _ = self.vlm_with_expert.forward(
        attention_mask=att_2d_masks,
        position_ids=position_ids,
        inputs_embeds=[prefix_embs, suffix_embs],
    )
    suffix_out = suffix_out[:, -self.config.chunk_size :]

    # 7. 预测 velocity
    v_t = self.action_out_proj(suffix_out)  # (B, 50, 8)

    # 8. MSE Loss: ||u_t - v_t||^2
    losses = F.mse_loss(u_t, v_t, reduction="none")  # (B, 50, 8)
    return losses
```

### 5.4 推理 (Sample Actions)

**文件**: `src/lerobot/policies/smolvla/modeling_smolvla.py:801-860`

推理使用**迭代式 Flow Matching 采样**（共 `num_steps=10` 步）：

```python
def sample_actions(self, images, ...):
    noise = self.sample_noise((bsize, chunk_size, action_dim), device)
    x_t = noise

    # 迭代去噪
    for step in range(num_steps):
        time = 1.0 + step * (-1.0 / num_steps)  # 从 1.0 到 0.0
        time_tensor = torch.tensor(time, dtype=torch.float32, device=device)

        # 预测 velocity
        v_t = self._forward_impl(images, x_t, time_tensor, ...)

        # 采样下一步
        x_t = x_t + (-1.0 / num_steps) * v_t

    return x_t  # 最终去噪的动作
```

---

## 6. 启动训练与评测

### 6.1 完整训练命令

```bash
# 设置 HuggingFace Token (访问 gated 数据集)
export HF_TOKEN=hf_your_token_here

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh && conda activate lerobot_env

# 启动训练 (需要代理访问 HF)
cd ~/lerobot && proxychains4 -q python -m lerobot.scripts.lerobot_train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=your_username/smolvla_finetune \
  --dataset.repo_id=lerobot/pusht \
  --batch_size=8 \
  --steps=20000 \
  --eval_freq=500 \
  --eval.n_episodes=10 \
  --eval.batch_size=10 \
  --output_dir=outputs/smolvla_finetune
```

### 6.2 完整评测命令

```bash
export HF_TOKEN=hf_your_token_here

source ~/anaconda3/etc/profile.d/conda.sh && conda activate lerobot_env

cd ~/lerobot && proxychains4 -q python -m lerobot.scripts.lerobot_eval \
  --policy.path=outputs/smolvla_finetune/checkpoints/latest/pretrained_model \
  --env.type=pusht \
  --eval.n_episodes=10 \
  --eval.batch_size=10 \
  --policy.device=cuda
```

### 6.3 关键参数说明

| CLI 参数 | 配置字段 | 说明 | 推荐值 |
|----------|----------|------|--------|
| `--batch_size` | `batch_size` | 训练 batch size | 2-16 (根据显存) |
| `--steps` | `steps` | 总训练步数 | 5000-50000 |
| `--eval_freq` | `eval_freq` | 评测频率 | 500-2000 |
| `--eval.n_episodes` | `eval.n_episodes` | 每个任务 episode 数 | 10-50 |
| `--eval.batch_size` | `eval.batch_size` | 并行评测环境数 | >= n_episodes |
| `--num_workers` | `num_workers` | DataLoader workers | 4-8 |

---

## 7. SmolVLA 配置解析

**文件**: `src/lerobot/policies/smolvla/configuration_smolvla.py`

```python
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    # 输入输出结构
    n_obs_steps: int = 1           # 观察步数
    chunk_size: int = 50           # 动作 chunk 长度
    n_action_steps: int = 50       # 预测的动作步数

    # 图像预处理
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # 训练设置
    freeze_vision_encoder: bool = True   # 冻结视觉编码器
    train_expert_only: bool = True        # 只训练 Action Expert
    train_state_proj: bool = True        # 训练状态投影层

    optimizer_lr: float = 1e-4           # 学习率
    optimizer_grad_clip_norm: float = 10 # 梯度裁剪

    # VLM backbone
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    # Flow Matching
    num_steps: int = 10                  # 推理去噪步数
```

---

## 8. 数据集格式

LeRobot 数据集必须是 LeRobot 格式（HuggingFace Hub 上的 `lerobot/` 命名空间）。结构：

```
lerobot/dataset_name/
├── meta/
│   ├── info.json              # 数据集元信息
│   ├── episodes.json           # episode 列表
│   └── features.json           # 数据特征定义
├── data/
│   ├── episode_000000/
│   │   ├── observation.images.camera1/
│   │   │   └── 000000.jpg, 000001.jpg, ...
│   │   ├── observation.state/
│   │   │   └── 000000.parquet
│   │   └── action/
│   │       └── 000000.parquet
│   └── episode_000001/
│       └── ...
└── videos/
    └── episode_000000.camera1.mp4, ...
```

---

## 9. 常见问题

### Q: 显存不足 (OOM)
```bash
# 减小 batch_size
--batch_size=4  # 或 2

# 或增加梯度累积
--gradient_accumulation_steps=4
```

### Q: 网络超时
```bash
# 使用代理
export HF_ENDPOINT=https://hf-mirror.com
# 或
proxychains4 -q python ...
```

### Q: 如何恢复中断的训练
```bash
--checkpoint_path=outputs/smolvla_finetune/checkpoints/005000/pretrained_model
```

### Q: 如何启用 LoRA 微调
```bash
--peft.lora_rank=16 \
--peft.lora_alpha=32 \
--peft.lora_dropout=0.1
```

---

## 10. 参考链接

- [SmolVLA 官方文档](https://huggingface.co/docs/lerobot/smolvla)
- [SmolVLA 模型卡](https://hf.co/lerobot/smolvla_base)
- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [LeRobot 数据集格式](./lerobot_dataset_format.md)





cd ~/lerobot && python -m lerobot.scripts.lerobot_train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=mcx/smolvla_libero \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --batch_size=8 \
  --steps=20000 \
  --eval_freq=500 \
  --eval.n_episodes=10 \
  --eval.batch_size=10 \
  --output_dir=outputs/smolvla_libero_finetune \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'


cd /home/mcx/lerobot

python scripts/eval_LIBERO.py \
  --policy_path /home/mcx/lerobot/outputs/smolvla_libero_finetune/checkpoints/002000/pretrained_model \
  --task_suite_name libero_spatial \
  --num_trials_per_task 10 \
  --device cuda \
  --video_out_path outputs/eval_libero_smolvla/videos
