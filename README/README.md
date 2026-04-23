# LeRobot 学习资料索引

本目录存放 LeRobot VLA 相关的学习笔记和指南。

## 文档列表

### [VLA_Training_Eval_Guide.md](./VLA_Training_Eval_Guide.md) ⭐ 推荐先读

**内容概要**：
- LeRobot 训练流程详解 (`lerobot_train.py`)
- LeRobot 评测流程详解 (`lerobot_eval.py`)
- SmolVLA 模型架构（VLM + Action Expert）
- **SmolVLA Forward Loss 计算**（Flow Matching MSE）
- SmolVLA 输入输出详解
- 训练/评测命令模板
- 常见问题解答

**适用场景**：学习 LeRobot 训练/评测全流程、理解 SmolVLA 如何计算 loss

---

## 快速开始

### 训练 SmolVLA

```bash
export HF_TOKEN=hf_your_token
source ~/anaconda3/etc/profile.d/conda.sh && conda activate lerobot_env
cd ~/lerobot && proxychains4 -q python -m lerobot.scripts.lerobot_train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=your_username/smolvla_finetune \
  --dataset.repo_id=lerobot/pusht \
  --batch_size=8 \
  --steps=5000 \
  --eval_freq=500 \
  --eval.n_episodes=8 \
  --eval.batch_size=8 \
  --output_dir=outputs/smolvla_finetune
```

### 评测

```bash
cd ~/lerobot && proxychains4 -q python -m lerobot.scripts.lerobot_eval \
  --policy.path=outputs/smolvla_finetune/checkpoints/latest/pretrained_model \
  --env.type=pusht \
  --eval.n_episodes=10 \
  --eval.batch_size=10
```

---

## 核心代码位置

| 功能 | 文件路径 |
|------|----------|
| 训练入口 | `src/lerobot/scripts/lerobot_train.py` |
| 评测入口 | `src/lerobot/scripts/lerobot_eval.py` |
| SmolVLA 模型+Loss | `src/lerobot/policies/smolvla/modeling_smolvla.py` |
| SmolVLA 配置 | `src/lerobot/policies/smolvla/configuration_smolvla.py` |
| 训练配置 | `src/lerobot/configs/train.py` |
| 评测配置 | `src/lerobot/configs/eval.py` |
| 数据集加载 | `src/lerobot/datasets/factory.py` |
| 环境创建 | `src/lerobot/envs/factory.py` |

---

## SmolVLA 训练核心：Forward Loss

SmolVLA 使用 **Flow Matching** 进行动作生成，loss 计算在 `modeling_smolvla.py:763-799`：

```python
# Flow 插值
x_t = t * noise + (1-t) * actions
# 目标 velocity
u_t = noise - actions
# 预测 velocity
v_t = model(...)
# MSE Loss
loss = F.mse_loss(u_t, v_t, reduction="none")
```

详见 [VLA_Training_Eval_Guide.md](./VLA_Training_Eval_Guide.md#53-forward-loss-计算-flow-matching)



训练：
```
cd ~/lerobot && proxychains4 -q python -m lerobot.scripts.lerobot_train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/svla_so100_stacking \
  --batch_size=8 \
  --steps=200 \
  --eval_freq=100 \
  --eval.n_episodes=8 \
  --eval.batch_size=8 \
  --output_dir=outputs/smolvla_stacking \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --dataset.video_backend=pyav \
  --policy.push_to_hub=false

```