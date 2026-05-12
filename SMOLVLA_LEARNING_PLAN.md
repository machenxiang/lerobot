# SmolVLA 进阶学习与优化计划 (V2.0)

## 背景
- 当前项目：在 LIBERO 数据集上微调 SmolVLA
- 当前状态：92k 步训练，Loss 下降 18% 后趋于平稳（0.374 → 0.306 → 0.306）
- Task 5/9 成功率低：30% / 40%
- 硬件环境：4060 Ti 16GB
- 数据集：HuggingFaceVLA/libero（273,465 样本，40 个任务）
- 预训练模型：HuggingFaceVLA/smolvla_libero（已适配 LIBERO 维度）

---

## 核心目标

1. **突破 Loss 平台期**：通过解冻策略和 LoRA 寻找更优局部解
2. **攻克长尾任务**：将 Task 5/9 的成功率从 30-40% 提升至 60% 以上
3. **架构深度理解**：掌握从视觉特征到 Flow Matching 动作生成的全链路逻辑

---

## 学习计划（4 周）

### Week 1: 理论深化与动作对齐

**目标**：理解"视觉-动作"的翻译机制

**重点**：Flow Matching 原理解析、视觉端到动作端的对齐

#### 1.1 VLM 架构拆解
- 视觉端：SigLIP 的特征图如何映射到动作空间
- 语言端：Gemma 模型在动作预测中的作用
- 动作端：深入 Flow Matching 机制
  - 理解它如何通过回归"向量场"来预测动作
  - 对比 Diffusion：Flow Matching 不需要不断去噪，直接预测速度场

#### 1.2 微调范式研究
- Prompt Tuning（改变输入提示）
- Adapter/LoRA（改变模型权重）
- 对机器人轨迹平滑度的影响对比

#### 1.3 参数分布与显存分析
- 整理 SmolVLA 的参数分布表（哪些部分占显存最大）
- 分析 4060 Ti 16GB 环境下的训练瓶颈

**任务**：
- [ ] 整理 SmolVLA 的参数分布表（VLM / Expert / StateProj 各占多少）
- [ ] 对比基准：记录当前 92k 步的推理视频，分析 Task 5/9 失败的具体瞬间
- [ ] 分析失败原因：是"找不准物体"还是"抓取位置偏移"

**验证**：能画出 SmolVLA 的完整数据流图（像素 → 特征 → 动作）

---

### Week 2: 工程分析与环境对齐

**目标**：排查状态维度与相机视角问题

**重点**：8D vs 6D 状态空间、视觉注意力诊断

#### 2.1 状态空间排查
- 深入 `modeling_smolvla.py`
- 确认 LIBERO 的 8 维状态（6D 姿态 + 1D 夹爪 + 1D 填充）是否与预训练模型期望完美对齐

#### 2.2 视觉注意力诊断
- 在推理代码中插入特征图可视化
- 观察 Task 5 (Ramekin 上) 时，SigLIP 的注意力是否被 Ramekin 的边缘干扰
- 分析模型是否分不清"碗"和"架子"

#### 2.3 针对性代码修改
- 在 `lerobot_train.py` 中尝试修改 `train_expert_only` 的开关逻辑
- 验证不同的冻结策略对训练效果的影响

**任务**：
- [ ] 绘制 SmolVLA 数据流图：从像素输入到最后 7 维动作输出
- [ ] 在 `prepare_state` 函数中加入数据检查点，确保归一化没有导致精度损失
- [ ] 实现简单的视觉注意力可视化

**验证**：能回答"SmolVLA 的状态为什么是 8 维"这个问题

---

### Week 3: 瓶颈诊断与数据增强

**目标**：解决 Loss 平稳与高处拾取难题

**重点**：LoRA 引入、针对 Task 5/9 的数据增强

#### 3.1 Loss 平台期分析
- **假设**：Expert Action 层参数量过小，已达到拟合极限
- **对策**：引入 LoRA 微调语言模型 Backbone（Gemma）
- 增强模型对复杂场景（如 Task 5/9 的高处空间关系）的理解能力

#### 3.2 Task 5/9 专项强化
- **数据重采样**：在训练 Dataloader 中，对 Task 5 和 Task 9 进行 2x Oversampling
- **数据增强**：针对高处任务的光影变化，增加随机亮度与对比度增强

**任务**：
- [ ] 使用 TensorBoard 对比不同 Task 的分支 Loss
- [ ] 验证：如果只训练 Task 5/9，模型能否快速收敛？（排除数据质量问题）
- [ ] 实现 Task 5/9 的 Oversampling

**验证**：能提出 3 个具体的改进方案，并说明原理

---

### Week 4: 策略实验与显存优化

**目标**：在 4060 Ti 上进行高效微调

**重点**：LoRA vs Partial Unfreeze 对比实验

#### 4.1 显存友好型微调策略

| 实验 | 修改内容 | 预期显存 | 说明 |
|------|----------|----------|------|
| Baseline | Expert Only (当前) | ~8GB | 基准 |
| Exp A | LoRA on LLM | ~12GB | 仅在语言模型上应用 LoRA |
| Exp B | Partial Unfreeze Vision | ~15GB | 解冻视觉编码器最后 2 层 |

#### 4.2 训练参数调优
- 学习率调度：余弦退火（Cosine Annealing）
- 混合精度：使用 bf16 或 fp16
- 梯度累积：`gradient_accumulation_steps=4` 以获得更稳定的梯度

#### 4.3 综合评测
- 每隔 5k 步保存 Checkpoint
- 在 LIBERO 仿真器中进行多轮（50 episodes/task）评测

**任务**：
- [ ] 执行对冲实验：Expert Only vs LoRA vs Partial Unfreeze
- [ ] 撰写最终报告
- [ ] 总结提升 Task 5/9 成功率的最佳超参数组合

**验证**：Task 5/9 成功率提升到 60%+

---

## 针对 4060 Ti (16GB) 的实操建议

### 混合精度训练
务必使用 bf16（如果显卡支持）或 fp16

### 梯度累积
由于显存限制，Batch Size 可能被迫设得很小（如 4 或 8）
建议设置 `gradient_accumulation_steps=4` 以获得更稳定的梯度

### 内存管理
使用 `bitsandbytes` 库开启 4-bit 或 8-bit 加载 Backbone，为微调层腾出空间

### 关键配置参考
```
batch_size: 4-8
gradient_accumulation_steps: 4
mixed_precision: bf16
lr: 1e-4 (with cosine annealing)
```

---

## 实验记录模板

| 实验编号 | 核心修改 | Loss 趋势 | Task 5 成功率 | Task 9 成功率 | 显存占用 |
|---------|----------|-----------|--------------|--------------|----------|
| Baseline | Expert Only | 0.306 (平稳) | 30% | 40% | ~8GB |
| Exp 01 | LoRA on LLM | - | - | - | ~12GB |
| Exp 02 | Unfreeze Vision (Last 2) | - | - | - | ~15GB |
| Exp 03 | Task 5/9 Oversampling | - | - | - | - |

---

## 关键文件路径

### 代码
- SmolVLA 模型：`src/lerobot/policies/smolvla/`
- 训练脚本：`src/lerobot/scripts/lerobot_train.py`
- 评测脚本：`scripts/eval_local_libero.py`

### 数据
- 训练数据：`~/.cache/huggingface/datasets/HuggingFaceVLA___libero/`
- 训练输出：`outputs/smolvla_libero_finetune_0427/`
- TensorBoard：`outputs/smolvla_libero_finetune_0427/tensorboard/`

### Checkpoints
- 当前最新：`outputs/smolvla_libero_finetune_0427/checkpoints/092000/`

---

## 学习资源

- LeRobot 官方文档：https://lerobot.github.io/
- HuggingFace Transformers：https://huggingface.co/docs/transformers/
- LIBERO benchmark：https://github.com/facebookresearch/LIBERO