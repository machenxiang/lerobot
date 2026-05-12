# Week 1: 理论深化与动作对齐

## 本周目标
理解"视觉-动作"的翻译机制，掌握 Flow Matching 原理，整理 SmolVLA 参数分布

## 学习内容

### 1.1 VLM 架构拆解

#### 视觉端：SigLIP
- SigLIP 是视觉编码器，将图像转换为特征向量
- 特征图尺寸：通常 16x16 或 24x24
- 输出：patch embeddings → 动作空间的映射关系

**SigLIP 在 SmolVLA 中的实现细节：**

```python
# 位于 smolvlm_with_expert.py 的 embed_image 函数
def embed_image(self, image: torch.Tensor):
    # 1. 通过 vision_model 获取图像特征
    image_hidden_states = (
        self.get_vlm_model()
        .vision_model(
            pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype),
            patch_attention_mask=patch_attention_mask,
        )
        .last_hidden_state
    )
    # 2. 通过 connector (模态投影层) 进行处理
    image_hidden_states = self.get_vlm_model().connector(image_hidden_states)
    return image_hidden_states
```

**数据流：**
```
相机图像 (H, W, 3)
    ↓
像素值转换 (归一化到 [-1, 1] 给 SigLIP)
    ↓
vision_model 处理得到 patch embeddings
    ↓
patch 序列 (batch, num_patches, hidden_dim)
    ↓
connector 模态投影 + resampling
    ↓
图像特征 (batch, seq_len, hidden_dim)
```

**关键参数：**
- SmolVLA 使用的 VLM 模型：`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
- 图像预处理： resize 到 512x512，保持宽高比 padding
- 像素值范围：从 [0, 1] 转换到 [-1, 1]（SigLIP 要求）
- 特征维度：由 VLM 的 `hidden_size` 决定（约 768-2048）

**Patch 到动作的映射机制：**
1. 图像 patch embeddings 与语言 embeddings 拼接
2. 通过 VLM Transformer 进行多模态融合
3. Action Expert (Flow Matching) 基于融合后的特征预测动作
4. 动作输出经过 `action_out_proj` 投影到实际动作维度

#### 语言端：Gemma
- Gemma 是语言模型，负责理解任务指令
- 将自然语言指令转换为动作预测的条件

**Gemma 在 SmolVLA 中的实现细节：**

```python
# 位于 smolvlm_with_expert.py
def embed_language_tokens(self, tokens: torch.Tensor):
    # 1. 获取 text_model 的嵌入层
    embed_layer = self.get_vlm_model().text_model.get_input_embeddings()
    # 2. 将 token IDs 转换为嵌入向量
    return embed_layer(tokens)
```

**数据流：**
```
任务指令文本: "pick up the red bowl"
    ↓
Tokenizer → token IDs: [1234, 5678, 9012, ...]
    ↓
embed_language_tokens() → 嵌入向量 (batch, seq_len, hidden_dim)
    ↓
与图像特征拼接，一起送入 VLM Transformer
```

**关键参数：**
- VLM 模型：`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
- Gemma hidden_size：约 1024（取决于模型配置）
- Gemma 层数：16 层（可通过 `num_vlm_layers` 配置）
- 位置编码：RoPE（Rotary Position Embedding）

**Gemma 在 VLM 中的作用：**
1. **理解指令**：将自然语言解析为语义向量
2. **跨模态融合**：与 SigLIP 图像特征交互
3. **指令控制**：根据任务决定关注图像的哪个区域

**与其他模块的关系：**
```
图像特征 → SigLIP encoder → connector → 图像嵌入
指令 token → Gemma embed_layer → 语言嵌入
                              ↓
                    VLM Transformer (Gemma layers)
                              ↓
                    输出融合后的多模态特征
```

#### 动作端：Flow Matching
- **核心问题**：动作预测本质是什么？
- **Diffusion vs Flow Matching**：
  - Diffusion：逐步去噪，从噪声生成动作
  - Flow Matching：直接回归速度场，预测每个时间步应该往哪个方向移动
- SmolVLA 使用 10-step Euler 积分进行动作生成

### 1.2 微调范式对比

| 范式 | 方法 | 对轨迹的影响 |
|------|------|-------------|
| Prompt Tuning | 改变输入提示 | 效果有限，不改变模型权重 |
| Adapter | 在层间插入小型网络 | 保留原始能力，专注新任务 |
| LoRA | 低秩矩阵分解 | 参数高效，训练稳定 |
| Full Fine-tune | 训练所有参数 | 容易过拟合，需要大量数据 |

### 1.3 参数分布与显存分析

## 任务清单

### Task 1: 整理 SmolVLA 参数分布表

**目标**：了解各部分占用的显存和参数量

**执行步骤**：
1. 读取 `HuggingFaceVLA/smolvla_libero` 的 config.json
2. 计算 VLM、Expert、StateProj 的参数量
3. 估算不同冻结策略下的显存占用

```python
# 建议代码：统计参数量
import torch
from transformers import AutoModel

# 加载模型
model = ...  # 你的模型加载代码

# 统计各部分参数
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
```

**产出**：
- [ ] 创建 `smolvla_params_table.md`，记录：
  - VLM 参数量
  - Expert 参数量
  - StateProj 参数量
  - 各部分显存占用估算

---

### Task 2: 分析 Task 5/9 失败原因

**目标**：记录 92k 步模型的推理视频，分析失败模式

**执行步骤**：
1. 使用 `debug_inference.py` 在 92k checkpoint 上评测 Task 5 和 Task 9
2. 保存评测视频
3. 逐帧分析失败原因

**失败类型分类**：
- **类型 A：找不准物体** - 模型在图像中定位碗的位置失败
- **类型 B：抓取位置偏移** - 找到了物体但夹爪伸过去位置不对
- **类型 C：运动学问题** - 夹爪到达位置但无法完成抓取
- **类型 D：放置失败** - 抓到了但放不到正确位置

**产出**：
- [ ] 创建 `task5_9_failure_analysis.md`
- [ ] 记录每种失败类型的出现次数
- [ ] 附上典型失败帧截图

---

### Task 3: 绘制 SmolVLA 数据流图

**目标**：画出从像素输入到最后 7 维动作输出的完整流程

**数据流**：
```
相机图像 (H, W, 3)
    ↓
SigLIP 视觉编码器
    ↓
图像特征 (batch, seq_len, hidden_dim)
    ↓
Gemma 语言编码器 (处理任务指令)
    ↓
多模态特征融合
    ↓
State 投影层 (8D → hidden_dim)
    ↓
Action Expert (Flow Matching)
    ↓
动作预测 (batch, action_dim)
    ↓
后处理 (归一化 → 原始动作空间)
    ↓
7 维动作输出
```

**产出**：
- [ ] 创建 `smolvla_architecture.drawio` 或手绘版
- [ ] 标注每个步骤的输入输出维度

---

### Task 4: 理解 Flow Matching

**目标**：理解 SmolVLA 如何预测动作

**关键概念**：
1. **向量场 (Vector Field)**：在动作空间中，Flow Matching 预测的是"速度方向"
2. **Euler 积分**：10 步 Euler 积分，从当前状态逐步推进到目标动作
3. **Flow Matching Loss**：简单来说，就是让模型学习"给定当前状态和目标，动作应该往哪个方向走"

**阅读材料**：
- 如果有 Flow Matching 论文，阅读 Abstract 和 Introduction
- 查看 `modeling_smolvla.py` 中的 `forward` 函数

**产出**：
- [ ] 创建 `flow_matching_notes.md`
- [ ] 解释 Flow Matching vs Diffusion 的区别
- [ ] 描述 SmolVLA 的 10-step 积分过程

---

## 验证标准

完成 Week 1 后，你应该能够：

1. ✅ 说出 SigLIP、Gemma、Action Expert 在 SmolVLA 中的作用
2. ✅ 解释 Flow Matching 和 Diffusion 的核心区别
3. ✅ 画出 SmolVLA 从图像到动作的完整数据流图
4. ✅ 统计各部分的参数量和显存占用
5. ✅ 分析 Task 5/9 的失败原因，并分类

---

## 周产出清单

| 产出 | 文件 | 状态 |
|------|------|------|
| 参数分布表 | `smolvla_params_table.md` | ⬜ |
| Task 5/9 失败分析 | `task5_9_failure_analysis.md` | ⬜ |
| 架构图 | `smolvla_architecture.drawio` 或手绘 | ⬜ |
| Flow Matching 笔记 | `flow_matching_notes.md` | ⬜ |

---

## 下周预告

Week 2 将深入代码工程：
- 在 `prepare_state` 中加入数据检查点
- 实现视觉注意力可视化
- 修改 `train_expert_only` 开关逻辑

---

## 参考资料

- [Flow Matching 论文](https://arxiv.org/abs/2402.03082)
- [SigLIP 论文](https://arxiv.org/abs/2309.17420)
- [Gemma 模型](https://huggingface.co/google/gemma-2b)
- LeRobot SmolVLA: `src/lerobot/policies/smolvla/modeling_smolvla.py`