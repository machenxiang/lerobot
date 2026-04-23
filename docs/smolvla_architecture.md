# SmolVLA Architecture Analysis

> 基于 LeRobot 框架中 SmolVLA 实现（v0.5.2）的深度分析文档

---

## 1. Model Overview

### 1.1 SmolVLA 定位

- **参数量**：450M
- **基础模型**：HuggingFaceTB/SmolVLM2-500M-Video-Instruct
- **任务**：视觉-语言-动作（VLA）预测
- **核心范式**：Flow Matching（流匹配）

### 1.2 类层次结构

```
SmolVLAPolicy (lerobot wrapper)
└── VLAFlowMatching (core model)
    ├── vlm_with_expert: SmolVLMWithExpertModel
    │   ├── vlm: SmolVLMForConditionalGeneration
    │   │   ├── vision_model: SigLIP
    │   │   ├── connector: Modality Projection
    │   │   └── text_model: Gemma (16 layers, 3B equiv)
    │   └── lm_expert: AutoModel (smaller Gemma, 0.75x width)
    ├── state_proj: nn.Linear (32 → VLM hidden)
    ├── action_in_proj: nn.Linear (action_dim → expert hidden)
    ├── action_out_proj: nn.Linear (expert hidden → action_dim)
    ├── action_time_mlp_in: nn.Linear
    └── action_time_mlp_out: nn.Linear
```

---

## 2. Input/Output Specification

### 2.1 Configuration Parameters

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_obs_steps` | 1 | 观测帧数 |
| `chunk_size` | 50 | 动作块大小 |
| `n_action_steps` | 50 | 每次调用的动作步数 |
| `max_state_dim` | 32 | 状态维度（padding后） |
| `max_action_dim` | 32 | 动作维度（padding后） |
| `resize_imgs_with_padding` | (512, 512) | 图像 resize 尺寸 |
| `vlm_model_name` | "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" | VLM 骨干 |
| `num_vlm_layers` | 16 | Gemma 层数（原始38层） |
| `expert_width_multiplier` | 0.75 | Expert 相对 VLM 的宽度比例 |
| `attention_mode` | "cross_attn" | 注意力模式 |

### 2.2 Inputs

**`forward()` 输入参数：**

| 参数 | Shape | 说明 |
|------|-------|------|
| `images` | List[(B, C, H, W)] | 多相机图像列表 |
| `img_masks` | List[(B, num_patches)] | 有效图像区域 mask |
| `lang_tokens` | (B, 48) | 语言任务描述 tokens |
| `lang_masks` | (B, 48) | 语言注意力 mask |
| `state` | (B, 32) | 机器人状态（关节位置） |
| `actions` | (B, 50, 32) | 目标动作序列 |
| `noise` | (B, 50, 32) | Flow matching 噪声 |
| `time` | (B,) | 时间步 t ∈ [0.001, 1.0] |

### 2.3 Outputs

| 参数 | Shape | 说明 |
|------|-------|------|
| `losses` | (B, 50, 32) | Flow matching MSE 损失 |

---

## 3. Encoder Components

### 3.1 Vision Encoder: SigLIP

**位置**: `smolvlm_with_expert.py:179-192`

```python
def embed_image(self, image: torch.Tensor):
    image_hidden_states = (
        self.get_vlm_model()
        .vision_model(
            pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype),
        )
        .last_hidden_state
    )
    # Modality projection & resampling
    image_hidden_states = self.get_vlm_model().connector(image_hidden_states)
    return image_hidden_states
```

**为什么选择 SigLIP：**
1. **Sigmoid Loss Training**：对比损失函数，更好的图文对齐
2. **视觉-语言预训练**：与 Gemma LLM 联合训练效果好
3. **高效**：相比 CLIP 推理速度更快

### 3.2 Language Encoder: Gemma Embeddings

**位置**: `smolvlm_with_expert.py:194-195`

```python
def embed_language_tokens(self, tokens: torch.Tensor):
    return self.get_vlm_model().text_model.get_input_embeddings()(tokens)
```

**为什么选择 Gemma：**
1. **轻量高性能**：3B 参数级别效果优秀
2. **大规模预训练**：海量 web 数据
3. **指令遵循能力强**：适合机器人任务描述

### 3.3 Modality Connector

- 将 SigLIP 输出的 vision hidden states 投影到 Gemma 的 hidden dimension
- 实现视觉和语言特征空间的统一

---

## 4. Decoder Components: Action Expert + Flow Matching

### 4.1 Action Expert Architecture

**位置**: `smolvlm_with_expert.py:61-132`

```python
def __init__(self, ...):
    # Load pretrained VLM (38 layer Gemma)
    self.vlm = AutoModelForImageTextToText.from_pretrained(model_id, ...)

    # Create smaller action expert (75% width)
    lm_expert_config = copy.deepcopy(config.text_config)
    hidden_size = lm_expert_config.hidden_size
    lm_expert_config.hidden_size = int(hidden_size * 0.75)
    self.lm_expert = AutoModel.from_config(lm_expert_config)
```

**Expert 参数：**
- 使用 16 层（与 VLM 相同）
- Hidden size = VLM hidden × 0.75
- 接收来自 VLM KV cache 的 cross-attention

### 4.2 Cross-Attention Mechanism

**位置**: `smolvlm_with_expert.py:274-387`

```python
def forward_cross_attn_layer(self, ...):
    # Expert query 投影
    expert_query_state = expert_layer.self_attn.q_proj(expert_hidden_states)

    # VLM 的 K, V 投影到 Expert 维度
    _key_states = key_states.to(dtype=expert_layer.self_attn.k_proj.weight.dtype)
    expert_key_states = expert_layer.self_attn.k_proj(_key_states)
    expert_value_states = expert_layer.self_attn.v_proj(_value_states)

    # Expert cross-attention 到 VLM KV
    att_output = attention_interface(
        expert_attention_mask, batch_size, head_dim,
        expert_query_states, expert_key_states, expert_value_states,
    )
```

**两种注意力模式：**
1. `self_attn`: Expert 自注意力（每 N 层）
2. `cross_attn`: Expert cross-attention 到 VLM KV cache（默认）

### 4.3 为什么使用 Flow Matching

**传统 Diffusion 的问题：**
- 需要多步 reverse 过程（50-100步）
- 推理慢，不适合实时控制

**Flow Matching 的优势：**
1. **连续动作空间**：自然处理连续动作
2. **简化的训练**：无需 reverse process
3. **快速采样**：Euler 积分，10 步即可
4. **理论保证**：最优传输路径

---

## 5. Forward Pass: How Outputs Are Computed

### 5.1 Prefix Embedding (Observations)

**位置**: `smolvlm_with_expert.py:626-718`

```python
def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state):
    embs = []

    # 1. Image embeddings (SigLIP → connector)
    for img, img_mask in zip(images, img_masks):
        img_emb = self.vlm_with_expert.embed_image(img)
        img_emb = img_emb * sqrt(img_emb_dim)
        embs.append(img_emb)

    # 2. Language embeddings (Gemma)
    lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
    lang_emb = lang_emb * sqrt(lang_emb_dim)
    embs.append(lang_emb)

    # 3. State embedding (project to VLM hidden dim)
    state_emb = self.state_proj(state)  # 32 → VLM hidden
    embs.append(state_emb)

    return embs, pad_masks, att_masks
```

### 5.2 Suffix Embedding (Noisy Actions + Timestep)

**位置**: `smolvlm_with_expert.py:720-761`

```python
def embed_suffix(self, noisy_actions, timestep):
    # Project actions to expert dimension
    action_emb = self.action_in_proj(noisy_actions)  # 32 → expert_hidden

    # Timestep encoding (sinusoidal)
    time_emb = create_sinusoidal_pos_embedding(
        timestep,
        self.vlm_with_expert.expert_hidden_size,
        min_period=4e-3,
        max_period=4.0,
    )

    # Fuse action + timestep
    action_time_emb = torch.cat([action_emb, time_emb], dim=2)
    action_time_emb = self.action_time_mlp_in(action_time_emb)
    action_time_emb = F.silu(action_time_emb)
    action_time_emb = self.action_time_mlp_out(action_time_emb)

    return action_time_emb
```

**Timestep Encoding 公式：**
```
period = min_period * (max_period / min_period) ^ fraction
scaling_factor = 2π / period
pos_emb = [sin(scaling_factor * t), cos(scaling_factor * t)]
```

### 5.3 Full Forward Pass

**位置**: `modeling_smolvla.py:763-799`

```python
def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
    # 1. Sample noise and time
    if noise is None:
        noise = self.sample_noise(actions.shape, actions.device)
    if time is None:
        time = self.sample_time(actions.shape[0], actions.device)

    # 2. Flow interpolation: x_t = t*noise + (1-t)*actions
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions

    # 3. Target velocity: u_t = noise - actions
    u_t = noise - actions

    # 4. Embed prefix (images, language, state)
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(...)

    # 5. Embed suffix (noisy actions + timestep)
    suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

    # 6. Build 2D attention masks
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)

    # 7. Forward through VLM + Expert
    (_, suffix_out), _ = self.vlm_with_expert.forward(
        attention_mask=att_2d_masks,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
    )

    # 8. Extract action predictions
    suffix_out = suffix_out[:, -self.config.chunk_size:]
    v_t = self.action_out_proj(suffix_out)

    # 9. Compute flow matching loss
    losses = F.mse_loss(u_t, v_t, reduction="none")
    return losses
```

---

## 6. Loss Calculation: Flow Matching Loss

### 6.1 Time Sampling

**位置**: `modeling_smolvla.py:620-624`

```python
def sample_time(self, bsize, device):
    beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
    time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
    time = time_beta * 0.999 + 0.001  # Map to [0.001, 1.0]
    return time
```

- 使用 Beta(1.5, 1.0) 分布
- 中间时间步采样更多
- 偏移到 [0.001, 1.0] 避免边界

### 6.2 Noise Sampling

**位置**: `modeling_smolvla.py:610-618`

```python
def sample_noise(self, shape, device):
    noise = torch.normal(
        mean=0.0,
        std=1.0,
        size=shape,
        dtype=torch.float32,
        device=device,
    )
    return noise
```

### 6.3 Flow Matching Loss Formula

```python
# 线性插值
x_t = t * noise + (1-t) * actions

# 目标速度场
u_t = noise - actions

# 预测速度场
v_t = model(x_t, t)

# MSE 损失
losses = F.mse_loss(u_t, v_t, reduction="none")
```

### 6.4 完整公式

| 步骤 | 公式 | 说明 |
|------|------|------|
| 1. 采样时间 | t ~ Beta(1.5, 1.0) | 中间步更多 |
| 2. 采样噪声 | ε ~ N(0, 1) | 标准高斯 |
| 3. 插值 | x_t = tε + (1-t)x₀ | 噪声与数据之间 |
| 4. 速度目标 | u_t = ε - x₀ | 导数 |
| 5. 模型预测 | v_θ(x_t, t) | 网络输出 |
| 6. 损失 | L = \|\|u_t - v_θ\|\|² | MSE |

---

## 7. Inference: Euler Integration

### 7.1 Sampling Process

**位置**: `modeling_smolvla.py:801-870`

```python
def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
    # 1. Start from random noise
    x_t = noise if noise is not None else self.sample_noise(...)

    # 2. Pre-compute VLM KV cache (once for all steps)
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(...)
    _, past_key_values = self.vlm_with_expert.forward(
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
        fill_kv_cache=True,
    )

    # 3. Euler integration (10 steps)
    dt = -1.0 / num_steps  # -0.1
    x_t = noise

    for step in range(num_steps):
        time = 1.0 + step * dt  # 1.0 → 0.0

        # Denoise one step
        v_t = self.denoise_step(
            x_t=x_t,
            past_key_values=past_key_values,
            timestep=time,
        )

        # Euler update
        x_t = x_t + dt * v_t

    return x_t
```

### 7.2 Euler Integration Formula

```
x_{t+dt} = x_t + dt * v_t

其中:
- dt = -1/num_steps = -0.1
- time 从 1.0 降到 0.0
- num_steps 默认 = 10
```

---

## 8. Key Classes Summary

### 8.1 `modeling_smolvla.py`

| 类/函数 | 行号 | 功能 |
|---------|------|------|
| `SmolVLAPolicy` | 224 | LeRobot 包装类 |
| `VLAFlowMatching` | 530 | 核心 VLM + Expert 模型 |
| `create_sinusoidal_pos_embedding` | 80 | 时间步编码 |
| `make_att_2d_masks` | 101 | 2D 注意力掩码 |
| `resize_with_pad` | 134 | 图像预处理 |

### 8.2 `smolvlm_with_expert.py`

| 类/函数 | 行号 | 功能 |
|---------|------|------|
| `SmolVLMWithExpertModel` | 61 | VLM + Action Expert 组合 |
| `forward_cross_attn_layer` | 274 | Cross-attention 层 |
| `embed_image` | 179 | 图像 embedding |
| `embed_language_tokens` | 194 | 语言 embedding |

---

## 9. Architecture Diagram

```
                    ┌─────────────────────────────┐
                    │                             │
                    │   ┌───────────────────┐     │
                    │   │    SigLIP Vision   │     │
                    │   │      Encoder       │     │
                    │   └─────────┬─────────┘     │
                    │             │              │
                    │   ┌─────────▼─────────┐     │
                    │   │   Modality Conn   │     │
                    │   └─────────┬─────────┘     │
                    │             │              │
                    │   ┌─────────▼─────────┐     │
                    │   │   Gemma (16L)    │     │
                    │   │      VLM         │     │
                    │   └───────┬───────────┘     │
                    │           │                │
                    │           │ KV Cache       │
                    │           ▼                │
                    │   ┌───────────────────┐     │
                    │   │  Expert (16L,    │     │
                    │   │  0.75x width)    │     │
                    │   │  Cross-Attn      │     │
                    │   └───────┬───────────┘     │
                    │           │                │
                    │           ▼                │
                    │   ┌───────────────────┐     │
                    │   │  action_out_proj │     │
                    │   │   (linear)       │     │
                    │   └───────┬───────────┘     │
                    │           │                │
                    └───────────┼────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │    Flow Matching Loss   │
                    │  L = MSE(u_t, v_t)     │
                    └─────────────────────────┘
```

---

## 10. References

- LeRobot v0.5.2: https://github.com/huggingface/lerobot
- SmolVLA Paper: https://arxiv.org/abs/2506.01844
- Flow Matching: https://arxiv.org/abs/2206.00306
- SigLIP: https://arxiv.org/abs/2309.14819
- Gemma: https://arxiv.org/abs/2403.20595