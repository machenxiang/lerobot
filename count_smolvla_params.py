"""
统计 SmolVLA 模型参数分布
Usage: python count_smolvla_params.py
"""

import os
import sys

os.environ["HF_TOKEN"] = ""
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def count_params(module, name=""):
    """统计参数量"""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def format_params(n):
    """格式化参数量"""
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    elif n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n/1_000:.2f}K"
    else:
        return str(n)


def main():
    print("=" * 60)
    print("SmolVLA 参数统计")
    print("=" * 60)

    # 加载模型
    checkpoint_path = "outputs/smolvla_libero_finetune_0427/checkpoints/last/pretrained_model"
    print(f"\nLoading model from: {checkpoint_path}")

    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    policy.to("cpu")
    policy.eval()

    print("Model loaded successfully!\n")

    # 统计各部分参数
    results = []

    # 1. 整体统计
    total_all, train_all = count_params(policy, "Total")
    results.append(("Total", total_all, train_all))

    # 2. VLM 部分
    vlm = policy.model.vlm_with_expert.vlm
    v_total, v_train = count_params(vlm, "VLM")
    results.append(("VLM (包含 Vision + Connector + Text)", v_total, v_train))

    # 3. Vision Encoder (SigLIP)
    vision = vlm.model.vision_model
    vis_total, vis_train = count_params(vision, "Vision Encoder (SigLIP)")
    results.append(("  Vision Encoder", vis_total, vis_train))

    # 4. Connector
    connector = vlm.model.connector
    conn_total, conn_train = count_params(connector, "Connector")
    results.append(("  Connector", conn_total, conn_train))

    # 5. Text Model (Gemma)
    text_model = vlm.model.text_model
    text_total, text_train = count_params(text_model, "Text Model (Gemma)")
    results.append(("  Text Model (Gemma)", text_total, text_train))

    # 6. Action Expert
    expert = policy.model.vlm_with_expert.lm_expert
    expert_total, expert_train = count_params(expert, "Action Expert")
    results.append(("Action Expert (lm_expert)", expert_total, expert_train))

    # 7. State Projection
    state_proj = policy.model.state_proj
    sp_total, sp_train = count_params(state_proj, "State Projection")
    results.append(("  state_proj", sp_total, sp_train))

    # 8. Action Projections
    action_in = policy.model.action_in_proj
    action_out = policy.model.action_out_proj
    action_time_in = policy.model.action_time_mlp_in
    action_time_out = policy.model.action_time_mlp_out

    act_in_total, act_in_train = count_params(action_in, "action_in_proj")
    act_out_total, act_out_train = count_params(action_out, "action_out_proj")
    act_time_in_total, act_time_in_train = count_params(action_time_in, "action_time_mlp_in")
    act_time_out_total, act_time_out_train = count_params(action_time_out, "action_time_mlp_out")

    act_proj_total = act_in_total + act_out_total + act_time_in_total + act_time_out_total
    act_proj_train = act_in_train + act_out_train + act_time_in_train + act_time_out_train

    results.append(("Action Projections", act_proj_total, act_proj_train))
    results.append(("  action_in_proj", act_in_total, act_in_train))
    results.append(("  action_out_proj", act_out_total, act_out_train))
    results.append(("  action_time_mlp_in", act_time_in_total, act_time_in_train))
    results.append(("  action_time_mlp_out", act_time_out_total, act_time_out_train))

    # 打印结果
    print(f"{'Component':<40} {'Total':>10} {'Trainable':>10} {'Frozen':>10}")
    print("-" * 75)
    for name, total, trainable in results:
        frozen = total - trainable
        print(f"{name:<40} {format_params(total):>10} {format_params(trainable):>10} {format_params(frozen):>10}")

    print("-" * 75)

    # 汇总
    vlm_total = v_total
    expert_total = expert_total
    proj_total = sp_total + act_proj_total
    other_total = total_all - vlm_total - expert_total - proj_total

    print(f"\n{'Summary':<40}")
    print("-" * 75)
    print(f"{'VLM (包含 Vision + Connector + Text + lm_head):':<50} {format_params(vlm_total):>10}")
    print(f"{'Action Expert (lm_expert):':<50} {format_params(expert_total):>10}")
    print(f"{'Projections (state + action):':<50} {format_params(proj_total):>10}")
    print(f"{'Other/RTC (if any):':<50} {format_params(other_total):>10}")
    print(f"{'Trainable ratio:':<50} {train_all/total_all*100:.2f}%")

    # 验证：component 相加 ≈ Total
    component_sum = vlm_total + expert_total + proj_total + other_total
    print(f"\n{'Verification:':<40}")
    print(f"{'Component sum:':<40} {format_params(component_sum):>10}")
    print(f"{'Total (from policy):':<40} {format_params(total_all):>10}")
    if abs(component_sum - total_all) > 1000:
        print(f"  ⚠️  Difference: {format_params(abs(component_sum - total_all))} (possible missing layers)")
    else:
        print(f"  ✓ Match!")

    # 显存估算 (更精确)
    print(f"\n{'Memory Estimation (BF16):':<40}")
    print("-" * 55)
    bytes_per_param = 2  # BF16 = 2 bytes
    total_mem = total_all * bytes_per_param / 1e9
    train_mem = train_all * bytes_per_param / 1e9
    frozen_mem = (total_all - train_all) * bytes_per_param / 1e9

    print(f"{'Total model (parameters):':<40} {total_mem:.2f} GB")
    print(f"{'Trainable (requires_grad=True):':<40} {train_mem:.2f} GB")
    print(f"{'Frozen (requires_grad=False):':<40} {frozen_mem:.2f} GB")

    # 梯度显存 (训练时)
    if train_all > 0:
        grad_mem = train_mem  # 梯度通常等于参数大小
        print(f"\n{'Training Memory Breakdown:':<40}")
        print(f"{'  Parameters (frozen):':<40} {frozen_mem:.2f} GB")
        print(f"{'  Parameters (trainable):':<40} {train_mem:.2f} GB")
        print(f"{'  Gradients (trainable):':<40} {grad_mem:.2f} GB")
        print(f"{'  Total (no optimizer/activations):':<40} {frozen_mem + train_mem + grad_mem:.2f} GB")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
