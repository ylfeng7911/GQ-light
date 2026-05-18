import torch
import torch.nn as nn
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.zoo.rtdetr.modules import MSDeformableAttention


def fmt_num(x):
    """格式化大数字"""
    if x >= 1e9:
        return f"{x / 1e9:.4f} G"
    elif x >= 1e6:
        return f"{x / 1e6:.4f} M"
    elif x >= 1e3:
        return f"{x / 1e3:.4f} K"
    else:
        return str(x)


def mha_self_attention_macs(
    batch_size: int,
    num_queries: int,
    embed_dim: int,
    num_heads: int,
    include_linear: bool = True,
):
    """
    统计 nn.MultiheadAttention self-attention 的 MACs。

    约定：
    - 这里统计 MACs，即 multiply-accumulate 次数。
    - 如果你想换成 FLOPs，常见做法是 FLOPs = 2 * MACs。
    """

    B = batch_size
    N = num_queries
    C = embed_dim
    H = num_heads
    D = C // H

    assert C % H == 0

    macs = {}

    if include_linear:
        # Q, K, V 三个线性投影: 3 * B * N * C * C
        macs["qkv_proj"] = 3 * B * N * C * C
    else:
        macs["qkv_proj"] = 0

    # attention score: Q @ K^T
    # 每个 head: [N, D] @ [D, N] -> [N, N]
    # 总计: B * H * N * N * D = B * N * N * C
    macs["attn_score"] = B * H * N * N * D

    # attention value: Attn @ V
    # 每个 head: [N, N] @ [N, D] -> [N, D]
    # 总计: B * H * N * N * D = B * N * N * C
    macs["attn_value"] = B * H * N * N * D

    if include_linear:
        # output projection: B * N * C * C
        macs["out_proj"] = B * N * C * C
    else:
        macs["out_proj"] = 0

    total = sum(macs.values())
    return macs, total


def ms_deformable_cross_attention_macs(
    batch_size: int,
    num_queries: int,
    num_memory_tokens: int,
    embed_dim: int,
    num_heads: int,
    num_levels: int,
    num_points: int,
    include_linear: bool = True,
    include_sampling: bool = True,
):
    """
    统计 MSDeformableAttention cross-attention 的近似 MACs。

    对应常见 MSDeformableAttention 结构：
    - value_proj: memory -> value
    - sampling_offsets: query -> offsets
    - attention_weights: query -> weights
    - deformable sampling and weighted sum
    - output_proj

    注意：
    deformable sampling 通常涉及 grid_sample / 双线性插值，
    这个部分不同实现统计方式不同，这里给一个近似估计。
    """

    B = batch_size
    Nq = num_queries
    S = num_memory_tokens
    C = embed_dim
    H = num_heads
    L = num_levels
    P = num_points
    D = C // H

    assert C % H == 0

    macs = {}

    if include_linear:
        # value projection: memory [B, S, C] -> [B, S, C]
        macs["value_proj"] = B * S * C * C

        # sampling_offsets projection:
        # query [B, Nq, C] -> [B, Nq, H * L * P * 2]
        offset_dim = H * L * P * 2
        macs["sampling_offsets"] = B * Nq * C * offset_dim

        # attention_weights projection:
        # query [B, Nq, C] -> [B, Nq, H * L * P]
        weight_dim = H * L * P
        macs["attention_weights"] = B * Nq * C * weight_dim

        # output projection: [B, Nq, C] -> [B, Nq, C]
        macs["out_proj"] = B * Nq * C * C
    else:
        macs["value_proj"] = 0
        macs["sampling_offsets"] = 0
        macs["attention_weights"] = 0
        macs["out_proj"] = 0

    if include_sampling:
        # 近似统计 weighted sum:
        # 对每个 query、每个 head、每个 level、每个 point、每个 head_dim 做加权求和
        #
        # B * Nq * H * L * P * D = B * Nq * L * P * C
        #
        # 如果严格统计双线性插值，通常还要乘一个常数因子，比如 4 个邻居点插值。
        # 这里先统计核心 weighted aggregation。
        macs["sampling_weighted_sum"] = B * Nq * H * L * P * D
    else:
        macs["sampling_weighted_sum"] = 0

    total = sum(macs.values())
    return macs, total


def print_macs_table(name, macs_dict, total_macs):
    print("=" * 80)
    print(name)
    print("=" * 80)
    for k, v in macs_dict.items():
        print(f"{k:<30}: {fmt_num(v)} MACs, {fmt_num(2 * v)} FLOPs")
    print("-" * 80)
    print(f"{'Total':<30}: {fmt_num(total_macs)} MACs, {fmt_num(2 * total_macs)} FLOPs")
    print()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------
    # 基本配置
    # ------------------------------------------------------------
    B = 1
    num_queries = 300
    embed_dim = 256
    num_heads = 8

    num_levels = 3
    num_points = 4

    # memory tokens = 80*80 + 40*40 + 20*20 = 8400
    memory_spatial_shapes = torch.tensor(
        [[80, 80], [40, 40], [20, 20]],
        dtype=torch.long,
        device=device,
    )
    num_memory_tokens = int((memory_spatial_shapes[:, 0] * memory_spatial_shapes[:, 1]).sum().item())

    # ------------------------------------------------------------
    # 正确创建输入张量
    # ------------------------------------------------------------
    x = torch.randn(B, num_queries, embed_dim, device=device)

    # reference_points 这里按照你原来的接口写
    reference_points = torch.rand(B, num_queries, 1, 4, device=device)

    memory = torch.randn(B, num_memory_tokens, embed_dim, device=device)

    memory_mask = None

    # ------------------------------------------------------------
    # 定义模块
    # ------------------------------------------------------------
    self_attn = nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0,
        batch_first=True,
    ).to(device)

    cross_attn = MSDeformableAttention(
        embed_dim,
        num_heads,
        num_levels,
        num_points,
    ).to(device)

    self_attn.eval()
    cross_attn.eval()

    # ------------------------------------------------------------
    # 前向检查
    # ------------------------------------------------------------
    with torch.no_grad():
        # nn.MultiheadAttention 返回的是 tuple:
        # output1: [B, Nq, C]
        # attn_weights: [B, Nq, Nq] 或相关格式
        output1, attn_weights = self_attn(x, x, x)

        output2 = cross_attn(
            x,
            reference_points,
            memory,
            memory_spatial_shapes,
            memory_mask,
        )

    print("Forward check:")
    print(f"self_attn output shape : {tuple(output1.shape)}")
    print(f"cross_attn output shape: {tuple(output2.shape)}")
    print()

    # ------------------------------------------------------------
    # FLOPs / MACs 统计
    # ------------------------------------------------------------

    # 1. Self-Attention 完整统计：包含 QKV 投影、注意力矩阵、输出投影
    self_macs, self_total = mha_self_attention_macs(
        batch_size=B,
        num_queries=num_queries,
        embed_dim=embed_dim,
        num_heads=num_heads,
        include_linear=True,
    )

    # 2. Cross-Attention 完整统计：包含 value_proj、offset/weight 投影、sampling、out_proj
    cross_macs, cross_total = ms_deformable_cross_attention_macs(
        batch_size=B,
        num_queries=num_queries,
        num_memory_tokens=num_memory_tokens,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        include_linear=True,
        include_sampling=True,
    )

    print_macs_table("Self-Attention in Decoder", self_macs, self_total)
    print_macs_table("MSDeformable Cross-Attention in Decoder", cross_macs, cross_total)

    print("=" * 80)
    print("Comparison")
    print("=" * 80)
    print(f"Self-Attention total MACs : {fmt_num(self_total)}")
    print(f"Cross-Attention total MACs: {fmt_num(cross_total)}")
    print(f"Cross / Self MACs ratio   : {cross_total / self_total:.4f}")
    print()

    # ------------------------------------------------------------
    # 只比较 attention 交互本身，不包含线性投影
    # ------------------------------------------------------------
    self_core_macs, self_core_total = mha_self_attention_macs(
        batch_size=B,
        num_queries=num_queries,
        embed_dim=embed_dim,
        num_heads=num_heads,
        include_linear=False,
    )

    cross_core_macs, cross_core_total = ms_deformable_cross_attention_macs(
        batch_size=B,
        num_queries=num_queries,
        num_memory_tokens=num_memory_tokens,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        include_linear=False,
        include_sampling=True,
    )

    print("=" * 80)
    print("Core Attention Interaction Only")
    print("=" * 80)
    print(f"Self core attention MACs : {fmt_num(self_core_total)}")
    print(f"Cross core sampling MACs : {fmt_num(cross_core_total)}")
    print(f"Cross / Self core ratio  : {cross_core_total / self_core_total:.4f}")