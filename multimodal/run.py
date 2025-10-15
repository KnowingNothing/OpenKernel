import torch 
import torch.profiler 
import traceback 
import os 
import random 
from typing import List, Tuple 
import time 
import torch.nn.functional as F

# Ensure the bagel_mot_attention.py file is in the same directory
try: 
    from bagel_mot_attention import PackedAttentionMoT, Qwen2RotaryEmbedding, NaiveCache 
except ImportError: 
    print("Error: Please ensure bagel_mot_attention.py is in the same directory as this script.") 
    exit() 

def verify_custom_kernel_vs_pytorch():
    import custom_attention
    print("\n=== Custom Kernel vs PyTorch算子顺序实现 验证 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    # 可通过环境变量覆盖初始规模
    def _get_env_int(name, default):
        try:
            return int(os.getenv(name, default))
        except Exception:
            return default
    # 默认使用区间内的随机值，使每次运行的数据集大小不一致且更均匀；如设置了环境变量则优先生效
    batch_size = _get_env_int('ATTN_B', random.randint(1, 4))
    num_heads = _get_env_int('ATTN_H', random.randint(4, 32))
    seq_len = _get_env_int('ATTN_S', random.randint(256, 8192))
    head_dim = _get_env_int('ATTN_D', random.randint(64, 128))
    # 基于显存预算对 seq_len 做内存安全裁剪（主要约束来自 S^2 的 scores/weights）
    def estimate_bytes(B,H,S,D):
        bytes_scores = B*H*S*S*4
        bytes_weights = bytes_scores
        bytes_qkv = 3*B*H*S*D*4
        bytes_out = B*H*S*D*4
        bytes_mask = S*S*1
        return bytes_scores + bytes_weights + bytes_qkv + bytes_out + bytes_mask
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory
    else:
        total_mem = 8 * (1024**3)
    budget = int(total_mem * 0.6)  # 留出空间给系统/缓存
    while seq_len > 16 and estimate_bytes(batch_size, num_heads, seq_len, head_dim) > budget:
        seq_len = max(16, seq_len // 2)
    print(f"[verify] 使用尺寸: B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim} (预算≈{budget/1024/1024/1024:.1f} GB)")
    q = torch.rand(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.rand(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.rand(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    # causal mask: True=mask
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    print(f"Tensor shapes: q {q.shape}, k {k.shape}, v {v.shape}, mask {mask.shape}")
    # Custom kernel output
    out_custom = custom_attention.forward(q, k, v, mask)
    # PyTorch reference（先做一次正确性验证，不计时）
    scale = 1.0 / (head_dim ** 0.5)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_scores = attn_scores.masked_fill(mask, float('-inf'))
    # 正确性验证
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    attn_weights_once = torch.softmax(attn_scores, dim=-1)
    out_ref_once = torch.matmul(attn_weights_once, v)
    abs_diff_once = (out_custom - out_ref_once).abs()
    rel_diff_once = abs_diff_once / (out_ref_once.abs() + 1e-8)
    max_abs_once = abs_diff_once.max().item()
    mean_abs_once = abs_diff_once.mean().item()
    max_rel_once = rel_diff_once.max().item()
    allclose_once = torch.allclose(out_custom, out_ref_once, atol=1e-4, rtol=1e-4)
    print(f"正确性: max_abs={max_abs_once:.6e}, mean_abs={mean_abs_once:.6e}, max_rel={max_rel_once:.6e}, allclose={allclose_once}")
    if allclose_once:
        print("✅ SUCCESS: The outputs of the custom operator and PyTorch are consistent!")
    else:
        print("❌ FAILURE: The outputs do not match.")
        print(f"   Max absolute difference: {max_abs_once:.6e}")
        print("   Possible causes:")
        print("   - Mask dtype/semantics mismatch: use boolean mask where True=masked，apply -inf before softmax")
        print("   - Scaling mismatch: ensure logits scaled by 1/sqrt(D)")
        print("   - Softmax stability: subtract row-wise max before exp/softmax")
        print("   - Dtype issues: compute QK^T/softmax/weights@V in float32")
        print("   - Broadcasting/shape: verify mask is [S,S] and broadcasted correctly across [B,H]")
    # 计时 softmax：改为每次迭代随机生成不同数据集大小，并对各自 softmax 计时后再求平均
    import time
    times = []
    shapes = []
    N = int(os.getenv('ATTN_N', '10'))
    print(f"\n--- Variable-size Softmax timing across {N} randomly sampled shapes ---")
    for i in range(N):
        # 为本次计时随机采样一个新形状，并做内存裁剪
        bs_i = _get_env_int('ATTN_B', random.randint(1, 4))
        nh_i = _get_env_int('ATTN_H', random.randint(4, 32))
        sl_i = _get_env_int('ATTN_S', random.randint(256, 8192))
        hd_i = _get_env_int('ATTN_D', random.randint(64, 128))
        while sl_i > 16 and estimate_bytes(bs_i, nh_i, sl_i, hd_i) > budget:
            sl_i = max(16, sl_i // 2)

        # 构造张量并计算 attn_scores（仅用于 softmax 计时）
        q_i = torch.rand(bs_i, nh_i, sl_i, hd_i, device=device, dtype=dtype)
        k_i = torch.rand(bs_i, nh_i, sl_i, hd_i, device=device, dtype=dtype)
        mask_i = torch.triu(torch.ones(sl_i, sl_i, device=device, dtype=torch.bool), diagonal=1)
        scale_i = 1.0 / (hd_i ** 0.5)
        scores_i = torch.matmul(q_i, k_i.transpose(-2, -1)) * scale_i
        scores_i = scores_i.masked_fill(mask_i, float('-inf'))

        # 每个形状做一次短 warmup，避免首次开销影响
        _ = torch.softmax(scores_i, dim=-1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 正式计时
        t0 = time.perf_counter()
        _ = torch.softmax(scores_i, dim=-1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        elapsed = (t1 - t0) * 1000
        times.append(elapsed)
        shapes.append((bs_i, nh_i, sl_i, hd_i))
        print(f"#{i+1}: B={bs_i}, H={nh_i}, S={sl_i}, D={hd_i} -> Softmax: {elapsed:.4f} ms")

    # 剔除最大/最小后求平均
    times_sorted = sorted(times)
    if N > 2:
        trimmed = times_sorted[1:-1]
    else:
        trimmed = times_sorted
    avg = sum(trimmed) / len(trimmed)
    print(f"Variable-size Softmax去除极值后平均耗时: {avg:.4f} ms")

    # 首个样本的详细对比打印（避免与不同形状的计时变量混淆）
    abs_diff = abs_diff_once
    out_ref = out_ref_once
    # 输出前2个batch、前2个head、前2个seq、前8个特征（基于首个样本）
    for b in range(min(2, out_custom.shape[0])):
        for h in range(min(2, out_custom.shape[1])):
            for s in range(min(2, out_custom.shape[2])):
                print(f"[b={b},h={h},s={s}] custom:   ", out_custom[b,h,s,:8].cpu().numpy())
                print(f"[b={b},h={h},s={s}] pytorch:  ", out_ref[b,h,s,:8].cpu().numpy())
                print(f"[b={b},h={h},s={s}] abs_diff: ", abs_diff[b,h,s,:8].cpu().numpy())

# Copyright 2025 Bytedance Ltd. and/or its affiliates. 
# SPDX-License-Identifier: Apache-2.0 

if __name__ == "__main__":
    verify_custom_kernel_vs_pytorch()