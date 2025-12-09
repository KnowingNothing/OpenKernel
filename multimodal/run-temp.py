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
    # run verification multiple times (silent inner runs)
    VERIFY_RUNS = 5
    max_abs_list = []
    mean_abs_list = []
    max_rel_list = []
    allclose_list = []
    scale = 1.0 / (head_dim ** 0.5)
    for _ in range(VERIFY_RUNS):
        # Custom kernel output
        out_custom = custom_attention.forward(q, k, v, mask)
        # reference
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        attn_weights_once = torch.softmax(attn_scores, dim=-1)
        out_ref_once = torch.matmul(attn_weights_once, v)
        abs_diff_once = (out_custom - out_ref_once).abs()
        rel_diff_once = abs_diff_once / (out_ref_once.abs() + 1e-8)
        max_abs_list.append(abs_diff_once.max().item())
        mean_abs_list.append(abs_diff_once.mean().item())
        max_rel_list.append(rel_diff_once.max().item())
        allclose_list.append(torch.allclose(out_custom, out_ref_once, atol=1e-4, rtol=1e-4))

    # concise summary
    worst_max_abs = max(max_abs_list)
    avg_mean_abs = sum(mean_abs_list) / len(mean_abs_list)
    any_allclose = all(allclose_list)
    print(f"验证（{VERIFY_RUNS}次）汇总: worst_max_abs={worst_max_abs:.6e}, avg_mean_abs={avg_mean_abs:.6e}, allclose_every_run={any_allclose}")
    # 性能计时（多样本/多 seed 测试）已移至 seed_based_timing()
    # 如果需要对 softmax/operator 性能做更系统的测量，请运行 seed_based_timing() 或在主脚本中启用对应功能。

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


def seed_based_timing():
    """
    在验证之后使用固定 20 个 seed 做性能计时：
    - 对每个 seed 设置 RNG
    - 生成缩小幅度的 q/k/v（避免过大数值）和 causal mask
    - warm-up 1 次，计时 5 次，取平均
    - 遇到异常跳过并记录失败的 seed
    """
    import custom_attention
    # print("\n=== Seed-based timing (20 fixed seeds) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # 20 固定 seed（可按需修改）
    seeds = [466, 19543, 1200, 8732, 250, 16011, 4321, 9999, 2048, 15000, 678, 3021, 18999, 4200, 7777, 201, 1560, 11000, 840, 13245]

    # 最大尺寸上限（防止生成过大张量导致 OOM），可通过环境变量覆盖
    max_B = int(os.getenv('ATTN_MAX_B', '2'))
    max_H = int(os.getenv('ATTN_MAX_H', '16'))
    # 根据 GPU 总显存选择更保守的默认 max_S（如检测到 >=24GB 则使用 1024）
    if torch.cuda.is_available():
        try:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            if gpu_mem >= 24 * (1024 ** 3):
                default_max_S = '1024'
            else:
                default_max_S = '4096'
        except Exception:
            default_max_S = '2048'
    else:
        default_max_S = '2048'
    max_S = int(os.getenv('ATTN_MAX_S', default_max_S))
    max_D = int(os.getenv('ATTN_MAX_D', '64'))
    # 保守因子：在估计内存时乘以此因子以留出额外头寸，默认 1.8
    try:
        safety_factor = float(os.getenv('ATTN_SAFETY_FACTOR', '1.8'))
    except Exception:
        safety_factor = 1.8

    # helper: env aware int
    def _get_env_int(name, default):
        try:
            return int(os.getenv(name, default))
        except Exception:
            return default

    # estimate bytes (same logic as verify)
    def estimate_bytes(B,H,S,D):
        bytes_scores = B*H*S*S*4
        bytes_weights = bytes_scores
        bytes_qkv = 3*B*H*S*D*4
        bytes_out = B*H*S*D*4
        bytes_mask = S*S*1
        return bytes_scores + bytes_weights + bytes_qkv + bytes_out + bytes_mask

    # 更保守的 required_bytes 估计（包含 softmax 临时权重/工作内存）
    def required_bytes_for_run(B,H,S,D):
        base = estimate_bytes(B,H,S,D)
        # softmax 临时可能需要额外的 weights 张量（B*H*S*S）和一些工作缓冲
        softmax_extra = B*H*S*S*4
        misc_buf = B*H*S*D*4
        return base + softmax_extra + misc_buf

    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory
    else:
        total_mem = 8 * (1024**3)
    budget = int(total_mem * 0.6)

    results = []
    failed = []

    for s in seeds:
        try:
            random.seed(s)
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)
            # If we have a fixed shape mapping for reproducibility, use it.
            fixed_shapes = {
                466:  (1, 5, 2651, 56),
                19543:(1, 5, 3366, 64),
                1200: (2,13,2872,44),
                8732: (2,14,4096,64),
                250:  (2,10,4096,52),
                16011:(2,7,4096,64),
                4321: (2,4,3594,40),
                9999: (1,6,4096,41),
                2048: (1,11,4096,64),
                15000:(2,14,1808,64),
                678:  (1,9,270,64),
                3021: (1,4,279,64),
                18999:(2,10,3348,64),
                4200: (2,10,704,64),
                7777: (2,12,2916,64),
                201:  (1,15,2923,64),
                1560: (2,14,1510,64),
                11000:(1,12,3416,64),
                840:  (2,10,3998,64),
                13245:(2,5,2990,64),
            }
            if s in fixed_shapes:
                B, H, S, D = fixed_shapes[s]
            else:
                # shape sampling (deterministic because RNG seeded)
                B = _get_env_int('ATTN_B', random.randint(1, 4))
                H = _get_env_int('ATTN_H', random.randint(4, 16))
                S = _get_env_int('ATTN_S', random.randint(256, 8192)) # or 4096
                D = _get_env_int('ATTN_D', random.randint(32, 128))
            # enforce user / default maxima to avoid OOM
            # If using fixed_shapes, skip clipping so we reproduce exactly; otherwise clip
            if s not in fixed_shapes:
                B = max(1, min(B, max_B))
                H = max(1, min(H, max_H))
                D = max(1, min(D, max_D))
                S = max(16, min(S, max_S))
            # 更严格地按安全因子裁剪 S（避免实际峰值超出估计）
            effective_budget = int(budget / max(1.0, safety_factor))
            while S > 16 and estimate_bytes(B, H, S, D) > effective_budget:
                S = max(16, S // 2)

            # 计算完整运行所需内存，如果仍然超出 effective_budget，则跳过该 seed
            req = required_bytes_for_run(B, H, S, D)
            if req > effective_budget:
                print(f"Seed {s}: required_bytes ({req/1024/1024:.2f} MB) exceeds effective_budget ({effective_budget/1024/1024:.2f} MB); skipping seed. shape={(B,H,S,D)}")
                failed.append(s)
                continue

            # 在分配前做一次 try 清理缓存以减小碎片影响
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            # (removed per-seed GPU memory debug prints)

            # build small-valued inputs to avoid large activations
            q = (torch.rand(B, H, S, D, device=device, dtype=dtype) - 0.5) * 0.04
            k = (torch.rand(B, H, S, D, device=device, dtype=dtype) - 0.5) * 0.04
            v = (torch.rand(B, H, S, D, device=device, dtype=dtype) - 0.5) * 0.04
            mask = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)

            # warm-up
            _ = custom_attention.forward(q, k, v, mask)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # timed runs
            times = []
            import time as _time
            for r in range(5):
                t0 = _time.perf_counter()
                _ = custom_attention.forward(q, k, v, mask)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = _time.perf_counter()
                times.append((t1 - t0) * 1000.0)

            avg_ms = sum(times) / len(times)
            results.append((s, avg_ms, times, (B, H, S, D)))

        except Exception:
            print(f"Seed {s} FAILED; recording inputs and continuing.")
            traceback.print_exc()
            # ensure logs dir exists
            try:
                os.makedirs('./logs', exist_ok=True)
            except Exception:
                pass
            try:
                save_path = f"./logs/seed_{s}_fail.pt"
                meta = {
                    'shape': (B, H, S, D),
                    'seed': s,
                }
                # 尝试收集显存摘要（如果可用）以便离线分析
                try:
                    if torch.cuda.is_available():
                        meta['cuda_reserved'] = torch.cuda.memory_reserved()
                        meta['cuda_allocated'] = torch.cuda.memory_allocated()
                        meta['cuda_max_allocated'] = torch.cuda.max_memory_allocated()
                        # 更详细的内存报告（字符串）
                        meta['cuda_summary'] = torch.cuda.memory_summary()
                except Exception:
                    pass

                # 保存尽可能多的张量（移动到 CPU），部分情况下分配失败时 q/k/v 可能未定义
                save_dict = {'meta': meta}
                try:
                    save_dict['q'] = q.cpu()
                    save_dict['k'] = k.cpu()
                    save_dict['v'] = v.cpu()
                    save_dict['mask'] = mask.cpu()
                except Exception:
                    # 如果张量未成功分配，仍保存 meta
                    pass

                torch.save(save_dict, save_path)
                print(f"  Saved failed inputs and memory summary to: {save_path}")
            except Exception:
                print("  Failed to save inputs for seed", s)
                traceback.print_exc()
            failed.append(s)
            # try to clear CUDA error state before continuing
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
            continue

    # Print numbered summary 1..N (aligned with seeds order). Show avg ms or FAILED.
    print("\n=== Seed timing summary ===")
    # build a mapping from seed to result for lookup
    result_map = {s: (avg_ms, times, shape) for (s, avg_ms, times, shape) in results}
    for idx, s in enumerate(seeds, start=1):
        if s in result_map:
            avg_ms, times, _shape = result_map[s]
            print(f"{idx:2d} : {avg_ms:.3f} ms")
        else:
            print(f"{idx:2d} : FAILED")
    if failed:
        print("Failed seeds:", failed)


    return results, failed

if __name__ == "__main__":
    verify_custom_kernel_vs_pytorch()
    # 在验证之后运行 seed-based timing
    seed_based_timing()