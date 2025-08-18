# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.profiler
import traceback
import os
import random
from typing import List, Tuple

# 确保 bagel_mot_attention.py 文件在同一个目录下
try:
    from bagel_mot_attention import PackedAttentionMoT, Qwen2RotaryEmbedding, NaiveCache
except ImportError:
    print("错误：请确保 bagel_mot_attention.py 与此脚本位于同一目录中。")
    exit()

##########################
# 创建掩码部分
##########################
def create_structured_attention_mask(
    samples_config: List[Tuple[str, int]], 
    device: str, 
    dtype: torch.dtype
) -> List[torch.Tensor]:

    total_length = sum(length for _, length in samples_config)
    
    # 1. 初始化一个完全被屏蔽的掩码张量 (-inf)
    attention_mask_tensor = torch.full(
        (total_length, total_length), 
        -float('inf'), 
        device=device, 
        dtype=dtype
    )

    current_offset = 0
    for sample_type, sample_length in samples_config:
        # 定义当前样本在整个序列中的切片范围
        s = slice(current_offset, current_offset + sample_length)
        
        if sample_type == 'text':
            # 2. 为 'text' 样本应用因果掩码
            causal_mask = torch.tril(
                torch.ones((sample_length, sample_length), device=device, dtype=torch.bool)
            )
            attention_mask_tensor[s, s] = torch.where(causal_mask, 0.0, -float('inf'))
        else: # 'image' 或 'noise'
            # 3. 为 'image' 和 'noise' 样本应用全注意力掩码
            attention_mask_tensor[s, s] = 0.0
        
        current_offset += sample_length

    return [attention_mask_tensor]

# ==============================================================================
# 修改点 1: 修改 run_train_test_with_profiler 函数的定义
# ==============================================================================
def run_train_test_with_profiler(use_custom_kernel: float = False): # <--- 接收一个布尔值参数
    
    print("\n" + "="*80)
    print("--- forward_train (动态数据) 开始 ---")
    print("="*80)

    # ==============================================================================
    # 步骤 0: 环境设置和初始化
    # ==============================================================================

    hidden_size = 4096
    num_attention_heads = 32
    num_key_value_heads = 32
    max_position_embeddings = 16384 # 增加以适应更大的随机序列
    rope_theta = 10000.0
    rms_norm_eps = 1e-6
    head_dim = hidden_size // num_attention_heads
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    if not torch.cuda.is_available():
        print("警告：未检测到 CUDA 设备。程序将在 CPU 上运行，性能可能会较慢。")

    # ==============================================================================
    # 修改点 2: 创建 PackedAttentionMoT 实例时，传入开关状态
    # ==============================================================================
    attention_layer = PackedAttentionMoT(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
        is_causal=False,
        attention_dropout=0.0,
        qk_norm=True,
        rms_norm_eps=rms_norm_eps,
        freeze_und=False,
        layer_idx=0,
        use_custom_kernel=use_custom_kernel  # <--- 将接收到的开关状态传递给模型
    ).to(device=device, dtype=dtype).train()

    rotary_emb = Qwen2RotaryEmbedding(
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta
    ).to(device=device, dtype=dtype)
    
    print(f"步骤 0: 环境设置完成，模块已加载到 {device}。")

    # ==============================================================================
    # 步骤 1: 动态生成样本配置并构建索引
    # ==============================================================================
    # 设置种子以确保可复现性
    random.seed(42)
    torch.manual_seed(42)
    
    num_samples = random.randint(1, 10)
    sample_types = ['text', 'image', 'noise']
    samples_config = []
    for _ in range(num_samples):
        stype = random.choice(sample_types)
        if stype == 'text':
            slen = random.randint(1, 1024)
        else: # image or noise
            slen = random.randint(1024, 10000)
        samples_config.append((stype, slen))

    sequence_length = sum(length for _, length in samples_config)
    sample_lens = [length for _, length in samples_config]

    print("\n步骤 1: 动态生成数据配置完成。")
    print(f"  - 生成的样本数量: {num_samples}")
    print(f"  - 样本配置 (类型, 长度): {samples_config}")
    print(f"  - 总序列长度: {sequence_length}")

    und_indexes, gen_indexes = [], []
    current_offset = 0
    for stype, slen in samples_config:
        indices = torch.arange(current_offset, current_offset + slen, dtype=torch.long, device=device)
        if stype == 'text':
            und_indexes.append(indices)
        else:
            gen_indexes.append(indices)
        current_offset += slen
    
    packed_und_token_indexes = torch.cat(und_indexes) if und_indexes else torch.tensor([], dtype=torch.long, device=device)
    packed_gen_token_indexes = torch.cat(gen_indexes) if gen_indexes else torch.tensor([], dtype=torch.long, device=device)
    
    packed_position_ids = torch.arange(0, sequence_length, dtype=torch.long, device=device)

    # ==============================================================================
    # 步骤 2: 构建核心输入张量 `packed_sequence`
    # ==============================================================================
    packed_sequence = torch.randn(
        sequence_length,
        hidden_size,
        dtype=dtype,
        device=device
    )
    print(f"步骤 2: `packed_sequence` 创建成功，形状: {packed_sequence.shape}")

    # ==============================================================================
    # 步骤 3: 构建位置编码 `packed_position_embeddings`
    # ==============================================================================
    dummy_x_for_rope = torch.randn(1, sequence_length, head_dim, device=device, dtype=dtype)
    packed_cos, packed_sin = rotary_emb(dummy_x_for_rope, position_ids=packed_position_ids.unsqueeze(0))
    packed_position_embeddings = (packed_cos.squeeze(0), packed_sin.squeeze(0))
    print(f"步骤 3: `packed_position_embeddings` 创建成功，形状: (cos: {packed_position_embeddings[0].shape}, sin: {packed_position_embeddings[1].shape})")

    # ==============================================================================
    # 步骤 4: 构建注意力掩码 `attention_mask`
    # ==============================================================================
    print("步骤 4: 正在构建注意力掩码 `attention_mask`...")
    attention_mask = create_structured_attention_mask(samples_config, device, dtype)
    print(f"  掩码列表创建成功，包含 {len(attention_mask)} 个张量，每个形状为 {attention_mask[0].shape}")

    # ==============================================================================
    # 步骤 5: MoE 路由索引已在步骤 1 中创建
    # ==============================================================================
    print(f"步骤 5: MoE 路由索引创建成功。")
    print(f"  - und (text) tokens 数量: {len(packed_und_token_indexes)}")
    print(f"  - gen (image/noise) tokens 数量: {len(packed_gen_token_indexes)}")


    # ==============================================================================
    # 步骤 6: 使用 Profiler 整合并调用 `forward_train`
    # ==============================================================================
    inputs = {
        "packed_sequence": packed_sequence,
        "sample_lens": sample_lens,
        "attention_mask": attention_mask,
        "packed_position_embeddings": packed_position_embeddings,
        "packed_und_token_indexes": packed_und_token_indexes,
        "packed_gen_token_indexes": packed_gen_token_indexes,
    }

    print("\n" + "="*50)
    print("--- 所有输入参数准备就绪，即将调用 forward_train ---")
    print("="*50 + "\n")

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  参数: {k:<28} | 形状: {v.shape}")
        elif isinstance(v, tuple):
            print(f"  参数: {k:<28} | 形状: (cos: {v[0].shape}, sin: {v[1].shape})")
        elif isinstance(v, list) and all(isinstance(i, int) for i in v):
             print(f"  参数: {k:<28} | 值: {v}")
        elif isinstance(v, list) and all(isinstance(i, torch.Tensor) for i in v):
             print(f"  参数: {k:<28} | 类型: List[Tensor], 长度: {len(v)}")
        else:
            print(f"  参数: {k:<28} | 类型: {type(v)}")
    print("="*50 + "\n")

    try:
        log_dir_train = './logs/train'
        print(f"训练模式的性能追踪文件将保存在: {log_dir_train}")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir_train),
            record_shapes=True, profile_memory=True, with_stack=True
        ) as prof:
            for i in range(5): # (1+1+3)*1 = 5
                with torch.autocast(device_type=device, dtype=dtype):
                    output = attention_layer.forward_train(**inputs)
                prof.step()
                print(f"训练模式 Profiler 步骤 {i+1}/5 完成...")

        print("\n--- forward_train 调用成功! ---")
        print(f"训练模式测试通过！性能追踪文件已生成在 {log_dir_train} 目录。")

        print(f"输出张量形状: {output.shape}")
        
        expected_shape = (sequence_length, hidden_size)
        assert output.shape == expected_shape, f"输出形状错误！期望 {expected_shape}, 但得到 {output.shape}"
        print(f"输出形状正确！期望: {expected_shape}, 得到: {output.shape}")

        # --- 添加验证 ---
        assert not torch.isnan(output).any(), "模型输出包含 NaN 值！训练可能不稳定。"
        assert not torch.isinf(output).any(), "模型输出包含 Inf 值！训练可能不稳定。"
        print("✅ 模型输出已通过数值稳定性检查！")

        # 检查并打印输出值
        print("\n--- 输出值概览 ---")
        output_float = output.to(torch.float32)
        print(f"  - 输出张量总和 (Sum): {output_float.sum().item():.4f}")
        print(f"  - 输出张量均值 (Mean): {output_float.mean().item():.4f}")
        print(f"  - 输出张量标准差 (Std): {output_float.std().item():.4f}")

    except Exception as e:
        print(f"\n--- 调用 forward_train 时发生错误 ---")
        traceback.print_exc()


def run_inference_test_with_profiler(use_custom_kernel: float = False):
    """
    测试推理模式 (forward_inference) 并为预填充和解码阶段分别进行性能分析。
    （此函数保持不变）
    """
    print("\n" + "="*80)
    print("--- forward_inference begin ---")
    print("="*80)

    # ==============================================================================
    # 步骤 0: 环境设置和初始化
    # ==============================================================================
    hidden_size = 4096
    num_attention_heads = 32
    num_key_value_heads = 32
    max_position_embeddings = 8192
    rope_theta = 10000.0
    rms_norm_eps = 1e-6
    head_dim = hidden_size // num_attention_heads
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    num_layers = 1

    attention_layer = PackedAttentionMoT(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
        is_causal=True,
        attention_dropout=0.0,
        qk_norm=True,
        rms_norm_eps=rms_norm_eps,
        freeze_und=False,
        layer_idx=0,
        use_custom_kernel=use_custom_kernel  # <--- 将接收到的开关状态传递给模型
    ).to(device=device, dtype=dtype).eval() # 确保是评估模式

    rotary_emb = Qwen2RotaryEmbedding(
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta
    ).to(device=device, dtype=dtype)

    print(f"步骤 0: 推理模式环境设置完成。")

    # ==============================================================================
    # 步骤 1: 模拟推理模式的输入数据
    # ==============================================================================
    num_text_tokens = 6
    num_vae_tokens = 4096
    sequence_length = num_text_tokens + num_vae_tokens
    
    packed_text_indexes = torch.arange(0, num_text_tokens, dtype=torch.long, device=device)
    packed_vae_token_indexes = torch.arange(num_text_tokens, sequence_length, dtype=torch.long, device=device)
    
    # ==============================================================================
    # 阶段 A: 分析预填充 (Prefill) 性能
    # ==============================================================================
    print("\n--- 阶段 A: 预填充 (Prefill) 性能分析 ---")
    
    # 1. 构建预填充阶段的输入张量
    prefill_sequence = torch.randn(sequence_length, hidden_size, dtype=dtype, device=device)
    prefill_position_ids = torch.arange(0, sequence_length, dtype=torch.long, device=device)
    
    # 2. 构建预填充阶段的位置编码
    dummy_x = torch.randn(1, sequence_length, head_dim, device=device, dtype=dtype)
    cos, sin = rotary_emb(dummy_x, position_ids=prefill_position_ids.unsqueeze(0))
    prefill_pos_emb = (cos.squeeze(0), sin.squeeze(0))

    try:
        log_dir_prefill = './logs/inference_prefill'
        print(f"预填充阶段的追踪文件将保存在: {log_dir_prefill}")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir_prefill),
            with_stack=True
        ) as prof_prefill:
            for i in range(4): # (1+1+2)*1 = 4
                # 在循环内重新初始化 KV 缓存，确保每次分析都是独立的
                past_key_values = NaiveCache(num_layers=num_layers)
                with torch.no_grad():
                    output_prefill, updated_past = attention_layer.forward_inference(
                        packed_query_sequence=prefill_sequence,
                        query_lens=torch.tensor([sequence_length], device=device),
                        packed_query_position_embeddings=prefill_pos_emb,
                        packed_query_indexes=torch.arange(0, sequence_length, device=device),
                        past_key_values=past_key_values,
                        key_values_lens=None,
                        packed_key_value_indexes=None,
                        mode="gen", 
                        packed_vae_token_indexes=packed_vae_token_indexes,
                        packed_text_indexes=packed_text_indexes
                    )
                prof_prefill.step()
                print(f"预填充 Profiler 步骤 {i+1}/4 完成...")

        print("预填充性能分析成功！")
        assert output_prefill.shape == (sequence_length, hidden_size), "预填充输出形状错误！"
        print(f"预填充输出形状正确: {output_prefill.shape}")
        assert updated_past.key_cache[0] is not None, "KV 缓存未被更新！"
        assert updated_past.key_cache[0].shape[0] == sequence_length, "KV 缓存中的序列长度不正确！"
        print(f"KV 缓存已成功更新，缓存中 K 张量的形状: {updated_past.key_cache[0].shape}")

    except Exception as e:
        print(f"\n--- 预填充性能分析期间发生错误 ---")
        traceback.print_exc()

    # ==============================================================================
    # 阶段 B: 解码 (Decoding) 性能分析
    # ==============================================================================
    print("\n--- 阶段 B: 解码 (Decoding) 性能分析 ---")

    print("正在为解码阶段准备 KV 缓存 (运行一次预填充)...")
    with torch.no_grad():
        initial_past_key_values = NaiveCache(num_layers=num_layers)
        _, updated_past_for_decode = attention_layer.forward_inference(
            packed_query_sequence=prefill_sequence,
            query_lens=torch.tensor([sequence_length], device=device),
            packed_query_position_embeddings=prefill_pos_emb,
            packed_query_indexes=torch.arange(0, sequence_length, device=device),
            past_key_values=initial_past_key_values,
            mode="gen", 
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes
        )
    print("KV 缓存已准备就绪。")

    # 1. 模拟一个新 token
    new_token_sequence = torch.randn(1, hidden_size, dtype=dtype, device=device)
    new_token_position_ids = torch.tensor([sequence_length], dtype=torch.long, device=device)

    # 2. 为新 token 构建位置编码
    dummy_x_new = torch.randn(1, 1, head_dim, device=device, dtype=dtype)
    cos_new, sin_new = rotary_emb(dummy_x_new, position_ids=new_token_position_ids.unsqueeze(0))
    new_token_pos_emb = (cos_new.squeeze(0), sin_new.squeeze(0))
    
    try:
        log_dir_decode = './logs/inference_decode'
        print(f"解码阶段的追踪文件将保存在: {log_dir_decode}")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir_decode),
            with_stack=True
        ) as prof_decode:
            for i in range(10): # (2+2+6)*1 = 10
                with torch.no_grad():
                    past_key_values_copy = NaiveCache(num_layers)
                    past_key_values_copy.key_cache[0] = updated_past_for_decode.key_cache[0].clone()
                    past_key_values_copy.value_cache[0] = updated_past_for_decode.value_cache[0].clone()

                    output_decode, final_past = attention_layer.forward_inference(
                        packed_query_sequence=new_token_sequence,
                        query_lens=torch.tensor([1], device=device),
                        packed_query_position_embeddings=new_token_pos_emb,
                        packed_query_indexes=torch.tensor([sequence_length], device=device),
                        past_key_values=past_key_values_copy,
                        key_values_lens=torch.tensor([sequence_length], device=device),
                        packed_key_value_indexes=torch.arange(0, sequence_length, device=device),
                        mode="gen",
                        packed_vae_token_indexes=torch.tensor([], dtype=torch.long, device=device), 
                        packed_text_indexes=torch.tensor([0], device=device) # 假设新token是文本
                    )
                prof_decode.step()
                print(f"解码 Profiler 步骤 {i+1}/10 完成...")

        print("解码性能分析成功！")
        assert output_decode.shape == (1, hidden_size), "解码输出形状错误！"
        print(f"解码输出形状正确: {output_decode.shape}")
        final_cache_len = sequence_length + 1
        assert final_past.key_cache[0].shape[0] == final_cache_len, "最终缓存序列长度错误！"
        print(f"缓存已再次更新，最终缓存中 K 张量的形状: {final_past.key_cache[0].shape}")
        print("\n--- run_inference_test 通过! ---")

    except Exception as e:
        print(f"\n--- 解码性能分析失败 ---")
        traceback.print_exc()


if __name__ == "__main__":

    # 在调试时，设置此环境变量有助于定位异步 CUDA 操作中的错误
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # ==============================================================================
    # 修改点 3: 在这里定义手动开关并控制调用
    # ==============================================================================
    
    # --- 手动开关 ---
    # 设置为 True  -> 运行并计时你的自定义 CUDA Kernel
    # 设置为 False -> 运行并计时 PyTorch 的原生实现
    USE_CUSTOM_KERNEL = True

    if USE_CUSTOM_KERNEL:
        print("🚀 开关已打开，将运行并计时【自定义 CUDA Kernel】。")
    else:
        print("🐢 开关已关闭，将运行并计时【原生 PyTorch 实现】。")

    run_train_test_with_profiler(use_custom_kernel=USE_CUSTOM_KERNEL)
    run_inference_test_with_profiler(use_custom_kernel=USE_CUSTOM_KERNEL)
