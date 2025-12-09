# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.profiler
import traceback
import os

# 确保 bagel_mot_attention.py 文件在同一个目录下
try:
    from bagel_mot_attention import PackedAttentionMoT, Qwen2RotaryEmbedding, NaiveCache
    from torch.nn.attention.flex_attention import create_block_mask
except ImportError:
    print("Error: Ensure bagel_mot_attention.py is in the same directory as this script.")
    exit()
##########################
# 创建掩码部分
##########################
def create_structured_attention_mask(num_text_tokens, sequence_length, device, dtype):
    """
    创建一个混合模式的注意力掩码。
    文本部分是因果的，其余部分是完全的。
    """
    # 1. 创建一个默认为 0.0 的掩码张量
    attention_mask_tensor = torch.zeros(
        (sequence_length, sequence_length),
        device=device,
        dtype=dtype
    )

    # 2. 在文本-到-文本区域应用因果掩码
    if num_text_tokens > 0:
        text_causal_mask = torch.triu(
            torch.ones((num_text_tokens, num_text_tokens), device=device, dtype=torch.bool),
            diagonal=1
        )
        attention_mask_tensor[:num_text_tokens, :num_text_tokens][text_causal_mask] = -float('inf')

    # 3. 将最终的掩码张量放入列表中
    return [attention_mask_tensor]

def run_train_test_with_profiler():
    """
    测试训练模式 (forward_train) 并使用 profiler 进行性能分析。
    """
    print("\n" + "="*80)
    print("--- forward_train begin ---")
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

    if not torch.cuda.is_available():
        print("Error: CUDA device not detected. The program will run on the CPU, which may result in slower performance.")

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
        layer_idx=0
    ).to(device=device, dtype=dtype).train() # 确保是训练模式

    rotary_emb = Qwen2RotaryEmbedding(
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta
    ).to(device=device, dtype=dtype)
    
    print(f"Step 0: Environment setup is complete, and modules are loaded on {device}.")

    # ==============================================================================
    # 步骤 1: 模拟输入数据并构建基础索引
    # ==============================================================================
    prompt_ids = [151644, 8948, 198, 2610, 6435, 25]
    image_height, image_width = 1024, 1024
    vae_downsample_ratio = 8
    latent_patch_size = 2

    num_text_tokens = len(prompt_ids)
    num_vae_h = (image_height // vae_downsample_ratio) // latent_patch_size
    num_vae_w = (image_width // vae_downsample_ratio) // latent_patch_size
    num_vae_tokens = num_vae_h * num_vae_w

    sequence_length = num_text_tokens + num_vae_tokens

    print("\nStep 0: Begin to verify the output.")
    
    sample_lens = [sequence_length]
    print(f"Step 1: Simulation data calculation is complete. Text Token count: {num_text_tokens}, VAE Token count: {num_vae_tokens}, Total sequence length: {sequence_length}")

    packed_text_indexes = torch.arange(0, num_text_tokens, dtype=torch.long, device=device)
    packed_vae_token_indexes = torch.arange(num_text_tokens, sequence_length, dtype=torch.long, device=device)
    packed_position_ids = torch.zeros(sequence_length, dtype=torch.long, device=device)

    # ==============================================================================
    # 步骤 2: 构建核心输入张量 `packed_sequence`
    # ==============================================================================
    packed_sequence = torch.randn(
        sequence_length,
        hidden_size,
        dtype=dtype,
        device=device
    )
    print(f"Step 2: `packed_sequence` created successfully with shape: {packed_sequence.shape}")

    # ==============================================================================
    # 步骤 3: 构建位置编码 `packed_position_embeddings`
    # ==============================================================================
    dummy_x_for_rope = torch.randn(1, sequence_length, head_dim, device=device, dtype=dtype)
    packed_cos, packed_sin = rotary_emb(dummy_x_for_rope, position_ids=packed_position_ids.unsqueeze(0))
    packed_position_embeddings = (packed_cos.squeeze(0), packed_sin.squeeze(0))
    print(f"Step 3: `packed_position_embeddings` created successfully with shapes: (cos: {packed_position_embeddings[0].shape}, sin: {packed_position_embeddings[1].shape})")

    # # ==============================================================================
    # # 步骤 4: 构建注意力掩码 `attention_mask`
    # # ==============================================================================
 
    print("Step 4: Constructing attention mask `attention_mask`")
    attention_mask = create_structured_attention_mask(num_text_tokens, sequence_length, device, dtype)
    print(f"  Mask list created successfully, containing {len(attention_mask)} tensors, each with shape {attention_mask[0].shape}")
    # ...

    print(f"Step 4: `attention_mask` (mixed mode) created successfully.")
    print(f"  Mask list contains {len(attention_mask)} tensors, each with shape {attention_mask[0].shape}")

    # ==============================================================================
    # 步骤 5: 构建 MoE 路由索引
    # ==============================================================================
    packed_und_token_indexes = packed_text_indexes
    packed_gen_token_indexes = packed_vae_token_indexes
    print(f"Step 5: MoE routing indexes created successfully.")

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
    print("--- All input parameters are ready, about to call forward_train ---")
    print("="*50 + "\n")

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  Parameter: {k:<28} | Shape: {v.shape}")
        elif isinstance(v, tuple):
            print(f"  Parameter: {k:<28} | Shape: (cos: {v[0].shape}, sin: {v[1].shape})")
        else:
            print(f"  Parameter: {k:<28} | Value: {type(v)}")
    print("="*50 + "\n")

    try:
        log_dir_train = './logs/train'
        print(f"Training mode trace files will be saved at: {log_dir_train}")
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
                print(f"Training mode Profiler step {i+1}/5 completed...")

        print("\n--- forward_train call successful! ---")
        print(f"Training mode test passed! Trace files have been generated in the {log_dir_train} directory.")

        print(f"Output tensor shape: {output.shape}")
        
        expected_shape = (sequence_length, hidden_size)
        assert output.shape == expected_shape, f"Output shape error! Expected {expected_shape}, but got {output.shape}"
        print(f"Output shape is correct! Expected: {expected_shape}, Got: {output.shape}")

        # --- 添加验证 ---
        assert not torch.isnan(output).any(), "The model output contains NaN values! Training may be unstable."
        assert not torch.isinf(output).any(), "The model output contains Inf values! Training may be unstable."
        print("✅ The model output has passed the numerical stability check!")

        # Check and print output values
        print("\n--- Output Value Overview ---")
        # Convert output to float32 for more stable calculations
        output_float = output.to(torch.float32)
        print(f"  Output tensor sum (Sum): {output_float.sum().item():.4f}")
        print(f"  Output tensor mean (Mean): {output_float.mean().item():.4f}")
        print(f"  Output tensor std (Std): {output_float.std().item():.4f}")
        # Print top-left 3x5 slice
        print("  The top-left 3x5 slice of the output tensor:")
        print(output_float[:3, :5])

    except Exception as e:
        print(f"\n--- An error occurred while calling forward_train ---")
        traceback.print_exc()


def run_inference_test_with_profiler():
    """
    测试推理模式 (forward_inference) 并为预填充和解码阶段分别进行性能分析。
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
        layer_idx=0
    ).to(device=device, dtype=dtype).eval() # 确保是评估模式

    rotary_emb = Qwen2RotaryEmbedding(
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta
    ).to(device=device, dtype=dtype)

    print(f"Step 0: The inference mode environment setup is complete.")

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
    print("\n--- Phase A: Analysis of Prefill ---")
    
    # 1. 构建预填充阶段的输入张量
    prefill_sequence = torch.randn(sequence_length, hidden_size, dtype=dtype, device=device)
    prefill_position_ids = torch.arange(0, sequence_length, dtype=torch.long, device=device)
    
    # 2. 构建预填充阶段的位置编码
    dummy_x = torch.randn(1, sequence_length, head_dim, device=device, dtype=dtype)
    cos, sin = rotary_emb(dummy_x, position_ids=prefill_position_ids.unsqueeze(0))
    prefill_pos_emb = (cos.squeeze(0), sin.squeeze(0))

    try:
        log_dir_prefill = './logs/inference_prefill'
        print(f"Trace files for the prefill phase will be saved at: {log_dir_prefill}")
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
                print(f"Prefill Profiler step {i+1}/4 complete...")

        # 验证部分被移到 profiler 循环之外，只验证最后一次的结果
        print("Prefill performance analysis successful!")
        assert output_prefill.shape == (sequence_length, hidden_size), "Prefill output shape error!"
        print(f"Prefill output shape is correct: {output_prefill.shape}")
        assert updated_past.key_cache[0] is not None, "Cache has not been updated!"
        assert updated_past.key_cache[0].shape[0] == sequence_length, "Sequence length in cache is incorrect!"
        print(f"Cache has been successfully updated, shape of K tensor in cache: {updated_past.key_cache[0].shape}")

    except Exception as e:
        print(f"\n--- An error occurred during prefill performance analysis ---")
        traceback.print_exc()

    # ==============================================================================
    # Phase B: Analysis of Decoding Performance
    # ==============================================================================
    print("\n--- Phase B: Analysis of Decoding ---")

    # Fix 1: Run prefill once outside of profiler to prepare a valid KV cache for decoding
    print("Preparing KV cache for decoding (running prefill once)...")
    with torch.no_grad():
        # 使用上面准备好的预填充输入
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
    print("KV cache is ready.")

    # 1. 模拟一个新 token
    new_token_sequence = torch.randn(1, hidden_size, dtype=dtype, device=device)
    new_token_position_ids = torch.tensor([sequence_length], dtype=torch.long, device=device)

    # 2. 为新 token 构建位置编码
    dummy_x_new = torch.randn(1, 1, head_dim, device=device, dtype=dtype)
    cos_new, sin_new = rotary_emb(dummy_x_new, position_ids=new_token_position_ids.unsqueeze(0))
    new_token_pos_emb = (cos_new.squeeze(0), sin_new.squeeze(0))
    
    try:
        log_dir_decode = './logs/inference_decode'
        print(f"Trace files for the decoding phase will be saved at: {log_dir_decode}")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir_decode),
            with_stack=True
        ) as prof_decode:
            for i in range(10): # (2+2+6)*1 = 10
                with torch.no_grad():
                    # 每次循环都使用准备好的 updated_past_for_decode 的一个副本
                    # 这是为了防止上一次循环修改了缓存的状态
                    past_key_values_copy = NaiveCache(num_layers)
                    past_key_values_copy.key_cache[0] = updated_past_for_decode.key_cache[0].clone()
                    past_key_values_copy.value_cache[0] = updated_past_for_decode.value_cache[0].clone()

                    output_decode, final_past = attention_layer.forward_inference(
                        packed_query_sequence=new_token_sequence,
                        query_lens=torch.tensor([1], device=device),
                        packed_query_position_embeddings=new_token_pos_emb,
                        packed_query_indexes=torch.tensor([sequence_length], device=device),
                        past_key_values=past_key_values_copy, # 使用副本
                        key_values_lens=torch.tensor([sequence_length], device=device),
                        packed_key_value_indexes=torch.arange(0, sequence_length, device=device),
                        mode="gen",
                        packed_vae_token_indexes=torch.tensor([0], device=device), 
                        packed_text_indexes=torch.tensor([], dtype=torch.long, device=device)
                    )
                prof_decode.step()
                print(f"Decoding Profiler step {i+1}/10 complete...")

        # 验证部分被移到 profiler 循环之外，只验证最后一次的结果
        print("Decoding performance analysis succeeded!")
        assert output_decode.shape == (1, hidden_size), "Decoding output shape error!"
        print(f"Decoding output shape is correct: {output_decode.shape}")
        final_cache_len = sequence_length + 1
        assert final_past.key_cache[0].shape[0] == final_cache_len, "Final cache sequence length error!"
        print(f"Cache has been updated again, shape of K tensor in final cache: {final_past.key_cache[0].shape}")
        print("\n--- run_inference_test passed! ---")
        


    except Exception as e:
        print(f"\n--- Decoding performance analysis failed ---")
        traceback.print_exc()


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    run_train_test_with_profiler()
    run_inference_test_with_profiler()

