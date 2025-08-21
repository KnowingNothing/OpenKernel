# Copyright 2025 Bytedance Ltd. and/or its affiliates. 
# SPDX-License-Identifier: Apache-2.0 

import torch 
import torch.profiler 
import traceback 
import os 
import random 
from typing import List, Tuple 
import time 

# Ensure the bagel_mot_attention.py file is in the same directory
try: 
    from bagel_mot_attention import PackedAttentionMoT, Qwen2RotaryEmbedding, NaiveCache 
except ImportError: 
    print("Error: Please ensure bagel_mot_attention.py is in the same directory as this script.") 
    exit() 

def create_structured_attention_mask( 
    samples_config: List[Tuple[str, int]],  
    device: str,  
    dtype: torch.dtype 
) -> List[torch.Tensor]: 

    total_length = sum(length for _, length in samples_config) 
    
    attention_mask_tensor = torch.full( 
        (total_length, total_length),  
        -float('inf'),  
        device=device,  
        dtype=dtype 
    ) 

    current_offset = 0 
    for sample_type, sample_length in samples_config: 
        s = slice(current_offset, current_offset + sample_length) 
        
        if sample_type == 'text': 
            causal_mask = torch.tril( 
                torch.ones((sample_length, sample_length), device=device, dtype=torch.bool) 
            ) 
            attention_mask_tensor[s, s] = torch.where(causal_mask, 0.0, -float('inf')) 
        else: # image or noise 
            attention_mask_tensor[s, s] = 0.0 
        
        current_offset += sample_length 

    return [attention_mask_tensor] 

def run_train_test_with_profiler(use_custom_kernel: float = False):  
    
    print("\n" + "="*80) 
    print("--- forward_train (Dynamic Data) Start ---") 
    print("="*80) 

    # ============================================================================== 
    # 0: Environment setup and initialization
    # ============================================================================== 

    hidden_size = 2048
    num_attention_heads = 32 
    num_key_value_heads = 32 
    max_position_embeddings = 16384  
    rope_theta = 10000.0 
    rms_norm_eps = 1e-6 
    head_dim = hidden_size // num_attention_heads 
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32 

    if not torch.cuda.is_available(): 
        print("Warning: CUDA device not detected. The program will run on the CPU, which may be slow.") 

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
        use_custom_kernel=use_custom_kernel 
    ).to(device=device, dtype=dtype).train() 

    rotary_emb = Qwen2RotaryEmbedding( 
        head_dim=head_dim, 
        max_position_embeddings=max_position_embeddings, 
        rope_theta=rope_theta 
    ).to(device=device, dtype=dtype) 
    
    print(f"Step 0: Environment setup complete. Modules loaded to {device}.") 

    # ============================================================================== 
    # 1: Dynamically generate sample configuration and build indices
    # ============================================================================== 
    # Set seed for reproducibility
    # SEED = 32
    # SEED = 32
    # torch.manual_seed(SEED)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(SEED)
    
    num_samples = random.randint(1, 5) 
    sample_types = ['text', 'image', 'noise'] 
    samples_config = [] 
    for _ in range(num_samples): 
        stype = random.choice(sample_types) 
        if stype == 'text': 
            slen = random.randint(1, 1024) 
        else: # image or noise 
            slen = random.randint(1024, 2048) 
        samples_config.append((stype, slen)) 

    ## The test of datasize 
    # num_samples = random.randint(1, 5) 
    # sample_types = ['image', 'noise'] 
    # samples_config = [] 
    # for _ in range(num_samples): 
    #     stype = random.choice(sample_types) 
    #     slen = random.randint(1024, 2048) 
    #     samples_config.append((stype, slen)) 

    sequence_length = sum(length for _, length in samples_config) 
    sample_lens = [length for _, length in samples_config] 

    print("Step 1: Dynamic data configuration generated.") 
    print(f"  - Number of samples generated: {num_samples}") 
    print(f"  - Sample configuration (type, length): {samples_config}") 
    print(f"  - Total sequence length: {sequence_length}") 

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
    # 2: Build the core input tensor packed_sequence
    # ============================================================================== 
    packed_sequence = torch.randn( 
        sequence_length, 
        hidden_size, 
        dtype=dtype, 
        device=device 
    ) 
    print(f"Step 2: `packed_sequence` created successfully, shape: {packed_sequence.shape}") 

    # ============================================================================== 
    # 3: Build position embeddings packed_position_embeddings
    # ============================================================================== 
    dummy_x_for_rope = torch.randn(1, sequence_length, head_dim, device=device, dtype=dtype) 
    packed_cos, packed_sin = rotary_emb(dummy_x_for_rope, position_ids=packed_position_ids.unsqueeze(0)) 
    packed_position_embeddings = (packed_cos.squeeze(0), packed_sin.squeeze(0)) 
    print(f"Step 3: `packed_position_embeddings` created successfully, shape: (cos: {packed_position_embeddings[0].shape}, sin: {packed_position_embeddings[1].shape})") 

    # ============================================================================== 
    # 4: Build the attention mask attention_mask
    # ============================================================================== 
    print("Step 4: Building attention_mask...") 
    attention_mask = create_structured_attention_mask(samples_config, device, dtype) 
    print(f"  Mask list created successfully, containing {len(attention_mask)} tensors, each with shape {attention_mask[0].shape}") 

    # ============================================================================== 
    # 5: Create MoE routing indices (already created in Step 1)
    # ============================================================================== 
    print(f"Step 5: MoE routing indices created successfully.") 
    print(f"  - Number of und (text) tokens: {len(packed_und_token_indexes)}") 
    print(f"  - Number of gen (image/noise) tokens: {len(packed_gen_token_indexes)}") 


    # ============================================================================== 
    # 6: Integrate and call forward_train using the Profiler
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
    print("--- All input parameters are ready. Calling forward_train... ---") 
    print("="*50 + "\n") 

    for k, v in inputs.items(): 
        if isinstance(v, torch.Tensor): 
            print(f"  Parameter: {k:<28} | Shape: {v.shape}") 
        elif isinstance(v, tuple): 
            print(f"  Parameter: {k:<28} | Shape: (cos: {v[0].shape}, sin: {v[1].shape})") 
        elif isinstance(v, list) and all(isinstance(i, int) for i in v): 
             print(f"  Parameter: {k:<28} | Value: {v}") 
        elif isinstance(v, list) and all(isinstance(i, torch.Tensor) for i in v): 
             print(f"  Parameter: {k:<28} | Type: List[Tensor], Length: {len(v)}") 
        else: 
            print(f"  Parameter: {k:<28} | Type: {type(v)}") 
    print("="*50 + "\n") 

    try: 
        log_dir_train = './logs/train' 
        print(f"Profiler traces for training mode will be saved in: {log_dir_train}") 

        durations_ms = [0.0] * 5
        total_time = 0.0 

        with torch.profiler.profile( 
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], 
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1), 
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir_train), 
            record_shapes=True, profile_memory=True, with_stack=True 
        ) as prof: 
            for i in range(5): # (1+1+3)*1 = 5 
                torch.cuda.synchronize()

                start_time = time.perf_counter()

                with torch.autocast(device_type=device, dtype=dtype): 
                    output = attention_layer.forward_train(**inputs) 
                prof.step() 
                torch.cuda.synchronize()

                end_time = time.perf_counter()
                durations_ms[i] = (end_time - start_time) * 1000

                print(f"Training mode Profiler step {i+1}/5 complete...") 
                
            print("\n"+"The initial and final times had additional expenses, "+ "\n"+ "so they were not included in the average time calculation."+"\n")
            for i in range(3) :
                total_time += durations_ms[i+1]
                print(f"Step {i+2}/5 cost {durations_ms[i+1]:.4f} ms")
            average_time = total_time / 3
            print("\n"+f"average time is {average_time:.4f} ms")

        print("\n--- forward_train called successfully! ---") 
        print(f"Training mode test passed! Profiler traces generated in {log_dir_train} directory.") 

        print(f"Output tensor shape: {output.shape}") 
        
        # Verify if the output shape matches the expected shape
        expected_shape = (sequence_length, hidden_size) 
        assert output.shape == expected_shape, f"Output shape mismatch! Expected {expected_shape}, but got {output.shape}" 
        print(f"Output shape correct! Expected: {expected_shape}, Got: {output.shape}") 

        # --- Additional validations ---
        assert not torch.isnan(output).any(), "Model output contains NaN values! Training may be unstable." 
        assert not torch.isinf(output).any(), "Model output contains Inf values! Training may be unstable." 
        print("✅ Model output passed numerical stability check!") 

        # Check and print output values
        print("\n--- Output Value Overview ---") 
        output_float = output.to(torch.float32) 
        print(f"  - Output Tensor Sum: {output_float.sum().item():.4f}") 
        print(f"  - Output Tensor Mean: {output_float.mean().item():.4e}") 
        print(f"  - Output Tensor Std Dev: {output_float.std().item():.4f}") 

    except Exception as e: 
        print(f"\n--- An error occurred while calling forward_train ---") 
        traceback.print_exc() 

def run_inference_test_with_profiler(use_custom_kernel: float = False): 
    """ 
    Test inference mode (forward_inference) and profile the prefill and decoding stages separately.
    """ 
    print("\n" + "="*80) 
    print("--- forward_inference begin ---") 
    print("="*80) 

    # ============================================================================== 
    # 0: Environment setup and initialization
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
        use_custom_kernel=use_custom_kernel  
    ).to(device=device, dtype=dtype).eval()  

    rotary_emb = Qwen2RotaryEmbedding( 
        head_dim=head_dim, 
        max_position_embeddings=max_position_embeddings, 
        rope_theta=rope_theta 
    ).to(device=device, dtype=dtype) 

    print(f"Step 0: Inference mode environment setup complete.") 

    # ============================================================================== 
    # 1: Simulate input data for inference mode
    # ============================================================================== 
    num_text_tokens = 6 
    num_vae_tokens = 4096 
    sequence_length = num_text_tokens + num_vae_tokens 
    
    packed_text_indexes = torch.arange(0, num_text_tokens, dtype=torch.long, device=device) 
    packed_vae_token_indexes = torch.arange(num_text_tokens, sequence_length, dtype=torch.long, device=device) 
    
    # ============================================================================== 
    # Phase A: Analyze Prefill Performance
    # ============================================================================== 
    print("\n--- Phase A: Prefill Performance Analysis ---") 
    
    # 1. Build input tensors for the prefill stage
    prefill_sequence = torch.randn(sequence_length, hidden_size, dtype=dtype, device=device) 
    prefill_position_ids = torch.arange(0, sequence_length, dtype=torch.long, device=device) 
    
    # 2. Build position embeddings for the prefill stage
    dummy_x = torch.randn(1, sequence_length, head_dim, device=device, dtype=dtype) 
    cos, sin = rotary_emb(dummy_x, position_ids=prefill_position_ids.unsqueeze(0)) 
    prefill_pos_emb = (cos.squeeze(0), sin.squeeze(0)) 

    try: 
        log_dir_prefill = './logs/inference_prefill' 
        print(f"Profiler traces for prefill phase will be saved in: {log_dir_prefill}") 
        
        durations_ms = [0.0] * 5
        total_time = 0.0 

        with torch.profiler.profile( 
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], 
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1), 
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir_prefill), 
            with_stack=True 
        ) as prof_prefill: 
            for i in range(5): # (1+1+3)*1 = 5 
                torch.cuda.synchronize()
                # Reinitialize KV cache inside the loop to ensure each analysis is independent
                past_key_values = NaiveCache(num_layers=num_layers) 

                start_time = time.perf_counter()
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
                torch.cuda.synchronize()

                end_time = time.perf_counter()
                durations_ms[i] = (end_time - start_time) * 1000

                print(f"Prefill Profiler step {i+1}/4 complete...") 
            
            print("\n"+"The initial and final times had additional expenses, "+ "\n"+ "so they were not included in the average time calculation."+"\n")
            for i in range(3) :
                total_time += durations_ms[i+1]
                print(f"Step {i+2}/5 cost {durations_ms[i+1]:.4f} ms")
            average_time = total_time / 3
            print("\n"+f"average time is {average_time:.4f} ms")

        print("\n"+"Prefill performance analysis successful!") 
        assert output_prefill.shape == (sequence_length, hidden_size), "Prefill output shape mismatch!" 
        print(f"Prefill output shape correct: {output_prefill.shape}") 
        assert updated_past.key_cache[0] is not None, "KV cache was not updated!" 
        assert updated_past.key_cache[0].shape[0] == sequence_length, "Sequence length in KV cache is incorrect!" 
        print(f"KV cache updated successfully. Shape of K tensor in cache: {updated_past.key_cache[0].shape}") 

    except Exception as e: 
        print(f"\n--- An error occurred during prefill performance analysis ---") 
        traceback.print_exc() 

    # ============================================================================== 
    # Phase B: Analyze Decoding Performance
    # ============================================================================== 
    print("\n--- Phase B: Decoding Performance Analysis ---") 

    print("Preparing KV cache for decoding phase (running one prefill step)...") 
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
    print("KV cache is ready.") 

    # 1. Simulate a new token
    new_token_sequence = torch.randn(1, hidden_size, dtype=dtype, device=device) 
    new_token_position_ids = torch.tensor([sequence_length], dtype=torch.long, device=device) 

    # 2. Build position embeddings for the new token
    dummy_x_new = torch.randn(1, 1, head_dim, device=device, dtype=dtype) 
    cos_new, sin_new = rotary_emb(dummy_x_new, position_ids=new_token_position_ids.unsqueeze(0)) 
    new_token_pos_emb = (cos_new.squeeze(0), sin_new.squeeze(0)) 
    
    try: 
        log_dir_decode = './logs/inference_decode' 
        print(f"Profiler traces for decoding phase will be saved in: {log_dir_decode}") 

        durations_ms = [0.0] * 10
        total_time = 0.0 

        with torch.profiler.profile( 
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], 
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1), 
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir_decode), 
            with_stack=True 
        ) as prof_decode: 
            for i in range(10): # (2+2+6)*1 = 10 
                torch.cuda.synchronize()

                start_time = time.perf_counter()
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
                        packed_text_indexes=torch.tensor([0], device=device) # Assume the new token is text
                    ) 
                prof_decode.step() 
                torch.cuda.synchronize()

                end_time = time.perf_counter()
                durations_ms[i] = (end_time - start_time) * 1000

                print(f"Decoding Profiler step {i+1}/10 complete...") 

            print("\n"+"The initial and final times had additional expenses, "+ "\n"+ "so they were not included in the average time calculation."+"\n")
            for i in range(6) :
                total_time += durations_ms[i+2]
                print(f"Step {i+3}/6 cost {durations_ms[i+2]:.4f} ms")
            average_time = total_time / 6
            print("\n"+f"average time is {average_time:.4f} ms")

        print("\n"+"Decoding performance analysis successful!") 
        assert output_decode.shape == (1, hidden_size), "Decoding output shape mismatch!" 
        print(f"Decoding output shape correct: {output_decode.shape}") 
        final_cache_len = sequence_length + 1 
        assert final_past.key_cache[0].shape[0] == final_cache_len, "The final cache sequence length is incorrect!" 
        print(f"KV cache updated again. Shape of final K tensor in cache: {final_past.key_cache[0].shape}") 
        print("\n--- run_inference_test passed! ---") 

    except Exception as e: 
        print(f"\n--- An error occurred during the decoding performance analysis ---") 
        traceback.print_exc() 


if __name__ == "__main__": 
    # --- Manual Switch ---
    # Set to True -> Run and time your custom CUDA Kernel
    # Set to False -> Run and time the native PyTorch implementation
    USE_CUSTOM_KERNEL = True 

    if USE_CUSTOM_KERNEL: 
        print("🚀 Switch is ON. Running and timing [Custom CUDA Kernel].") 
    else: 
        print("🐢 Switch is OFF. Running and timing [Native PyTorch Implementation].") 

    run_train_test_with_profiler(use_custom_kernel=USE_CUSTOM_KERNEL) 
    run_inference_test_with_profiler(use_custom_kernel=USE_CUSTOM_KERNEL)