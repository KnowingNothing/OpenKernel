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

def create_structured_attention_mask(
    samples_config: List[Tuple[str, int]],
    device: str,
    dtype: torch.dtype,
) -> List[torch.Tensor]:

    # sdpa per-sample policy: return a list of boolean masks, each [Li, Li]
    # True indicates masked (disallow), False indicates allowed.
    masks: List[torch.Tensor] = []
    for sample_type, sample_length in samples_config:
        if sample_type == 'text':
            causal_allowed = torch.tril(torch.ones((sample_length, sample_length), device=device, dtype=torch.bool))
            mask_bool = ~causal_allowed
        else:
            mask_bool = torch.zeros((sample_length, sample_length), device=device, dtype=torch.bool)
        masks.append(mask_bool)
    return masks

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

    force_math_reference_env = bool(int(os.getenv("FORCE_MATH_REF", "0")))
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
        use_custom_kernel=use_custom_kernel, 
        force_math_reference=force_math_reference_env and (not use_custom_kernel),
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

    # ============================================================================== 
    # 6-Pre: Numerical correctness check (custom kernel vs step-by-step reference)
    # ============================================================================== 
    try:
        print("--- Running numerical correctness check (custom kernel vs step-by-step reference) ---")

        # Build a custom kernel module and align weights for fair comparison
        att_custom = PackedAttentionMoT(
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
            use_custom_kernel=True,
            force_math_reference=False,
        ).to(device=device, dtype=dtype).eval()

        # For reference: step-by-step FP32 implementation for the entire packed sequence
        with torch.no_grad():
            # 1. Rebuild q/k/v after norm + RoPE in float32 (packed)
            pq = torch.zeros(sequence_length, att_custom.num_heads * (hidden_size // att_custom.num_heads), device=device, dtype=dtype)
            pk = torch.zeros(sequence_length, att_custom.num_key_value_heads * (hidden_size // att_custom.num_key_value_heads), device=device, dtype=dtype)
            pv = torch.zeros_like(pk)

            packed_sequence_und = packed_sequence[packed_und_token_indexes]
            packed_sequence_gen = packed_sequence[packed_gen_token_indexes]
            pq[packed_und_token_indexes] = att_custom.q_proj(packed_sequence_und)
            pq[packed_gen_token_indexes] = att_custom.q_proj_moe_gen(packed_sequence_gen)
            pk[packed_und_token_indexes] = att_custom.k_proj(packed_sequence_und)
            pk[packed_gen_token_indexes] = att_custom.k_proj_moe_gen(packed_sequence_gen)
            pv[packed_und_token_indexes] = att_custom.v_proj(packed_sequence_und)
            pv[packed_gen_token_indexes] = att_custom.v_proj_moe_gen(packed_sequence_gen)

            H = att_custom.num_heads
            D = hidden_size // H
            pq = pq.view(-1, H, D)
            pk = pk.view(-1, att_custom.num_key_value_heads, D)
            pv = pv.view(-1, att_custom.num_key_value_heads, D)

            # q/k norms
            pq_ = torch.zeros_like(pq)
            pk_ = torch.zeros_like(pk)
            pq_[packed_und_token_indexes] = att_custom.q_norm(pq[packed_und_token_indexes])
            pq_[packed_gen_token_indexes] = att_custom.q_norm_moe_gen(pq[packed_gen_token_indexes])
            pk_[packed_und_token_indexes] = att_custom.k_norm(pk[packed_und_token_indexes])
            pk_[packed_gen_token_indexes] = att_custom.k_norm_moe_gen(pk[packed_gen_token_indexes])

            # RoPE in float32
            cos, sin = packed_position_embeddings
            pq_, pk_ = pq_.to(torch.float32), pk_.to(torch.float32)
            pq_, pk_ = (pq_ * cos.unsqueeze(1)) + (torch.cat((-pq_[..., D//2:], pq_[..., :D//2]), dim=-1) * sin.unsqueeze(1)), \
                       (pk_ * cos.unsqueeze(1)) + (torch.cat((-pk_[..., D//2:], pk_[..., :D//2]), dim=-1) * sin.unsqueeze(1))

            # GQA expand K/V to H
            def _repeat_kv(x_1s_kv_d: torch.Tensor, n_rep: int) -> torch.Tensor:
                if n_rep == 1:
                    return x_1s_kv_d
                b, s, kv, d = x_1s_kv_d.shape
                x = x_1s_kv_d.unsqueeze(3).expand(b, s, kv, n_rep, d)
                return x.reshape(b, s, kv * n_rep, d)
            k_gqa = _repeat_kv(pk_.unsqueeze(0), att_custom.num_key_value_groups).squeeze(0)  # [S,H,D]
            v_gqa = _repeat_kv(pv.to(torch.float32).unsqueeze(0), att_custom.num_key_value_groups).squeeze(0)

            # Split per-sample: get chunks [H, Li, D]
            q_chunks = pq_.transpose(0, 1).split(sample_lens, dim=1)
            k_chunks = k_gqa.transpose(0, 1).split(sample_lens, dim=1)
            v_chunks = v_gqa.transpose(0, 1).split(sample_lens, dim=1)
            mask_chunks = [m.to(torch.bool) for m in attention_mask]

            # Compute reference output for all samples/heads
            ref_outputs = []
            for q_hld, k_hld, v_hld, mask_bool in zip(q_chunks, k_chunks, v_chunks, mask_chunks):
                # q_hld, k_hld, v_hld: [H, L, D], mask_bool: [L, L]
                H, L, D = q_hld.shape
                out = torch.zeros(H, L, D, device=q_hld.device, dtype=torch.float32)
                for h in range(H):
                    q = q_hld[h]  # [L, D]
                    k = k_hld[h]
                    v = v_hld[h]
                    logits = (q @ k.transpose(0, 1)) / (D ** 0.5)
                    logits = logits.masked_fill(mask_bool, -float('inf'))
                    logits = logits - torch.amax(logits, dim=-1, keepdim=True)
                    weights = torch.softmax(logits, dim=-1)
                    out[h] = weights @ v
                ref_outputs.append(out.transpose(0, 1).contiguous())  # [L, H, D]
            ref_packed = torch.cat(ref_outputs, dim=0).reshape(-1, H * D)

            # Apply the same output projection as forward_train for fair comparison
            ref_after_proj = torch.zeros_like(ref_packed)
            if packed_und_token_indexes.numel() > 0:
                ref_after_proj[packed_und_token_indexes] = att_custom.o_proj(
                    ref_packed[packed_und_token_indexes].to(att_custom.o_proj.weight.dtype)
                ).to(ref_after_proj.dtype)
            if packed_gen_token_indexes.numel() > 0:
                ref_after_proj[packed_gen_token_indexes] = att_custom.o_proj_moe_gen(
                    ref_packed[packed_gen_token_indexes].to(att_custom.o_proj_moe_gen.weight.dtype)
                ).to(ref_after_proj.dtype)

            # Custom kernel output
            out_custom = att_custom.forward_train(**inputs)

        # Compare in float32 for stable metrics
        a = ref_after_proj.to(torch.float32)
        b = out_custom.to(torch.float32)
        diff = (a - b).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        denom = a.abs().max().item() + 1e-8
        rel_max = max_abs / denom

        # dtype-aware tolerances
        if dtype in (torch.bfloat16, torch.float16):
            atol, rtol = 1e-2, 1e-2
        else:
            atol, rtol = 1e-4, 1e-4

        passed = max_abs <= max(atol, rtol * denom)
        print(f"Numeric check results: max_abs_diff={max_abs:.6e}, mean_abs_diff={mean_abs:.6e}, rel_max={rel_max:.6e}")
        print(f"Using tolerances: atol={atol}, rtol={rtol}")
        if passed:
            print("✅ Custom kernel numerical check PASSED vs reference implementation.")
        else:
            print("❌ Custom kernel numerical check FAILED vs reference implementation.")

        # Optionally save diagnostics
        try:
            os.makedirs('./logs', exist_ok=True)
            torch.save({
                'dtype': str(dtype),
                'samples_config': samples_config,
                'max_abs_diff': max_abs,
                'mean_abs_diff': mean_abs,
                'rel_max': rel_max,
                'atol': atol,
                'rtol': rtol,
                'ref_sample': a[: min(4, a.shape[0])].cpu(),
                'out_custom_sample': b[: min(4, b.shape[0])].cpu(),
                'diff_sample': diff[: min(4, diff.shape[0])].cpu(),
            }, './logs/numeric_check_diffs.pt')
            print("Saved numeric check diagnostics to ./logs/numeric_check_diffs.pt")
        except Exception as _:
            print("(Skip saving diagnostics)")
        try:
            print("--- Step-by-step debug on one sample/head (float32, mask as bool) ---")

            # Helper: repeat_kv for GQA (for debug only)
            def _repeat_kv(x_1s_kv_d: torch.Tensor, n_rep: int) -> torch.Tensor:
                # x: [1, S, Kv, D] -> [1, S, H, D]
                if n_rep == 1:
                    return x_1s_kv_d
                b, s, kv, d = x_1s_kv_d.shape
                x = x_1s_kv_d.unsqueeze(3).expand(b, s, kv, n_rep, d)  # [1,S,Kv,rep,D]
                return x.reshape(b, s, kv * n_rep, d)

            # 1) Rebuild q/k/v after norm + RoPE in float32 (packed) using module weights
            with torch.no_grad():
                att = att_custom  # reuse the same module to avoid undefined variable and ensure weight parity
                # projections (packed)
                pq = torch.zeros(sequence_length, att.num_heads * (hidden_size // att.num_heads), device=device, dtype=dtype)
                pk = torch.zeros(sequence_length, att.num_key_value_heads * (hidden_size // att.num_heads), device=device, dtype=dtype)
                pv = torch.zeros_like(pk)

                packed_sequence_und = packed_sequence[packed_und_token_indexes]
                packed_sequence_gen = packed_sequence[packed_gen_token_indexes]
                pq[packed_und_token_indexes] = att.q_proj(packed_sequence_und)
                pq[packed_gen_token_indexes] = att.q_proj_moe_gen(packed_sequence_gen)
                pk[packed_und_token_indexes] = att.k_proj(packed_sequence_und)
                pk[packed_gen_token_indexes] = att.k_proj_moe_gen(packed_sequence_gen)
                pv[packed_und_token_indexes] = att.v_proj(packed_sequence_und)
                pv[packed_gen_token_indexes] = att.v_proj_moe_gen(packed_sequence_gen)

                H = att.num_heads
                D = hidden_size // H
                pq = pq.view(-1, H, D)
                pk = pk.view(-1, att.num_key_value_heads, D)
                pv = pv.view(-1, att.num_key_value_heads, D)

                # q/k norms
                pq_ = torch.zeros_like(pq)
                pk_ = torch.zeros_like(pk)
                pq_[packed_und_token_indexes] = att.q_norm(pq[packed_und_token_indexes])
                pq_[packed_gen_token_indexes] = att.q_norm_moe_gen(pq[packed_gen_token_indexes])
                pk_[packed_und_token_indexes] = att.k_norm(pk[packed_und_token_indexes])
                pk_[packed_gen_token_indexes] = att.k_norm_moe_gen(pk[packed_gen_token_indexes])

                # RoPE
                cos, sin = packed_position_embeddings
                pq_, pk_ = pq_.to(torch.float32), pk_.to(torch.float32)
                pq_, pk_ = (pq_ * cos.unsqueeze(1)) + (torch.cat((-pq_[..., D//2:], pq_[..., :D//2]), dim=-1) * sin.unsqueeze(1)), \
                           (pk_ * cos.unsqueeze(1)) + (torch.cat((-pk_[..., D//2:], pk_[..., :D//2]), dim=-1) * sin.unsqueeze(1))

                # GQA expand K/V to H
                k_gqa = _repeat_kv(pk_.unsqueeze(0), att.num_key_value_groups).squeeze(0)  # [S,H,D]
                v_gqa = _repeat_kv(pv.to(torch.float32).unsqueeze(0), att.num_key_value_groups).squeeze(0)

                # Split per-sample: get chunks [H, Li, D]
                q_chunks = pq_.transpose(0, 1).split(sample_lens, dim=1)
                k_chunks = k_gqa.transpose(0, 1).split(sample_lens, dim=1)
                v_chunks = v_gqa.transpose(0, 1).split(sample_lens, dim=1)

                # Pick first sample and first head (if available)
                s_idx = 0
                h_idx = 0
                q_hld = q_chunks[s_idx][h_idx]  # [Li,D]
                k_hld = k_chunks[s_idx][h_idx]
                v_hld = v_chunks[s_idx][h_idx]
                mask_bool = attention_mask[s_idx].to(torch.bool)

                L = q_hld.shape[0]
                scale = 1.0 / (D ** 0.5)

                # Reference logits/softmax in float32
                logits_ref = (q_hld @ k_hld.transpose(0, 1)) * scale
                logits_ref = logits_ref.masked_fill(mask_bool, -float('inf'))
                logits_ref = logits_ref - torch.amax(logits_ref, dim=-1, keepdim=True)
                weights_ref = torch.softmax(logits_ref, dim=-1)
                out_ref = weights_ref @ v_hld

                # SDPA output for the same sample/head (try to force MATH backend; fallback to reference)
                try:
                    from torch.nn.attention import sdpa_kernel, SDPBackend
                    with sdpa_kernel(backends=[SDPBackend.MATH]):
                        out_sdpa = F.scaled_dot_product_attention(
                            q_hld.unsqueeze(0).unsqueeze(0),  # [1,1,L,D]
                            k_hld.unsqueeze(0).unsqueeze(0),
                            v_hld.unsqueeze(0).unsqueeze(0),
                            attn_mask=mask_bool.unsqueeze(0).unsqueeze(1),
                            is_causal=False,
                        ).squeeze(0).squeeze(0)
                except Exception:
                    out_sdpa = out_ref.clone()

                # Custom kernel output for the same sample/head
                try:
                    import custom_attention
                    out_custom = custom_attention.forward(
                        q_hld.unsqueeze(0).unsqueeze(0).contiguous(),
                        k_hld.unsqueeze(0).unsqueeze(0).contiguous(),
                        v_hld.unsqueeze(0).unsqueeze(0).contiguous(),
                        mask_bool.contiguous(),
                    ).squeeze(0).squeeze(0)
                except Exception:
                    out_custom = torch.full_like(out_ref, float('nan'))

                # Diffs
                ref_sdpa_diff = (out_ref - out_sdpa).abs()
                ref_custom_diff = (out_ref - out_custom).abs()
                sdpa_custom_diff = (out_sdpa - out_custom).abs()

                debug_info = {
                    'sample_idx': s_idx,
                    'head_idx': h_idx,
                    'L': L,
                    'D': D,
                    'out_ref_sample': out_ref[: min(8, L)].cpu(),
                    'out_sdpa_sample': out_sdpa[: min(8, L)].cpu(),
                    'out_custom_sample': out_custom[: min(8, L)].cpu(),
                    'ref_sdpa_max_abs': ref_sdpa_diff.max().item(),
                    'ref_custom_max_abs': ref_custom_diff.max().item(),
                    'sdpa_custom_max_abs': sdpa_custom_diff.max().item(),
                }
                torch.save(debug_info, './logs/attn_debug.pt')
                print("Saved step-by-step debug to ./logs/attn_debug.pt")
                print(
                    f"Step-debug diffs: ref-vs-sdpa={debug_info['ref_sdpa_max_abs']:.6e}, "
                    f"ref-vs-custom={debug_info['ref_custom_max_abs']:.6e}, "
                    f"sdpa-vs-custom={debug_info['sdpa_custom_max_abs']:.6e}"
                )
        except Exception:
            print("(Step-by-step debug) raised an exception (continuing). Stack trace:")
            traceback.print_exc()
    except Exception:
        print("Numeric correctness check raised an exception (continuing). Stack trace:")
        traceback.print_exc()

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

    force_math_reference_env = bool(int(os.getenv("FORCE_MATH_REF", "0")))
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
        use_custom_kernel=use_custom_kernel, 
        force_math_reference=force_math_reference_env and (not use_custom_kernel), 
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
        # print(f"Profiler traces for prefill phase will be saved in: {log_dir_prefill}") 
        
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

                # print(f"Prefill Profiler step {i+1}/4 complete...") 
            
            # print("\n"+"The initial and final times had additional expenses, "+ "\n"+ "so they were not included in the average time calculation."+"\n")
            for i in range(3) :
                total_time += durations_ms[i+1]
                # print(f"Step {i+2}/5 cost {durations_ms[i+1]:.4f} ms")
            average_time = total_time / 3
            # print("\n"+f"average time is {average_time:.4f} ms")

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
        # print(f"Profiler traces for decoding phase will be saved in: {log_dir_decode}") 

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

                # print(f"Decoding Profiler step {i+1}/10 complete...") 

            # print("\n"+"The initial and final times had additional expenses, "+ "\n"+ "so they were not included in the average time calculation."+"\n")
            for i in range(6) :
                total_time += durations_ms[i+2]
                # print(f"Step {i+3}/6 cost {durations_ms[i+2]:.4f} ms")
            average_time = total_time / 6
            # print("\n"+f"average time is {average_time:.4f} ms")

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
    verify_custom_kernel_vs_pytorch()