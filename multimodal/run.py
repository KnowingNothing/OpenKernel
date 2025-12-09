import torch 
import torch.profiler 
import traceback 
import os 
import random 
from typing import List, Tuple 
import time 
import torch.nn.functional as F

# 参数不与注释匹配，具体测例决赛结束前公布
FIXED_SHAPES = {
    1001: (1, 5, 2651, 56),   # 极短指令/对话
    1002: (1, 5, 3366, 64),   # 标准 LLaVA 单图问答
    1003: (2, 13, 2872, 44),  # 1k 上下文边界测试
    1004: (2, 14, 4096, 64),  # 多图对比分析。
    1005: (2, 10, 4096, 52),  # 并发多图复杂任务
    1006: (2, 7, 4096, 64),   # 高清切图 OCR
    1007: (2, 4, 3594, 40),   # 短视频理解  
    1008: (1, 6, 4096, 41),   # 长文档/长视频
    1009: (1, 11, 4096, 64),  # 投机采样验证/高吞吐
    1010: (2, 14, 1808, 64),  # 8k 上下文溢出鲁棒性测试
}


from bagel_mot_attention import PackedAttentionMoT, Qwen2RotaryEmbedding, NaiveCache 


def prepare_multimodal_inputs(
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=32,
    max_position_embeddings=8192,
    rope_theta=10000.0,
    image_height=1024,
    image_width=1024,
    vae_downsample_ratio=8,
    latent_patch_size=2,
    device: torch.device = None,
    dtype: torch.dtype = None,
    forced_num_text_tokens: int | None = None,
    forced_num_vae_tokens: int | None = None,
):
    """Prepare packed multimodal sequence: text + VAE image tokens.

    This simulates realistic distributions for text embeddings (small-variance
    token embeddings with deterministic token-specific offsets) and VAE
    latents (spatially correlated feature map flattened to tokens).

    Environment variables (optional):
      - MM_NUM_TEXT: number of text tokens (default 6)
      - MM_TEXT_STD: std for text embedding noise (default 0.02)
      - MM_VAE_STD: std for VAE latent noise (default 0.06)
      - MM_VAE_BLUR: blur kernel size for spatial smoothing (default 3)
      - MM_VAE_CHANNELS: number of latent channels before projection (default = hidden_size)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    head_dim = hidden_size // num_attention_heads

    # text tokens
    if forced_num_text_tokens is not None:
        num_text_tokens = int(forced_num_text_tokens)
    else:
        num_text_tokens = int(os.getenv('MM_NUM_TEXT', '6'))
    prompt_ids = torch.randint(0, 200000, (num_text_tokens,), device=device)

    text_std = float(os.getenv('MM_TEXT_STD', '0.02'))
    if num_text_tokens > 0:
        text_emb = torch.randn(num_text_tokens, hidden_size, device=device, dtype=torch.float32) * text_std
        # deterministic token-specific small offsets to mimic embedding table
        token_ids_f = prompt_ids.to(dtype=torch.float32).unsqueeze(1)
        pos_vec = torch.arange(hidden_size, device=device, dtype=torch.float32).unsqueeze(0)
        token_offset = (torch.sin(token_ids_f * 0.123456 + pos_vec * 0.00037) * 0.005).to(dtype=torch.float32)
        text_emb = text_emb + token_offset
        text_emb = text_emb.to(dtype=dtype)
    else:
        text_emb = torch.empty((0, hidden_size), device=device, dtype=dtype)

    # VAE latent tokens as spatially correlated map
    if forced_num_vae_tokens is not None:
        num_vae_tokens = int(forced_num_vae_tokens)
        num_vae_h = None
        num_vae_w = None
    else:
        num_vae_h = (image_height // vae_downsample_ratio) // latent_patch_size
        num_vae_w = (image_width // vae_downsample_ratio) // latent_patch_size
        num_vae_tokens = num_vae_h * num_vae_w

    vae_std = float(os.getenv('MM_VAE_STD', '0.06'))
    blur_k = int(os.getenv('MM_VAE_BLUR', '3'))
    if blur_k % 2 == 0:
        blur_k = max(3, blur_k - 1)

    vae_channels = int(os.getenv('MM_VAE_CHANNELS', str(hidden_size)))

    if num_vae_tokens > 0:
        if forced_num_vae_tokens is not None:
            packed_vae = torch.randn(num_vae_tokens, hidden_size, device=device, dtype=dtype) * vae_std
        else:
            vae_map = torch.randn(1, vae_channels, num_vae_h, num_vae_w, device=device, dtype=torch.float32) * vae_std
            try:
                k = blur_k
                kernel = torch.ones(1, 1, k, k, device=device, dtype=torch.float32) / (k * k)
                kernel = kernel.repeat(vae_channels, 1, 1, 1)
                vae_map = F.conv2d(vae_map, kernel, padding=k//2, groups=vae_channels)
            except Exception:
                pass
            if vae_channels != hidden_size:
                proj = torch.randn(vae_channels, hidden_size, device=device, dtype=torch.float32) * (1.0 / (vae_channels ** 0.5))
                flat = vae_map.squeeze(0).permute(1, 2, 0).reshape(num_vae_tokens, vae_channels)
                packed_vae = flat.matmul(proj)
                packed_vae = packed_vae.to(dtype=dtype)
            else:
                packed_vae = vae_map.squeeze(0).permute(1, 2, 0).reshape(num_vae_tokens, hidden_size).to(dtype=dtype)
    else:
        packed_vae = torch.empty((0, hidden_size), device=device, dtype=dtype)

    # Concatenate text then vae to form packed_sequence
    packed_sequence = torch.cat([text_emb, packed_vae], dim=0)

    # Indices
    packed_text_indexes = torch.arange(0, num_text_tokens, dtype=torch.long, device=device)
    packed_vae_token_indexes = torch.arange(num_text_tokens, num_text_tokens + num_vae_tokens, dtype=torch.long, device=device)

    # Position ids and embeddings (RoPE)
    rotary_emb = Qwen2RotaryEmbedding(head_dim=head_dim, max_position_embeddings=max_position_embeddings, rope_theta=rope_theta).to(device=device, dtype=dtype)
    packed_position_ids = torch.arange(0, packed_sequence.shape[0], dtype=torch.long, device=device)
    dummy = torch.randn(1, packed_sequence.shape[0], head_dim, device=device, dtype=dtype)
    packed_cos, packed_sin = rotary_emb(dummy, position_ids=packed_position_ids.unsqueeze(0))
    packed_position_embeddings = (packed_cos.squeeze(0), packed_sin.squeeze(0))

    sample_lens = [packed_sequence.shape[0]]

    # ========= build multimodal attention mask (start) =========
    # Attention mask: build modality-aware mask for text and VAE tokens.
    seq_len = packed_sequence.shape[0]
    # build additive mask in float32 (0 = allow, -inf = block)
    attn = torch.zeros((seq_len, seq_len), device=device, dtype=torch.float32)

    # read flags from env (strings to ints)
    def _env_flag(name, default):
        try:
            return int(os.getenv(name, str(int(default)))) != 0
        except Exception:
            return bool(default)

    text_causal = _env_flag('MM_TEXT_CAUSAL', True)
    text_sees_vae = _env_flag('MM_TEXT_SEES_VAE', True)
    vae_sees_text = _env_flag('MM_VAE_SEES_TEXT', False)
    vae_self_causal = _env_flag('MM_VAE_SELF_CASUAL', False)
    try:
        vae_local_window = int(os.getenv('MM_VAE_LOCAL_WINDOW', '0'))
    except Exception:
        vae_local_window = 0

    text_start = 0
    text_end = num_text_tokens
    vae_start = num_text_tokens
    vae_end = seq_len

    # 1) Text-to-text: causal (default) or full
    if num_text_tokens > 0:
        if text_causal:
            causal_upper = torch.triu(torch.ones((num_text_tokens, num_text_tokens), device=device, dtype=torch.bool), diagonal=1)
            attn[text_start:text_end, text_start:text_end][causal_upper] = -float('inf')

    # 2) Text <-> VAE cross-attention
    if num_vae_tokens > 0 and num_text_tokens > 0:
        # By default text can see VAE, VAE cannot see text. Use flags to override.
        if not text_sees_vae:
            # block columns corresponding to VAE for text rows
            attn[text_start:text_end, vae_start:vae_end] = -float('inf')
        if not vae_sees_text:
            # block columns corresponding to text for vae rows
            attn[vae_start:vae_end, text_start:text_end] = -float('inf')

    # 3) VAE self-attention: full, causal, or local window
    if num_vae_tokens > 0:
        if vae_self_causal:
            # causal among VAE tokens
            k = num_vae_tokens
            causal_upper = torch.triu(torch.ones((k, k), device=device, dtype=torch.bool), diagonal=1)
            attn[vae_start:vae_end, vae_start:vae_end][causal_upper] = -float('inf')
        elif vae_local_window > 0:
            # local window attention: for each token, block far-away columns
            for i in range(num_vae_tokens):
                left = max(0, i - vae_local_window)
                right = min(num_vae_tokens, i + vae_local_window + 1)
                if left > 0:
                    attn[vae_start + i, vae_start:vae_start + left] = -float('inf')
                if right < num_vae_tokens:
                    attn[vae_start + i, vae_start + right:vae_end] = -float('inf')

    # Note: attn is additive mask (0 = allow, -inf = block)
    # Convert additive mask to per-sample boolean masks expected by PackedAttentionMoT.forward_train
    # (List of tensors with shape (Li, Li), dtype=torch.bool, True == masked)
    mask_bool = (attn == float('-inf')).to(torch.bool)
    masks = []
    idx = 0
    for L in sample_lens:
        if L <= 0:
            masks.append(torch.empty((0, 0), dtype=torch.bool, device=device))
        else:
            masks.append(mask_bool[idx:idx+L, idx:idx+L].contiguous())
        idx += L
    # ========= build multimodal attention mask (end) =========

    return {
        'hidden_size': hidden_size,
        'num_attention_heads': num_attention_heads,
        'num_key_value_heads': num_key_value_heads,
        'max_position_embeddings': max_position_embeddings,
        'rope_theta': rope_theta,
        'device': device,
        'dtype': dtype,
        'head_dim': head_dim,
        'num_text_tokens': num_text_tokens,
        'num_vae_tokens': num_vae_tokens,
        'sequence_length': packed_sequence.shape[0],
        'packed_sequence': packed_sequence,
        'sample_lens': sample_lens,
        'attention_mask': masks,
        'packed_position_embeddings': packed_position_embeddings,
        'packed_text_indexes': packed_text_indexes,
        'packed_vae_token_indexes': packed_vae_token_indexes,
    }


def verify_custom_kernel_vs_pytorch():
    import custom_attention
    print("\n=== Custom Kernel vs PyTorch算子顺序实现 验证（支持多模态掩码） ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Input sampler: multimodal-only path
    def sample_inputs(seed=None):
        if seed is not None and seed in FIXED_SHAPES:
            batch_size, num_heads, seq_len, head_dim = FIXED_SHAPES[seed]
            hidden_size = num_heads * head_dim
            text_tokens = max(1, min(seq_len, seq_len // 4))
            vae_tokens = max(0, seq_len - text_tokens)
            params = prepare_multimodal_inputs(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                num_key_value_heads=num_heads,
                device=device,
                dtype=dtype,
                forced_num_text_tokens=text_tokens,
                forced_num_vae_tokens=vae_tokens,
            )
        else:
            params = prepare_multimodal_inputs(device=device, dtype=dtype)
            batch_size = 1
            num_heads = params['num_attention_heads']
            seq_len = params['sequence_length']
            head_dim = params['head_dim']
        q = (torch.rand(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype) - 0.5) * 0.04
        k = (torch.rand(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype) - 0.5) * 0.04
        v = (torch.rand(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype) - 0.5) * 0.04
        mask_bool = params['attention_mask'][0].to(torch.bool).contiguous()
        return q, k, v, mask_bool
    # Show the mode being used
    print("[verify] Using multimodal mask (prepare_multimodal_inputs)")

    # run verification on multiple independently sampled inputs
    VERIFY_RUNS = 5
    max_abs_list = []
    mean_abs_list = []
    max_rel_list = []
    allclose_list = []
    worst_shape = None

    # keep the tensors from the worst run for later detailed diagnostics
    worst_run_abs = -1.0
    worst_abs_diff = None
    worst_out_custom = None
    worst_out_ref = None

    worst_shape = None  # initialize worst_shape to avoid scope issues

    for run_i in range(VERIFY_RUNS):
        keys_list = list(FIXED_SHAPES.keys())
        seed_for_shape = keys_list[run_i % len(keys_list)]
        q, k, v, mask = sample_inputs(seed=seed_for_shape)
        
        # current tensor shapes
        batch_size = q.shape[0]
        num_heads = q.shape[1]
        seq_len = q.shape[2]
        head_dim = q.shape[3]
        scale = 1.0 / (head_dim ** 0.5)
        
        # Custom kernel output
        out_custom = custom_attention.forward(q, k, v, mask)
        
        # Reference implementation
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        attn_weights_once = torch.softmax(attn_scores, dim=-1)
        out_ref_once = torch.matmul(attn_weights_once, v)
        
        abs_diff_once = (out_custom - out_ref_once).abs()
        rel_diff_once = abs_diff_once / (out_ref_once.abs() + 1e-8)

        max_abs_val = float(abs_diff_once.max().item())
        mean_abs_val = float(abs_diff_once.mean().item())
        max_rel_val = float(rel_diff_once.max().item())
        
        # tiered tolerance for precision
        is_half = q.dtype in [torch.float16, torch.bfloat16]
        # half precision allows 5e-3 error; full precision uses 1e-5
        atol_val = 5e-3 if is_half else 1e-5
        rtol_val = 5e-3 if is_half else 1e-5
        tol_desc = f"atol={atol_val}, rtol={rtol_val}"
        
        is_allclose = bool(torch.allclose(out_custom, out_ref_once, atol=atol_val, rtol=rtol_val))

        max_abs_list.append(max_abs_val)
        mean_abs_list.append(mean_abs_val)
        max_rel_list.append(max_rel_val)
        allclose_list.append(is_allclose)

        # record the worst run so far
        if max_abs_val > worst_run_abs:
            worst_run_abs = max_abs_val
            # persist the shape when the worst run happens
            worst_shape = (batch_size, num_heads, seq_len, head_dim)
            try:
                worst_abs_diff = abs_diff_once.detach().clone()
                worst_out_custom = out_custom.detach().clone()
                worst_out_ref = out_ref_once.detach().clone()
            except Exception:
                worst_abs_diff = abs_diff_once
                worst_out_custom = out_custom
                worst_out_ref = out_ref_once
        
        # free per-run tensors to avoid accumulation
        try:
            del q, k, v, mask
        except Exception:
            pass

    # aggregated summary
    worst_max_abs = max(max_abs_list)
    avg_mean_abs = sum(mean_abs_list) / len(mean_abs_list)
    worst_max_rel = max(max_rel_list) if max_rel_list else float('nan')
    any_allclose = all(allclose_list)
    print(f"验证（{VERIFY_RUNS}次）汇总: worst_max_abs={worst_max_abs:.6e}, avg_mean_abs={avg_mean_abs:.6e}, worst_max_rel={worst_max_rel:.6e}, allclose_every_run={any_allclose}, {tol_desc}")

    # failure diagnostics
    max_abs_thresh = float(os.getenv('MM_MAX_ABS_THRESH', '1e-3'))
    mean_abs_thresh = float(os.getenv('MM_MEAN_ABS_THRESH', '1e-5'))
    
    # Note: if allclose passes but worst_abs is still high (within tolerance), this does not trigger
    if (not any_allclose):
        print('\n' + '='*12 + ' NUMERIC VALIDATION FAILED ' + '='*12)
        print('===  数值验证失败：custom_attention 与 PyTorch 参考不一致  ===')
        print(f'===  worst_max_abs={worst_max_abs:.6e}, avg_mean_abs={avg_mean_abs:.6e}, worst_max_rel={worst_max_rel:.6e}, allclose_every_run={any_allclose}, {tol_desc}  ===\n')
        
        try:
            if worst_abs_diff is None:
                flat = abs_diff_once.view(-1)
                ref_out_for_diag = out_ref
                custom_out_for_diag = out_custom
                # fallback: if worst_shape is missing, we ran once or logic failed
                B, H, S, D = batch_size, num_heads, seq_len, head_dim
            else:
                flat = worst_abs_diff.view(-1)
                ref_out_for_diag = worst_out_ref
                custom_out_for_diag = worst_out_custom
                # use recorded worst_shape to unravel indices
                if worst_shape is not None:
                    B, H, S, D = worst_shape
                else:
                    B, H, S, D = batch_size, num_heads, seq_len, head_dim
            
            topk = min(8, flat.numel())
            vals, idxs = torch.topk(flat, topk)
            print(f'  Top {topk} absolute diffs:')
            for rank, (val, idx) in enumerate(zip(vals, idxs), start=1):
                idx = int(idx.item())
                v = float(val.item())
                
                # unravel with correct dims (B, H, S, D)
                d = idx % D
                idx2 = idx // D
                s = idx2 % S
                idx3 = idx2 // S
                h = idx3 % H
                b = idx3 // H
                
                try:
                    custom_val = float(custom_out_for_diag[b, h, s, d].detach().cpu().item())
                    ref_val = float(ref_out_for_diag[b, h, s, d].detach().cpu().item())
                    rel = abs(custom_val - ref_val) / (abs(ref_val) + 1e-8)
                except Exception:
                    custom_val, ref_val, rel = None, None, None
                print(f'    #{rank}: abs={v:.6e} at (b={b},h={h},s={s},d={d})  custom={custom_val}, ref={ref_val}, rel={rel}')
        except Exception:
            print('  Failed to compute top-k diagnostics:', traceback.format_exc())
        raise RuntimeError('custom_attention numerical comparison FAILED')
    else:
        print('\n' + '='*12 + ' NUMERIC VALIDATION PASSED ' + '='*12)
        print('===  数值验证通过：custom_attention 与 PyTorch 参考一致  ===')
        print('===  worst_max_abs={:.6e}, avg_mean_abs={:.6e}, worst_max_rel={:.6e}, {}  ==='.format(worst_max_abs, avg_mean_abs, worst_max_rel, tol_desc)+'\n')
    
    # clean up resources
    try:
        del q, k, v, mask, out_custom, out_ref, abs_diff_once
        if 'worst_abs_diff' in locals(): del worst_abs_diff
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def seed_based_timing():
    """
    After verification, run performance timing using a fixed set of seeds:
    - set RNGs for each seed
    - generate scaled-down q/k/v (to avoid large magnitudes) and a multimodal mask
    - perform 2 warm-up runs, then 5 timed runs and take the average
    - on exception, skip the seed and record it as failed
    """
    import custom_attention
    # print("\n=== Seed-based timing (20 fixed seeds) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    seeds = list(FIXED_SHAPES.keys())

    # Maximum size caps (avoid huge tensors/OOM); overridable via environment variables
    max_B = int(os.getenv('ATTN_MAX_B', '2'))
    max_H = int(os.getenv('ATTN_MAX_H', '16'))
    # Choose a conservative default max_S based on GPU memory (>=24GB -> 1024 else 4096)
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
    # Safety factor: scale estimated memory to leave headroom (default 1.8)
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

    # Conservative required_bytes estimate (includes softmax temp weights/workspace)
    def required_bytes_for_run(B,H,S,D):
        base = estimate_bytes(B,H,S,D)
        # softmax may need an extra weights tensor (B*H*S*S) plus some workspace
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

            if s in FIXED_SHAPES:
                B, H, S, D = FIXED_SHAPES[s]
                hidden_size = H * D
                text_tokens = max(1, min(S, S // 4))
                vae_tokens = max(0, S - text_tokens)
                params = prepare_multimodal_inputs(
                    hidden_size=hidden_size,
                    num_attention_heads=H,
                    num_key_value_heads=H,
                    device=device,
                    dtype=dtype,
                    forced_num_text_tokens=text_tokens,
                    forced_num_vae_tokens=vae_tokens,
                )
            else:
                params = prepare_multimodal_inputs(device=device, dtype=dtype)
                B = 1
                H = params['num_attention_heads']
                S = params['sequence_length']
                D = params['head_dim']

            effective_budget = int(budget / max(1.0, safety_factor))
            # if required_bytes_for_run(B, H, S, D) > effective_budget:
            #     print(f"Seed {s}: multimodal shape {(B,H,S,D)} exceeds budget; skipping.")
            #     failed.append(s)
            #     continue

            q = (torch.rand(B, H, S, D, device=device, dtype=dtype) - 0.5) * 0.04
            k = (torch.rand(B, H, S, D, device=device, dtype=dtype) - 0.5) * 0.04
            v = (torch.rand(B, H, S, D, device=device, dtype=dtype) - 0.5) * 0.04
            mask = params['attention_mask'][0].to(torch.bool).contiguous()

            # warm-up twice
            for _ in range(2):
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
            # print(f"Seed {s}: shape (B={B}, H={H}, S={S}, D={D}), avg={avg_ms:.3f} ms, runs={[f'{t:.3f}' for t in times]}")
            results.append((s, avg_ms, times, (B, H, S, D)))
            # free per-seed large tensors to avoid accumulating GPU memory
            try:
                del q, k, v, mask
            except Exception:
                pass
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

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
                # Try to collect CUDA memory summary for offline analysis when available
                try:
                    if torch.cuda.is_available():
                        meta['cuda_reserved'] = torch.cuda.memory_reserved()
                        meta['cuda_allocated'] = torch.cuda.memory_allocated()
                        meta['cuda_max_allocated'] = torch.cuda.max_memory_allocated()
                        # detailed memory report (string)
                        meta['cuda_summary'] = torch.cuda.memory_summary()
                except Exception:
                    pass

                # Save as many tensors as possible (move to CPU); q/k/v may be undefined on failure
                save_dict = {'meta': meta}
                try:
                    save_dict['q'] = q.cpu()
                    save_dict['k'] = k.cpu()
                    save_dict['v'] = v.cpu()
                    save_dict['mask'] = mask.cpu()
                except Exception:
                    # If tensors were not allocated, still save meta
                    pass

                torch.save(save_dict, save_path)
                print(f"  Saved failed inputs and memory summary to: {save_path}")
            except Exception:
                print("  Failed to save inputs for seed", s)
                traceback.print_exc()
            failed.append(s)
            # Clean up potentially large tensors and clear cache
            try:
                del q, k, v, mask
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
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
    # run the seed-based timing after verification
    seed_based_timing()
