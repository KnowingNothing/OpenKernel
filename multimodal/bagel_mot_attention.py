from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention, or_masks, and_masks
from torch.nn.functional import scaled_dot_product_attention
from transformers.utils import ModelOutput
import torch.nn.functional as F


# Attempt to import your own package
try:
    import custom_attention
    IS_CUSTOM_ATTENTION_AVAILABLE = True
    print("✅ Custom attention kernel successfully imported.")
except ImportError:
    IS_CUSTOM_ATTENTION_AVAILABLE = False
    print("⚠️ Custom attention kernel not found. Falling back to native PyTorch implementation.")


# from flash_attn import flash_attn_varlen_func


import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)


if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
    
    
from transformers.modeling_rope_utils import rope_config_validation

import time
from functools import wraps
def timeit(func):
    """A decorator to measure the execution time of a function."""
    @wraps(func)  # @wraps ensures that the metadata of the decorated function (like name, docstring) is preserved
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()

        result = func(*args, **kwargs)

        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time)*1000
        print(f"function '{func.__name__}' costs {elapsed_time:.4f} ms.")

        return result
    return wrapper


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)



def pad_sequence(tensor, pad_size):
    H, L, D = tensor.shape
    pad_tensor = tensor.new_zeros((H, pad_size, D))
    return torch.cat([tensor, pad_tensor], dim=1)


def create_sparse_mask(document_lens, split_lens, attn_modes, device):
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def full_and_noise_mask(b, h, q_idx, kv_idx):
        return (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx]) & (full_and_noise_seq_id[q_idx] >= 0)

    def remove_noise_mask(b, h, q_idx, kv_idx):
        return (~((noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx])))

    def sample_mask(b, h, q_idx, kv_idx):
        return document_id[q_idx] == document_id[kv_idx]

    full_and_noise_tmp = []
    noise_tmp = []

    for i, (length, model) in enumerate(zip(split_lens, attn_modes)):
        value = i if model in ['full', 'noise'] else -1
        full_and_noise_tmp.extend([value] * length)
        value_noise = i if model == 'noise' else -1
        noise_tmp.extend([value_noise] * length)

    full_and_noise_seq_id = torch.Tensor(full_and_noise_tmp).to(device)
    noise_seq_id = torch.Tensor(noise_tmp).to(device)

    document_id = torch.cat([torch.full((l,), i) for i, l in enumerate(document_lens, start=1)]).to(device)

    return and_masks(or_masks(causal_mask, full_and_noise_mask), remove_noise_mask, sample_mask)


 
class NaiveCache:
    def __init__(self, num_layers):
        self.key_cache = {k: None for k in range(num_layers)}
        self.value_cache = {k: None for k in range(num_layers)}

    @property
    def num_layers(self):
        return len(self.key_cache)

    @property
    def seq_lens(self):
        if self.key_cache[0] is not None:
            return self.key_cache[0].shape[0]
        else:
            return 0
        
        
# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen2
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
    
    
    
def _compute_default_rope_parameters(
    rope_theta,
    head_dim,
    partial_rotary_factor = 1.0,
    device: Optional["torch.device"] = None,
) -> tuple["torch.Tensor", float]:
    base = rope_theta

    dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor
    
    
    
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim,
        max_position_embeddings=2048,
        rope_theta=10000,
        device=None,
    ):
        super().__init__()

        self.rope_theta = rope_theta
        self.head_dim = head_dim

        self.rope_type = "default"
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings

        inv_freq, self.attention_scaling = _compute_default_rope_parameters(
            rope_theta, head_dim, device=device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)



class Qwen2Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(
        self, 
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        max_position_embeddings,
        rope_theta,
        is_causal,
        attention_dropout,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.is_causal = is_causal
        self.attention_dropout = attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    (batch, num_key_value_heads, seqlen, head_dim) -> (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class PackedAttentionMoT(Qwen2Attention):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        max_position_embeddings,
        rope_theta,
        is_causal,
        attention_dropout,
        qk_norm: bool,
        rms_norm_eps,
        freeze_und: bool,
        layer_idx,
        use_custom_kernel: bool = False # judge which kernel to use
    ):
        super().__init__(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            max_position_embeddings,
            rope_theta,
            is_causal,
            attention_dropout,
        )
        self.freeze_und = freeze_und
        self.layer_idx = layer_idx
        if qk_norm:
            self.q_norm = Qwen2RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = Qwen2RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.q_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.q_norm_moe_gen = nn.Identity()
            self.k_norm_moe_gen = nn.Identity()

        self.q_proj_moe_gen = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj_moe_gen = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj_moe_gen = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj_moe_gen = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        ############# new add ################
        if use_custom_kernel and not IS_CUSTOM_ATTENTION_AVAILABLE:
            raise ImportError("`use_custom_kernel=True` but the 'custom_attention' package could not be imported.")
        self.use_custom_kernel = use_custom_kernel

    # @timeit
    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
    ):
        packed_query_states = packed_sequence.new_zeros((packed_sequence.shape[0], self.num_heads * self.head_dim))
        packed_key_states = packed_sequence.new_zeros((packed_sequence.shape[0], self.num_key_value_heads * self.head_dim))
        packed_value_states = packed_sequence.new_zeros((packed_sequence.shape[0], self.num_key_value_heads * self.head_dim))

        packed_sequence_und = packed_sequence[packed_und_token_indexes]
        packed_sequence_gen = packed_sequence[packed_gen_token_indexes]

        packed_query_states[packed_und_token_indexes] = self.q_proj(packed_sequence_und)
        packed_query_states[packed_gen_token_indexes] = self.q_proj_moe_gen(packed_sequence_gen)

        packed_key_states[packed_und_token_indexes] = self.k_proj(packed_sequence_und)
        packed_key_states[packed_gen_token_indexes] = self.k_proj_moe_gen(packed_sequence_gen)

        packed_value_states[packed_und_token_indexes] = self.v_proj(packed_sequence_und)
        packed_value_states[packed_gen_token_indexes] = self.v_proj_moe_gen(packed_sequence_gen)

        packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
        packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)
        if self.freeze_und:
            packed_value_states[packed_und_token_indexes] = packed_value_states[packed_und_token_indexes].detach()

        packed_query_states_ = packed_query_states.new_zeros(packed_query_states.shape)
        packed_key_states_ = packed_key_states.new_zeros(packed_key_states.shape)

        packed_query_states_[packed_und_token_indexes] = self.q_norm(packed_query_states[packed_und_token_indexes])
        if self.freeze_und:
            packed_query_states_[packed_und_token_indexes] = packed_query_states_[packed_und_token_indexes].detach()
        packed_query_states_[packed_gen_token_indexes] = self.q_norm_moe_gen(packed_query_states[packed_gen_token_indexes])

        packed_key_states_[packed_und_token_indexes] = self.k_norm(packed_key_states[packed_und_token_indexes])
        if self.freeze_und:
            packed_key_states_[packed_und_token_indexes] = packed_key_states_[packed_und_token_indexes].detach()
        packed_key_states_[packed_gen_token_indexes] = self.k_norm_moe_gen(packed_key_states[packed_gen_token_indexes])

        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states_, packed_key_states_ = apply_rotary_pos_emb(
            packed_query_states_, packed_key_states_, packed_cos, packed_sin, unsqueeze_dim=1
        )

        if self.use_custom_kernel:
            # --- Path A: Call our own CUDA Kernel ---
            print("(Using a custom CUDA Kernel...)")

            # [Important Fix]: Handle GQA before reshaping
            # Repeat the heads of K and V to match the number of heads in Q
            # The shape of packed_..._ is (TotalSeqLen, NumKeyValueHeads, HeadDim)
            # First, we add a batch dimension
            k_gqa = packed_key_states_.unsqueeze(0)
            v_gqa = packed_value_states.unsqueeze(0)
            
            # Call repeat_kv for replication
            k_gqa = repeat_kv(k_gqa, self.num_key_value_groups)
            v_gqa = repeat_kv(v_gqa, self.num_key_value_groups)

            # Remove the batch dimension to restore the packed shape
            k_repeated = k_gqa.squeeze(0)
            v_repeated = v_gqa.squeeze(0)
            
            # Prepare inputs: Our custom kernel expects the shape (B, H, S, D)
            # Use the GQA-processed k_repeated and v_repeated
            q = packed_query_states_.permute(1, 0, 2).unsqueeze(0)
            k = k_repeated.permute(1, 0, 2).unsqueeze(0)
            v = v_repeated.permute(1, 0, 2).unsqueeze(0)

            # Adapt the Mask
            mask = attention_mask[0] if isinstance(attention_mask, List) else attention_mask

            attn_output_custom = custom_attention.forward(q, k, v, mask)

            # Reshape to match the subsequent code
            packed_attn_output = attn_output_custom.squeeze(0).permute(1, 0, 2)
            
        else:
            # --- Path B: Use the original PyTorch implementation (unchanged) ---
            print("   (Using native PyTorch Attention...)")

            if isinstance(attention_mask, List):
                # help understand the else branch
                packed_key_states_ = packed_key_states_[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
                packed_key_states_ = packed_key_states_.reshape(-1, self.num_heads, self.head_dim)
                packed_value_states = packed_value_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
                packed_value_states = packed_value_states.reshape(-1, self.num_heads, self.head_dim)

                unpacked_query_states = packed_query_states_.transpose(0, 1).split(sample_lens, dim=1)
                unpacked_key_states = packed_key_states_.transpose(0, 1).split(sample_lens, dim=1)
                unpacked_value_states = packed_value_states.transpose(0, 1).split(sample_lens, dim=1)
                upacked_attn_output = []
                for query_states, key_states, value_states, attention_mask_per_sample in zip(
                    unpacked_query_states, unpacked_key_states, unpacked_value_states, attention_mask
                ):
                    with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                        # Attention mechanism call part
                        attn_output = scaled_dot_product_attention(
                            query_states.to(torch.bfloat16).unsqueeze(0), 
                            key_states.to(torch.bfloat16).unsqueeze(0), 
                            value_states.to(torch.bfloat16).unsqueeze(0),
                            attn_mask=attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                        )
                    upacked_attn_output.append(attn_output.squeeze(0))
                packed_attn_output = torch.cat(upacked_attn_output, dim=1)
            else:
                pad_size = sum(sample_lens) - packed_query_states.shape[0]
                packed_query_states_ = pad_sequence(packed_query_states_.permute(1, 0, 2), pad_size)
                packed_key_states_ = pad_sequence(packed_key_states_.permute(1, 0, 2), pad_size)
                packed_value_states = pad_sequence(packed_value_states.permute(1, 0, 2), pad_size)
                # Attention mechanism call part
                packed_attn_output = flex_attention(
                    packed_query_states_.unsqueeze(0), # 1, num_head, L, head_dim
                    packed_key_states_.unsqueeze(0), 
                    packed_value_states.unsqueeze(0), 
                    enable_gqa=True,
                    block_mask=attention_mask,
                )
                end_index = packed_attn_output.shape[2] - pad_size
                packed_attn_output = packed_attn_output[0, :, :end_index, :]

        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(-1, self.num_heads * self.head_dim)
        packed_attn_output_ = packed_attn_output.new_zeros(packed_attn_output.shape)
        packed_attn_output_[packed_und_token_indexes] = self.o_proj(packed_attn_output[packed_und_token_indexes])
        packed_attn_output_[packed_gen_token_indexes] = self.o_proj_moe_gen(packed_attn_output[packed_gen_token_indexes])

        return packed_attn_output_
    # @timeit
    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ):
        if mode == 'und':
            # 投影并调整形状为 [seq_len, num_heads, head_dim]
            packed_query_states = self.q_proj(packed_query_sequence).view(-1, self.num_heads, self.head_dim)
            packed_key_states = self.k_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states = self.v_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
            packed_query_states = self.q_norm(packed_query_states)
            packed_key_states = self.k_norm(packed_key_states)
        elif mode == 'gen':
            packed_query_sequence = packed_query_sequence.to(torch.bfloat16)
            packed_query_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_heads * self.head_dim))
            packed_key_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim))
            packed_value_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim))

            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]

            packed_query_states[packed_text_indexes] = self.q_proj(packed_text_query_sequence)
            packed_query_states[packed_vae_token_indexes] = self.q_proj_moe_gen(packed_vae_query_sequence)

            packed_key_states[packed_text_indexes] = self.k_proj(packed_text_query_sequence)
            packed_key_states[packed_vae_token_indexes] = self.k_proj_moe_gen(packed_vae_query_sequence)

            packed_value_states[packed_text_indexes] = self.v_proj(packed_text_query_sequence)
            packed_value_states[packed_vae_token_indexes] = self.v_proj_moe_gen(packed_vae_query_sequence)

            # 调整形状为 [seq_len, num_heads, head_dim]
            packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
            packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)

            packed_query_states = packed_query_states.to(torch.float32)
            packed_query_states[packed_text_indexes] = self.q_norm(packed_query_states[packed_text_indexes])
            packed_query_states[packed_vae_token_indexes] = self.q_norm_moe_gen(packed_query_states[packed_vae_token_indexes])

            packed_key_states = packed_key_states.to(torch.float32)
            packed_key_states[packed_text_indexes] = self.k_norm(packed_key_states[packed_text_indexes])
            packed_key_states[packed_vae_token_indexes] = self.k_norm_moe_gen(packed_key_states[packed_vae_token_indexes])

        # 应用旋转位置编码
        packed_cos, packed_sin = packed_query_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states, packed_key_states, packed_cos, packed_sin, unsqueeze_dim=1
        )

        # 转换数据类型
        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)

        # 处理历史键值对
        if past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None:
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]

            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
            merged_value_states = past_key_states.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        # 1. 处理GQA (Grouped Query Attention)
        # 重复key和value的头以匹配query的头数量
        if self.num_key_value_groups > 1:
            # 增加批次维度以便重复
            packed_query_states = packed_query_states.unsqueeze(0)  # [1, seq_len, num_heads, head_dim]
            
            # 对key和value进行重复以匹配query的头数量
            merged_key_states = repeat_kv(merged_key_states.unsqueeze(0), self.num_key_value_groups)  # [1, seq_len, num_heads, head_dim]
            merged_value_states = repeat_kv(merged_value_states.unsqueeze(0), self.num_key_value_groups)  # [1, seq_len, num_heads, head_dim]
        else:
            # 增加批次维度
            packed_query_states = packed_query_states.unsqueeze(0)
            merged_key_states = merged_key_states.unsqueeze(0)
            merged_value_states = merged_value_states.unsqueeze(0)

        # 2. 解包并填充张量
        from torch.nn.utils.rnn import pad_sequence

        # 将打包的张量分割成一个列表，每个元素是一个序列
        query_list = torch.split(packed_query_states[0], query_lens.tolist(), dim=0)  # 移除批次维度后分割
        key_list = torch.split(merged_key_states[0], key_values_lens.tolist(), dim=0)
        value_list = torch.split(merged_value_states[0], key_values_lens.tolist(), dim=0)

        # 将列表中的序列进行填充，使它们具有相同的最大长度
        # 填充后形状: [batch_size, max_seq_len, num_heads, head_dim]
        padded_query = pad_sequence(query_list, batch_first=True)
        padded_key = pad_sequence(key_list, batch_first=True)
        padded_value = pad_sequence(value_list, batch_first=True)

        # 3. 调整维度顺序为SDPA所需的 [batch_size, num_heads, seq_len, head_dim]
        padded_query = padded_query.transpose(1, 2)  # [batch, num_heads, seq_len_q, head_dim]
        padded_key = padded_key.transpose(1, 2)      # [batch, num_heads, seq_len_k, head_dim]
        padded_value = padded_value.transpose(1, 2)  # [batch, num_heads, seq_len_k, head_dim]

        # 4. 创建注意力掩码
        # 掩码形状需要是 [batch_size, 1, seq_len_q, seq_len_k] 以匹配SDPA要求
        mask_list = [torch.ones(l, dtype=torch.bool, device=padded_key.device) for l in key_values_lens]
        padding_mask = pad_sequence(mask_list, batch_first=True, padding_value=False)  # [batch, seq_len_k]
        
        # 扩展掩码维度以匹配注意力计算
        attn_mask = ~padding_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len_k]
        
        # 如果查询和键的长度不同，需要调整掩码
        if padded_query.shape[2] != padded_key.shape[2]:
            # 创建新的掩码并复制值
            batch_size, _, seq_len_q, _ = padded_query.shape
            _, _, seq_len_k, _ = padded_key.shape
            new_mask = torch.zeros(batch_size, 1, seq_len_q, seq_len_k, dtype=torch.bool, device=padding_mask.device)
            new_mask[:, :, :, :padding_mask.shape[1]] = attn_mask
            attn_mask = new_mask

        # 5. 调用SDPA函数
        if is_causal:
            # 因果模式下，SDPA会自动处理因果掩码
            padded_attn_output = torch.nn.functional.scaled_dot_product_attention(
                padded_query,
                padded_key,
                padded_value,
                attn_mask=attn_mask,
                is_causal=True,
            )
        else:
            padded_attn_output = torch.nn.functional.scaled_dot_product_attention(
                padded_query,
                padded_key,
                padded_value,
                attn_mask=attn_mask,
                is_causal=False,
            )

        # 6. 调整输出维度顺序并重新打包
        padded_attn_output = padded_attn_output.transpose(1, 2)  # [batch, seq_len_q, num_heads, head_dim]
        
        # 重新打包输出张量
        attn_output_list = [padded_attn_output[i, :query_lens[i]] for i in range(len(query_lens))]
        packed_attn_output = torch.cat(attn_output_list, dim=0)  # [total_seq_len, num_heads, head_dim]
        
        # 重塑为 [total_seq_len, hidden_size]
        packed_attn_output = packed_attn_output.reshape(-1, self.num_heads * self.head_dim)
        
        # 应用输出投影
        if mode == 'und':
            packed_attn_output = self.o_proj(packed_attn_output)
        elif mode == 'gen':
            packed_attn_output[packed_text_indexes] = self.o_proj(packed_attn_output[packed_text_indexes])
            packed_attn_output[packed_vae_token_indexes] = self.o_proj_moe_gen(packed_attn_output[packed_vae_token_indexes])

        # 更新历史键值对
        if update_past_key_values and past_key_values is not None:
            # 移除批次维度后保存
            past_key_values.key_cache[self.layer_idx] = merged_key_states[0]
            past_key_values.value_cache[self.layer_idx] = merged_value_states[0]

        return packed_attn_output, past_key_values

    
    
if __name__ == "__main__":
    pass