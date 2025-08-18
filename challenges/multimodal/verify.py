########## 验证custom kernel 的正确性 ###########
import torch
import math
# 确保导入你刚刚安装的、正确的包名
import custom_attention 

print("--- Custom Attention Operator Verification ---")

# 1. 检查 CUDA 环境是否可用
if not torch.cuda.is_available():
    print("❌ CUDA is not available. This custom operator requires a GPU.")
    exit()
else:
    print("✅ CUDA is available. Running tests...")

# 2. 定义 Attention 模型的超参数
# 为了快速测试，我们使用一组较小的尺寸
batch_size = 2
num_heads = 4
seq_len = 32  # 序列长度
head_dim = 16   # 每个头的维度
device = torch.device("cuda")
dtype = torch.float32

# 3. 创建随机的输入张量 (Q, K, V)
# 使用 torch.rand 确保数值稳定，并设置 requires_grad=False 因为我们只做前向验证
q = torch.rand(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
k = torch.rand(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
v = torch.rand(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

print(f"\nTensor shapes (B, H, S, D): ({batch_size}, {num_heads}, {seq_len}, {head_dim})")

# 4. 执行计算和验证
try:
    # --- 核心修改：将 Mask 的创建和使用逻辑合并到这里 ---
    
    # a. 创建 Mask 张量
    print("Creating a causal attention mask...")
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    # b. 调用我们的自定义 CUDA 算子 (传入 mask)
    print("\n[1] Running custom CUDA operator with mask...")
    output_custom = custom_attention.forward(q, k, v, mask)
    print("   Custom operator executed successfully.")

    # c. 调用 PyTorch 原生实现作为对比基准 (同样使用 mask)
    print("[2] Running PyTorch native operator for verification...")
    
    # 定义 scale 变量
    scale = 1.0 / math.sqrt(head_dim)
    
    # 计算原始分数
    attn_scores_pytorch = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # 在 softmax 之前应用 mask
    attn_scores_pytorch = attn_scores_pytorch.masked_fill(mask, -torch.inf)
    
    # 计算 softmax 和最终输出
    attn_weights_pytorch = torch.softmax(attn_scores_pytorch, dim=-1)
    output_pytorch = torch.matmul(attn_weights_pytorch, v)
    print("   PyTorch native operator executed successfully.")

    # --- 5. 验证结果 ---
    print("\n[3] Verifying results...")
    
    # 使用 allclose 来比较浮点数张量，atol 是绝对容忍度
    are_outputs_close = torch.allclose(output_custom, output_pytorch, atol=1e-4, rtol=1e-4)

    if are_outputs_close:
        print("✅ SUCCESS: The outputs of the custom operator and PyTorch are consistent!")
    else:
        print("❌ FAILURE: The outputs do not match.")
        difference = torch.abs(output_custom - output_pytorch).max().item()
        print(f"   Max absolute difference: {difference}")

    # 打印一小部分输出进行直观对比
    print("\n--- Output Slice Comparison ---")
    print("Custom op output[0,0,0,:8]:")
    print(output_custom[0, 0, 0, :8])
    print("\nPyTorch op output[0,0,0,:8]:")
    print(output_pytorch[0, 0, 0, :8])
    print("---------------------------------")

except Exception as e:
    print(f"\n❌ An error occurred during execution: {e}")
    print("   Please check your C++/CUDA code and ensure it has been compiled correctly with the latest changes.")