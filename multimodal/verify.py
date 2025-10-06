########## Verify the correctness of the custom kernel ###########
import torch
import math
# Ensure to import the correct package name you just installed
import custom_attention 

print("--- Custom Attention Operator Verification ---")

# Check whether the CUDA environment is available
if not torch.cuda.is_available():
    print("❌ CUDA is not available. This custom operator requires a GPU.")
    exit()
else:
    print("✅ CUDA is available. Running tests...")

# Define hyperparameters for the attention model
batch_size = 2
num_heads = 4
seq_len = 32 
head_dim = 16  
device = torch.device("cuda")
dtype = torch.float32

### 466 is a magical number that can bring great good luck ###
# SEED = 466
# torch.manual_seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(SEED)

# Create random input tensors
q = torch.rand(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
k = torch.rand(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
v = torch.rand(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

print(f"\nTensor shapes (B, H, S, D): ({batch_size}, {num_heads}, {seq_len}, {head_dim})")

# Execute computation and verification
try:
    print("Creating a causal attention mask...")
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    print("\n[1] Running custom CUDA operator with mask...")
    output_custom = custom_attention.forward(q, k, v, mask)
    print("   Custom operator executed successfully.")

    print("[2] Running PyTorch native operator for verification...")
    
    scale = 1.0 / math.sqrt(head_dim)
    
    attn_scores_pytorch = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    attn_scores_pytorch = attn_scores_pytorch.masked_fill(mask, -torch.inf)
    
    attn_weights_pytorch = torch.softmax(attn_scores_pytorch, dim=-1)
    output_pytorch = torch.matmul(attn_weights_pytorch, v)
    print("   PyTorch native operator executed successfully.")

    print("[3] Verifying results...")
    
    are_outputs_close = torch.allclose(output_custom, output_pytorch, atol=1e-4, rtol=1e-4)

    if are_outputs_close:
        print("✅ SUCCESS: The outputs of the custom operator and PyTorch are consistent!")
    else:
        print("❌ FAILURE: The outputs do not match.")
        difference = torch.abs(output_custom - output_pytorch).max().item()
        print(f"   Max absolute difference: {difference}")

    # Print a small slice of the output for visual comparison
    print("\n--- Output Slice Comparison ---")
    print("Custom op output[0,0,0,:8]:")
    print(output_custom[0, 0, 0, :8])
    print("\nPyTorch op output[0,0,0,:8]:")
    print(output_pytorch[0, 0, 0, :8])
    print("---------------------------------")

except Exception as e:
    print(f"\n❌ An error occurred during execution: {e}")
    print("   Please check your C++/CUDA code and ensure it has been compiled correctly with the latest changes.")