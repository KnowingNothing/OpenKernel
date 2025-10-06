# ====== Only check the correctness of costurm kernel ======
import torch
import math
# Attention! Self-written python packages need to be imported after torch
import custom_attention 

print("--- Custom Attention Operator Verification ---")

if not torch.cuda.is_available():
    print("CUDA is not available. This custom operator requires a GPU.")
    exit()
else:
    print("CUDA is available. Running tests...")

# Define the hyperparameters of the Attention model
batch_size = 2
num_heads = 4
seq_len = 32  
head_dim = 16   
device = torch.device("cuda")
dtype = torch.float32

q = torch.rand(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
k = torch.rand(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
v = torch.rand(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

print(f"\nTensor shapes (B, H, S, D): ({batch_size}, {num_heads}, {seq_len}, {head_dim})")

# Perform calculations and verifications
try:
    # --- Merge the creation and usage logic of the mask here ---

    # Create the mask tensor
    print("Creating a causal attention mask...")
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.float), diagonal=1)

    # Call the custom CUDA operator (pass the mask)
    print("\n[1] Running custom CUDA operator with mask...")
    output_custom = custom_attention.forward(q, k, v, mask)
    print("   Custom operator executed successfully.")

    # c. Call the PyTorch native implementation as a comparison baseline (also using the mask)
    print("[2] Running PyTorch native operator for verification...")
    
    # Define the scale variable
    scale = 1.0 / math.sqrt(head_dim)

    # Compute the raw attention scores
    attn_scores_pytorch = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Apply the mask before softmax
    attn_scores_pytorch = attn_scores_pytorch.masked_fill(mask, -torch.inf)

    # Compute softmax and final output
    attn_weights_pytorch = torch.softmax(attn_scores_pytorch, dim=-1)
    output_pytorch = torch.matmul(attn_weights_pytorch, v)
    print("   PyTorch native operator executed successfully.")

    # --- 5. Verify results ---
    print("\n[3] Verifying results...")

    # Use allclose to compare floating-point tensors, atol is the absolute tolerance
    are_outputs_close = torch.allclose(output_custom, output_pytorch, atol=1e-4)

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