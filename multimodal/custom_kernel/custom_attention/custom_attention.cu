#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Kernel 1: Scaled Dot-Product
__global__ void scaled_dot_product_kernel_naive(
    const float* q_data, const float* k_data, 
    const float* mask_data,     
    float* scores_data,
    int seq_len, int head_dim, float scale) 
{
    const int query_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int key_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (query_idx >= seq_len || key_idx >= seq_len) return;

    const int mask_idx = query_idx * seq_len + key_idx;
    if (mask_data != nullptr && mask_data[mask_idx]) {
        scores_data[mask_idx] = -1.0e9f;
        return;
    }
    
    float sum = 0.0f;
    const int q_row_offset = query_idx * head_dim;
    const int k_row_offset = key_idx * head_dim;

    for (int i = 0; i < head_dim; ++i) {
        sum += q_data[q_row_offset + i] * k_data[k_row_offset + i];
    }
    
    scores_data[query_idx * seq_len + key_idx] = sum * scale;
}

// Kernel 2: Row-wise Softmax
__global__ void softmax_kernel_naive(const float* scores_data, float* weights_data, int seq_len)
{
    const int row = blockIdx.x;
    const int col = threadIdx.x;
    if (col >= seq_len) return;
    const float* row_input = scores_data + row * seq_len;
    float* row_output = weights_data + row * seq_len;
    float max_val = -__FLT_MAX__;
    for (int i = 0; i < seq_len; ++i) max_val = max(max_val, row_input[i]);
    float sum_val = 0.0f;
    for (int i = 0; i < seq_len; ++i) sum_val += expf(row_input[i] - max_val);
    row_output[col] = expf(row_input[col] - max_val) / sum_val;
}

// Kernel 3: Matrix Multiply 
__global__ void matrix_multiply_kernel_naive(
    const float* A, const float* B, float* C, int M, int N, int K)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) sum += A[row * K + i] * B[i * N + col];
    C[row * N + col] = sum;
}

// Host-side C++ function 
extern "C" {
void attention_forward_cuda(
    const float* q_data,
    const float* k_data,
    const float* v_data,
    const float* mask_data,
    float* output_data,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim) 
{
    float *scores_data, *weights_data;
    size_t matrix_size_qk = (size_t)seq_len * seq_len * sizeof(float);
    CUDA_CHECK(cudaMalloc(&scores_data, matrix_size_qk * batch_size * num_heads));
    CUDA_CHECK(cudaMalloc(&weights_data, matrix_size_qk * batch_size * num_heads));
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    for (int i = 0; i < batch_size * num_heads; ++i) {
        const float* q_ptr = q_data + i * seq_len * head_dim;
        const float* k_ptr = k_data + i * seq_len * head_dim;
        const float* v_ptr = v_data + i * seq_len * head_dim;
        float* score_ptr = scores_data + i * seq_len * seq_len;
        float* weight_ptr = weights_data + i * seq_len * seq_len;
        float* output_ptr = output_data + i * seq_len * head_dim;

        dim3 threads_scores(16, 16);
        dim3 blocks_scores((seq_len + threads_scores.x - 1) / threads_scores.x,
                           (seq_len + threads_scores.y - 1) / threads_scores.y);
        
        scaled_dot_product_kernel_naive<<<blocks_scores, threads_scores>>>(
            q_ptr, k_ptr, mask_data, score_ptr, seq_len, head_dim, scale
        );
        
        dim3 threads_softmax(256);
        dim3 blocks_softmax(seq_len);
        softmax_kernel_naive<<<blocks_softmax, threads_softmax>>>(
            score_ptr, weight_ptr, seq_len
        );
        
        dim3 threads_gemm(16, 16);
        dim3 blocks_gemm((head_dim + threads_gemm.x - 1) / threads_gemm.x,
                         (seq_len + threads_gemm.y - 1) / threads_gemm.y);
        matrix_multiply_kernel_naive<<<blocks_gemm, threads_gemm>>>(
            weight_ptr, v_ptr, output_ptr, seq_len, head_dim, seq_len
        );
    }
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(scores_data));
    CUDA_CHECK(cudaFree(weights_data));
}
} // extern "C"