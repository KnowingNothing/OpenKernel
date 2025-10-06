#include <torch/extension.h>
#include <vector>

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
    int head_dim);
}

torch::Tensor attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor mask) {
    TORCH_CHECK(q.dim() == 4, "Query must be a 4D tensor");

    auto q_fp32 = q.to(torch::kFloat32);
    auto k_fp32 = k.to(torch::kFloat32);
    auto v_fp32 = v.to(torch::kFloat32);
    auto mask_fp32 = mask.to(torch::kFloat32);

    const int batch_size = q_fp32.size(0);
    const int num_heads = q_fp32.size(1);
    const int seq_len = q_fp32.size(2);
    const int head_dim = q_fp32.size(3);

    torch::Tensor output = torch::empty_like(q);
    auto output_fp32 = torch::empty_like(q_fp32);

    attention_forward_cuda(
        q_fp32.data_ptr<float>(),
        k_fp32.data_ptr<float>(),
        v_fp32.data_ptr<float>(),
        mask_fp32.data_ptr<float>(),
        output_fp32.data_ptr<float>(),
        batch_size,
        num_heads,
        seq_len,
        head_dim
    );

    output.copy_(output_fp32);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &attention_forward, "Full Attention Forward with Mask (CUDA)");
}