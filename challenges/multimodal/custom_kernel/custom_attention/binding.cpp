#include <torch/extension.h>
#include <vector>

// ----------------------------------------------------------------------------
// 1. 更新 CUDA 函数声明
//    使其与 .cu 文件中 attention_forward_cuda 的签名完全一致
// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
// 2. 更新 C++ 包装函数
//    函数名改为 attention_forward，并接收第三个张量 v
// ----------------------------------------------------------------------------
// 在 binding.cpp 中

torch::Tensor attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor mask) {
    // --- 输入校验部分保持不变 ---
    TORCH_CHECK(q.dim() == 4, "Query must be a 4D tensor");
    // // ... (省略其他所有 TORCH_CHECK, 它们都是正确的)
    // TORCH_CHECK(
    //     (mask.scalar_type() == torch.kFloat32 || 
    //     mask.scalar_type() == torch.kBFloat16 || 
    //     mask.scalar_type() == torch.kFloat16), // <--- 关键修正：在这里添加了额外的括号
    //     "Mask must be a float-like tensor (float32, bfloat16, or float16)"
    // );

    // --- 核心修正：在这里统一将所有输入转换为 float32 ---
    // .to() 方法会创建一个新的张量副本（如果类型不同），或者返回原始张量（如果类型相同）。
    // 这确保了传递给 data_ptr<float>() 的张量一定是 float32 类型。
    auto q_fp32 = q.to(torch::kFloat32);
    auto k_fp32 = k.to(torch::kFloat32);
    auto v_fp32 = v.to(torch::kFloat32);
    auto mask_fp32 = mask.to(torch::kFloat32);
    // --- 修正结束 ---

    // 提取维度信息 (现在从转换后的张量中提取)
    const int batch_size = q_fp32.size(0);
    const int num_heads = q_fp32.size(1);
    const int seq_len = q_fp32.size(2);
    const int head_dim = q_fp32.size(3);

    // 准备输出张量 (输出类型与输入 q 保持一致，而不是 float32)
    torch::Tensor output = torch::empty_like(q);
    // 同时创建一个 float32 版本的输出张量用于计算
    auto output_fp32 = torch::empty_like(q_fp32);


    // 调用 CUDA 主函数 (现在使用转换后的 fp32 张量指针)
    attention_forward_cuda(
        q_fp32.data_ptr<float>(),
        k_fp32.data_ptr<float>(),
        v_fp32.data_ptr<float>(),
        mask_fp32.data_ptr<float>(),
        output_fp32.data_ptr<float>(), // <-- 将结果写入 fp32 的输出缓存
        batch_size,
        num_heads,
        seq_len,
        head_dim
    );
    
    // 将计算结果从 float32 转回原始的输入类型 (例如 bfloat16)，并存入最终的 output 张量
    output.copy_(output_fp32);

    return output;
}

// Pybind11 模块定义保持不变，因为它绑定的 C++ 函数签名中的参数类型对 Python 是透明的
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &attention_forward, "Full Attention Forward with Mask (CUDA)");
}