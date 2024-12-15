#include "unpack2b_cpu.h"
#ifdef USE_CUDA
#include "unpack2b_cuda.h"
#endif
#include <torch/extension.h>

torch::Tensor unpack2b(torch::Tensor A)
{
    // Common checks for both CPU and CUDA implementations
    TORCH_CHECK(A.dtype() == torch::kInt8, "Tensor 'A' must be of type int8");
    TORCH_CHECK(A.dim() == 2, "Tensor 'A' must be 2-dimensional");

    int64_t M = A.size(0);
    int64_t packed_cols = A.size(1);
    int64_t N = packed_cols * 4;

    // Check device and use appropriate implementation
    if (A.device().is_cuda())
    {
#ifdef USE_CUDA
        torch::Tensor B = torch::zeros({M, N}, torch::dtype(torch::kInt8).device(torch::kCUDA));
        _unpack2b_cuda(A.data_ptr<int8_t>(), M, N, B.data_ptr<int8_t>());
        return B;
#else
        TORCH_CHECK(false, "CUDA implementation not available. Recompile with USE_CUDA defined.");
#endif
    }
    else if (A.device().is_cpu())
    {
        torch::Tensor B = torch::zeros({M, N}, torch::dtype(torch::kInt8).device(torch::kCPU));
        _unpack2b_cpu(A.data_ptr<int8_t>(), M, N, B.data_ptr<int8_t>());
        return B;
    }
    else
    {
        TORCH_CHECK(false, "Tensor 'A' must be on CPU or CUDA.");
    }
}
