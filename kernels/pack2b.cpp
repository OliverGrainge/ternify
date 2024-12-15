#include "pack2b_cpu.h"
#ifdef USE_CUDA
#include "pack2b_cuda.h"
#endif
#include <torch/extension.h>

torch::Tensor pack2b(torch::Tensor A)
{
    // Common checks for both CPU and CUDA implementations
    TORCH_CHECK(A.dtype() == torch::kInt8, "Tensor 'A' must be of type int8");
    TORCH_CHECK(A.dim() == 2, "Tensor 'A' must be 2-dimensional");
    TORCH_CHECK(A.size(1) % 4 == 0, "Width must be a multiple of 4");

    A = A.contiguous(); // Ensure contiguous layout

    int64_t M = A.size(0);
    int64_t N = A.size(1);
    int64_t packed_cols = N / 4;

    // Check device and use appropriate implementation
    if (A.device().is_cuda())
    {
#ifdef USE_CUDA
        torch::Tensor C = torch::zeros({M, packed_cols}, torch::dtype(torch::kInt8).device(torch::kCUDA));
        _pack2b_cuda(A.data_ptr<int8_t>(), M, N, C.data_ptr<int8_t>());
        return C;
#else
        TORCH_CHECK(false, "CUDA implementation not available. Recompile with USE_CUDA defined.");
#endif
    }
    else if (A.device().is_cpu())
    {
        torch::Tensor C = torch::zeros({M, packed_cols}, torch::dtype(torch::kInt8).device(torch::kCPU));
        _pack2b_cpu(A.data_ptr<int8_t>(), M, N, C.data_ptr<int8_t>());
        return C;
    }
    else
    {
        TORCH_CHECK(false, "Tensor 'A' must be on CPU or CUDA.");
    }
}
