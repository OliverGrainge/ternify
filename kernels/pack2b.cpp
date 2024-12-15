#include "pack2b_cpu.h"
#include <torch/extension.h>

torch::Tensor pack2b(torch::Tensor A)
{
    TORCH_CHECK(A.device().is_cpu(), "Tensor 'A' must be on CPU");
    TORCH_CHECK(A.dtype() == torch::kInt8, "Tensor 'A' must be of type int8");
    TORCH_CHECK(A.dim() == 2, "Tensor 'A' must be 2-dimensional");
    TORCH_CHECK(A.size(1) % 4 == 0, "Width must be a multiple of 4");

    A = A.contiguous(); // Ensure contiguous layout

    int64_t M = A.size(0);
    int64_t N = A.size(1);
    int64_t packed_cols = N / 4;

    torch::Tensor C = torch::zeros({M, packed_cols}, torch::dtype(torch::kInt8).device(torch::kCPU));
    _pack2b_cpu(A.data_ptr<int8_t>(), M, N, C.data_ptr<int8_t>());

    return C;
}