#include <torch/extension.h>
#include "unpack2b_cpu.h"

torch::Tensor unpack2b(torch::Tensor A)
{
    TORCH_CHECK(A.device().is_cpu(), "Tensor 'A' must be on CPU");
    TORCH_CHECK(A.dtype() == torch::kInt8, "Tensor 'A' must be of type int8");
    TORCH_CHECK(A.dim() == 2, "Tensor 'A' must be 2-dimensional");

    int64_t M = A.size(0);
    int64_t N = A.size(1) * 4;

    torch::Tensor B = torch::zeros({M, N}, torch::dtype(torch::kInt8).device(torch::kCPU));
    _unpack2b_cpu(A.data_ptr<int8_t>(), M, N, B.data_ptr<int8_t>());
    return B;
}
