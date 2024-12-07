#include "unpack2b.h"
#include <iostream>

void _unpack2b_cpu(const int8_t* A, int64_t M, int64_t N, int8_t* B) {
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            int64_t idx = i * N + j;
            int64_t shift = (3 - (j % 4)) * 2;
            B[idx] = (A[idx/4] >> shift) & 0b11;
        }
    }
}


torch::Tensor unpack2b_cpu(torch::Tensor A) {
    TORCH_CHECK(A.device().is_cpu(), "Tensor 'A' must be on CPU");
    TORCH_CHECK(A.dtype() == torch::kInt8, "Tensor 'A' must be of type int8");
    TORCH_CHECK(A.dim() == 2, "Tensor 'A' must be 2-dimensional");

    int64_t M = A.size(0);
    int64_t N = A.size(1) * 4; 

    torch::Tensor B = torch::zeros({M, N}, torch::dtype(torch::kInt8).device(torch::kCPU));

    _unpack2b_cpu(A.data_ptr<int8_t>(), M, N, B.data_ptr<int8_t>());
    return B;
}

