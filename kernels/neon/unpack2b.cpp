#include <omp.h>
#include "unpack2b.h"
#include <iostream>

void _unpack2b_cpu(const int8_t* A, int64_t M, int64_t N, int8_t* B) {
    // N is the full count after unpacking (A has N/4 columns)
    int64_t packed_cols = (N + 3) / 4;

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < M; ++i) {
        const int8_t* A_row = A + i * packed_cols;
        int8_t* B_row = B + i * N;

        for (int64_t j = 0; j < N; ++j) {
            int64_t b_idx = j >> 2;         // j/4
            int64_t shift = (3 - (j & 3)) * 2; // (j%4)
            B_row[j] = (A_row[b_idx] >> shift) & 0b11;
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
