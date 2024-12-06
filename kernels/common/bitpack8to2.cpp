#include "bitpack8to2.h"
#include <omp.h>

void _bitpack8to2_cpu(const int8_t* A, int64_t M, int64_t N, int8_t* B) {
    #pragma omp parallel for
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j += 4) {
            int64_t idx = i * N + j;
            int64_t packed_idx = i * ((N + 3) / 4) + j / 4;
            int8_t packed_value = 0;

            for (int k = 0; k < 4 && j + k < N; ++k) {
                packed_value |= (A[idx + k] & 0x3) << (k * 2);
            }

            B[packed_idx] = packed_value;
        }
    }
}

torch::Tensor bitpack8to2_cpu(torch::Tensor A) {
    TORCH_CHECK(A.device().is_cpu(), "Tensor 'A' must be on CPU");
    TORCH_CHECK(A.dtype() == torch::kInt8, "Tensor 'A' must be of type int8");

    int64_t M = A.size(0);
    int64_t N = A.size(1);
    TORCH_CHECK(A.dim() == 2, "Tensor 'A' must be 2-dimensional");

    // Adjust the size of C to accommodate the packed data
    int64_t packed_cols = (N + 3) / 4; // Ensure enough space for all elements
    torch::Tensor C = torch::zeros({M, packed_cols}, torch::dtype(torch::kInt8).device(torch::kCPU));
    _bitpack8to2_cpu(A.data_ptr<int8_t>(), M, N, C.data_ptr<int8_t>());
    return C;
}
