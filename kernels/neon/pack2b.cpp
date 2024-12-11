#include <omp.h>
#include "pack2b.h"

void _pack2b_cpu(const int8_t* A, int64_t M, int64_t N, int8_t* B) {
    int64_t packed_cols = (N + 3) / 4;

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j += 4) {
            int8_t packed_value = 0;
            int64_t base_idx = i * N + j;
            int64_t out_idx = i * packed_cols + (j / 4);

            // Manually unroll this loop since k is always small (0 to 3)
            // Also handle the boundary condition if (j + k >= N)
            {
                int8_t val0 = (j + 0 < N) ? (A[base_idx + 0] & 0x3) : 0;
                int8_t val1 = (j + 1 < N) ? (A[base_idx + 1] & 0x3) : 0;
                int8_t val2 = (j + 2 < N) ? (A[base_idx + 2] & 0x3) : 0;
                int8_t val3 = (j + 3 < N) ? (A[base_idx + 3] & 0x3) : 0;

                // Pack them into a single byte (val0 << 6 | val1 << 4 | val2 << 2 | val3)
                packed_value = (val0 << 6) | (val1 << 4) | (val2 << 2) | val3;
            }

            B[out_idx] = packed_value;
        }
    }
}

torch::Tensor pack2b_cpu(torch::Tensor A) {
    TORCH_CHECK(A.device().is_cpu(), "Tensor 'A' must be on CPU");
    TORCH_CHECK(A.dtype() == torch::kInt8, "Tensor 'A' must be of type int8");

    int64_t M = A.size(0);
    int64_t N = A.size(1);
    TORCH_CHECK(A.dim() == 2, "Tensor 'A' must be 2-dimensional");

    int64_t packed_cols = (N + 3) / 4;
    torch::Tensor C = torch::zeros({M, packed_cols}, torch::dtype(torch::kInt8).device(torch::kCPU));
    _pack2b_cpu(A.data_ptr<int8_t>(), M, N, C.data_ptr<int8_t>());
    return C;
}
