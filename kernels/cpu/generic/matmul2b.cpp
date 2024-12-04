#include "matmul2b.h"
#include <omp.h>
#include <algorithm>

void _matmul2b_cpu(const int8_t* A, const int8_t* B_packed, int8_t* C, int64_t BATCH, int64_t M, int64_t N, int64_t K) {
    const int64_t BLOCK_SIZE = 64; // Tunable parameter for blocking

    #pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < BATCH; ++b) {
        for (int64_t i_block = 0; i_block < M; i_block += BLOCK_SIZE) {
            for (int64_t j_block = 0; j_block < K; j_block += BLOCK_SIZE) {
                for (int64_t k_block = 0; k_block < N; k_block += BLOCK_SIZE) {
                    // Compute the boundaries of the block
                    int64_t i_max = std::min(i_block + BLOCK_SIZE, M);
                    int64_t j_max = std::min(j_block + BLOCK_SIZE, K);
                    int64_t k_max = std::min(k_block + BLOCK_SIZE, N);

                    // Iterate over the block
                    for (int64_t i = i_block; i < i_max; ++i) {
                        for (int64_t j = j_block; j < j_max; ++j) {
                            int32_t sum = 0;
                            for (int64_t k = k_block; k < k_max; ++k) {
                                int64_t idx = (k * K / 4) + j / 4;
                                int64_t shift = (j % 4) * 2;
                                sum += A[b * M * N + i * N + k] * (((B_packed[idx] >> shift) & 0b11) - 1);
                            }
                            #pragma omp atomic
                            C[b * M * K + i * K + j] += sum;
                        }
                    }
                }
            }
        }
    }
}

torch::Tensor matmul2b_cpu(torch::Tensor A, torch::Tensor B) {
    // Ensure the tensors are on CPU
    TORCH_CHECK(A.device().is_cpu(), "Tensor 'A' must be on CPU");
    TORCH_CHECK(B.device().is_cpu(), "Tensor 'B' must be on CPU");

    int64_t BATCH, M, N, K;
    torch::Tensor C;

    if (A.dim() == 3) {
        // Batched case
        BATCH = A.size(0);
        M = A.size(1);
        N = A.size(2);
        K = B.size(1);
        // Create an output tensor
        C = torch::zeros({BATCH, M, K * 4}, torch::dtype(torch::kChar).device(torch::kCPU));
    } else if (A.dim() == 2) {
        // Non-batched case
        BATCH = 1;
        M = A.size(0);
        N = A.size(1);
        K = B.size(1);
        // Create an output tensor
        C = torch::zeros({M, K * 4}, torch::dtype(torch::kChar).device(torch::kCPU));
    } else {
        TORCH_CHECK(false, "Tensor 'A' must be either 2-dimensional or 3-dimensional");
    }

    // Get pointers to the underlying data
    const int8_t* A_ptr = A.data_ptr<int8_t>();
    const int8_t* B_ptr = B.data_ptr<int8_t>();
    int8_t* C_ptr = C.data_ptr<int8_t>();

    // Compute the matrix multiplication (optimized version)
    _matmul2b_cpu(A_ptr, B_ptr, C_ptr, BATCH, M, N, K * 4);

    return C;
}
