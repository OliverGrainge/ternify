#include "matmul2b.h"

void _matmul2b_cpu(const int8_t* A, const int8_t* B_packed, int32_t* C, int64_t BATCH, int64_t M, int64_t N, int64_t K) {
    for (int64_t b = 0; b < BATCH; ++b) {
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < K; ++j) {
                int32_t sum = 0;
                for (int64_t k = 0; k < N; ++k) {
                    int64_t idx = (k * K + j)/4;
                    int64_t shift = (idx % 4) * 2;
                    sum += A[b * M * N + i * N + k] * ((B_packed[idx] >> shift) & 0b11);
                }
                C[b * M * K + i * K + j] = sum;
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
        TORCH_CHECK(B.size(0) == N, "Tensor dimensions are not compatible for matrix multiplication");

        // Create an output tensor
        C = torch::zeros({BATCH, M, K}, torch::dtype(A.dtype()).device(torch::kCPU));
    } else if (A.dim() == 2) {
        // Non-batched case
        BATCH = 1;
        M = A.size(0);
        N = A.size(1);
        K = B.size(1);
        TORCH_CHECK(B.size(0) == N/4, "Tensor dimensions are not compatible for matrix multiplication");

        // Create an output tensor
        torch::Tensor C = torch::zeros({BATCH, M, K}, torch::dtype(torch::kInt32).device(torch::kCPU));
    } else {
        TORCH_CHECK(false, "Tensor 'A' must be either 2-dimensional or 3-dimensional");
    }

    // Get pointers to the underlying data
    const int8_t* A_ptr = A.data_ptr<int8_t>();
    const int8_t* B_ptr = B.data_ptr<int8_t>();
    int32_t* C_ptr = C.data_ptr<int32_t>();

    // Compute the matrix multiplication (naive_matmul)
    _matmul2b_cpu(A_ptr, B_ptr, C_ptr, BATCH, M, N, K);

    return C;
}
