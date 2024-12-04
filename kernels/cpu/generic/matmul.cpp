#include "matmul.h"

void _matmul_cpu(const float* A, const float* B, float* C, int64_t BATCH, int64_t M, int64_t N, int64_t K) {
    for (int64_t b = 0; b < BATCH; ++b) {
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < K; ++j) {
                float sum = 0.0f;
                for (int64_t k = 0; k < N; ++k) {
                    sum += A[b * M * N + i * N + k] * B[k * K + j];
                }
                C[b * M * K + i * K + j] = sum;
            }
        }
    }
}


torch::Tensor matmul_cpu(torch::Tensor A, torch::Tensor B) {
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
        TORCH_CHECK(B.size(0) == N, "Tensor dimensions are not compatible for matrix multiplication");

        // Create an output tensor
        C = torch::zeros({M, K}, torch::dtype(A.dtype()).device(torch::kCPU));
    } else {
        TORCH_CHECK(false, "Tensor 'A' must be either 2-dimensional or 3-dimensional");
    }

    // Get pointers to the underlying data
    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Compute the matrix multiplication (naive_matmul)
    _matmul_cpu(A_ptr, B_ptr, C_ptr, BATCH, M, N, K);

    return C;
}