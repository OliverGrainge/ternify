#include "sgemm.h"
#include <cassert>
#include <cstring>   // For memset

void sgemm(
    const float* A,          // First input matrix (M x K)
    const float* B,          // Second input matrix (K x N)
    float* C,                // Output matrix (M x N), can be pre-allocated or an empty tensor
    float alpha,             // Scalar multiplier for A * B
    float beta,              // Scalar multiplier for C (if accumulation is needed)
    bool transA,             // Whether to transpose A
    bool transB              // Whether to transpose B
) {
    assert(A != nullptr && B != nullptr && C != nullptr);

    // Determine the dimensions based on transposition flags
    size_t M, K, N;
    if (transA) {
        // If A is transposed, M and K are swapped
        M = K;
        K = M;
    } else {
        // Original dimensions for A
        // Assuming M and K are globally defined or passed in some other way
    }
    if (transB) {
        // If B is transposed, K and N are swapped
        N = K;
    } else {
        // Original dimensions for B
        // Assuming N and K are globally defined or passed in some other way
    }

    // Initialize C with beta * C if beta is not zero
    if (beta != 0.0f) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                C[i * N + j] *= beta;
            }
        }
    } else {
        memset(C, 0, M * N * sizeof(float));
    }

    // Perform the matrix multiplication
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                float a_val = transA ? A[k * M + i] : A[i * K + k];
                float b_val = transB ? B[j * K + k] : B[k * N + j];
                sum += a_val * b_val;
            }
            C[i * N + j] += alpha * sum;
        }
    }
}
