#include "sgemm.h"
#include <cassert>
#include <cstring> 
#include <cstddef> 

void sgemm(
    const float* A,          // First input matrix (M x K)
    const float* B,          // Second input matrix (K x N)
    float* C,                // Output matrix (M x N), can be pre-allocated or an empty tensor
    size_t M,   // Number of rows in A
    size_t K,   // Number of columns in A   
    size_t J,   // Number of rows in B
    size_t N,   // Number of columns in B
    float alpha,             // Scalar multiplier for A* B
    float beta,              // Scalar multiplier for C (if accumulation is needed)
    bool transA,            // Whether to transpose A
    bool transB              // Whether to transpose B
) {
    assert(A != nullptr && B != nullptr && C != nullptr);


    size_t C_rows = transA ? K : M;
    size_t C_cols = transB ? J : N;
    // Initialize C with beta * C if beta is not zero
    if (beta != 0.0f && beta != 1.0f) {
        for (size_t i = 0; i < C_rows; ++i) {
            for (size_t j = 0; j < C_cols; ++j) {
                C[i * C_cols + j] *= beta;
            }
        }
    } else if (beta == 0.0f) {
        memset(C, 0, C_rows * C_cols * sizeof(float));
    }

    // Perform the matrix multiplication
    for (size_t i = 0; i < C_rows; ++i) {
        for (size_t j = 0; j < C_cols; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                float a_val = transA ? A[k * K + i] : A[i * K + k];
                float b_val = transB ? B[j * N + k] : B[k * N + j];
                sum += a_val * b_val;
            }
            C[i * C_cols + j] += alpha * sum;
        }
    }
}