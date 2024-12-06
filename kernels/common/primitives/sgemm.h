#ifndef SGEMM_H
#define SGEMM_H

#include <cstddef> // For size_t

// Function declaration for SGEMM
void sgemm(
    const float* A,          // First input matrix (M x K)
    const float* B,          // Second input matrix (K x N)
    float* C,                // Output matrix (M x N), can be pre-allocated or an empty tensor
    float alpha,             // Scalar multiplier for A * B
    float beta,              // Scalar multiplier for C (if accumulation is needed)
    bool transA,             // Whether to transpose A
    bool transB              // Whether to transpose B
);

#endif // SGEMM_H
