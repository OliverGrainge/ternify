#ifndef SGEMM_H
#define SGEMM_H
#include <cstddef> 

// Function declaration for SGEMM
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
    bool transA,            // Whether A is transposed
    bool transB              // Whether B is transposed
);

#endif // SGEMM_H