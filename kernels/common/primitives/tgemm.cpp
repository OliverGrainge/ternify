#ifndef SGEMM_H
#define SGEMM_H

// Function declaration for SGEMM
void sgemm(
    const float* A,          // First input matrix (M x K)
    const float* B,          // Second input matrix (K x N)
    float* C,                // Output matrix (M x N), can be pre-allocated or an empty tensor
    float alpha = 1.0f,       // Scalar multiplier for A * B
    float beta = 0.0f,        // Scalar multiplier for C (if accumulation is needed)
    bool transA = false,      // Whether to transpose A
    bool transB = false       // Whether to transpose B
);

#endif // SGEMM_H
