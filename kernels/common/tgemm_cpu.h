#ifndef TGEMM_CPU_H
#define TGEMM_CPU_H
#include <cstddef>
#include <cstdint>

// Function declaration for SGEMM
void tgemm_cpu(
    const int8_t *A, // First input matrix (M x K)
    const int8_t *B, // Second input matrix (K x N)
    int32_t *C,      // Output matrix (M x N), can be pre-allocated or an empty tensor
    size_t M,        // Number of rows in A
    size_t K,        // Number of columns in A
    size_t J,        // Number of rows in B
    size_t N         // Number of columns in B
);

#endif // TGEMM_CPU_H