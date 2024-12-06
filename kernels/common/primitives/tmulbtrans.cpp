#include "tmulbtrans.h"
#include <cassert>
#include <cstring> 
#include <cstddef> 

void tmulbtrans(
    const int8_t* A,          // First input matrix (M x K)
    const int8_t* B,          // Second input matrix (K x N)
    int32_t* C,                // Output matrix (M x N), can be pre-allocated or an empty tensor
    size_t M,   // Number of rows in A
    size_t K,   // Number of columns in A   
    size_t J,   // Number of rows in B
    size_t N   // Number of columns in B
) {
    assert(A != nullptr && B != nullptr && C != nullptr);

    // Perform the matrix multiplication
    size_t C_rows = M; 
    size_t C_cols = J;

    for (size_t i = 0; i < C_rows; ++i) {
        for (size_t j = 0; j < C_cols; ++j) {
            int32_t sum = 0;
            for (size_t k = 0; k < J; ++k) {
                int idx = j * N + k;
                int shift = (idx % 4) * 2;
                sum += A[i * K + k] * ((B[idx] >> shift) & 0b11);
            }
            C[i * C_cols + j] = sum;
        }
    }
}