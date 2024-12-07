#include "tmulbtrans.h"
#include <cassert>
#include <cstring> 
#include <cstddef> 
#include <iostream>

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
    int64_t C_rows = M; 
    int64_t C_cols = J; 
    for (int64_t i = 0; i < C_rows; ++i) {
        for (int64_t j = 0; j < C_cols; ++j) {
            int32_t sum = 0;
            for (int64_t k = 0; k < K; ++k) {
                int64_t idx = j * K + k;
                int8_t val = ((B[idx/4] >> ((3 - (idx % 4)) * 2)) & 0b11) - 1;
                sum += A[i * K + k] * val;
            }
            C[i * C_cols + j] = sum;
        }
    }
}
