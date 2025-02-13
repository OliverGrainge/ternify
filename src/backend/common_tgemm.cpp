#include "../../include/backend/common_tgemm.hpp"

#include <cstdint> 


void common_tgemm(const int8_t* A_packed, const int8_t* B, int32_t* C,
            int M, int N, int K,
            int lda, int ldb, int ldc) {
    // Iterate over each row of matrix A.
    for (int i = 0; i < M; ++i) {
        // Iterate over each column of matrix B.
        for (int j = 0; j < N; ++j) {
            int32_t sum = 0;
            // Compute the dot product of the i-th row of A and the j-th column of B.
            for (int k = 0; k < K; ++k) {
                // Access A and B using their respective leading dimensions.
                sum += static_cast<int32_t>(A_packed[i * lda + k]) *
                       static_cast<int32_t>(B[k * ldb + j]);
            }
            // Store the computed sum in matrix C.
            C[i * ldc + j] = sum;
        }
    }
}

