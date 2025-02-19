#include "../../include/backend/common_tgemm.hpp"
#include <cstdint>
#include <cassert>
#include <omp.h>

/**
 * @brief Matrix multiplication using a packed ternary weight matrix.
 *
 * The left-hand side matrix A is stored in a packed 2-bit per weight format,
 * where every byte holds 4 weights (packed in row-major order). The ternary
 * weights are encoded as:
 *   -1 -> 0b00
 *    0 -> 0b01
 *    1 -> 0b10
 *
 * @param A_packed Pointer to the packed A matrix (each row has lda bytes).
 * @param B        Pointer to matrix B (int8_t elements).
 * @param C        Pointer to the output matrix C (int32_t elements).
 * @param M        Number of rows in matrix A.
 * @param N        Number of columns in matrix B.
 * @param K        The inner dimension (number of original weights per row of A).
 *                 Must be divisible by 4.
 * @param lda      Leading dimension (in bytes) for A_packed.
 * @param ldb      Leading dimension for B.
 * @param ldc      Leading dimension for C.
 */
void common_tgemm(const int8_t* A, const uint8_t* B_packed, int32_t* C,
                  int M, int N, int K,
                  int lda, int ldb, int ldc) {
    // Ensure that K is divisible by 4, as required by the packing scheme.
    assert((K % 4) == 0 && "K must be divisible by 4 for packed matrix A.");

    // Iterate over each row of A.
    #pragma omp parallel for collapse(2)
    for (int m = 0; m < M; ++m) {
        // Iterate over each column of B.
        for (int n = 0; n < N; ++n) {
            int32_t sum = 0;

            // shape of A is M x K
            // shape of B is K x N 
            // shape of B_packed is K x (N/4)
            // shape of C is M x N
            
            // Process each of the original K weights.
            for (int k = 0; k < K; k += 4) {
                // Compute the index of the byte in the packed A.
                uint8_t packed_byte = static_cast<uint8_t>(B_packed[m * (K/4) + (k/4)]);

                sum += static_cast<int32_t>(A[m * K + k]) * static_cast<int32_t>(static_cast<int8_t>((packed_byte >> 6) & 0x03) - 1);
                sum += static_cast<int32_t>(A[m * K + k + 1]) * static_cast<int32_t>(static_cast<int8_t>((packed_byte >> 4) & 0x03) - 1);
                sum += static_cast<int32_t>(A[m * K + k + 2]) * static_cast<int32_t>(static_cast<int8_t>((packed_byte >> 2) & 0x03) - 1);
                sum += static_cast<int32_t>(A[m * K + k + 3]) * static_cast<int32_t>(static_cast<int8_t>((packed_byte) & 0x03) - 1);
            }

            // Store the result in matrix C.
            C[m * ldc + n] = sum;
        }
    }
}
