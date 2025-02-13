#include "../../include/backend/common_tgemm.hpp"
#include <cstdint>
#include <cassert>

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
void common_tgemm(const uint8_t* A_packed, const int8_t* B, int32_t* C,
                  int M, int N, int K,
                  int lda, int ldb, int ldc) {
    // Ensure that K is divisible by 4, as required by the packing scheme.
    assert((K % 4) == 0 && "K must be divisible by 4 for packed matrix A.");

    // Iterate over each row of A.
    for (int i = 0; i < M; ++i) {
        // Iterate over each column of B.
        for (int j = 0; j < N; ++j) {
            int32_t sum = 0;
            // Process each of the original K weights.
            for (int k = 0; k < K; ++k) {
                // Compute the index of the byte in the packed A.
                int byte_index = i * lda + (k / 4);
                uint8_t packed_byte = static_cast<uint8_t>(A_packed[byte_index]);
                // Determine the bit shift needed to extract the 2-bit weight.
                int shift = 6 - 2 * (k % 4);
                uint8_t two_bit = (packed_byte >> shift) & 0x03;

                // Decode the 2-bit weight to its actual ternary value.
                int8_t a_val;
                if (two_bit == 0b00) {
                    a_val = -1;
                } else if (two_bit == 0b01) {
                    a_val = 0;
                } else if (two_bit == 0b10) {
                    a_val = 1;
                } else {
                    // This should not happen if A_packed was produced by the packer.
                    assert(false && "Invalid 2-bit weight encountered in A_packed.");
                    a_val = 0; // To silence compiler warnings.
                }

                // Multiply the unpacked A value with the corresponding B element.
                int8_t b_val = B[k * ldb + j];
                sum += static_cast<int32_t>(a_val) * static_cast<int32_t>(b_val);
            }
            // Store the result in matrix C.
            C[i * ldc + j] = sum;
        }
    }
}
