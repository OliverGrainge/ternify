#ifndef TERNARY_COMMON_TGEMM_HPP
#define TERNARY_COMMON_TGEMM_HPP

#include <cstdint> 


/**
 * @brief Matrix multiplication using a packed ternary weight matrix.
 *
 * The right-hand side matrix B is stored in a packed 2-bit per weight format,
 * where every byte holds 4 weights (packed in row-major order). The ternary
 * weights are encoded as:
 *   -1 -> 0b00
 *    0 -> 0b01
 *    1 -> 0b10
 *
 * @param A        Pointer to matrix A (int8_t elements).
 * @param B_packed Pointer to the packed B matrix (each row has K/4 bytes).
 * @param C        Pointer to the output matrix C (int32_t elements).
 * @param M        Number of rows in matrix A.
 * @param N        Number of columns in matrix B.
 * @param K        The inner dimension (number of columns in A / rows in B).
 *                 Must be divisible by 4.
 * @param lda      Leading dimension for A.
 * @param ldb      Leading dimension for B (before packing).
 * @param ldc      Leading dimension for C.
 */
void common_tgemm(const int8_t* A, const uint8_t* B_packed, int32_t* C,
          int M, int N, int K,
          int lda, int ldb, int ldc);


#endif // TERNARY_GEMM_GEMM_OPERATOR_HPP
