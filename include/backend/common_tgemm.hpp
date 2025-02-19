#ifndef TERNARY_COMMON_TGEMM_HPP
#define TERNARY_COMMON_TGEMM_HPP

#include <cstdint> 


/**
 * @brief Performs a basic GEMM operation: C = A * B.
 *
 * @param A   Pointer to the first input matrix (size: M x K) in row-major order.
 * @param B   Pointer to the second input matrix (size: K x N) in row-major order.
 * @param C   Pointer to the output matrix (size: M x N) in row-major order.
 * @param M   Number of rows in matrix A and C.
 * @param N   Number of columns in matrix B and C.
 * @param K   Number of columns in matrix A and rows in matrix B.
 * @param lda Leading dimension (stride) for matrix A.
 * @param ldb Leading dimension (stride) for matrix B.
 * @param ldc Leading dimension (stride) for matrix C.
 */
void common_tgemm(const int8_t* A, const uint8_t* B_packed, int32_t* C,
          int M, int N, int K,
          int lda, int ldb, int ldc);


#endif // TERNARY_GEMM_GEMM_OPERATOR_HPP
