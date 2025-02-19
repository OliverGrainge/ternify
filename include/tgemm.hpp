#ifndef TERNARY_TGEMM_HPP
#define TERNARY_TGEMM_HPP

#include <cstdint>


/**
 * @brief Performs a ternary General Matrix Multiplication (GEMM) operation
 * 
 * Computes the matrix multiplication C = A * B where A and B are int8_t matrices
 * and C is the resulting int32_t matrix.
 * 
 * @param A Input matrix A of size M x K
 * @param B Input matrix B of size K x N
 * @param C Output matrix C of size M x N
 * @param M Number of rows in matrix A and C
 * @param N Number of columns in matrix B and C
 * @param K Number of columns in A and rows in B
 * @param lda Leading dimension of matrix A
 * @param ldb Leading dimension of matrix B
 * @param ldc Leading dimension of matrix C
 */
void tgemm(const int8_t* A, const uint8_t* B_packed, int32_t* C,
           int M, int N, int K,
           int lda, int ldb, int ldc);

#endif // TERNARY_GEMM_OPERATOR_HPP