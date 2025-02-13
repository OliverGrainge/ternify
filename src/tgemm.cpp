// File: src/gemm_operator.cpp

#include "../include/backend/common_tgemm.hpp"
#include <stdexcept>
#include <cstdint> 



// The high-level GEMM operator which optionally fuses bias and requantization.
// It dispatches to the appropriate backend if available.
void tgemm(const int8_t* A_packed, const int8_t* B, int32_t* C,
          int M, int N, int K,
          int lda, int ldb, int ldc) {

    if (!A_packed || !B || !C) {
        throw std::invalid_argument("Input matrices cannot be null");
    }
    
    if (M <= 0 || N <= 0 || K <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    
    // Example dispatching based on compile-time macros.
    // (In a more advanced implementation, you might detect CPU capabilities at runtime.)
#ifdef USE_AVX2
    throw std::runtime_error("AVX2 GEMM implementation not yet available");
#endif
#ifdef USE_NEON
    throw std::runtime_error("NEON GEMM implementation not yet available");
#else
    // Fallback: a simple reference implementation
    common_tgemm(A_packed, B, C, M, N, K, lda, ldb, ldc);
#endif
}

