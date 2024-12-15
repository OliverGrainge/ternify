#include <omp.h>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>

#include "tgemm.h"

void tgemm(
    const int8_t* A,  // M x K
    const int8_t* B,  // K x J
    int32_t* C,       // M x J
    size_t M, 
    size_t K, 
    size_t J, 
    size_t N   // Not used
) {
    assert(A != nullptr && B != nullptr && C != nullptr);

    // Initialize C to zero first
    std::memset(C, 0, M * J * sizeof(int32_t));

    // Add thread safety checks
    #pragma omp parallel for schedule(static) if(M >= 32)
    for (int64_t i = 0; i < (int64_t)M; ++i) {
        const int8_t* A_row = A + i * K;
        int32_t* C_row = C + i * J;  // Pre-compute C row pointer
        
        for (int64_t j = 0; j < (int64_t)J; ++j) {
            int32_t sum = 0;
            int64_t base_jk = j * (int64_t)K;

            // Unroll the inner loop by 4 for better performance
            int64_t k = 0;
            for (; k < ((int64_t)K - 3); k += 4) {
                int64_t idx = base_jk + k;
                
                // Process 4 elements at once
                for (int offset = 0; offset < 4; offset++) {
                    int64_t curr_idx = idx + offset;
                    int64_t b_idx = curr_idx >> 2;
                    int64_t b_shift = (3 - (curr_idx & 3)) * 2;
                    int8_t val = ((B[b_idx] >> b_shift) & 0b11) - 1;
                    sum += A_row[k + offset] * val;
                }
            }

            // Handle remaining elements
            for (; k < (int64_t)K; ++k) {
                int64_t idx = base_jk + k;
                int64_t b_idx = idx >> 2;
                int64_t b_shift = (3 - (idx & 3)) * 2;
                int8_t val = ((B[b_idx] >> b_shift) & 0b11) - 1;
                sum += A_row[k] * val;
            }

            C_row[j] = sum;  // Use pre-computed row pointer
        }
    }
}
