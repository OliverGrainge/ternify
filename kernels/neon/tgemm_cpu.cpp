#include <arm_neon.h>
#include <omp.h>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>

#include "tgemm_cpu.h"

void tgemm_cpu(
    const int8_t *A, // M x K
    const int8_t *B, // K x J (quantized, 2 bits per value)
    int32_t *C,      // M x J
    size_t M,
    size_t K,
    size_t J,
    size_t N // Not used as per your code snippet, assuming J = output cols
)
{
    assert(A != nullptr && B != nullptr && C != nullptr);

#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)M; ++i)
    {
        const int8_t *A_row = A + i * K;
        for (int64_t j = 0; j < (int64_t)J; ++j)
        {
            int32_t sum = 0;
            int64_t base_jk = j * (int64_t)K;

            int64_t k = 0;
            for (; k <= (int64_t)K - 16; k += 16)
            {
                // Load 16 values from A_row
                int8x16_t a_vec = vld1q_s8(A_row + k);

                // Decode 16 values from B on-the-fly
                int8_t b_decoded[16];
                for (int x = 0; x < 16; ++x)
                {
                    int64_t idx = base_jk + k + x;
                    int64_t b_idx = idx >> 2;
                    int64_t b_shift = (3 - (idx & 3)) * 2;
                    b_decoded[x] = ((B[b_idx] >> b_shift) & 0b11) - 1;
                }
                int8x16_t b_vec = vld1q_s8(b_decoded);

                // Multiply and accumulate
                int16x8_t prod1 = vmull_s8(vget_low_s8(a_vec), vget_low_s8(b_vec));
                int16x8_t prod2 = vmull_s8(vget_high_s8(a_vec), vget_high_s8(b_vec));

                int32x4_t sum1 = vpaddlq_s16(prod1);
                int32x4_t sum2 = vpaddlq_s16(prod2);

                sum += vaddvq_s32(sum1) + vaddvq_s32(sum2);
            }

            // Handle the remaining elements
            for (; k < (int64_t)K; ++k)
            {
                int64_t idx = base_jk + k;
                int64_t b_idx = idx >> 2;
                int64_t b_shift = (3 - (idx & 3)) * 2;
                int8_t val = ((B[b_idx] >> b_shift) & 0b11) - 1;

                sum += A_row[k] * val;
            }

            C[i * J + j] = sum;
        }
    }
}
