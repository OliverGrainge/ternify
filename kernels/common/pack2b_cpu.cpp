#include "pack2b_cpu.h"
#include <omp.h>

void _pack2b_cpu(const int8_t *A, int64_t M, int64_t N, int8_t *B)
{
    int64_t packed_cols = N / 4;

#pragma omp parallel for schedule(static) if (M >= 32)
    for (int64_t i = 0; i < M; i++)
    {
        for (int64_t j = 0; j < packed_cols; j++)
        {
            const int64_t base_idx = i * N + (j * 4);
            int8_t packed_value = 0;
            const int64_t out_idx = i * packed_cols + j;

            const int8_t val0 = A[base_idx + 0] & 0x3;
            const int8_t val1 = A[base_idx + 1] & 0x3;
            const int8_t val2 = A[base_idx + 2] & 0x3;
            const int8_t val3 = A[base_idx + 3] & 0x3;

            packed_value = (val0 << 6) | (val1 << 4) | (val2 << 2) | val3;
            B[out_idx] = packed_value;
        }
    }
}
