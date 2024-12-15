#include <omp.h>
#include "unpack2b_cpu.h"
#include <iostream>

void _unpack2b_cpu(const int8_t *A, int64_t M, int64_t N, int8_t *B)
{
    // N is the full count after unpacking (A has N/4 columns)
    int64_t packed_cols = (N + 3) / 4;

#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < M; ++i)
    {
        const int8_t *A_row = A + i * packed_cols;
        int8_t *B_row = B + i * N;

        for (int64_t j = 0; j < N; ++j)
        {
            int64_t b_idx = j >> 2;            // j/4
            int64_t shift = (3 - (j & 3)) * 2; // (j%4)
            B_row[j] = (A_row[b_idx] >> shift) & 0b11;
        }
    }
}
