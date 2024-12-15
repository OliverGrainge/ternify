#include "pack2b_cuda.h"
#include <cuda_runtime.h>

__global__ void _pack2b_kernel(const int8_t *d_A, int64_t M, int64_t N, int8_t *d_B)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N / 4)
    {
        int64_t base_idx = row * N + (col * 4);
        int8_t packed_value = 0;
        int64_t out_idx = row * (N / 4) + col;

        int8_t val0 = d_A[base_idx + 0] & 0x3;
        int8_t val1 = d_A[base_idx + 1] & 0x3;
        int8_t val2 = d_A[base_idx + 2] & 0x3;
        int8_t val3 = d_A[base_idx + 3] & 0x3;

        packed_value = (val0 << 6) | (val1 << 4) | (val2 << 2) | val3;
        d_B[out_idx] = packed_value;
    }
}

void _pack2b_cuda(const int8_t *d_A, int64_t M, int64_t N, int8_t *d_B)
{
    dim3 blockDim(16, 16);                                                                  // Configure block size
    dim3 gridDim((N / 4 + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y); // Configure grid size

    // Launch kernel
    _pack2b_kernel<<<gridDim, blockDim>>>(d_A, M, N, d_B);
}
