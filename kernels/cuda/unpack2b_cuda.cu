#include "unpack2b_cuda.h"
#include <cuda_runtime.h>
#include <cstdio>

__global__ void _unpack2b_kernel(const int8_t *d_A, int64_t M, int64_t N, int8_t *d_B)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // N is the total number of columns after unpacking
    int64_t packed_cols = (N + 3) / 4;

    if (row < M && col < N)
    {
        // Calculate the index of the packed value
        int64_t b_idx = col >> 2;            // col / 4
        int64_t shift = (3 - (col & 3)) * 2; // (col % 4)

        // Each element in A_row contains 4 2-bit values
        const int8_t *A_row = d_A + row * packed_cols;

        // Unpack the 2-bit value
        d_B[row * N + col] = (A_row[b_idx] >> shift) & 0b11;
    }
}

void _unpack2b_cuda(const int8_t *d_A, int64_t M, int64_t N, int8_t *d_B)
{
    // Define CUDA thread block and grid dimensions
    dim3 blockDim(16, 16); // 16x16 threads per block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    _unpack2b_kernel<<<gridDim, blockDim>>>(d_A, M, N, d_B);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}
