#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdint>

__global__ void _pack2b_cuda_kernel(const int8_t *A, int64_t M, int64_t N, int8_t *B)
{
    int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N / 4)
    {
        const int64_t base_idx = row * N + col * 4;
        int8_t packed_value = 0;

        const int8_t val0 = A[base_idx + 0] & 0x3;
        const int8_t val1 = A[base_idx + 1] & 0x3;
        const int8_t val2 = A[base_idx + 2] & 0x3;
        const int8_t val3 = A[base_idx + 3] & 0x3;

        packed_value = (val0 << 6) | (val1 << 4) | (val2 << 2) | val3;
        B[row * (N / 4) + col] = packed_value;
    }
}

torch::Tensor pack2b_cuda(torch::Tensor A)
{
    TORCH_CHECK(A.device().is_cuda(), "Tensor 'A' must be on CUDA device");
    TORCH_CHECK(A.dtype() == torch::kInt8, "Tensor 'A' must be of type int8");
    TORCH_CHECK(A.dim() == 2, "Tensor 'A' must be 2-dimensional");
    TORCH_CHECK(A.size(1) % 4 == 0, "Width must be a multiple of 4");

    A = A.contiguous(); // Ensure contiguous layout

    int64_t M = A.size(0);
    int64_t N = A.size(1);
    int64_t packed_cols = N / 4;

    auto C = torch::zeros({M, packed_cols}, torch::dtype(torch::kInt8).device(torch::kCUDA));

    dim3 blockDim(32, 32); // Adjust for optimal performance
    dim3 gridDim((packed_cols + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    _pack2b_cuda_kernel<<<gridDim, blockDim>>>(A.data_ptr<int8_t>(), M, N, C.data_ptr<int8_t>());

    cudaDeviceSynchronize(); // Ensure kernel completion

    return C;
}