#ifndef PACK2B_CUDA_H
#define PACK2B_CUDA_H

#include <torch/extension.h>
#include <cstdint>

// CUDA kernel declaration
__global__ void _pack2b_cuda_kernel(const int8_t *A, int64_t M, int64_t N, int8_t *B);

// Function to pack 2-bit values into an int8 tensor using CUDA
torch::Tensor pack2b_cuda(torch::Tensor A);

#endif // PACK2B_CUDA_H