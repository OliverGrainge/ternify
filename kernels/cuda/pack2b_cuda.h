#ifndef PACK2B_CUDA_H
#define PACK2B_CUDA_H

#include <cstdint> // For int8_t and int64_t

extern "C" void _pack2b_cuda(const int8_t *d_A, int64_t M, int64_t N, int8_t *d_B);

#endif // PACK2B_CUDA_H