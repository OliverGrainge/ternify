#ifndef UNPACK2B_CUDA_H
#define UNPACK2B_CUDA_H

#include <cstdint> // For standard integer types

// Function prototype for the 2-bit unpacking function in CUDA
void _unpack2b_cuda(const int8_t *d_A, int64_t M, int64_t N, int8_t *d_B);

#endif // UNPACK2B_CUDA_H