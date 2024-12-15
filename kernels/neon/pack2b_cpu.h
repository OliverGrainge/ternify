#ifndef PACK2B_CPU_H
#define PACK2B_CPU_H

#include <stdint.h> // For standard integer types

// Function prototype for the 2-bit packing function
void _pack2b_cpu(const int8_t *A, int64_t M, int64_t N, int8_t *B);

#endif // PACK2B_CPU_H
