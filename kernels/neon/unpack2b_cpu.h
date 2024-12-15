#ifndef UNPACK2B_CPU_H
#define UNPACK2B_CPU_H

#include <stdint.h> // For standard integer types

// Function prototype for the 2-bit unpacking function
void _unpack2b_cpu(const int8_t *A, int64_t M, int64_t N, int8_t *B);

#endif // UNPACK2B_CPU_H