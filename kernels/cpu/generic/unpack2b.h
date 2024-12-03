#ifndef UNPACK2B_H
#define UNPACK2B_H

#include <torch/extension.h>

// Function declaration
torch::Tensor unpack2b_cpu(torch::Tensor A, int64_t cols = 0);

#endif // PACK2B_H