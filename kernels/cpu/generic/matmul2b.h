#ifndef MATMUL2B_H
#define MATMUL2B_H

#include <torch/extension.h>

// Function declaration
torch::Tensor matmul2b_cpu(torch::Tensor A, torch::Tensor B);

#endif // PACK2B_H