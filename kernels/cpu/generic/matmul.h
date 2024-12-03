#ifndef MATMUL_H
#define MATMUL_H

#include <torch/extension.h>

// Function declaration
torch::Tensor matmul_cpu(torch::Tensor A, torch::Tensor B);

#endif // MATMUL_H
