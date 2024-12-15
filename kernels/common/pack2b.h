#ifndef PACK2B_H
#define PACK2B_H

#include <torch/extension.h>

// Function declaration
torch::Tensor pack2b_cpu(torch::Tensor A);

#endif // PACK2B_H