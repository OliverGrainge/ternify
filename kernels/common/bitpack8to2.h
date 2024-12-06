#ifndef PACK2B_H
#define PACK2B_H

#include <torch/extension.h>

// Function declaration
torch::Tensor bitpack8to2_cpu(torch::Tensor A);

#endif // PACK2B_H