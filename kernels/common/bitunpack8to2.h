#ifndef UNPACK2B_H
#define UNPACK2B_H

#include <torch/extension.h>

// Function declaration
torch::Tensor bitunpack8to2_cpu(torch::Tensor A);

#endif // PACK2B_H