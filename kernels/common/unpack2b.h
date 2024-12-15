#ifndef UNPACK2B_H
#define UNPACK2B_H

#include <torch/extension.h>

// Function declaration
torch::Tensor unpack2b_cpu(torch::Tensor A);

#endif // UNPACK2B_H