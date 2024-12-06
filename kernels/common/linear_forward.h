#ifndef LINEAR_FORWARD_H
#define LINEAR_FORWARD_H

#include <torch/extension.h>

// Function declaration for the Linear Forward using SGEMM
torch::Tensor linear_forward(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor bias = torch::Tensor());

#endif // LINEAR_FORWARD_H
