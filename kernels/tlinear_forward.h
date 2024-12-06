#ifndef TLINEAR_FORWARD_H
#define TLINEAR_FORWARD_H

#include <torch/extension.h>

// Function declaration for the Linear Forward using SGEMM
torch::Tensor tlinear_forward(
    torch::Tensor X,
    torch::Tensor W_packed,
    torch::Tensor bias = torch::Tensor());

#endif // LINEAR_FORWARD_H
