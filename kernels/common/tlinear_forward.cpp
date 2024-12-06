#include "tlinear_forward.h"
#include "primitives/tmulbtrans.h"
#include <cassert>
#include <cstring> 
#include <cstddef> 

// Implementation of the Linear Forward using SGEMM
torch::Tensor tlinear_forward(
    torch::Tensor X,
    torch::Tensor W_packed,
    torch::Tensor bias) {


    // Check input dimensions
    TORCH_CHECK(X.dim() == 2, "Input X must be a 2D matrix");
    TORCH_CHECK(W_packed.dim() == 2, "Weight W must be a 2D matrix");
    TORCH_CHECK(!bias.defined() || bias.dim() == 1, "Bias must be a 1D vector or None");

    // Check for correct dimensions
    TORCH_CHECK(X.size(1) == W_packed.size(1)*4, "The number of input features in X must be 4 times layer thatn input features in W");

    // Allocate output tensor
    auto Y = torch::zeros({X.size(0), W_packed.size(0)}, torch::TensorOptions().dtype(torch::kInt32).device(X.device()));

    // Get pointers to raw data
    const int8_t* X_data = X.data_ptr<int8_t>();
    const int8_t* W_data = W_packed.data_ptr<int8_t>();
    int32_t* Y_data = Y.data_ptr<int32_t>();

    // Call the tmulbtrans function
    tmulbtrans(X_data, W_data, Y_data, X.size(0), X.size(1), W_packed.size(0), W_packed.size(1));

    // Add bias if provided
    if (bias.defined()) {
        Y.add_(bias);
    }

    return Y;
}
