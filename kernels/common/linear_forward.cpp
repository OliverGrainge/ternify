#include "linear_forward.h"
#include "primitives/sgemm.h"

// Implementation of the Linear Forward using SGEMM
torch::Tensor linear_forward(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor bias) {


    // Check input dimensions
    TORCH_CHECK(X.dim() == 2, "Input X must be a 2D matrix");
    TORCH_CHECK(W.dim() == 2, "Weight W must be a 2D matrix");
    TORCH_CHECK(!bias.defined() || bias.dim() == 1, "Bias must be a 1D vector or None");

    // Check for correct dimensions
    TORCH_CHECK(X.size(1) == W.size(1), "The number of input features in X must match the number of input features in W");

    // Allocate output tensor
    auto Y = torch::zeros({X.size(0), W.size(0)}, torch::TensorOptions().dtype(X.dtype()).device(X.device()));

    // Get pointers to raw data
    const float* X_data = X.data_ptr<float>();
    const float* W_data = W.data_ptr<float>();
    float* Y_data = Y.data_ptr<float>();

    // Call the SGEMM function
    sgemm(X_data, W_data, Y_data, X.size(0), X.size(1), W.size(0),W.size(1), 1.0f, 0.0f, false, true);

    // Add bias if provided
    if (bias.defined()) {
        Y.add_(bias);
    }

    return Y;
}
