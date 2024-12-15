#include "tlinear_forward.h"
#include "primitives/tgemm.h"
#include <cassert>
#include <cstring> 
#include <cstddef> 

torch::Tensor tlinear_forward(
    torch::Tensor X,
    torch::Tensor W_packed,
    torch::Tensor bias) {

    TORCH_CHECK(X.dim() == 2, "Input X must be a 2D matrix");
    TORCH_CHECK(W_packed.dim() == 2, "Weight W must be a 2D matrix");
    TORCH_CHECK(!bias.defined() || bias.dim() == 1, "Bias must be a 1D vector or None");
    TORCH_CHECK(X.size(1) == W_packed.size(1)*4, "The number of input features in X must be 4 times that in W");

    auto Y = torch::zeros({X.size(0), W_packed.size(0)}, torch::dtype(torch::kInt32).device(X.device()));

    const int8_t* X_data = X.data_ptr<int8_t>();
    const int8_t* W_data = W_packed.data_ptr<int8_t>();
    int32_t* Y_data = Y.data_ptr<int32_t>();

    // tgemm: M = X.size(0), K = X.size(1), J = W_packed.size(0), 
    // Remember W_packed.size(1) is compressed by factor of 4, but original code uses that logic in tgemm itself.
    tgemm(X_data, W_data, Y_data, X.size(0), X.size(1), W_packed.size(0), W_packed.size(1));

    if (bias.defined()) {
        // Broadcast bias across the batch dimension
        Y.add_(bias.unsqueeze(0));
    }

    return Y;
}
