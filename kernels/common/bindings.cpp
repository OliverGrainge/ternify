#include <torch/extension.h>
#include "linear_forward.h"

// Bind the linear_forward function to Python
PYBIND11_MODULE(functional, m) {
    m.def("linear_forward", &linear_forward, "Linear forward using SGEMM",
          py::arg("X"), py::arg("W"), py::arg("bias") = torch::Tensor());
}