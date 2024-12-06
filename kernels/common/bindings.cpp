#include <torch/extension.h>
#include "linear_forward.h"
#include "bitpack8to2.h"
#include "bitunpack8to2.h"

// Bind the linear_forward function to Python
PYBIND11_MODULE(functional, m) {
    m.def("linear_forward", &linear_forward, "Linear forward using SGEMM",
          py::arg("X"), py::arg("W"), py::arg("bias") = torch::Tensor());
    m.def("bitpack8to2_cpu", &bitpack8to2_cpu, "Pack to 2-bit (CPU)",
          py::arg("A"));
    m.def("bitunpack8to2_cpu", &bitunpack8to2_cpu, "Unpack to 8-bit (CPU)",
          py::arg("A"));
}