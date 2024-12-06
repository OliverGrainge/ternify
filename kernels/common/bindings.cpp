#include <torch/extension.h>
#include "linear_forward.h"
#include "tlinear_forward.h"
#include "pack2b.h"
#include "unpack2b.h"

// Bind the linear_forward function to Python
PYBIND11_MODULE(functional, m) {
    m.def("linear_forward", &linear_forward, "Linear forward using SGEMM",
        py::arg("X"), py::arg("W"), py::arg("bias") = torch::Tensor());
    m.def("tlinear_forward", &tlinear_forward, "Linear forward using SGEMM",
            py::arg("X"), py::arg("W_packed"), py::arg("bias") = torch::Tensor());
    m.def("pack2b_cpu", &pack2b_cpu, "Pack to 2-bit (CPU)",
          py::arg("A"));
    m.def("unpack2b_cpu", &unpack2b_cpu, "Unpack to 8-bit (CPU)",
          py::arg("A"));
}