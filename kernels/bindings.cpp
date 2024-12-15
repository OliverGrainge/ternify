#include <torch/extension.h>
#include "tlinear_forward.h"
#include "pack2b.h"
#include "unpack2b.h"

namespace py = pybind11;

// Bind the linear_forward function to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      m.def("tlinear_forward", &tlinear_forward, "Linear forward using TGEMM",
            py::arg("X"), py::arg("W_packed"), py::arg("bias") = torch::Tensor());
      m.def("pack2b", &pack2b, "Pack to 2-bit (CPU)",
            py::arg("A"));
      m.def("unpack2b", &unpack2b, "Unpack to 8-bit (CPU)",
            py::arg("A"));
}