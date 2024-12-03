#include <torch/extension.h>
#include "pack2b.h"
#include "matmul.h"
#include "matmul2b.h"
#include "unpack2b.h"

// Bindings
PYBIND11_MODULE(functional, m) {
    m.def("matmul_cpu", &matmul_cpu, "Matrix Multiplication (CPU)",
          py::arg("A"), py::arg("B"));
    m.def("matmul2b_cpu", &matmul_cpu, "Matrix Multiplication (CPU)",
          py::arg("A"), py::arg("B"));
    m.def("pack2b_cpu", &pack2b_cpu, "Pack to 2-bit (CPU)",
          py::arg("A"));
    m.def("unpack2b_cpu", &unpack2b_cpu, "Pack to 2-bit (CPU)",
          py::arg("A"), py::arg("cols") = 0);
}

