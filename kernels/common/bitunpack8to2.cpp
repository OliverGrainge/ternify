#include "bitunpack8to2.h"
#include <omp.h>

void _bitunpack8to2_cpu(const int8_t* A, int64_t M, int64_t N, int8_t* B) {
    #pragma omp parallel for
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            //int64_t idx = i * (N / 4) + j / 4;  // Index in the packed tensor
            int64_t idx = (i * N + j) / 4;
            //int64_t shift = (j % 4) * 2; // Calculate the shift based on position
            int64_t shift = ((i * N + j) % 4) * 2; 
            B[i * N + j] = (A[idx] >> shift) & 0b11; // Extract 2 bits from packed value
        }
    }
}



torch::Tensor bitunpack8to2_cpu(torch::Tensor A) {
    TORCH_CHECK(A.device().is_cpu(), "Tensor 'A' must be on CPU");
    TORCH_CHECK(A.dtype() == torch::kInt8, "Tensor 'A' must be of type int8");
    TORCH_CHECK(A.dim() == 2, "Tensor 'A' must be 2-dimensional");

    int64_t M = A.size(0);
    int64_t N = A.size(1) * 4; 

    torch::Tensor B = torch::zeros({M, N}, torch::dtype(torch::kInt8).device(torch::kCPU));

    _bitunpack8to2_cpu(A.data_ptr<int8_t>(), M, N, B.data_ptr<int8_t>());
    return B;
}

