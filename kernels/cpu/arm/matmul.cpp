#include <torch/extension.h>
#include <omp.h>
#include <algorithm>
#include <arm_neon.h>

constexpr int BLOCK_SIZE = 4096;  // Adjust according to L1/L2 cache size of your CPU

// Optimized kernel for computing batched matrix multiplication with ARM SIMD acceleration
void _matmul_cpu_optimized(const float* A, const float* B, float* C, int64_t BATCH, int64_t M, int64_t N, int64_t K) {
    #pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < BATCH; ++b) {
        for (int64_t i = 0; i < M; i += BLOCK_SIZE) {
            const float* A_batch = A + b * M * N;
            float* C_batch = C + b * M * K;
            for (int64_t j = 0; j < K; j += BLOCK_SIZE) {
                for (int64_t k = 0; k < N; k += BLOCK_SIZE) {
                    // Iterate over the block
                    for (int64_t ii = i; ii < std::min(i + BLOCK_SIZE, M); ++ii) {
                        for (int64_t kk = k; kk < std::min(k + BLOCK_SIZE, N); ++kk) {
                            float A_val = A_batch[ii * N + kk];
                            for (int64_t jj = j; jj < std::min(j + BLOCK_SIZE, K); jj += 4) {
                                if (jj + 4 <= K) {
                                    float32x4_t B_val = vld1q_f32(B + kk * K + jj);
                                    float32x4_t C_val = vld1q_f32(C_batch + ii * K + jj);
                                    float32x4_t A_vec = vdupq_n_f32(A_val);
                                    C_val = vmlaq_f32(C_val, A_vec, B_val);
                                    vst1q_f32(C_batch + ii * K + jj, C_val);
                                } else {
                                    // Handle remaining elements that don't fit in a vector of 4
                                    for (int64_t jj_rem = jj; jj_rem < std::min(j + BLOCK_SIZE, K); ++jj_rem) {
                                        C_batch[ii * K + jj_rem] += A_val * B[kk * K + jj_rem];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

torch::Tensor matmul_cpu(torch::Tensor A, torch::Tensor B) {
    // Ensure the tensors are on CPU
    TORCH_CHECK(A.device().is_cpu(), "Tensor 'A' must be on CPU");
    TORCH_CHECK(B.device().is_cpu(), "Tensor 'B' must be on CPU");

    int64_t BATCH, M, N, K;
    torch::Tensor C;

    if (A.dim() == 3) {
        // Batched case
        BATCH = A.size(0);
        M = A.size(1);
        N = A.size(2);
        K = B.size(1);
        TORCH_CHECK(B.size(0) == N, "Tensor dimensions are not compatible for matrix multiplication");

        // Create an output tensor
        C = torch::zeros({BATCH, M, K}, torch::dtype(A.dtype()).device(torch::kCPU));
    } else if (A.dim() == 2) {
        // Non-batched case
        BATCH = 1;
        M = A.size(0);
        N = A.size(1);
        K = B.size(1);
        TORCH_CHECK(B.size(0) == N, "Tensor dimensions are not compatible for matrix multiplication");

        // Create an output tensor
        C = torch::zeros({M, K}, torch::dtype(A.dtype()).device(torch::kCPU));
    } else {
        TORCH_CHECK(false, "Tensor 'A' must be either 2-dimensional or 3-dimensional");
    }

    // Get pointers to the underlying data
    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Compute the matrix multiplication (optimized)
    _matmul_cpu_optimized(A_ptr, B_ptr, C_ptr, BATCH, M, N, K);

    return C;
}

// Use PyBind11 to bind the function to Python
PYBIND11_MODULE(functional, m) {
    m.def("matmul_cpu", &matmul_cpu, "Optimized Matrix Multiplication (CPU)");
}
