#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// CUDA Kernel for computing batched matrix multiplication (naive_matmul)
__global__ void matmul_kernel(const float* A, const float* B, float* C, int64_t BATCH, int64_t M, int64_t N, int64_t K) {
    int batch_idx = blockIdx.z;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx < BATCH && row < M && col < K) {
        const float* A_batch = A + batch_idx * M * N;
        const float* B_batch = B;
        float* C_batch = C + batch_idx * M * K;

        float sum = 0.0f;
        for (int64_t k = 0; k < N; ++k) {
            sum += A_batch[row * N + k] * B_batch[k * K + col];
        }
        C_batch[row * K + col] = sum;
    }
}

torch::Tensor matmul_gpu(torch::Tensor A, torch::Tensor B) {
    // Ensure the tensors are on CUDA
    TORCH_CHECK(A.device().is_cuda(), "Tensor 'A' must be on CUDA");
    TORCH_CHECK(B.device().is_cuda(), "Tensor 'B' must be on CUDA");

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
        C = torch::zeros({BATCH, M, K}, torch::dtype(A.dtype()).device(torch::kCUDA));
    } else if (A.dim() == 2) {
        // Non-batched case
        BATCH = 1;
        M = A.size(0);
        N = A.size(1);
        K = B.size(1);
        TORCH_CHECK(B.size(0) == N, "Tensor dimensions are not compatible for matrix multiplication");

        // Create an output tensor
        C = torch::zeros({M, K}, torch::dtype(A.dtype()).device(torch::kCUDA));
    } else {
        TORCH_CHECK(false, "Tensor 'A' must be either 2-dimensional or 3-dimensional");
    }

    // Define block and grid sizes
    dim3 block_size(16, 16);
    dim3 grid_size((M + block_size.x - 1) / block_size.x, (K + block_size.y - 1) / block_size.y, BATCH);

    // Get pointers to the underlying data
    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Launch the CUDA kernel
    matmul_kernel<<<grid_size, block_size>>>(A_ptr, B_ptr, C_ptr, BATCH, M, N, K);
    cudaDeviceSynchronize();

    return C;
}

// Use PyBind11 to bind the function to Python
PYBIND11_MODULE(functional, m) {
    m.def("matmul_gpu", &matmul_gpu, "Naive Matrix Multiplication (GPU)");
}
