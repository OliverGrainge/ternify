import torch 
import os 
import sys 
import time
# Add the path to the shared object files to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import torch
import time
import matplotlib.pyplot as plt
import ternify.tnn.functional as TF
# Define a range of matrix sizes
matrix_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

torch_times = []
naive_times = []

for size in matrix_sizes:
    A = torch.randn(size, size).contiguous()
    B = torch.randn(size, size).contiguous()

    # Benchmark torch.matmul
    start_time = time.time()
    exp_output = torch.matmul(A, B)
    torch_time = time.time() - start_time
    torch_times.append(torch_time)

    # Benchmark TF.matmul_cpu
    start_time = time.time()
    output = TF.matmul_cpu(A, B)
    naive_time = time.time() - start_time
    naive_times.append(naive_time)

    # Ensure outputs are close
    assert torch.allclose(exp_output, output, atol=1e-3), "Outputs differ!"

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot([size for size in matrix_sizes], torch_times, label='torch.matmul', marker='o')
plt.plot([size for size in matrix_sizes], naive_times, label='TF.matmul_cpu', marker='o')
plt.xlabel('Matrix Size (N x N)')
plt.ylabel('Execution Time (seconds)')
plt.title('Benchmarking torch.matmul vs TF.matmul_cpu')
plt.legend()
plt.grid(True)
plt.show()