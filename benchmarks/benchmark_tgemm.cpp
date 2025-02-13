// File: benchmarks/benchmark_gemm.cpp

#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdlib>

// Include your GEMM operator header.
#include "../include/tgemm.hpp"

// Add this function near the top of the file, after the includes
std::string get_implementation_type() {
    std::string impl = "common";
    #ifdef USE_NEON
        impl = "neon";
    #endif
    #ifdef USE_AVX
        impl = "avx";
    #endif
    return impl;
}

// This function runs the benchmark for one matrix size.
// It allocates matrices of size M x K, K x N, runs the GEMM operator repeatedly,
// and writes out the average latency and GFLOPS to the provided output stream.
void run_benchmark(int M, int N, int K, int iterations, std::ofstream &outfile) {
    // For simplicity, we assume row-major storage:
    const int lda = K; // A is M x K
    const int ldb = N; // B is K x N
    const int ldc = N; // C is M x N

    // Allocate matrices.
    std::vector<int8_t> A(M * K);
    std::vector<int8_t> B(K * N);
    std::vector<int32_t> C(M * N, 0);

    // Initialize matrices A and B with random data.
    for (auto &val : A) {
        val = static_cast<int8_t>((rand() % 256) - 128);
    }
    for (auto &val : B) {
        val = static_cast<int8_t>((rand() % 256) - 128);
    }

    // Warm-up call to help with caching effects.
    tgemm(A.data(), B.data(), C.data(), M, N, K, lda, ldb, ldc);

    // Start timing over a number of iterations.
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        tgemm(A.data(), B.data(), C.data(), M, N, K, lda, ldb, ldc);
    }
    auto end = std::chrono::high_resolution_clock::now();

    // Compute elapsed time in seconds.
    std::chrono::duration<double> elapsed = end - start;
    double avg_time = elapsed.count() / iterations;

    // Compute the number of operations (assume 2 per inner-loop iteration: multiply and add).
    double total_ops = 2.0 * M * N * K;
    double gflops = (total_ops / avg_time) / 1e9;

    // Modify the CSV output line to include the implementation type
    outfile << get_implementation_type() << "," 
            << M << "," << N << "," << K << "," << avg_time << "," << gflops << "\n";
    std::cout << "Implementation: " << get_implementation_type()
              << " | Matrix sizes: M=" << M << " N=" << N << " K=" << K 
              << " | Avg Time: " << avg_time << " s, GFLOPS: " << gflops << "\n";
}

int main() {
    std::vector<int> sizes = {128, 256, 512, 1024};
    int iterations = 10;

    // Get the implementation type for the filename
    std::string impl_type = get_implementation_type();
    
    // Modify the output filename to include the implementation type
    std::string filename = "benchmark_results_" + impl_type + ".csv";
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open output file.\n";
        return 1;
    }
    
    // Add implementation type to the CSV header
    outfile << "Implementation,M,N,K,AvgTime,GFLOPS\n";

    // Loop over each size and run the benchmark.
    for (int size : sizes) {
        run_benchmark(size, size, size, iterations, outfile);
    }

    outfile.close();
    std::cout << "Benchmark data written to " << filename << "\n";
    return 0;
}
