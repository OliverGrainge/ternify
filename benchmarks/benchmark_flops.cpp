#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cblas.h>

#include "../include/tgemm.hpp"
#include "../include/tgemm_pack_weights.hpp"

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

void run_benchmark_comparison(int M, int N, int K, int iterations, std::ofstream &outfile) {
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    // Allocate matrices for TGEMM
    std::vector<int8_t> A_tgemm(M * K);
    std::vector<int8_t> B_tgemm(K * N);
    std::vector<uint8_t> B_tgemm_packed(K * (N/4));
    std::vector<int32_t> C_tgemm(M * N, 0);

    // Allocate matrices for BLAS (using float for cblas_sgemm)
    std::vector<float> A_blas(M * K);
    std::vector<float> B_blas(K * N);
    std::vector<float> C_blas(M * N, 0);

    // Initialize matrices with random data
    for (int i = 0; i < M * K; i++) {
        int8_t val = static_cast<int8_t>((rand() % 256) - 128);
        A_tgemm[i] = val;
        A_blas[i] = static_cast<float>(val);
    }
    for (int i = 0; i < K * N; i++) {
        int8_t val = static_cast<int8_t>((rand() % 3) - 1);
        B_tgemm[i] = val;
        B_blas[i] = static_cast<float>(val);
    }
    tgemm_pack_weights(B_tgemm.data(), B_tgemm_packed.data(), K, N);
    // Warm-up calls
    tgemm(A_tgemm.data(), B_tgemm_packed.data(), C_tgemm.data(), M, N, K, lda, ldb, ldc);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f,
                A_blas.data(), lda,
                B_blas.data(), ldb,
                0.0f, C_blas.data(), ldc);

    // Benchmark TGEMM
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        tgemm(A_tgemm.data(), B_tgemm_packed.data(), C_tgemm.data(), M, N, K, lda, ldb, ldc);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_tgemm = end - start;
    double avg_time_tgemm = elapsed_tgemm.count() / iterations;

    // Benchmark BLAS
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f,
                    A_blas.data(), lda,
                    B_blas.data(), ldb,
                    0.0f, C_blas.data(), ldc);
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_blas = end - start;
    double avg_time_blas = elapsed_blas.count() / iterations;

    // Calculate GFLOPS
    double total_ops = 2.0 * M * N * K;
    double gflops_tgemm = (total_ops / avg_time_tgemm) / 1e9;
    double gflops_blas = (total_ops / avg_time_blas) / 1e9;

    // Write results
    outfile << get_implementation_type() << "," 
            << M << "," << N << "," << K << ","
            << "TGEMM," << avg_time_tgemm << "," << gflops_tgemm << "\n";
    outfile << get_implementation_type() << "," 
            << M << "," << N << "," << K << ","
            << "BLAS," << avg_time_blas << "," << gflops_blas << "\n";

    std::cout << "Matrix sizes: M=" << M << " N=" << N << " K=" << K << "\n"
              << "TGEMM (" << get_implementation_type() << "): "
              << avg_time_tgemm << " s, " << gflops_tgemm << " GFLOPS\n"
              << "BLAS: " << avg_time_blas << " s, " << gflops_blas << " GFLOPS\n"
              << "Speedup ratio (BLAS/TGEMM): " << avg_time_blas/avg_time_tgemm << "\n\n";
}

int main() {
    std::vector<int> sizes = {128, 256, 512, 1024};
    int iterations = 10;

    std::string impl_type = get_implementation_type();
    std::string filename = "benchmark_results_comparison_" + impl_type + ".csv";
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open output file.\n";
        return 1;
    }

    outfile << "Implementation,M,N,K,Method,AvgTime,GFLOPS\n";

    for (int size : sizes) {
        run_benchmark_comparison(size, size, size, iterations, outfile);
    }

    outfile.close();
    std::cout << "Benchmark comparison data written to " << filename << "\n";
    return 0;
}