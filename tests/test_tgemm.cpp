#include <gtest/gtest.h>
#include "../include/tgemm.hpp"
#include <cstdint>
#include <vector>
#include <stdexcept>

class TGemmTest : public ::testing::Test {
protected:
    // Helper function to create a simple matrix
    std::vector<int8_t> createMatrix(int rows, int cols, int8_t value) {
        return std::vector<int8_t>(rows * cols, value);
    }

    // Helper function to verify if all elements match expected value
    void verifyResult(const int32_t* C, int rows, int cols, int32_t expected) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                EXPECT_EQ(C[i * cols + j], expected) 
                    << "Mismatch at position (" << i << "," << j << ")";
            }
        }
    }
};

// Test basic multiplication with ones
TEST_F(TGemmTest, BasicMultiplication) {
    const int M = 2, N = 2, K = 2;
    auto A = createMatrix(M, K, 1);
    auto B = createMatrix(K, N, 1);
    std::vector<int32_t> C(M * N, 0);

    // Each output element should be K * (1 * 1) = 2
    tgemm(A.data(), B.data(), C.data(), M, N, K, K, N, N);
    verifyResult(C.data(), M, N, 2);
}

// Test multiplication with zeros
TEST_F(TGemmTest, ZeroMultiplication) {
    const int M = 3, N = 3, K = 3;
    auto A = createMatrix(M, K, 0);
    auto B = createMatrix(K, N, 1);
    std::vector<int32_t> C(M * N, -1);  // Initialize with -1 to catch errors

    tgemm(A.data(), B.data(), C.data(), M, N, K, K, N, N);
    verifyResult(C.data(), M, N, 0);
}

// Test invalid dimensions
TEST_F(TGemmTest, InvalidDimensions) {
    const int M = 0, N = 2, K = 2;
    auto A = createMatrix(M, K, 1);
    auto B = createMatrix(K, N, 1);
    std::vector<int32_t> C(M * N, 0);

    EXPECT_THROW(
        tgemm(A.data(), B.data(), C.data(), M, N, K, K, N, N),
        std::invalid_argument
    );
}

// Test null pointer handling
TEST_F(TGemmTest, NullPointers) {
    const int M = 2, N = 2, K = 2;
    auto A = createMatrix(M, K, 1);
    auto B = createMatrix(K, N, 1);
    std::vector<int32_t> C(M * N, 0);

    EXPECT_THROW(
        tgemm(nullptr, B.data(), C.data(), M, N, K, K, N, N),
        std::invalid_argument
    );
}
