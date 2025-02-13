#include <gtest/gtest.h>
#include "../include/tgemm.hpp"  // Assumed to declare both tgemm() and tgemm_pack_weights()
#include "../include/tgemm_pack_weights.hpp"
#include <cstdint>
#include <vector>
#include <stdexcept>

// The test harness assumes that the raw A matrix (with ternary weights)
// is first packed via tgemm_pack_weights into a compact 2-bit-per-weight format.
// Each row of A is packed into (K / 4) bytes (since 4 weights fit in one byte).

class TGemmTest : public ::testing::Test {
protected:
    // Helper function to create a simple matrix of raw ternary weights.
    // Valid values are -1, 0, or 1.
    std::vector<int8_t> createMatrix(int rows, int cols, int8_t value) {
        return std::vector<int8_t>(rows * cols, value);
    }

    // Helper function to verify that every element in C equals the expected value.
    void verifyResult(const int32_t* C, int rows, int cols, int32_t expected) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                EXPECT_EQ(C[i * cols + j], expected)
                    << "Mismatch at position (" << i << "," << j << ")";
            }
        }
    }
};

// Test basic multiplication with ones.
// Note: We use K = 4 so that the number of columns is divisible by 4.
TEST_F(TGemmTest, BasicMultiplication) {
    const int M = 2, N = 2, K = 4;
    // Create a raw A matrix filled with 1s (ternary value 1)
    auto A_raw = createMatrix(M, K, 1);
    // Create a B matrix filled with 1s
    auto B = createMatrix(K, N, 1);
    std::vector<int32_t> C(M * N, 0);

    // Allocate buffer for packed A.
    // Each row of A packs K weights into K/4 bytes.
    std::vector<uint8_t> A_packed(M * (K / 4), 0);
    tgemm_pack_weights(A_raw.data(), A_packed.data(), M, K);

    // For the packed matrix, the leading dimension (lda) is K/4.
    tgemm(reinterpret_cast<const int8_t*>(A_packed.data()), B.data(), C.data(),
          M, N, K, K / 4, N, N);

    // Each element in C is the dot product of a row of ones with a column of ones:
    // Expected sum = K * (1 * 1) = 4.
    verifyResult(C.data(), M, N, 4);
}

// Test multiplication with zeros.
TEST_F(TGemmTest, ZeroMultiplication) {
    const int M = 3, N = 3, K = 4;
    // Create a raw A matrix filled with 0s.
    auto A_raw = createMatrix(M, K, 0);
    // Create a B matrix filled with 1s.
    auto B = createMatrix(K, N, 1);
    std::vector<int32_t> C(M * N, -1);  // Initialize with -1 to catch errors

    std::vector<uint8_t> A_packed(M * (K / 4), 0);
    tgemm_pack_weights(A_raw.data(), A_packed.data(), M, K);

    tgemm(reinterpret_cast<const int8_t*>(A_packed.data()), B.data(), C.data(),
          M, N, K, K / 4, N, N);

    verifyResult(C.data(), M, N, 0);
}

// Test invalid dimensions: Here we test that packing fails when K is not divisible by 4.
TEST_F(TGemmTest, InvalidDimensions) {
    // K not divisible by 4.
    const int M = 2, N = 2, K = 3;
    auto A_raw = createMatrix(M, K, 1);
    // Allocate a buffer that is large enough (though the packer will assert).
    std::vector<uint8_t> A_packed(M * ((K + 3) / 4), 0);
    
    EXPECT_THROW(
        tgemm_pack_weights(A_raw.data(), A_packed.data(), M, K),
        std::invalid_argument
    );
}

// Test null pointer handling: These tests expect a crash (death)
// when a null pointer is passed.
TEST_F(TGemmTest, NullPointers) {
    const int M = 2, N = 2, K = 4;
    auto A_raw = createMatrix(M, K, 1);
    auto B = createMatrix(K, N, 1);
    std::vector<int32_t> C(M * N, 0);
    std::vector<uint8_t> A_packed(M * (K / 4), 0);
    tgemm_pack_weights(A_raw.data(), A_packed.data(), M, K);

    // Passing nullptr for the packed A matrix.
    EXPECT_THROW(
        tgemm(nullptr, B.data(), C.data(), M, N, K, K / 4, N, N),
        std::invalid_argument
    );
    // Passing nullptr for B.
    EXPECT_THROW(
        tgemm(reinterpret_cast<const int8_t*>(A_packed.data()), nullptr, C.data(),
            M, N, K, K / 4, N, N),
        std::invalid_argument
    );
    // Passing nullptr for C.
    EXPECT_THROW(
        tgemm(reinterpret_cast<const int8_t*>(A_packed.data()), B.data(), nullptr,
            M, N, K, K / 4, N, N),
        std::invalid_argument
    );
}
