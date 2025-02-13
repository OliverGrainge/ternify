#include <gtest/gtest.h>
#include "../include/tgemm_pack_weights.hpp"  // Assumes tgemm_pack_weights is declared here.
#include <cstdint>
#include <vector>

// Test fixture for the packing function.
class TGemmPackWeightsTest : public ::testing::Test {
protected:
    // Helper to create a matrix (vector) of given dimensions and fill it with a constant value.
    std::vector<int8_t> createMatrix(int rows, int cols, int8_t value) {
        return std::vector<int8_t>(rows * cols, value);
    }
};

// Test a single row with exactly 4 weights.
TEST_F(TGemmPackWeightsTest, SingleRowPack) {
    const int rows = 1;
    const int cols = 4;  // Must be divisible by 4.
    // Create a row with specific ternary weights: -1, 0, 1, 0.
    std::vector<int8_t> input_weights = { -1, 0, 1, 0 };

    // Expected mapping:
    //   -1 -> 0b00,  0 -> 0b01,  1 -> 0b10,  0 -> 0b01.
    // Packed into a byte: 00 (bits 7-6), 01 (bits 5-4), 10 (bits 3-2), 01 (bits 1-0)
    // That is: 0b00011001 (or 0x19 in hex).
    const uint8_t expected = 0b00011001;

    std::vector<uint8_t> packed_weights(rows * (cols / 4), 0);
    tgemm_pack_weights(input_weights.data(), packed_weights.data(), rows, cols);

    ASSERT_EQ(packed_weights.size(), 1);
    EXPECT_EQ(packed_weights[0], expected);
}

// Test multiple rows and multiple groups per row.
TEST_F(TGemmPackWeightsTest, MultiRowPack) {
    const int rows = 2;
    const int cols = 8;  // Must be divisible by 4.
    // Define a 2x8 matrix with a mix of weights.
    // Row 0: [-1, 0, 1, 0, 1, 1, -1, 0]
    // Row 1: [ 1, 1, 1, 1, 0, 0, 0, 0]
    std::vector<int8_t> input_weights = {
         -1,  0,  1,  0,  1,  1, -1,  0,
          1,  1,  1,  1,  0,  0,  0,  0
    };

    // Expected packed output:
    // Row 0:
    //   Byte 0 (first 4 weights): -1->00, 0->01, 1->10, 0->01 = 0b00011001 (0x19)
    //   Byte 1 (next 4 weights): 1->10, 1->10, -1->00, 0->01 = 0b10100001 (0xA1)
    // Row 1:
    //   Byte 2 (first 4 weights): 1->10, 1->10, 1->10, 1->10 = 0b10101010 (0xAA)
    //   Byte 3 (next 4 weights): 0->01, 0->01, 0->01, 0->01 = 0b01010101 (0x55)
    std::vector<uint8_t> expected = { 0x19, 0xA1, 0xAA, 0x55 };

    std::vector<uint8_t> packed_weights(rows * (cols / 4), 0);
    tgemm_pack_weights(input_weights.data(), packed_weights.data(), rows, cols);

    ASSERT_EQ(packed_weights.size(), expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
        EXPECT_EQ(packed_weights[i], expected[i])
            << "Mismatch at index " << i;
    }
}

// Death test: columns not divisible by 4 should trigger an assertion.
TEST_F(TGemmPackWeightsTest, ColumnsNotDivisibleBy4) {
    const int rows = 1;
    const int cols = 5;  // Not divisible by 4.
    std::vector<int8_t> input_weights = createMatrix(rows, cols, 1);
    // Allocate enough memory (the function expects rows * (cols/4) bytes,
    // but since cols is invalid here, we don't care about the size).
    std::vector<uint8_t> packed_weights(rows * ((cols + 3) / 4), 0);

    // Using EXPECT_DEATH to catch the assertion failure.
    EXPECT_THROW(
        tgemm_pack_weights(input_weights.data(), packed_weights.data(), rows, cols),
        std::invalid_argument
    );
}

// Death test: invalid weight value (not -1, 0, or 1) should trigger an assertion.
TEST_F(TGemmPackWeightsTest, InvalidWeightValue) {
    const int rows = 1;
    const int cols = 4;
    // Introduce an invalid value (e.g., 2) among valid weights.
    std::vector<int8_t> input_weights = { -1, 0, 2, 0 };
    std::vector<uint8_t> packed_weights(rows * (cols / 4), 0);

    EXPECT_THROW(
        tgemm_pack_weights(input_weights.data(), packed_weights.data(), rows, cols),
        std::invalid_argument
    );
}
