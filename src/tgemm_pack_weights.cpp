#include <cstdint>
#include <stdexcept>
#include <cassert>

/**
 * @brief Packs a matrix of ternary weights into a 2-bit per weight format.
 *
 * The input matrix is assumed to be stored in row-major order and contains
 * only ternary weights (-1, 0, or 1), which are mapped as follows:
 *   -1 -> 00
 *    0 -> 01
 *    1 -> 10
 *
 * The number of columns must be divisible by 4, since four 2-bit weights pack into one byte.
 *
 * @param src_unpacked Pointer to the input weight matrix.
 * @param dst_packed Pointer to the output buffer where packed weights will be stored.
 *                       The buffer should have at least rows * (cols / 4) bytes.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix (must be divisible by 4).
 */
void tgemm_pack_weights(const int8_t* src_unpacked, uint8_t* dst_packed, int rows, int cols) {
    // Ensure that the number of columns is divisible by 4.
    if (cols % 4 != 0) {
        throw std::invalid_argument("Number of columns must be divisible by 4.");
    }

    int packed_index = 0;

    // Process each row.
    for (int row = 0; row < rows; ++row) {
        // Process each group of 4 weights (each group forms one byte).
        for (int col = 0; col < cols; col += 4) {
            uint8_t byte_val = 0;
            for (int i = 0; i < 4; ++i) {
                int index = row * cols + col + i;
                int8_t w = src_unpacked[index];
                uint8_t two_bit = 0;

                // Map the ternary weight to its 2-bit representation.
                if (w == -1) {
                    two_bit = 0b00;
                } else if (w == 0) {
                    two_bit = 0b01;
                } else if (w == 1) {
                    two_bit = 0b10;
                } else {
                    throw std::invalid_argument("Input weight must be -1, 0, or 1");
                }

                // Place the 2-bit weight in the correct position in the byte,
                // starting from the most significant bits
                byte_val |= (two_bit << (6 - i * 2));
            }
            // Write the packed byte to the output buffer.
            dst_packed[packed_index++] = byte_val;
        }
    }
}
