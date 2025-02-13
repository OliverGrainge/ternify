
#ifndef TERNARY_TGEMM_PACK_WEIGHTS_HPP
#define TERNARY_TGEMM_PACK_WEIGHTS_HPP

#include <cstdint>

/**
 * @brief Packs ternary weights into an optimized format for tgemm.
 *
 * This function converts the input weight matrix into a packed format that is
 * better suited for the tgemm operation. The input weights are assumed to be in
 * row-major order and consist of ternary values (typically -1, 0, and 1).
 * The caller must ensure that the output buffer, `packed_weights`, is allocated
 * with sufficient size.
 *
 * @param input_weights Pointer to the original weight matrix.
 * @param packed_weights Pointer to the buffer where the packed weights will be stored.
 * @param rows Number of rows in the weight matrix.
 * @param cols Number of columns in the weight matrix.
 */
void tgemm_pack_weights(const int8_t* input_weights, uint8_t* packed_weights, int rows, int cols);
