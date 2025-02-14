#ifndef TERNARY_TGEMM_PACK_WEIGHTS_HPP
#define TERNARY_TGEMM_PACK_WEIGHTS_HPP

#include <cstdint>

/**
 * @brief Packs ternary weights into an optimized format for tgemm.
 *
 * This function converts the input weight matrix into a packed format that is
 * better suited for the tgemm operation. The input weights are assumed to be in
 * row-major order and consist of ternary values (typically -1, 0, and 1).
 * The caller must ensure that the output buffer, `dst_packed`, is allocated
 * with sufficient size.
 *
 * @param src_unpacked Pointer to the original weight matrix.
 * @param dst_packed Pointer to the buffer where the packed weights will be stored.
 * @param rows Number of rows in the weight matrix.
 * @param cols Number of columns in the weight matrix.
 */
void tgemm_pack_weights(const int8_t* src_unpacked, uint8_t* dst_packed, int rows, int cols);

#endif // TERNARY_TGEMM_PACK_WEIGHTS_HPP