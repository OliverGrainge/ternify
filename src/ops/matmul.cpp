#include <cassert>

#include "ops/matmul.h"


void matmul(QT_S_I8_PT* Y, QT_S_I8_PT* A, QT_S_I8_PT* W, T_FP* B) {
    assert(A->cols == W->cols); 
    assert(Y->rows == A->rows);
    assert(Y->cols == W->rows); 

    int m, n, k; 
    float scale = (A->scales[0] * W->scales[0]) / Y->scales[0];
    for (m = 0; m < A->rows; m++) {
        for (n = 0; n < W->rows; n++) {
            int32_t acc = 0;
            for (k = 0; k < A->cols; k++) {
                acc += A->data[m * A->cols + k] * W->data[n * W->cols + k];
            }
            // Add bias if provided
            float result = acc * scale;
            if (B != nullptr) {
                result += B->data[n];
            }
            Y->data[m * Y->cols + n] = result;
        }
    }
}