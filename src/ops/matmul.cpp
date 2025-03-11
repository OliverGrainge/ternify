#include "ops/matmul.h"
#include <cassert>

void matmul(QT_S_I8_PT* A, QT_S_I8_PT* W, QT_S_I8_PT* Y) {
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
            Y->data[m * Y->cols + n] = acc * scale;
        }
    }
}