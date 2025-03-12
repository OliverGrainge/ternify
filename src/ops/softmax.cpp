#include "ops/softmax.h" 

#include <cassert>


void softmax(T_FP* Y, T_FP* A) {
    assert(Y->rows == A->rows);
    assert(Y->cols == A->cols);

    for (int m = 0; m < A->rows; m++) {
        float max = 0.0f; 
        for (int k = 0; k < A->cols; k++) {
            if (A->data[m * A->cols + k] > max) {
                max = A->data[m * A->cols + k];
            }
        }
        float exp_sum = 0.0f; 
        for (int k = 0; k < A->cols; k++) {
            exp_sum += std::exp(A->data[m * A->cols + k] - max);
        }
        for (int k = 0; k < A->cols; k++) {
            Y->data[m * Y->cols + k] = std::exp(A->data[m * A->cols + k] - max) / exp_sum;
        }
    }
}


void softmax(QT_S_I8_PT* Y, QT_S_I8_PT* A) {
    assert(Y->rows == A->rows);
    assert(Y->cols == A->cols);

    for (int m = 0; m < A->rows; m++) {
        int8_t max = 0.0f; 
        for (int k = 0; k < A->cols; k++) {
            if (A->data[m * A->cols + k] > max) {
                max = A->data[m * A->cols + k];
            }
        }
        float exp_sum = 0.0f; 
        for (int k = 0; k < A->cols; k++) {
            exp_sum += std::exp((A->data[m * A->cols + k] - max) * A->scales[0]);
        }

        for (int k = 0; k < A->cols; k++) {
            float ytmp = std::exp((A->data[m * A->cols + k] - max) * A->scales[0]) / exp_sum;
            Y->data[m * A->cols + k] = std::round(ytmp / Y->scales[0]);
        }
    }
}