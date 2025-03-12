#include <cassert>
#include <algorithm>
#include "ops/relu.h"

void relu(T_FP* Y, T_FP* A) {
    assert(Y->rows == A->rows); 
    assert(Y->cols == A->cols);

    for (int i = 0; i < Y->rows * Y->cols; i++) {
        Y->data[i] = std::max(0.0f, A->data[i]);
    }
}

void relu(QT_S_I8_PT* Y, QT_S_I8_PT* A) {
    assert(Y->rows == A->rows); 
    assert(Y->cols == A->cols); 

    for (int i = 0; i < Y->rows * Y->cols; i++) {
        Y->data[i] = std::max<int8_t>(0, A->data[i]);
    }
}

