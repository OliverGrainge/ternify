#include <cassert>
#include <algorithm>
#include "ops/relu.h"

void relu(T_FP* Y) {
    for (int i = 0; i < Y->rows * Y->cols; i++) {
        Y->data[i] = std::max(0.0f, Y->data[i]);
    }
}

void relu(QT_S_I8_PT* Y) {
    for (int i = 0; i < Y->rows * Y->cols; i++) {
        Y->data[i] = std::max<int8_t>(0, Y->data[i]);
    }
}

