#include <cstdint> 
#include <cmath>
#include <cassert>

#include "ops/gelu.h" 



void gelu(T_FP* Y, T_FP* A) {
    assert(Y->rows == A->rows);
    assert(Y->cols == A->cols);

    const float sqrt_2_over_pi = std::sqrt(2.0f/M_PI);
    const float coef = 0.044715f;
    
    for (int64_t i = 0; i < Y->rows * Y->cols; i++) {
        const float x = A->data[i];
        const float x_cubed = x * x * x;
        Y->data[i] = 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + coef * x_cubed)));
    }
}



void gelu(QT_S_I8_PT* Y, QT_S_I8_PT* A) {
    assert(Y->rows == A->rows);
    assert(Y->cols == A->cols);

    const float sqrt_2_over_pi = std::sqrt(2.0f/M_PI);
    const float coef = 0.044715f;
    
    for (int64_t i = 0; i < Y->rows * Y->cols; i++) {
        const float x = A->data[i] * A->scales[0];
        const float x_cubed = x * x * x;
        float ytmp = 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + coef * x_cubed)));
        Y->data[i] = std::round(ytmp/Y->scales[0]);
    }
}