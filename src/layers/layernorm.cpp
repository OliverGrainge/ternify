#include "layers/layernorm.h"
#include "ops/layernorm.h"
#include <cassert>

LayerNorm::LayerNorm(int dim, float eps) {
    this->dim = dim;
    this->eps = eps;
    this->weight = nullptr;
    this->bias = nullptr;
}



void LayerNorm::forward(T_FP* Y, T_FP* A) {
    layernorm(Y, A, this->weight, this->bias, this->eps); 
}

void LayerNorm::forward(QT_S_I8_PT* Y, QT_S_I8_PT* A) {
    layernorm(Y, A, this->weight, this->bias, this->eps); 
}

void LayerNorm::set_weight(T_FP* weight) {
    assert(weight->cols == this->dim); 
    assert(weight->rows == 1); 
    this->weight = weight; 
}

void LayerNorm::set_bias(T_FP* bias) {
    assert(bias->cols == this->dim); 
    assert(bias->rows == 1); 
    this->bias = bias; 
}