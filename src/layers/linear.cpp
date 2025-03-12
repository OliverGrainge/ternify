#include "layers/linear.h"
#include "ops/matmul.h"
#include "types/types.h"
#include <cassert> 

Linear::Linear(int in_features, int out_features, bool bias, std::string weight_type) {
    this->in_features = in_features; 
    this->out_features = out_features; 
    this->has_bias = bias; 
    this->weight_type = weight_type; 
}

void Linear::set_weight(T_FP* weight) {
    assert(this->weight_type == "T_FP"); 
    if (this->weight != nullptr) {
        free_T((T_FP*)this->weight); 
    }
    this->weight = weight; 
}

void Linear::set_weight(QT_S_I8_PT* weight) {
    assert(this->weight_type == "QT_S_I8_PT"); 
    assert(weight->cols == this->in_features);
    assert(weight->rows == this->out_features);
    if (this->weight != nullptr) {
        free_QT((QT_S_I8_PT*)this->weight); 
    }
    this->weight = weight; 
}

    
void Linear::set_bias(T_FP* bias) {
    assert(this->has_bias); 
    assert(bias->cols == this->out_features);
    assert(bias->rows == 1);

    if (this->bias != nullptr) {
        free_T(this->bias); 
    }
    this->bias = bias; 
}

void Linear::forward(T_FP* Y, T_FP* A) {
    assert(this->weight_type == "T_FP"); 
    assert(Y->cols == this->out_features);
    assert(A->cols == this->in_features);

    if (this->has_bias) {
        matmul(Y, A, (T_FP*)this->weight); 
    } else {
        matmul(Y, A, (T_FP*)this->weight, (T_FP*)this->bias); 
    }
}


void Linear::forward(QT_S_I8_PT* Y, QT_S_I8_PT* A) {
    assert(this->weight_type == "QT_S_I8_PT"); 
    assert(Y->cols == this->out_features);
    assert(A->cols == this->in_features);

    if (this->has_bias) {
        matmul(Y, A, (QT_S_I8_PT*)this->weight); 
    } else {
        matmul(Y, A, (QT_S_I8_PT*)this->weight, (T_FP*)this->bias); 
    }
}

