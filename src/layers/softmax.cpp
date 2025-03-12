#include "layers/softmax.h"
#include "ops/softmax.h"



void Softmax::forward(T_FP* Y, T_FP* A) {
    softmax(Y, A); 
}

void Softmax::forward(QT_S_I8_PT* Y, QT_S_I8_PT* A) {
    softmax(Y, A); 
}