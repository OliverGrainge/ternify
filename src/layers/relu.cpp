#include "layers/relu.h"
#include "ops/relu.h"



void ReLU::forward(T_FP* Y, T_FP* A) {
    relu(Y, A); 
}

void ReLU::forward(QT_S_I8_PT* Y, QT_S_I8_PT* A) {
    relu(Y, A); 
}