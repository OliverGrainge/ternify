#include "layers/gelu.h"
#include "ops/gelu.h"


void GeLU::forward(T_FP* Y, T_FP* A) {
    gelu(Y, A); 
}

void GeLU::forward(QT_S_I8_PT* Y, QT_S_I8_PT* A) {
    gelu(Y, A); 
}