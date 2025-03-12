#include "layers/add.h"
#include "ops/add.h"

void Add::forward(T_FP* Y, T_FP* A1, T_FP* A2) {
    add(Y, A1, A2); 
}


void Add::forward(QT_S_I8_PT* Y, QT_S_I8_PT* A1, QT_S_I8_PT* A2) {
    add(Y, A1, A2); 
}