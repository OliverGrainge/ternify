#include "ops/add.h"


void add(T_FP* Y, T_FP* A1, T_FP* A2) {
    assert(A1->rows == A2->rows);
    assert(A1->cols == A2->cols); 
    assert(Y->rows == A1->rows); 
    assert(Y->cols == A1->cols); 

    for (int m = 0; m < A1->rows * A1->cols; m++) {
        Y->data[m] = A1->data[m] + A2->data[m]; 
    }
}

void add(QT_S_I8_PT* Y, QT_S_I8_PT* A1, QT_S_I8_PT* A2) {
    assert(A1->rows == A2->rows);
    assert(A1->cols == A2->cols); 
    assert(Y->rows == A1->rows); 
    assert(Y->cols == A1->cols); 

    float scale = A1->scales[0] * A2->scales[0]; 
    for (int m = 0; m < A1->rows + A1->cols; m++) {
        Y->data[m] = std::round((scale * (A1->data[m] + A2->data[m])) / Y->scales[0]); 
    }
}