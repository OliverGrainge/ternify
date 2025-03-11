#include <cstdint>

#include "types/types.h"

void layernorm(T_FP *Y, T_FP *A, T_FP *scale = nullptr, T_FP *bias = nullptr, float eps=1e-5); 
void layernorm(QT_S_I8_PT *Y, QT_S_I8_PT *A, T_FP *scale = nullptr, T_FP *bias = nullptr, float eps=1e-5); 
void layernorm(QT_S_I8_PT *Y, QT_S_I8_PC *A, T_FP *scale = nullptr, T_FP *bias = nullptr, float eps=1e-5); 
void layernorm(QT_S_I8_PT *Y, QT_S_I8_PG *A, T_FP *scale = nullptr, T_FP *bias = nullptr, float eps=1e-5); 

// void layernorm(QT_S_T_PT *A); 
// void layernorm(QT_S_T_PC *A); 
// void layernorm(QT_S_T_PG *A); 
