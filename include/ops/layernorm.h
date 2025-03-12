#include <cstdint>

#include "types/types.h"

void layernorm(T_FP *Y, T_FP *A, T_FP *weight = nullptr, T_FP *bias = nullptr, float eps=1e-5); 

// ============= Integer Quantized Layernorm =============
void layernorm(QT_S_I8_PT *Y, QT_S_I8_PT *A, T_FP *weight = nullptr, T_FP *bias = nullptr, float eps=1e-5); 
// void layernorm(QT_S_I8_PC *Y, QT_S_I8_PC *A, T_FP *weight = nullptr, T_FP *bias = nullptr, float eps=1e-5); 
// void layernorm(QT_S_I8_PG *Y, QT_S_I8_PG *A, T_FP *weight = nullptr, T_FP *bias = nullptr, float eps=1e-5); 


// ============= Ternary Quantized Layernorm =============
// void layernorm(QT_S_T_PT *A); 
// void layernorm(QT_S_T_PC *A); 
// void layernorm(QT_S_T_PG *A); 
