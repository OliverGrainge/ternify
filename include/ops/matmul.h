#include <cstdint>

#include "types/types.h"

void matmul(T_FP* Y, T_FP* A, T_FP* W, T_FP* B = nullptr);


// ============= Integer Quantized Matmul =============
void matmul(QT_S_I8_PT* Y, QT_S_I8_PT* A, QT_S_I8_PT* W, T_FP* B = nullptr);

// void matmul(QT_S_I8_PC* Y, QT_S_I8_PC* A, QT_S_I8_PC* W, T_FP* B = nullptr);

// void matmul(QT_S_I8_PG* Y, QT_S_I8_PG* A, QT_S_I8_PG* W, T_FP* B = nullptr);



// ============= Ternary Quantized Matmul =============
// void matmul(QT_S_I8_PT* Y, QT_S_I8_PT* A, QT_S_T_PT* W, T_FP* B = nullptr);

// void matmul(QT_S_I8_PC* Y, QT_S_I8_PC* A, QT_S_T_PC* W, T_FP* B = nullptr);

// void matmul(QT_S_I8_PG* Y, QT_S_I8_PG* A, QT_S_T_PG* W, T_FP* B = nullptr);






