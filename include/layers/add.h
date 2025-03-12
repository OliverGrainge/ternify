#include "types/types.h"

class Add { 
    public: 
        Add() {}; 
        void forward(T_FP* Y, T_FP* A1, T_FP* A2); 
        void forward(QT_S_I8_PT* Y, QT_S_I8_PT* A1, QT_S_I8_PT* A2); 
};