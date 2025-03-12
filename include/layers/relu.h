#include "types/types.h"

class ReLU {
    public: 
        ReLU() {}; 
        void forward(T_FP* Y, T_FP* A); 
        void forward(QT_S_I8_PT* Y, QT_S_I8_PT* A); 
}; 