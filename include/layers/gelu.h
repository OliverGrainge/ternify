#include "types/types.h"

class GeLU {
    public: 
        GeLU() {}; 
        void forward(T_FP* Y, T_FP* A); 
        void forward(QT_S_I8_PT* Y, QT_S_I8_PT* A); 
}; 