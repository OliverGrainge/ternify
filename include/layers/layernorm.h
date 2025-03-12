#include <string>

#include "types/types.h"


class LayerNorm {
    public: 
        LayerNorm(int dim, float eps=1e-5); 
        void forward(T_FP* Y, T_FP* A); 
        void forward(QT_S_I8_PT* Y, QT_S_I8_PT* A);

        void set_weight(T_FP* weight); 
        void set_bias(T_FP* bias); 

    private: 
        int dim; 
        std::string weight_type; 
        float eps;
        T_FP* weight; 
        T_FP* bias; 
}; 

