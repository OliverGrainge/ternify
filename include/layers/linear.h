#include <string>

#include "types/types.h"


class Linear {
    public: 
        Linear(int in_features, int out_features, bool bias=true, std::string weight_type="T_FP"); 
        void forward(T_FP* Y, T_FP* A); 
        void forward(QT_S_I8_PT* Y, QT_S_I8_PT* A);

        void set_weight(T_FP* weight); 
        void set_weight(QT_S_I8_PT* weight); 
        void set_bias(T_FP* bias); 

    private: 
        int in_features; 
        int out_features; 
        bool has_bias; 
        std::string weight_type; 
        void* weight = nullptr; 
        T_FP* bias = nullptr; 
}; 


