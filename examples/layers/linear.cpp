#include "layers/linear.h" 
#include "types/types.h"

void fill_weights(float *W_data, int n) {
    for (int i = 0; i < n; i++) {
        W_data[i] = (float)i; 
    }
}

void fill_bias(float *B_data, int n) {
    for (int i = 0; i < n; i++) {
        B_data[i] = 0.0f; 
    }
}


int main() {
    int in_features = 6; 
    int out_features = 4; 
    int batch_size = 2; 

    float *W_data = new float[out_features * in_features]; 
    float *B_data = new float[out_features]; 
    fill_weights(W_data, out_features * in_features); 
    fill_bias(B_data, out_features); 

    T_FP* W = new T_FP(W_data, out_features, in_features); 
    T_FP* B = new T_FP(B_data, 1, out_features); 

    Linear* layer = new Linear(in_features, out_features, true, "T_FP"); 
    layer->set_weight(W); 
    layer->set_bias(B); 


    // float *A_data = new float[batch_size * in_features]; 
    // fill_weights(A_data, batch_size * in_features); 

    //float *Y_data = new float[batch_size * out_features]; 

    //T_FP* X = new T_FP(A_data, batch_size, in_features); 
    //T_FP* Y = new T_FP(A_data, batch_size, out_features); 


    
}