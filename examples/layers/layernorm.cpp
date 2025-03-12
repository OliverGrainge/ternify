#include "layers/layernorm.h" 
#include "types/types.h"
#include <iostream>

void fill_data(float *W_data, int n) {
    for (int i = 0; i < n; i++) {
        W_data[i] = (float)i; 
    }
}

void fill_weight(float *W_data, int n) {
    for (int i = 0; i < n; i++) {
        W_data[i] = 1.0f; 
    }
}


void fill_bias(float *B_data, int n) {
    for (int i = 0; i < n; i++) {
        B_data[i] = 0.0f; 
    }
}


void print_matrix(float * data, int rows, int cols) {
    // Print top border
    printf("┌");
    for (int j = 0; j < cols; j++) {
        printf("────────");  // Widened to accommodate float format
    }
    printf("┐\n");

    // Print matrix contents
    for (int i = 0; i < rows; i++) {
        printf("│");
        for (int j = 0; j < cols; j++) {
            printf(" %6.2f ", data[i * cols + j]);
        }
        printf("│\n");
    }

    // Print bottom border
    printf("└");
    for (int j = 0; j < cols; j++) {
        printf("────────");  // Widened to accommodate float format
    }
    printf("┘\n");
}


int main() {
    int dim = 6; 
    int batch_size = 2; 

    float *A_data = new float[batch_size * dim]; 
    float *weight_data = new float[dim]; 
    float *bias_data = new float[dim]; 
    float *Y_data = new float[batch_size * dim]; 

    fill_data(A_data, batch_size * dim); 
    fill_weight(weight_data, dim); 
    fill_bias(bias_data, dim); 

    T_FP* A = new T_FP(A_data, batch_size, dim); 
    T_FP* weight = new T_FP(weight_data, 1, dim); 
    T_FP* bias = new T_FP(bias_data, 1, dim); 

    T_FP* Y = new T_FP(Y_data, batch_size, dim); 

    LayerNorm* layer = new LayerNorm(dim); 
    layer->set_weight(weight); 
    layer->set_bias(bias); 

    layer->forward(Y, A); 

    std::cout << "A: " << std::endl; 
    print_matrix(A_data, batch_size, dim); 
    std::cout << "weight: " << std::endl; 
    print_matrix(weight_data, 1, dim); 
    std::cout << "bias: " << std::endl; 
    print_matrix(bias_data, 1, dim); 
    std::cout << "Y: " << std::endl; 
    print_matrix(Y_data, batch_size, dim); 

}