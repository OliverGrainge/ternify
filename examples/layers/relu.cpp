#include "layers/relu.h" 
#include "types/types.h"
#include <iostream>

void fill_data(float *W_data, int n) {
    for (int i = 0; i < n; i++) {
        W_data[i] = (float)(rand() % 100) - 50.0f; 
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
    int in_features = 2; 
    int out_features = 4; 
    int batch_size = 2; 

    float *A_data = new float[out_features * in_features]; 
    float *Y_data = new float[out_features * in_features]; 
    fill_data(A_data, out_features * in_features); 
    
    T_FP* A = new T_FP(A_data, out_features, in_features); 
    T_FP* Y = new T_FP(Y_data, out_features, in_features); 

    ReLU* layer = new ReLU(); 

    layer->forward(Y, A); 

    std::cout << "A: " << std::endl; 
    print_matrix(A_data, out_features, in_features); 
    std::cout << "Y: " << std::endl; 
    print_matrix(Y_data, out_features, in_features); 

    return 0; 
}