#include "types/types.h"
#include "ops/layernorm.h"
#include <cstdlib>
#include <iostream>

void fill_data(int8_t *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = (rand() % 256) - 128;  // Generate values from -128 to 127
    }
}


void fill_scales(float *scales, int n) {
    for (int i = 0; i < n; i++) {
        scales[i] = 1.0f;//(float)rand() / (float)RAND_MAX;
    }
}


void fill_bias(float *bias, int n) {
    for (int i = 0; i < n; i++) {
        bias[i] = 0.0f;//(float)rand() / (float)RAND_MAX;
    }
}

void print_matrix(int8_t * data, int rows, int cols) {
    // Print top border
    printf("┌");
    for (int j = 0; j < cols; j++) {
        printf("──────");
    }
    printf("┐\n");

    // Print matrix contents
    for (int i = 0; i < rows; i++) {
        printf("│");
        for (int j = 0; j < cols; j++) {
            printf(" %4d ", data[i * cols + j]);
        }
        printf("│\n");
    }

    // Print bottom border
    printf("└");
    for (int j = 0; j < cols; j++) {
        printf("──────");
    }
    printf("┘\n");
}


int main() {
    int rows = 2; 
    int cols = 6; 
    // input activation data and scales
    int8_t *A_data = new int8_t[rows * cols]; 
    float *A_scales = new float[1];
    // weight parameters 
    float *scale_data = new float[cols]; 
    float *bias_data = new float[cols]; 
    // output data 
    int8_t *Y_data = new int8_t[rows * cols]; 
    float *Y_scales = new float[1];

    fill_data(A_data, rows * cols); 
    fill_scales(A_scales, 1); 

    fill_scales(scale_data, cols); 
    fill_bias(bias_data, cols);

    fill_data(Y_data, rows * cols); 
    fill_scales(Y_scales, 1); 

    QT_S_I8_PT *A = new QT_S_I8_PT(A_data, A_scales, rows, cols); 
    T_FP *scales = new T_FP(scale_data, 1, cols); 
    T_FP *bias = new T_FP(bias_data, 1, cols); 
    QT_S_I8_PT *Y = new QT_S_I8_PT(Y_data, Y_scales, rows, cols); 
    
    std::cout << "Before layernorm:" << std::endl;
    print_matrix(A_data, rows, cols);
    layernorm(Y, A, scales, bias);
    std::cout << "After layernorm:" << std::endl;
    print_matrix(Y_data, rows, cols);

    return 0;
}