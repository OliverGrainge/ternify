#include "types/types.h"
#include "ops/matmul.h"
#include <iostream>
#include <cstdlib>


void fill_data(int8_t *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = i; //(rand() % 256) - 128;  // Generate values from -128 to 127
    }
}

void fill_scales(float *scales, int n) {
    for (int i = 0; i < n; i++) {
        //scales[i] = (float)rand() / (float)RAND_MAX;
        scales[i] = 1.0f;
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
    int M = 2; 
    int N = 2; 
    int K = 2;
    int8_t *A_data = new int8_t[M * K]; 
    float *A_scales = new float[1];

    int8_t *W_data = new int8_t[N * K]; 
    float *W_scales = new float[1];

    int8_t *Y_data = new int8_t[N * M]; 
    float *Y_scales = new float[1];

    fill_data(A_data, M * K); 
    fill_scales(A_scales, 1); 

    fill_data(W_data, N * K); 
    fill_scales(W_scales, 1);

    fill_data(Y_data, M * N); 
    fill_scales(Y_scales, 1);  

    QT_S_I8_PT *A = new QT_S_I8_PT(A_data, A_scales, M, K); 
    QT_S_I8_PT *W = new QT_S_I8_PT(W_data, W_scales, N, K); 
    QT_S_I8_PT *Y = new QT_S_I8_PT(Y_data, Y_scales, M, N); 

    matmul(A, W, Y);
    std::cout << "A: " << std::endl;
    print_matrix(A_data, M, K);
    std::cout << "W: " << std::endl;
    print_matrix(W_data, N, K);
    std::cout << "Y: " << std::endl;
    print_matrix(Y_data, M, N);
    return 0;
}