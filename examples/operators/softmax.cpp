#include "types/types.h"
#include "ops/softmax.h"
#include <cstdlib>
#include <iostream>

void fill_data(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = (float)i;
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
    int rows = 2; 
    int cols = 4; 
    // input activation data and scales
    float *A_data = new float[rows * cols]; 
    // output data and scales
    float *Y_data = new float[rows * cols]; 

    fill_data(A_data, rows * cols); 

    fill_data(Y_data, rows * cols); 

    T_FP *A = new T_FP(A_data, rows, cols); 
    T_FP *Y = new T_FP(Y_data, rows, cols); 
    
    std::cout << "Before softmax:" << std::endl;
    print_matrix(A_data, rows, cols);
    softmax(Y, A);
    std::cout << "After softmax:" << std::endl;
    print_matrix(Y_data, rows, cols);

    return 0;
}