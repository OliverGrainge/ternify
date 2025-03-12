#include "types/types.h"
#include "ops/gelu.h"
#include <cstdlib>
#include <iostream>

void fill_data(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = (float)rand() / (float)RAND_MAX;
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
    int rows = 6; 
    int cols = 6; 
    float *data = new float[rows * cols]; 
    float *Y_data = new float[rows * cols]; 

    fill_data(data, rows * cols); 

    T_FP *A = new T_FP(data, rows, cols); 
    T_FP *Y = new T_FP(Y_data, rows, cols); 


    std::cout << "Before GeLU:" << std::endl;
    print_matrix(data, rows, cols);
    gelu(Y, A);
    std::cout << "After GeLU:" << std::endl;
    print_matrix(data, rows, cols);

    return 0;
}