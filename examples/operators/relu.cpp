#include "types/types.h"
#include "ops/relu.h"
#include <cstdlib>
#include <iostream>

void fill_data(int8_t *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = (rand() % 256) - 128;  // Generate values from -128 to 127
    }
}

void fill_scales(float *scales, int n) {
    for (int i = 0; i < n; i++) {
        scales[i] = (float)rand() / (float)RAND_MAX;
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
    int rows = 6; 
    int cols = 6; 
    int8_t *data = new int8_t[rows * cols]; 
    float *scales = new float[1];

    fill_data(data, rows * cols); 
    fill_scales(scales, 1); 

    QT_S_I8_PT *A = new QT_S_I8_PT(data, scales, rows, cols); 
    std::cout << "Before ReLU:" << std::endl;
    print_matrix(data, rows, cols);
    relu(A, A);
    std::cout << "After ReLU:" << std::endl;
    print_matrix(data, rows, cols);

    return 0;
}