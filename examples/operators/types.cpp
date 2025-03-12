#include "types/types.h"
#include <cstdlib>


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


int main() {
    int rows = 128; 
    int cols = 256; 
    int8_t *data = new int8_t[rows * cols]; 
    float *scales = new float[rows];

    fill_data(data, rows * cols); 
    fill_scales(scales, rows); 

    QT_S_I8_PC A = QT_S_I8_PC(data, scales, rows, cols); 

    return 0;
}