#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <cstdint>


struct T_FP {
    float *data;
    int rows;
    int cols;

    T_FP(float* d, int r, int c) 
        : data(d), rows(r), cols(c) {}
};

// === Per-Tensor Quantization ===
struct QT_S_I8_PT { // Symmetric Int8 Per-Tensor
    int8_t *data;
    float *scales;
    int rows; 
    int cols;

    QT_S_I8_PT(int8_t* d, float* s, int r, int c) 
        : data(d), scales(s), rows(r), cols(c) {}
};


struct QT_S_T_PT { // Ternary Per-Tensor
    uint8_t *data; // Bit-packed ternary values using 2 bits per weight:
                   // 00 -> -1
                   // 01 ->  0
                   // 10 -> +1
                   // (4 weights per byte)
    float *scales;
    int rows; 
    int cols;

    QT_S_T_PT(uint8_t* d, float* s, int r, int c) 
        : data(d), scales(s), rows(r), cols(c) {}
};

// === Per-Channel Quantization ===
struct QT_S_I8_PC { // Symmetric Int8 Per-Channel
    int8_t *data;
    float *scales;
    int rows; 
    int cols;

    QT_S_I8_PC(int8_t* d, float* s, int r, int c) 
        : data(d), scales(s), rows(r), cols(c) {}
};


struct QT_S_T_PC { // Ternary Per-Channel
    uint8_t *data; // Bit-packed ternary values using 2 bits per weight:
                   // 00 -> -1
                   // 01 ->  0
                   // 10 -> +1
                   // (4 weights per byte)
    float *scales;
    int rows; 
    int cols; 

    QT_S_T_PC(uint8_t* d, float* s, int r, int c) 
        : data(d), scales(s), rows(r), cols(c) {}
};

// === Per-Group Quantization ===
struct QT_S_I8_PG { // Symmetric Int8 Per-Group
    int8_t *data;
    float *scales;
    int rows; 
    int cols; 
    int group_size;

    QT_S_I8_PG(int8_t* d, float* s, int r, int c, int gs) 
        : data(d), scales(s), rows(r), cols(c), group_size(gs) {}
};


struct QT_S_T_PG { // Ternary Per-Group
    uint8_t *data; // Bit-packed ternary values using 2 bits per weight:
                   // 00 -> -1
                   // 01 ->  0
                   // 10 -> +1
                   // (4 weights per byte)
    float *scales;
    int rows; 
    int cols; 
    int group_size;

    QT_S_T_PG(uint8_t* d, float* s, int r, int c, int gs) 
        : data(d), scales(s), rows(r), cols(c), group_size(gs) {}
};

void free_QT(QT_S_I8_PT *qt);
void free_QT(QT_S_T_PT *qt);
void free_QT(QT_S_I8_PC *qt);
void free_QT(QT_S_T_PC *qt);
void free_QT(QT_S_I8_PG *qt);
void free_QT(QT_S_T_PG *qt);



#endif // TYPES_H


