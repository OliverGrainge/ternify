#include "ops/layernorm.h"
#include <cmath>  // for std::pow
#include <cassert>

void layernorm(T_FP *Y, T_FP *A, T_FP *scale, T_FP *bias, float eps) {
    assert(A->rows == Y->rows);
    assert(A->cols == Y->cols);
    assert(scale->rows == A->cols);
    assert(bias->rows == A->cols);

    for (int m = 0; m < A->rows; m++) {
        float sum = 0.0f;
        float mean = 0.0f;
        float var = 0.0f;

        // First pass: calculate mean
        for (int k = 0; k < A->cols; k++) {
            sum += A->data[m * A->cols + k];
        }
        mean = sum / (float)A->cols;

        // Second pass: calculate variance
        for (int k = 0; k < A->cols; k++) {
            float diff = A->data[m * A->cols + k] - mean;
            var += diff * diff;  // More efficient than std::pow
        }
        var = var / (float)A->cols;  // Divide by N before sqrt
        float inv_std = 1.0f / std::sqrt(var + eps);

        // Third pass: normalize and apply scale/bias
        for (int k = 0; k < A->cols; k++) {
            Y->data[m * A->cols + k] = (A->data[m * A->cols + k] - mean) * inv_std;
            if (scale != nullptr) {
                Y->data[m * A->cols + k] *= scale->data[k];
            }
            if (bias != nullptr) {
                Y->data[m * A->cols + k] += bias->data[k];
            }
        }
    }
}