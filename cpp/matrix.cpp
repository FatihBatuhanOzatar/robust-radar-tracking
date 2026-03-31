/**
 * @file matrix.cpp
 * @brief Fixed-size matrix operations for Kalman Filter computations.
 *
 * All operations are straightforward triple-nested loops or closed-form
 * formulas. No heap allocation — everything is stack arrays.
 */

#include "matrix.h"
#include <cstring>  // memset, memcpy
#include <cstdlib>  // abort
#include <cstdio>   // fprintf
#include <cmath>    // fabs

// ============================================================
// Matrix multiply
// ============================================================

void mat4x4_mul(const double A[4][4], const double B[4][4], double C[4][4]) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 4; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

void mat4x4_mul_vec(const double A[4][4], const double x[4], double y[4]) {
    for (int i = 0; i < 4; ++i) {
        double sum = 0.0;
        for (int k = 0; k < 4; ++k) {
            sum += A[i][k] * x[k];
        }
        y[i] = sum;
    }
}

void mat2x4_mul_4x4(const double A[2][4], const double B[4][4], double C[2][4]) {
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 4; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

void mat4x4_mul_4x2(const double A[4][4], const double B[4][2], double C[4][2]) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 4; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

void mat2x4_mul_vec(const double A[2][4], const double x[4], double y[2]) {
    for (int i = 0; i < 2; ++i) {
        double sum = 0.0;
        for (int k = 0; k < 4; ++k) {
            sum += A[i][k] * x[k];
        }
        y[i] = sum;
    }
}

void mat2x4_mul_4x2(const double A[2][4], const double B[4][2], double C[2][2]) {
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 4; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

void mat4x2_mul_2x4(const double A[4][2], const double B[2][4], double C[4][4]) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 2; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

void mat4x2_mul_2x2(const double A[4][2], const double B[2][2], double C[4][2]) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 2; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

void mat2x2_mul_2x4(const double A[2][2], const double B[2][4], double C[2][4]) {
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 2; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// ============================================================
// Transpose
// ============================================================

void mat4x4_transpose(const double A[4][4], double AT[4][4]) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            AT[j][i] = A[i][j];
        }
    }
}

void mat2x4_transpose(const double A[2][4], double AT[4][2]) {
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            AT[j][i] = A[i][j];
        }
    }
}

// ============================================================
// Addition / Subtraction
// ============================================================

void mat4x4_add(const double A[4][4], const double B[4][4], double C[4][4]) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

void mat4x4_sub(const double A[4][4], const double B[4][4], double C[4][4]) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}

void mat2x2_add(const double A[2][2], const double B[2][2], double C[2][2]) {
    C[0][0] = A[0][0] + B[0][0];
    C[0][1] = A[0][1] + B[0][1];
    C[1][0] = A[1][0] + B[1][0];
    C[1][1] = A[1][1] + B[1][1];
}

// ============================================================
// Inverse (2x2 only — closed-form)
// ============================================================

void mat2x2_inv(const double A[2][2], double Ainv[2][2]) {
    // For 2x2 matrix [[a, b], [c, d]]:
    // inv = 1/(ad - bc) * [[d, -b], [-c, a]]
    double det = A[0][0] * A[1][1] - A[0][1] * A[1][0];

    if (fabs(det) < 1e-15) {
        fprintf(stderr, "ERROR: 2x2 matrix is singular (det = %.2e)\n", det);
        abort();
    }

    double inv_det = 1.0 / det;
    Ainv[0][0] =  A[1][1] * inv_det;
    Ainv[0][1] = -A[0][1] * inv_det;
    Ainv[1][0] = -A[1][0] * inv_det;
    Ainv[1][1] =  A[0][0] * inv_det;
}

// ============================================================
// Utility
// ============================================================

void mat4x4_identity(double A[4][4]) {
    memset(A, 0, 16 * sizeof(double));
    A[0][0] = 1.0;
    A[1][1] = 1.0;
    A[2][2] = 1.0;
    A[3][3] = 1.0;
}

void mat4x4_zero(double A[4][4]) {
    memset(A, 0, 16 * sizeof(double));
}

void mat4x4_copy(const double src[4][4], double dst[4][4]) {
    memcpy(dst, src, 16 * sizeof(double));
}
