/**
 * @file matrix.h
 * @brief Fixed-size matrix operations for Kalman Filter computations.
 *
 * Provides hardcoded matrix operations for 4x4, 2x4, 4x2, and 2x2 matrices.
 * No templates or generics — sizes are exactly what the constant-velocity
 * 2D Kalman Filter needs. All operations use stack-allocated C arrays.
 *
 * Design choice: fixed-size over generic to avoid heap allocation and
 * demonstrate understanding of the underlying linear algebra.
 */

#ifndef MATRIX_H
#define MATRIX_H

// ============================================================
// Matrix multiply
// ============================================================

/// C = A * B, where A, B, C are 4x4
void mat4x4_mul(const double A[4][4], const double B[4][4], double C[4][4]);

/// y = A * x, where A is 4x4, x and y are 4x1
void mat4x4_mul_vec(const double A[4][4], const double x[4], double y[4]);

/// C = A * B, where A is 2x4, B is 4x4, C is 2x4
void mat2x4_mul_4x4(const double A[2][4], const double B[4][4], double C[2][4]);

/// C = A * B, where A is 4x4, B is 4x2, C is 4x2
void mat4x4_mul_4x2(const double A[4][4], const double B[4][2], double C[4][2]);

/// y = A * x, where A is 2x4, x is 4x1, y is 2x1
void mat2x4_mul_vec(const double A[2][4], const double x[4], double y[2]);

/// C = A * B, where A is 2x4, B is 4x2, C is 2x2
void mat2x4_mul_4x2(const double A[2][4], const double B[4][2], double C[2][2]);

/// C = A * B, where A is 4x2, B is 2x4, C is 4x4
void mat4x2_mul_2x4(const double A[4][2], const double B[2][4], double C[4][4]);

/// C = A * B, where A is 4x2, B is 2x2, C is 4x2
void mat4x2_mul_2x2(const double A[4][2], const double B[2][2], double C[4][2]);

/// C = A * B, where A is 2x2, B is 2x4, C is 2x4
void mat2x2_mul_2x4(const double A[2][2], const double B[2][4], double C[2][4]);

// ============================================================
// Transpose
// ============================================================

/// AT = A^T, where A and AT are 4x4
void mat4x4_transpose(const double A[4][4], double AT[4][4]);

/// AT = A^T, where A is 2x4 and AT is 4x2
void mat2x4_transpose(const double A[2][4], double AT[4][2]);

// ============================================================
// Addition / Subtraction
// ============================================================

/// C = A + B, where A, B, C are 4x4
void mat4x4_add(const double A[4][4], const double B[4][4], double C[4][4]);

/// C = A - B, where A, B, C are 4x4
void mat4x4_sub(const double A[4][4], const double B[4][4], double C[4][4]);

/// C = A + B, where A, B, C are 2x2
void mat2x2_add(const double A[2][2], const double B[2][2], double C[2][2]);

// ============================================================
// Inverse
// ============================================================

/// Ainv = A^(-1), where A is 2x2. Uses closed-form: 1/(ad-bc) * [[d,-b],[-c,a]]
void mat2x2_inv(const double A[2][2], double Ainv[2][2]);

// ============================================================
// Utility
// ============================================================

/// Set A to 4x4 identity matrix
void mat4x4_identity(double A[4][4]);

/// Set all elements of a 4x4 matrix to zero
void mat4x4_zero(double A[4][4]);

/// Copy src to dst (4x4)
void mat4x4_copy(const double src[4][4], double dst[4][4]);

#endif // MATRIX_H
