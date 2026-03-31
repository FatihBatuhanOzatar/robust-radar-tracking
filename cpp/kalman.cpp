/**
 * @file kalman.cpp
 * @brief Constant-velocity Kalman Filter implementation.
 *
 * Direct C++ port of radarsim/tracker/kf.py.
 * Every step of the KF equations maps to the Python version:
 *
 *   predict:  x = F @ x,  P = F @ P @ F.T + Q
 *   update:   y = z - H @ x
 *             S = H @ P @ H.T + R
 *             K = P @ H.T @ inv(S)
 *             x = x + K @ y
 *             P = (I-KH) @ P @ (I-KH).T + K @ R @ K.T   (Joseph form)
 *
 * All intermediate matrices are stack-allocated scratch arrays.
 */

#include "kalman.h"
#include "matrix.h"
#include <cstring>  // memset

KalmanFilter::KalmanFilter(double dt, double q, double r_x, double r_y)
    : dt_(dt)
{
    // Zero everything first
    memset(x, 0, sizeof(x));
    mat4x4_zero(P);
    mat4x4_zero(F);
    mat4x4_zero(Q);
    memset(H, 0, sizeof(H));
    memset(R, 0, sizeof(R));

    // State transition matrix — constant velocity model
    // [1  0  dt  0 ]     x_new  = x + vx*dt
    // [0  1  0   dt]     y_new  = y + vy*dt
    // [0  0  1   0 ]     vx_new = vx
    // [0  0  0   1 ]     vy_new = vy
    mat4x4_identity(F);
    F[0][2] = dt;
    F[1][3] = dt;

    // Measurement matrix — observe position only
    // z = H * x = [x, y]
    H[0][0] = 1.0;
    H[1][1] = 1.0;

    // Process noise covariance — physically derived from acceleration
    // uncertainty (Bar-Shalom formulation).
    //
    // Models unknown acceleration as white noise with spectral density q.
    // Q = q * [[dt^4/4,  0,       dt^3/2,  0      ],
    //          [0,       dt^4/4,  0,       dt^3/2 ],
    //          [dt^3/2,  0,       dt^2,    0      ],
    //          [0,       dt^3/2,  0,       dt^2   ]]
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    double dt4 = dt3 * dt;

    Q[0][0] = q * dt4 / 4.0;
    Q[0][2] = q * dt3 / 2.0;
    Q[1][1] = q * dt4 / 4.0;
    Q[1][3] = q * dt3 / 2.0;
    Q[2][0] = q * dt3 / 2.0;
    Q[2][2] = q * dt2;
    Q[3][1] = q * dt3 / 2.0;
    Q[3][3] = q * dt2;

    // Measurement noise covariance — R = diag(r_x^2, r_y^2)
    R[0][0] = r_x * r_x;
    R[1][1] = r_y * r_y;

    // Default covariance: large uncertainty everywhere
    mat4x4_identity(P);
    for (int i = 0; i < 4; ++i) {
        P[i][i] = 500.0;
    }
}

void KalmanFilter::init_state(double z_x, double z_y) {
    // Position from measurement, velocity unknown (zero)
    x[0] = z_x;
    x[1] = z_y;
    x[2] = 0.0;
    x[3] = 0.0;

    // Covariance: measurement noise for position, large for velocity
    mat4x4_zero(P);
    P[0][0] = R[0][0];   // position uncertainty = measurement noise variance
    P[1][1] = R[1][1];
    P[2][2] = 500.0;     // velocity uncertainty — large, unknown
    P[3][3] = 500.0;
}

void KalmanFilter::predict() {
    // x = F * x
    double x_new[4];
    mat4x4_mul_vec(F, x, x_new);
    memcpy(x, x_new, sizeof(x));

    // P = F * P * F^T + Q
    double FP[4][4];
    mat4x4_mul(F, P, FP);

    double FT[4][4];
    mat4x4_transpose(F, FT);

    double FPFT[4][4];
    mat4x4_mul(FP, FT, FPFT);

    mat4x4_add(FPFT, Q, P);
}

void KalmanFilter::update(double z_x, double z_y) {
    // Innovation (measurement residual): y = z - H * x
    double Hx[2];
    mat2x4_mul_vec(H, x, Hx);
    double y[2] = {z_x - Hx[0], z_y - Hx[1]};

    // Innovation covariance: S = H * P * H^T + R
    double HP[2][4];
    mat2x4_mul_4x4(H, P, HP);

    double HT[4][2];
    mat2x4_transpose(H, HT);

    double HPHT[2][2];
    mat2x4_mul_4x2(HP, HT, HPHT);

    double S[2][2];
    mat2x2_add(HPHT, R, S);

    // Kalman gain: K = P * H^T * inv(S)
    double PHT[4][2];
    mat4x4_mul_4x2(P, HT, PHT);

    double S_inv[2][2];
    mat2x2_inv(S, S_inv);

    double K[4][2];
    mat4x2_mul_2x2(PHT, S_inv, K);

    // State update: x = x + K * y
    for (int i = 0; i < 4; ++i) {
        x[i] += K[i][0] * y[0] + K[i][1] * y[1];
    }

    // Covariance update — Joseph form for numerical stability:
    // P = (I - K*H) * P * (I - K*H)^T + K * R * K^T

    // I_KH = I - K * H   (4x4)
    double KH[4][4];
    mat4x2_mul_2x4(K, H, KH);

    double I_KH[4][4];
    mat4x4_identity(I_KH);
    mat4x4_sub(I_KH, KH, I_KH);

    // term1 = (I-KH) * P * (I-KH)^T
    double IKH_P[4][4];
    mat4x4_mul(I_KH, P, IKH_P);

    double I_KH_T[4][4];
    mat4x4_transpose(I_KH, I_KH_T);

    double term1[4][4];
    mat4x4_mul(IKH_P, I_KH_T, term1);

    // term2 = K * R * K^T
    double KR[4][2];
    mat4x2_mul_2x2(K, R, KR);

    double KT[2][4];
    // Transpose K (4x2) -> KT (2x4): manual since we don't have mat4x2_transpose
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) {
            KT[j][i] = K[i][j];
        }
    }

    double term2[4][4];
    mat4x2_mul_2x4(KR, KT, term2);

    // P = term1 + term2
    mat4x4_add(term1, term2, P);
}

void KalmanFilter::step(double z_x, double z_y) {
    predict();
    update(z_x, z_y);
}

double KalmanFilter::get_position_error_sq(double true_x, double true_y) const {
    double dx = x[0] - true_x;
    double dy = x[1] - true_y;
    return dx * dx + dy * dy;
}
