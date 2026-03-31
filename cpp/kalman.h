/**
 * @file kalman.h
 * @brief Constant-velocity Kalman Filter for 2D target tracking.
 *
 * Direct C++ port of radarsim/tracker/kf.py.
 *
 * State vector: [x, y, vx, vy]
 * Measurement:  [x, y]
 *
 * Uses the Bar-Shalom physically-derived process noise covariance (Q)
 * and Joseph-form covariance update for numerical stability.
 * All matrices are stack-allocated fixed-size arrays — zero heap allocation.
 */

#ifndef KALMAN_H
#define KALMAN_H

class KalmanFilter {
public:
    /**
     * Construct a constant-velocity Kalman Filter.
     *
     * @param dt  Time step duration (seconds).
     * @param q   Process noise intensity — acceleration variance (m²/s⁴).
     * @param r_x Measurement noise std dev in x (meters). Squared for R matrix.
     * @param r_y Measurement noise std dev in y (meters). Squared for R matrix.
     */
    KalmanFilter(double dt, double q, double r_x, double r_y);

    /**
     * Initialize state from the first measurement.
     * Position set to measurement, velocity set to zero.
     * Covariance: measurement noise for position, 500 for velocity (high uncertainty).
     */
    void init_state(double z_x, double z_y);

    /**
     * Prediction step: project state and covariance forward one time step.
     * x = F * x
     * P = F * P * F^T + Q
     */
    void predict();

    /**
     * Measurement update step: incorporate a new radar measurement.
     * Uses Joseph form: P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
     *
     * @param z_x Measured x position (meters).
     * @param z_y Measured y position (meters).
     */
    void update(double z_x, double z_y);

    /**
     * Full predict-then-update cycle.
     *
     * @param z_x Measured x position (meters).
     * @param z_y Measured y position (meters).
     */
    void step(double z_x, double z_y);

    // ---- Getters ----

    double get_x()  const { return x[0]; }
    double get_y()  const { return x[1]; }
    double get_vx() const { return x[2]; }
    double get_vy() const { return x[3]; }

    /**
     * Squared position error against ground truth (for RMSE accumulation).
     * Returns (x - true_x)^2 + (y - true_y)^2.
     */
    double get_position_error_sq(double true_x, double true_y) const;

private:
    double dt_;

    double x[4];       // State vector [x, y, vx, vy]
    double P[4][4];    // State covariance
    double F[4][4];    // State transition matrix
    double H[2][4];    // Measurement matrix
    double Q[4][4];    // Process noise covariance (Bar-Shalom)
    double R[2][2];    // Measurement noise covariance
};

#endif // KALMAN_H
