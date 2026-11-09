import math

import numpy as np


# Continuous-time process-noise spectral densities:
# gyro white noise: rad^2/s; gyro-bias random walk: rad^2/s^3.
Q_GYRO_NOISE = math.radians(0.5) ** 2
Q_BIAS_RANDOM_WALK = math.radians(0.02) ** 2
# Magnetometer heading measurement variance: rad^2.
R_MAGNETOMETER = math.radians(8.0) ** 2

# Calibration placeholders. Set the expected field strength to the local,
# calibrated magnitude in the units returned by Sense HAT (usually microtesla).
MAG_HARD_IRON_OFFSET = np.array([0.0, 0.0, 0.0])
MAG_SOFT_IRON_MATRIX = np.eye(3)
MAG_EXPECTED_FIELD = 50.0
MAG_FIELD_TOLERANCE = 0.30  # Fractional deviation from expected field magnitude.

MAX_PREDICT_DT = 0.5  # seconds; larger timestamp gaps are skipped.


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def accelerometer_roll_pitch(accel_xyz):
    ax, ay, az = accel_xyz
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.hypot(ay, az))
    return roll, pitch


def tilt_compensated_heading(mag_xyz, roll: float, pitch: float):
    calibrated = MAG_SOFT_IRON_MATRIX @ (
        np.asarray(mag_xyz, dtype=float) - MAG_HARD_IRON_OFFSET
    )
    mx, my, mz = calibrated
    horizontal_x = mx * math.cos(pitch) + mz * math.sin(pitch)
    horizontal_y = (
        mx * math.sin(roll) * math.sin(pitch)
        + my * math.cos(roll)
        - mz * math.sin(roll) * math.cos(pitch)
    )
    heading = math.atan2(horizontal_y, horizontal_x)
    return wrap_angle(heading), float(np.linalg.norm(calibrated))


class HeadingKalmanFilter:
    """Two-state heading/bias filter; angles are radians and rates rad/s."""

    def __init__(self) -> None:
        self.x = np.array([0.0, 0.0])
        # P = cov([heading, gyro_z bias]); initial diagonal variances are rad^2
        # and (rad/s)^2 respectively.
        self.P = np.diag([math.pi**2, math.radians(20.0) ** 2])
        self.initialized = False

    @property
    def heading(self) -> float:
        return float(self.x[0])

    @property
    def gyro_bias(self) -> float:
        return float(self.x[1])

    def predict(self, gyro_z: float, dt: float) -> bool:
        if not math.isfinite(dt) or dt <= 0.0 or dt > MAX_PREDICT_DT:
            return False
        if not math.isfinite(gyro_z):
            return False

        if not self.initialized:
            return False

        transition = np.array([[1.0, -dt], [0.0, 1.0]])
        dt2 = dt * dt
        dt3 = dt2 * dt
        process_noise = np.array([
            [
                Q_GYRO_NOISE * dt + Q_BIAS_RANDOM_WALK * dt3 / 3.0,
                -Q_BIAS_RANDOM_WALK * dt2 / 2.0,
            ],
            [
                -Q_BIAS_RANDOM_WALK * dt2 / 2.0,
                Q_BIAS_RANDOM_WALK * dt,
            ],
        ])

        self.x[0] = wrap_angle(self.x[0] + (gyro_z - self.x[1]) * dt)
        self.P = transition @ self.P @ transition.T + process_noise
        return True

    def update(self, magnetometer_heading: float, field_magnitude: float) -> bool:
        if not math.isfinite(magnetometer_heading) or not math.isfinite(field_magnitude):
            return False
        if MAG_EXPECTED_FIELD <= 0.0 or abs(
            field_magnitude - MAG_EXPECTED_FIELD
        ) > MAG_FIELD_TOLERANCE * MAG_EXPECTED_FIELD:
            return False

        measurement = wrap_angle(magnetometer_heading)
        if not self.initialized:
            self.x[0] = measurement
            self.P[0, 0] = R_MAGNETOMETER
            self.initialized = True
            return True

        # H = [1, 0]; K = P H^T (H P H^T + R)^-1
        # = P[:,0] / (P[0,0] + R).
        innovation = wrap_angle(measurement - self.x[0])
        gain = self.P[:, 0] / (self.P[0, 0] + R_MAGNETOMETER)
        self.x += gain * innovation
        self.x[0] = wrap_angle(self.x[0])

        # Joseph form preserves a symmetric, non-negative covariance P.
        observation = np.array([[1.0, 0.0]])
        residual = np.eye(2) - np.outer(gain, observation[0])
        self.P = (
            residual @ self.P @ residual.T
            + R_MAGNETOMETER * np.outer(gain, gain)
        )
        self.P = (self.P + self.P.T) / 2.0
        return True
