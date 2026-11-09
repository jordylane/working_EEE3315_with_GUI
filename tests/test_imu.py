import math
import unittest

from pc.imu import format_imu_reading, parse_imu_payload


class ImuTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "accelerometer_g": [0.1, -0.2, 0.9],
            "gyroscope_rad_s": [0.01, -0.02, 0.03],
            "magnetometer_uT": [12.0, -13.0, 14.0],
            "orientation_rad": {
                "roll": 0.1,
                "pitch": -0.2,
                "yaw": math.pi / 2,
            },
        }

    def test_parses_all_raw_axes_and_orientation(self):
        reading = parse_imu_payload(self.payload)

        self.assertEqual(reading.accelerometer_g, (0.1, -0.2, 0.9))
        self.assertEqual(reading.gyroscope_rad_s, (0.01, -0.02, 0.03))
        self.assertEqual(reading.magnetometer_uT, (12.0, -13.0, 14.0))
        self.assertEqual(
            reading.orientation_rpy_rad,
            (0.1, -0.2, math.pi / 2),
        )

    def test_formats_every_axis_and_orientation_for_dashboard(self):
        summary = format_imu_reading(parse_imu_payload(self.payload))

        self.assertIn("A g:    X +0.100  Y -0.200  Z +0.900", summary)
        self.assertIn("G rad/s X +0.010  Y -0.020  Z +0.030", summary)
        self.assertIn("M uT:   X +12.0  Y -13.0  Z +14.0", summary)
        self.assertIn("R/P/Y deg: +5.7 / -11.5 / +90.0", summary)

    def test_accepts_legacy_packets_without_imu_telemetry(self):
        self.assertIsNone(parse_imu_payload(None))
        self.assertEqual(
            format_imu_reading(None),
            "Waiting for IMU telemetry",
        )

    def test_formats_unavailable_gyro_axes_without_inventing_values(self):
        payload = {
            **self.payload,
            "gyroscope_rad_s": [None, None, 0.03],
        }

        reading = parse_imu_payload(payload)

        self.assertEqual(reading.gyroscope_rad_s, (None, None, 0.03))
        self.assertIn("G rad/s X --  Y --  Z +0.030", format_imu_reading(reading))

    def test_rejects_malformed_or_nonfinite_sensor_values(self):
        with self.assertRaises(ValueError):
            parse_imu_payload({**self.payload, "gyroscope_rad_s": [0.0, 1.0]})

        invalid_payload = {
            **self.payload,
            "magnetometer_uT": [0.0, math.nan, 0.0],
        }
        with self.assertRaises(ValueError):
            parse_imu_payload(invalid_payload)


if __name__ == "__main__":
    unittest.main()
