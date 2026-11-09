import argparse
import csv
import math
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kalman import (
    HeadingKalmanFilter,
    MAG_EXPECTED_FIELD,
    accelerometer_roll_pitch,
    tilt_compensated_heading,
    wrap_angle,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compare recorded magnetometer and Kalman headings."
    )
    parser.add_argument(
        "filename",
        nargs="?",
        help="optional attitude capture filename",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"directory containing sensor data files (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--bias-interval",
        type=float,
        default=5.0,
        help="Seconds between printed gyro-bias estimates.",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    capture_path = Path(args.filename) if args.filename else (
        data_dir / "attitude_capture.csv"
    )
    if not capture_path.is_absolute():
        capture_path = data_dir / capture_path
    capture_path = capture_path.resolve()
    if not capture_path.is_file():
        raise FileNotFoundError(f"Attitude capture file not found: {capture_path}")

    heading_filter = HeadingKalmanFilter()
    errors = []
    previous_timestamp = None
    previous_yaw = None
    next_bias_report = 0.0

    with open(capture_path, newline="") as capture:
        rows = csv.reader(capture)
        for row in rows:
            if not row or row[0].startswith("raw_mag_heading_deg"):
                continue

            values = [float(value) for value in row]
            if len(values) >= 15:
                timestamp = values[4]
                mag_xyz = values[5:8]
                accel_xyz = values[8:11]
                gyro_z = values[11]
                roll, pitch = accelerometer_roll_pitch(accel_xyz)
                raw_heading, field_magnitude = tilt_compensated_heading(
                    mag_xyz, roll, pitch
                )
            elif len(values) == 5:
                raw_heading = math.radians(values[0])
                roll, pitch, yaw, timestamp = values[1:5]
                field_magnitude = None
                if previous_timestamp is not None:
                    dt = timestamp - previous_timestamp
                    gyro_z = wrap_angle(yaw - previous_yaw) / dt if dt > 0 else 0.0
                else:
                    gyro_z = 0.0
            else:
                raise ValueError(f"Unrecognized capture row with {len(values)} columns")

            if len(values) >= 15:
                yaw = values[3]

            if previous_timestamp is not None:
                dt = timestamp - previous_timestamp
                heading_filter.predict(gyro_z, dt)
            if field_magnitude is None:
                heading_filter.update(raw_heading, MAG_EXPECTED_FIELD)
            else:
                heading_filter.update(raw_heading, field_magnitude)

            errors.append(wrap_angle(heading_filter.heading - raw_heading))
            if timestamp >= next_bias_report:
                print(
                    f"t={timestamp:.2f}s  "
                    f"gyro bias={math.degrees(heading_filter.gyro_bias):.3f} deg/s"
                )
                next_bias_report = timestamp + max(args.bias_interval, 0.01)

            previous_timestamp = timestamp
            previous_yaw = yaw

    if not errors:
        raise ValueError(f"No capture samples found in {capture_path}")
    rms = math.degrees(math.sqrt(sum(error * error for error in errors) / len(errors)))
    print(f"RMS raw-magnetometer vs filtered heading difference: {rms:.3f} deg")


if __name__ == "__main__":
    main()
