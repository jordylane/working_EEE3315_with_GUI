import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ImuReading:
    accelerometer_g: tuple[float, float, float]
    gyroscope_rad_s: tuple[float | None, float | None, float | None]
    magnetometer_uT: tuple[float, float, float]
    orientation_rpy_rad: tuple[float, float, float]


def _vector3(value, field: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"IMU {field} must contain three values")
    vector = (
        float(value[0]),
        float(value[1]),
        float(value[2]),
    )
    if not all(math.isfinite(component) for component in vector):
        raise ValueError(f"IMU {field} values must be finite")
    return vector


def _vector3_with_missing(
    value, field: str
) -> tuple[float | None, float | None, float | None]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"IMU {field} must contain three values")
    vector = tuple(
        None if component is None else float(component)
        for component in value
    )
    if not all(
        component is None or math.isfinite(component)
        for component in vector
    ):
        raise ValueError(f"IMU {field} values must be finite when present")
    return vector


def _format_component(value, format_spec):
    return "--" if value is None else format(value, format_spec)


def parse_imu_payload(payload):
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("IMU telemetry must be an object")

    orientation = payload.get("orientation_rad")
    if not isinstance(orientation, dict):
        raise ValueError("IMU orientation_rad must be an object")
    orientation_rpy = _vector3(
        [orientation.get(axis) for axis in ("roll", "pitch", "yaw")],
        "orientation_rad",
    )

    return ImuReading(
        accelerometer_g=_vector3(payload.get("accelerometer_g"), "accelerometer_g"),
        gyroscope_rad_s=_vector3_with_missing(
            payload.get("gyroscope_rad_s"), "gyroscope_rad_s"
        ),
        magnetometer_uT=_vector3(payload.get("magnetometer_uT"), "magnetometer_uT"),
        orientation_rpy_rad=orientation_rpy,
    )


def format_imu_reading(reading):
    if reading is None:
        return "Waiting for IMU telemetry"

    ax, ay, az = reading.accelerometer_g
    gx, gy, gz = reading.gyroscope_rad_s
    mx, my, mz = reading.magnetometer_uT
    roll, pitch, yaw = map(math.degrees, reading.orientation_rpy_rad)
    return (
        f"A g:    X {_format_component(ax, '+.3f')}  "
        f"Y {_format_component(ay, '+.3f')}  "
        f"Z {_format_component(az, '+.3f')}\n"
        f"G rad/s X {_format_component(gx, '+.3f')}  "
        f"Y {_format_component(gy, '+.3f')}  "
        f"Z {_format_component(gz, '+.3f')}\n"
        f"M uT:   X {_format_component(mx, '+.1f')}  "
        f"Y {_format_component(my, '+.1f')}  "
        f"Z {_format_component(mz, '+.1f')}\n"
        f"R/P/Y deg: {roll:+.1f} / {pitch:+.1f} / {yaw:+.1f}"
    )
