import argparse
import math
import os
import socket
import struct
import time
from pathlib import Path

import numpy as np

from common import (
    MAX_PACKET_SIZE,
    PC_IP,
    SEND_RATE_HZ,
    UDP_PORT,
    anchors,
    encode_packet,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "pi"
LIDAR_PACKET_FORMAT = struct.Struct("<xBHH" + "HB" * 12 + "HHB")
LIDAR_PACKET_LENGTH = 47
LIDAR_MEASUREMENTS_PER_SCAN = 480
UDP_MAX_PAYLOAD = 65507


class CaptureTail:
    def __init__(self, path):
        self.path = Path(path)
        self.stream = None
        self.identity = None

    def read_available(self):
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            if self.stream is not None:
                self.stream.close()
                self.stream = None
                self.identity = None
            return b"", False

        identity = (stat.st_dev, stat.st_ino)
        if (
            self.stream is None
            or identity != self.identity
            or stat.st_size < self.stream.tell()
        ):
            if self.stream is not None:
                self.stream.close()
            self.stream = self.path.open("rb")
            self.stream.seek(0, os.SEEK_END)
            self.identity = identity
            return b"", True

        return self.stream.read(), False

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


class LineCaptureTail:
    def __init__(self, path):
        self.tail = CaptureTail(path)
        self.pending = bytearray()

    def read_lines(self):
        data, reset = self.tail.read_available()
        if reset:
            self.pending.clear()
        self.pending.extend(data)
        lines = self.pending.split(b"\n")
        self.pending = bytearray(lines.pop())
        return [
            line.decode("utf-8", errors="replace").strip()
            for line in lines
            if line.strip()
        ]

    def close(self):
        self.tail.close()


class LidarScanParser:
    def __init__(self):
        self.buffer = bytearray()
        self.measurements = []

    def reset(self):
        self.buffer.clear()
        self.measurements.clear()

    def feed(self, data):
        self.buffer.extend(data)
        latest_scan = None

        while True:
            header = self.buffer.find(b"\x54\x2c")
            if header < 0:
                keep_last_byte = self.buffer[-1:] == b"\x54"
                self.buffer[:] = self.buffer[-1:] if keep_last_byte else b""
                break
            if header:
                del self.buffer[:header]
            if len(self.buffer) < LIDAR_PACKET_LENGTH:
                break

            frame = bytes(self.buffer[:LIDAR_PACKET_LENGTH])
            try:
                fields = LIDAR_PACKET_FORMAT.unpack(frame)
            except struct.error:
                del self.buffer[0]
                continue
            del self.buffer[:LIDAR_PACKET_LENGTH]

            start_angle = fields[2] / 100.0
            stop_angle = fields[-3] / 100.0
            if stop_angle < start_angle:
                stop_angle += 360.0
            angle_step = (stop_angle - start_angle) / 11
            distances = fields[3:-3:2]
            confidence = fields[4:-3:2]

            for index, (distance_mm, quality) in enumerate(
                zip(distances, confidence)
            ):
                angle = math.radians(start_angle + angle_step * index)
                distance_m = distance_mm / 1000.0
                self.measurements.append(
                    (
                        math.sin(angle) * distance_m,
                        math.cos(angle) * distance_m,
                        quality,
                    )
                )

            if len(self.measurements) > LIDAR_MEASUREMENTS_PER_SCAN:
                latest_scan = self.measurements
                self.measurements = []

        return latest_scan


def trilaterate_uwb(distances):
    anchor_points = np.asarray(anchors, dtype=float)
    ranges = np.asarray(distances, dtype=float)
    first_x, first_y, first_z = anchor_points[0]
    first_range = ranges[0]
    matrix = []
    vector = []

    for index in range(1, 4):
        x, y, z = anchor_points[index]
        distance = ranges[index]
        matrix.append(
            [
                2 * (x - first_x),
                2 * (y - first_y),
                2 * (z - first_z),
            ]
        )
        vector.append(
            first_range**2
            - distance**2
            + x**2
            - first_x**2
            + y**2
            - first_y**2
            + z**2
            - first_z**2
        )

    position = np.linalg.solve(matrix, vector)
    if not np.all(np.isfinite(position)):
        raise ValueError("UWB position is not finite")
    return float(position[0]), float(position[1])


def parse_accelerometer_record(line):
    values = [float(value) for value in line.split(",")]
    if len(values) < 3 or not all(math.isfinite(value) for value in values[:3]):
        raise ValueError("invalid accelerometer capture row")
    return tuple(values[:3])


def parse_attitude_record(line):
    values = [float(value) for value in line.split(",")]
    if len(values) < 15 or not all(math.isfinite(value) for value in values[:15]):
        raise ValueError("invalid attitude capture row")
    return {
        "accelerometer_g": values[8:11],
        "gyroscope_rad_s": [None, None, values[11]],
        "magnetometer_uT": values[5:8],
        "orientation_rad": {
            "roll": values[1],
            "pitch": values[2],
            "yaw": values[3],
        },
        "heading": values[13],
        "yaw": values[3],
    }


class TelemetrySources:
    def __init__(self, data_dir):
        self.lidar_tail = CaptureTail(data_dir / "lidar_capture.bin")
        self.lidar_parser = LidarScanParser()
        self.lidar_scan = None
        self.lidar_updated_at = None

        self.uwb_tail = LineCaptureTail(data_dir / "uwb_capture.txt")
        self.distances = [None] * 4
        self.uwb_updated_at = None

        self.accelerometer_tail = LineCaptureTail(
            data_dir / "accelerometer_capture.csv"
        )
        self.accelerometer_g = None
        self.accelerometer_updated_at = None

        self.attitude_tail = LineCaptureTail(data_dir / "attitude_capture.csv")
        self.attitude = None
        self.attitude_updated_at = None

    def poll(self, now):
        lidar_bytes, reset = self.lidar_tail.read_available()
        if reset:
            self.lidar_parser.reset()
        scan = self.lidar_parser.feed(lidar_bytes)
        if scan is not None:
            self.lidar_scan = scan
            self.lidar_updated_at = now

        for line in self.uwb_tail.read_lines():
            if not line.startswith(("1> ", "2> ", "3> ", "4> ")):
                continue
            try:
                anchor_index = int(line[0]) - 1
                value = float(line[3:].split("\t", 1)[0].strip())
            except (ValueError, IndexError):
                continue
            if math.isfinite(value):
                self.distances[anchor_index] = value
                self.uwb_updated_at = now

        for line in self.accelerometer_tail.read_lines():
            try:
                self.accelerometer_g = parse_accelerometer_record(line)
            except ValueError:
                continue
            self.accelerometer_updated_at = now

        for line in self.attitude_tail.read_lines():
            if line.startswith("raw_mag_heading_deg,"):
                continue
            try:
                self.attitude = parse_attitude_record(line)
            except ValueError:
                continue
            self.attitude_updated_at = now

    def packet(self, sequence, timestamp, now, stale_after):
        if self.lidar_scan is None:
            return None, "waiting for a complete LiDAR scan"
        if any(distance is None for distance in self.distances):
            return None, "waiting for all four UWB anchor distances"
        if self.accelerometer_g is None:
            return None, "waiting for Sensor HAT accelerometer capture"
        if self.attitude is None:
            return None, "waiting for Sensor HAT attitude capture"

        sample_times = (
            self.lidar_updated_at,
            self.uwb_updated_at,
            self.accelerometer_updated_at,
            self.attitude_updated_at,
        )
        if any(
            sample_time is None or now - sample_time > stale_after
            for sample_time in sample_times
        ):
            return None, "one or more sensor capture files are stale"

        try:
            uwb_x, uwb_y = trilaterate_uwb(self.distances)
        except (np.linalg.LinAlgError, ValueError) as error:
            return None, f"cannot calculate UWB position: {error}"

        lidar_x = [point[0] for point in self.lidar_scan]
        lidar_y = [point[1] for point in self.lidar_scan]
        lidar_c = [point[2] for point in self.lidar_scan]
        imu = {
            "accelerometer_g": list(self.attitude["accelerometer_g"]),
            "gyroscope_rad_s": list(self.attitude["gyroscope_rad_s"]),
            "magnetometer_uT": list(self.attitude["magnetometer_uT"]),
            "orientation_rad": dict(self.attitude["orientation_rad"]),
        }

        return {
            "seq": sequence,
            "timestamp": timestamp,
            "lidar_x": lidar_x,
            "lidar_y": lidar_y,
            "lidar_c": lidar_c,
            "distances": list(self.distances),
            "uwb_x": uwb_x,
            "uwb_y": uwb_y,
            "ax": self.accelerometer_g[0],
            "ay": self.accelerometer_g[1],
            "az": self.accelerometer_g[2],
            "imu": imu,
            "heading": self.attitude["heading"],
            "yaw": self.attitude["yaw"],
        }, None

    def close(self):
        for tail in (
            self.lidar_tail,
            self.uwb_tail,
            self.accelerometer_tail,
            self.attitude_tail,
        ):
            tail.close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Forward live Pi sensor capture files to the PC dashboard."
    )
    parser.add_argument(
        "--pc-ip",
        default=PC_IP,
        help=f"Windows PC address reachable from the Pi (default: {PC_IP})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=(
            "directory containing the Pi program's capture files "
            f"(default: {DEFAULT_DATA_DIR})"
        ),
    )
    parser.add_argument(
        "--stale-after",
        type=float,
        default=5.0,
        help=(
            "stop forwarding if any sensor capture has not advanced "
            "for this many seconds"
        ),
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.stale_after) or args.stale_after <= 0:
        parser.error("--stale-after must be a positive finite number")

    data_dir = args.data_dir.expanduser().resolve()
    sources = TelemetrySources(data_dir)
    destination = (args.pc_ip, UDP_PORT)
    interval = 1.0 / SEND_RATE_HZ
    sequence = 0
    next_send = time.monotonic()
    waiting_reason = None
    send_error = None
    packet_count = 0
    print(
        f"Forwarding capture files from {data_dir} to "
        f"{destination[0]}:{destination[1]} at up to {SEND_RATE_HZ} Hz",
        flush=True,
    )

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
            while True:
                now = time.monotonic()
                try:
                    sources.poll(now)
                    packet, reason = sources.packet(
                        sequence,
                        time.time(),
                        now,
                        args.stale_after,
                    )
                except OSError as error:
                    packet = None
                    reason = f"capture-file error: {error}"

                if reason != waiting_reason:
                    if reason is not None:
                        print(f"Telemetry paused: {reason}", flush=True)
                    elif waiting_reason is not None:
                        print(
                            "Sensor capture data is current; sending telemetry",
                            flush=True,
                        )
                    waiting_reason = reason

                now = time.monotonic()
                if packet is not None and now >= next_send:
                    try:
                        payload = encode_packet(packet)
                        if len(payload) > min(MAX_PACKET_SIZE, UDP_MAX_PAYLOAD):
                            raise ValueError(
                                f"encoded packet is too large for UDP ({len(payload)} bytes)"
                            )
                        udp_socket.sendto(payload, destination)
                    except (OSError, ValueError, TypeError) as error:
                        if str(error) != send_error:
                            print(f"Telemetry send failed: {error}", flush=True)
                            send_error = str(error)
                    else:
                        if send_error is not None:
                            print("Telemetry transmission resumed", flush=True)
                            send_error = None
                        sequence = (sequence + 1) & 0xFFFFFFFF
                        packet_count += 1
                        if packet_count % 100 == 0:
                            print(
                                f"Sent {packet_count} packets; "
                                f"LiDAR points={len(packet['lidar_x'])} "
                                f"UWB=({packet['uwb_x']:.2f}, {packet['uwb_y']:.2f})",
                                flush=True,
                            )
                    next_send += interval
                    if next_send < now:
                        next_send = now + interval

                time.sleep(0.02)
    except KeyboardInterrupt:
        print("Stopping Pi telemetry sender.", flush=True)
    finally:
        sources.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
