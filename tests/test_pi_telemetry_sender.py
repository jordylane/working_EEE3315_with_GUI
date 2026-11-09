import tempfile
import unittest
from pathlib import Path

import numpy as np

from common import anchors, decode_packet, encode_packet
from pc.imu import parse_imu_payload
from pi_telemetry_sender import (
    CaptureTail,
    LIDAR_PACKET_FORMAT,
    LidarScanParser,
    TelemetrySources,
    parse_accelerometer_record,
    parse_attitude_record,
    trilaterate_uwb,
)


def make_lidar_frame(start_angle):
    fields = [0x2C, 3600, start_angle * 100]
    for _ in range(12):
        fields.extend((1000, 200))
    fields.extend(((start_angle + 11) * 100, 1, 0))
    frame = bytearray(LIDAR_PACKET_FORMAT.pack(*fields))
    frame[0] = 0x54
    return bytes(frame)


class CaptureTailTests(unittest.TestCase):
    def test_tails_new_data_without_replaying_existing_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.csv"
            path.write_bytes(b"old row\n")
            tail = CaptureTail(path)
            try:
                self.assertEqual(tail.read_available(), (b"", True))
                with path.open("ab") as capture:
                    capture.write(b"new row\n")
                self.assertEqual(tail.read_available(), (b"new row\n", False))
            finally:
                tail.close()


class LidarScanParserTests(unittest.TestCase):
    def test_parses_frames_across_reads_and_converts_mm_to_metres(self):
        parser = LidarScanParser()
        frame = make_lidar_frame(0)
        self.assertIsNone(parser.feed(frame[:9]))
        self.assertIsNone(parser.feed(frame[9:]))

        scan = None
        for start_angle in range(12, 492, 12):
            scan = parser.feed(make_lidar_frame(start_angle))

        self.assertIsNotNone(scan)
        self.assertEqual(len(scan), 492)
        self.assertAlmostEqual(scan[0][0], 0.0)
        self.assertAlmostEqual(scan[0][1], 1.0)
        self.assertEqual(scan[0][2], 200)


class CaptureParsingTests(unittest.TestCase):
    def test_parses_accelerometer_and_attitude_rows_with_missing_gyro_axes(self):
        self.assertEqual(
            parse_accelerometer_record("0.1,-0.2,0.9,10.0"),
            (0.1, -0.2, 0.9),
        )
        attitude = parse_attitude_record(
            "1,0.1,-0.2,0.3,1,12,13,14,0.1,-0.2,0.9,0.04,2,3,0"
        )
        self.assertEqual(attitude["gyroscope_rad_s"], [None, None, 0.04])
        self.assertEqual(attitude["heading"], 3.0)
        self.assertEqual(attitude["yaw"], 0.3)

    def test_reuses_pi_anchor_geometry_for_uwb_position(self):
        expected = np.array([1.0, 2.0, 0.8])
        distances = np.linalg.norm(np.asarray(anchors) - expected, axis=1)

        actual = trilaterate_uwb(distances)

        self.assertAlmostEqual(actual[0], expected[0])
        self.assertAlmostEqual(actual[1], expected[1])

    def test_encoded_telemetry_keeps_null_gyro_values_as_null(self):
        packet = decode_packet(
            encode_packet(
                {
                    "imu": {
                        "gyroscope_rad_s": [None, None, 0.04],
                    }
                }
            )
        )
        self.assertEqual(packet["imu"]["gyroscope_rad_s"], [None, None, 0.04])

    def test_packet_maps_captures_to_the_pc_telemetry_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = TelemetrySources(Path(directory))
            sources.lidar_scan = [(1.0, 2.0, 200)]
            sources.lidar_updated_at = 10.0
            sources.distances = [1.0, 2.0, 3.0, 4.0]
            sources.uwb_updated_at = 10.0
            sources.accelerometer_g = (0.1, -0.2, 0.9)
            sources.accelerometer_updated_at = 10.0
            sources.attitude = parse_attitude_record(
                "1,0.1,-0.2,0.3,1,12,13,14,0.1,-0.2,0.9,0.04,2,3,0"
            )
            sources.attitude_updated_at = 10.0
            try:
                packet, reason = sources.packet(9, 100.0, 10.1, 5.0)
                self.assertIsNone(reason)
                decoded = decode_packet(encode_packet(packet))
                imu = parse_imu_payload(decoded["imu"])
            finally:
                sources.close()

        self.assertEqual(decoded["seq"], 9)
        self.assertEqual(decoded["lidar_x"], [1.0])
        self.assertEqual(decoded["lidar_y"], [2.0])
        self.assertEqual(decoded["lidar_c"], [200])
        self.assertEqual(decoded["distances"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(decoded["ax"], 0.1)
        self.assertEqual(decoded["az"], 0.9)
        self.assertAlmostEqual(decoded["heading"], 3.0)
        self.assertAlmostEqual(decoded["yaw"], 0.3)
        self.assertEqual(imu.gyroscope_rad_s, (None, None, 0.04))


if __name__ == "__main__":
    unittest.main()
