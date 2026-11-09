import struct
import unittest

from common import (
    ACK_MAGIC,
    CONTROL_PACKET_FORMAT,
    HELLO_MAGIC,
    control_destinations,
    decode_packet,
    encode_packet,
    is_ack_packet,
    is_hello_packet,
    make_ack_packet,
    make_hello_packet,
    fragment_telemetry_packet,
    mix,
    TelemetryPacketReassembler,
    unpack_control_packet,
)


class MixTests(unittest.TestCase):
    def test_all_nine_key_combinations(self):
        cases = {
            frozenset(): (0.0, 0.0),
            frozenset({"up"}): (1.0, 1.0),
            frozenset({"down"}): (-1.0, -1.0),
            frozenset({"left"}): (-1.0, 1.0),
            frozenset({"right"}): (1.0, -1.0),
            frozenset({"up", "left"}): (0.0, 1.0),
            frozenset({"up", "right"}): (1.0, 0.0),
            frozenset({"down", "left"}): (-1.0, 0.0),
            frozenset({"down", "right"}): (0.0, -1.0),
        }

        for keys, expected in cases.items():
            with self.subTest(keys=keys):
                self.assertEqual(mix(keys), expected)

    def test_auxiliary_commands_do_not_change_drive_mixing(self):
        self.assertEqual(mix({"up", "grabber_up"}), (1.0, 1.0))


class ControlPacketTests(unittest.TestCase):
    def test_unpack_control_packet(self):
        packet = struct.pack(CONTROL_PACKET_FORMAT, 42, 12.5, 0x05)
        self.assertEqual(
            unpack_control_packet(packet),
            (42, 12.5, {"up", "left"}),
        )

    def test_unpack_control_packet_with_grabber_and_claw_commands(self):
        packet = struct.pack(CONTROL_PACKET_FORMAT, 43, 12.6, 0xF0)
        self.assertEqual(
            unpack_control_packet(packet),
            (
                43,
                12.6,
                {"grabber_up", "grabber_down", "claw_open", "claw_close"},
            ),
        )

    def test_rejects_malformed_packet(self):
        with self.assertRaises(ValueError):
            unpack_control_packet(b"short")

    def test_control_destinations_include_sensor_and_monitor_udp_ports(self):
        self.assertEqual(
            control_destinations("192.0.2.10"),
            (("192.0.2.10", 5006), ("192.0.2.10", 5007)),
        )

    def test_hello_and_ack_packets_are_distinct_from_control_packets(self):
        hello = make_hello_packet()
        ack = make_ack_packet()
        control = struct.pack(CONTROL_PACKET_FORMAT, 1, 1.0, 0)

        self.assertEqual(hello, HELLO_MAGIC + b"CONNECTION OK")
        self.assertEqual(ack, ACK_MAGIC + b"CONNECTION ACK")
        self.assertTrue(is_hello_packet(hello))
        self.assertTrue(is_ack_packet(ack))
        self.assertFalse(is_hello_packet(ack))
        self.assertFalse(is_hello_packet(control))
        self.assertFalse(is_ack_packet(hello))
        self.assertFalse(is_ack_packet(control))
        with self.assertRaises(ValueError):
            unpack_control_packet(hello)


class TelemetryFragmentTests(unittest.TestCase):
    def test_reassembles_packet_from_out_of_order_fragments(self):
        payload = b'{"lidar_x":[' + b"1.234," * 700 + b"0]}"
        fragments = fragment_telemetry_packet(payload, 123)
        self.assertTrue(all(len(fragment) <= 1200 for fragment in fragments))

        reassembler = TelemetryPacketReassembler()
        result = None
        for fragment in reversed(fragments):
            result = reassembler.add(fragment, now=1.0)

        self.assertEqual(result, payload)

    def test_round_trips_encoded_lidar_telemetry(self):
        source = {
            "seq": 91,
            "lidar_x": [index / 1000 for index in range(1200)],
            "lidar_y": [index / -1000 for index in range(1200)],
            "lidar_c": [200] * 1200,
        }
        fragments = fragment_telemetry_packet(encode_packet(source), source["seq"])
        reassembler = TelemetryPacketReassembler()
        payload = None
        for fragment in fragments:
            payload = reassembler.add(fragment)

        self.assertIsNotNone(payload)
        self.assertEqual(decode_packet(payload), source)

    def test_plain_json_packet_passes_through_unchanged(self):
        packet = b'{"seq":1}'
        self.assertEqual(TelemetryPacketReassembler().add(packet), packet)

    def test_fragment_streams_from_different_senders_do_not_mix(self):
        first = fragment_telemetry_packet(b"first" * 300, 11)
        second = fragment_telemetry_packet(b"other" * 300, 11)
        reassembler = TelemetryPacketReassembler()

        self.assertIsNone(reassembler.add(first[0], now=1.0, source_key=("a", 1)))
        self.assertIsNone(reassembler.add(second[0], now=1.0, source_key=("b", 1)))
        first_result = second_result = None
        for fragment in first[1:]:
            first_result = reassembler.add(
                fragment,
                now=1.0,
                source_key=("a", 1),
            )
        for fragment in second[1:]:
            second_result = reassembler.add(
                fragment,
                now=1.0,
                source_key=("b", 1),
            )

        self.assertEqual(first_result, b"first" * 300)
        self.assertEqual(second_result, b"other" * 300)

    def test_incomplete_packet_expires(self):
        fragments = fragment_telemetry_packet(b"x" * 2000, 7)
        reassembler = TelemetryPacketReassembler(timeout_s=0.5)

        self.assertIsNone(reassembler.add(fragments[0], now=1.0))
        self.assertIsNone(reassembler.add(fragments[1], now=2.0))
        self.assertEqual(len(reassembler._assemblies), 1)

    def test_rejects_malformed_fragment(self):
        with self.assertRaises(ValueError):
            TelemetryPacketReassembler().add(b"\x00UWF")

    def test_fragmenter_rejects_datagrams_larger_than_safe_limit(self):
        with self.assertRaises(ValueError):
            fragment_telemetry_packet(b"data", 1, datagram_size=1300)


if __name__ == "__main__":
    unittest.main()
