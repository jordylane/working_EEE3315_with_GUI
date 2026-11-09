import unittest

from common import make_ack_packet, make_hello_packet
from pc.control_connection import (
    CONNECTION_ATTEMPTS,
    CONNECTION_HEARTBEAT_SECONDS,
    CONNECTION_RETRY_SECONDS,
    ControlHandshake,
)


class FakeSocket:
    def __init__(self):
        self.sent = []
        self.received = []
        self.bound = None
        self.blocking = None

    def setblocking(self, value):
        self.blocking = value

    def bind(self, address):
        self.bound = address

    def sendto(self, packet, address):
        self.sent.append((packet, address))

    def recvfrom(self, _size):
        if self.received:
            return self.received.pop(0)
        raise BlockingIOError


class ControlHandshakeTests(unittest.TestCase):
    def setUp(self):
        self.udp_socket = FakeSocket()
        self.status_updates = []
        self.handshake = ControlHandshake(
            self.udp_socket,
            "192.0.2.20",
            lambda state, message: self.status_updates.append(
                (state, message)
            ),
        )

    def test_sends_hello_retries_and_accepts_pi_ack(self):
        self.assertFalse(self.handshake.poll(10.0))
        self.assertEqual(
            self.udp_socket.sent[-1],
            (make_hello_packet(), ("192.0.2.20", 5007)),
        )
        self.assertIn("(attempt 1/10)", self.status_updates[-1][1])

        self.assertFalse(
            self.handshake.poll(10.0 + CONNECTION_RETRY_SECONDS)
        )
        self.assertIn("(attempt 2/10)", self.status_updates[-1][1])

        self.udp_socket.received.append(
            (make_ack_packet(), ("192.0.2.20", 5007))
        )
        self.assertTrue(self.handshake.poll(11.1))
        self.assertEqual(self.handshake.generation, 1)
        self.assertEqual(self.status_updates[-1][0], "Connected")
        self.assertEqual(
            self.status_updates[-1][1], "CONNECTION OK - Pi confirmed"
        )

    def test_failed_attempts_continue_retrying(self):
        for attempt in range(CONNECTION_ATTEMPTS):
            self.handshake.poll(20.0 + attempt * CONNECTION_RETRY_SECONDS)

        self.assertFalse(
            self.handshake.poll(
                20.0 + CONNECTION_ATTEMPTS * CONNECTION_RETRY_SECONDS
            )
        )
        self.assertEqual(self.status_updates[-1][0], "Failed")
        self.assertIn("CONNECTION FAILED - no response", self.status_updates[-1][1])

    def test_heartbeat_timeout_marks_disconnected_and_retries(self):
        self.udp_socket.received.append(
            (make_ack_packet(), ("192.0.2.20", 5007))
        )
        self.assertTrue(self.handshake.poll(30.0))
        heartbeat_at = 30.0 + CONNECTION_HEARTBEAT_SECONDS
        self.assertTrue(self.handshake.poll(heartbeat_at))
        self.assertFalse(
            self.handshake.poll(heartbeat_at + 1.5)
        )
        self.assertEqual(self.status_updates[-1][0], "Failed")
        self.assertFalse(self.handshake.poll(heartbeat_at + 1.6))
        self.assertTrue(
            any(packet == make_hello_packet() for packet, _ in self.udp_socket.sent)
        )

    def test_ignores_ack_from_wrong_endpoint(self):
        self.udp_socket.received.append(
            (make_ack_packet(), ("192.0.2.21", 5007))
        )
        self.assertFalse(self.handshake.poll(40.0))
        self.assertFalse(self.handshake.connected)


if __name__ == "__main__":
    unittest.main()
