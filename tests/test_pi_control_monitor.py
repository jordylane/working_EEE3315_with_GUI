import unittest
from unittest.mock import patch

from common import make_ack_packet, make_hello_packet, pack_control_packet
from pi import pi_control_monitor


class FakeSocket:
    def __init__(self, packets=()):
        self.bound_address = None
        self.packets = list(packets)
        self.sent = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def bind(self, address):
        self.bound_address = address

    def settimeout(self, _timeout):
        pass

    def recvfrom(self, _size):
        if self.packets:
            return self.packets.pop(0)
        raise KeyboardInterrupt

    def sendto(self, packet, address):
        self.sent.append((packet, address))


class ControlMonitorTests(unittest.TestCase):
    def _run_monitor(self, arguments, packets=()):
        fake_socket = FakeSocket(packets)
        with (
            patch(
                "pi.pi_control_monitor.socket.socket",
                return_value=fake_socket,
            ) as socket_factory,
            patch("builtins.print") as print_output,
        ):
            result = pi_control_monitor.main(arguments)
        socket_factory.assert_called_once_with(
            pi_control_monitor.socket.AF_INET,
            pi_control_monitor.socket.SOCK_DGRAM,
        )
        return fake_socket.bound_address, fake_socket, print_output, result

    def test_defaults_bind_udp_on_all_interfaces_port_5007(self):
        address, _socket, output, _result = self._run_monitor([])

        self.assertEqual(address, ("0.0.0.0", 5007))
        self.assertTrue(
            any(
                "Listening on 0.0.0.0:5007 at" in str(call)
                for call in output.call_args_list
            )
        )

    def test_host_and_port_can_be_overridden(self):
        address, _socket, output, _result = self._run_monitor(
            ["--host", "192.0.2.10", "--port", "6000"]
        )

        self.assertEqual(address, ("192.0.2.10", 6000))
        self.assertTrue(
            any(
                "Listening on 192.0.2.10:6000 at" in str(call)
                for call in output.call_args_list
            )
        )

    def test_hello_gets_ack_and_is_excluded_from_control_statistics(self):
        sender = ("192.0.2.30", 43210)
        control = pack_control_packet(0, 1.0, {"up"})
        address, fake_socket, output, _result = self._run_monitor(
            [], [(make_hello_packet(), sender), (control, sender)]
        )

        self.assertEqual(address, ("0.0.0.0", 5007))
        self.assertEqual(fake_socket.sent, [(make_ack_packet(), sender)])
        rendered = [str(call) for call in output.call_args_list]
        self.assertTrue(
            any("CONNECTION OK from 192.0.2.30:43210" in line for line in rendered)
        )
        self.assertTrue(
            any(
                "received=1 dropped=0 missing=0 hellos=1" in line
                for line in rendered
            )
        )

    def test_repeated_hello_replies_each_time_but_prints_once(self):
        sender = ("192.0.2.30", 43210)
        address, fake_socket, output, _result = self._run_monitor(
            [],
            [
                (make_hello_packet(), sender),
                (make_hello_packet(), sender),
            ],
        )
        connection_lines = [
            call for call in output.call_args_list
            if "CONNECTION OK from" in str(call)
        ]

        self.assertEqual(address, ("0.0.0.0", 5007))
        self.assertEqual(len(fake_socket.sent), 2)
        self.assertTrue(all(packet == make_ack_packet()
                            for packet, _ in fake_socket.sent))
        self.assertEqual(len(connection_lines), 1)

    def test_status_accepts_auxiliary_control_keys(self):
        with patch("builtins.print") as print_output:
            pi_control_monitor.print_status(
                0.0, 1, {"up", "grabber_up", "claw_open"}, 10.0
            )

        self.assertIn(
            "keys=UP+GRABBER_UP+CLAW_OPEN",
            str(print_output.call_args),
        )


if __name__ == "__main__":
    unittest.main()
