import socket
import time
from collections.abc import Callable

from common import (
    CONTROL_MONITOR_PORT,
    is_ack_packet,
    make_hello_packet,
)


CONNECTION_RETRY_SECONDS = 0.5
CONNECTION_ATTEMPTS = 10
CONNECTION_HEARTBEAT_SECONDS = 0.1
CONNECTION_ACK_TIMEOUT_SECONDS = 1.5
CONNECTION_RETRY_DELAY_SECONDS = 1.0
LIKELY_CONNECTION_CAUSES = (
    "wrong IP, Pi script not running, firewall, or different network"
)


class ControlHandshake:
    def __init__(
        self,
        udp_socket,
        pi_ip: str,
        status_callback: Callable[[str, str], None] | None = None,
    ):
        self.udp_socket = udp_socket
        self.pi_ip = pi_ip
        self.address = (pi_ip, CONTROL_MONITOR_PORT)
        self.status_callback = status_callback
        self.connected = False
        self.generation = 0
        self.attempts = 0
        self.next_attempt = 0.0
        self.awaiting_ack_until = None
        self.next_heartbeat = None
        self.state = "Connecting"
        self.udp_socket.setblocking(False)
        self.udp_socket.bind(("", 0))

    def _set_status(self, state: str, message: str):
        self.state = state
        print(message, flush=True)
        if self.status_callback is not None:
            self.status_callback(state, message)

    def _matches_pi(self, sender):
        if sender[1] != CONTROL_MONITOR_PORT:
            return False
        try:
            expected_ip = socket.gethostbyname(self.pi_ip)
        except OSError:
            expected_ip = self.pi_ip
        return sender[0] == expected_ip

    def _send_hello(self):
        try:
            self.udp_socket.sendto(make_hello_packet(), self.address)
        except OSError as error:
            print(f"Connection HELLO send failed: {error}", flush=True)

    def _receive_acknowledgements(self, now):
        while True:
            try:
                packet, sender = self.udp_socket.recvfrom(1024)
            except BlockingIOError:
                return
            except OSError as error:
                print(f"Connection ACK receive failed: {error}", flush=True)
                return

            if not is_ack_packet(packet) or not self._matches_pi(sender):
                continue

            newly_connected = not self.connected
            self.connected = True
            self.attempts = 0
            self.awaiting_ack_until = None
            self.next_heartbeat = now + CONNECTION_HEARTBEAT_SECONDS
            self.next_attempt = now
            if newly_connected:
                self.generation += 1
                self._set_status("Connected", "CONNECTION OK - Pi confirmed")

    def _connection_failed(self, now):
        self.connected = False
        self.awaiting_ack_until = None
        self.attempts = 0
        self.next_attempt = now + CONNECTION_RETRY_DELAY_SECONDS
        self._set_status(
            "Failed",
            f"CONNECTION FAILED - no response from "
            f"{self.pi_ip}:{CONTROL_MONITOR_PORT} "
            f"({LIKELY_CONNECTION_CAUSES}); retrying",
        )

    def poll(self, now=None) -> bool:
        now = time.perf_counter() if now is None else now
        self._receive_acknowledgements(now)

        if self.connected:
            if (
                self.awaiting_ack_until is not None
                and now >= self.awaiting_ack_until
            ):
                self._connection_failed(now)
            elif (
                self.awaiting_ack_until is None
                and now >= self.next_heartbeat
            ):
                self._send_hello()
                self.awaiting_ack_until = (
                    now + CONNECTION_ACK_TIMEOUT_SECONDS
                )
                self.next_heartbeat = (
                    now + CONNECTION_HEARTBEAT_SECONDS
                )
            return self.connected

        if now < self.next_attempt:
            return False

        if self.attempts >= CONNECTION_ATTEMPTS:
            self._set_status(
                "Failed",
                f"CONNECTION FAILED - no response from "
                f"{self.pi_ip}:{CONTROL_MONITOR_PORT} "
                f"({LIKELY_CONNECTION_CAUSES}); retrying",
            )
            self.attempts = 0
            self.next_attempt = now + CONNECTION_RETRY_DELAY_SECONDS
            return False

        self.attempts += 1
        self._set_status(
            "Connecting",
            f"Connecting to {self.pi_ip}:{CONTROL_MONITOR_PORT} ... "
            f"(attempt {self.attempts}/{CONNECTION_ATTEMPTS})",
        )
        self._send_hello()
        self.next_attempt = now + CONNECTION_RETRY_SECONDS
        return False
