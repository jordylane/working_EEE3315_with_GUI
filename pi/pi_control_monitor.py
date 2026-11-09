import argparse
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (
    CONTROL_RATE_HZ,
    CONTROL_TIMEOUT_SECONDS,
    is_hello_packet,
    make_ack_packet,
    mix,
    unpack_control_packet,
)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5007
HEARTBEAT_SECONDS = 1.0
KEY_ORDER = (
    "up", "down", "left", "right",
    "grabber_up", "grabber_down", "claw_open", "claw_close",
)


def print_status(start, sequence, keys, age_ms):
    names = "+".join(key.upper() for key in KEY_ORDER if key in keys) or "NONE"
    drive_keys = keys & {"up", "down", "left", "right"}
    left, right = mix(drive_keys)
    print(
        f"[{time.perf_counter() - start:.3f}] seq={sequence} keys={names} "
        f"left={left:.2f} right={right:.2f} age={age_ms:.1f}ms",
        flush=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Monitor PC vehicle-control packets.")
    parser.add_argument("--all", action="store_true", help="Print every valid packet.")
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"UDP bind address (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"UDP bind port (default: {DEFAULT_PORT})",
    )
    args = parser.parse_args(argv)
    listen_address = (args.host, args.port)
    start = time.perf_counter()
    last_packet_time = None
    last_print_time = 0.0
    last_keys = None
    last_sequence = None
    received = dropped = missing = hellos = 0
    signal_lost = False
    hello_senders = set()

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as control_socket:
        try:
            control_socket.bind(listen_address)
        except OSError as exc:
            # Test with this script instead of the main Pi script, or configure
            # SO_REUSEADDR/SO_REUSEPORT; UDP may still deliver to only one socket
            # unless multicast/broadcast is used.
            print(
                f"Cannot bind UDP {args.host}:{args.port}: {exc}. "
                "Stop any other process using this address "
                "for testing, or configure socket reuse (UDP packet delivery "
                "may be exclusive).",
                flush=True,
            )
            return 1

        control_socket.settimeout(0.05)
        print(
            f"Listening on {listen_address[0]}:{listen_address[1]} "
            f"at {CONTROL_RATE_HZ} Hz"
        )
        try:
            while True:
                try:
                    packet, sender_address = control_socket.recvfrom(1024)
                except socket.timeout:
                    now = time.perf_counter()
                    reference = last_packet_time if last_packet_time is not None else start
                    if not signal_lost and now - reference >= CONTROL_TIMEOUT_SECONDS:
                        print(f"[{now - start:.3f}] NO SIGNAL - STOP", flush=True)
                        signal_lost = True
                        last_keys = None
                    elif (
                        not signal_lost and last_keys is not None
                        and now - last_print_time >= HEARTBEAT_SECONDS
                    ):
                        age_ms = (now - last_packet_time) * 1000.0
                        print_status(start, last_sequence, last_keys, age_ms)
                        last_print_time = now
                    continue

                now = time.perf_counter()
                if is_hello_packet(packet):
                    hellos += 1
                    sender_seen = sender_address in hello_senders
                    was_signal_lost = signal_lost
                    hello_senders.add(sender_address)
                    last_sequence = None
                    last_packet_time = now
                    signal_lost = False
                    if not sender_seen or was_signal_lost:
                        print(
                            f"[{now - start:.3f}] CONNECTION OK from "
                            f"{sender_address[0]}:{sender_address[1]}",
                            flush=True,
                        )
                    try:
                        control_socket.sendto(
                            make_ack_packet(), sender_address
                        )
                    except OSError as exc:
                        print(
                            f"Cannot send CONNECTION ACK to "
                            f"{sender_address[0]}:{sender_address[1]}: {exc}",
                            flush=True,
                        )
                    continue

                received += 1
                try:
                    sequence, _, keys = unpack_control_packet(packet)
                except ValueError:
                    dropped += 1
                    continue

                if last_sequence is not None:
                    difference = (sequence - last_sequence) & 0xFFFFFFFF
                    if difference == 0 or difference >= 0x80000000:
                        dropped += 1
                        print(f"[{now - start:.3f}] WARNING out-of-order "
                              f"dropped={dropped} missing={missing}", flush=True)
                        continue
                    missing += difference - 1
                    if difference > 1:
                        print(f"[{now - start:.3f}] WARNING sequence gap "
                              f"dropped={dropped} missing={missing}", flush=True)

                age_ms = (
                    (now - last_packet_time) * 1000.0
                    if last_packet_time is not None else 0.0
                )
                changed = keys != last_keys
                restored = signal_lost
                if restored:
                    print(f"[{now - start:.3f}] SIGNAL RESTORED", flush=True)
                    signal_lost = False

                last_sequence = sequence
                last_packet_time = now
                if args.all or changed or restored:
                    print_status(start, sequence, keys, age_ms)
                    last_print_time = now
                last_keys = keys
        except KeyboardInterrupt:
            elapsed = time.perf_counter() - start
            rate = received / elapsed if elapsed else 0.0
            print(f"Stopped: received={received} dropped={dropped} "
                  f"missing={missing} hellos={hellos} "
                  f"average_rate={rate:.2f} packets/s")


if __name__ == "__main__":
    raise SystemExit(main())
