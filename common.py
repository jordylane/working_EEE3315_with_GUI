import json
import math
import struct
import time
from collections.abc import Hashable
from collections.abc import Collection

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


PC_IP = "192.168.15.14"
PI_IP = "192.168.15.16"
UDP_PORT = 5005
CONTROL_PORT = 5006
CONTROL_MONITOR_PORT = 5007
HELLO_MAGIC = b"\x00UWB-HELLO\xff"
ACK_MAGIC = b"\x00UWB-ACK\xff"
HELLO_PAYLOAD = b"CONNECTION OK"
ACK_PAYLOAD = b"CONNECTION ACK"
SEND_RATE_HZ = 20
EXPECTED_RATE_HZ = SEND_RATE_HZ
CONTROL_RATE_HZ = 20
CONTROL_TIMEOUT_SECONDS = 0.3
COMMAND_TIMEOUT_S = 0.5
TRACK_WIDTH = 0.35  # metres between the tank's tracks
MAX_SPEED = 1.0  # metres per second at full track command
CONTROL_PACKET_FORMAT = "!IdB"  # sequence, PC perf_counter timestamp, key bits
CONTROL_KEY_BITS = {
    "up": 0x01,
    "down": 0x02,
    "left": 0x04,
    "right": 0x08,
    "grabber_up": 0x10,
    "grabber_down": 0x20,
    "claw_open": 0x40,
    "claw_close": 0x80,
}
MAX_PACKET_SIZE = 60 * 1024
UDP_SAFE_DATAGRAM_SIZE = 1200
TELEMETRY_FRAGMENT_MAGIC = b"\x00UWF"
TELEMETRY_FRAGMENT_HEADER = struct.Struct("!4sIHHI")
TELEMETRY_FRAGMENT_TIMEOUT_S = 2.0
MAX_TELEMETRY_ASSEMBLIES = 16
_ANCHOR_COORDINATES = (
    [0.0, 0.0, 1.2],
    [0.0, 4.0, 0.75],
    [4.0, 4.0, 1.2],
    [4.0, 0.0, 0.75],
)
anchors = np.array(_ANCHOR_COORDINATES) if np is not None else _ANCHOR_COORDINATES


def make_hello_packet() -> bytes:
    return HELLO_MAGIC + HELLO_PAYLOAD


def is_hello_packet(data: bytes) -> bool:
    return data == make_hello_packet()


def make_ack_packet() -> bytes:
    return ACK_MAGIC + ACK_PAYLOAD


def is_ack_packet(data: bytes) -> bool:
    return data == make_ack_packet()


def control_destinations(pi_ip: str) -> tuple[tuple[str, int], ...]:
    """Return UDP endpoints for Pi sensor control and control monitoring."""
    return (
        (pi_ip, CONTROL_PORT),
        (pi_ip, CONTROL_MONITOR_PORT),
    )


def mix(keys: Collection[str]) -> tuple[float, float]:
    """Mix held arrow keys into normalized left/right differential-track speeds."""
    unknown_keys = set(keys) - CONTROL_KEY_BITS.keys()
    if unknown_keys:
        raise ValueError(f"Unknown control keys: {sorted(unknown_keys)}")

    throttle = float("up" in keys) - float("down" in keys)
    steering = float("right" in keys) - float("left" in keys)
    left = throttle + steering
    right = throttle - steering
    scale = max(1.0, abs(left), abs(right))
    return left / scale, right / scale


def pack_control_packet(
    sequence: int,
    pc_timestamp: float,
    keys: Collection[str],
) -> bytes:
    unknown_keys = set(keys) - CONTROL_KEY_BITS.keys()
    if unknown_keys:
        raise ValueError(f"Unknown control keys: {sorted(unknown_keys)}")
    key_bits = 0
    for key in keys:
        key_bits |= CONTROL_KEY_BITS[key]
    return struct.pack(CONTROL_PACKET_FORMAT, sequence, pc_timestamp, key_bits)


def unpack_control_packet(packet: bytes) -> tuple[int, float, set[str]]:
    packet_struct = struct.Struct(CONTROL_PACKET_FORMAT)
    if len(packet) != packet_struct.size:
        raise ValueError(f"Expected {packet_struct.size}-byte control packet")

    sequence, pc_timestamp, key_bits = packet_struct.unpack(packet)
    valid_key_bits = sum(CONTROL_KEY_BITS.values())
    if key_bits & ~valid_key_bits:
        raise ValueError("Control packet contains unknown key bits")
    if not math.isfinite(pc_timestamp):
        raise ValueError("Control packet timestamp must be finite")

    keys = {
        key for key, bit in CONTROL_KEY_BITS.items()
        if key_bits & bit
    }
    return sequence, pc_timestamp, keys


def _to_builtin(value):
    if np is not None and isinstance(value, np.ndarray):
        return _to_builtin(value.tolist())
    if np is not None and isinstance(value, np.generic):
        return _to_builtin(value.item())
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def _round_floats(value):
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    return value


def encode_packet(packet):
    """Encode a packet as compact JSON, reducing float precision if needed."""
    payload = _to_builtin(packet)
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")

    if len(encoded) > MAX_PACKET_SIZE:
        payload = _round_floats(payload)
        encoded = json.dumps(
            payload,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    return encoded


def decode_packet(packet):
    """Decode JSON bytes and require the top-level packet to be an object."""
    decoded = json.loads(packet)
    if not isinstance(decoded, dict):
        raise ValueError("UDP packet must contain a JSON object")
    return decoded


def fragment_telemetry_packet(
    payload: bytes,
    sequence: int,
    datagram_size: int = UDP_SAFE_DATAGRAM_SIZE,
) -> tuple[bytes, ...]:
    """Split a JSON telemetry packet into bounded UDP datagrams."""
    if not isinstance(payload, bytes):
        raise TypeError("telemetry payload must be bytes")
    if not payload:
        raise ValueError("telemetry payload must not be empty")
    if len(payload) > MAX_PACKET_SIZE:
        raise ValueError(f"telemetry payload exceeds {MAX_PACKET_SIZE} bytes")
    if not 0 <= sequence <= 0xFFFFFFFF:
        raise ValueError("telemetry sequence must fit in an unsigned 32-bit integer")
    if (
        not isinstance(datagram_size, int)
        or isinstance(datagram_size, bool)
        or datagram_size > UDP_SAFE_DATAGRAM_SIZE
    ):
        raise ValueError(
            f"datagram size must be an integer no greater than "
            f"{UDP_SAFE_DATAGRAM_SIZE}"
        )
    chunk_size = datagram_size - TELEMETRY_FRAGMENT_HEADER.size
    if chunk_size <= 0:
        raise ValueError("datagram size is too small for the fragment header")

    chunk_count = max(1, (len(payload) + chunk_size - 1) // chunk_size)
    if chunk_count > 0xFFFF:
        raise ValueError("telemetry packet requires too many fragments")
    return tuple(
        TELEMETRY_FRAGMENT_HEADER.pack(
            TELEMETRY_FRAGMENT_MAGIC,
            sequence,
            index,
            chunk_count,
            len(payload),
        )
        + payload[index * chunk_size : (index + 1) * chunk_size]
        for index in range(chunk_count)
    )


class TelemetryPacketReassembler:
    """Bounded, timeout-based reassembly for fragmented telemetry packets."""

    def __init__(
        self,
        timeout_s: float = TELEMETRY_FRAGMENT_TIMEOUT_S,
        max_assemblies: int = MAX_TELEMETRY_ASSEMBLIES,
    ):
        self.timeout_s = timeout_s
        self.max_assemblies = max_assemblies
        self._assemblies = {}

    def add(
        self,
        datagram: bytes,
        now: float | None = None,
        source_key: Hashable | None = None,
    ) -> bytes | None:
        now = time.monotonic() if now is None else now
        self._expire(now)
        if not datagram.startswith(TELEMETRY_FRAGMENT_MAGIC):
            return datagram
        if len(datagram) < TELEMETRY_FRAGMENT_HEADER.size:
            raise ValueError("telemetry fragment is shorter than its header")

        magic, sequence, index, count, total_size = (
            TELEMETRY_FRAGMENT_HEADER.unpack_from(datagram)
        )
        chunk = datagram[TELEMETRY_FRAGMENT_HEADER.size :]
        if magic != TELEMETRY_FRAGMENT_MAGIC:
            raise ValueError("invalid telemetry fragment marker")
        if (
            count == 0
            or index >= count
            or total_size == 0
            or total_size > MAX_PACKET_SIZE
            or count > total_size
            or len(datagram) > UDP_SAFE_DATAGRAM_SIZE
        ):
            raise ValueError("invalid telemetry fragment dimensions")
        if not chunk:
            raise ValueError("telemetry fragment has an empty data chunk")

        key = (source_key, sequence)
        assembly = self._assemblies.get(key)
        if assembly is None:
            if len(self._assemblies) >= self.max_assemblies:
                oldest_key = min(
                    self._assemblies,
                    key=lambda item: self._assemblies[item]["updated_at"],
                )
                del self._assemblies[oldest_key]
            assembly = {
                "count": count,
                "total_size": total_size,
                "chunks": {},
                "updated_at": now,
            }
            self._assemblies[key] = assembly
        elif (
            assembly["count"] != count
            or assembly["total_size"] != total_size
        ):
            del self._assemblies[key]
            raise ValueError("inconsistent telemetry fragment metadata")

        previous = assembly["chunks"].get(index)
        if previous is not None and previous != chunk:
            del self._assemblies[key]
            raise ValueError("conflicting duplicate telemetry fragment")
        assembly["chunks"][index] = chunk
        assembly["updated_at"] = now

        if len(assembly["chunks"]) != count:
            return None

        payload = b"".join(assembly["chunks"][part] for part in range(count))
        del self._assemblies[key]
        if len(payload) != total_size:
            raise ValueError("reassembled telemetry size does not match header")
        return payload

    def _expire(self, now: float) -> None:
        expired = [
            key
            for key, assembly in self._assemblies.items()
            if now - assembly["updated_at"] > self.timeout_s
        ]
        for key in expired:
            del self._assemblies[key]
