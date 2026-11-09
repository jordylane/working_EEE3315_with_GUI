import socket
import sys
import threading
import time
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
import tkinter as tk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (
    CONTROL_KEY_BITS,
    CONTROL_RATE_HZ,
    PI_IP,
    UDP_PORT,
    anchors,
    control_destinations,
    decode_packet,
    pack_control_packet,
)
from pc.control_connection import ControlHandshake

PLOT_MAX_RANGE = 4.0
PLOT_AUTO_RANGE = False
PLOT_CONFIDENCE = True
PLOT_CONFIDENCE_COLOUR_MAP = "bwr_r"
PRINT_EVERY = 10


class ControlSender:
    def __init__(self, pi_ip):
        self.addresses = control_destinations(pi_ip)
        self.lock = threading.Lock()
        self.keys = set()
        self.pending_states = deque()
        self.sent_timestamps = {}
        self.wake = threading.Event()
        self.closing = False
        self.thread = threading.Thread(target=self._send_loop, daemon=True)
        self.thread.start()

    def set_keys(self, keys):
        keys = set(keys)
        with self.lock:
            if self.closing:
                return
            if keys == self.keys:
                return
            self.keys = keys
            self.pending_states.append(keys.copy())
        self.wake.set()

    def _send_loop(self):
        interval = 1.0 / CONTROL_RATE_HZ
        sequence = 0
        next_send = time.perf_counter()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
            handshake = ControlHandshake(udp_socket, self.addresses[0][0])
            connection_generation = 0
            while True:
                timeout = max(0.0, next_send - time.perf_counter())
                self.wake.wait(timeout)
                self.wake.clear()

                timestamp = time.perf_counter()
                connected = handshake.poll(timestamp)
                with self.lock:
                    pending_states = list(self.pending_states)
                    self.pending_states.clear()
                    current_keys = self.keys.copy()
                    closing = self.closing

                if connected:
                    if handshake.generation != connection_generation:
                        connection_generation = handshake.generation
                        sequence = 0
                        pending_states.clear()
                    states_to_send = pending_states
                    if timestamp >= next_send or (closing and not states_to_send):
                        if not states_to_send or states_to_send[-1] != current_keys:
                            states_to_send.append(current_keys)
                        next_send = timestamp + interval
                else:
                    states_to_send = []
                    pending_states.clear()
                    next_send = timestamp + interval

                for keys in states_to_send:
                    send_timestamp = time.perf_counter()
                    with self.lock:
                        self.sent_timestamps[sequence] = send_timestamp
                        while len(self.sent_timestamps) > 256:
                            oldest_sequence = next(iter(self.sent_timestamps))
                            del self.sent_timestamps[oldest_sequence]
                    packet = pack_control_packet(sequence, send_timestamp, keys)
                    for address in self.addresses:
                        try:
                            udp_socket.sendto(packet, address)
                        except BlockingIOError:
                            pass
                        except OSError as exc:
                            print(
                                f"control UDP send to {address[0]}:{address[1]} "
                                f"failed: {exc}",
                                flush=True,
                            )
                    sequence = (sequence + 1) & 0xFFFFFFFF
                if closing:
                    break

    def close(self):
        with self.lock:
            self.pending_states.append(set())
            self.keys.clear()
            self.closing = True
        self.wake.set()
        self.thread.join()

    def round_trip_ms(self, sequence, echoed_timestamp):
        if not isinstance(sequence, int) or not isinstance(echoed_timestamp, (int, float)):
            return None
        with self.lock:
            sent_timestamp = self.sent_timestamps.get(sequence)
        if sent_timestamp is None or sent_timestamp != echoed_timestamp:
            return None
        elapsed = time.perf_counter() - sent_timestamp
        return elapsed * 1000.0 if elapsed >= 0.0 else None


def bind_control_keys(fig, control_sender):
    window = fig.canvas.manager.window
    if not isinstance(window, tk.Misc):
        raise RuntimeError(
            "Arrow-key control requires Matplotlib's Tkinter backend."
        )

    held = set()

    def on_key_press(event):
        key = event.keysym.lower()
        if key in CONTROL_KEY_BITS and key not in held:
            held.add(key)
            control_sender.set_keys(held)

    def on_key_release(event):
        key = event.keysym.lower()
        if key in held:
            held.remove(key)
            control_sender.set_keys(held)

    def on_focus_loss(_event):
        if held:
            held.clear()
            control_sender.set_keys(held)

    window.bind("<KeyPress>", on_key_press, add="+")
    window.bind("<KeyRelease>", on_key_release, add="+")
    window.bind("<FocusOut>", on_focus_loss, add="+")
    window.focus_set()


def display_heading(angle):
    hx = -np.sin(angle)
    hy = np.cos(angle)
    return (np.degrees(np.arctan2(hx, hy)) - 90 + 360) % 360


class DisplayData:
    def __init__(self):
        self.lock = threading.Lock()
        self.lidar_x = np.array([])
        self.lidar_y = np.array([])
        self.lidar_c = np.array([])
        self.uwb_x = 0.0
        self.uwb_y = 0.0
        self.distances = [0.0] * 4
        self.ax_buffer = deque(maxlen=100)
        self.ay_buffer = deque(maxlen=100)
        self.heading = 0.0
        self.raw_heading = 0.0
        self.gyro_heading = 0.0
        self.kalman_heading = 0.0
        self.yaw = 0.0
        self.vehicle_x = 0.0
        self.vehicle_y = 0.0
        self.vehicle_heading = 0.0
        self.rtt_ms = None


def receiver_loop(shared_data, stop_event, control_sender):
    packet_count = 0
    failures = 0
    last_packet_time = time.monotonic()
    warned = False

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
        udp_socket.bind(("0.0.0.0", UDP_PORT))
        udp_socket.settimeout(0.5)
        print(f"listening on port {UDP_PORT}", flush=True)

        while not stop_event.is_set():
            try:
                payload, _ = udp_socket.recvfrom(65535)
            except socket.timeout:
                if not warned and time.monotonic() - last_packet_time >= 5:
                    print("no packets for 5 s", flush=True)
                    warned = True
                continue

            last_packet_time = time.monotonic()
            warned = False
            packet_count += 1

            try:
                packet = decode_packet(payload)
                round_trip_ms = control_sender.round_trip_ms(
                    packet.get("control_sequence"),
                    packet.get("control_timestamp"),
                )
                with shared_data.lock:
                    shared_data.lidar_x = packet["lidar_x"]
                    shared_data.lidar_y = packet["lidar_y"]
                    shared_data.lidar_c = packet["lidar_c"]
                    shared_data.uwb_x = packet["uwb_x"]
                    shared_data.uwb_y = packet["uwb_y"]
                    shared_data.distances = packet["distances"]
                    shared_data.ax_buffer.append(packet["ax"])
                    shared_data.ay_buffer.append(packet["ay"])
                    shared_data.heading = packet["heading"]
                    shared_data.raw_heading = packet.get(
                        "raw_heading", np.radians(packet["heading"])
                    )
                    shared_data.gyro_heading = packet.get(
                        "gyro_heading", shared_data.raw_heading
                    )
                    shared_data.kalman_heading = packet.get(
                        "kalman_heading", shared_data.raw_heading
                    )
                    shared_data.yaw = packet["yaw"]
                    shared_data.vehicle_x = packet.get("vehicle_x", 0.0)
                    shared_data.vehicle_y = packet.get("vehicle_y", 0.0)
                    if round_trip_ms is not None:
                        shared_data.rtt_ms = round_trip_ms
                if packet_count % PRINT_EVERY:
                    continue
                distances = ", ".join(f"{value:.2f}" for value in packet["distances"])
                age = time.time() - packet["timestamp"]
                print(
                    f"#{packet_count} {len(payload)} B lidar={len(packet['lidar_x'])} "
                    f"uwb=({packet['uwb_x']:.2f}, {packet['uwb_y']:.2f}) "
                    f"distances=[{distances}] ax/ay=({packet['ax']:.3f}, "
                    f"{packet['ay']:.3f}) heading={packet['heading']:.1f} "
                    f"yaw={packet['yaw']:.1f} age={age:.3f}s",
                    flush=True,
                )
            except (KeyError, TypeError, ValueError, RecursionError) as exc:
                failures += 1
                if failures % 50 == 0:
                    print(f"decode failure {failures}: {exc}", flush=True)


def display_loop(shared_data, stop_event, control_sender):
    fig = plt.figure(figsize=(8, 8))

    ax_pos = fig.add_subplot(221)

    x_history = []
    y_history = []
    vehicle_x_history = []
    vehicle_y_history = []

    circle_1 = Circle(
        (anchors[0, 0], anchors[0, 1]),
        0,
        fill=False,
        color='red',
        linewidth=0.5
    )

    ax_pos.add_patch(circle_1)

    circle_2 = Circle(
        (anchors[1, 0], anchors[1, 1]),
        0,
        fill=False,
        color='red',
        linewidth=0.5
    )

    ax_pos.add_patch(circle_2)

    circle_3 = Circle(
        (anchors[2, 0], anchors[2, 1]),
        0,
        fill=False,
        color='red',
        linewidth=0.5
    )

    ax_pos.add_patch(circle_3)

    circle_4 = Circle(
        (anchors[3, 0], anchors[3, 1]),
        0,
        fill=False,
        color='red',
        linewidth=0.5
    )

    ax_pos.add_patch(circle_4)

    graph_history = ax_pos.plot([], [], 'b-', label="Path")[0]
    graph_current = ax_pos.plot([], [], 'ro', label="Current Position")[0]
    graph_vehicle = ax_pos.plot([], [], 'g-', label="Simulated vehicle")[0]
    graph_vehicle_current = ax_pos.plot([], [], 'go', label="Sim vehicle")[0]
    ax_pos.legend()

    ax_pos.set_xlim(0, 4.0)
    ax_pos.set_ylim(0, 4.0)
    ax_pos.set_aspect('equal')

    ax_pos.set_title("UWB Position")
    ax_pos.set_xlabel("X (m)")
    ax_pos.set_ylabel("Y (m)")
    ax_pos.grid(True)

    ax_accel = fig.add_subplot(222)

    graph_accel_x = ax_accel.plot([], 'b-', label='x acceleration')[0]
    graph_accel_y = ax_accel.plot([], 'r-', label='y acceleration')[0]

    ax_accel.legend()
    ax_accel.set_title("Acceleration")
    ax_accel.grid(True)

    ax_compass = fig.add_subplot(223)

    t = np.linspace(
        0,
        2*np.pi,
        500
    )

    ax_compass.plot(
        np.cos(t),
        np.sin(t),
        'k',
        lw=2
    )

    for ang in range(0, 360, 10):

        a = np.radians(ang)

        r1 = 0.92
        r2 = 1.00

        ax_compass.plot(
            [r1*np.sin(a), r2*np.sin(a)],
            [r1*np.cos(a), r2*np.cos(a)],
            'k'
        )

    graph_compass_red = ax_compass.plot(
        [],
        [],
        'r',
        linewidth=2,
        label="Raw mag"
    )[0]

    graph_compass_blue = ax_compass.plot(
        [],
        [],
        'b',
        linewidth=2,
        label="Gyro only"
    )[0]

    graph_compass_green = ax_compass.plot(
        [],
        [],
        'g',
        linewidth=4,
        label="Kalman"
    )[0]
    ax_compass.legend(loc="lower center", ncol=3, fontsize=8)

    ax_compass.text(
        0, 1.12, 'N',
        fontsize=16,
        ha='center'
    )

    ax_compass.text(
        1.12, 0, 'E',
        fontsize=16,
        ha='center'
    )

    ax_compass.text(
        0, -1.12, 'S',
        fontsize=16,
        ha='center'
    )

    ax_compass.text(
        -1.12, 0, 'W',
        fontsize=16,
        ha='center'
    )

    ax_compass.set_xlim([-1.2, 1.2])
    ax_compass.set_ylim([-1.2, 1.2])

    ax_compass.set_aspect('equal')
    ax_compass.axis('off')

    ax_compass.set_title(f"Compass Heading\n")

    ax_lidar = fig.add_subplot(224)

    if PLOT_CONFIDENCE:
        graph = ax_lidar.scatter([], [], c=[], marker=".", s=8, vmin=0,
                            vmax=255, cmap=PLOT_CONFIDENCE_COLOUR_MAP)
    else:
        graph = ax_lidar.plot([], [], ".")[0]

    ax_lidar.set_xlim(-PLOT_MAX_RANGE, PLOT_MAX_RANGE)
    ax_lidar.set_ylim(-PLOT_MAX_RANGE, PLOT_MAX_RANGE)

    ax_lidar.set_xlim(-4, 4)
    ax_lidar.set_ylim(-4, 4)

    yaw_arrow = FancyArrowPatch(
        (0, 0),
        (0, 1),
        color='red',
        mutation_scale=15,
        linewidth=0.5
    )

    ax_lidar.add_patch(yaw_arrow)

    yaw_text = ax_lidar.text(
        0,
        -PLOT_MAX_RANGE + 0.3,
        "Yaw = 0.0°",
        color='red'
    )

    plt.ion()
    plt.show(block=False)
    bind_control_keys(fig, control_sender)
    fig.canvas.mpl_connect("close_event", lambda event: stop_event.set())

    while not stop_event.is_set() and plt.fignum_exists(fig.number):
        with shared_data.lock:
            x = np.asarray(shared_data.lidar_x)
            y = np.asarray(shared_data.lidar_y)
            c = np.asarray(shared_data.lidar_c)
            distances = shared_data.distances.copy()
            uwb_x = shared_data.uwb_x
            uwb_y = shared_data.uwb_y
            raw_heading = shared_data.raw_heading
            gyro_heading = shared_data.gyro_heading
            kalman_heading = shared_data.kalman_heading
            yaw = shared_data.yaw
            vehicle_x = shared_data.vehicle_x
            vehicle_y = shared_data.vehicle_y
            rtt_ms = shared_data.rtt_ms
            ax_hist = np.asarray(shared_data.ax_buffer)
            ay_hist = np.asarray(shared_data.ay_buffer)

        x_history.append(uwb_x)
        y_history.append(uwb_y)
        vehicle_x_history.append(vehicle_x)
        vehicle_y_history.append(vehicle_y)

        raw_display_heading = display_heading(raw_heading)
        gyro_display_heading = display_heading(gyro_heading)
        kalman_display_heading = display_heading(kalman_heading)

        yaw = (yaw - (np.pi / 2) + (2 * np.pi)) % (2 * np.pi)

        circle_1.set_radius(distances[0])
        circle_2.set_radius(distances[1])
        circle_3.set_radius(distances[2])
        circle_4.set_radius(distances[3])

        graph_history.set_data(x_history, y_history)
        graph_vehicle.set_data(vehicle_x_history, vehicle_y_history)
        graph_vehicle_current.set_data([vehicle_x], [vehicle_y])

        if x_history and y_history:
            graph_current.set_data([x_history[-1]], [y_history[-1]])
        else:
            graph_current.set_data([], [])

        graph_accel_x.set_data(
            np.arange(len(ax_hist)),
            ax_hist
        )

        graph_accel_y.set_data(
            np.arange(len(ay_hist)),
            ay_hist
        )

        if len(ax_hist) and len(ay_hist):
            ax_accel.set_xlim(0, len(ax_hist))
            ymin = min(ax_hist.min(), ay_hist.min()) - 0.25
            ymax = max(ax_hist.max(), ay_hist.max()) + 0.25
            ax_accel.set_ylim(ymin, ymax)

        a = np.radians(raw_display_heading)
        graph_compass_red.set_data(
            [0, 0.8*np.sin(a)],
            [0, 0.8*np.cos(a)]
        )

        a = np.radians(gyro_display_heading)
        graph_compass_blue.set_data(
            [0, -0.3*np.sin(a)],
            [0, -0.3*np.cos(a)]
        )

        a = np.radians(kalman_display_heading)
        graph_compass_green.set_data(
            [0, np.sin(a)],
            [0, np.cos(a)]
        )

        ax_compass.set_title(
            f"Compass Heading: {kalman_display_heading:.1f}°"
        )

        if PLOT_AUTO_RANGE and len(x) and len(y):
            mav_val = max(np.max(np.abs(x)), np.max(np.abs(y))) * 1.2
            ax_lidar.set_xlim(-mav_val, mav_val)
            ax_lidar.set_ylim(-mav_val, mav_val)

        if PLOT_CONFIDENCE:
            points = np.column_stack((x, y))
            graph.set_offsets(points)
            graph.set_array(np.asarray(c))
        else:
            graph = ax_lidar.plot(x, y, 'b.')[0]

        arrow_length = 1.0

        arrow_x = arrow_length * np.sin(yaw)
        arrow_y = arrow_length * np.cos(yaw)

        yaw_arrow.set_positions(
            (0, 0),
            (arrow_x, arrow_y)
        )

        yaw_text.set_text(
            f"Yaw = {np.degrees(yaw):.1f}°"
        )
        fig.suptitle(
            "Control RTT: -- ms" if rtt_ms is None
            else f"Control RTT: {rtt_ms:.1f} ms"
        )

        plt.pause(0.03)


def main():
    shared_data = DisplayData()
    stop_event = threading.Event()
    control_sender = ControlSender(PI_IP)
    receiver = threading.Thread(
        target=receiver_loop,
        args=(shared_data, stop_event, control_sender),
        daemon=True,
    )
    receiver.start()

    try:
        display_loop(shared_data, stop_event, control_sender)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        control_sender.close()
        plt.close("all")
        receiver.join()


if __name__ == "__main__":
    main()
