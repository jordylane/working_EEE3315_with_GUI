import argparse
import math
import os
import socket
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from urllib.request import urlopen

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"

import numpy as np
import pyqtgraph as pg
try:
    import pygame
except ImportError:
    pygame = None

from PySide6.QtCore import (
    QEvent,
    QObject,
    QPoint,
    QRect,
    Qt,
    QSize,
    QTimer,
    Signal,
)
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pc.mjpeg import extract_jpeg_frames
from common import (
    COMMAND_TIMEOUT_S,
    CONTROL_PORT,
    CONTROL_RATE_HZ,
    EXPECTED_RATE_HZ,
    PI_IP,
    UDP_PORT,
    TelemetryPacketReassembler,
    anchors,
    control_destinations,
    decode_packet,
    pack_control_packet,
)
from pc.controller_input import controller_commands
from pc.control_connection import ControlHandshake
from pc.imu import format_imu_reading, parse_imu_payload

DEFAULT_PI_IP = PI_IP
CAMERA_STREAM_URL = "http://192.168.137.115:5000"
PRINT_EVERY = 10
PLOT_SPAN = 5.0
LIDAR_PLOT_SPAN = 8.0
HISTORY_LENGTH = 100
AUTO_DISARM_ON_LINK_LOST = True
LINK_THRESHOLDS = {
    "packet_age": {"green": 0.2, "amber": 1.0},
    "loss": {"green": 0.02, "amber": 0.10},
    "rate_ratio": {"green": 0.90, "amber": 0.70},
    "rssi_dbm": {"green": -60.0, "amber": -70.0},
    "command_late_s": 1.0,
    "hysteresis_s": 1.0,
}
CONTROL_KEYS = {
    b"L": "left",
    b"R": "right",
    b"U": "up",
    b"D": "down",
    b"w": "grabber_up",
    b"s": "grabber_down",
    b"d": "claw_open",
    b"a": "claw_close",
}

WINDOW = "#1e1f22"
PANEL = "#2b2d31"
BORDER = "#3a3c41"
TEXT = "#dbdee1"
MUTED = "#949ba4"
BLUE = "#4f9cf9"
GREEN = "#3ba55d"
RED = "#ed4245"
AMBER = "#f0b232"


class LinkMonitor:
    def __init__(self):
        self.arrival_times = deque(maxlen=HISTORY_LENGTH)
        self.sequence_history = deque(maxlen=HISTORY_LENGTH)
        self.packet_count = 0
        self.total_sequence_gaps = 0
        self.last_packet_time = None
        self.latest_sequence = None
        self.latest_packet_has_sequence = False
        self.rssi_dbm = None
        self.cmd_age_s = None
        self.failsafe = None
        self.history = deque(maxlen=30)
        self.last_history_sample = None
        self.state = "waiting"
        self.better_state = None
        self.better_state_since = None

    def record(self, packet, received_at):
        self.packet_count += 1
        self.last_packet_time = received_at
        self.arrival_times.append(received_at)

        seq = packet.get("seq")
        self.latest_packet_has_sequence = (
            isinstance(seq, int)
            and not isinstance(seq, bool)
            and 0 <= seq <= 0xFFFFFFFF
        )
        if self.latest_packet_has_sequence:
            if self.latest_sequence is None:
                self.latest_sequence = seq
                self.sequence_history.append(seq)
            else:
                difference = (seq - self.latest_sequence) & 0xFFFFFFFF
                if 0 < difference < 0x80000000:
                    self.total_sequence_gaps += difference - 1
                    self.latest_sequence = seq
                    self.sequence_history.append(seq)

        rssi_dbm = packet.get("rssi_dbm")
        self.rssi_dbm = (
            float(rssi_dbm)
            if isinstance(rssi_dbm, (int, float))
            and not isinstance(rssi_dbm, bool)
            and math.isfinite(rssi_dbm)
            else None
        )
        cmd_age_s = packet.get("cmd_age_s")
        self.cmd_age_s = (
            float(cmd_age_s)
            if isinstance(cmd_age_s, (int, float))
            and not isinstance(cmd_age_s, bool)
            and math.isfinite(cmd_age_s)
            and cmd_age_s >= 0
            else None
        )
        failsafe = packet.get("failsafe")
        self.failsafe = failsafe if isinstance(failsafe, bool) else None

    def _rate(self, now):
        recent = [
            received_at
            for received_at in self.arrival_times
            if now - received_at <= 2.0
        ]
        if len(recent) >= 2 and recent[-1] > recent[0]:
            return (len(recent) - 1) / (recent[-1] - recent[0])
        if self.last_packet_time is not None and now - self.last_packet_time > 2.0:
            return 0.0
        return None

    def _loss(self):
        if not self.latest_packet_has_sequence or not self.sequence_history:
            return None
        sequences = list(self.sequence_history)
        gaps = 0
        expected = 0
        for previous, sequence in zip(sequences, sequences[1:]):
            difference = (sequence - previous) & 0xFFFFFFFF
            if 0 < difference < 0x80000000:
                gaps += difference - 1
                expected += difference
        return gaps / (expected + 1) if expected else 0.0

    def _target_state(self, age, rate, loss, rssi_dbm):
        severities = {"green": 0, "amber": 1, "red": 2}
        measures = []

        if age is not None:
            thresholds = LINK_THRESHOLDS["packet_age"]
            if age < thresholds["green"]:
                measures.append("green")
            elif age <= thresholds["amber"]:
                measures.append("amber")
            else:
                measures.append("red")

        if loss is not None:
            thresholds = LINK_THRESHOLDS["loss"]
            if loss < thresholds["green"]:
                measures.append("green")
            elif loss <= thresholds["amber"]:
                measures.append("amber")
            else:
                measures.append("red")

        if rate is not None:
            ratio = rate / EXPECTED_RATE_HZ
            thresholds = LINK_THRESHOLDS["rate_ratio"]
            if ratio > thresholds["green"]:
                measures.append("green")
            elif ratio >= thresholds["amber"]:
                measures.append("amber")
            else:
                measures.append("red")

        if rssi_dbm is not None:
            thresholds = LINK_THRESHOLDS["rssi_dbm"]
            if rssi_dbm > thresholds["green"]:
                measures.append("green")
            else:
                measures.append("amber")

        return max(measures, key=severities.get) if measures else "green"

    def snapshot(self, now):
        age = (
            max(0.0, now - self.last_packet_time)
            if self.last_packet_time is not None
            else None
        )
        rate = self._rate(now)
        loss = self._loss()

        if self.last_packet_time is None:
            self.state = "waiting"
            self.better_state = None
            self.better_state_since = None
        else:
            target = self._target_state(age, rate, loss, self.rssi_dbm)
            severity = {"green": 0, "amber": 1, "red": 2}
            current_severity = severity.get(self.state, -1)
            target_severity = severity[target]
            if target_severity > current_severity:
                self.state = target
                self.better_state = None
                self.better_state_since = None
            elif target_severity < current_severity:
                if target != self.better_state:
                    self.better_state = target
                    self.better_state_since = now
                elif now - self.better_state_since >= LINK_THRESHOLDS["hysteresis_s"]:
                    self.state = target
                    self.better_state = None
                    self.better_state_since = None
            else:
                self.better_state = None
                self.better_state_since = None

        if (
            self.last_history_sample is None
            or now - self.last_history_sample >= 1.0
        ):
            self.history.append((now, rate, loss))
            self.last_history_sample = now

        return {
            "state": self.state,
            "packet_count": self.packet_count,
            "total_sequence_gaps": self.total_sequence_gaps,
            "last_packet_age": age,
            "rate_hz": rate,
            "loss": loss,
            "rssi_dbm": self.rssi_dbm,
            "cmd_age_s": self.cmd_age_s,
            "failsafe": self.failsafe,
            "history": list(self.history),
            "sampled_at": now,
        }


def display_heading(angle):
    hx = -np.sin(angle)
    hy = np.cos(angle)
    return (np.degrees(np.arctan2(hx, hy)) - 90 + 360) % 360


def filter_position(x, y):
    return x, y


class DisplayData:
    def __init__(self):
        self.lock = threading.Lock()
        self.link_monitor = LinkMonitor()
        self.lidar_x = np.array([])
        self.lidar_y = np.array([])
        self.lidar_c = np.array([])
        self.ax_buffer = deque(maxlen=100)
        self.ay_buffer = deque(maxlen=100)
        self.az_buffer = deque(maxlen=100)
        self.imu_reading = None
        self.uwb_x = 0.0
        self.uwb_y = 0.0
        self.distances = [0.0] * 4
        self.heading = 0.0
        self.yaw = 0.0
        self.seq = None
        self.rssi_dbm = None
        self.packet_count = 0
        self.last_packet_time = None
        self.sender_address = None
        self.arrival_times = deque(maxlen=HISTORY_LENGTH)
        self.sequence_history = deque(maxlen=HISTORY_LENGTH)


def receiver_loop(shared_data, stop_event, udp_socket):
    failures = 0
    warned = False
    reassembler = TelemetryPacketReassembler()
    print(f"listening on port {UDP_PORT}", flush=True)

    while not stop_event.is_set():
        try:
            payload, sender_address = udp_socket.recvfrom(65535)
        except socket.timeout:
            with shared_data.lock:
                last_packet_time = shared_data.link_monitor.last_packet_time
            if (
                not warned
                and (
                    last_packet_time is None
                    or time.monotonic() - last_packet_time >= 5
                )
            ):
                print("no packets for 5 s", flush=True)
                warned = True
            continue
        except OSError as exc:
            if not stop_event.is_set():
                print(f"UDP receive failed: {exc}", flush=True)
            break

        try:
            received_at = time.monotonic()
            payload = reassembler.add(
                payload,
                received_at,
                source_key=sender_address,
            )
            if payload is None:
                continue
            packet = decode_packet(payload)
            lidar_x = np.asarray(packet["lidar_x"])
            lidar_y = np.asarray(packet["lidar_y"])
            lidar_c = np.asarray(packet["lidar_c"])
            distances = list(packet["distances"])
            imu_reading = parse_imu_payload(packet.get("imu"))
            seq = packet.get("seq")
            rssi_dbm = packet.get("rssi_dbm")
            with shared_data.lock:
                shared_data.link_monitor.record(packet, received_at)
                shared_data.lidar_x = lidar_x
                shared_data.lidar_y = lidar_y
                shared_data.lidar_c = lidar_c
                shared_data.imu_reading = imu_reading
                if imu_reading is None:
                    shared_data.ax_buffer.append(float(packet["ax"]))
                    shared_data.ay_buffer.append(float(packet["ay"]))
                    shared_data.az_buffer.append(float("nan"))
                else:
                    accel_x, accel_y, accel_z = imu_reading.accelerometer_g
                    shared_data.ax_buffer.append(accel_x)
                    shared_data.ay_buffer.append(accel_y)
                    shared_data.az_buffer.append(accel_z)
                shared_data.uwb_x = float(packet["uwb_x"])
                shared_data.uwb_y = float(packet["uwb_y"])
                shared_data.distances = distances
                shared_data.heading = float(packet["heading"])
                shared_data.yaw = float(packet["yaw"])
                shared_data.seq = (
                    seq
                    if isinstance(seq, int) and not isinstance(seq, bool)
                    else None
                )
                shared_data.rssi_dbm = (
                    float(rssi_dbm)
                    if isinstance(rssi_dbm, (int, float))
                    and math.isfinite(rssi_dbm)
                    else None
                )
                shared_data.packet_count = shared_data.link_monitor.packet_count
                shared_data.last_packet_time = received_at
                shared_data.sender_address = sender_address
                shared_data.arrival_times.append(received_at)
                if shared_data.seq is not None:
                    shared_data.sequence_history.append(shared_data.seq)
                packet_count = shared_data.link_monitor.packet_count
            warned = False

            if packet_count % PRINT_EVERY:
                continue
            distances_text = ", ".join(
                f"{value:.2f}" for value in packet["distances"]
            )
            age = time.time() - packet["timestamp"]
            print(
                f"#{packet_count} {len(payload)} B "
                f"lidar={len(packet['lidar_x'])} "
                f"uwb=({packet['uwb_x']:.2f}, {packet['uwb_y']:.2f}) "
                f"distances=[{distances_text}] "
                f"heading={packet['heading']:.1f} "
                f"yaw={packet['yaw']:.1f} age={age:.3f}s",
                flush=True,
            )
        except Exception:
            failures += 1
            if failures % 50 == 0:
                print(f"decode failure {failures} traceback:", flush=True)
                traceback.print_exc()


class CompassWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle = 0.0
        self.setFixedSize(150, 150)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def set_heading(self, angle):
        self.angle = angle
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = self.rect().center()
        radius = min(self.width(), self.height()) * 0.43

        painter.setPen(QPen(QColor(BORDER), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(center, int(radius), int(radius))

        for degrees in range(0, 360, 10):
            angle = math.radians(degrees)
            inner = radius * (0.88 if degrees % 30 == 0 else 0.93)
            outer_x = center.x() + math.sin(angle) * radius
            outer_y = center.y() - math.cos(angle) * radius
            inner_x = center.x() + math.sin(angle) * inner
            inner_y = center.y() - math.cos(angle) * inner
            painter.setPen(QPen(QColor(MUTED), 1))
            painter.drawLine(
                int(inner_x), int(inner_y), int(outer_x), int(outer_y)
            )

        painter.setPen(QColor(TEXT))
        for label, degrees in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
            angle = math.radians(degrees)
            x = center.x() + math.sin(angle) * radius * 0.70
            y = center.y() - math.cos(angle) * radius * 0.70
            painter.drawText(
                int(x - 6), int(y + 5), label
            )

        angle = math.radians(self.angle)
        painter.setPen(QPen(QColor(RED), 3))
        painter.drawLine(
            center,
            center + QPoint(
                int(math.sin(angle) * radius * 0.65),
                int(-math.cos(angle) * radius * 0.65),
            ),
        )
        painter.setPen(QPen(QColor(BLUE), 3))
        painter.drawLine(
            center,
            center + QPoint(
                int(-math.sin(angle) * radius * 0.28),
                int(math.cos(angle) * radius * 0.28),
            ),
        )


class ControllerSticksWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(190, 100)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.axes = (0.0, 0.0, 0.0, 0.0)
        self.active_sticks = (False, False)

    def set_sticks(self, axes, active_sticks):
        self.axes = axes
        self.active_sticks = active_sticks
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        radius = 29
        centers = (QPoint(47, 39), QPoint(143, 39))
        labels = ("Grabber / claw", "Movement")
        colors = (AMBER, BLUE)

        for index, center in enumerate(centers):
            active = self.active_sticks[index]
            color = QColor(colors[index] if active else BORDER)
            painter.setPen(QPen(color, 2))
            painter.setBrush(QColor("#222428"))
            painter.drawEllipse(center, radius, radius)

            painter.setPen(QPen(QColor(BORDER), 1))
            painter.drawLine(
                center.x() - radius + 5,
                center.y(),
                center.x() + radius - 5,
                center.y(),
            )
            painter.drawLine(
                center.x(),
                center.y() - radius + 5,
                center.x(),
                center.y() + radius - 5,
            )

            axis_index = index * 2
            knob_x = center.x() + round(self.axes[axis_index] * radius * 0.65)
            knob_y = center.y() + round(self.axes[axis_index + 1] * radius * 0.65)
            painter.setPen(QPen(color, 2))
            painter.setBrush(QColor(colors[index] if active else MUTED))
            painter.drawEllipse(QPoint(knob_x, knob_y), 8, 8)
            painter.setPen(QColor(TEXT))
            painter.drawText(
                QRect(center.x() - 47, 76, 94, 18),
                Qt.AlignmentFlag.AlignCenter,
                labels[index],
            )


class SquarePlotWidget(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        policy = QSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return width


class LidarPlot(SquarePlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setBackground(PANEL)
        self.showGrid(x=True, y=True, alpha=0.25)
        self.setLabel("bottom", "X", units="m")
        self.setLabel("left", "Y", units="m")
        self.setMouseEnabled(False, False)
        self.setXRange(-LIDAR_PLOT_SPAN / 2, LIDAR_PLOT_SPAN / 2, padding=0)
        self.setYRange(-LIDAR_PLOT_SPAN / 2, LIDAR_PLOT_SPAN / 2, padding=0)
        self.setAspectLocked(True, ratio=1)
        self.disableAutoRange()

        self.lidar_points = pg.ScatterPlotItem(size=3, pen=None)
        self.plotItem.addItem(self.lidar_points)
        self.heading_line = self.plot([], [], pen=pg.mkPen(RED, width=2))
        self.heading_head_left = self.plot([], [], pen=pg.mkPen(RED, width=2))
        self.heading_head_right = self.plot([], [], pen=pg.mkPen(RED, width=2))
        self.yaw_text = pg.TextItem(
            "Yaw = 0.0°", color=RED, anchor=(0, 1)
        )
        self.plotItem.addItem(self.yaw_text)
        self.yaw_text.setPos(
            -LIDAR_PLOT_SPAN / 2 + 0.1,
            -LIDAR_PLOT_SPAN / 2 + 0.1,
        )

    def update_data(self, lidar_x, lidar_y, lidar_c, yaw):
        lidar_x = np.asarray(lidar_x, dtype=float)
        lidar_y = np.asarray(lidar_y, dtype=float)
        lidar_c = np.asarray(lidar_c, dtype=float)
        point_count = min(len(lidar_x), len(lidar_y))
        lidar_x = lidar_x[:point_count]
        lidar_y = lidar_y[:point_count]
        confidence = (
            lidar_c[:point_count]
            if len(lidar_c) >= point_count
            else np.zeros(point_count)
        )
        confidence = np.clip(confidence, 0, 255) / 255.0
        colors = [
            QColor(
                int(int(BLUE[1:3], 16) * (1 - value) + int(RED[1:3], 16) * value),
                int(int(BLUE[3:5], 16) * (1 - value) + int(RED[3:5], 16) * value),
                int(int(BLUE[5:7], 16) * (1 - value) + int(RED[5:7], 16) * value),
            )
            for value in confidence
        ]
        self.lidar_points.setData(
            x=lidar_x, y=lidar_y, brush=colors, pen=None, size=3, symbol="o"
        )
        arrow_x = math.sin(yaw)
        arrow_y = math.cos(yaw)
        end_x = arrow_x
        end_y = arrow_y
        head_base_x = end_x - arrow_x * 0.18
        head_base_y = end_y - arrow_y * 0.18
        head_side_x = math.cos(yaw) * 0.08
        head_side_y = -math.sin(yaw) * 0.08
        self.heading_line.setData([0, end_x], [0, end_y])
        self.heading_head_left.setData(
            [end_x, head_base_x + head_side_x],
            [end_y, head_base_y + head_side_y],
        )
        self.heading_head_right.setData(
            [end_x, head_base_x - head_side_x],
            [end_y, head_base_y - head_side_y],
        )
        self.yaw_text.setText(f"Yaw = {math.degrees(yaw):.1f}°")


class UwbPlot(SquarePlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setBackground(PANEL)
        self.showGrid(x=True, y=True, alpha=0.25)
        self.setLabel("bottom", "X", units="m")
        self.setLabel("left", "Y", units="m")
        self.setMouseEnabled(False, False)
        self.setXRange(0, PLOT_SPAN, padding=0)
        self.setYRange(0, PLOT_SPAN, padding=0)
        self.setAspectLocked(True, ratio=1)
        self.disableAutoRange()
        self.raw_path = self.plot(
            [], [], pen=pg.mkPen(AMBER, width=1.5, style=Qt.PenStyle.DashLine)
        )
        self.filtered_path = self.plot([], [], pen=pg.mkPen(BLUE, width=2))
        self.anchor_markers = self.plot(
            [float(anchor[0]) for anchor in anchors],
            [float(anchor[1]) for anchor in anchors],
            pen=None,
            symbol="o",
            symbolSize=6,
            symbolBrush=TEXT,
            symbolPen=pg.mkPen(BORDER),
        )
        self.anchor_circles = [
            self.plot([], [], pen=pg.mkPen(RED, width=0.8))
            for _ in range(4)
        ]
        self.current_position = self.plot(
            [], [], pen=None, symbol="o", symbolSize=8,
            symbolBrush=RED, symbolPen=pg.mkPen(RED)
        )

    def update_data(
        self, distances, raw_x, raw_y, filtered_x, filtered_y, x, y,
        show_current=True,
    ):
        self.raw_path.setData(raw_x, raw_y)
        self.filtered_path.setData(filtered_x, filtered_y)
        angles = np.linspace(0, 2 * np.pi, 65)
        for index, circle in enumerate(self.anchor_circles):
            radius = float(distances[index]) if index < len(distances) else 0.0
            anchor_x = float(anchors[index][0])
            anchor_y = float(anchors[index][1])
            circle.setData(
                anchor_x + radius * np.cos(angles),
                anchor_y + radius * np.sin(angles),
            )
        if show_current:
            self.current_position.setData([x], [y])
        else:
            self.current_position.setData([], [])


class CameraStream(QObject):
    frame_ready = Signal(QImage)
    status_changed = Signal(str)

    def __init__(self, base_url=CAMERA_STREAM_URL):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(
            target=self._run,
            name="camera-stream",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def _open_stream(self):
        errors = []
        for path in ("/video_feed", "/"):
            try:
                response = urlopen(self.base_url + path, timeout=2)
            except OSError as error:
                errors.append(f"{path}: {error}")
                continue
            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type.lower():
                response.close()
                errors.append(f"{path}: returned an HTML page")
                continue
            return response
        raise OSError("; ".join(errors))

    def _read_stream(self, response):
        buffer = bytearray()
        while not self._stop_event.is_set():
            chunk = response.read1(4096)
            if not chunk:
                return
            buffer.extend(chunk)
            for encoded_frame in extract_jpeg_frames(buffer):
                image = QImage.fromData(encoded_frame)
                if not image.isNull():
                    self.frame_ready.emit(image)

            if len(buffer) > 8 * 1024 * 1024:
                buffer.clear()

    def _run(self):
        self.status_changed.emit("Connecting to camera...")
        while not self._stop_event.is_set():
            try:
                with self._open_stream() as response:
                    self.status_changed.emit("Camera connected; waiting for frames...")
                    self._read_stream(response)
                    if not self._stop_event.is_set():
                        raise OSError("camera stream ended")
            except OSError as error:
                if self._stop_event.is_set():
                    break
                self.status_changed.emit(
                    f"Camera unavailable: {error}. Retrying..."
                )
                self._stop_event.wait(1)


class CameraView(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("cameraView")
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self._frame = None
        self.setText("Connecting to camera...")

    def sizeHint(self):
        return QSize(300, 300)

    def set_frame(self, image):
        if image.isNull():
            return
        self._frame = QPixmap.fromImage(image)
        self.setText("")
        self._update_frame()

    def set_status(self, message):
        self._frame = None
        self.setPixmap(QPixmap())
        self.setText(message)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_frame()

    def _update_frame(self):
        if self._frame is not None:
            self.setPixmap(
                self._frame.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )


class StaleOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("staleOverlay")
        self.setStyleSheet(
            "QWidget#staleOverlay { background-color: rgba(18, 19, 21, 210); }"
            "QLabel { background: transparent; color: #dbdee1; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        self.message = QLabel("No data")
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.hide()

    def set_age(self, age):
        self.message.setText(f"No data, last update {age:.1f} s ago")


class HealthPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.stale_overlay = StaleOverlay(self)

    def set_link_state(self, state, age):
        if state == "red" and age is not None:
            self.stale_overlay.set_age(age)
            self.stale_overlay.setGeometry(self.rect())
            self.stale_overlay.raise_()
            self.stale_overlay.show()
        else:
            self.stale_overlay.hide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.stale_overlay.setGeometry(self.rect())


class LinkDetailsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Link details")
        self.setModal(False)
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        self.rate_plot = pg.PlotWidget()
        self.rate_plot.setBackground(PANEL)
        self.rate_plot.setTitle("Packet rate (Hz)")
        self.rate_plot.showGrid(x=True, y=True, alpha=0.25)
        self.rate_plot.setLabel("bottom", "Seconds ago")
        self.rate_plot.setXRange(-30, 0, padding=0)
        self.rate_line = self.rate_plot.plot([], [], pen=pg.mkPen(BLUE, width=2))
        layout.addWidget(self.rate_plot)

        self.loss_plot = pg.PlotWidget()
        self.loss_plot.setBackground(PANEL)
        self.loss_plot.setTitle("Packet loss (%)")
        self.loss_plot.showGrid(x=True, y=True, alpha=0.25)
        self.loss_plot.setLabel("bottom", "Seconds ago")
        self.loss_plot.setLabel("left", "Loss", units="%")
        self.loss_plot.setXRange(-30, 0, padding=0)
        self.loss_plot.setYRange(0, 100, padding=0)
        self.loss_line = self.loss_plot.plot(
            [], [], pen=pg.mkPen(AMBER, width=2)
        )
        layout.addWidget(self.loss_plot)

        self.details = {}
        for name, title in (
            ("packet_count", "Packets received"),
            ("total_sequence_gaps", "Total sequence gaps"),
            ("last_packet_age", "Last packet age"),
            ("rssi_dbm", "RSSI"),
            ("cmd_age_s", "Command age"),
            ("failsafe", "Pi failsafe"),
            ("pi_ip", "Pi IP"),
            ("pc_ip", "PC IP"),
        ):
            row = QHBoxLayout()
            label = QLabel(title)
            value = QLabel("--")
            value.setAlignment(Qt.AlignmentFlag.AlignRight)
            row.addWidget(label)
            row.addWidget(value)
            layout.addLayout(row)
            self.details[name] = value

    def update_details(self, snapshot, pi_ip, pc_ip):
        now = snapshot["sampled_at"]
        history = snapshot["history"]
        rate_points = [
            (sample_time - now, rate)
            for sample_time, rate, _loss in history
            if rate is not None
        ]
        loss_points = [
            (sample_time - now, loss * 100)
            for sample_time, _rate, loss in history
            if loss is not None
        ]
        self.rate_line.setData(
            [point[0] for point in rate_points],
            [point[1] for point in rate_points],
        )
        self.loss_line.setData(
            [point[0] for point in loss_points],
            [point[1] for point in loss_points],
        )

        age = snapshot["last_packet_age"]
        rssi = snapshot["rssi_dbm"]
        cmd_age = snapshot["cmd_age_s"]
        self.details["packet_count"].setText(str(snapshot["packet_count"]))
        self.details["total_sequence_gaps"].setText(
            str(snapshot["total_sequence_gaps"])
        )
        self.details["last_packet_age"].setText(
            f"{age:.2f} s" if age is not None else "--"
        )
        self.details["rssi_dbm"].setText(
            f"{rssi:.0f} dBm" if rssi is not None else "--"
        )
        self.details["cmd_age_s"].setText(
            f"{cmd_age:.2f} s" if cmd_age is not None else "--"
        )
        failsafe = snapshot["failsafe"]
        self.details["failsafe"].setText(
            "STOPPED" if failsafe is True else "Clear" if failsafe is False else "--"
        )
        self.details["pi_ip"].setText(pi_ip)
        self.details["pc_ip"].setText(pc_ip)


class MainWindow(QMainWindow):
    def __init__(self, pi_ip=DEFAULT_PI_IP):
        super().__init__()
        self.setWindowTitle("UWB / IMU / Lidar Dashboard")
        self.setMinimumSize(1200, 700)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.shared_data = DisplayData()
        self.stop_event = threading.Event()
        self.controller_joystick = None
        self.held = set()
        self.controller_held = set()
        self.estop_active = False
        self.last_command = ""
        self.control_sequence = 0
        self.connection_generation = 0
        self.pi_ip = pi_ip
        self.raw_x = []
        self.raw_y = []
        self.filtered_x = []
        self.filtered_y = []
        self.latest_x = 0.0
        self.latest_y = 0.0
        self.heading_degrees = 0.0
        self.yaw = 0.0
        self.pc_ip = self._get_local_ip(self.pi_ip)
        self.displayed_link_state = "waiting"
        self.original_title = "UWB / IMU / Lidar Dashboard"
        self.health_panels = []

        self.receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver_socket.bind(("0.0.0.0", UDP_PORT))
        self.receiver_socket.settimeout(0.25)
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.handshake = ControlHandshake(
            self.control_socket,
            self.pi_ip,
            self._update_connection_status,
        )

        self._build_ui()
        self.camera_stream = CameraStream()
        self.camera_stream.frame_ready.connect(self.set_camera_frame)
        self.camera_stream.status_changed.connect(self.camera_view.set_status)
        self.camera_stream.start()
        QApplication.instance().installEventFilter(self)
        self._init_controller()
        self.link_details = LinkDetailsDialog(self)
        self.receiver = threading.Thread(
            target=receiver_loop,
            args=(self.shared_data, self.stop_event, self.receiver_socket),
            daemon=True,
        )
        self.receiver.start()

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_dashboard)
        self.update_timer.start(33)
        self.control_timer = QTimer(self)
        self.control_timer.timeout.connect(self.send_held_commands)
        self.control_timer.start(round(1000 / CONTROL_RATE_HZ))

    @staticmethod
    def _get_local_ip(pi_ip):
        address_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            address_socket.connect((pi_ip, CONTROL_PORT))
            return address_socket.getsockname()[0]
        except OSError:
            return "--"
        finally:
            address_socket.close()

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 8)
        root.setSpacing(8)

        top_section = QWidget()
        top_section.setFixedHeight(190)
        top_bar = QHBoxLayout(top_section)
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(10)

        compass_panel = QFrame()
        compass_panel.setObjectName("panel")
        compass_panel.setMinimumWidth(190)
        compass_layout = QVBoxLayout(compass_panel)
        compass_layout.setContentsMargins(8, 4, 8, 4)
        compass_layout.setSpacing(0)
        self.compass = CompassWidget()
        compass_layout.addWidget(self.compass, 0, Qt.AlignmentFlag.AlignHCenter)
        self.heading_label = QLabel("Heading 0.0°")
        self.heading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        compass_layout.addWidget(self.heading_label)
        top_bar.addWidget(compass_panel, 0)

        acceleration_panel = QFrame()
        acceleration_panel.setObjectName("panel")
        acceleration_panel.setFixedWidth(370)
        acceleration_layout = QVBoxLayout(acceleration_panel)
        acceleration_layout.setContentsMargins(8, 4, 8, 6)
        acceleration_layout.setSpacing(2)
        acceleration_title = QLabel("Sense HAT IMU")
        acceleration_title.setObjectName("panelTitle")
        acceleration_layout.addWidget(acceleration_title)
        self.imu_values_label = QLabel("Waiting for IMU telemetry")
        self.imu_values_label.setObjectName("imuValues")
        acceleration_layout.addWidget(self.imu_values_label)
        self.acceleration_plot = pg.PlotWidget()
        self.acceleration_plot.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.acceleration_plot.setBackground(PANEL)
        self.acceleration_plot.showGrid(x=True, y=True, alpha=0.25)
        self.acceleration_plot.setLabel("bottom", "Sample")
        self.acceleration_plot.setLabel("left", "Acceleration", units="g")
        self.acceleration_plot.setYRange(-2, 2, padding=0)
        self.acceleration_plot.addLegend()
        self.acceleration_x_line = self.acceleration_plot.plot(
            [], [], pen=pg.mkPen(BLUE, width=1.5), name="X"
        )
        self.acceleration_y_line = self.acceleration_plot.plot(
            [], [], pen=pg.mkPen(RED, width=1.5), name="Y"
        )
        self.acceleration_z_line = self.acceleration_plot.plot(
            [], [], pen=pg.mkPen(GREEN, width=1.5), name="Z"
        )
        acceleration_layout.addWidget(self.acceleration_plot, 1)
        top_bar.addWidget(acceleration_panel, 0)

        controls_panel = QFrame()
        controls_panel.setObjectName("panel")
        controls_layout = QHBoxLayout(controls_panel)
        controls_layout.setContentsMargins(8, 6, 8, 6)
        controls_layout.setSpacing(12)
        self.key_indicators = {}
        self._add_keypad(
            controls_layout,
            "Grabber / claw",
            (
                (None, (b"w", "W"), None),
                ((b"a", "A"), (b"s", "S"), (b"d", "D")),
            ),
            "actionKey",
        )
        self._add_keypad(
            controls_layout,
            "Movement",
            (
                (None, (b"U", "↑"), None),
                ((b"L", "←"), (b"D", "↓"), (b"R", "→")),
            ),
            "arrowKey",
        )
        self.controller_sticks = ControllerSticksWidget()
        controls_layout.addWidget(self.controller_sticks)
        top_bar.addWidget(controls_panel, 0)

        top_bar.addStretch(1)
        status_block = QVBoxLayout()
        status_block.setSpacing(3)
        status_block.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.estop_button = QPushButton("E-STOP CLEAR")
        self.estop_button.setObjectName("estopButton")
        self.estop_button.setProperty("active", "false")
        self.estop_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.estop_button.clicked.connect(self._trigger_estop)
        status_block.addWidget(self.estop_button, 0, Qt.AlignmentFlag.AlignRight)
        self.arm_button = QPushButton("DISARMED")
        self.arm_button.setObjectName("armButton")
        self.arm_button.setCheckable(True)
        self.arm_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.arm_button.toggled.connect(self._set_armed)
        status_block.addWidget(self.arm_button, 0, Qt.AlignmentFlag.AlignRight)
        shortcut_label = QLabel("Enter: E-stop | Space: reset / arm")
        shortcut_label.setObjectName("mutedText")
        status_block.addWidget(shortcut_label, 0, Qt.AlignmentFlag.AlignRight)
        link_status_row = QHBoxLayout()
        link_status_row.setSpacing(6)
        self.link_status_button = QPushButton()
        self.link_status_button.setObjectName("linkStatusButton")
        self.link_status_button.setFlat(True)
        status_button_layout = QHBoxLayout(self.link_status_button)
        status_button_layout.setContentsMargins(2, 0, 2, 0)
        status_button_layout.setSpacing(5)
        self.link_dot = QLabel()
        self.link_dot.setObjectName("linkDot")
        self.link_dot.setFixedSize(10, 10)
        self.link_word = QLabel("Waiting for data")
        self.link_word.setObjectName("linkWord")
        status_button_layout.addWidget(self.link_dot)
        status_button_layout.addWidget(self.link_word)
        self.link_status_button.clicked.connect(self._show_link_details)
        link_status_row.addWidget(self.link_status_button)
        self.link_metric_label = QLabel("--")
        self.link_metric_label.setObjectName("mutedText")
        link_status_row.addWidget(self.link_metric_label)
        status_block.addLayout(link_status_row)
        self.command_health_label = QLabel("--")
        self.command_health_label.setObjectName("mutedText")
        status_block.addWidget(
            self.command_health_label, 0, Qt.AlignmentFlag.AlignRight
        )
        ip_label = QLabel(f"Pi: {self.pi_ip} | PC: {self.pc_ip}")
        ip_label.setObjectName("mutedText")
        status_block.addWidget(ip_label, 0, Qt.AlignmentFlag.AlignRight)
        self.sender_label = QLabel("Waiting for data")
        self.sender_label.setObjectName("waitingStatus")
        status_block.addWidget(self.sender_label, 0, Qt.AlignmentFlag.AlignRight)
        self.command_label = QLabel("Command: idle")
        self.command_label.setObjectName("mutedText")
        status_block.addWidget(self.command_label, 0, Qt.AlignmentFlag.AlignRight)
        self.connection_status_label = QLabel("Connecting")
        self.connection_status_label.setObjectName("connectionStatus")
        status_block.addWidget(
            self.connection_status_label, 0, Qt.AlignmentFlag.AlignRight
        )
        top_bar.addLayout(status_block, 0)
        root.addWidget(top_section, 0)

        bottom_section = QWidget()
        bottom_layout = QHBoxLayout(bottom_section)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(10)

        lidar_panel = HealthPanel()
        lidar_panel.setObjectName("panel")
        self.health_panels.append(lidar_panel)
        lidar_panel.setMinimumSize(300, 300)
        lidar_layout = QVBoxLayout(lidar_panel)
        lidar_layout.setContentsMargins(10, 8, 10, 10)
        lidar_layout.setSpacing(4)
        lidar_title = QLabel("Lidar")
        lidar_title.setObjectName("panelTitle")
        lidar_layout.addWidget(lidar_title)
        self.lidar_plot = LidarPlot()
        lidar_layout.addWidget(self.lidar_plot, 1)
        bottom_layout.addWidget(lidar_panel, 1)

        uwb_panel = HealthPanel()
        uwb_panel.setObjectName("panel")
        self.health_panels.append(uwb_panel)
        uwb_panel.setMinimumSize(300, 300)
        uwb_layout = QVBoxLayout(uwb_panel)
        uwb_layout.setContentsMargins(10, 8, 10, 10)
        uwb_layout.setSpacing(4)
        uwb_title = QLabel("UWB position")
        uwb_title.setObjectName("panelTitle")
        uwb_layout.addWidget(uwb_title)
        self.uwb_plot = UwbPlot()
        uwb_layout.addWidget(self.uwb_plot, 1)
        bottom_layout.addWidget(uwb_panel, 1)

        camera_panel = HealthPanel()
        camera_panel.setObjectName("panel")
        self.health_panels.append(camera_panel)
        camera_panel.setMinimumSize(300, 300)
        camera_layout = QVBoxLayout(camera_panel)
        camera_layout.setContentsMargins(10, 8, 10, 10)
        camera_layout.setSpacing(4)
        camera_title = QLabel("Camera")
        camera_title.setObjectName("panelTitle")
        camera_layout.addWidget(camera_title)
        self.camera_view = CameraView()
        camera_layout.addWidget(self.camera_view, 1)
        bottom_layout.addWidget(camera_panel, 1)
        root.addWidget(bottom_section, 1)
        self.setCentralWidget(central)

        status = self.statusBar()
        status.setObjectName("dashboardStatus")
        self.position_status = QLabel("UWB: --, -- m")
        self.anchor_status = QLabel("Anchors: 4/4")
        self.packet_status = QLabel("Packets: 0")
        self.controller_status = QLabel("Controller: checking")
        self.controller_axes = QLabel("Axes: --")
        self.controller_buttons = QLabel("Buttons: --")
        status.addWidget(self.position_status)
        status.addWidget(self.anchor_status)
        status.addWidget(self.packet_status)
        status.addWidget(self.controller_status)
        status.addWidget(self.controller_axes)
        status.addWidget(self.controller_buttons)

    def _init_controller(self):
        if pygame is None:
            self.controller_status.setText("Controller: pygame unavailable")
            return
        try:
            pygame.display.init()
            pygame.joystick.init()
            self._update_controller_status()
        except pygame.error as exc:
            self.controller_status.setText(f"Controller: unavailable ({exc})")

    def _update_controller_status(self):
        if pygame is None:
            return
        try:
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.JOYDEVICEREMOVED:
                    self.controller_joystick = None

            if self.controller_joystick is None and pygame.joystick.get_count():
                self.controller_joystick = pygame.joystick.Joystick(0)
                self.controller_joystick.init()

            joystick = self.controller_joystick
            if joystick is None:
                self.controller_status.setText("Controller: not detected")
                self.controller_axes.setText("Axes: --")
                self.controller_buttons.setText("Buttons: --")
                self._set_controller_input([])
                return

            self.controller_status.setText(
                f"Controller: {joystick.get_name()}"
            )
            axis_values = [
                joystick.get_axis(index)
                for index in range(joystick.get_numaxes())
            ]
            axes = [
                f"{index}={value:+.2f}"
                for index, value in enumerate(axis_values)
            ]
            pressed = [
                str(index)
                for index in range(joystick.get_numbuttons())
                if joystick.get_button(index)
            ]
            self.controller_axes.setText(
                "Axes: " + (" ".join(axes) if axes else "--")
            )
            self.controller_buttons.setText(
                "Buttons: " + (", ".join(pressed) if pressed else "none")
            )
            self._set_controller_input(axis_values)
        except pygame.error:
            if self.controller_joystick is not None:
                try:
                    self.controller_joystick.quit()
                except pygame.error:
                    pass
            self.controller_joystick = None
            self.controller_status.setText("Controller: disconnected")
            self.controller_axes.setText("Axes: --")
            self.controller_buttons.setText("Buttons: --")
            self._set_controller_input([])

    def _set_controller_input(self, axes):
        stick_axes = tuple((list(axes[:4]) + [0.0] * 4)[:4])
        commands = controller_commands(stick_axes)
        movement_commands = {b"L", b"R", b"U", b"D"}
        grabber_commands = {b"w", b"s", b"a", b"d"}
        self.controller_sticks.set_sticks(
            stick_axes,
            (
                bool(commands & grabber_commands),
                bool(commands & movement_commands),
            ),
        )
        active_commands = commands if self.arm_button.isChecked() else set()
        if active_commands != self.controller_held:
            self.controller_held = active_commands
            self._update_command_label()

    def _add_keypad(self, parent_layout, title, rows, key_style):
        keypad = QVBoxLayout()
        keypad.setSpacing(3)
        title_label = QLabel(title)
        title_label.setObjectName("controlGroupTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        keypad.addWidget(title_label)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(3)
        for row_index, row in enumerate(rows):
            for column_index, key_info in enumerate(row):
                if key_info is None:
                    continue
                key, label_text = key_info
                indicator = QLabel(label_text)
                indicator.setObjectName("controlKey")
                indicator.setProperty("keyStyle", key_style)
                indicator.setProperty("pressed", "false")
                indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
                indicator.setFixedSize(30, 28)
                self.key_indicators[key] = indicator
                grid.addWidget(indicator, row_index, column_index)
        keypad.addLayout(grid)
        parent_layout.addLayout(keypad)

    def _set_armed(self, armed):
        if armed and self.estop_active:
            self.arm_button.setChecked(False)
            return
        self.arm_button.setText("ARMED" if armed else "DISARMED")
        self.arm_button.setProperty("armed", "true" if armed else "false")
        self.arm_button.style().unpolish(self.arm_button)
        self.arm_button.style().polish(self.arm_button)
        if not armed:
            self._clear_held()
            self._send_control_state()

    def _trigger_estop(self):
        if self.estop_active:
            return
        self.estop_active = True
        self.estop_button.setText("E-STOP ACTIVE — Space to reset")
        self.estop_button.setProperty("active", "true")
        self.estop_button.style().unpolish(self.estop_button)
        self.estop_button.style().polish(self.estop_button)
        if self.arm_button.isChecked():
            self.arm_button.setChecked(False)
        else:
            self._clear_held()
            self._send_control_state()

    def _reset_estop(self):
        if not self.estop_active:
            return
        self.estop_active = False
        self.estop_button.setText("E-STOP CLEAR")
        self.estop_button.setProperty("active", "false")
        self.estop_button.style().unpolish(self.estop_button)
        self.estop_button.style().polish(self.estop_button)

    def _clear_held(self):
        was_armed = self.arm_button.isChecked()
        self.held.clear()
        self.controller_held.clear()
        self._update_command_label()
        if was_armed:
            self._send_control_state()

    def _update_command_label(self):
        command = "+".join(
            key.decode("ascii")
            for key in (b"L", b"R", b"U", b"D", b"w", b"s", b"d", b"a")
            if key in self.held or key in self.controller_held
        )
        for key, indicator in self.key_indicators.items():
            pressed = key in self.held
            indicator.setProperty("pressed", "true" if pressed else "false")
            indicator.style().unpolish(indicator)
            indicator.style().polish(indicator)
        if command == self.last_command:
            return
        self.last_command = command
        self.command_label.setText(f"Command: {command or 'idle'}")
        print(command or "released", flush=True)
        if self.arm_button.isChecked():
            self._send_control_state()

    def send_held_commands(self):
        connected = self.handshake.poll()
        if connected and self.handshake.generation != self.connection_generation:
            self.connection_generation = self.handshake.generation
            self.control_sequence = 0
        if not connected:
            return
        if self.estop_active:
            self._send_control_state()
            return
        if not self.arm_button.isChecked():
            return
        self._send_control_state()

    def _send_control_state(self):
        if not self.handshake.connected:
            return
        keys = set()
        if not self.estop_active:
            keys = {
                CONTROL_KEYS[key]
                for key in self.held | self.controller_held
            }
        packet = pack_control_packet(
            self.control_sequence,
            time.perf_counter(),
            keys,
        )
        for destination in control_destinations(self.pi_ip):
            try:
                self.control_socket.sendto(packet, destination)
            except OSError as exc:
                print(
                    f"control UDP send to {destination[0]}:{destination[1]} "
                    f"failed: {exc}",
                    flush=True,
                )
        self.control_sequence = (self.control_sequence + 1) & 0xFFFFFFFF

    def _show_link_details(self):
        self.link_details.show()
        self.link_details.raise_()
        self.update_dashboard()

    def _update_connection_status(self, state, _message):
        self.connection_status_label.setText(state)
        colors = {
            "Connecting": AMBER,
            "Connected": GREEN,
            "Failed": RED,
        }
        self.connection_status_label.setStyleSheet(
            f"color: {colors[state]};"
        )

    def _update_link_health(self, snapshot):
        state = snapshot["state"]
        state_labels = {
            "waiting": "Waiting for data",
            "green": "Connected",
            "amber": "Degraded",
            "red": "Link lost",
        }
        state_colors = {
            "waiting": MUTED,
            "green": GREEN,
            "amber": AMBER,
            "red": RED,
        }
        color = state_colors[state]
        self.link_dot.setStyleSheet(
            f"background-color: {color}; border-radius: 5px;"
        )
        self.link_word.setText(state_labels[state])
        self.link_word.setStyleSheet(f"color: {color};")

        if state in ("green", "amber"):
            loss = snapshot["loss"]
            metric_text = f"{loss * 100:.1f}% loss" if loss is not None else "--"
        elif state == "red":
            age = snapshot["last_packet_age"]
            metric_text = (
                f"last packet {age:.1f} s ago" if age is not None else "--"
            )
        else:
            metric_text = "--"
        self.link_metric_label.setText(metric_text)

        cmd_age = snapshot["cmd_age_s"]
        if snapshot["failsafe"] is True:
            command_text = "Pi failsafe: STOPPED"
            command_color = RED
        elif cmd_age is None:
            command_text = "--"
            command_color = MUTED
        elif cmd_age < COMMAND_TIMEOUT_S or (
            (
                not self.arm_button.isChecked()
                or not (self.held | self.controller_held)
            )
            and not snapshot["failsafe"]
        ):
            command_text = "Commands OK"
            command_color = GREEN
        elif (
            self.arm_button.isChecked()
            and (self.held or self.controller_held)
            and cmd_age > LINK_THRESHOLDS["command_late_s"]
        ):
            command_text = "Commands not reaching Pi"
            command_color = AMBER
        else:
            command_text = "Commands delayed"
            command_color = AMBER
        self.command_health_label.setText(command_text)
        self.command_health_label.setStyleSheet(f"color: {command_color};")

        for panel in self.health_panels:
            panel.set_link_state(state, snapshot["last_packet_age"])

        if state == "red":
            self.setWindowTitle("LINK LOST: Rescue vehicle control")
        else:
            self.setWindowTitle(self.original_title)

        previous = self.displayed_link_state
        if state != previous:
            age = snapshot["last_packet_age"]
            loss = snapshot["loss"]
            rate = snapshot["rate_hz"]
            rssi = snapshot["rssi_dbm"]
            if (
                state == "red"
                and age is not None
                and age > LINK_THRESHOLDS["packet_age"]["amber"]
            ):
                reason = f"last packet {age:.1f} s ago"
            elif (
                state == "red"
                and loss is not None
                and loss > LINK_THRESHOLDS["loss"]["amber"]
            ):
                reason = f"loss {loss * 100:.1f}%"
            elif (
                state == "red"
                and rate is not None
                and rate / EXPECTED_RATE_HZ
                < LINK_THRESHOLDS["rate_ratio"]["amber"]
            ):
                reason = f"rate {rate:.1f} Hz"
            elif age is not None and age >= LINK_THRESHOLDS["packet_age"]["green"]:
                reason = f"last packet {age:.1f} s ago"
            elif loss is not None and loss >= LINK_THRESHOLDS["loss"]["green"]:
                reason = f"loss {loss * 100:.1f}%"
            elif rate is not None and (
                rate / EXPECTED_RATE_HZ
                <= LINK_THRESHOLDS["rate_ratio"]["green"]
            ):
                reason = f"rate {rate:.1f} Hz"
            elif rssi is not None and (
                rssi <= LINK_THRESHOLDS["rssi_dbm"]["green"]
            ):
                reason = f"RSSI {rssi:.0f} dBm"
            else:
                reason = "measurements healthy"
            print(f"link: {previous} -> {state} ({reason})", flush=True)
            self.displayed_link_state = state
            if (
                state == "red"
                and previous != "red"
                and AUTO_DISARM_ON_LINK_LOST
                and self.arm_button.isChecked()
            ):
                self.arm_button.setChecked(False)
                print("AUTO-DISARMED: link lost", flush=True)

        if self.link_details.isVisible():
            self.link_details.update_details(snapshot, self.pi_ip, self.pc_ip)

    def _fit_square_plots(self):
        for plot in (self.lidar_plot, self.uwb_plot):
            view_box = plot.getViewBox().geometry()
            if view_box.width() <= 0 or view_box.height() <= 0:
                continue
            height_adjustment = round(view_box.width() - view_box.height())
            if height_adjustment:
                plot.setFixedHeight(
                    max(1, plot.height() + height_adjustment)
                )

    def update_dashboard(self):
        self._fit_square_plots()
        self._update_controller_status()
        now = time.monotonic()
        with self.shared_data.lock:
            snapshot = self.shared_data.link_monitor.snapshot(now)
            lidar_x = self.shared_data.lidar_x.copy()
            lidar_y = self.shared_data.lidar_y.copy()
            lidar_c = self.shared_data.lidar_c.copy()
            distances = self.shared_data.distances.copy()
            ax_buffer = list(self.shared_data.ax_buffer)
            ay_buffer = list(self.shared_data.ay_buffer)
            az_buffer = list(self.shared_data.az_buffer)
            imu_reading = self.shared_data.imu_reading
            uwb_x = self.shared_data.uwb_x
            uwb_y = self.shared_data.uwb_y
            heading = self.shared_data.heading
            raw_yaw = self.shared_data.yaw
            sender_address = self.shared_data.sender_address

        self._update_link_health(snapshot)
        if snapshot["last_packet_age"] is None:
            return

        link_lost = snapshot["state"] == "red"
        if not link_lost:
            self.raw_x.append(uwb_x)
            self.raw_y.append(uwb_y)
            filtered_x, filtered_y = filter_position(uwb_x, uwb_y)
            self.filtered_x.append(filtered_x)
            self.filtered_y.append(filtered_y)
        self.latest_x = uwb_x
        self.latest_y = uwb_y
        self.heading_degrees = display_heading(math.radians(heading))
        self.yaw = (raw_yaw - (np.pi / 2) + (2 * np.pi)) % (2 * np.pi)

        self.acceleration_x_line.setData(
            np.arange(len(ax_buffer)), ax_buffer
        )
        self.acceleration_y_line.setData(
            np.arange(len(ay_buffer)), ay_buffer
        )
        self.acceleration_z_line.setData(
            np.arange(len(az_buffer)), az_buffer
        )
        self.imu_values_label.setText(format_imu_reading(imu_reading))
        sample_count = max(len(ax_buffer), len(ay_buffer), len(az_buffer))
        if sample_count:
            self.acceleration_plot.setXRange(0, sample_count, padding=0)

        self.compass.set_heading(self.heading_degrees)
        self.heading_label.setText(f"Heading {self.heading_degrees:.1f}°")
        self.lidar_plot.update_data(
            lidar_x,
            lidar_y,
            lidar_c,
            self.yaw,
        )
        self.uwb_plot.update_data(
            distances,
            self.raw_x,
            self.raw_y,
            self.filtered_x,
            self.filtered_y,
            uwb_x,
            uwb_y,
            show_current=not link_lost,
        )
        self.position_status.setText(f"UWB: {uwb_x:.2f}, {uwb_y:.2f} m")
        self.packet_status.setText(f"Packets: {snapshot['packet_count']}")
        if sender_address is not None:
            self.sender_label.setText(
                f"Receiving from {sender_address[0]}:{sender_address[1]}"
            )
            self.sender_label.setObjectName("connectedStatus")
            self.sender_label.style().unpolish(self.sender_label)
            self.sender_label.style().polish(self.sender_label)

    def set_camera_frame(self, qimage):
        self.camera_view.set_frame(qimage)

    def keyPressEvent(self, event):
        if self._handle_control_key_press(event):
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if self._handle_control_key_release(event):
            return
        super().keyReleaseEvent(event)

    def eventFilter(self, watched, event):
        if (
            isinstance(watched, QWidget)
            and (watched is self or self.isAncestorOf(watched))
            and self.isActiveWindow()
        ):
            if event.type() == QEvent.Type.KeyPress:
                return self._handle_control_key_press(event)
            if event.type() == QEvent.Type.KeyRelease:
                return self._handle_control_key_release(event)
        return super().eventFilter(watched, event)

    def _handle_control_key_press(self, event):
        if event.isAutoRepeat():
            return True
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._trigger_estop()
            event.accept()
            return True
        if event.key() == Qt.Key.Key_Space:
            if self.estop_active:
                self._reset_estop()
                event.accept()
                return True
            self._clear_held()
            self.arm_button.setChecked(not self.arm_button.isChecked())
            event.accept()
            return True
        key = self._control_key(event.key())
        if key is not None and self.arm_button.isChecked():
            self.held.add(key)
            self._update_command_label()
            event.accept()
            return True
        return False

    def _handle_control_key_release(self, event):
        if event.isAutoRepeat():
            return True
        key = self._control_key(event.key())
        if key is not None and key in self.held:
            self.held.remove(key)
            self._update_command_label()
            event.accept()
            return True
        return False

    @staticmethod
    def _control_key(key):
        key_map = {
            Qt.Key.Key_Left: b"L",
            Qt.Key.Key_Right: b"R",
            Qt.Key.Key_Up: b"U",
            Qt.Key.Key_Down: b"D",
            Qt.Key.Key_W: b"w",
            Qt.Key.Key_S: b"s",
            Qt.Key.Key_D: b"d",
            Qt.Key.Key_A: b"a",
        }
        return key_map.get(key)

    def focusOutEvent(self, event):
        self._clear_held()
        super().focusOutEvent(event)

    def closeEvent(self, event):
        QApplication.instance().removeEventFilter(self)
        self.stop_event.set()
        self.update_timer.stop()
        self.control_timer.stop()
        self.camera_stream.stop()
        self.receiver_socket.close()
        self.control_socket.close()
        self.receiver.join()
        if pygame is not None:
            if self.controller_joystick is not None:
                try:
                    self.controller_joystick.quit()
                except pygame.error:
                    pass
            pygame.quit()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="UWB / IMU / lidar dashboard.")
    parser.add_argument(
        "--pi-ip",
        default=DEFAULT_PI_IP,
        help=f"Raspberry Pi IP address (default: {DEFAULT_PI_IP})",
    )
    args = parser.parse_args()
    app = QApplication([sys.argv[0]])
    pg.setConfigOption("background", PANEL)
    pg.setConfigOption("foreground", TEXT)
    app.setStyleSheet(
        f"""
        QWidget {{
            background: {WINDOW};
            color: {TEXT};
            font-size: 13px;
        }}
        QFrame#panel {{
            background: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 8px;
        }}
        QLabel#panelTitle, QLabel#mutedText {{
            color: {MUTED};
        }}
        QLabel#panelTitle {{
            font-size: 11px;
        }}
        QLabel#controlGroupTitle {{
            color: {MUTED};
            font-size: 10px;
        }}
        QLabel#controlKey {{
            background: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 4px;
            color: {MUTED};
            font-size: 14px;
            font-weight: bold;
        }}
        QLabel#controlKey[keyStyle="arrowKey"][pressed="true"] {{
            background: {BLUE};
            border-color: {BLUE};
            color: {WINDOW};
        }}
        QLabel#controlKey[keyStyle="actionKey"][pressed="true"] {{
            background: {AMBER};
            border-color: {AMBER};
            color: {WINDOW};
        }}
        QPushButton#linkStatusButton {{
            background: transparent;
            border: none;
            padding: 1px;
        }}
        QPushButton#linkStatusButton:hover {{ background: {BORDER}; }}
        QPushButton#armButton {{
            background: #4a4d54;
            border: 1px solid {BORDER};
            border-radius: 6px;
            padding: 7px 14px;
            font-weight: bold;
        }}
        QPushButton#armButton[armed="true"] {{ background: {GREEN}; }}
        QPushButton#armButton:hover {{ border-color: {TEXT}; }}
        QPushButton#estopButton {{
            background: {GREEN};
            border: 1px solid {GREEN};
            border-radius: 6px;
            color: {WINDOW};
            padding: 7px 14px;
            font-weight: bold;
        }}
        QPushButton#estopButton[active="true"] {{
            background: {RED};
            border-color: {RED};
            color: white;
        }}
        QLabel#cameraView {{
            background: #121315;
            border-radius: 6px;
            color: {MUTED};
            font-size: 14px;
            padding: 8px;
        }}
        QStatusBar {{
            background: {PANEL};
            border-top: 1px solid {BORDER};
        }}
        QStatusBar::item {{ border: none; padding: 0 10px; }}
        QLabel#waitingStatus {{ color: {MUTED}; }}
        QLabel#connectionStatus {{ font-weight: bold; }}
        QLabel#connectedStatus {{ color: {GREEN}; }}
        """
    )
    window = MainWindow(args.pi_ip)
    window.show()
    window.setFocus()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
