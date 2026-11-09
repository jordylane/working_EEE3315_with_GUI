import argparse
import math
import serial
import threading
import time
import struct
from enum import Enum
from collections import deque
import socket
import numpy as np
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (
    CONTROL_PORT,
    CONTROL_TIMEOUT_SECONDS,
    COMMAND_TIMEOUT_S,
    MAX_SPEED,
    PC_IP,
    SEND_RATE_HZ,
    TRACK_WIDTH,
    UDP_PORT,
    anchors,
    encode_packet,
    mix,
    unpack_control_packet,
)
from kalman import (
    HeadingKalmanFilter,
    MAG_EXPECTED_FIELD,
    MAX_PREDICT_DT,
    accelerometer_roll_pitch,
    tilt_compensated_heading,
    wrap_angle,
)

USE_LIVE_DATA = True
PRINT_EVERY = 10
MAX_SIMULATION_DT = 0.1


def read_wifi_rssi_dbm():
    try:
        with open("/proc/net/wireless", "r") as wireless_file:
            for line in wireless_file:
                interface, separator, stats = line.partition(":")
                if separator and interface.strip() == "wlan0":
                    fields = stats.split()
                    return int(round(float(fields[2].rstrip("."))))
    except Exception:
        return None
    return None


class ControlReceiver:
    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.latest_sequence = None
        self.pc_timestamp = None
        self.keys = set()
        self.last_received = None
        self.reported_stick_x = 0.0
        self.reported_stick_y = 0.0
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)

    @staticmethod
    def _is_newer(sequence, previous):
        difference = (sequence - previous) & 0xFFFFFFFF
        return 0 < difference < 0x80000000

    @staticmethod
    def _parse_joystick_packet(packet):
        try:
            fields = packet.decode("ascii").split(",")
            if len(fields) != 3 or fields[0] != "J":
                return None
            x, y = float(fields[1]), float(fields[2])
            if not math.isfinite(x) or not math.isfinite(y):
                return None
        except (UnicodeDecodeError, ValueError):
            return None
        return round(max(-1.0, min(1.0, x)), 2), round(
            max(-1.0, min(1.0, y)), 2
        )

    def _report_stick_change(self, stick_x, stick_y):
        if (
            abs(stick_x - self.reported_stick_x) > 0.05
            or abs(stick_y - self.reported_stick_y) > 0.05
        ):
            print(f"stick: x={stick_x:.2f} y={stick_y:.2f}", flush=True)
            self.reported_stick_x = stick_x
            self.reported_stick_y = stick_y

    def _receive_loop(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as control_socket:
            control_socket.bind(("", CONTROL_PORT))
            control_socket.settimeout(0.05)
            while not self.stop_event.is_set():
                packet = None
                try:
                    packet, _ = control_socket.recvfrom(65535)
                except socket.timeout:
                    pass
                except OSError as exc:
                    if not self.stop_event.is_set():
                        print(f"control UDP receive failed: {exc}", flush=True)
                    return

                if packet is not None:
                    joystick_values = (
                        self._parse_joystick_packet(packet)
                        if packet.startswith(b"J,")
                        else None
                    )
                    is_joystick_packet = packet.startswith(b"J,")
                    valid_command = False
                    previous_failsafe = False

                    if is_joystick_packet:
                        if joystick_values is not None:
                            with self.shared_data.lock:
                                previous_failsafe = self.shared_data.failsafe
                                self.shared_data.last_cmd_time = time.monotonic()
                                self.shared_data.command_received = True
                                self.shared_data.failsafe = False
                                self.shared_data.stick_x = joystick_values[0]
                                self.shared_data.stick_y = joystick_values[1]
                            valid_command = True
                            self._report_stick_change(*joystick_values)
                    else:
                        try:
                            sequence, pc_timestamp, keys = unpack_control_packet(
                                packet
                            )
                        except ValueError:
                            pass
                        else:
                            with self.shared_data.lock:
                                previous_failsafe = self.shared_data.failsafe
                                self.shared_data.last_cmd_time = time.monotonic()
                                self.shared_data.command_received = True
                                self.shared_data.failsafe = False
                            valid_command = True

                            with self.lock:
                                if self.latest_sequence is None or self._is_newer(
                                    sequence, self.latest_sequence
                                ):
                                    self.latest_sequence = sequence
                                    self.pc_timestamp = pc_timestamp
                                    self.keys = keys
                                    self.last_received = time.perf_counter()

                    if valid_command and previous_failsafe:
                        print("command resumed", flush=True)

                now = time.monotonic()
                with self.shared_data.lock:
                    command_time = self.shared_data.last_cmd_time
                    if (
                        command_time is not None
                        and now - command_time > COMMAND_TIMEOUT_S
                        and not self.shared_data.failsafe
                    ):
                        self.shared_data.stick_x = 0.0
                        self.shared_data.stick_y = 0.0
                        self.shared_data.failsafe = True
                        stopped = True
                        stick_x = self.shared_data.stick_x
                        stick_y = self.shared_data.stick_y
                    else:
                        stopped = False
                if stopped:
                    print("failsafe: stopped", flush=True)
                    self._report_stick_change(stick_x, stick_y)

    def current(self, now):
        with self.lock:
            if (
                self.last_received is None
                or now - self.last_received > CONTROL_TIMEOUT_SECONDS
            ):
                return set(), None, None
            return self.keys.copy(), self.latest_sequence, self.pc_timestamp

    def close(self):
        self.stop_event.set()
        self.thread.join()


class VehicleSimulation:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.last_update = None

    def update(self, left_track, right_track, now):
        dt = 0.0 if self.last_update is None else now - self.last_update
        self.last_update = now
        dt = min(max(dt, 0.0), MAX_SIMULATION_DT)

        left_speed = left_track * MAX_SPEED
        right_speed = right_track * MAX_SPEED
        linear_speed = (left_speed + right_speed) / 2.0
        angular_speed = (right_speed - left_speed) / TRACK_WIDTH
        midpoint_heading = self.heading + angular_speed * dt / 2.0
        self.x += linear_speed * math.cos(midpoint_heading) * dt
        self.y += linear_speed * math.sin(midpoint_heading) * dt
        self.heading = (self.heading + angular_speed * dt + math.pi) % (
            2.0 * math.pi
        ) - math.pi
        return self.x, self.y, self.heading


def apply_vehicle_command(vehicle, left_track, right_track, now):
    """Simulation sink; replace this function's body with motor-driver output."""
    return vehicle.update(left_track, right_track, now)


# --------------------------------------------------
# Data Sources
# --------------------------------------------------

class SerialLidarSource:

    def __init__(self, port, record_filename=None):

        self.source = serial.Serial(
            port,
            230400,
            timeout=0.5
        )

        self.record_file = None

        if record_filename:
            self.record_file = open(record_filename, "wb")

    def read(self, size=1):

        data = self.source.read(size)

        if self.record_file and data:
            self.record_file.write(data)

        return data

    def close(self):

        if self.record_file:
            self.record_file.close()

        self.source.close()


class FileLidarSource:

    def __init__(self, filename, baudrate=230400):
        self.source = open(filename, "rb")
        self.baudrate = baudrate

        # Playback timing
        self.next_read_time = time.perf_counter()

    def read(self, size=1):

        data = self.source.read(size)

        if len(data) < size:
            self.source.seek(0)
            data += self.source.read(size - len(data))

            with shared.lock:
                shared.lidar_file_looped = True

        # Serial transfer time for this block
        transfer_time = len(data) * 12.925 / self.baudrate

        # Schedule next transfer completion time
        self.next_read_time += transfer_time

        # Wait until that time arrives
        while True:

            remaining = self.next_read_time - time.perf_counter()

            if remaining <= 0:
                break

            # Sleep when far away from target
            if remaining > 0.001:
                time.sleep(remaining - 0.0005)

        return data

    def close(self):
        self.source.close()


class SerialUWBSource:

    def __init__(self, port, baudrate=115200,
                 record_filename=None):

        self.source = serial.Serial(
            port,
            baudrate,
            timeout=1
        )

        self.record_file = None

        if record_filename:
            self.record_file = open(
                record_filename,
                "w"
            )

    def readline(self):

        data = self.source.readline()
            
        if self.record_file and data:

            line = data.decode("utf-8")

            self.record_file.write(line)
            self.record_file.flush()

        return data

    def close(self):

        if self.record_file:
            self.record_file.close()

        self.source.close()
        

class FileUWBSource:

    def __init__(self,
                 filename,
                 baudrate=115200):

        self.filename = filename
        self.source = open(filename, "r")
        self.baudrate = baudrate

        # Playback timing
        self.next_read_time = time.perf_counter()

    def readline(self):

        line = self.source.readline()

        if line == "":
            self.source.seek(0)
            line = self.source.readline()
            with shared.lock:
                shared.uwb_file_looped = True

        # Serial transfer time for this block
        transfer_time = len(line) * 10.0 / self.baudrate

        # Schedule next transfer completion time
        self.next_read_time += transfer_time

        # Wait until that time arrives
        while True:

            remaining = self.next_read_time - time.perf_counter()

            if remaining <= 0:
                break

            # Sleep when far away from target
            if remaining > 0.001:
                time.sleep(remaining - 0.0005)

        return line.encode("utf-8")

    def close(self):

        self.source.close()


class LiveAccSource:

    def __init__(self, record_filename=None):

        from sense_hat import SenseHat

        self.sense = SenseHat()
        self.record_file = None

        if record_filename:
            self.record_file = open(record_filename, "w")

    def get_data(self, elapsed = 0.0):

        accel = self.sense.get_accelerometer_raw()
        
        accel_x = accel['x']
        accel_y = accel['y']
        accel_z = accel['z']
        
        if self.record_file:
            self.record_file.write(
                f"{accel_x:.4f},{accel_y:.4f},{accel_z:.4f},{elapsed}\n"
            )
            self.record_file.flush()         
    
        return accel_x, accel_y, accel_z, elapsed
    
    def close(self):

        if self.record_file:
            self.record_file.close()
            
class FileAccSource:

    def __init__(self,
                 filename,
                 sample_period=0.01):

        self.file = open(filename, "r")
        self.sample_period = sample_period

    def get_data(self, elapsed = 0.0):

        line = self.file.readline()

        if line == "":
            self.file.seek(0)
            line = self.file.readline()
            with shared.lock:
                shared.acc_file_looped = True

        accel_x, accel_y, accel_z, elapsed = map(
            float,
            line.strip().split(",")
        )

        time.sleep(self.sample_period)

        return accel_x, accel_y, accel_z, elapsed

    def close(self):

        self.file.close()


class LiveAttSource:

    def __init__(self, record_filename=None):

        from sense_hat import SenseHat

        self.sense = SenseHat()
        self.start_time = None

        self.record_file = None

        if record_filename:
            self.record_file = open(record_filename, "w")
            self.record_file.write(
                "raw_mag_heading_deg,roll_rad,pitch_rad,yaw_rad,elapsed_s,"
                "mag_x_uT,mag_y_uT,mag_z_uT,accel_x_g,accel_y_g,accel_z_g,"
                "gyro_z_rad_s,gyro_heading_deg,kalman_heading_deg,gyro_bias_deg_s,"
                "gyro_x_rad_s,gyro_y_rad_s,orientation_roll_rad,"
                "orientation_pitch_rad\n"
            )

    def get_data(self):
        sample_timestamp = time.perf_counter()
        if self.start_time is None:
            self.start_time = sample_timestamp
        accel = self.sense.get_accelerometer_raw()
        gyro = self.sense.get_gyroscope_raw()
        mag = self.sense.get_compass_raw()
        attitude = self.sense.get_orientation_radians()
        gyro_xyz = (gyro["x"], gyro["y"], gyro["z"])
        orientation_rpy = (
            attitude["roll"],
            attitude["pitch"],
            attitude["yaw"],
        )

        return {
            "mag_xyz": (mag["x"], mag["y"], mag["z"]),
            "accel_xyz": (accel["x"], accel["y"], accel["z"]),
            "gyro_xyz": gyro_xyz,
            "gyro_z": gyro["z"],
            "orientation_rpy": orientation_rpy,
            "yaw": attitude["yaw"],
            "elapsed": sample_timestamp - self.start_time,
        }

    def record(self, sample, roll, pitch, raw_heading, gyro_heading, kalman_heading, bias):
        if self.record_file:
            mx, my, mz = sample["mag_xyz"]
            ax, ay, az = sample["accel_xyz"]
            roll_sense, pitch_sense = sample["orientation_rpy"][:2]
            gx, gy = sample["gyro_xyz"][:2]
            self.record_file.write(
                f"{math.degrees(raw_heading):.4f},{roll:.6f},{pitch:.6f},"
                f"{sample['yaw']:.6f},{sample['elapsed']:.6f},"
                f"{mx:.5f},{my:.5f},{mz:.5f},{ax:.5f},{ay:.5f},{az:.5f},"
                f"{sample['gyro_z']:.6f},{math.degrees(gyro_heading):.4f},"
                f"{math.degrees(kalman_heading):.4f},"
                f"{math.degrees(bias):.4f},{gx:.6f},{gy:.6f},"
                f"{roll_sense:.6f},{pitch_sense:.6f}\n"
            )
            self.record_file.flush()
    

    def close(self):

        if self.record_file:
            self.record_file.close()
            
            
class FileAttSource:

    def __init__(self,
                 filename,
                 sample_period=0.05):

        self.file = open(filename, "r")
        self.sample_period = sample_period

    def get_data(self, elapsed = 0.0):

        line = self.file.readline()

        if line == "":
            self.file.seek(0)
            line = self.file.readline()
            with shared.lock:
                shared.att_file_looped = True

        if line.startswith("raw_mag_heading_deg,"):
            line = self.file.readline()
        if line == "":
            raise ValueError(f"No attitude samples in {self.file.name}")

        values = list(map(float, line.strip().split(",")))
        if len(values) >= 15:
            sample = {
                "mag_xyz": tuple(values[5:8]),
                "accel_xyz": tuple(values[8:11]),
                "gyro_z": values[11],
                "gyro_xyz": (
                    (values[15], values[16], values[11])
                    if len(values) >= 17
                    else (0.0, 0.0, values[11])
                ),
                "orientation_rpy": (
                    (values[17], values[18], values[3])
                    if len(values) >= 19
                    else (values[1], values[2], values[3])
                ),
                "yaw": values[3],
                "elapsed": values[4],
            }
        elif len(values) == 5:
            heading, roll, pitch, yaw, elapsed = values
            sample = {
                "mag_xyz": None,
                "accel_xyz": None,
                "gyro_z": None,
                "gyro_xyz": None,
                "orientation_rpy": None,
                "yaw": yaw,
                "elapsed": elapsed,
                "legacy_heading": math.radians(heading),
                "legacy_roll": roll,
                "legacy_pitch": pitch,
            }
        else:
            raise ValueError(
                f"Unrecognized attitude row in {self.file.name}: {line.strip()}"
            )

        time.sleep(self.sample_period)

        return sample

    def close(self):

        self.file.close()
            
# --------------------------------------------------
# Shared data
# --------------------------------------------------

class SharedData:
    def __init__(self):
        self.lock = threading.Lock()
        self.stick_x = 0.0
        self.stick_y = 0.0
        self.last_cmd_time = None
        self.command_received = False
        self.failsafe = False
        
        self.ax_buffer = deque(maxlen=100)
        self.ay_buffer = deque(maxlen=100)
        self.accel_xyz = (0.0, 0.0, 1.0)
        self.imu_accel_xyz = (0.0, 0.0, 1.0)
        self.imu_gyro_xyz = (0.0, 0.0, 0.0)
        self.imu_mag_xyz = (0.0, 0.0, 0.0)
        self.imu_orientation_rpy = (0.0, 0.0, 0.0)

        self.uwb_x = 0.0
        self.uwb_y = 0.0
        
        self.x = np.array([])
        self.y = np.array([])
        self.c = np.array([])
        self.distances = [0.0, 0.0, 0.0, 0.0]
        
        self.heading = 0.0
        self.raw_heading = 0.0
        self.gyro_heading = 0.0
        self.kalman_heading = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
        self.uwb_elapsed_time = 0.0
        self.lidar_elapsed_time = 0.0
        self.acc_elapsed_time = 0.0
        self.att_elapsed_time = 0.0
        
        self.uwb_file_looped = False
        self.acc_file_looped = False
        self.att_file_looped = False
        self.lidar_file_looped = False

        
def trilaterate_3d(anchors, distances):
    """
    3D trilateration using four anchors.

    Parameters
    ----------
    anchors : ndarray (4,3)
        Anchor coordinates [[x1,y1,z1], ... [x4,y4,z4]]

    distances : ndarray (4,)
        Measured distances [d1,d2,d3,d4]

    Returns
    -------
    ndarray (3,)
        Estimated position [x,y,z]
    """

    anchors = np.asarray(anchors, dtype=float)
    distances = np.asarray(distances, dtype=float)

    x1, y1, z1 = anchors[0]
    d1 = distances[0]

    A = []
    b = []

    for i in range(1, 4):
        xi, yi, zi = anchors[i]
        di = distances[i]

        A.append([
            2 * (xi - x1),
            2 * (yi - y1),
            2 * (zi - z1)
        ])

        b.append(
            d1**2 - di**2
            + xi**2 - x1**2
            + yi**2 - y1**2
            + zi**2 - z1**2
        )

    A = np.array(A)
    b = np.array(b)

    position = np.linalg.solve(A, b)

    return position


# ----------------------------------------------------------------------
# Lidar System Constants
# ----------------------------------------------------------------------
# Serial port of the LIDAR unit
SERIAL_PORT = "/dev/serial0"

# At the default rotation speed (~3600 deg/s) the system outputs about 
# 480 measurements in a full rotation. We want to plot at least this
# many in order to get a 360 degree plot
MEASUREMENTS_PER_PLOT = 480

# System will plot between +/-PLOT_MAX_RANGE on both X and Y axis
PLOT_MAX_RANGE = 4.0 # in meters

# Set the plot area to autoscale to the max distance in X or Y
PLOT_AUTO_RANGE = False

# Show the confidence as a colour gradiant on the plot
PLOT_CONFIDENCE = True

# Set the colour gradiant to use. See this URL for options:
# https://matplotlib.org/stable/gallery/color/colormap_reference.html
PLOT_CONFIDENCE_COLOUR_MAP = "bwr_r"

#Enable debug messages
PRINT_DEBUG = False

# ----------------------------------------------------------------------
# Main Packet Format
# ----------------------------------------------------------------------

# All fields are little endian
# Header (1 byte) = 0x54
# Length (1 byte) = 0x2C (assumed to be constant)
# Speed (2 bytes) Rotation speed in degrees per second
# Start angle (2 bytes) divide by 100.0 to get angle in degrees
# Data Measurements (MEASUREMENT_LENGTH * 3 bytes)
#                   See "Format of each data measurement" below
# Stop angle (2 bytes) divide by 100.0 to get angle in degrees
# Timestamp (2 bytes) In milliseconds
# CRC (1 bytes) Poly: 0x4D, Initial Value: 0x00, Final Xor Value: 0x00
#               Input reflected: False, Result Reflected: False
#               http://www.sunshine2k.de/coding/javascript/crc/crc_js.html
# Format of each data measurement
# Distance (2 bytes) # In millimeters
# Confidence (1 byte)

# ----------------------------------------------------------------------
# Packet format constants
# ----------------------------------------------------------------------

PACKET_LENGTH = 47
MEASUREMENT_LENGTH = 12 
MESSAGE_FORMAT = "<xBHH" + "HB" * MEASUREMENT_LENGTH + "HHB"

State = Enum("State", ["SYNC0", "SYNC1", "SYNC2", "LOCKED", "UPDATE_PLOT"])


def parse_lidar_data(data):
    
    # Extract data
    length, speed, start_angle, *pos_data, stop_angle, timestamp, crc = \
        struct.unpack(MESSAGE_FORMAT, data)
    
    # Scale values
    start_angle = float(start_angle) / 100.0
    stop_angle = float(stop_angle) / 100.0
    
    # Unwrap angle if needed and calculate angle step size
    if stop_angle < start_angle:
        stop_angle += 360.0
    step_size = (stop_angle - start_angle) / (MEASUREMENT_LENGTH - 1)
    
    # Get the angle for each measurement in packet
    angle = [start_angle + step_size * i for i in range(0,MEASUREMENT_LENGTH)]
    distance = pos_data[0::2] # in millimeters
    confidence = pos_data[1::2]
    N = len(angle)
    time_arr = np.full(12, timestamp)

    if PRINT_DEBUG:
        print(length, speed, start_angle, *pos_data, stop_angle, timestamp, crc)
        
    return list(zip(angle, distance, confidence, time_arr))

def get_xyc_data(measurements):
    # Unpack the tuples
    angle = np.array([measurement[0] for measurement in measurements])
    distance = np.array([measurement[1] for measurement in measurements])
    confidence = np.array([measurement[2] for measurement in measurements])
    time = np.array([measurement[3] for measurement in measurements])
    # Convert to cartesian coordinates in meters
    x = np.sin(np.radians(angle)) * (distance / 1000.0)
    y = np.cos(np.radians(angle)) * (distance / 1000.0)
    return x, y, confidence, time[0]


shared = SharedData()
running = True

# --------------------------------------------------
# Lidar Thread
# --------------------------------------------------

def lidar_loop():
    
# --------------------------------------------------
# Choose source
# --------------------------------------------------

    if USE_LIVE_DATA:
        source = SerialLidarSource(
            SERIAL_PORT, record_filename=DATA_DIR / "lidar_capture.bin"
        )
    else:
        source = FileLidarSource(DATA_DIR / "lidar_capture.bin")
    

    new_sample = False
    sample_time = 0.0
    prev_sample_time = 0.0
    num_samples = 0
    elapsed_sample_time = 0
    elapsed = 0
    start_time = time.perf_counter()
    
    # Set up initial state
    measurements = []
    data = b''
    state = State.SYNC0

    # Main state machine
    while running:
        # Find 1st header byte
        if state == State.SYNC0:
            
            data = b''
            measurements = []
            
            if source.read() == b'\x54':
                data = b'\x54'
                state = State.SYNC1
                
        # Find 2nd header byte
        # Technically this is the length field but the packet length
        # is fixed, so it can be treated as a constant.
        elif state == State.SYNC1:
            
            if source.read() == b'\x2C':
                state = State.SYNC2
                data += b'\x2C'
            else:
                state = State.SYNC0
                
        # Read remainder of the packet (PACKET_LENGTH minus the 2 header
        # bytes we have already read).
        elif state == State.SYNC2:
            
            data += source.read(size=PACKET_LENGTH - 2)
            
            if len(data) != PACKET_LENGTH:
                state = State.SYNC0
                continue
            
            measurements += parse_lidar_data(data)
            
            state = State.LOCKED
            
        elif state == State.LOCKED:
            
            data = source.read(size=PACKET_LENGTH)
            
            if data[0] != 0x54 or len(data) != PACKET_LENGTH:
                print("WARNING: Serial sync lost")
                state = State.SYNC0
                continue
            
            measurements += parse_lidar_data(data)
            
            if len(measurements) > MEASUREMENTS_PER_PLOT:
                state = State.UPDATE_PLOT
                
        elif state == State.UPDATE_PLOT:
            
            x, y, c, sample_time = get_xyc_data(measurements)
            
            with shared.lock:
                shared.x = x
                shared.y = y
                shared.c = c
                
            # Get the next packet
            state = State.LOCKED
            measurements = []
            
            
        if sample_time != prev_sample_time:
            new_sample = True
            num_samples += 1
            prev_sample_time = sample_time
        else:
            new_sample = False
            
        if num_samples == 1:
            sample_start_time = sample_time
            
        elapsed += time.perf_counter() - start_time
        start_time = time.perf_counter()
            
        if new_sample:
            
            if shared.lidar_file_looped:
                shared.lidar_file_looped = False
                sample_start_time = sample_time - sample_duration
            
            sample_duration = sample_time - sample_start_time 
                
            if sample_duration < 0:
                elapsed_sample_time += sample_duration + 30000
            else:
                elapsed_sample_time += sample_duration
                
            sample_start_time = sample_time
                
        with shared.lock:
            shared.lidar_elapsed_time = elapsed_sample_time/1000
            
#         if num_samples % 10 == 0.0 and new_sample:
#             print(f"lidar elapsed = {elapsed_sample_time/1000}")
            
            
# --------------------------------------------------
# UWB Thread
# --------------------------------------------------

def uwb_loop():
        
    if USE_LIVE_DATA:
        source = SerialUWBSource(
            "/dev/ttyUSB0", record_filename=DATA_DIR / "uwb_capture.txt"
        )
    else:
        source = FileUWBSource(DATA_DIR / "uwb_capture.txt", baudrate=115200)
    
    
    new_sample = False
    sample_time = 0.0
    prev_sample_time = 0.0
    num_samples = 0
    uwb_elapsed_millis = 0
    uwb_elapsed = 0.0
    elapsed = 0.0
    start_time = time.perf_counter()
    time_delay = 0.0
    
    time_step = 1.0
    
    Kp = 0.1
    Ki = 0.01
    integral = 0.0
    
    while running:
    
        # Read UWB position
        line = source.readline().decode().strip()
        
        if line.startswith("time: "):
            try:
                sample_time = float(line.split(":")[1].strip())
            except ValueError:
                pass
        
        with shared.lock:
            if line.startswith("1> "):
                try:
                    shared.distances[0], _ = map(float, line.replace("1> ", "").split("\t"))
                except ValueError:
                    pass
            if line.startswith("2> "):
                try:
                    shared.distances[1], _ = map(float, line.replace("2> ", "").split("\t"))
                except ValueError:
                    pass
            if line.startswith("3> "):
                try:
                    shared.distances[2], _ = map(float, line.replace("3> ", "").split("\t"))
                except ValueError:
                    pass
            if line.startswith("4> "):
                try:
                    shared.distances[3], _ = map(float, line.replace("4> ", "").split("\t"))
                except ValueError:
                    pass
            current_distances = shared.distances.copy()

        estimated_pos = trilaterate_3d(anchors, current_distances)
        
        with shared.lock:
            shared.uwb_x = estimated_pos[0]
            shared.uwb_y = estimated_pos[1]
        
        if sample_time != prev_sample_time:
            new_sample = True
            num_samples += 1
            prev_sample_time = sample_time
        else:
            new_sample = False
            
        if num_samples == 1:
            sample_start_time = sample_time
            
            
        if new_sample:
            
            if shared.uwb_file_looped:
                shared.uwb_file_looped = False
                sample_start_time = sample_time - sample_duration
            
            sample_duration = sample_time - sample_start_time
                
            uwb_elapsed_millis += sample_duration
            uwb_elapsed = uwb_elapsed_millis/1000
            sample_start_time = sample_time
                
        with shared.lock:
            lidar_elapsed = shared.lidar_elapsed_time
            shared.uwb_elapsed_time = uwb_elapsed
        
        now = time.perf_counter()
        dt = now - start_time
        elapsed += dt
        start_time = now
        
        error = lidar_elapsed - uwb_elapsed
        
        integral += error * dt
        correction = Kp*error + Ki*integral
        time_delay -= correction
        
        if time_delay < 0.0:
            time_delay = 0.0
            
        # if uwb_elapsed > time_step:
        #     time_step += 1.0
        #     print(f"elapsed = {elapsed}")
        #     print(f"lidar elapsed = {lidar_elapsed}")
        #     print(f"uwb elapsed = {uwb_elapsed}")
        #     print(f"uwb time delay = {time_delay}\n")
            
        time.sleep(time_delay)


# --------------------------------------------------
# Accelerometer Thread
# --------------------------------------------------

def accelerometer_loop():
    
# --------------------------------------------------
# Choose source
# --------------------------------------------------

    if USE_LIVE_DATA:
        acc_source = LiveAccSource(
            record_filename=DATA_DIR / "accelerometer_capture.csv"
        )
    else:
        acc_source = FileAccSource(
            DATA_DIR / "accelerometer_capture.csv", sample_period=0.001
        )


    elapsed = 0.0
    prev_acc_elapsed = 0.0
    acc_elapsed_loop_offset = 0.0
    start_time = time.perf_counter()
    time_delay = 0.0
    
    time_step = 1.0
    
    Kp = 0.1
    Ki = 0.01
    integral = 0.0

    while running:

        accel_x, accel_y, accel_z, acc_elapsed = acc_source.get_data(elapsed)

        with shared.lock:
            shared.ax_buffer.append(accel_x)
            shared.ay_buffer.append(accel_y)
            shared.accel_xyz = (accel_x, accel_y, accel_z)
            lidar_elapsed = shared.lidar_elapsed_time
            shared.acc_elapsed_time = acc_elapsed
            
            if shared.acc_file_looped:
                acc_elapsed_loop_offset = prev_acc_elapsed
                shared.acc_file_looped = False

        acc_elapsed += acc_elapsed_loop_offset
        prev_acc_elapsed = acc_elapsed
        
        now = time.perf_counter()
        dt = now - start_time
        elapsed += dt
        start_time = now
        
        error = lidar_elapsed - acc_elapsed
        
        integral += error * dt
        correction = Kp*error + Ki*integral
        time_delay -= correction
        
        if time_delay < 0:
            time_delay = 0
        
#         if acc_elapsed > time_step:
#             time_step += 1.0
#             print(f"lidar elapsed = {lidar_elapsed}")
#             print(f"acc elapsed = {acc_elapsed}")
#             print(f"acc time delay = {time_delay}\n")
        
        if USE_LIVE_DATA:
            time.sleep(0.1)
        else:
            time.sleep(time_delay)
            
        


# --------------------------------------------------
# Attitude Thread
# --------------------------------------------------

def attitude_loop():
    
    if USE_LIVE_DATA:
        att_source = LiveAttSource(
            record_filename=DATA_DIR / "attitude_capture.csv"
        )
    else:
        att_source = FileAttSource(
            DATA_DIR / "attitude_capture.csv", sample_period=0.0365
        )

    elapsed = 0.0
    prev_att_elapsed = 0.0
    att_elapsed_loop_offset = 0.0
    previous_filter_timestamp = None
    previous_yaw = None
    gyro_heading = None
    heading_filter = HeadingKalmanFilter()
    start_time = time.perf_counter()
    time_delay = 0.2
    
    time_step = 1.0
    
    Kp = 0.1
    Ki = 0.01
    integral = 0.0

    while running:

        sample = att_source.get_data()
        yaw = sample["yaw"]
        att_elapsed = sample["elapsed"]

        with shared.lock:
            lidar_elapsed = shared.lidar_elapsed_time
            replay_accel_xyz = shared.accel_xyz
            shared.att_elapsed_time = att_elapsed

            if shared.att_file_looped:
                att_elapsed_loop_offset = prev_att_elapsed
                shared.att_file_looped = False
                
        att_elapsed += att_elapsed_loop_offset
        prev_att_elapsed = att_elapsed

        dt = (
            att_elapsed - previous_filter_timestamp
            if previous_filter_timestamp is not None
            else 0.0
        )

        if sample["accel_xyz"] is not None:
            roll, pitch = accelerometer_roll_pitch(sample["accel_xyz"])
        else:
            roll, pitch = accelerometer_roll_pitch(replay_accel_xyz)

        if sample["mag_xyz"] is not None:
            raw_heading, field_magnitude = tilt_compensated_heading(
                sample["mag_xyz"], roll, pitch
            )
        else:
            raw_heading = sample["legacy_heading"]
            field_magnitude = MAG_EXPECTED_FIELD

        gyro_z = sample["gyro_z"]
        if gyro_z is None:
            gyro_z = (
                wrap_angle(yaw - previous_yaw) / dt
                if previous_yaw is not None and 0.0 < dt <= MAX_PREDICT_DT
                else 0.0
            )

        previous_filter_timestamp = att_elapsed
        previous_yaw = yaw

        if gyro_heading is None:
            gyro_heading = raw_heading
        elif 0.0 < dt <= MAX_PREDICT_DT:
            gyro_heading = wrap_angle(gyro_heading + gyro_z * dt)

        heading_filter.predict(gyro_z, dt)
        heading_filter.update(raw_heading, field_magnitude)
        kalman_heading = heading_filter.heading

        if USE_LIVE_DATA:
            att_source.record(
                sample,
                roll,
                pitch,
                raw_heading,
                gyro_heading,
                kalman_heading,
                heading_filter.gyro_bias,
            )

        with shared.lock:
            shared.raw_heading = raw_heading
            shared.gyro_heading = gyro_heading
            shared.kalman_heading = kalman_heading
            shared.heading = math.degrees(kalman_heading)
            shared.roll = roll
            shared.pitch = pitch
            shared.yaw = yaw
            shared.imu_accel_xyz = sample["accel_xyz"] or replay_accel_xyz
            if sample["gyro_xyz"] is not None:
                shared.imu_gyro_xyz = sample["gyro_xyz"]
            if sample["mag_xyz"] is not None:
                shared.imu_mag_xyz = sample["mag_xyz"]
            if sample["orientation_rpy"] is not None:
                shared.imu_orientation_rpy = sample["orientation_rpy"]

        now = time.perf_counter()
        dt = now - start_time
        elapsed += dt
        start_time = now
        
        error = lidar_elapsed - att_elapsed
        
        integral += error * dt
        correction = Kp*error + Ki*integral
        time_delay -= correction
        
        if time_delay < 0.05:
            time_delay = 0.05
            
#         if att_elapsed > time_step:
#             time_step += 1.0
#             print(f"lidar elapsed = {lidar_elapsed}")
#             print(f"att elapsed = {att_elapsed}")
#             print(f"att time delay = {time_delay}\n")
        
        
        if USE_LIVE_DATA:
            time.sleep(0.1)
        else:
            time.sleep(time_delay)

    att_source.close()
# --------------------------------------------------
# UDP Sender Thread
# --------------------------------------------------

def sender_loop(pc_ip, control_receiver=None, vehicle=None):
    interval = 1.0 / SEND_RATE_HZ
    packet_count = 0
    sequence = 0
    last_rssi_read = None
    rssi_dbm = None

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
        while running:
            send_start = time.perf_counter()
            now = time.monotonic()
            if last_rssi_read is None or now - last_rssi_read >= 1.0:
                rssi_dbm = read_wifi_rssi_dbm()
                last_rssi_read = now
            vehicle_x, vehicle_y, vehicle_heading = (0.0, 0.0, 0.0)
            control_sequence = None
            control_timestamp = None
            if USE_LIVE_DATA and control_receiver is not None and vehicle is not None:
                keys, control_sequence, control_timestamp = control_receiver.current(
                    send_start
                )
                left_track, right_track = mix(keys)
                vehicle_x, vehicle_y, vehicle_heading = apply_vehicle_command(
                    vehicle, left_track, right_track, send_start
                )

            with shared.lock:
                cmd_age_s = (
                    time.monotonic() - shared.last_cmd_time
                    if control_receiver is not None and shared.command_received
                    else None
                )
                packet = {
                    "seq": sequence,
                    "rssi_dbm": rssi_dbm,
                    "cmd_age_s": cmd_age_s,
                    "failsafe": (
                        shared.failsafe if control_receiver is not None else False
                    ),
                    "lidar_x": shared.x.copy(),
                    "lidar_y": shared.y.copy(),
                    "lidar_c": shared.c.copy(),
                    "uwb_x": shared.uwb_x,
                    "uwb_y": shared.uwb_y,
                    "distances": shared.distances.copy(),
                    "ax": shared.ax_buffer[-1] if shared.ax_buffer else 0.0,
                    "ay": shared.ay_buffer[-1] if shared.ay_buffer else 0.0,
                    "imu": {
                        "accelerometer_g": list(shared.imu_accel_xyz),
                        "gyroscope_rad_s": list(shared.imu_gyro_xyz),
                        "magnetometer_uT": list(shared.imu_mag_xyz),
                        "orientation_rad": {
                            "roll": shared.imu_orientation_rpy[0],
                            "pitch": shared.imu_orientation_rpy[1],
                            "yaw": shared.imu_orientation_rpy[2],
                        },
                    },
                    "heading": shared.heading,
                    "raw_heading": shared.raw_heading,
                    "gyro_heading": shared.gyro_heading,
                    "kalman_heading": shared.kalman_heading,
                    "yaw": shared.yaw,
                    "vehicle_x": vehicle_x,
                    "vehicle_y": vehicle_y,
                    "vehicle_heading": vehicle_heading,
                    "control_sequence": control_sequence,
                    "control_timestamp": control_timestamp,
                    "timestamp": time.time(),
                }

            payload = encode_packet(packet)
            packet_count += 1
            sequence = (sequence + 1) & 0xFFFFFFFF
            if packet_count % PRINT_EVERY == 0:
                distances = ", ".join(f"{value:.2f}" for value in packet["distances"])
                age = time.time() - packet["timestamp"]
                imu = packet["imu"]
                accel_text = ", ".join(
                    f"{value:+.3f}" for value in imu["accelerometer_g"]
                )
                gyro_text = ", ".join(
                    f"{value:+.3f}" for value in imu["gyroscope_rad_s"]
                )
                mag_text = ", ".join(
                    f"{value:+.1f}" for value in imu["magnetometer_uT"]
                )
                orientation = imu["orientation_rad"]
                print(
                    f"#{packet_count} {len(payload)} B lidar={len(packet['lidar_x'])} "
                    f"uwb=({packet['uwb_x']:.2f}, {packet['uwb_y']:.2f}) "
                    f"distances=[{distances}] "
                    f"imu_accel_g=[{accel_text}] "
                    f"imu_gyro_rad_s=[{gyro_text}] "
                    f"imu_mag_uT=[{mag_text}] "
                    f"imu_rpy_rad=({orientation['roll']:+.3f},"
                    f"{orientation['pitch']:+.3f},"
                    f"{orientation['yaw']:+.3f}) "
                    f"heading={packet['heading']:.1f} "
                    f"yaw={packet['yaw']:.1f} age={age:.3f}s",
                    flush=True,
                )

            try:
                udp_socket.sendto(payload, (pc_ip, UDP_PORT))
            except OSError:
                pass

            time.sleep(max(0.0, interval - (time.perf_counter() - send_start)))


# --------------------------------------------------
# Start Threads
# --------------------------------------------------

def main():
    global running, DATA_DIR, SERIAL_PORT

    parser = argparse.ArgumentParser(description="Send UWB, IMU, and lidar data over UDP")
    parser.add_argument("--pc-ip", default=PC_IP, help="IP address of the PC display")
    parser.add_argument(
        "--lidar-port",
        default=SERIAL_PORT,
        help="serial device used by the live lidar (default: /dev/serial0)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"directory containing sensor data files (default: {DATA_DIR})",
    )
    args = parser.parse_args()
    DATA_DIR = args.data_dir.expanduser().resolve()
    SERIAL_PORT = args.lidar_port
    print(f"Sensor data directory: {DATA_DIR}", flush=True)
    print(f"Reading live lidar from {SERIAL_PORT} at 230400 baud", flush=True)

    control_receiver = ControlReceiver(shared)
    vehicle = VehicleSimulation()
    threads = [
        threading.Thread(target=lidar_loop),
        threading.Thread(target=uwb_loop),
        threading.Thread(target=accelerometer_loop),
        threading.Thread(target=attitude_loop),
        threading.Thread(
            target=sender_loop,
            args=(args.pc_ip, control_receiver, vehicle),
        ),
    ]
    if control_receiver is not None:
        threads.append(control_receiver.thread)

    for thread in threads:
        thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        running = False
        if control_receiver is not None:
            control_receiver.close()
        for thread in threads:
            thread.join()
        print("Stopped")


if __name__ == "__main__":
    main()
