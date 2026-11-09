import argparse
import subprocess
import sys
import time
from pathlib import Path

from common import CONTROL_MONITOR_PORT


ROOT = Path(__file__).resolve().parent
SENSOR_SCRIPT = ROOT / "pi" / "pi_sensors.py"
CONTROL_MONITOR_SCRIPT = ROOT / "pi" / "pi_control_monitor.py"
DEFAULT_DATA_DIR = ROOT / "pi"


def process_commands(pc_ip, lidar_port, data_dir):
    return (
        (
            "Pi Control Monitor",
            [
                sys.executable,
                "-u",
                str(CONTROL_MONITOR_SCRIPT),
                "--host",
                "0.0.0.0",
                "--port",
                str(CONTROL_MONITOR_PORT),
            ],
        ),
        (
            "Pi Sensors",
            [
                sys.executable,
                "-u",
                str(SENSOR_SCRIPT),
                "--pc-ip",
                pc_ip,
                "--lidar-port",
                str(lidar_port),
                "--data-dir",
                str(data_dir),
            ],
        ),
    )


def stop_processes(processes):
    for _, process in processes:
        if process.poll() is None:
            try:
                process.terminate()
            except OSError as error:
                if process.poll() is None:
                    print(
                        f"Could not stop child process {process.pid}: {error}",
                        file=sys.stderr,
                    )

    for _, process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def run_processes(commands):
    processes = []
    try:
        for name, command in commands:
            print(f"Starting {name}", flush=True)
            try:
                process = subprocess.Popen(command, cwd=ROOT)
            except OSError as error:
                print(f"Cannot start {name}: {error}", file=sys.stderr)
                return 1
            processes.append((name, process))

        while True:
            for name, process in processes:
                return_code = process.poll()
                if return_code is not None:
                    print(
                        f"{name} exited with status {return_code}; "
                        "stopping remaining Pi processes.",
                        flush=True,
                    )
                    return return_code
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping Pi sensor and control monitoring.", flush=True)
        return 130
    finally:
        stop_processes(processes)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Raspberry Pi sensor and control monitoring."
    )
    parser.add_argument(
        "--pc-ip",
        required=True,
        help=(
            "PC address reachable from the Pi for sensor UDP packets"
        ),
    )
    parser.add_argument(
        "--lidar-port",
        default="/dev/serial0",
        help="serial device used by the live lidar (default: /dev/serial0)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"directory for sensor recordings (default: {DEFAULT_DATA_DIR})",
    )
    args = parser.parse_args(argv)

    for name, script in (
        ("Pi Control Monitor", CONTROL_MONITOR_SCRIPT),
        ("Pi Sensors", SENSOR_SCRIPT),
    ):
        if not script.is_file():
            print(f"Cannot launch {name}: script not found: {script}",
                  file=sys.stderr)
            return 1

    return run_processes(
        process_commands(
            args.pc_ip,
            args.lidar_port,
            args.data_dir.expanduser().resolve(),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
