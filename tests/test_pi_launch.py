import os
import unittest
from pathlib import Path
from unittest.mock import patch

import pi_launch


class FakeProcess:
    def __init__(self, returncode=None):
        self.returncode = returncode
        self.pid = 1
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


class PiLauncherTests(unittest.TestCase):
    def test_runs_sensor_and_monitor_in_one_session_with_pc_ip(self):
        monitor = FakeProcess(returncode=0)
        sensors = FakeProcess()
        with patch(
            "pi_launch.subprocess.Popen",
            side_effect=(monitor, sensors),
        ) as popen:
            with patch.dict(os.environ, {}, clear=True):
                result = pi_launch.main([
                    "--pc-ip",
                    "192.0.2.14",
                    "--lidar-port",
                    "/dev/ttyUSB1",
                    "--data-dir",
                    "captures",
                ])

        self.assertEqual(result, 0)
        self.assertEqual(popen.call_count, 2)
        self.assertEqual(
            popen.call_args_list[0].args[0],
            [
                pi_launch.sys.executable,
                "-u",
                str(pi_launch.CONTROL_MONITOR_SCRIPT),
                "--host",
                "0.0.0.0",
                "--port",
                "5007",
            ],
        )
        self.assertEqual(
            popen.call_args_list[1].args[0],
            [
                pi_launch.sys.executable,
                "-u",
                str(pi_launch.SENSOR_SCRIPT),
                "--pc-ip",
                "192.0.2.14",
                "--lidar-port",
                "/dev/ttyUSB1",
                "--data-dir",
                str(Path("captures").resolve()),
            ],
        )
        for call in popen.call_args_list:
            self.assertEqual(call.kwargs["cwd"], pi_launch.ROOT)
        self.assertTrue(sensors.terminated)

    def test_stops_started_process_if_another_child_cannot_start(self):
        monitor = FakeProcess()
        with patch(
            "pi_launch.subprocess.Popen",
            side_effect=(monitor, OSError("launch failed")),
        ):
            result = pi_launch.main(["--pc-ip", "192.0.2.14"])

        self.assertEqual(result, 1)
        self.assertTrue(monitor.terminated)

    def test_reports_missing_child_script(self):
        with (
            patch.object(
                pi_launch,
                "CONTROL_MONITOR_SCRIPT",
                Path("monitor.py"),
            ),
            patch.object(
                pi_launch,
                "SENSOR_SCRIPT",
                Path("sensors.py"),
            ),
            patch("pathlib.Path.is_file", side_effect=(True, False)),
            patch("pi_launch.subprocess.Popen") as popen,
        ):
            result = pi_launch.main(["--pc-ip", "192.0.2.14"])

        self.assertEqual(result, 1)
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
