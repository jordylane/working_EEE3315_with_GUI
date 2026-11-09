import unittest

from pc_launch import LinkMonitor


class LinkMonitorTests(unittest.TestCase):
    def test_missing_optional_fields_are_skipped(self):
        monitor = LinkMonitor()
        monitor.record({}, 10.0)

        snapshot = monitor.snapshot(10.0)

        self.assertEqual(snapshot["state"], "green")
        self.assertIsNone(snapshot["loss"])
        self.assertIsNone(snapshot["rssi_dbm"])
        self.assertIsNone(snapshot["cmd_age_s"])
        self.assertIsNone(snapshot["failsafe"])

    def test_sequence_gaps_handle_wraparound_and_ignore_old_packets(self):
        monitor = LinkMonitor()
        for seq, received_at in (
            (0xFFFFFFFE, 1.0),
            (0xFFFFFFFF, 1.05),
            (1, 1.10),
            (0, 1.15),
        ):
            monitor.record({"seq": seq}, received_at)

        snapshot = monitor.snapshot(1.15)

        self.assertEqual(snapshot["total_sequence_gaps"], 1)
        self.assertAlmostEqual(snapshot["loss"], 0.25)

    def test_red_state_recovers_only_after_hysteresis(self):
        monitor = LinkMonitor()
        monitor.record({"seq": 10}, 0.0)
        self.assertEqual(monitor.snapshot(0.0)["state"], "green")
        self.assertEqual(monitor.snapshot(1.01)["state"], "red")

        for index in range(46):
            received_at = 1.1 + index * 0.05
            monitor.record({"seq": 11 + index}, received_at)
            state = monitor.snapshot(received_at)["state"]
            if 2.05 <= received_at <= 3.0:
                self.assertEqual(state, "red")
        self.assertEqual(state, "green")

    def test_low_rssi_can_warn_but_not_make_state_red(self):
        monitor = LinkMonitor()
        monitor.record({"seq": 1, "rssi_dbm": -85}, 2.0)

        self.assertEqual(monitor.snapshot(2.0)["state"], "amber")


if __name__ == "__main__":
    unittest.main()
