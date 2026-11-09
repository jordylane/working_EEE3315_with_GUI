import unittest

from pc.controller_input import controller_commands


class ControllerCommandTests(unittest.TestCase):
    def test_right_stick_drives_movement(self):
        self.assertEqual(controller_commands((0.0, 0.0, 0.0, -1.0)), {b"U"})
        self.assertEqual(controller_commands((0.0, 0.0, -1.0, 0.0)), {b"L"})

    def test_left_stick_controls_grabber_and_claw(self):
        self.assertEqual(controller_commands((0.0, -1.0, 0.0, 0.0)), {b"w"})
        self.assertEqual(controller_commands((-1.0, 0.0, 0.0, 0.0)), {b"a"})
        self.assertEqual(controller_commands((1.0, 1.0, 0.0, 0.0)), {b"d", b"s"})

    def test_diagonals_and_deadzone(self):
        self.assertEqual(
            controller_commands((-0.8, -0.8, 0.8, -0.8)),
            {b"R", b"U", b"a", b"w"},
        )
        self.assertEqual(controller_commands((0.2, -0.3, 0.1, 0.0)), set())


if __name__ == "__main__":
    unittest.main()
