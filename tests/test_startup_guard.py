#!/usr/bin/env python3
"""
Test startup calibration timeout guard in SpeedTrackApp.
"""

import os
import sys
import unittest
import numpy as np

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.speed_track.config import V3Config
from src.speed_track.main_speed_track import SpeedRacingV3


class TestStartupGuardTimeout(unittest.TestCase):
    def test_startup_guard_timeout_after_60_frames(self):
        """Verify that if dual lines are never seen during startup, app aborts after 60 frames."""
        cfg = V3Config()
        cfg.startup_timeout_frames = 60
        app = SpeedRacingV3(cfg)

        # Blank black frame (no lines visible)
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Run for 59 frames -> calibration_checked should still be False, calibration_valid True
        for i in range(59):
            steer, throttle, dash, state = app.process_frame(black_frame)
            self.assertEqual(throttle, 0.0)

        self.assertFalse(app.calibration_checked)

        # Frame 60 -> should trigger timeout fail-safe
        steer, throttle, dash, state = app.process_frame(black_frame)
        self.assertTrue(app.calibration_checked, "Should trigger calibration check timeout")
        self.assertFalse(app.calibration_valid, "Calibration valid should become False on timeout")
        self.assertEqual(throttle, 0.0)


if __name__ == '__main__':
    unittest.main()
