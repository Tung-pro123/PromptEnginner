#!/usr/bin/env python3
"""
Unit and integration tests for V3 Perception & Obstacle Avoidance enhancements:
1. Curvature-adaptive continuity check (sharp curved solid line vs dashed line).
2. Dashed-line confidence calculation.
3. LaserScan angle sign convention and avoidance direction.
4. Flank clearance gating (70°-110° sector check) during obstacle return.
5. Virtual Offset trajectory generation and safety corridor clamping.
"""

import os
import sys
import math
import time
import unittest
import numpy as np

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.speed_track.config import V3Config
from src.speed_track.perception.lane_detector import MultiLaneDetector, LineDetection
from src.speed_track.perception.bev import BEVTransform
from src.speed_track.estimation.lane_state import LaneState, TrackingState
from src.speed_track.control.obstacle_avoidance import (
    ObstacleAvoidance, ObstacleDetector, AvoidState, ObstacleSide
)
from src.speed_track.control.trajectory import TrajectoryGenerator


class MockLaserScan:
    """Mock ROS sensor_msgs/LaserScan for testing."""
    def __init__(self, ranges, angle_min=-math.pi, angle_max=math.pi, angle_increment=math.radians(1.0)):
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.angle_increment = angle_increment
        self.range_min = 0.05
        self.range_max = 5.0
        self.ranges = list(ranges)


def create_curved_synthetic_bev(bev_w=640, bev_h=480, poly_a=0.0008, base_x=150, line_type='solid'):
    """Create synthetic BEV image of a curved line: x(y) = a*(y-480)^2 + base_x."""
    bev = np.zeros((bev_h, bev_w), dtype=np.uint8)
    for y in range(bev_h):
        # Quadratic curve bending right as y goes from 480 to 0
        x = int(base_x + poly_a * ((bev_h - y) ** 2))
        if 0 <= x < bev_w:
            if line_type == 'solid':
                bev[y, max(0, x - 5):min(bev_w, x + 5)] = 255
            elif line_type == 'dashed':
                # Realistic track dashed pattern: 40px dash (4cm), 80px gap (8cm) -> 120px cycle
                if (y % 120) < 40:
                    bev[y, max(0, x - 5):min(bev_w, x + 5)] = 255
    return bev


class TestPerceptionCurvatureAndConfidence(unittest.TestCase):
    def setUp(self):
        self.cfg = V3Config()
        self.detector = MultiLaneDetector(self.cfg)

    def test_curved_solid_line_continuity(self):
        """Test that sharp curved solid line still achieves high continuity ratio (>= 0.70)."""
        # A curved line that drifts by >150px at the top
        bev = create_curved_synthetic_bev(poly_a=0.0008, base_x=150, line_type='solid')
        ratio = self.detector._check_continuity(bev, base_x=150)
        self.assertGreaterEqual(ratio, 0.75, f"Curved solid line should have high continuity ratio, got {ratio}")

    def test_curved_dashed_line_continuity(self):
        """Test that curved dashed line achieves dashed continuity ratio (< 0.68)."""
        bev = create_curved_synthetic_bev(poly_a=0.0008, base_x=320, line_type='dashed')
        ratio = self.detector._check_continuity(bev, base_x=320)
        self.assertLess(ratio, 0.68, f"Curved dashed line should have <0.68 continuity ratio, got {ratio}")
        self.assertGreater(ratio, 0.15, f"Curved dashed line should have >0.15 continuity ratio, got {ratio}")

    def test_dashed_confidence_not_penalized(self):
        """Test that dashed line with 100 inliers gets good confidence when is_dashed=True."""
        conf_solid = self.detector._compute_confidence(n_inliers=100, rmse=2.0, inlier_ratio=0.8, is_dashed=False)
        conf_dashed = self.detector._compute_confidence(n_inliers=100, rmse=2.0, inlier_ratio=0.8, is_dashed=True)
        self.assertGreater(conf_dashed, conf_solid, "Dashed line confidence should be higher with dedicated thresholds")
        self.assertGreater(conf_dashed, self.cfg.min_confidence_gate, "Dashed line must pass min confidence gate")


class TestObstacleAvoidanceLogic(unittest.TestCase):
    def setUp(self):
        self.cfg = V3Config()
        self.cfg.lidar_offset_deg = 0.0  # standard frame: 0 = front, +90 = left, -90 = right
        self.detector = ObstacleDetector(self.cfg)
        self.avoidance = ObstacleAvoidance(self.cfg)

    def _create_scan_with_obstacle(self, obs_angle_deg, obs_dist=0.40, n_points=360):
        """Create a mock 360-degree scan with an obstacle at a specific angle."""
        ranges = [3.0] * n_points
        # Angle min = -180 deg, increment = 1 deg
        idx = int(obs_angle_deg + 180) % n_points
        # Span obstacle across +/- 5 degrees
        for offset in range(-5, 6):
            i = (idx + offset) % n_points
            ranges[i] = obs_dist
        return MockLaserScan(ranges)

    def test_obstacle_on_left_evades_right(self):
        """When obstacle is on the LEFT (+30 deg), lateral force and offset must push RIGHT (>0)."""
        scan = self._create_scan_with_obstacle(obs_angle_deg=30.0, obs_dist=0.45)
        apf_steer, min_front, speed_factor, side = self.detector.detect(scan)

        self.assertEqual(side, ObstacleSide.LEFT, "Obstacle at +30 deg should be detected on LEFT")
        self.assertGreater(apf_steer, 0.0, "Repulsive force for obstacle on LEFT must push RIGHT (>0)")

        res = self.avoidance.process(scan, lane_steer=0.0, lateral_error_m=0.0, dt=0.05)
        self.assertEqual(res.state, AvoidState.EVADING)
        self.assertGreater(res.target_offset_m, 0.0, "Target offset must be positive (RIGHT) when evading left obstacle")

    def test_obstacle_on_right_evades_left(self):
        """When obstacle is on the RIGHT (-30 deg), lateral force and offset must push LEFT (<0)."""
        scan = self._create_scan_with_obstacle(obs_angle_deg=-30.0, obs_dist=0.45)
        apf_steer, min_front, speed_factor, side = self.detector.detect(scan)

        self.assertEqual(side, ObstacleSide.RIGHT, "Obstacle at -30 deg should be detected on RIGHT")
        self.assertLess(apf_steer, 0.0, "Repulsive force for obstacle on RIGHT must push LEFT (<0)")

        res = self.avoidance.process(scan, lane_steer=0.0, lateral_error_m=0.0, dt=0.05)
        self.assertEqual(res.state, AvoidState.EVADING)
        self.assertLess(res.target_offset_m, 0.0, "Target offset must be negative (LEFT) when evading right obstacle")

    def test_flank_clearance_prevents_early_return(self):
        """Test that EVADING does NOT transition to RETURNING if side flank (80 deg) is still occupied."""
        # 1. Start evading left obstacle
        scan_front = self._create_scan_with_obstacle(obs_angle_deg=20.0, obs_dist=0.45)
        self.avoidance.process(scan_front, dt=0.05)
        self.assertEqual(self.avoidance.state, AvoidState.EVADING)

        # 2. Obstacle moves to the side flank (80 deg, 0.25m away - still beside car)
        # Front is clear (min_front = 3.0 > clear_dist)
        scan_side = self._create_scan_with_obstacle(obs_angle_deg=80.0, obs_dist=0.25)
        is_clear = self.detector.is_side_clear(scan_side, ObstacleSide.LEFT)
        self.assertFalse(is_clear, "Flank should NOT be clear when obstacle is at 80 deg within 0.25m")

        res = self.avoidance.process(scan_side, dt=0.05)
        self.assertEqual(res.state, AvoidState.EVADING, "Must stay in EVADING while flank is blocked")

        # 3. Obstacle clears flank (moves behind car or vanishes)
        scan_clear = MockLaserScan([3.0] * 360)
        is_clear = self.detector.is_side_clear(scan_clear, ObstacleSide.LEFT)
        self.assertTrue(is_clear, "Flank should be clear when obstacle is gone")

        res = self.avoidance.process(scan_clear, dt=0.05)
        self.assertEqual(res.state, AvoidState.RETURNING, "Should transition to RETURNING once front and flank are clear")


class TestVirtualOffsetTrajectory(unittest.TestCase):
    def setUp(self):
        self.cfg = V3Config()
        self.bev = BEVTransform(self.cfg)
        self.traj_gen = TrajectoryGenerator(self.cfg, self.bev)

    def test_trajectory_shifted_by_offset(self):
        """Test that trajectory points and lateral error shift accurately by lateral_offset_m."""
        # Centerline straight down the middle (x = 320 px = 0.0 m)
        poly = np.array([0.0, 0.0, 320.0])
        state = LaneState(centerline_poly=poly, tracking_state=TrackingState.TRACKING)

        # Unshifted trajectory
        traj_0 = self.traj_gen.generate(state, current_speed=0.5, lateral_offset_m=0.0)
        self.assertAlmostEqual(traj_0.points[0].x_px, 320.0, delta=1.0)
        self.assertAlmostEqual(traj_0.lateral_error_m, 0.0, delta=0.01)

        # Shifted +0.15m (0.15 * 640 = 96 px right -> 416.0 px)
        traj_shift = self.traj_gen.generate(state, current_speed=0.5, lateral_offset_m=0.15)
        expected_x_px = 320.0 + 0.15 * self.cfg.px_per_meter_x
        # Rollout starts from vehicle base at y=0 (320px) without discontinuous jump
        self.assertAlmostEqual(traj_shift.points[0].x_px, 320.0, delta=2.0)
        # Rollout reaches full target offset at steady state (416px)
        self.assertAlmostEqual(traj_shift.points[-1].x_px, expected_x_px, delta=2.0)
        self.assertAlmostEqual(traj_shift.lateral_error_m, 0.15, delta=0.01)


if __name__ == '__main__':
    unittest.main()
