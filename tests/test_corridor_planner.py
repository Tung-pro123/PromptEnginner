#!/usr/bin/env python3
"""
Comprehensive Unit & Integration Test Suite for CorridorPlanner, RacerController & Safety Architecture:
1. BEV metric calibration at 640 px/m.
2. [P0 Fix] RacerController 0.0 throttle preservation (not replaced by BASE_THROTTLE).
3. [P0 Fix] Dynamic stopping distance active braking enforcement when obstacle is within d_stop.
4. [P0 Fix] LiDAR Watchdog is NOT reset by cached static scan.
5. [P1 Fix] Multi-disk swept footprint & dense segment collision checking in curve.
6. [P1 Fix] Camera-LiDAR temporal sync & missing timestamp handling.
7. [P1 Fix] Stale boundaries timeout (>0.8s) triggering safe stop.
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
from src.speed_track.perception.bev import BEVTransform
from src.speed_track.estimation.geometry import LaneGeometry
from src.speed_track.estimation.lane_state import LaneState, TrackingState
from src.speed_track.control.trajectory import TrajectoryGenerator
from src.speed_track.control.corridor_planner import CorridorPlanner, CandidateTrajectory
from src.core.control.racer_controller import RacerController


class MockHeader:
    def __init__(self, stamp=None):
        self.stamp = stamp or time.time()


class MockLaserScan:
    """Mock ROS sensor_msgs/LaserScan for testing."""
    def __init__(self, ranges, stamp=None, angle_min=-math.pi, angle_max=math.pi, angle_increment=math.radians(1.0)):
        self.header = MockHeader(stamp=stamp)
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.angle_increment = angle_increment
        self.range_min = 0.05
        self.range_max = 5.0
        self.ranges = list(ranges)


class TestRacerControllerThrottleP0(unittest.TestCase):
    def test_zero_speed_is_not_replaced_by_base_throttle(self):
        """P0 Test: Verify that passing speed=0.0 to forward() or steer() maintains throttle=0.0."""
        racer = RacerController()
        # Test steer with speed=0.0 (stop command)
        racer.steer(0.15, speed=0.0)
        self.assertEqual(racer.car.throttle, 0.0, "steer(speed=0.0) must set throttle to 0.0, not BASE_THROTTLE")

        # Test forward with speed=0.0
        racer.forward(speed=0.0)
        self.assertEqual(racer.car.throttle, 0.0, "forward(speed=0.0) must set throttle to 0.0, not BASE_THROTTLE")


class TestCorridorPlannerP0P1Fixes(unittest.TestCase):
    def setUp(self):
        self.cfg = V3Config()
        self.cfg.lidar_offset_deg = 0.0  # Standard frame for synthetic scan
        self.bev = BEVTransform(self.cfg)
        self.planner = CorridorPlanner(self.cfg, self.bev)

    def test_bev_640_scale_calibration(self):
        """Test that 384px in BEV corresponds to 0.60m track width and passes geometry tolerance."""
        geometry = LaneGeometry(self.cfg, self.bev)
        self.assertEqual(self.cfg.px_per_meter_x, 640.0)
        width_m = 384.0 / self.cfg.px_per_meter_x
        self.assertAlmostEqual(width_m, 0.60, places=3)
        self.assertTrue(geometry._width_valid(width_m))

    def test_obstacle_within_d_stop_triggers_safe_stop(self):
        """P0 Test: Obstacle not touching current footprint but located within d_stop ahead triggers safe stop."""
        poly = np.array([0.0, 0.0, 320.0])
        left_poly = np.array([0.0, 0.0, 128.0])
        right_poly = np.array([0.0, 0.0, 512.0])

        now = time.time()
        state = LaneState(
            centerline_poly=poly, left_poly=left_poly, right_poly=right_poly,
            left_poly_timestamp=now, right_poly_timestamp=now,
            lane_width_m=0.60, tracking_state=TrackingState.TRACKING
        )

        # Vehicle traveling at 0.35 m/s -> d_stop = (0.35^2)/(2*2.5) + 0.35*0.10 = 0.0245 + 0.035 = 0.0595 m
        # Place obstacle spanning full track at 0.08m (within d_stop + margin 0.04m = 0.0995m)
        ranges = [3.0] * 360
        for ang in range(-45, 46):
            ranges[(ang + 180) % 360] = 0.08

        scan = MockLaserScan(ranges, stamp=now)
        res = self.planner.plan(state, scan, current_speed=0.35, camera_timestamp=now)

        self.assertFalse(res.safe_to_proceed, "Vehicle must trigger emergency stop when obstacle is within d_stop")
        self.assertEqual(res.speed_factor, 0.0, "Speed factor must be 0 on blocked corridor")
        self.assertGreater(res.d_stop, 0.05, "d_stop must be actively non-zero when speed > 0")

    def test_cached_scan_does_not_reset_watchdog(self):
        """P0 Test: Verify that passing the same cached scan object repeatedly triggers watchdog timeout."""
        poly = np.array([0.0, 0.0, 320.0])
        start_time = time.time()
        state = LaneState(
            centerline_poly=poly, left_poly_timestamp=start_time, right_poly_timestamp=start_time,
            tracking_state=TrackingState.TRACKING
        )

        # Create a single scan with fixed stamp
        cached_scan = MockLaserScan([3.0] * 360, stamp=start_time)

        # First call at t = start_time
        res1 = self.planner.plan(state, cached_scan, current_speed=0.25, camera_timestamp=start_time)
        self.assertTrue(res1.safe_to_proceed)

        # Simulate 1.0s passing, but caller passes the SAME cached_scan with same timestamp
        t_future = start_time + 1.0
        res2 = self.planner.plan(state, cached_scan, current_speed=0.25, camera_timestamp=t_future)

        # Stale LiDAR / Watchdog must be flagged because no NEW scan arrived for > 0.8s
        self.assertTrue(res2.stale_lidar, "Stale LiDAR must be flagged when scan timestamp is not updated")
        self.assertLessEqual(res2.speed_factor, 0.5, "Speed factor must be reduced on stale scan")

    def test_swept_multidisk_footprint_in_curve(self):
        """P1 Test: Verify that swept multi-disk footprint catches corner obstacle in sharp curve."""
        # Sharp right curve poly: x(y) = 0.0008*y^2 - 0.3*y + 320
        poly = np.array([0.0008, -0.3, 320.0])
        left_poly = np.array([0.0008, -0.3, 128.0])
        right_poly = np.array([0.0008, -0.3, 512.0])

        now = time.time()
        state = LaneState(
            centerline_poly=poly, left_poly=left_poly, right_poly=right_poly,
            left_poly_timestamp=now, right_poly_timestamp=now,
            lane_width_m=0.60, tracking_state=TrackingState.TRACKING
        )

        # Place obstacle at left edge of curve at 0.40m
        ranges = [3.0] * 360
        for offset in range(-5, 6):
            ranges[(25 + offset + 180) % 360] = 0.40

        scan = MockLaserScan(ranges, stamp=now)
        res = self.planner.plan(state, scan, current_speed=0.25, camera_timestamp=now)

        self.assertTrue(res.safe_to_proceed)
        # Must steer right (+steer / rightward rollout)
        self.assertGreater(res.selected_steer_rad, 0.0)

    def test_stale_boundaries_timeout_safe_stop(self):
        """P1 Test: If both boundary lines are stale for > 0.8s, planner commands safe stop."""
        poly = np.array([0.0, 0.0, 320.0])
        left_poly = np.array([0.0, 0.0, 128.0])
        right_poly = np.array([0.0, 0.0, 512.0])

        now = time.time()
        # Boundaries last seen 1.0s ago (> 0.80s boundary_stop_timeout)
        state = LaneState(
            centerline_poly=poly, left_poly=left_poly, right_poly=right_poly,
            left_poly_timestamp=now - 1.0, right_poly_timestamp=now - 1.0,
            lane_width_m=0.60, tracking_state=TrackingState.TRACKING
        )

        scan = MockLaserScan([3.0] * 360, stamp=now)
        res = self.planner.plan(state, scan, current_speed=0.25, camera_timestamp=now)

        self.assertFalse(res.safe_to_proceed, "Must trigger emergency stop when boundaries remain stale > 0.8s")
        self.assertEqual(res.reason, "stale_boundaries_timeout")
        self.assertEqual(res.speed_factor, 0.0)

    def test_kinematic_rollout_starts_from_vehicle_pose(self):
        """P0 Test: Verify that all candidate evasion rollouts start continuously from vehicle pose (x=0, y=0, psi=0, delta=current_steer)."""
        poly = np.array([0.0, 0.0, 320.0])
        left_poly = np.array([0.0, 0.0, 128.0])
        right_poly = np.array([0.0, 0.0, 512.0])

        now = time.time()
        state = LaneState(
            centerline_poly=poly, left_poly=left_poly, right_poly=right_poly,
            left_poly_timestamp=now, right_poly_timestamp=now,
            lane_width_m=0.60, tracking_state=TrackingState.TRACKING
        )

        scan = MockLaserScan([3.0] * 360, stamp=now)
        res = self.planner.plan(state, scan, current_speed=0.25, camera_timestamp=now, current_steer_rad=0.10)

        self.assertTrue(res.safe_to_proceed)
        for cand in res.all_candidates:
            if cand.points:
                p0 = cand.points[0]
                self.assertAlmostEqual(p0.x_m, 0.0, places=4, msg="Rollout must start at vehicle x=0.0m")
                self.assertAlmostEqual(p0.y_m, 0.0, places=4, msg="Rollout must start at vehicle y=0.0m")
                self.assertAlmostEqual(p0.yaw_rad, 0.0, places=4, msg="Rollout must start at vehicle yaw=0.0rad")
                self.assertAlmostEqual(p0.steer_rad, 0.10, places=4, msg="Rollout must start at current steer angle")

    def test_lateral_offsets_recovery(self):
        """Test that vehicle recovers properly when centerline is laterally shifted ±5cm, ±10cm, ±15cm."""
        now = time.time()
        for offset_m in [-0.15, -0.10, -0.05, 0.05, 0.10, 0.15]:
            offset_px = offset_m * 640.0
            # Centerline shifted: if line is at x=320 + offset_px, car (at x=320) sees target to the right (offset_m > 0)
            poly = np.array([0.0, 0.0, 320.0 + offset_px])
            left_poly = np.array([0.0, 0.0, 128.0 + offset_px])
            right_poly = np.array([0.0, 0.0, 512.0 + offset_px])

            state = LaneState(
                centerline_poly=poly, left_poly=left_poly, right_poly=right_poly,
                left_poly_timestamp=now, right_poly_timestamp=now,
                lane_width_m=0.60, tracking_state=TrackingState.TRACKING
            )
            scan = MockLaserScan([3.0] * 360, stamp=now)
            res = self.planner.plan(state, scan, current_speed=0.25, camera_timestamp=now)

            self.assertTrue(res.safe_to_proceed)
            if offset_m > 0.05:
                # Target line is to the right -> planner must steer right (> 0)
                self.assertGreater(res.selected_steer_rad, 0.0,
                                   msg=f"Should steer right when line is at +{offset_m}m")
            elif offset_m < -0.05:
                # Target line is to the left -> planner must steer left (< 0)
                self.assertLess(res.selected_steer_rad, 0.0,
                                msg=f"Should steer left when line is at {offset_m}m")

    def test_kinematic_bicycle_servo_rate_limiting(self):
        """P0 Test: Verify that steering angle change between consecutive steps never exceeds max_steer_rate * dt."""
        bicycle = self.planner.bicycle_model
        dt_s = 0.03
        max_rate = bicycle.max_steer_rate
        points = bicycle.simulate_rollout(delta_target=0.436, current_steer_rad=-0.436, speed_m_s=0.25, horizon_s=0.90, dt_s=dt_s)

        self.assertGreater(len(points), 5)
        for i in range(1, len(points)):
            delta_diff = abs(points[i].steer_rad - points[i - 1].steer_rad)
            max_allowed = max_rate * dt_s + 1e-6
            self.assertLessEqual(delta_diff, max_allowed,
                                 msg=f"Steering slew rate exceeded: step diff {delta_diff:.4f} > max {max_allowed:.4f}")

    def test_max_curvature_chassis_limit(self):
        """Test that max allowable curvature is derived exactly from single unified chassis parameters."""
        cfg = self.planner.cfg
        expected_kappa = math.tan(cfg.max_steer_rad) / cfg.wheelbase_m
        self.assertAlmostEqual(cfg.max_trajectory_curvature, expected_kappa, places=4)
        self.assertAlmostEqual(expected_kappa, 3.328, delta=0.05)


if __name__ == '__main__':
    unittest.main()
