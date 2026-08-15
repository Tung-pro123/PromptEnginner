#!/usr/bin/env python3
"""
Integration & End-to-End Tests for Single Source of Truth Steering Pipeline.

Verifies:
1. Direct Steering Command: Planner selection (selected_steer_rad) feeds directly to servo actuator without controller overwriting.
2. Hardware Inversion Isolation: steer_invert=True only flips normalized actuator signal, while physical simulation & last_steer_rad remain true physical radians (+right, -left).
3. Slew Rate Constraint: Servo velocity limits (max_steer_rate_rad_s * dt) are strictly honored on every loop cycle.
4. Kinematic Consistency: Simulated rollout points and actuator commands share identical delta_i sequences.
5. Obstacle Safety Enforcement: Pure Pursuit cannot override a collision-free planner evasion trajectory.
"""

import math
import os
import sys
import time
import unittest
import numpy as np

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.speed_track.config import V3Config
from src.speed_track.perception.bev import BEVTransform
from src.speed_track.estimation.lane_state import LaneState, TrackingState
from src.speed_track.control.corridor_planner import CorridorPlanner
from src.speed_track.control.steering_filter import SteeringFilter
from src.speed_track.control.trajectory import TrajectoryGenerator
from src.speed_track.main_speed_track import SpeedRacingV3


class MockActuator:
    """Mock JetRacer hardware actuator that records received steering and throttle commands."""
    def __init__(self):
        self.last_steer_normalized = 0.0
        self.last_throttle = 0.0
        self.history = []

    def steer(self, angle, speed):
        self.last_steer_normalized = float(angle)
        self.last_throttle = float(speed)
        self.history.append((self.last_steer_normalized, self.last_throttle))

    def stop(self):
        self.last_steer_normalized = 0.0
        self.last_throttle = 0.0
        self.history.append((0.0, 0.0))


class MockLaserScan:
    """Mock ROS LaserScan message."""
    def __init__(self, ranges, stamp=None):
        self.ranges = list(ranges)
        self.angle_min = 0.0
        self.angle_max = 2.0 * math.pi
        self.angle_increment = 2.0 * math.pi / len(ranges)
        self.range_min = 0.05
        self.range_max = 10.0
        self.header = MockHeader(stamp)


class MockHeader:
    def __init__(self, stamp=None):
        self.stamp = MockTime(stamp or time.time())


class MockTime:
    def __init__(self, t):
        self._t = float(t)

    def to_sec(self):
        return self._t


class TestSteeringPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.cfg = V3Config()
        self.cfg.wheelbase_m = 0.14
        self.cfg.max_steer_rad = 0.436
        self.cfg.max_steer_rate_rad_s = 2.5
        self.cfg.steer_lpf_alpha = 1.0  # 1.0 for exact rate-limiting assertions without LPF lag
        self.cfg.lidar_offset_deg = 0.0

    def test_planner_direct_command_to_actuator(self):
        """Test 1: When planner selects delta = +0.2 rad, actuator receives exactly (+0.2 / max_steer_rad)."""
        app = SpeedRacingV3(self.cfg)
        actuator = MockActuator()
        app.racer = actuator

        # Target command from planner
        target_delta_rad = 0.20
        dt = 0.10  # dt * 2.5 = 0.25 >= 0.20 -> reaches target in 1 step

        steer_filtered_rad = app.steering_filter.filter_rad(target_delta_rad, dt=dt)
        steer_hw = app.to_hardware_steering(steer_filtered_rad)
        actuator.steer(steer_hw, 0.25)

        expected_normalized = target_delta_rad / self.cfg.max_steer_rad
        self.assertAlmostEqual(actuator.last_steer_normalized, expected_normalized, places=4,
                               msg="Actuator must receive normalized steering corresponding to planner's selected_steer_rad")
        self.assertGreater(actuator.last_steer_normalized, 0.0, "Positive rad must result in positive (right) hardware steer")

    def test_hardware_inversion_isolation(self):
        """Test 2: steer_invert=True flips only the normalized hardware command, keeping internal physical rad untouched."""
        cfg_normal = V3Config(steer_invert=False)
        cfg_inverted = V3Config(steer_invert=True)

        app_normal = SpeedRacingV3(cfg_normal)
        app_inverted = SpeedRacingV3(cfg_inverted)

        delta_rad = 0.20
        norm_cmd = app_normal.to_hardware_steering(delta_rad)
        inv_cmd = app_inverted.to_hardware_steering(delta_rad)

        self.assertAlmostEqual(norm_cmd, 0.20 / 0.436, places=4)
        self.assertAlmostEqual(inv_cmd, -0.20 / 0.436, places=4)
        self.assertEqual(norm_cmd, -inv_cmd, "Hardware inversion must strictly negate actuator output only")

    def test_slew_rate_limiting_per_loop(self):
        """Test 3: Sudden jump in steering target is rate-limited to max_steer_rate_rad_s * dt."""
        filter_unit = SteeringFilter(self.cfg)
        filter_unit.prev_steer_rad = 0.0

        dt = 0.025  # 25ms loop
        max_allowed_change = self.cfg.max_steer_rate_rad_s * dt  # 2.5 * 0.025 = 0.0625 rad

        # Attempt massive step jump to +0.40 rad
        filtered_1 = filter_unit.filter_rad(0.40, dt=dt)
        self.assertAlmostEqual(filtered_1, max_allowed_change, places=4,
                               msg=f"Step 1 change {filtered_1} must equal max allowed slew rate {max_allowed_change}")

        # Step 2
        filtered_2 = filter_unit.filter_rad(0.40, dt=dt)
        self.assertAlmostEqual(filtered_2, 2.0 * max_allowed_change, places=4,
                               msg="Step 2 must increment by exactly max_steer_rate_rad_s * dt")

    def test_rollout_simulation_matches_actuator_slew_rate(self):
        """Test 4: Bicycle model rollout integration uses identical slew rate limits as steering filter."""
        bev = BEVTransform(self.cfg)
        traj_gen = TrajectoryGenerator(self.cfg, bev)
        planner = CorridorPlanner(self.cfg, bev, traj_gen)

        dt = self.cfg.rollout_dt_s
        points = planner.bicycle_model.simulate_rollout(
            delta_target=self.cfg.max_steer_rad,
            current_steer_rad=-self.cfg.max_steer_rad,
            speed_m_s=0.25,
            horizon_s=self.cfg.rollout_horizon_s,
            dt_s=dt
        )

        filter_unit = SteeringFilter(self.cfg)
        filter_unit.reset(initial_steer_rad=-self.cfg.max_steer_rad)

        for i in range(1, len(points)):
            filtered_sim = filter_unit.filter_rad(self.cfg.max_steer_rad, dt=dt)
            self.assertAlmostEqual(points[i].steer_rad, filtered_sim, places=4,
                                   msg=f"Point {i} steer_rad in rollout must match SteeringFilter simulation exactly")

    def test_pure_pursuit_cannot_override_safety_planner_evasion(self):
        """Test 5: In an obstacle evasion scenario, the planner commands safe avoidance and Pure Pursuit is NOT used to overwrite it."""
        bev = BEVTransform(self.cfg)
        traj_gen = TrajectoryGenerator(self.cfg, bev)
        planner = CorridorPlanner(self.cfg, bev, traj_gen)

        # Straight corridor
        poly = np.array([0.0, 0.0, 320.0])
        left_poly = np.array([0.0, 0.0, 128.0])
        right_poly = np.array([0.0, 0.0, 512.0])

        now = time.time()
        state = LaneState(
            centerline_poly=poly, left_poly=left_poly, right_poly=right_poly,
            left_poly_timestamp=now, right_poly_timestamp=now,
            lane_width_m=0.60, tracking_state=TrackingState.TRACKING
        )

        # Obstacle placed directly in front of the vehicle at (x=0, y=0.45m)
        ranges = [3.0] * 360
        for offset in range(-5, 6):
            ranges[offset % 360] = 0.45

        scan = MockLaserScan(ranges, stamp=now)
        plan_res = planner.plan(state, scan, current_speed=0.25, camera_timestamp=now, current_steer_rad=0.0)

        self.assertTrue(plan_res.safe_to_proceed)
        # Planner MUST choose a non-zero evasion steering angle
        self.assertNotAlmostEqual(plan_res.selected_steer_rad, 0.0, places=2,
                                  msg="Planner must select evasion steering to navigate around frontal obstacle")

        # If a naive Pure Pursuit was applied on the straight centerline target (x=0, y=Ld), steer would be 0.0 (crash)
        # Our architecture directly executes plan_res.selected_steer_rad
        app = SpeedRacingV3(self.cfg)
        steer_cmd_rad = plan_res.selected_steer_rad
        steer_filtered = app.steering_filter.filter_rad(steer_cmd_rad, dt=0.05)
        hw_steer = app.to_hardware_steering(steer_filtered)

        self.assertNotAlmostEqual(hw_steer, 0.0, places=2,
                                  msg="Actuator command must reflect the safe collision-free evasion angle, NOT centerline zero-steer")


if __name__ == '__main__':
    unittest.main()
