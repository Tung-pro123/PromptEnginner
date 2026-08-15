#!/usr/bin/env python3
"""
Unit and integration tests for advanced curved dead reckoning, normal-vector offset,
corridor priority & passability checks:
1. Dead Reckoning holds curvature in curves during line dropouts.
2. Re-acquisition gating prevents jumping to wrong line.
3. Normal-vector offset maintains true orthogonal distance in curves.
4. Passability check stops the vehicle if corridor space is insufficient.
5. Speed controller applies crawl speed on single-line reconstruction.
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
from src.speed_track.estimation.geometry import GeometryObservation, LaneGeometry
from src.speed_track.estimation.lane_state import LaneStateEstimator, TrackingState, LaneState
from src.speed_track.control.trajectory import TrajectoryGenerator
from src.speed_track.control.speed_controller import SpeedController
from src.speed_track.control.obstacle_avoidance import ObstacleAvoidance, ObstacleSide


class MockLaserScan:
    """Mock ROS sensor_msgs/LaserScan for testing."""
    def __init__(self, ranges, angle_min=-math.pi, angle_max=math.pi, angle_increment=math.radians(1.0)):
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.angle_increment = angle_increment
        self.range_min = 0.05
        self.range_max = 5.0
        self.ranges = list(ranges)


class TestCurvedDeadReckoning(unittest.TestCase):
    def setUp(self):
        self.cfg = V3Config()
        self.bev = BEVTransform(self.cfg)
        self.estimator = LaneStateEstimator(self.cfg, self.bev)

    def test_dead_reckoning_holds_curvature_in_curve(self):
        """Test that estimator keeps 100% curvature for 0.25s during line dropout in a curve."""
        # Initial confident curve detection (turning right with curvature kappa = 1.2)
        poly = np.array([0.001, 0.2, 320.0])
        obs = GeometryObservation(
            centerline_poly=poly,
            overall_confidence=0.9,
            valid=True,
            lane_width_m=0.60,
            method='L+C+R_fused'
        )
        
        # 1. Update with good measurement
        state = self.estimator.update(obs)
        initial_kappa = state.curvature
        self.assertGreater(abs(initial_kappa), 0.2, "Initial curvature should be significant in curve")

        # 2. Line drops out (e.g. gap between dashed lines in curve for 0.20s)
        now = time.time()
        invalid_obs = GeometryObservation(valid=False)

        # Call update 5 times simulating 0.15s of prediction
        self.estimator._last_confident_time = now  # Anchor time
        for i in range(5):
            state = self.estimator.update(invalid_obs)
            # Curvature must remain exactly equal to initial_kappa (100% hold)
            self.assertAlmostEqual(state.curvature, initial_kappa, places=4,
                                   msg=f"Curvature should be held during hold duration, got {state.curvature}")

    def test_reacquisition_spatial_gate(self):
        """Test that line re-acquisition rejects impossible jumps when recovering."""
        # Setup estimator in PREDICTING state with centerline near x=320 (0m lateral error)
        poly = np.array([0.0, 0.0, 320.0])
        obs = GeometryObservation(centerline_poly=poly, overall_confidence=0.9, valid=True, lane_width_m=0.60)
        self.estimator.update(obs)

        # Force state to PREDICTING
        self.estimator.state.tracking_state = TrackingState.PREDICTING

        # Candidate falsely tries to snap to outer boundary (lateral jump = 0.35m > 0.25m gate)
        jump_poly = np.array([0.0, 0.0, 320.0 + 350.0])  # +350px = +0.35m
        jump_obs = GeometryObservation(centerline_poly=jump_poly, overall_confidence=0.8, valid=True, lane_width_m=0.60)

        state = self.estimator.update(jump_obs)
        # Should reject and remain in prediction
        self.assertEqual(state.reconstruction_method, 'prediction', "Jump observation should be rejected by reacquisition gate")


class TestNormalVectorOffset(unittest.TestCase):
    def setUp(self):
        self.cfg = V3Config()
        self.bev = BEVTransform(self.cfg)
        self.traj_gen = TrajectoryGenerator(self.cfg, self.bev)

    def test_normal_offset_in_sharp_curve(self):
        """Test that offset points are shifted perpendicular to curve tangent."""
        # Curved line x(y) = a*y^2 + b*y + c
        poly = np.array([0.0005, -0.2, 320.0])
        state = LaneState(centerline_poly=poly, tracking_state=TrackingState.TRACKING)

        d_offset = 0.15  # 15cm right offset
        traj = self.traj_gen.generate(state, current_speed=0.3, lateral_offset_m=d_offset)

        # In vehicle position (y_px = 480):
        a, b = poly[0], poly[1]
        dx_dy_px = 2.0 * a * 480.0 + b
        sx = 1.0 / self.cfg.px_per_meter_x
        sy = 1.0 / self.cfg.px_per_meter_y
        dx_dy_m = -dx_dy_px * (sx / sy)
        theta_road = math.atan(dx_dy_m)

        # Check that target points exist and follow smooth kinematic rollout
        self.assertTrue(len(traj.points) > 0)
        p0 = traj.points[0]
        base_xm_0, base_ym_0 = self.bev.px_to_metric(np.polyval(poly, 480.0), 480.0)
        dist_0 = math.sqrt((p0.x_m - base_xm_0)**2 + (p0.y_m - base_ym_0)**2)
        # At vehicle bumper (y=0), rollout must start from actual vehicle pose (0 offset)
        self.assertAlmostEqual(dist_0, 0.0, places=3,
                               msg="Rollout must start from actual vehicle pose at y=0 without jump")

        # At steady-state lookahead distance (y_m >= transition_distance), offset must reach full d_offset
        p_steady = traj.points[-1]
        y_px_steady = p_steady.y_px
        base_xm_s, base_ym_s = self.bev.px_to_metric(np.polyval(poly, y_px_steady), y_px_steady)
        dist_s = math.sqrt((p_steady.x_m - base_xm_s)**2 + (p_steady.y_m - base_ym_s)**2)
        self.assertAlmostEqual(dist_s, abs(d_offset), delta=0.015,
                               msg="Steady state shift distance must reach full lateral offset along normal vector")


class TestPassabilityAndSpeedControl(unittest.TestCase):
    def setUp(self):
        self.cfg = V3Config()
        self.cfg.lidar_offset_deg = 0.0  # standard coordinate frame for synthetic scan
        self.speed_ctrl = SpeedController(self.cfg)
        self.avoidance = ObstacleAvoidance(self.cfg)

    def test_passability_check_blocks_narrow_passage(self):
        """Test that ObstacleAvoidance stops if remaining passage is narrower than car width."""
        # Obstacle directly in front (0 deg, 0.35m away), and BOTH left & right flanks blocked (0.12m away)
        ranges = [3.0] * 360
        # Obstacle in front (0 deg, 0.35m)
        for offset in range(-10, 11):
            ranges[(0 + offset + 180) % 360] = 0.35
        # Obstacle on left (45 deg, 0.12m away)
        for offset in range(-10, 11):
            ranges[(45 + offset + 180) % 360] = 0.12
        # Obstacle on right (-45 deg, 0.12m away)
        for offset in range(-10, 11):
            ranges[(-45 + offset + 180) % 360] = 0.12

        scan = MockLaserScan(ranges)
        res = self.avoidance.process(scan, dt=0.05)
        self.assertEqual(res.speed_factor, 0.0, "Vehicle must stop when passage clearance is insufficient")

    def test_speed_controller_crawl_mode(self):
        """Test that single-line mode ('L_only') reduces speed to crawl speed."""
        # Normal tracking: cruise speed (0.25)
        throttle_normal = self.speed_ctrl.compute(
            curvature=0.1, confidence=0.9, tracking_state=TrackingState.TRACKING, reconstruction_method='L+C+R_fused'
        )
        # Single line mode: crawl speed (0.12)
        throttle_single = self.speed_ctrl.compute(
            curvature=0.1, confidence=0.5, tracking_state=TrackingState.TRACKING, reconstruction_method='L_only'
        )
        self.assertLess(throttle_single, throttle_normal, "Single line mode should reduce speed")
        self.assertAlmostEqual(throttle_single, self.cfg.crawl_speed, delta=0.02)


if __name__ == '__main__':
    unittest.main()
