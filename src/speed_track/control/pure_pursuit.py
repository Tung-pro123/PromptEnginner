#!/usr/bin/env python3
"""
V3 Control — Pure Pursuit Controller (V1 Baseline)

Implements the Pure Pursuit lateral controller:
    kappa_pp = 2 * y_target / Ld²
    delta = atan(L * kappa_pp)

where:
    y_target = lateral offset of target point (in vehicle frame)
    Ld = lookahead distance
    L = wheelbase

This is the baseline controller. It must demonstrate stable tracking
before any other controller (Stanley) is enabled.

Pure Pursuit is geometrically intuitive and well-behaved:
- It naturally converges to the path
- It doesn't require a separate heading error calculation
- It degrades gracefully at low speeds
"""

import math


class PurePursuitController:
    """Pure Pursuit lateral controller.

    Takes a target point from the trajectory generator and
    computes a steering angle to drive toward it.
    """

    def __init__(self, config):
        """
        Args:
            config: V3Config instance.
        """
        self.wheelbase = getattr(config, 'wheelbase_m', getattr(config, 'wheelbase', 0.14))
        self.max_steer_rad = getattr(config, 'max_steer_rad', getattr(config, 'max_steer_angle_rad', 0.436))

    def compute(self, target_point, lookahead_m):
        """Compute steering angle using Pure Pursuit.

        Args:
            target_point: TrajectoryPoint with x_m, y_m in vehicle frame.
                          x_m > 0 = right, y_m > 0 = forward
            lookahead_m: Lookahead distance (meters). Must be > 0.

        Returns:
            Steering command in [-1.0, 1.0] (normalized for servo).
        """
        if lookahead_m < 1e-6:
            return 0.0

        # Pure Pursuit curvature
        # The target point lateral offset is x_m (positive = right)
        x_target = target_point.x_m
        kappa_pp = 2.0 * x_target / (lookahead_m ** 2)

        # Steering angle
        delta_rad = math.atan(self.wheelbase * kappa_pp)

        # Normalize to [-1, 1] for servo
        delta_norm = delta_rad / self.max_steer_rad
        delta_norm = max(-1.0, min(1.0, delta_norm))

        return delta_norm
