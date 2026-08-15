#!/usr/bin/env python3
"""
V3 Control — Stanley + Curvature Feedforward Controller (V2)

Implements the Stanley lateral controller with curvature feedforward:
    delta = e_psi + atan(k * e_y / (v + epsilon)) + atan(L * kappa)

where:
    e_psi = heading error (radians)
    e_y = lateral error (meters)
    v = vehicle speed (m/s)
    k = Stanley gain
    L = wheelbase
    kappa = curvature (1/m)

This controller is DISABLED by default (config.stanley_enabled = False).
It should only be enabled after Pure Pursuit demonstrates stable tracking.

Differences from V2's Stanley:
- Uses proper metric units (not px_to_m scalar conversion)
- Receives pre-computed heading error from trajectory module
- Has configurable enable/disable flag
"""

import math


class StanleyController:
    """Stanley + curvature feedforward lateral controller."""

    def __init__(self, config):
        """
        Args:
            config: V3Config instance.
        """
        self.k = config.stanley_k
        self.wheelbase = getattr(config, 'wheelbase_m', getattr(config, 'wheelbase', 0.14))
        self.max_steer_rad = getattr(config, 'max_steer_rad', getattr(config, 'max_steer_angle_rad', 0.436))
        self.epsilon = 0.01  # Prevent division by zero at low speed

    def compute(self, heading_error, lateral_error_m, curvature, speed):
        """Compute steering angle using Stanley + feedforward.

        Args:
            heading_error: e_psi in radians (positive = road curves right).
            lateral_error_m: e_y in meters (positive = vehicle is right of center).
            curvature: kappa in 1/m (positive = curving right).
            speed: Vehicle speed in m/s.

        Returns:
            Steering command in [-1.0, 1.0] (normalized for servo).
        """
        # Heading correction
        delta_heading = heading_error

        # Cross-track correction (Stanley term)
        delta_crosstrack = math.atan(self.k * lateral_error_m / (speed + self.epsilon))

        # Curvature feedforward
        delta_feedforward = math.atan(self.wheelbase * curvature)

        # Total steering angle
        delta_rad = delta_heading + delta_crosstrack + delta_feedforward

        # Normalize to [-1, 1] for servo
        delta_norm = delta_rad / self.max_steer_rad
        delta_norm = max(-1.0, min(1.0, delta_norm))

        return delta_norm
