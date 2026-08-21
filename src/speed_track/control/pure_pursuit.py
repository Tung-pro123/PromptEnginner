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
    """Pure Pursuit lateral controller with Curvature Feedforward.

    Takes a target point and road curvature to compute predictive steering
    that never understeers in sharp hairpin chicanes.
    """

    def __init__(self, config):
        """
        Args:
            config: V3Config instance.
        """
        self.wheelbase = config.wheelbase
        self.max_steer_rad = config.max_steer_angle_rad
        self.k_ff = getattr(config, 'curvature_feedforward_gain', 0.50)
        self.config = config

    def compute(self, target_point, lookahead_m, curvature=0.0, k_ff=None):
        """Compute steering angle using Pure Pursuit + Curvature Feedforward.

        Args:
            target_point: TrajectoryPoint with x_m, y_m in vehicle frame.
                          x_m > 0 = right, y_m > 0 = forward
            lookahead_m: Lookahead distance (meters). Must be > 0.
            curvature: Road curvature (1/m) at vehicle position (+ = right, - = left).
            k_ff: Optional dynamic feedforward gain (from Sector Profile).

        Returns:
            Steering command in [-1.0, 1.0] (normalized for servo).
        """
        if lookahead_m < 1e-6:
            return 0.0

        # Pure Pursuit feedback steering (bẻ lái sửa sai lệch tâm)
        x_target = target_point.x_m
        kappa_pp = 2.0 * x_target / (lookahead_m ** 2)
        delta_fb = math.atan(self.wheelbase * kappa_pp)

        # ACTIVE DRIFT CORRECTION BOOST: Khi phát hiện lệch tim đường > 6cm, tăng lực bẻ lái +25%
        if abs(x_target) > 0.06:
            delta_fb *= 1.25

        # Curvature Feedforward with deadzone:
        # Chỉ kích hoạt bù góc lái khi vào cua gắt (|kappa| >= deadzone), tuyệt đối không kích hoạt trên đường thẳng
        effective_k_ff = k_ff if k_ff is not None else self.k_ff
        ff_deadzone = getattr(self.config, 'curvature_feedforward_deadzone', 0.40)
        if abs(curvature) >= ff_deadzone:
            delta_ff = math.atan(self.wheelbase * curvature)
        else:
            delta_ff = 0.0

        # Total steering command
        delta_rad = delta_fb + effective_k_ff * delta_ff

        # Normalize to [-1, 1] for servo
        delta_norm = delta_rad / self.max_steer_rad
        return max(-1.0, min(1.0, delta_norm))
