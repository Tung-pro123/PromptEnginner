#!/usr/bin/env python3
"""
V3 Control — Physical-Radian Steering Rate Limiter & Smoothing Filter

Applies real-world physical constraints:
1. Physical Saturation: Clamped strictly to [-max_steer_rad, +max_steer_rad].
2. Physical Servo Slew Rate Limiting: Limits steering delta per loop to (max_steer_rate_rad_s * dt).
3. Configurable Low-Pass Filter (EMA) for high-frequency noise rejection.

Convention:
    δ > 0: Steer RIGHT (Rẽ phải)
    δ < 0: Steer LEFT (Rẽ trái)
    Unit: Radians
"""

import math


class SteeringFilter:
    """Rate-limited and low-pass steering filter in physical radians."""

    def __init__(self, config):
        """
        Args:
            config: V3Config instance.
        """
        self.cfg = config
        self.max_steer_rad = getattr(config, 'max_steer_rad', 0.436)
        self.max_steer_rate_rad_s = getattr(config, 'max_steer_rate_rad_s', 2.5)
        self.lpf_alpha = getattr(config, 'steer_lpf_alpha', 0.8)
        self.prev_steer_rad = 0.0

    def filter_rad(self, target_rad: float, dt: float = 0.025) -> float:
        """Apply physical radian rate-limiting (servo slew rate) and optional LPF.

        Args:
            target_rad: Target steering angle in physical radians (+ = right, - = left).
            dt: Measured loop time in seconds.

        Returns:
            Filtered steering angle in physical radians clamped to [-max_steer_rad, +max_steer_rad].
        """
        dt_clamped = max(0.001, min(0.20, dt))

        # Step 1: Chassis physical clamp
        target_clamped = max(-self.max_steer_rad, min(self.max_steer_rad, target_rad))

        # Step 2: Rate limiting based on physical servo slew rate (rad/s) and real loop dt
        max_change = self.max_steer_rate_rad_s * dt_clamped
        delta = target_clamped - self.prev_steer_rad
        if abs(delta) > max_change:
            delta = math.copysign(max_change, delta)
        rate_limited = self.prev_steer_rad + delta

        # Step 3: Low-pass filter (EMA)
        filtered_rad = self.lpf_alpha * rate_limited + (1.0 - self.lpf_alpha) * self.prev_steer_rad

        # Final safety clamp
        filtered_rad = max(-self.max_steer_rad, min(self.max_steer_rad, filtered_rad))
        self.prev_steer_rad = filtered_rad
        return filtered_rad

    def filter(self, raw_steering: float, dt: float = 0.025) -> float:
        """Backward-compatible filter for normalized steering in [-1.0, 1.0].

        Args:
            raw_steering: Normalized steering command in [-1.0, 1.0].
            dt: Measured loop time in seconds.

        Returns:
            Filtered normalized steering command in [-1.0, 1.0].
        """
        target_rad = raw_steering * self.max_steer_rad
        filtered_rad = self.filter_rad(target_rad, dt=dt)
        return filtered_rad / self.max_steer_rad

    def reset(self, initial_steer_rad: float = 0.0):
        """Reset filter state (e.g., on emergency stop or state transition)."""
        self.prev_steer_rad = initial_steer_rad
