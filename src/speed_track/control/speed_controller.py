#!/usr/bin/env python3
"""
V3 Control — Curvature-Based Speed Controller

Computes target speed based on:
1. Road curvature:  v_curve = sqrt(a_lat_max / (|kappa| + eps))
2. Lane confidence: reduce speed when perception is uncertain
3. Tracking state: reduce/stop based on state machine

Supports optional encoder-based PID for closed-loop speed control.
Falls back to open-loop throttle mapping when no encoder is available.

V2 used a fixed px_to_m scalar and got incorrect v_max values.
V3 uses proper metric curvature from the trajectory module.
"""

import math
import time


class SpeedController:
    """Curvature-aware speed controller with optional PID."""

    def __init__(self, config):
        """
        Args:
            config: V3Config instance.
        """
        self.cfg = config

        # PID state (for encoder feedback mode)
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def compute(self, curvature, confidence, tracking_state, actual_speed=None, reconstruction_method='none'):
        """Compute target throttle.

        Args:
            curvature: Road curvature (1/m) from trajectory module.
            confidence: Lane confidence [0,1] from state estimator.
            tracking_state: TrackingState enum value.
            actual_speed: Actual speed from encoder (m/s). None if no encoder.
            reconstruction_method: Method used to reconstruct centerline ('L+C+R_fused', 'L+R_midpoint', 'L_only', 'R_only', 'prediction').

        Returns:
            Throttle command (0.0 to max_speed).
        """
        from src.speed_track.estimation.lane_state import TrackingState

        cfg = self.cfg
        crawl = getattr(cfg, 'crawl_speed', 0.12)

        # Step 1: Curvature-based target speed
        epsilon = 1e-4
        v_curve = math.sqrt(cfg.a_lat_max / (abs(curvature) + epsilon))
        v_target = min(v_curve, cfg.cruise_speed)

        # Step 2: Confidence scaling
        if confidence < cfg.speed_confidence_thresh:
            scale = max(0.3, confidence / cfg.speed_confidence_thresh)
            v_target *= scale

        # Step 3: Single-line reconstruction speed limit (Caution mode)
        if reconstruction_method in ('L_only', 'R_only'):
            v_target = min(v_target, crawl)

        # Step 4: State machine speed limits
        if tracking_state == TrackingState.SEARCH:
            v_target = cfg.min_speed
        elif tracking_state == TrackingState.UNCERTAIN:
            v_target = min(v_target, cfg.cruise_speed * 0.7)
        elif tracking_state == TrackingState.PREDICTING:
            v_target = crawl
        elif tracking_state == TrackingState.RECOVERY:
            v_target = crawl
        elif tracking_state == TrackingState.E_STOP:
            v_target = 0.0

        # Clamp to valid range
        if v_target > 0:
            v_target = max(crawl, min(cfg.max_speed, v_target))
        else:
            v_target = 0.0

        # Step 4: Convert to throttle
        if cfg.use_encoder and actual_speed is not None:
            # Closed-loop PID
            throttle = self._pid_compute(v_target, actual_speed)
        else:
            # Open-loop mapping
            throttle = v_target * cfg.speed_to_throttle_factor

        # Final clamp
        throttle = max(0.0, min(cfg.max_speed, throttle))

        return throttle

    def _pid_compute(self, target_speed, actual_speed):
        """PID control for speed (encoder feedback).

        Args:
            target_speed: Desired speed (m/s).
            actual_speed: Measured speed from encoder (m/s).

        Returns:
            Throttle command.
        """
        cfg = self.cfg
        now = time.time()

        error = target_speed - actual_speed

        if self._prev_time is None:
            dt = 0.05
        else:
            dt = max(0.01, now - self._prev_time)

        # P
        p = cfg.speed_pid_kp * error

        # I (with anti-windup)
        self._integral += error * dt
        self._integral = max(-1.0, min(1.0, self._integral))
        i = cfg.speed_pid_ki * self._integral

        # D
        d = cfg.speed_pid_kd * (error - self._prev_error) / dt

        self._prev_error = error
        self._prev_time = now

        return p + i + d

    def reset(self):
        """Reset PID state."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None
