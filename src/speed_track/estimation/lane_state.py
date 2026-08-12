#!/usr/bin/env python3
"""
V3 Estimation — Temporal Lane State & Measurement Gating

Maintains a persistent lane state across frames using temporal filtering.
This is the core module that makes V3 stable compared to V2's frame-by-frame
approach.

Key behaviors:
- Accepts or rejects new observations (measurement gating)
- Smooths accepted measurements with EMA (alpha-beta filter)
- Predicts from previous state when observations are rejected
- Decays confidence over time when no valid measurement arrives
- Tracks the overall tracking state (SEARCH/TRACKING/UNCERTAIN/etc.)

Design is extensible for Kalman filter replacement later.
"""

import time
import math
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class TrackingState(Enum):
    """Tracking state machine states."""
    SEARCH = 0       # Looking for lane lines
    TRACKING = 1     # Normal operation, good detections
    UNCERTAIN = 2    # Partial line loss, confidence dropping
    PREDICTING = 3   # All lines lost, using prediction only
    RECOVERY = 4     # Slow, searching for lines
    E_STOP = 5       # Emergency stop


@dataclass
class LaneState:
    """Persistent lane state maintained across frames."""
    centerline_poly: Optional[np.ndarray] = None  # [a, b, c]
    lane_width_m: float = 0.0
    curvature: float = 0.0              # 1/m at vehicle position
    heading_error: float = 0.0          # radians
    lateral_error_m: float = 0.0        # meters
    confidence: float = 0.0             # [0, 1]
    timestamp: float = 0.0
    tracking_state: TrackingState = TrackingState.SEARCH
    state_entry_time: float = field(default_factory=time.time)       # when current state was entered

    # Per-line confidences (for debug)
    left_conf: float = 0.0
    center_conf: float = 0.0
    right_conf: float = 0.0
    reconstruction_method: str = 'none'


class LaneStateEstimator:
    """Temporal lane state estimator with measurement gating.

    Updates:
    1. Receive observation from LaneGeometry
    2. Gate: accept or reject based on physical plausibility
    3. If accepted: EMA update
    4. If rejected: predict from previous state, decay confidence
    5. Update tracking state machine
    """

    def __init__(self, config, bev_transform):
        """
        Args:
            config: V3Config instance.
            bev_transform: BEVTransform instance.
        """
        self.cfg = config
        self.bev = bev_transform
        self.state = LaneState()

    def update(self, observation):
        """Process a new geometry observation and update lane state.

        Args:
            observation: GeometryObservation from LaneGeometry.

        Returns:
            Updated LaneState.
        """
        now = time.time()

        if not observation.valid or observation.centerline_poly is None:
            # No valid observation — pure prediction
            self._predict(now)
            self._update_state_machine(now, measurement_accepted=False)
            return self.state

        # Compute derived quantities from observation
        obs_e_lat, obs_e_psi, obs_kappa = self._compute_errors(
            observation.centerline_poly
        )

        # Measurement gating
        if self.state.centerline_poly is not None:
            # We have a previous state — check for impossible jumps
            if not self._passes_gating(obs_e_lat, obs_kappa, observation.overall_confidence):
                # Observation rejected — predict instead
                self._predict(now)
                self._update_state_machine(now, measurement_accepted=False)
                return self.state

        # Measurement accepted — EMA update
        self._ema_update(observation, obs_e_lat, obs_e_psi, obs_kappa, now)
        self._update_state_machine(now, measurement_accepted=True)

        return self.state

    def _compute_errors(self, centerline_poly):
        """Compute lateral error, heading error, and curvature from a centerline.

        The vehicle is at the bottom-center of the BEV image.

        Args:
            centerline_poly: [a, b, c] polynomial coefficients (x = f(y) in pixels).

        Returns:
            (e_lat_m, e_psi_rad, kappa) in metric units.
        """
        bev_h = self.cfg.image_height
        bev_w = self.cfg.image_width

        # Vehicle position in BEV: bottom center
        y_vehicle = float(bev_h)
        x_center = bev_w / 2.0

        a, b, c = centerline_poly[0], centerline_poly[1], centerline_poly[2]

        # Lateral error: where is the centerline at the vehicle's y-position?
        x_line_at_vehicle = np.polyval(centerline_poly, y_vehicle)
        e_lat_px = x_line_at_vehicle - x_center
        e_lat_m = e_lat_px / self.cfg.px_per_meter_x

        # Heading error: tangent angle of the centerline at vehicle position
        # dx/dy = 2*a*y + b (in pixel space)
        dx_dy_px = 2.0 * a * y_vehicle + b

        # Convert to metric derivative
        sx = 1.0 / self.cfg.px_per_meter_x
        sy = 1.0 / self.cfg.px_per_meter_y
        dx_dy_m = dx_dy_px * (sx / sy)

        # Heading error = angle of tangent vector
        # In the vehicle frame, the road should go straight up (dy direction).
        # The tangent in (dy_m, dx_m) = (1, dx_dy_m), so the heading error is:
        e_psi = math.atan(dx_dy_m)

        # Curvature in metric coordinates
        d2x_dy2_px = 2.0 * a
        d2x_dy2_m = d2x_dy2_px * (sx / (sy * sy))
        kappa = d2x_dy2_m / ((1.0 + dx_dy_m ** 2) ** 1.5)

        return e_lat_m, e_psi, kappa

    def _passes_gating(self, obs_e_lat, obs_kappa, obs_confidence):
        """Check if a new observation is physically plausible.

        Rejects observations that jump too far from the previous state.

        Args:
            obs_e_lat: Observed lateral error (meters).
            obs_kappa: Observed curvature (1/m).
            obs_confidence: Observation confidence [0,1].

        Returns:
            True if the observation passes all gates.
        """
        prev = self.state

        # Gate 1: Confidence too low
        if obs_confidence < self.cfg.min_confidence_gate:
            return False

        # Gate 2: Lateral position jump
        if abs(obs_e_lat - prev.lateral_error_m) > self.cfg.max_lateral_jump_m:
            return False

        # Gate 3: Curvature jump
        if abs(obs_kappa - prev.curvature) > self.cfg.max_curvature_jump:
            return False

        return True

    def _ema_update(self, observation, e_lat, e_psi, kappa, now):
        """Apply EMA (exponential moving average) update.

        Blends the new observation with the previous state.

        Args:
            observation: GeometryObservation.
            e_lat, e_psi, kappa: Computed errors from the observation.
            now: Current timestamp.
        """
        cfg = self.cfg

        if self.state.centerline_poly is None:
            # First valid observation — initialize directly
            self.state.centerline_poly = observation.centerline_poly.copy()
            self.state.lateral_error_m = e_lat
            self.state.heading_error = e_psi
            self.state.curvature = kappa
            self.state.lane_width_m = observation.lane_width_m
            self.state.confidence = observation.overall_confidence
        else:
            # EMA blend
            a_pos = cfg.alpha_position
            a_head = cfg.alpha_heading
            a_curv = cfg.alpha_curvature
            a_width = cfg.alpha_width

            self.state.centerline_poly = (
                a_pos * observation.centerline_poly +
                (1.0 - a_pos) * self.state.centerline_poly
            )
            self.state.lateral_error_m = (
                a_pos * e_lat + (1.0 - a_pos) * self.state.lateral_error_m
            )
            self.state.heading_error = (
                a_head * e_psi + (1.0 - a_head) * self.state.heading_error
            )
            self.state.curvature = (
                a_curv * kappa + (1.0 - a_curv) * self.state.curvature
            )

            if observation.lane_width_m > 0:
                self.state.lane_width_m = (
                    a_width * observation.lane_width_m +
                    (1.0 - a_width) * self.state.lane_width_m
                )

            self.state.confidence = (
                a_pos * observation.overall_confidence +
                (1.0 - a_pos) * self.state.confidence
            )

        self.state.left_conf = observation.left_conf
        self.state.center_conf = observation.center_conf
        self.state.right_conf = observation.right_conf
        self.state.reconstruction_method = observation.method
        self.state.timestamp = now

    def _predict(self, now):
        """Predict lane state when no valid measurement is available.

        Keeps the previous state but decays confidence and steers towards straight ahead.

        Args:
            now: Current timestamp.
        """
        self.state.confidence *= self.cfg.confidence_decay
        self.state.timestamp = now
        self.state.reconstruction_method = 'prediction'
        
        # Ý TƯỞNG 1: Trả lái về thẳng khi mất vạch
        # Từ từ đưa độ cong và độ lệch về 0 (nhân 0.8) để vô lăng nhả thẳng
        self.state.curvature *= 0.8
        self.state.lateral_error_m *= 0.8
        self.state.heading_error *= 0.8

    def _update_state_machine(self, now, measurement_accepted):
        """Update the tracking state based on confidence and timing.

        Args:
            now: Current timestamp.
            measurement_accepted: Whether the latest measurement was accepted.
        """
        prev_state = self.state.tracking_state
        conf = self.state.confidence
        cfg = self.cfg
        time_in_state = now - self.state.state_entry_time

        new_state = prev_state

        if prev_state == TrackingState.SEARCH:
            if measurement_accepted and conf >= cfg.tracking_confidence_min:
                new_state = TrackingState.TRACKING
            elif time_in_state > cfg.search_timeout:
                new_state = TrackingState.E_STOP

        elif prev_state == TrackingState.TRACKING:
            if conf < cfg.uncertain_confidence_thresh:
                new_state = TrackingState.UNCERTAIN

        elif prev_state == TrackingState.UNCERTAIN:
            if measurement_accepted and conf >= cfg.tracking_confidence_min:
                new_state = TrackingState.TRACKING
            elif time_in_state > cfg.uncertain_timeout:
                new_state = TrackingState.PREDICTING

        elif prev_state == TrackingState.PREDICTING:
            if measurement_accepted and conf >= cfg.tracking_confidence_min:
                new_state = TrackingState.TRACKING
            elif time_in_state > cfg.predicting_timeout:
                new_state = TrackingState.RECOVERY

        elif prev_state == TrackingState.RECOVERY:
            if measurement_accepted and conf >= cfg.tracking_confidence_min:
                new_state = TrackingState.TRACKING
            elif time_in_state > cfg.recovery_timeout:
                new_state = TrackingState.E_STOP

        # Transition
        if new_state != prev_state:
            self.state.tracking_state = new_state
            self.state.state_entry_time = now

    def reset(self):
        """Reset the estimator to initial state (e.g., for restart)."""
        self.state = LaneState()
