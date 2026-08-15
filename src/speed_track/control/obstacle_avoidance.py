#!/usr/bin/env python3
"""
V3 Control — Obstacle Avoidance (Virtual Offset + Flank Gating + State Machine)

State machine:
    CLEAR     — no obstacle, lane following along centerline (offset = 0.0)
    EVADING   — obstacle detected, virtual target centerline is ramped smoothly
                away from obstacle with safety corridor clamping
    RETURNING — obstacle passed in front AND cleared side flank (70°-110°),
                virtual offset ramps back to 0.0

Virtual Offset Architecture:
    Instead of conflicting APF steering force directly against Pure Pursuit (Control Fight),
    we shift the target trajectory by d_offset(t). Pure Pursuit naturally follows this
    collision-free virtual path without oscillation or jerk.
"""

import math
import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class AvoidState(Enum):
    CLEAR = 0
    EVADING = 1
    RETURNING = 2


class ObstacleSide(Enum):
    NONE = 0
    LEFT = 1      # obstacle is left of center → evade right (offset > 0)
    RIGHT = 2     # obstacle is right of center → evade left (offset < 0)


@dataclass
class ObstacleResult:
    """Output of one avoidance cycle."""
    state: AvoidState = AvoidState.CLEAR
    side: ObstacleSide = ObstacleSide.NONE
    virtual_offset_m: float = 0.0  # Ramped lateral offset for trajectory (meters, + = right)
    target_offset_m: float = 0.0   # Target steady-state offset (meters)
    apf_steer: float = 0.0         # Raw APF lateral force [-1, 1] for debug/fallback
    min_front: float = float('inf')
    speed_factor: float = 1.0      # [0, 1] multiplier for throttle
    e_stop: bool = False           # True = emergency stop


class ObstacleDetector:
    """Reads LaserScan and computes APF repulsive force, flank clearance, and obstacle side."""

    def __init__(self, config):
        self.cfg = config
        self.side_clear_dist = getattr(config, 'side_clear_dist', 0.35)
        self.side_scan_start = getattr(config, 'side_scan_start_deg', 70.0)
        self.side_scan_end = getattr(config, 'side_scan_end_deg', 110.0)

    def detect(self, scan_msg):
        """Compute APF steering and identify obstacle side.

        Args:
            scan_msg: sensor_msgs/LaserScan or None.

        Returns:
            (apf_steer, min_front, speed_factor, side)
        """
        if scan_msg is None:
            return 0.0, float('inf'), 1.0, ObstacleSide.NONE

        cfg = self.cfg
        lateral_force = 0.0
        min_front = float('inf')

        # Scan side spaces to determine bias for frontal obstacles
        # In ROS: Angle > 0 is LEFT, Angle < 0 is RIGHT
        left_min = self._sector_min(scan_msg, 30, 70)
        right_min = self._sector_min(scan_msg, -70, -30)

        # If left has more clearance (left_min >= right_min), evade LEFT (bias_dir = -1.0)
        # If right has more clearance (left_min < right_min), evade RIGHT (bias_dir = +1.0)
        bias_dir = -1.0 if left_min >= right_min else 1.0

        # Track weighted obstacle angle for side detection
        obstacle_angle_sum = 0.0
        obstacle_weight_sum = 0.0

        for i, d in enumerate(scan_msg.ranges):
            if not (scan_msg.range_min < d < scan_msg.range_max):
                continue
            if d > cfg.apf_influence_dist:
                continue

            deg = math.degrees(
                scan_msg.angle_min + i * scan_msg.angle_increment
            ) + cfg.lidar_offset_deg
            angle = math.radians((deg + 180) % 360 - 180)
            angle_deg = math.degrees(angle)

            if abs(angle_deg) > 90:
                continue

            # Track front distance (forward cone +/- 35 deg in front of bumper)
            if abs(angle_deg) < 35:
                min_front = min(min_front, d)

            # APF repulsive force (inversely proportional to distance)
            force = cfg.apf_gain * (1.0 / d - 1.0 / cfg.apf_influence_dist)

            if abs(angle_deg) < 15:
                # Frontal obstacle: push toward open side (bias_dir)
                lateral_force += force * cfg.apf_frontal_bias * bias_dir
            else:
                # Side obstacle:
                # Obstacle on LEFT (angle > 0, sin > 0) -> Push RIGHT (lateral_force > 0)
                # Obstacle on RIGHT (angle < 0, sin < 0) -> Push LEFT (lateral_force < 0)
                lateral_force += force * math.sin(angle)

            # Weight obstacle positions for side classification
            if d < cfg.obstacle_trigger_dist:
                w = 1.0 / (d * d)
                obstacle_angle_sum += angle_deg * w
                obstacle_weight_sum += w

        apf_steer = max(-1.0, min(1.0, lateral_force))

        # Speed factor: slow down near obstacles
        speed_factor = 1.0
        if min_front < cfg.apf_influence_dist:
            speed_factor = max(0.3, min_front / cfg.apf_influence_dist)

        # Determine which side the obstacle is on
        side = ObstacleSide.NONE
        if obstacle_weight_sum > 0:
            avg_angle = obstacle_angle_sum / obstacle_weight_sum
            if avg_angle > 5:
                # Obstacle on left -> evade right
                side = ObstacleSide.LEFT
            elif avg_angle < -5:
                # Obstacle on right -> evade left
                side = ObstacleSide.RIGHT
            else:
                # Frontal — pick side with more space
                # If left has more space, treat obstacle as biased RIGHT -> evade LEFT
                side = ObstacleSide.RIGHT if left_min >= right_min else ObstacleSide.LEFT

        return apf_steer, min_front, speed_factor, side

    def is_side_clear(self, scan_msg, side: ObstacleSide) -> bool:
        """Check if the obstacle beside the vehicle has cleared the flank.
        
        Args:
            scan_msg: ROS LaserScan.
            side: ObstacleSide (LEFT or RIGHT) where the obstacle was detected.
            
        Returns:
            True if no obstacle is detected within side_clear_dist in the flank sector.
        """
        if scan_msg is None or side == ObstacleSide.NONE:
            return True

        if side == ObstacleSide.LEFT:
            # Obstacle was on the left -> Check left flank sector (70° to 110°)
            flank_dist = self._sector_min(scan_msg, self.side_scan_start, self.side_scan_end)
        else:
            # Obstacle was on the right -> Check right flank sector (-110° to -70°)
            flank_dist = self._sector_min(scan_msg, -self.side_scan_end, -self.side_scan_start)

        return flank_dist > self.side_clear_dist

    def _sector_min(self, msg, a_min_deg, a_max_deg):
        """Get filtered minimum distance in an angular sector."""
        dists = []
        for i, d in enumerate(msg.ranges):
            if not (msg.range_min < d < msg.range_max):
                continue
            deg = math.degrees(
                msg.angle_min + i * msg.angle_increment
            ) + self.cfg.lidar_offset_deg
            a = (deg + 180) % 360 - 180
            if a_min_deg <= a <= a_max_deg:
                dists.append(d)
        if not dists:
            return float('inf')
        # Percentile filter: skip bottom 10% to reject noise spikes
        dists.sort()
        idx = min(len(dists) - 1, max(1, len(dists) // 10))
        return dists[idx]


class ObstacleAvoidance:
    """State machine that manages Virtual Offset for trajectory-level collision avoidance.

    Steering convention: negative = left, positive = right.
    Offset convention: negative = left (meters), positive = right (meters).

    State transitions:
        CLEAR → EVADING:     min_front < trigger_dist
        EVADING → RETURNING: min_front > clear_dist AND flank is clear (obstacle passed)
        EVADING → CLEAR:     timeout (obstacle vanished)
        RETURNING → CLEAR:   |lateral_error| < threshold AND |current_offset| < 0.02
        RETURNING → EVADING: new obstacle appears in front
    """

    def __init__(self, config):
        self.cfg = config
        self.detector = ObstacleDetector(config)

        self.state = AvoidState.CLEAR
        self.evade_side = ObstacleSide.NONE  # which side the obstacle was on
        self._state_time = time.time()
        self._last_time = time.time()
        self._evade_timeout = 5.0   # seconds max in EVADING
        self._return_timeout = 4.0  # seconds max in RETURNING

        # Virtual Offset states (meters)
        self.current_offset_m = 0.0
        self.target_offset_m = 0.0

    def get_max_safe_offset(self) -> float:
        """Calculate max allowable lateral evasion offset to stay inside road corridor."""
        cfg = self.cfg
        track_w = getattr(cfg, 'track_width_m', 0.60)
        car_w = getattr(cfg, 'car_width_m', 0.18)
        margin = getattr(cfg, 'safety_margin_m', 0.04)
        configured_max = getattr(cfg, 'max_evade_offset_m', 0.18)
        
        # Max safe displacement from centerline = Half Track - Half Car - Safety Margin
        corridor_max = (track_w / 2.0) - (car_w / 2.0) - margin
        return max(0.05, min(configured_max, corridor_max))

    def check_passability(self, scan_msg, side: ObstacleSide) -> bool:
        """Check if there is sufficient lateral clearance on the evasion side to pass safely.

        If remaining clearance < (car_width + safety_margin), the corridor is too narrow
        (e.g. obstacle on inside of a sharp curve blocking the full lane). The car must stop
        rather than forcing an evasion that crosses outer boundary lines.
        """
        if scan_msg is None or side == ObstacleSide.NONE:
            return True
        cfg = self.cfg
        car_w = getattr(cfg, 'car_width_m', 0.18)
        margin = getattr(cfg, 'safety_margin_m', 0.04)
        min_required_space = car_w + margin

        # Check clearance on the intended evasion side (30° to 70°)
        if side == ObstacleSide.LEFT:
            # Evading RIGHT -> Check right space (-70° to -30°)
            clearance = self.detector._sector_min(scan_msg, -70, -30)
        else:
            # Evading LEFT -> Check left space (30° to 70°)
            clearance = self.detector._sector_min(scan_msg, 30, 70)

        return clearance >= min_required_space

    def process(self, scan_msg, lane_steer=0.0, lateral_error_m=0.0, dt: Optional[float] = None) -> ObstacleResult:
        """Update obstacle avoidance state and calculate smooth virtual offset.

        Args:
            scan_msg: sensor_msgs/LaserScan or None.
            lane_steer: current lane steering [-1, 1].
            lateral_error_m: current lateral error (meters, + = right of center).
            dt: time step in seconds (computed automatically if None).

        Returns:
            ObstacleResult containing state, virtual_offset_m, speed_factor, e_stop.
        """
        cfg = self.cfg
        now = time.time()
        
        if dt is None:
            dt = max(0.001, min(0.1, now - self._last_time))
        self._last_time = now

        apf_steer, min_front, speed_factor, side = self.detector.detect(scan_msg)

        result = ObstacleResult(
            apf_steer=apf_steer,
            min_front=min_front,
            speed_factor=speed_factor,
        )

        # Emergency stop if dangerously close
        if min_front < cfg.obstacle_e_stop_dist:
            result.e_stop = True
            result.state = self.state
            result.side = self.evade_side
            result.virtual_offset_m = self.current_offset_m
            result.target_offset_m = self.target_offset_m
            return result

        # Passability Check: if corridor is too narrow on evasion side, decelerate to stop
        if side != ObstacleSide.NONE and min_front < cfg.obstacle_trigger_dist:
            if not self.check_passability(scan_msg, side):
                result.speed_factor = 0.0  # Safe stop: lane blocked without sufficient room
                result.state = AvoidState.EVADING
                result.side = side
                return result

        max_safe_offset = self.get_max_safe_offset()
        time_in_state = now - self._state_time

        # --- State Machine Transitions ---
        if self.state == AvoidState.CLEAR:
            self.target_offset_m = 0.0
            if min_front < cfg.obstacle_trigger_dist and side != ObstacleSide.NONE:
                self._set_state(AvoidState.EVADING, now)
                self.evade_side = side
                # Obstacle on LEFT -> Evade RIGHT (+offset)
                # Obstacle on RIGHT -> Evade LEFT (-offset)
                self.target_offset_m = max_safe_offset if side == ObstacleSide.LEFT else -max_safe_offset

        elif self.state == AvoidState.EVADING:
            if side != ObstacleSide.NONE:
                self.evade_side = side
                self.target_offset_m = max_safe_offset if self.evade_side == ObstacleSide.LEFT else -max_safe_offset

            # Check if obstacle has cleared both front AND side flank
            side_cleared = self.detector.is_side_clear(scan_msg, self.evade_side)
            if min_front > cfg.obstacle_clear_dist and side_cleared:
                # Obstacle fully passed — ramp back to center line
                self._set_state(AvoidState.RETURNING, now)
                self.target_offset_m = 0.0
            elif time_in_state > self._evade_timeout:
                self._set_state(AvoidState.RETURNING, now)
                self.target_offset_m = 0.0

        elif self.state == AvoidState.RETURNING:
            if min_front < cfg.obstacle_trigger_dist and side != ObstacleSide.NONE:
                # New obstacle encountered while returning
                self._set_state(AvoidState.EVADING, now)
                self.evade_side = side
                self.target_offset_m = max_safe_offset if side == ObstacleSide.LEFT else -max_safe_offset
            elif abs(self.current_offset_m) < 0.02 and abs(lateral_error_m) < cfg.return_lateral_threshold:
                self._set_state(AvoidState.CLEAR, now)
                self.evade_side = ObstacleSide.NONE
                self.target_offset_m = 0.0
            elif time_in_state > self._return_timeout:
                self._set_state(AvoidState.CLEAR, now)
                self.evade_side = ObstacleSide.NONE
                self.target_offset_m = 0.0

        # --- Smooth Linear/S-Curve Ramp of Virtual Offset ---
        ramp_rate = getattr(cfg, 'offset_ramp_rate', 0.60)  # m/s
        max_delta = ramp_rate * dt
        diff = self.target_offset_m - self.current_offset_m

        if abs(diff) <= max_delta:
            self.current_offset_m = self.target_offset_m
        else:
            self.current_offset_m += math.copysign(max_delta, diff)

        # Clamping safety guard
        self.current_offset_m = max(-max_safe_offset, min(max_safe_offset, self.current_offset_m))

        result.state = self.state
        result.side = self.evade_side
        result.virtual_offset_m = self.current_offset_m
        result.target_offset_m = self.target_offset_m

        return result

    def fuse_steering(self, lane_steer: float, result: ObstacleResult) -> float:
        """Fallback steering clamp for safety if APF hybrid mode is enabled.

        When using Virtual Offset, Pure Pursuit handles steering directly.
        This function ensures steering does not re-enter the obstacle sector during RETURNING.

        Args:
            lane_steer: steering from lane following [-1, 1].
            result: ObstacleResult from process().

        Returns:
            Final steering [-1, 1].
        """
        if result.e_stop:
            return 0.0

        if result.state == AvoidState.RETURNING:
            if result.side == ObstacleSide.LEFT:
                # Was on left, evaded right, now returning left -> Do not allow steer right (>0)
                return min(0.0, lane_steer)
            elif result.side == ObstacleSide.RIGHT:
                # Was on right, evaded left, now returning right -> Do not allow steer left (<0)
                return max(0.0, lane_steer)

        return lane_steer

    def _set_state(self, new_state, now):
        self.state = new_state
        self._state_time = now
