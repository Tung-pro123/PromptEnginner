#!/usr/bin/env python3
"""
V3 Control — Look-Ahead Trajectory Generation

Generates a trajectory of multiple look-ahead points along the estimated
centerline, computes metric curvature, and selects the target point at
an adaptive lookahead distance.

Key differences from V2:
- V2 used a single point at the bottom of the BEV image
- V3 generates multiple points along the centerline and selects one at
  an adaptive distance that depends on speed and curvature
- V2 computed curvature with a wrong px_to_m scalar conversion
- V3 computes curvature properly in metric coordinates
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrajectoryPoint:
    """A single point on the look-ahead trajectory."""
    x_m: float = 0.0    # lateral position (meters, + = right)
    y_m: float = 0.0    # longitudinal position (meters, + = forward)
    x_px: float = 0.0   # BEV pixel x
    y_px: float = 0.0   # BEV pixel y
    yaw_rad: float = 0.0 # Vehicle heading angle relative to vehicle axis (radians)
    steer_rad: float = 0.0 # Front wheel steering angle (radians)


@dataclass
class TrajectoryResult:
    """Complete trajectory generation result."""
    points: list = None            # List of TrajectoryPoint
    target: Optional[TrajectoryPoint] = None  # Selected target at Ld
    lookahead_m: float = 0.0       # Adaptive lookahead distance (m)
    curvature: float = 0.0         # Metric curvature at vehicle (1/m)
    heading_error: float = 0.0     # Heading error (radians)
    lateral_error_m: float = 0.0   # Lateral error (meters)

    def __post_init__(self):
        if self.points is None:
            self.points = []


class KinematicBicycleModel:
    """Standard Single-Track Ackermann Kinematic Bicycle Model.
    
    State: [x, y, psi, delta] in vehicle rear-axle ego frame:
      +y = forward
      +x = right
      psi = vehicle heading / yaw (rad, positive = clockwise/turning right)
      delta = front steering angle (rad, positive = steering right)
    """

    def __init__(self, config, bev_transform):
        self.cfg = config
        self.bev = bev_transform
        self.wheelbase = getattr(config, 'wheelbase_m', 0.14)
        self.max_steer_rad = getattr(config, 'max_steer_rad', 0.436)
        self.max_steer_rate = getattr(config, 'max_steer_rate_rad_s', 2.5)

    def simulate_rollout(self, delta_target: float, current_steer_rad: float = 0.0,
                         speed_m_s: float = 0.25, horizon_s: float = 0.90, dt_s: float = 0.03) -> list:
        """Simulate forward trajectory for a given target steering angle from actual vehicle pose."""
        v = max(0.15, speed_m_s)  # Use effective planning velocity
        n_steps = max(10, int(horizon_s / dt_s))
        
        x = 0.0
        y = 0.0
        psi = 0.0
        delta = current_steer_rad
        
        points = []
        # Step 0 (Vehicle pose)
        x_px, y_px = self.bev.metric_to_px(x, y)
        points.append(TrajectoryPoint(x_m=x, y_m=y, x_px=x_px, y_px=y_px, yaw_rad=psi, steer_rad=delta))
        
        for _ in range(n_steps):
            # Servo rate limit
            delta_diff = delta_target - delta
            max_delta_change = self.max_steer_rate * dt_s
            delta_inc = max(-max_delta_change, min(max_delta_change, delta_diff))
            delta = max(-self.max_steer_rad, min(self.max_steer_rad, delta + delta_inc))
            
            # Bicycle kinematics
            psi_dot = (v / self.wheelbase) * math.tan(delta)
            psi = psi + psi_dot * dt_s
            x = x + v * math.sin(psi) * dt_s
            y = y + v * math.cos(psi) * dt_s
            
            x_px, y_px = self.bev.metric_to_px(x, y)
            points.append(TrajectoryPoint(x_m=x, y_m=y, x_px=x_px, y_px=y_px, yaw_rad=psi, steer_rad=delta))
            
        return points


class TrajectoryGenerator:
    """Generates look-ahead trajectory from estimated centerline.

    The trajectory consists of multiple points sampled along the
    centerline polynomial in BEV space, converted to metric coordinates.
    """

    def __init__(self, config, bev_transform):
        """
        Args:
            config: V3Config instance.
            bev_transform: BEVTransform for pixel-to-metric conversion.
        """
        self.cfg = config
        self.bev = bev_transform

    def generate(self, lane_state, current_speed=0.0, lateral_offset_m: float = 0.0):
        """Generate a look-ahead trajectory from the current lane state with Normal-Vector offset.

        In curved sections, lateral offset is applied perpendicular to the local road tangent
        (along the unit normal vector n = [cos(theta), -sin(theta)]), ensuring constant clearance
        regardless of curve sharpness.

        Args:
            lane_state: LaneState with centerline polynomial.
            current_speed: Current vehicle speed (m/s). Used for adaptive Ld.
            lateral_offset_m: Virtual lateral shift in meters (+ = right, - = left).

        Returns:
            TrajectoryResult with points, target, and derived quantities.
        """
        result = TrajectoryResult()

        if lane_state.centerline_poly is None:
            return result

        poly = lane_state.centerline_poly
        bev_h = self.cfg.image_height
        a, b = poly[0], poly[1]
        sx = 1.0 / self.cfg.px_per_meter_x
        sy = 1.0 / self.cfg.px_per_meter_y

        # Sample points along the centerline in BEV pixel space
        # Shifted along the local normal vector for curved obstacle avoidance
        n_pts = self.cfg.n_lookahead_points
        y_values = np.linspace(bev_h, bev_h * 0.1, n_pts)

        l_trans = getattr(self.cfg, 'transition_distance_m', 0.45)
        points = []

        for y_px in y_values:
            x_base_px = np.polyval(poly, y_px)
            x_m_base, y_m_base = self.bev.px_to_metric(x_base_px, y_px)

            if abs(lateral_offset_m) > 1e-4:
                # Tangent slope dx_m / dy_m (note dy_m = -sy * dy_px)
                dx_dy_px = 2.0 * a * y_px + b
                dx_dy_m = -dx_dy_px * (sx / sy)
                theta_road = math.atan(dx_dy_m)

                # Kinematic Hermite S-Curve Transition from vehicle pose (P0 Fix):
                # Starts at 0 offset at y_m=0 and smoothly establishes target offset at y_m >= l_trans
                if y_m_base <= 0.0:
                    blend = 0.0
                elif y_m_base < l_trans:
                    u = y_m_base / l_trans
                    blend = 3.0 * (u ** 2) - 2.0 * (u ** 3)
                else:
                    blend = 1.0

                eff_offset = lateral_offset_m * blend

                # Unit normal vector pointing right: n = (cos(theta), -sin(theta))
                x_m = x_m_base + eff_offset * math.cos(theta_road)
                y_m = y_m_base - eff_offset * math.sin(theta_road)
                x_px, y_px_shifted = self.bev.metric_to_px(x_m, y_m)
            else:
                x_m, y_m = x_m_base, y_m_base
                x_px, y_px_shifted = x_base_px, y_px

            points.append(TrajectoryPoint(x_m=x_m, y_m=y_m, x_px=x_px, y_px=y_px_shifted))

        result.points = points

        # Compute curvature at vehicle position (in metric coordinates)
        kappa_func = self.bev.curvature_px_to_metric(poly)
        result.curvature = kappa_func(float(bev_h))

        # Compute heading error at vehicle position
        dx_dy_px_veh = 2.0 * a * float(bev_h) + b
        dx_dy_m_veh = dx_dy_px_veh * (sx / sy)
        result.heading_error = math.atan(dx_dy_m_veh)

        # Lateral error at vehicle position relative to centerline / virtual target
        if abs(lateral_offset_m) > 1e-4:
            theta_0 = math.atan(-dx_dy_px_veh * (sx / sy))
            x_m_veh_base, _ = self.bev.px_to_metric(np.polyval(poly, float(bev_h)), float(bev_h))
            result.lateral_error_m = x_m_veh_base + lateral_offset_m * math.cos(theta_0)
        else:
            x_center_px = self.cfg.image_width / 2.0
            x_line_px = np.polyval(poly, float(bev_h))
            result.lateral_error_m = (x_line_px - x_center_px) / self.cfg.px_per_meter_x

        # Adaptive lookahead distance (contracts on sharp curves to capture next dash)
        Ld = self._adaptive_lookahead(current_speed, result.curvature)
        result.lookahead_m = Ld

        # Select target point at distance Ld along the trajectory
        result.target = self._select_target(points, Ld)

        return result

    def _adaptive_lookahead(self, speed, curvature):
        """Compute adaptive lookahead distance.

        Ld = clip(L0 + kv*v - kk*|kappa|, Lmin, Lmax)

        Longer lookahead for higher speed, shorter for tighter curves.

        Args:
            speed: Current speed (m/s).
            curvature: Current curvature (1/m).

        Returns:
            Lookahead distance in meters.
        """
        cfg = self.cfg
        Ld = cfg.lookahead_L0 + cfg.lookahead_kv * speed - cfg.lookahead_kk * abs(curvature)
        return max(cfg.lookahead_Lmin, min(cfg.lookahead_Lmax, Ld))

    def _select_target(self, points, Ld):
        """Select the trajectory point closest to the desired lookahead distance.

        Args:
            points: List of TrajectoryPoint (ordered from vehicle forward).
            Ld: Desired lookahead distance (meters).

        Returns:
            TrajectoryPoint closest to Ld, or the farthest point if Ld exceeds range.
        """
        if not points:
            return TrajectoryPoint()

        # Compute cumulative arc length from the vehicle
        best = points[0]
        best_dist_error = float('inf')
        cumulative_dist = 0.0

        for i in range(1, len(points)):
            dx = points[i].x_m - points[i - 1].x_m
            dy = points[i].y_m - points[i - 1].y_m
            cumulative_dist += math.sqrt(dx * dx + dy * dy)

            dist_error = abs(cumulative_dist - Ld)
            if dist_error < best_dist_error:
                best_dist_error = dist_error
                best = points[i]

        return best
