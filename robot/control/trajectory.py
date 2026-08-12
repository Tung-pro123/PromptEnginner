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

    def generate(self, lane_state, current_speed=0.0):
        """Generate a look-ahead trajectory from the current lane state.

        Args:
            lane_state: LaneState with centerline polynomial.
            current_speed: Current vehicle speed (m/s). Used for adaptive Ld.

        Returns:
            TrajectoryResult with points, target, and derived quantities.
        """
        result = TrajectoryResult()

        if lane_state.centerline_poly is None:
            return result

        poly = lane_state.centerline_poly
        bev_h = self.cfg.image_height

        # Sample points along the centerline in BEV pixel space
        # From vehicle position (bottom) to far ahead (top)
        n_pts = self.cfg.n_lookahead_points
        y_values = np.linspace(bev_h, bev_h * 0.1, n_pts)

        points = []
        for y_px in y_values:
            x_px = np.polyval(poly, y_px)
            x_m, y_m = self.bev.px_to_metric(x_px, y_px)
            points.append(TrajectoryPoint(x_m=x_m, y_m=y_m, x_px=x_px, y_px=y_px))

        result.points = points

        # Compute curvature at vehicle position (in metric coordinates)
        kappa_func = self.bev.curvature_px_to_metric(poly)
        result.curvature = kappa_func(float(bev_h))

        # Compute heading error at vehicle position
        a, b = poly[0], poly[1]
        dx_dy_px = 2.0 * a * float(bev_h) + b
        sx = 1.0 / self.cfg.px_per_meter_x
        sy = 1.0 / self.cfg.px_per_meter_y
        dx_dy_m = dx_dy_px * (sx / sy)
        result.heading_error = math.atan(dx_dy_m)

        # Lateral error at vehicle position
        x_center_px = self.cfg.image_width / 2.0
        x_line_px = np.polyval(poly, float(bev_h))
        result.lateral_error_m = (x_line_px - x_center_px) / self.cfg.px_per_meter_x

        # Adaptive lookahead distance
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
