#!/usr/bin/env python3
"""
V3 Estimation — Lane Geometry Validation & Center Reconstruction

Validates that detected lane lines form a geometrically consistent lane:
- Lane width within expected range
- Left/right symmetry reasonable
- Center line consistent with boundaries

Reconstructs the best center trajectory from whatever lines are available,
using a priority system that handles partial line loss gracefully.

This module does NOT do temporal filtering — that's lane_state.py's job.
This module produces a single-frame geometric observation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class GeometryObservation:
    """Single-frame lane geometry observation."""
    centerline_poly: Optional[np.ndarray] = None  # [a, b, c] polynomial
    lane_width_px: float = 0.0                     # pixels in BEV
    lane_width_m: float = 0.0                      # meters
    left_poly: Optional[np.ndarray] = None
    center_poly: Optional[np.ndarray] = None
    right_poly: Optional[np.ndarray] = None
    left_conf: float = 0.0
    center_conf: float = 0.0
    right_conf: float = 0.0
    overall_confidence: float = 0.0
    valid: bool = False
    method: str = 'none'   # How centerline was derived


class LaneGeometry:
    """Validates lane geometry and reconstructs centerline.

    Priority for center reconstruction:
    1. L + R + C (all three) → weighted fusion
    2. L + R (boundaries only, center missing) → C = (L + R) / 2
    3. L + C or R + C → use C, validate with boundary
    4. L only or R only → reconstruct using previous width
    5. None → return invalid (let temporal filter predict)
    """

    def __init__(self, config, bev_transform):
        """
        Args:
            config: V3Config instance.
            bev_transform: BEVTransform instance for metric conversion.
        """
        self.cfg = config
        self.bev = bev_transform
        self.bev_h = config.image_height

        # Previous lane width for reconstruction when lines are partially lost
        self._prev_width_px = config.expected_lane_width_m * config.px_per_meter_x

    def process(self, detection_result, prev_state=None):
        """Validate detection and reconstruct centerline.

        Args:
            detection_result: LaneDetectionResult from MultiLaneDetector.
            prev_state: Previous LaneState (for single-boundary reconstruction).

        Returns:
            GeometryObservation with centerline and validation info.
        """
        left = detection_result.left
        center = detection_result.center
        right = detection_result.right

        # Determine which lines are "usable" (confidence above minimum gate)
        min_conf = self.cfg.min_confidence_gate
        has_L = left.detected and left.confidence > min_conf
        has_C = center.detected and center.confidence > min_conf
        has_R = right.detected and right.confidence > min_conf

        obs = GeometryObservation(
            left_poly=left.poly if has_L else None,
            center_poly=center.poly if has_C else None,
            right_poly=right.poly if has_R else None,
            left_conf=left.confidence if has_L else 0.0,
            center_conf=center.confidence if has_C else 0.0,
            right_conf=right.confidence if has_R else 0.0,
        )

        # ---- Centerline reconstruction ----

        # ƯU TIÊN SỐ 1: Vạch giữa nét đứt (has_C) — Ground Truth của đường đua
        if has_C:
            obs.centerline_poly = center.poly.copy()
            if has_L and has_R:
                obs.method = 'L+C+R_fused'
                width_px = self._compute_width(left.poly, right.poly)
                self._prev_width_px = width_px
                obs.overall_confidence = max(0.7, (left.confidence + right.confidence + center.confidence) / 3.0)
            elif has_L:
                obs.method = 'L+C'
                obs.overall_confidence = max(0.6, (left.confidence * 0.4 + center.confidence * 0.6))
            elif has_R:
                obs.method = 'R+C'
                obs.overall_confidence = max(0.6, (right.confidence * 0.4 + center.confidence * 0.6))
            else:
                obs.method = 'CENTER_LOCKED'
                obs.overall_confidence = center.confidence
            
            obs.lane_width_px = self._prev_width_px
            obs.lane_width_m = self._prev_width_px / self.cfg.px_per_meter_x
            obs.valid = True

        elif has_L and has_R:
            # ƯU TIÊN SỐ 2: 2 vạch biên (không có vạch giữa) -> Lấy trung điểm
            center_from_boundaries = self._average_poly(left.poly, right.poly)
            width_px = self._compute_width(left.poly, right.poly)
            width_m = width_px / self.cfg.px_per_meter_x

            if self._width_valid(width_m):
                obs.centerline_poly = center_from_boundaries
                obs.method = 'L+R_midpoint'
                obs.lane_width_px = width_px
                obs.lane_width_m = width_m
                obs.overall_confidence = min(1.0, (left.confidence + right.confidence) / 2.0)
                obs.valid = True
                self._prev_width_px = width_px
            else:
                obs.valid = False
                obs.method = 'rejected_width'

        elif has_L:
            # ƯU TIÊN SỐ 3a: Chỉ có vạch biên TRÁI -> Dịch sang PHẢI 1 khoảng cố định chuẩn
            offset_m = getattr(self.cfg, 'single_line_offset_m', 0.225)
            offset_px = offset_m * self.cfg.px_per_meter_x
            obs.centerline_poly = left.poly.copy()
            obs.centerline_poly[-1] += offset_px  # Shift right
            obs.lane_width_px = self._prev_width_px
            obs.lane_width_m = self._prev_width_px / self.cfg.px_per_meter_x
            obs.overall_confidence = left.confidence * 0.5
            obs.valid = True
            obs.method = 'L_only'

        elif has_R:
            # ƯU TIÊN SỐ 3b: Chỉ có vạch biên PHẢI -> Dịch sang TRÁI 1 khoảng cố định chuẩn
            offset_m = getattr(self.cfg, 'single_line_offset_m', 0.225)
            offset_px = offset_m * self.cfg.px_per_meter_x
            obs.centerline_poly = right.poly.copy()
            obs.centerline_poly[-1] -= offset_px  # Shift left
            obs.lane_width_px = self._prev_width_px
            obs.lane_width_m = self._prev_width_px / self.cfg.px_per_meter_x
            obs.overall_confidence = right.confidence * 0.5
            obs.valid = True
            obs.method = 'R_only'

        else:
            # Không thấy bất kỳ vạch nào
            obs.valid = False
            obs.method = 'none'

        return obs

    def _average_poly(self, poly1, poly2):
        """Average two polynomials coefficient-wise.

        For L and R boundaries:  C(y) = (L(y) + R(y)) / 2

        Args:
            poly1, poly2: Polynomial coefficients [a, b, c].

        Returns:
            Averaged polynomial coefficients.
        """
        return (poly1 + poly2) / 2.0

    def _compute_width(self, left_poly, right_poly):
        """Compute average lane width (in pixels) between L and R.

        Evaluates at multiple y-positions and takes the mean.

        Args:
            left_poly, right_poly: Polynomial coefficients [a, b, c].

        Returns:
            Average lane width in BEV pixels.
        """
        y_eval = np.linspace(self.bev_h * 0.3, self.bev_h * 0.9, 10)
        x_left = np.polyval(left_poly, y_eval)
        x_right = np.polyval(right_poly, y_eval)
        widths = x_right - x_left
        # Use median to be robust against single-point anomalies
        return float(np.median(widths))

    def _compute_half_width(self, inner_poly, outer_poly):
        """Compute average half-width between two adjacent lines.

        Args:
            inner_poly: Polynomial closer to center.
            outer_poly: Polynomial farther from center.

        Returns:
            Average half-width in BEV pixels.
        """
        y_eval = np.linspace(self.bev_h * 0.3, self.bev_h * 0.9, 10)
        x_inner = np.polyval(inner_poly, y_eval)
        x_outer = np.polyval(outer_poly, y_eval)
        half_widths = np.abs(x_outer - x_inner)
        return float(np.median(half_widths))

    def _width_valid(self, width_m):
        """Check if the measured lane width is physically reasonable.

        Args:
            width_m: Measured width in meters.

        Returns:
            True if within tolerance of expected width.
        """
        expected = self.cfg.expected_lane_width_m
        tolerance = self.cfg.lane_width_tolerance
        return abs(width_m - expected) <= expected * tolerance
