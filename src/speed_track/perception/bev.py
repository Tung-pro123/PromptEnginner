#!/usr/bin/env python3
"""
V3 Perception — Bird's Eye View / Inverse Perspective Mapping

Transforms the camera image (or a binary mask) into a top-down view
using a perspective warp. Also provides metric conversion utilities.

The BEV calibration is defined by four source points (trapezoid on the
road surface in the camera image) mapped to four destination points
(rectangle in BEV space).

Key difference from V2:
- BEV is treated purely as a coordinate transformation, not as a
  processing step that introduces artifacts.
- Source/destination points are configurable, not hard-coded.
- Provides px ↔ metric conversion that respects x/y aspect ratio.
"""

import cv2
import numpy as np


class BEVTransform:
    """Perspective transform between camera view and bird's eye view.

    Computes the warp matrix once at initialization and reuses it.
    """

    def __init__(self, config):
        """
        Args:
            config: V3Config instance with BEV parameters.
        """
        self.cfg = config
        self.W = config.image_width
        self.H = config.image_height

        self.src_pts = config.bev_src_pts
        self.dst_pts = config.bev_dst_pts

        # Compute perspective transform matrices (once)
        self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        self.M_inv = cv2.getPerspectiveTransform(self.dst_pts, self.src_pts)

        # Metric calibration
        self.px_per_m_x = config.px_per_meter_x
        self.px_per_m_y = config.px_per_meter_y

    def warp_to_bev(self, image):
        """Transform an image (or mask) to bird's eye view.

        Args:
            image: Input image or binary mask.

        Returns:
            Warped image in BEV perspective, same size as input.
        """
        return cv2.warpPerspective(image, self.M, (self.W, self.H))

    def warp_to_camera(self, bev_image):
        """Transform a BEV image back to camera perspective.

        Args:
            bev_image: Image in BEV space.

        Returns:
            Image in camera perspective.
        """
        return cv2.warpPerspective(bev_image, self.M_inv, (self.W, self.H))

    def px_to_metric(self, x_px, y_px):
        """Convert BEV pixel coordinates to metric coordinates.

        Origin: bottom-center of BEV image (vehicle position).
        x_metric: positive = right
        y_metric: positive = forward (up in BEV)

        Args:
            x_px: x coordinate in BEV pixels.
            y_px: y coordinate in BEV pixels.

        Returns:
            (x_m, y_m): coordinates in meters relative to vehicle.
        """
        x_center_px = self.W / 2.0
        y_bottom_px = self.H

        x_m = (x_px - x_center_px) / self.px_per_m_x
        y_m = (y_bottom_px - y_px) / self.px_per_m_y

        return x_m, y_m

    def metric_to_px(self, x_m, y_m):
        """Convert metric coordinates back to BEV pixel coordinates.

        Args:
            x_m: x in meters (positive = right).
            y_m: y in meters (positive = forward).

        Returns:
            (x_px, y_px): coordinates in BEV pixels.
        """
        x_center_px = self.W / 2.0
        y_bottom_px = self.H

        x_px = x_m * self.px_per_m_x + x_center_px
        y_px = y_bottom_px - y_m * self.px_per_m_y

        return x_px, y_px

    def curvature_px_to_metric(self, poly_px):
        """Convert a polynomial fitted in pixel space to metric curvature.

        The polynomial is: x_px = a * y_px^2 + b * y_px + c
        We need curvature in metric: kappa = x_m'' / (1 + x_m'^2)^(3/2)

        The conversion accounts for different x and y scales.

        Args:
            poly_px: Polynomial coefficients [a, b, c] in pixel space.

        Returns:
            Function that computes metric curvature at a given y_px.
        """
        a, b, c = poly_px[0], poly_px[1], poly_px[2]

        # Scale factors
        sx = 1.0 / self.px_per_m_x   # px → m for x
        sy = 1.0 / self.px_per_m_y   # px → m for y

        # In metric space:
        # x_m(y_m) = a * (y_m/sy)^2 * sx + b * (y_m/sy) * sx + c * sx
        # But it's cleaner to compute derivatives in pixel space and scale:
        # dx_m/dy_m = (dx_px/dy_px) * (sx/sy)
        # d²x_m/dy_m² = (d²x_px/dy_px²) * (sx/sy²)

        def kappa_at_y_px(y_px):
            dx_dy_px = 2.0 * a * y_px + b
            d2x_dy2_px = 2.0 * a

            dx_dy_m = dx_dy_px * (sx / sy)
            d2x_dy2_m = d2x_dy2_px * (sx / (sy * sy))

            kappa = d2x_dy2_m / ((1.0 + dx_dy_m ** 2) ** 1.5)
            return kappa

        return kappa_at_y_px
