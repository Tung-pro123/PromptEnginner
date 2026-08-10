#!/usr/bin/env python3
"""
V3 Perception — Multi-Lane Detector (Histogram + Sliding Window + RANSAC)

Replaces V2's single largest-contour approach with proper per-line detection.
Detects LEFT, CENTER, and RIGHT lane markings separately using:
  1. Histogram peak initialization
  2. Sliding window search per line
  3. RANSAC polynomial fitting per line
  4. Per-line confidence scoring

Key differences from V2:
- V2 merged all lines into one contour → produced a centroid of mixed lines
- V3 identifies and tracks each line independently
- V3 uses RANSAC to reject outliers within each line's pixel set
- V3 produces a confidence score per line, enabling the state estimator
  to weight and fuse lines appropriately
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class LineDetection:
    """Detection result for a single lane line."""
    poly: Optional[np.ndarray] = None   # Polynomial coefficients [a, b, c]
    confidence: float = 0.0             # [0, 1]
    n_inliers: int = 0                  # Number of RANSAC inliers
    n_total: int = 0                    # Total pixels found by sliding window
    rmse: float = float('inf')          # Root mean square error of fit
    inlier_ratio: float = 0.0           # Fraction of inliers
    x_pixels: Optional[np.ndarray] = None  # Inlier x coordinates
    y_pixels: Optional[np.ndarray] = None  # Inlier y coordinates
    detected: bool = False              # Whether this line was found at all


@dataclass
class LaneDetectionResult:
    """Complete detection result for all three lines."""
    left: LineDetection
    center: LineDetection
    right: LineDetection
    histogram: Optional[np.ndarray] = None  # For debug visualization


class MultiLaneDetector:
    """Detects three lane lines (left, center, right) on a BEV binary mask.

    Algorithm:
    1. Compute column-sum histogram of the bottom half of the BEV mask.
    2. Find peaks in the histogram → candidate base x-positions for lines.
    3. Assign peaks to left/center/right based on position.
    4. Run sliding window search from each base upward, collecting pixels.
    5. Fit a 2nd-degree polynomial to each line's pixels using RANSAC.
    6. Compute confidence based on pixel count, RMSE, and inlier ratio.
    """

    def __init__(self, config):
        """
        Args:
            config: V3Config instance with sliding window and RANSAC params.
        """
        self.cfg = config
        self.bev_h = config.image_height
        self.bev_w = config.image_width
        self._last_center_x = None

    def detect(self, bev_mask):
        """Detect all three lane lines on a BEV binary mask.

        Args:
            bev_mask: Binary mask (uint8, 0 or 255) in BEV space.

        Returns:
            LaneDetectionResult containing left, center, right detections.
        """
        # Step 1: Histogram of bottom 75% (để bắt được các vạch biên bị cắt sớm)
        histogram = np.sum(bev_mask[self.bev_h // 4:, :], axis=0)

        # Step 2: Find peaks
        peaks = self._find_peaks(histogram)

        # Step 3: Assign peaks to L/C/R
        left_base, center_base, right_base = self._assign_peaks(peaks)

        # Step 4-6: Sliding window + RANSAC + confidence for each line
        left = self._detect_line(bev_mask, left_base) if left_base is not None else LineDetection()
        center = self._detect_line(bev_mask, center_base) if center_base is not None else LineDetection()
        right = self._detect_line(bev_mask, right_base) if right_base is not None else LineDetection()

        # Update last center x for temporal consistency in peak assignment
        if center_base is not None:
            # Slowly decay the last center x towards the middle to prevent permanent drift if errors occur
            if self._last_center_x is None:
                self._last_center_x = center_base
            else:
                self._last_center_x = int(self._last_center_x * 0.7 + center_base * 0.3)
        else:
            # If no center line detected, decay towards the actual middle of the screen
            if self._last_center_x is not None:
                self._last_center_x = int(self._last_center_x * 0.95 + (self.bev_w // 2) * 0.05)

        return LaneDetectionResult(
            left=left, center=center, right=right,
            histogram=histogram
        )

    def _find_peaks(self, histogram):
        """Find peaks in the column-sum histogram.

        Uses a simple local-maximum search with minimum distance
        and minimum height constraints.

        Args:
            histogram: 1D array of column sums.

        Returns:
            List of peak x-positions, sorted by x.
        """
        min_height = self.cfg.sw_min_peak_height
        min_distance = self.cfg.sw_min_peak_distance
        h = histogram

        peaks = []
        for i in range(1, len(h) - 1):
            if h[i] < min_height:
                continue
            # Local maximum check
            if h[i] > h[i - 1] and h[i] >= h[i + 1]:
                # Check minimum distance from existing peaks
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
                else:
                    # If too close, keep the taller one
                    if h[i] > h[peaks[-1]]:
                        peaks[-1] = i

        return sorted(peaks)

    def _assign_peaks(self, peaks):
        """Assign detected peaks to left, center, and right lines.

        Heuristic:
        - 3 peaks → left, center, right (sorted by x)
        - 2 peaks → left and right (center dashed line often missing)
        - 1 peak  → assign based on position in image
        - 0 peaks → all None

        Args:
            peaks: Sorted list of peak x-positions.

        Returns:
            (left_base, center_base, right_base) — each int or None.
        """
        # Sử dụng vị trí tâm của frame trước (nếu có) thay vì cố định ở giữa màn hình
        mid = self._last_center_x if self._last_center_x is not None else self.bev_w // 2

        if len(peaks) >= 3:
            # Take the 3 most prominent (sorted by x)
            # If more than 3, pick the 3 most spread out
            if len(peaks) == 3:
                return peaks[0], peaks[1], peaks[2]
            else:
                # Take leftmost, rightmost, and the one closest to middle
                left = peaks[0]
                right = peaks[-1]
                # Find the peak closest to the midpoint of left and right
                mid_expected = (left + right) // 2
                center_candidates = [p for p in peaks[1:-1]]
                if center_candidates:
                    center = min(center_candidates, key=lambda p: abs(p - mid_expected))
                else:
                    center = None
                return left, center, right

        elif len(peaks) == 2:
            # Two peaks detected. Are they (Left + Right) or (Center + Boundary)?
            # We can tell by the distance between them!
            p0, p1 = peaks[0], peaks[1]
            dist = p1 - p0
            
            expected_full_w = self.cfg.expected_lane_width_m * self.cfg.px_per_meter_x
            expected_half_w = expected_full_w / 2.0
            
            # If the distance is closer to the full width, they are L + R
            if abs(dist - expected_full_w) < abs(dist - expected_half_w):
                return p0, None, p1
            else:
                # The distance is closer to half width, so it's Center + Boundary.
                # The one closest to the middle of the camera is the Center line.
                if abs(p0 - mid) < abs(p1 - mid):
                    return None, p0, p1  # p0 is Center, p1 is Right
                else:
                    return p0, p1, None  # p0 is Left, p1 is Center

        elif len(peaks) == 1:
            # Only one line visible
            p = peaks[0]
            # If it's near the middle of the screen, it's highly likely the center line
            if abs(p - mid) < 100:
                return None, p, None
            elif p < mid:
                return p, None, None
            else:
                return None, None, p

        else:
            return None, None, None

    def _detect_line(self, bev_mask, base_x):
        """Run sliding window + RANSAC for one lane line.

        Args:
            bev_mask: Binary mask in BEV space.
            base_x: Starting x-position at the bottom of the image.

        Returns:
            LineDetection with polynomial, confidence, and pixel data.
        """
        if base_x is None:
            return LineDetection()

        # Sliding window search
        all_x, all_y = self._sliding_window(bev_mask, base_x)

        if len(all_x) < self.cfg.ransac_min_samples:
            return LineDetection(n_total=len(all_x))

        # RANSAC polynomial fit
        poly, inlier_mask, rmse = self._ransac_polyfit(all_y, all_x)

        if poly is None:
            return LineDetection(n_total=len(all_x))

        n_inliers = int(np.sum(inlier_mask))
        inlier_ratio = n_inliers / len(all_x) if len(all_x) > 0 else 0.0
        inlier_x = all_x[inlier_mask]
        inlier_y = all_y[inlier_mask]

        # Confidence scoring
        confidence = self._compute_confidence(n_inliers, rmse, inlier_ratio)

        return LineDetection(
            poly=poly,
            confidence=confidence,
            n_inliers=n_inliers,
            n_total=len(all_x),
            rmse=rmse,
            inlier_ratio=inlier_ratio,
            x_pixels=inlier_x,
            y_pixels=inlier_y,
            detected=True
        )

    def _sliding_window(self, bev_mask, base_x):
        """Sliding window search from bottom to top of the BEV mask.

        Args:
            bev_mask: Binary mask (0/255).
            base_x: Starting x-position at the bottom.

        Returns:
            (all_x, all_y): Arrays of found pixel coordinates.
        """
        n_windows = self.cfg.sw_n_windows
        margin = self.cfg.sw_margin
        min_pix = self.cfg.sw_min_pix

        window_height = self.bev_h // n_windows
        current_x = int(base_x)

        all_x = []
        all_y = []

        # Find all nonzero pixel positions in the mask
        nonzero_y, nonzero_x = bev_mask.nonzero()

        for win_idx in range(n_windows):
            # Window boundaries
            y_low = self.bev_h - (win_idx + 1) * window_height
            y_high = self.bev_h - win_idx * window_height
            x_low = max(0, current_x - margin)
            x_high = min(self.bev_w, current_x + margin)

            # Find pixels within the window
            in_window = (
                (nonzero_y >= y_low) & (nonzero_y < y_high) &
                (nonzero_x >= x_low) & (nonzero_x < x_high)
            )
            window_x = nonzero_x[in_window]
            window_y = nonzero_y[in_window]

            all_x.append(window_x)
            all_y.append(window_y)

            # Recenter window if enough pixels found
            if len(window_x) >= min_pix:
                current_x = int(np.mean(window_x))

        if len(all_x) == 0:
            return np.array([]), np.array([])

        return np.concatenate(all_x), np.concatenate(all_y)

    def _ransac_polyfit(self, y_data, x_data):
        """Fit a polynomial using RANSAC to reject outliers.

        Fits x = f(y) = a*y^2 + b*y + c (because lanes run vertically).

        Args:
            y_data: y-coordinates of pixels.
            x_data: x-coordinates of pixels.

        Returns:
            (poly_coeffs, inlier_mask, rmse) or (None, None, inf).
        """
        n = len(y_data)
        if n < self.cfg.ransac_min_samples:
            return None, None, float('inf')

        degree = self.cfg.poly_degree
        threshold = self.cfg.ransac_residual_threshold
        max_trials = self.cfg.ransac_max_trials
        min_samples = max(degree + 1, self.cfg.ransac_min_samples)

        best_inlier_count = 0
        best_poly = None
        best_inlier_mask = None

        for _ in range(max_trials):
            # Random sample
            indices = np.random.choice(n, size=min(min_samples, n), replace=False)
            sample_y = y_data[indices]
            sample_x = x_data[indices]

            # Fit polynomial on sample
            try:
                poly = np.polyfit(sample_y, sample_x, degree)
            except (np.linalg.LinAlgError, ValueError):
                continue

            # Compute residuals for all points
            x_pred = np.polyval(poly, y_data)
            residuals = np.abs(x_data - x_pred)

            # Count inliers
            inlier_mask = residuals < threshold
            n_inliers = np.sum(inlier_mask)

            if n_inliers > best_inlier_count:
                best_inlier_count = n_inliers
                best_inlier_mask = inlier_mask
                best_poly = poly

        if best_poly is None or best_inlier_count < self.cfg.ransac_min_samples:
            return None, None, float('inf')

        # Refit on all inliers for better coefficients
        inlier_y = y_data[best_inlier_mask]
        inlier_x = x_data[best_inlier_mask]
        try:
            refined_poly = np.polyfit(inlier_y, inlier_x, degree)
        except (np.linalg.LinAlgError, ValueError):
            refined_poly = best_poly

        # Compute RMSE on inliers
        x_pred_inliers = np.polyval(refined_poly, inlier_y)
        rmse = np.sqrt(np.mean((inlier_x - x_pred_inliers) ** 2))

        return refined_poly, best_inlier_mask, rmse

    def _compute_confidence(self, n_inliers, rmse, inlier_ratio):
        """Compute a confidence score in [0, 1] for a line detection.

        Components:
        1. Count: how many inlier pixels (more = better)
        2. RMSE: how tight the fit is (lower = better)
        3. Inlier ratio: what fraction survived RANSAC (higher = better)

        Args:
            n_inliers: Number of inlier pixels.
            rmse: Root mean square error of the polynomial fit.
            inlier_ratio: Fraction of total pixels that are inliers.

        Returns:
            Confidence score in [0, 1].
        """
        cfg = self.cfg

        # Count component
        if n_inliers < cfg.min_inlier_count:
            count_score = 0.0
        else:
            count_score = min(1.0, n_inliers / cfg.expected_inlier_count)

        # RMSE component
        rmse_score = max(0.0, 1.0 - rmse / cfg.max_fit_rmse)

        # Inlier ratio component
        ratio_score = inlier_ratio

        confidence = (
            cfg.conf_weight_count * count_score +
            cfg.conf_weight_rmse * rmse_score +
            cfg.conf_weight_inlier_ratio * ratio_score
        )

        return min(1.0, max(0.0, confidence))
