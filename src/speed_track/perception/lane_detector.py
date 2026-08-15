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
        # Learned dynamic lane width in pixels (adaptive, not hardcoded)
        self.learned_lane_width_px = float(config.expected_lane_width_m * config.px_per_meter_x)
        self._initial_lane_width_px = self.learned_lane_width_px

    def detect(self, bev_mask, current_steer: float = 0.0):
        """Detect all three lane lines on a BEV binary mask.

        Args:
            bev_mask: Binary mask (uint8, 0 or 255) in BEV space.
            current_steer: Current steering command [-1.0, 1.0] from previous frame
                           (used for kinematic drift compensation).

        Returns:
            LaneDetectionResult containing left, center, right detections.
        """
        # Step 1: Histogram of bottom 75%
        histogram = np.sum(bev_mask[self.bev_h // 4:, :], axis=0)

        # Step 2: Find peaks in the overall ROI
        peaks = self._find_peaks(histogram)
        
        # Step 3: Analyze continuity (Dash vs Solid) for each candidate peak
        continuities = [self._check_continuity(bev_mask, p) for p in peaks]

        # Step 4: Kinematic Anchor Prediction
        # When steering right (>0), vehicle rotates right -> features in camera shift left (<0)
        drift_px = -current_steer * 80.0  # Adaptive shift proportional to steering
        anchor_pred = (self._last_center_x + drift_px) if self._last_center_x is not None else float(self.bev_w // 2)
        anchor_pred = max(50.0, min(float(self.bev_w - 50), anchor_pred))

        # Bottom-Anchor (Bottom 10%) - near car bumper
        bottom_hist = np.sum(bev_mask[int(self.bev_h * 0.9):, :], axis=0)
        bottom_peaks = self._find_peaks(bottom_hist)
        if bottom_peaks:
            # Anchor to bottom peak closest to predicted center
            true_center = min(bottom_peaks, key=lambda p: abs(p - anchor_pred))
            if abs(true_center - anchor_pred) < self.learned_lane_width_px * 0.4:
                anchor_pred = float(true_center)
                self._last_center_x = true_center

        # Step 5: Intelligent 3-Layer Peak Assignment (L/C/R)
        left_base, center_base, right_base = self._assign_peaks(
            peaks, continuities, anchor_pred, current_steer
        )

        # Step 6: Sliding window + RANSAC + confidence for each line
        left = self._detect_line(bev_mask, left_base, is_dashed=False) if left_base is not None else LineDetection()
        center = self._detect_line(bev_mask, center_base, is_dashed=True) if center_base is not None else LineDetection()
        right = self._detect_line(bev_mask, right_base, is_dashed=False) if right_base is not None else LineDetection()

        # Step 7: Update dynamic lane width learning if confident lines found
        self._update_dynamic_width(left, center, right, left_base, center_base, right_base)

        return LaneDetectionResult(
            left=left, center=center, right=right,
            histogram=histogram
        )

    def _check_continuity(self, bev_mask, base_x, margin=40, n_slices=8):
        """Estimate vertical continuity of a line to distinguish Solid vs Dashed.
        
        Solid line (boundary): pixels present in almost all vertical slices (~0.75 - 1.0).
        Dashed line (centerline): periodic gaps in slices (~0.30 - 0.65).
        
        Uses adaptive slice centroid tracking from bottom to top so that curved lines
        do not drift out of the window at the top of the image.
        
        Returns:
            continuity_ratio in [0.0, 1.0]
        """
        slice_h = self.bev_h // n_slices
        current_x = float(base_x)
        
        active_slices = 0
        # Iterate from bottom slice (closest to car) upward to top slice
        for i in reversed(range(n_slices)):
            y_start = i * slice_h
            y_end = (i + 1) * slice_h
            x_min = max(0, int(current_x - margin))
            x_max = min(self.bev_w, int(current_x + margin))
            
            sub_mask = bev_mask[y_start:y_end, x_min:x_max]
            nonzeros = sub_mask.nonzero()
            pixel_count = len(nonzeros[0])
            
            if pixel_count > 15:
                active_slices += 1
                # Update current_x to the centroid of the detected line pixels in this slice
                mean_local_x = np.mean(nonzeros[1])
                current_x = float(x_min + mean_local_x)
                
        return active_slices / float(n_slices)

    def _update_dynamic_width(self, left, center, right, l_base, c_base, r_base):
        """Adaptively update learned track width from confident multi-line detections."""
        min_conf = self.cfg.min_confidence_gate
        measured_w = None
        
        if l_base is not None and r_base is not None and left.confidence > min_conf and right.confidence > min_conf:
            w = float(r_base - l_base)
            if 0.6 * self._initial_lane_width_px <= w <= 1.4 * self._initial_lane_width_px:
                measured_w = w
        elif l_base is not None and c_base is not None and left.confidence > min_conf and center.confidence > min_conf:
            w = float(c_base - l_base) * 2.0
            if 0.6 * self._initial_lane_width_px <= w <= 1.4 * self._initial_lane_width_px:
                measured_w = w
        elif r_base is not None and c_base is not None and right.confidence > min_conf and center.confidence > min_conf:
            w = float(r_base - c_base) * 2.0
            if 0.6 * self._initial_lane_width_px <= w <= 1.4 * self._initial_lane_width_px:
                measured_w = w

        if measured_w is not None:
            # Smooth EMA adaptation of physical track width
            self.learned_lane_width_px = 0.90 * self.learned_lane_width_px + 0.10 * measured_w

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

    def _assign_peaks(self, peaks, continuities, anchor_pred, current_steer):
        """Assign detected peaks to left, center, and right lines using 3-layer fusion.
        
        Args:
            peaks: Sorted list of peak x-positions.
            continuities: Continuity ratio for each peak.
            anchor_pred: Kinematically predicted center x-position.
            current_steer: Vehicle steering [-1, 1].
            
        Returns:
            (left_base, center_base, right_base)
        """
        if not peaks:
            return None, None, None

        W = self.learned_lane_width_px
        half_W = W / 2.0
        n = len(peaks)

        # -------------------------------------------------------------
        # CASE 1: 3 or more peaks visible -> [L, C, R]
        # -------------------------------------------------------------
        if n >= 3:
            # Find the peak closest to predicted center as candidate C
            c_idx = min(range(n), key=lambda i: abs(peaks[i] - anchor_pred))
            
            # Ensure C is not on the extreme boundary if 3 peaks exist
            if c_idx == 0:
                c_idx = 1
            elif c_idx == n - 1:
                c_idx = n - 2
                
            center = peaks[c_idx]
            left = peaks[c_idx - 1]
            right = peaks[c_idx + 1]
            self._last_center_x = center
            return left, center, right

        # -------------------------------------------------------------
        # CASE 2: Exactly 2 peaks visible [p0, p1]
        # -------------------------------------------------------------
        if n == 2:
            p0, p1 = peaks[0], peaks[1]
            c0, c1 = continuities[0], continuities[1]
            dist = p1 - p0

            # Subcase 2A: Distance is close to FULL lane width (L + R, center missing/dashed gap)
            # Midpoint between half-width (0.50W) and full-width (1.00W) is ~0.65W
            if dist > W * 0.65:
                # p0 is Left boundary, p1 is Right boundary
                self._last_center_x = (p0 + p1) // 2
                return p0, None, p1

            # Subcase 2B: Distance is close to HALF lane width (L+C or C+R)
            # Differentiate using Texture (Dash vs Solid) and Kinematics
            is_p0_dashed = c0 < 0.68
            is_p1_dashed = c1 < 0.68
            
            if is_p0_dashed and not is_p1_dashed:
                # p0 is Center (dashed), p1 is Right (solid)
                self._last_center_x = p0
                return None, p0, p1
            elif not is_p0_dashed and is_p1_dashed:
                # p0 is Left (solid), p1 is Center (dashed)
                self._last_center_x = p1
                return p0, p1, None
            else:
                # Both look similar: use Kinematic steering context
                if current_steer < -0.20:
                    # Turning Left (car drifts Right) -> Centerline is on Left (p0), Right boundary is p1
                    self._last_center_x = p0
                    return None, p0, p1
                elif current_steer > 0.20:
                    # Turning Right (car drifts Left) -> Left boundary is p0, Centerline is on Right (p1)
                    self._last_center_x = p1
                    return p0, p1, None
                else:
                    # Fallback: which peak is closer to anchor_pred?
                    if abs(p0 - anchor_pred) < abs(p1 - anchor_pred):
                        self._last_center_x = p0
                        return None, p0, p1
                    else:
                        self._last_center_x = p1
                        return p0, p1, None

        # -------------------------------------------------------------
        # CASE 3: Only 1 peak visible [p0] (Extreme drift or sharp curve)
        # -------------------------------------------------------------
        p0 = peaks[0]
        c0 = continuities[0]
        is_solid = c0 >= 0.70

        # If it is dashed OR very close to predicted center -> It's Center (C)
        if not is_solid and abs(p0 - anchor_pred) < half_W * 0.6:
            self._last_center_x = p0
            return None, p0, None

        # If it is Solid, it CANNOT be the Centerline!
        # Use steering kinematics to determine whether it's Left or Right boundary:
        if current_steer < -0.25:
            # Turning Left (drifting Right) -> Center is out of FOV to the left, p0 is RIGHT boundary!
            return None, None, p0
        elif current_steer > 0.25:
            # Turning Right (drifting Left) -> Center is out of FOV to the right, p0 is LEFT boundary!
            return p0, None, None
        else:
            # Ambiguous single peak: check relative to screen center
            if p0 < self.bev_w * 0.4:
                return p0, None, None  # Left
            elif p0 > self.bev_w * 0.6:
                return None, None, p0  # Right
            else:
                # Exactly in middle and we are driving straight -> treat as Center
                self._last_center_x = p0
                return None, p0, None

    def _detect_line(self, bev_mask, base_x, is_dashed: bool = False):
        """Run sliding window + RANSAC for one lane line.

        Args:
            bev_mask: Binary mask in BEV space.
            base_x: Starting x-position at the bottom of the image.
            is_dashed: Whether this line candidate is a dashed centerline.

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
        confidence = self._compute_confidence(n_inliers, rmse, inlier_ratio, is_dashed=is_dashed)

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

    def _compute_confidence(self, n_inliers, rmse, inlier_ratio, is_dashed: bool = False):
        """Compute a confidence score in [0, 1] for a line detection.

        Components:
        1. Count: how many inlier pixels (more = better)
        2. RMSE: how tight the fit is (lower = better)
        3. Inlier ratio: what fraction survived RANSAC (higher = better)

        Args:
            n_inliers: Number of inlier pixels.
            rmse: Root mean square error of the polynomial fit.
            inlier_ratio: Fraction of total pixels that are inliers.
            is_dashed: Whether this line candidate is a dashed line.

        Returns:
            Confidence score in [0, 1].
        """
        cfg = self.cfg

        min_inliers = getattr(cfg, 'min_inlier_count_dashed', 60) if is_dashed else cfg.min_inlier_count
        exp_inliers = getattr(cfg, 'expected_inlier_count_dashed', 180) if is_dashed else cfg.expected_inlier_count

        # Count component
        if n_inliers < min_inliers:
            count_score = 0.0
        else:
            count_score = min(1.0, float(n_inliers) / float(exp_inliers))

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
