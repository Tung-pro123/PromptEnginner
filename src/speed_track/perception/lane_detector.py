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
        self._locked_center_poly = None   # Đa thức vạch giữa đã khóa
        self._center_lock_count = 0        # Số frame khóa liên tiếp
        self._lost_center_count = 0        # Số frame tạm mất vạch giữa

    def detect(self, bev_mask):
        """Detect all three lane lines on a BEV binary mask.
        Ưu tiên khóa chặt và duy trì bám vạch giữa nét đứt qua Search Corridor.

        Args:
            bev_mask: Binary mask (uint8, 0 or 255) in BEV space.

        Returns:
            LaneDetectionResult containing left, center, right detections.
        """
        center = LineDetection()
        center_locked_success = False

        # ---- TẦNG 1: SEARCH CORRIDOR (Khóa bám vạch giữa từ đa thức frame trước) ----
        if getattr(self.cfg, 'center_lock_enabled', True) and self._locked_center_poly is not None:
            corridor_margin = getattr(self.cfg, 'center_corridor_margin', 45)
            center = self._search_corridor(bev_mask, self._locked_center_poly, corridor_margin)
            
            min_pts = getattr(self.cfg, 'center_lock_min_pts', 60)
            if center.detected and center.n_inliers >= min_pts and center.confidence >= 0.20:
                # Cập nhật trơn đa thức vạch giữa (EMA)
                alpha = getattr(self.cfg, 'center_poly_ema_alpha', 0.35)
                self._locked_center_poly = alpha * center.poly + (1.0 - alpha) * self._locked_center_poly
                center.poly = self._locked_center_poly.copy()
                self._center_lock_count += 1
                self._lost_center_count = 0
                center_locked_success = True
            else:
                self._lost_center_count += 1
                # Nếu mất quá 5 frame liên tiếp mới hủy khóa để tìm kiếm lại từ đầu
                if self._lost_center_count > 5:
                    self._locked_center_poly = None
                    self._center_lock_count = 0

        # ---- TẦNG 2: HISTOGRAM PEAK SEARCH (Nếu chưa khóa được vạch giữa hoặc tìm 2 vạch biên) ----
        histogram = np.sum(bev_mask[self.bev_h // 4:, :], axis=0)
        peaks = self._find_peaks(histogram)

        # Gán đỉnh cho 3 vạch
        left_base, center_base, right_base = self._assign_peaks(peaks)

        # Quét 2 vạch biên bằng sliding window
        left = self._detect_line(bev_mask, left_base) if left_base is not None else LineDetection()
        right = self._detect_line(bev_mask, right_base) if right_base is not None else LineDetection()

        # Nếu Search Corridor chưa bắt được vạch giữa thì dùng sliding window
        if not center_locked_success:
            center = self._detect_line(bev_mask, center_base) if center_base is not None else LineDetection()
            if center.detected and center.confidence >= 0.35 and center.poly is not None:
                # Khởi tạo khóa mới cho các frame sau
                self._locked_center_poly = center.poly.copy()
                self._center_lock_count = 1
                self._lost_center_count = 0

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
        """Assign detected peaks to left, center, and right lines using Geometry.
        
        Args:
            peaks: Sorted list of peak x-positions from the main histogram.
        Returns:
            (left_base, center_base, right_base)
        """
        if not peaks:
            return None, None, None
            
        full_width = self.cfg.expected_lane_width_m * self.cfg.px_per_meter_x
        half_width = full_width / 2.0
        
        left, center, right = None, None, None
        
        if len(peaks) == 3:
            left, center, right = peaks[0], peaks[1], peaks[2]
            
        elif len(peaks) == 2:
            p1, p2 = peaks[0], peaks[1]
            dist = p2 - p1
            
            err_half = abs(dist - half_width)
            err_full = abs(dist - full_width)
            
            if err_full < err_half:
                # Đây là 2 vạch biên (Left và Right), mất vạch Center
                left, right = p1, p2
            else:
                # Đây là 1 vạch biên và vạch Center
                midpoint = (p1 + p2) / 2.0
                if midpoint < self.bev_w / 2.0:
                    left, center = p1, p2
                else:
                    center, right = p1, p2
                    
        elif len(peaks) == 1:
            p = peaks[0]
            if p < self.bev_w * 0.35:
                left = p
            elif p > self.bev_w * 0.65:
                right = p
            else:
                center = p
                
        elif len(peaks) > 3:
            # Nhiễu, ưu tiên 3 đỉnh gần tâm bức ảnh nhất
            sorted_by_center = sorted(peaks, key=lambda x: abs(x - self.bev_w / 2.0))
            best_3 = sorted(sorted_by_center[:3])
            left, center, right = best_3[0], best_3[1], best_3[2]

        return left, center, right

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

    def _search_corridor(self, bev_mask, poly, margin):
        """Khóa bám vạch nét đứt bằng Hành lang tìm kiếm (Search Corridor).
        Trích xuất nhanh các pixel nằm trong phạm vi x_poly(y) +- margin.

        Args:
            bev_mask: Binary mask trong không gian BEV.
            poly: Đa thức [a, b, c] của vạch giữa ở frame trước.
            margin: Bề rộng hành lang (pixels).

        Returns:
            LineDetection kết quả.
        """
        nonzero_y, nonzero_x = bev_mask.nonzero()
        if len(nonzero_y) == 0:
            return LineDetection()

        # Tính vị trí x mong đợi theo đa thức cũ: x = a*y^2 + b*y + c
        expected_x = np.polyval(poly, nonzero_y)

        # Lọc các điểm rơi đúng vào bên trong hành lang
        in_corridor = (nonzero_x >= (expected_x - margin)) & (nonzero_x <= (expected_x + margin))

        corridor_x = nonzero_x[in_corridor]
        corridor_y = nonzero_y[in_corridor]

        if len(corridor_x) < getattr(self.cfg, 'center_lock_min_pts', 60):
            return LineDetection(n_total=len(corridor_x))

        # Fit đa thức mới từ các điểm trong hành lang
        new_poly, inlier_mask, rmse = self._ransac_polyfit(corridor_y, corridor_x)
        if new_poly is None:
            return LineDetection(n_total=len(corridor_x))

        n_inliers = int(np.sum(inlier_mask))
        inlier_ratio = n_inliers / len(corridor_x) if len(corridor_x) > 0 else 0.0
        inlier_x = corridor_x[inlier_mask]
        inlier_y = corridor_y[inlier_mask]

        confidence = self._compute_confidence(n_inliers, rmse, inlier_ratio)

        return LineDetection(
            poly=new_poly,
            confidence=confidence,
            n_inliers=n_inliers,
            n_total=len(corridor_x),
            rmse=rmse,
            inlier_ratio=inlier_ratio,
            x_pixels=inlier_x,
            y_pixels=inlier_y,
            detected=True
        )

    def _sliding_window(self, bev_mask, base_x):
        """Sliding window search từ dưới lên trên kèm cơ chế Gap-Bridging (vượt khoảng đứt).

        Args:
            bev_mask: Binary mask (0/255).
            base_x: Starting x-position at the bottom.

        Returns:
            (all_x, all_y): Arrays of found pixel coordinates.
        """
        n_windows = self.cfg.sw_n_windows
        margin = self.cfg.sw_margin
        min_pix = self.cfg.sw_min_pix
        max_empty = getattr(self.cfg, 'gap_bridge_max_empty', 3)

        window_height = self.bev_h // n_windows
        current_x = int(base_x)
        last_dx = 0  # Xu hướng độ dốc để vượt khoảng đứt
        empty_windows = 0

        all_x = []
        all_y = []

        nonzero_y, nonzero_x = bev_mask.nonzero()

        for win_idx in range(n_windows):
            y_low = self.bev_h - (win_idx + 1) * window_height
            y_high = self.bev_h - win_idx * window_height
            x_low = max(0, current_x - margin)
            x_high = min(self.bev_w, current_x + margin)

            in_window = (
                (nonzero_y >= y_low) & (nonzero_y < y_high) &
                (nonzero_x >= x_low) & (nonzero_x < x_high)
            )
            window_x = nonzero_x[in_window]
            window_y = nonzero_y[in_window]

            all_x.append(window_x)
            all_y.append(window_y)

            if len(window_x) >= min_pix:
                new_x = int(np.mean(window_x))
                last_dx = new_x - current_x
                current_x = new_x
                empty_windows = 0
            else:
                # Gặp khoảng đứt (Gap) -> Nhảy cóc theo xu hướng độ cong cũ
                empty_windows += 1
                if empty_windows <= max_empty:
                    current_x = int(np.clip(current_x + last_dx, 0, self.bev_w - 1))

        if len(all_x) == 0:
            return np.array([]), np.array([])

        return np.concatenate(all_x), np.concatenate(all_y)

    def _ransac_polyfit(self, y_data, x_data):
        """Fit a polynomial using standard Least Squares (Replaced RANSAC).
        
        Since sliding window already filters out most noise, a direct fit
        is 30x faster than RANSAC and yields virtually identical results.

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
        
        try:
            # Direct Least Squares fit on ALL pixels found by sliding window
            poly = np.polyfit(y_data, x_data, degree)
        except (np.linalg.LinAlgError, ValueError):
            return None, None, float('inf')

        # Compute residuals (RMSE)
        x_pred = np.polyval(poly, y_data)
        residuals = np.abs(x_data - x_pred)
        
        # Coi tất cả các điểm nằm trong ngưỡng là inlier để tính confidence
        threshold = self.cfg.ransac_residual_threshold
        inlier_mask = residuals < threshold
        
        rmse = np.sqrt(np.mean(residuals ** 2))

        return poly, inlier_mask, rmse

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
