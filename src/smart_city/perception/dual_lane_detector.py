#!/usr/bin/env python3
"""
Dual Lane Detector (Bám 2 Đường Biên)
Phát hiện line trái và phải, sau đó tính toán centerline ảo.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DualLaneResult:
    """Kết quả phát hiện lane."""
    center_x: Optional[int] = None
    left_x: Optional[int] = None
    right_x: Optional[int] = None
    mode: str = 'LOST'  # 'BOTH', 'LEFT_ONLY', 'RIGHT_ONLY', 'LOST'


class DualLaneDetector:
    def __init__(self, config):
        self.cfg = config

    def detect(self, image, roi_y, roi_h) -> DualLaneResult:
        """
        Phát hiện 2 line biên và tính centerline ảo.
        """
        if image is None:
            return DualLaneResult()

        # 1. Cắt ROI và tạo mask
        roi = image[roi_y:roi_y + roi_h, :]
        mask = self._create_line_mask(roi)
        mask = self._apply_focus_mask(mask)

        # 2. Chia mask thành nửa trái và phải
        w = mask.shape[1]
        mid_x = w // 2
        left_half = mask[:, :mid_x]
        right_half = mask[:, mid_x:]

        # 3. Tìm tâm của từng bên
        left_x = self._find_largest_centroid_x(left_half)
        right_x = self._find_largest_centroid_x(right_half)
        
        # Bù lại offset cho right_x vì nó tính trên ảnh nửa phải
        if right_x is not None:
            right_x += mid_x

        # 4. Tính centerline ảo
        center_x = None
        mode = 'LOST'

        if left_x is not None and right_x is not None:
            center_x = (left_x + right_x) // 2
            mode = 'BOTH'
        elif left_x is not None:
            center_x = left_x + self.cfg.lane_half_width_px
            mode = 'LEFT_ONLY'
        elif right_x is not None:
            center_x = right_x - self.cfg.lane_half_width_px
            mode = 'RIGHT_ONLY'

        return DualLaneResult(center_x=center_x, left_x=left_x, right_x=right_x, mode=mode)

    def get_execution_center(self, image) -> DualLaneResult:
        return self.detect(image, self.cfg.roi_y, self.cfg.roi_h)

    def get_lookahead_center(self, image) -> DualLaneResult:
        return self.detect(image, self.cfg.lookahead_y, self.cfg.lookahead_h)

    def is_line_visible(self, image) -> bool:
        res = self.get_execution_center(image)
        return res.mode != 'LOST'

    def _create_line_mask(self, roi):
        """Tạo binary mask lọc màu line."""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.cfg.line_hsv_lower, self.cfg.line_hsv_upper)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.cfg.morph_kernel_size, self.cfg.morph_kernel_size))
        if self.cfg.morph_open_iter > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.cfg.morph_open_iter)
        if self.cfg.morph_close_iter > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.cfg.morph_close_iter)

        return mask

    def _apply_focus_mask(self, mask):
        """Loại bỏ nhiễu mép."""
        h, w = mask.shape
        focus = np.zeros_like(mask)
        center_w = int(w * self.cfg.focus_width_percent)
        start_x = (w - center_w) // 2
        focus[:, start_x:start_x + center_w] = 255
        return cv2.bitwise_and(mask, focus)

    def _find_largest_centroid_x(self, mask):
        """Tìm tọa độ X của contour lớn nhất trong mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.cfg.min_contour_area:
            return None

        M = cv2.moments(largest)
        if M["m00"] > 0:
            return int(M["m10"] / M["m00"])
        return None

    def draw_debug(self, image, res_exec: DualLaneResult, res_look: DualLaneResult):
        """Vẽ thông tin debug lên ảnh."""
        if image is None:
            return None

        dbg = image.copy()
        
        # Vẽ khung ROI thực thi
        cv2.rectangle(dbg, (0, self.cfg.roi_y), (self.cfg.image_width - 1, self.cfg.roi_y + self.cfg.roi_h), (0, 255, 0), 1)
        # Vẽ khung ROI dự báo
        cv2.rectangle(dbg, (0, self.cfg.lookahead_y), (self.cfg.image_width - 1, self.cfg.lookahead_y + self.cfg.lookahead_h), (0, 255, 255), 1)

        # Vẽ kết quả ROI thực thi
        if res_exec.left_x is not None:
            cv2.circle(dbg, (res_exec.left_x, self.cfg.roi_y + self.cfg.roi_h // 2), 5, (255, 0, 0), -1) # Blue cho left
        if res_exec.right_x is not None:
            cv2.circle(dbg, (res_exec.right_x, self.cfg.roi_y + self.cfg.roi_h // 2), 5, (0, 255, 0), -1) # Green cho right
        if res_exec.center_x is not None:
            cv2.line(dbg, (res_exec.center_x, self.cfg.roi_y), (res_exec.center_x, self.cfg.roi_y + self.cfg.roi_h), (0, 0, 255), 2) # Red cho center

        # Vẽ kết quả ROI dự báo
        if res_look.center_x is not None:
            cv2.circle(dbg, (res_look.center_x, self.cfg.lookahead_y + self.cfg.lookahead_h // 2), 5, (0, 165, 255), -1) # Orange cho lookahead center

        # Vẽ trục giữa xe
        mid_x = self.cfg.image_width // 2
        for y in range(0, self.cfg.image_height, 10):
            cv2.line(dbg, (mid_x, y), (mid_x, y + 5), (255, 255, 255), 1)

        cv2.putText(dbg, f"Mode: {res_exec.mode}", (5, self.cfg.roi_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return dbg
