#!/usr/bin/env python3
"""
Crosswalk Detector
Phát hiện vạch qua đường bằng cách tính row sum của các pixel trắng.
"""

import cv2
import numpy as np


class CrosswalkDetector:
    def __init__(self, config):
        self.cfg = config
        self.last_detect_time = 0.0

    def detect(self, image) -> bool:
        """
        Kiểm tra xem có vạch qua đường trong ROI không.
        """
        if image is None:
            return False

        # Cắt ROI (thường nằm cao hơn ROI bám line một chút)
        roi = image[self.cfg.crosswalk_y:self.cfg.crosswalk_y + self.cfg.crosswalk_h, :]
        
        # Tái sử dụng logic lọc màu từ config
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.cfg.line_hsv_lower, self.cfg.line_hsv_upper)
        
        # Morphological open/close để giảm nhiễu
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.cfg.morph_kernel_size, self.cfg.morph_kernel_size))
        if self.cfg.morph_open_iter > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.cfg.morph_open_iter)
        if self.cfg.morph_close_iter > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.cfg.morph_close_iter)

        # Tính tổng số pixel trắng trên mỗi hàng ngang
        # mask > 0 sẽ cho mảng boolean, sum(axis=1) tính tổng theo hàng
        row_sums = np.sum(mask > 0, axis=1)
        
        # Một hàng được coi là "rộng" nếu số pixel trắng vượt quá một ngưỡng % của chiều rộng ROI
        width = roi.shape[1]
        wide_rows_count = np.sum(row_sums > (width * self.cfg.crosswalk_row_fill_ratio))

        # Nếu có đủ số hàng ngang thỏa mãn, coi như phát hiện crosswalk
        if wide_rows_count >= self.cfg.crosswalk_min_rows:
            return True
            
        return False
        
    def draw_debug(self, image, detected: bool):
        """Vẽ khung ROI của crosswalk để debug."""
        if image is None:
            return None
        dbg = image.copy()
        color = (0, 0, 255) if detected else (255, 255, 0) # Đỏ nếu detect, Cyan nếu không
        cv2.rectangle(dbg, (0, self.cfg.crosswalk_y), (self.cfg.image_width - 1, self.cfg.crosswalk_y + self.cfg.crosswalk_h), color, 2)
        if detected:
            cv2.putText(dbg, "CROSSWALK!", (5, self.cfg.crosswalk_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return dbg
