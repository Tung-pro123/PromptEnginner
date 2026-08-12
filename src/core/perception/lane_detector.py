#!/usr/bin/env python3
"""
Module Phát hiện Lane (Lane Detector) cho Speed Track.

Xử lý ảnh OpenCV thuần để tìm vạch kẻ trắng đứt khúc giữa đường đua.
Sử dụng Dual-ROI (ROI gần + ROI xa) cho việc dự báo đường đi.
"""

import cv2
import numpy as np


class LaneDetector:
    """Phát hiện đường kẻ trắng đứt khúc bằng xử lý ảnh OpenCV.
    
    Sa bàn Speed Track: đường đua tối (đen), line trắng đứt khúc ở giữa.
    """

    def __init__(self, width=300, height=300):
        """
        Args:
            width: Chiều rộng ảnh xử lý (pixels)
            height: Chiều cao ảnh xử lý (pixels)
        """
        self.WIDTH = width
        self.HEIGHT = height

        # === CẤU HÌNH ROI ===
        # ROI thực thi (gần, dùng để bám lane hiện tại)
        self.ROI_Y = int(self.HEIGHT * 0.80)
        self.ROI_H = int(self.HEIGHT * 0.18)

        # ROI dự báo (xa hơn, dùng để phát hiện sớm mất line / cua gấp)
        self.LOOKAHEAD_ROI_Y = int(self.HEIGHT * 0.55)
        self.LOOKAHEAD_ROI_H = int(self.HEIGHT * 0.15)

        # === CẤU HÌNH LỌC MÀU ===
        # Lọc line trắng trên đường tối
        # HLS: tìm vùng sáng (Lightness cao) bất kể Hue
        self.USE_HLS = True
        self.HLS_L_MIN = 150  # Ngưỡng Lightness tối thiểu cho line trắng
        self.HLS_S_MAX = 60   # Loại bỏ màu bão hoà cao (không phải trắng)

        # HSV backup: nếu USE_HLS=False
        self.HSV_WHITE_LOWER = np.array([0, 0, 180])    # V cao = sáng
        self.HSV_WHITE_UPPER = np.array([180, 60, 255])  # S thấp = trắng

        # === CẤU HÌNH CONTOUR ===
        self.MIN_CONTOUR_AREA = 80  # Diện tích tối thiểu contour hợp lệ
        self.ROI_CENTER_WIDTH_PERCENT = 0.7  # Focus mask width (70% giữa ảnh)

    def get_line_center(self, image, roi_y=None, roi_h=None):
        """Tìm tọa độ X trọng tâm của vạch kẻ đường trong ROI.
        
        Args:
            image: Ảnh BGR đầu vào (đã resize về WIDTH x HEIGHT)
            roi_y: Vị trí Y bắt đầu ROI (None = dùng ROI thực thi mặc định)
            roi_h: Chiều cao ROI (None = dùng ROI thực thi mặc định)
            
        Returns:
            int hoặc None: Tọa độ X trọng tâm, None nếu không tìm thấy
        """
        if image is None:
            return None

        if roi_y is None:
            roi_y = self.ROI_Y
        if roi_h is None:
            roi_h = self.ROI_H

        # Cắt ROI
        roi = image[roi_y:roi_y + roi_h, :]

        # Tạo mặt nạ phát hiện line trắng
        mask = self._create_white_mask(roi)

        # Áp dụng focus mask (chỉ giữ vùng giữa ảnh)
        mask = self._apply_focus_mask(mask)

        # Tìm contour lớn nhất
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) < self.MIN_CONTOUR_AREA:
            return None

        M = cv2.moments(largest)
        if M["m00"] > 0:
            return int(M["m10"] / M["m00"])

        return None

    def get_execution_center(self, image):
        """Lấy tâm line từ ROI thực thi (gần xe)."""
        return self.get_line_center(image, self.ROI_Y, self.ROI_H)

    def get_lookahead_center(self, image):
        """Lấy tâm line từ ROI dự báo (xa phía trước)."""
        return self.get_line_center(image, self.LOOKAHEAD_ROI_Y, self.LOOKAHEAD_ROI_H)

    def is_line_visible(self, image):
        """Kiểm tra nhanh xem line có hiển thị trong cả 2 ROI không."""
        exec_center = self.get_execution_center(image)
        look_center = self.get_lookahead_center(image)
        return exec_center is not None and look_center is not None

    def _create_white_mask(self, roi):
        """Tạo binary mask phát hiện vạch trắng.
        
        Args:
            roi: Ảnh BGR đã cắt ROI
            
        Returns:
            np.array: Binary mask (0/255)
        """
        if self.USE_HLS:
            hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
            # Line trắng: Lightness cao, Saturation thấp
            l_channel = hls[:, :, 1]
            s_channel = hls[:, :, 2]
            mask = ((l_channel >= self.HLS_L_MIN) & (s_channel <= self.HLS_S_MAX)).astype(np.uint8) * 255
        else:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.HSV_WHITE_LOWER, self.HSV_WHITE_UPPER)

        # Morphological operations để loại noise và nối các đoạn đứt khúc
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return mask

    def _apply_focus_mask(self, mask):
        """Áp dụng focus mask để chỉ giữ vùng giữa ảnh.
        
        Giúp loại bỏ nhiễu từ mép sa bàn hoặc vật thể bên rìa.
        """
        h, w = mask.shape
        focus = np.zeros_like(mask)

        center_w = int(w * self.ROI_CENTER_WIDTH_PERCENT)
        start_x = (w - center_w) // 2
        end_x = start_x + center_w

        focus[:, start_x:end_x] = 255

        return cv2.bitwise_and(mask, focus)

    def draw_debug(self, image):
        """Vẽ thông tin debug lên ảnh.
        
        Args:
            image: Ảnh gốc BGR
            
        Returns:
            np.array: Ảnh với debug overlay
        """
        if image is None:
            return None

        debug = image.copy()

        # Vẽ ROI thực thi (xanh lá)
        cv2.rectangle(debug, 
                     (0, self.ROI_Y), 
                     (self.WIDTH - 1, self.ROI_Y + self.ROI_H), 
                     (0, 255, 0), 1)

        # Vẽ ROI dự báo (vàng)
        cv2.rectangle(debug,
                     (0, self.LOOKAHEAD_ROI_Y),
                     (self.WIDTH - 1, self.LOOKAHEAD_ROI_Y + self.LOOKAHEAD_ROI_H),
                     (0, 255, 255), 1)

        # Vẽ tâm line thực thi (đỏ)
        exec_center = self.get_execution_center(image)
        if exec_center is not None:
            cv2.line(debug, 
                    (exec_center, self.ROI_Y), 
                    (exec_center, self.ROI_Y + self.ROI_H), 
                    (0, 0, 255), 2)

        # Vẽ tâm line dự báo (cam)
        look_center = self.get_lookahead_center(image)
        if look_center is not None:
            cv2.line(debug,
                    (look_center, self.LOOKAHEAD_ROI_Y),
                    (look_center, self.LOOKAHEAD_ROI_Y + self.LOOKAHEAD_ROI_H),
                    (0, 165, 255), 2)

        # Vẽ đường tâm ảnh (trắng, đứt nét)
        mid_x = self.WIDTH // 2
        for y in range(0, self.HEIGHT, 10):
            cv2.line(debug, (mid_x, y), (mid_x, y + 5), (255, 255, 255), 1)

        return debug
