"""
Traffic Sign & Traffic Light Detector - Nhận diện biển báo và đèn giao thông
==============================================================================
Module sử dụng Computer Vision (HSV Color Detection + Contour Analysis) để:
  1. Nhận diện đèn giao thông: ĐỎ (dừng), XANH (đi tiếp)
  2. Nhận diện biển báo chỉ dẫn: THẲNG, RẼ TRÁI, RẼ PHẢI

Thiết kế nhẹ nhàng, tối ưu cho Jetson Nano (không dùng AI/Deep Learning).

Output ghi vào Blackboard:
  - 'traffic_light': 'RED' | 'GREEN' | 'NONE'
  - 'traffic_sign':  'STRAIGHT' | 'LEFT' | 'RIGHT' | 'NONE'
"""

import cv2
import numpy as np
import sys
import os

class DummySettings:
    TRAFFIC_ROI_TOP_RATIO = 0.5
    TRAFFIC_MIN_AREA_LIGHT = 100
    TRAFFIC_MIN_AREA_SIGN = 200
    TRAFFIC_HISTORY_LEN = 3
    TRAFFIC_RED_HSV_LOWER1 = [0, 100, 100]
    TRAFFIC_RED_HSV_UPPER1 = [10, 255, 255]
    TRAFFIC_RED_HSV_LOWER2 = [160, 100, 100]
    TRAFFIC_RED_HSV_UPPER2 = [180, 255, 255]
    TRAFFIC_GREEN_HSV_LOWER = [40, 50, 50]
    TRAFFIC_GREEN_HSV_UPPER = [90, 255, 255]
    TRAFFIC_BLUE_HSV_LOWER = [100, 100, 100]
    TRAFFIC_BLUE_HSV_UPPER = [140, 255, 255]

settings = DummySettings()

class TrafficSign:
    """Hằng số biển báo chỉ dẫn."""
    NONE     = "NONE"
    STRAIGHT = "STRAIGHT"
    LEFT     = "LEFT"
    RIGHT    = "RIGHT"


class TrafficLight:
    """Hằng số đèn giao thông."""
    NONE  = "NONE"
    RED   = "RED"
    GREEN = "GREEN"


class TrafficDetector:
    """
    Bộ nhận diện biển báo và đèn giao thông bằng Computer Vision thuần túy.

    Pipeline xử lý:
      1. Cắt vùng ROI (Region of Interest) phía trên ảnh (nơi biển/đèn thường xuất hiện).
      2. Chuyển sang không gian màu HSV.
      3. Lọc theo dải màu (Color Masking) để tách riêng Đỏ, Xanh Lá, Xanh Dương.
      4. Tìm contour lớn nhất -> phân loại hình dạng.
    """

    def __init__(self, image_width=None, image_height=None):
        self.width = image_width or settings.IMAGE_WIDTH
        self.height = image_height or settings.IMAGE_HEIGHT

        # Vùng ROI: chỉ quét nửa trên bức ảnh
        self.roi_y_start = 0
        self.roi_y_end = int(self.height * settings.TRAFFIC_ROI_TOP_RATIO)

        # Ngưỡng diện tích tối thiểu (pixel^2)
        self.min_area_light = settings.TRAFFIC_MIN_AREA_LIGHT
        self.min_area_sign  = settings.TRAFFIC_MIN_AREA_SIGN

        # === DẢI MÀU HSV (từ settings) ===
        self.red_lower1   = np.array(settings.TRAFFIC_RED_HSV_LOWER1)
        self.red_upper1   = np.array(settings.TRAFFIC_RED_HSV_UPPER1)
        self.red_lower2   = np.array(settings.TRAFFIC_RED_HSV_LOWER2)
        self.red_upper2   = np.array(settings.TRAFFIC_RED_HSV_UPPER2)
        self.green_lower  = np.array(settings.TRAFFIC_GREEN_HSV_LOWER)
        self.green_upper  = np.array(settings.TRAFFIC_GREEN_HSV_UPPER)
        self.blue_lower   = np.array(settings.TRAFFIC_BLUE_HSV_LOWER)
        self.blue_upper   = np.array(settings.TRAFFIC_BLUE_HSV_UPPER)

        # Bộ đệm ổn định kết quả (Voting)
        self._light_history = []
        self._sign_history  = []
        self._history_len   = settings.TRAFFIC_HISTORY_LEN

    # ----------------------------------------------------------
    # API chính: Xử lý 1 frame ảnh
    # ----------------------------------------------------------
    def detect(self, frame):
        """
        Phân tích 1 frame ảnh và trả về kết quả phát hiện.

        Args:
            frame: Ảnh BGR gốc từ camera (numpy array).

        Returns:
            light (str): TrafficLight.RED / GREEN / NONE
            sign (str):  TrafficSign.STRAIGHT / LEFT / RIGHT / NONE
        """
        if frame is None:
            return TrafficLight.NONE, TrafficSign.NONE

        resized = cv2.resize(frame, (self.width, self.height))

        # Cắt vùng ROI (nửa trên ảnh)
        roi = resized[self.roi_y_start:self.roi_y_end, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # --- 1. PHÁT HIỆN ĐÈN GIAO THÔNG ---
        raw_light = self._detect_traffic_light(hsv)

        # --- 2. PHÁT HIỆN BIỂN BÁO CHỈ DẪN ---
        raw_sign = self._detect_traffic_sign(hsv, roi)

        # --- 3. ỔN ĐỊNH KẾT QUẢ BẰNG VOTING (Đa số thắng) ---
        light = self._stabilize(raw_light, self._light_history)
        sign = self._stabilize(raw_sign, self._sign_history)

        return light, sign

    # ----------------------------------------------------------
    # Nhận diện đèn giao thông (Đỏ / Xanh)
    # ----------------------------------------------------------
    def _detect_traffic_light(self, hsv):
        """Lọc màu Đỏ và Xanh Lá trong vùng ROI."""
        # Lọc Đỏ (2 dải)
        mask_red1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask_red2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # Lọc Xanh Lá
        mask_green = cv2.inRange(hsv, self.green_lower, self.green_upper)

        # Tìm contour lớn nhất cho từng màu
        red_area = self._largest_contour_area(mask_red)
        green_area = self._largest_contour_area(mask_green)

        # Quyết định dựa trên diện tích lớn hơn
        if red_area > self.min_area_light and red_area > green_area:
            return TrafficLight.RED
        elif green_area > self.min_area_light and green_area > red_area:
            return TrafficLight.GREEN

        return TrafficLight.NONE

    # ----------------------------------------------------------
    # Nhận diện biển báo chỉ dẫn (Thẳng / Trái / Phải)
    # ----------------------------------------------------------
    def _detect_traffic_sign(self, hsv, roi_bgr):
        """
        Phát hiện biển báo nền xanh dương, sau đó phân tích
        hướng mũi tên bên trong bằng thuật toán phân tích trọng tâm (Centroid).
        """
        mask_blue = cv2.inRange(hsv, self.blue_lower, self.blue_upper)

        # Tìm contour biển báo
        contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return TrafficSign.NONE

        # Lấy contour lớn nhất
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < self.min_area_sign:
            return TrafficSign.NONE

        # Cắt bounding box của biển báo
        x, y, w, h = cv2.boundingRect(largest)
        sign_crop = roi_bgr[y:y+h, x:x+w]

        if sign_crop.size == 0:
            return TrafficSign.NONE

        # Chuyển sang ảnh xám, nhị phân hóa phần trắng (mũi tên) bên trong biển
        gray_sign = cv2.cvtColor(sign_crop, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray_sign, 180, 255, cv2.THRESH_BINARY)

        # Phân tích hướng mũi tên bằng trọng tâm khối trắng
        return self._classify_arrow_direction(white_mask)

    # ----------------------------------------------------------
    # Phân loại hướng mũi tên
    # ----------------------------------------------------------
    def _classify_arrow_direction(self, white_mask):
        """
        Phân loại hướng mũi tên dựa trên vị trí trọng tâm khối trắng.
        - Trọng tâm lệch trái  -> Mũi tên chỉ TRÁI
        - Trọng tâm lệch phải  -> Mũi tên chỉ PHẢI
        - Trọng tâm ở giữa     -> Mũi tên chỉ THẲNG
        """
        moments = cv2.moments(white_mask)
        if moments['m00'] == 0:
            return TrafficSign.NONE

        cx = int(moments['m10'] / moments['m00'])
        h, w = white_mask.shape[:2]

        # Chia bức ảnh thành 3 vùng ngang: Trái (0-35%), Giữa (35-65%), Phải (65-100%)
        left_boundary = int(w * 0.35)
        right_boundary = int(w * 0.65)

        if cx < left_boundary:
            return TrafficSign.LEFT
        elif cx > right_boundary:
            return TrafficSign.RIGHT
        else:
            return TrafficSign.STRAIGHT

    # ----------------------------------------------------------
    # Tiện ích
    # ----------------------------------------------------------
    def _largest_contour_area(self, mask):
        """Tìm diện tích contour lớn nhất trong ảnh mask nhị phân."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0
        return cv2.contourArea(max(contours, key=cv2.contourArea))

    def _stabilize(self, raw_value, history):
        """Ổn định kết quả bằng phương pháp Voting (đa số thắng) trên N frame gần nhất."""
        history.append(raw_value)
        if len(history) > self._history_len:
            history.pop(0)

        # Đếm tần suất và lấy giá trị xuất hiện nhiều nhất
        counts = {}
        for v in history:
            counts[v] = counts.get(v, 0) + 1
        return max(counts, key=counts.get)

    # ----------------------------------------------------------
    # Tích hợp Blackboard
    # ----------------------------------------------------------
    def process(self, blackboard):
        """
        Hàm giao tiếp chuẩn với kiến trúc Blackboard.
        Đọc ảnh từ Blackboard, phân tích, ghi kết quả trở lại.
        """
        latest_image = blackboard.get('latest_image')
        light, sign = self.detect(latest_image)

        blackboard.set('traffic_light', light)
        blackboard.set('traffic_sign', sign)
