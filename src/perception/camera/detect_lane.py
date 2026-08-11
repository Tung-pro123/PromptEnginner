#!/usr/bin/env python3
"""
Module Phát hiện Lane (Lane Detector) cho Speed Track.
Thuật toán: Hybrid Lane Detection (Cách B)
- Quét tìm 2 đường biên trắng/đen
- Tìm vạch đứt khúc ở giữa trong vùng an toàn giữa 2 biên
- Fallback: Trả về trung điểm 2 biên nếu mất vạch
"""

import cv2
import numpy as np

class LaneDetector:
    def __init__(self, image_width=300, image_height=300):
        self.W = image_width
        self.H = image_height
        self.GRAY_THRESH = 180

    def process(self, frame):
        """
        Xử lý ảnh để tìm mục tiêu điều khiển.
        Returns:
            target_x (int): Điểm mục tiêu (tâm vạch giữa hoặc trung điểm 2 biên)
            L_n (int): Tọa độ biên trái (gần)
            R_n (int): Tọa độ biên phải (gần)
            has_center (bool): Có tìm thấy vạch giữa hay không
            dbg (np.array): Ảnh debug
        """
        # Nếu frame chưa đúng kích thước thì resize
        if frame.shape[1] != self.W or frame.shape[0] != self.H:
            resized = cv2.resize(frame, (self.W, self.H))
        else:
            resized = frame.copy()
            
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.GRAY_THRESH, 255, cv2.THRESH_BINARY)

        # Quét ở 2 khoảng cách: gần và xa
        y_near = int(self.H * 0.85)
        y_far = int(self.H * 0.55)

        def find_borders(y):
            mid = self.W // 2
            L, R = 0, self.W - 1
            # Quét từ giữa ra lề trái
            for x in range(mid, 0, -1):
                if thresh[y, x] == 255: 
                    L = x
                    break
            # Quét từ giữa ra lề phải
            for x in range(mid, self.W):
                if thresh[y, x] == 255: 
                    R = x
                    break
            return L, R, (L + R) // 2

        L_n, R_n, mid_n = find_borders(y_near)
        L_f, R_f, mid_f = find_borders(y_far)

        # ---------------------------------------------------------
        # Tìm vạch trắng đứt khúc giữa bằng contour trong vùng an toàn
        # ---------------------------------------------------------
        roi_y = int(self.H * 0.70)
        roi_h = int(self.H * 0.25)
        roi = thresh[roi_y:roi_y+roi_h, :]
        
        # Mask chỉ giữ vùng giữa 2 biên (loại bỏ đường biên cản trở)
        margin = 15  # pixel margin tránh biên
        mask_roi = np.zeros_like(roi)
        
        left_safe = min(L_n, L_f) + margin
        right_safe = max(R_n, R_f) - margin
        
        if left_safe < right_safe:
            mask_roi[:, left_safe:right_safe] = roi[:, left_safe:right_safe]
            
        # Tìm contour (tương thích OpenCV 3 và OpenCV 4)
        cnts = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cnts[0] if len(cnts) == 2 else cnts[1]

        center_line_x = None
        has_center = False
        if contours:
            largest = max(contours, key=cv2.contourArea)
            # Chỉ coi là vạch đứt khúc nếu diện tích đủ lớn (tránh nhiễu đốm nhỏ)
            if cv2.contourArea(largest) > 60:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    center_line_x = int(M["m10"] / M["m00"])
                    has_center = True

        # =========================================================
        # TARGET SELECTION (FALLBACK LOGIC)
        # =========================================================
        # Ưu tiên bám vạch giữa. Nếu không có vạch (do rẽ gắt hoặc bóng râm), lấy trung điểm 2 lề đường.
        target_x = center_line_x if has_center else mid_n

        # =========================================================
        # DEBUG VISUALIZATION
        # =========================================================
        dbg = resized.copy()
        # Vẽ vạch quét gần (màu vàng)
        cv2.line(dbg, (0, y_near), (self.W, y_near), (0, 255, 255), 1)
        # Vẽ 2 biên quét được
        cv2.circle(dbg, (L_n, y_near), 4, (0, 0, 255), -1)
        cv2.circle(dbg, (R_n, y_near), 4, (0, 0, 255), -1)
        # Vẽ điểm trung điểm biên (Fallback)
        cv2.circle(dbg, (mid_n, y_near), 5, (255, 0, 0), -1)
        
        # Vẽ điểm vạch giữa (nếu có)
        if has_center:
            cv2.circle(dbg, (center_line_x, roi_y + roi_h//2), 5, (0, 255, 0), -1)

        return target_x, L_n, R_n, has_center, dbg
