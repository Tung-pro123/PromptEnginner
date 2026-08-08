# -*- coding: utf-8 -*-
"""
CameraProcessor: Xử lý ảnh bám làn Dual-Filter (HSV Red + White Background) 
kết hợp phân loại biên theo ngữ cảnh State-Aware Segment Clustering.
"""

import cv2
import numpy as np
import sys
import os

# Nạp cấu hình settings
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import settings

class CameraProcessor:
    def __init__(self, blackboard=None):
        self.blackboard = blackboard
        self.WIDTH = settings.IMAGE_WIDTH
        self.HEIGHT = settings.IMAGE_HEIGHT
        self.estimated_lane_width = settings.DEFAULT_LANE_WIDTH
        self.last_known_direction = 0.0

    def process_frame(self, frame, fsm_state=1, dodge_direction=0.0, current_offset_px=0.0):
        """
        Xử lý khung hình camera và trả về:
        (C_near, C_far, y_near, y_far, debug_frame)
        """
        if frame is None:
            return settings.IMAGE_CENTER_X, settings.IMAGE_CENTER_X, int(self.HEIGHT * 0.85), int(self.HEIGHT * 0.55), None

        # 1. Resize ảnh về kích thước chuẩn (300x300)
        resized = cv2.resize(frame, (self.WIDTH, self.HEIGHT))
        
        y_near = int(self.HEIGHT * settings.ROI_Y_NEAR_RATIO)
        y_far = int(self.HEIGHT * settings.ROI_Y_FAR_RATIO)

        # 2. Xử lý màu kép (Dual-Filter): Lọc vạch ĐỎ + Lọc nền TRẮNG
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        
        # Lọc vạch màu ĐỎ (Red Lines: viền biên đỏ & vạch đứt đỏ ở giữa)
        lower_red1 = np.array(settings.HSV_RED_LOWER1)
        upper_red1 = np.array(settings.HSV_RED_UPPER1)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        
        lower_red2 = np.array(settings.HSV_RED_LOWER2)
        upper_red2 = np.array(settings.HSV_RED_UPPER2)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Lọc nền TRẮNG bên ngoài lòng đường đen
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, settings.WHITE_BACKGROUND_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Mặt nạ biên tổng hợp (Vạch Đỏ + Nền Trắng = Vùng không phải lòng đường đen)
        thresh = cv2.bitwise_or(red_mask, white_mask)

        # 3. Phân cụm vạch biên & gán nhãn theo ngữ cảnh trạng thái (State-Aware Segment Clustering)
        def find_borders(y_line):
            white_xs = [x for x in range(self.WIDTH) if thresh[y_line, x] == 255]
            
            segments = []
            if len(white_xs) > 0:
                current_segment = [white_xs[0]]
                for x in white_xs[1:]:
                    if x - current_segment[-1] > settings.MAX_GAP_BETWEEN_POINTS:
                        segments.append(int(np.mean(current_segment)))
                        current_segment = [x]
                    else:
                        current_segment.append(x)
                segments.append(int(np.mean(current_segment)))

            left_border = 0
            right_border = self.WIDTH - 1
            found_left = False
            found_right = False

            if len(segments) >= 2:
                # Tìm thấy >= 2 vạch -> Lấy 2 vạch ngoài cùng làm biên
                left_border = segments[0]
                right_border = segments[-1]
                found_left = True
                found_right = True
            elif len(segments) == 1:
                x_val = segments[0]
                # Phân loại phân vùng (Zone-Based) kết hợp FSM Prior chống chao đảo bánh lái:
                if x_val < settings.ZONE_LEFT_MAX:
                    left_border = x_val
                    found_left = True
                elif x_val > settings.ZONE_RIGHT_MIN:
                    right_border = x_val
                    found_right = True
                else:
                    # Vùng trung tâm (110 <= x <= 190): Định hướng theo trạng thái né
                    if fsm_state in [2, 3]: # DODGING hoặc REENTERING
                        if dodge_direction == -1.0: # Đang né trái -> Vạch trung tâm là biên trái
                            left_border = x_val
                            found_left = True
                        else: # Đang né phải -> Vạch trung tâm là biên phải
                            right_border = x_val
                            found_right = True
                    else:
                        if x_val < self.WIDTH / 2.0:
                            left_border = x_val
                            found_left = True
                        else:
                            right_border = x_val
                            found_right = True

            # Khôi phục biên đơn thích nghi (EMA Lane Width Estimation)
            if found_left and found_right:
                width = right_border - left_border
                if 160 < width < 280:
                    self.estimated_lane_width = 0.9 * self.estimated_lane_width + 0.1 * width
                center_x = int((left_border + right_border) / 2)
                self.last_known_direction = np.sign(center_x - self.WIDTH / 2.0)
            elif found_left:
                right_border = int(left_border + self.estimated_lane_width)
                center_x = int(left_border + self.estimated_lane_width / 2)
                self.last_known_direction = np.sign(center_x - self.WIDTH / 2.0)
            elif found_right:
                left_border = int(right_border - self.estimated_lane_width)
                center_x = int(right_border - self.estimated_lane_width / 2)
                self.last_known_direction = np.sign(center_x - self.WIDTH / 2.0)
            else:
                center_x = int(self.WIDTH / 2.0 + self.last_known_direction * (self.estimated_lane_width / 4.0))
                
            left_border = max(0, min(self.WIDTH - 1, left_border))
            right_border = max(0, min(self.WIDTH - 1, right_border))
            center_x = max(0, min(self.WIDTH - 1, center_x))
            
            return center_x, left_border, right_border

        C_near, L_near, R_near = find_borders(y_near)
        C_far, L_far, R_far = find_borders(y_far)

        # 4. Vẽ khung hình debug trực quan
        debug_frame = resized.copy()
        cv2.line(debug_frame, (0, y_near), (self.WIDTH, y_near), (0, 255, 255), 1)
        cv2.line(debug_frame, (0, y_far), (self.WIDTH, y_far), (0, 255, 255), 1)
        cv2.circle(debug_frame, (L_near, y_near), 5, (0, 0, 255), -1)
        cv2.circle(debug_frame, (R_near, y_near), 5, (0, 0, 255), -1)
        cv2.circle(debug_frame, (C_near, y_near), 6, (0, 255, 0), -1)
        cv2.circle(debug_frame, (C_far, y_far), 6, (0, 255, 0), -1)

        if self.blackboard:
            self.blackboard.set('C_near', C_near)
            self.blackboard.set('C_far', C_far)
            self.blackboard.set('y_near', y_near)
            self.blackboard.set('y_far', y_far)

        return C_near, C_far, y_near, y_far, debug_frame
