# -*- coding: utf-8 -*-
"""
LidarProcessor: Quét vật cản LiDAR 2D và vẽ Radar Overlay thu nhỏ.
"""

import math
import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import settings

class LidarProcessor:
    def __init__(self, blackboard=None):
        self.blackboard = blackboard

    def get_front_obstacle_distance(self, scan_msg):
        """Đo khoảng cách vật cản trước mặt trong hình nêm quét ±35 độ."""
        if scan_msg is None:
            return float('inf')
        
        distances = []
        for i, dist in enumerate(scan_msg.ranges):
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            angle_deg = math.degrees(angle) + 180.0
            angle_deg = (angle_deg + 180) % 360 - 180
            
            if settings.FRONT_ANGLE_MIN <= angle_deg <= settings.FRONT_ANGLE_MAX:
                if scan_msg.range_min < dist < scan_msg.range_max:
                    distances.append(dist)
                    
        front_dist = min(distances) if distances else float('inf')
        if self.blackboard:
            self.blackboard.set('front_dist', front_dist)
        return front_dist

    def get_closest_obstacle_angle_in_range(self, scan_msg, min_angle_deg, max_angle_deg, max_dist=0.80):
        """Tìm góc và khoảng cách của vật cản gần nhất trong dải quét."""
        if scan_msg is None:
            return None, float('inf')
        
        min_dist = float('inf')
        closest_angle = None
        
        for i, dist in enumerate(scan_msg.ranges):
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            angle_deg = math.degrees(angle) + 180.0
            angle_deg = (angle_deg + 180) % 360 - 180
            
            if min_angle_deg <= angle_deg <= max_angle_deg:
                if scan_msg.range_min < dist < max_dist:
                    if dist < min_dist:
                        min_dist = dist
                        closest_angle = angle_deg
                        
        return closest_angle, min_dist

    def draw_lidar_radar(self, img, scan_msg):
        """Vẽ bản đồ Radar LiDAR 2D thu nhỏ (80x80) ở góc trên bên phải khung hình debug."""
        if scan_msg is None or img is None:
            return img
            
        radar_size = 80
        center_x = settings.IMAGE_WIDTH - radar_size // 2 - 10
        center_y = radar_size // 2 + 10
        radius = radar_size // 2
        
        # 1. Vẽ vòng tròn nền
        cv2.circle(img, (center_x, center_y), radius, (30, 30, 30), -1)
        cv2.circle(img, (center_x, center_y), radius // 2, (70, 70, 70), 1)
        cv2.circle(img, (center_x, center_y), radius, (100, 100, 100), 1)
        
        # 2. Mũi xe hướng lên
        cv2.line(img, (center_x, center_y), (center_x, center_y - 6), (0, 0, 255), 2)
        
        # 3. Vẽ điểm LiDAR
        max_dist_visualize = 1.5
        scale = radius / max_dist_visualize
        
        for i, dist in enumerate(scan_msg.ranges):
            if math.isfinite(dist) and scan_msg.range_min < dist < max_dist_visualize:
                angle = scan_msg.angle_min + i * scan_msg.angle_increment
                angle_deg = math.degrees(angle) + 180.0
                angle_deg = (angle_deg + 180) % 360 - 180
                
                rad = math.radians(angle_deg)
                px = int(center_x - (dist * math.sin(rad)) * scale)
                py = int(center_y - (dist * math.cos(rad)) * scale)
                
                dist_to_center = math.sqrt((px - center_x) ** 2 + (py - center_y) ** 2)
                if dist_to_center <= radius:
                    if settings.FRONT_ANGLE_MIN <= angle_deg <= settings.FRONT_ANGLE_MAX and dist < settings.TRIGGER_DIST:
                        cv2.circle(img, (px, py), 1, (0, 0, 255), -1) # Màu đỏ = nguy hiểm
                    else:
                        cv2.circle(img, (px, py), 1, (0, 255, 0), -1) # Màu xanh = an toàn
                        
        return img
