import sys
import os
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from perception.lidar.base_lidar_processor import BaseLidarProcessor
from config import settings

class LidarProcessor(BaseLidarProcessor):
    def __init__(self):
        self.scan_data = [] # Giả lập danh sách các tuple (angle, distance)

    def initialize(self):
        print("[INFO] LidarProcessor initialized.")

    def get_scan(self):
        # Trong thực tế, đọc từ topic ROS /scan hoặc giao tiếp Serial
        return self.scan_data

    def process_scan(self, scan_data):
        """
        Phân tích mảng dữ liệu Lidar.
        Returns: (front_dist, closest_angle, side_clear)
        """
        front_dist = 999.0
        closest_angle = 0.0
        side_clear = True
        
        if not scan_data:
            return front_dist, closest_angle, side_clear

        for angle, dist in scan_data:
            # Nếu giá trị nhiễu hoặc quá nhỏ thì bỏ qua
            if dist < 0.05: continue
            
            # Góc quét mặt trước (ví dụ -35 đến 35 độ)
            if -settings.FRONT_ANGLE_RANGE <= angle <= settings.FRONT_ANGLE_RANGE:
                if dist < front_dist:
                    front_dist = dist
                    closest_angle = angle
                    
            # Góc quét sườn xe để tính an toàn nhập làn
            if angle > settings.SIDE_ANGLE_CLEAR or angle < -settings.SIDE_ANGLE_CLEAR:
                # Nếu có bất kỳ vật cản nào sườn xe gần hơn SIDE_CLEAR_DIST
                if dist < settings.SIDE_CLEAR_DIST:
                    side_clear = False
                    
        return front_dist, closest_angle, side_clear
