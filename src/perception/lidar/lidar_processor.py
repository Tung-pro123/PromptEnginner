import sys
import os
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from perception.lidar.base_lidar_processor import BaseLidarProcessor
from config import settings

class LidarProcessor(BaseLidarProcessor):
    def __init__(self, blackboard=None):
        self.blackboard = blackboard
        self.scan_data = [] # Giả lập danh sách các tuple (angle, distance)

    def initialize(self):
        print("[INFO] LidarProcessor initialized.")

    def get_scan(self):
        return self.scan_data

    def ros_callback(self, msg):
        """Chuyển đổi dữ liệu LaserScan ROS thành list[(angle_deg, dist)]"""
        import rospy
        rospy.loginfo_throttle(1.0, "[DEBUG] LidarProcessor: ĐÃ NHẬN được gói tin quét Laser từ ROS Topic!")
        scan_data = []
        for i, dist in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            angle_deg = math.degrees(angle)
            
            # Điều chỉnh góc Lidar theo hướng lắp đặt (quay 180 độ)
            angle_deg = angle_deg + 180.0
            angle_deg = (angle_deg + 180) % 360 - 180
            
            if msg.range_min < dist < msg.range_max:
                scan_data.append((angle_deg, dist))
                
        if self.blackboard:
            self.blackboard.set('latest_scan', scan_data)
        else:
            self.scan_data = scan_data

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

    def process(self, blackboard):
        scan_data = blackboard.get('latest_scan', [])
        front_dist, closest_angle, side_clear = self.process_scan(scan_data)
        blackboard.set('front_dist', front_dist)
        blackboard.set('closest_angle', closest_angle)
        blackboard.set('side_clear', side_clear)
