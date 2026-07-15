#!/usr/bin/env python3
import sys
import os

# Sắp xếp lại sys.path để ưu tiên thư viện Python 3, tránh xung đột với ROS Python 2.7
py3_paths = [p for p in sys.path if 'python2.7' not in p]
py2_paths = [p for p in sys.path if 'python2.7' in p]
sys.path = py3_paths + py2_paths

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rospy
from sensor_msgs.msg import LaserScan, Image
import cv2
import numpy as np
import math

from config import settings
from fsm.fsm_manager import FSMManager
from control.racer_controller import RacerController
from perception.camera.camera_processor import CameraProcessor
from perception.lidar.lidar_processor import LidarProcessor
from debug.debugger import Debugger

class ROSSpeedTrackNode:
    """Node ROS sử dụng kiến trúc module để chạy robot"""
    def __init__(self):
        rospy.init_node('speed_track_modular_node', anonymous=True)
        
        # Khởi tạo các module cốt lõi (Core Modules)
        self.fsm = FSMManager()
        self.controller = RacerController()
        self.camera = CameraProcessor()
        self.lidar = LidarProcessor()
        self.debugger = Debugger(debug_mode=True)
        
        self.controller.initialize()
        self.camera.initialize()
        self.lidar.initialize()
        
        self.latest_image = None
        self.latest_scan = None
        
        # ROS Subscribers
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera_callback)
        
        rospy.loginfo("Node ROS Speed Track (Modular) đã khởi động thành công.")
        
    def lidar_callback(self, msg):
        """Chuyển đổi dữ liệu LaserScan ROS thành list[(angle_deg, dist)]"""
        scan_data = []
        for i, dist in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            angle_deg = math.degrees(angle)
            
            # Điều chỉnh góc Lidar theo hướng lắp đặt (quay 180 độ)
            angle_deg = angle_deg + 180.0
            angle_deg = (angle_deg + 180) % 360 - 180
            
            if msg.range_min < dist < msg.range_max:
                scan_data.append((angle_deg, dist))
                
        self.latest_scan = scan_data

    def camera_callback(self, msg):
        """Chuyển đổi dữ liệu ảnh ROS Image thành numpy array OpenCV"""
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding == 'bgr8':
                self.latest_image = img.reshape((msg.height, msg.width, 3))
            elif msg.encoding == 'rgb8':
                img_rgb = img.reshape((msg.height, msg.width, 3))
                self.latest_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'mono8':
                self.latest_image = img.reshape((msg.height, msg.width))
        except Exception as e:
            rospy.logerr(f"Lỗi chuyển đổi ảnh: {e}")

    def run(self):
        """Vòng lặp điều khiển chính chạy ở 20Hz"""
        rate = rospy.Rate(20)
        
        while not rospy.is_shutdown():
            if self.latest_image is None or self.latest_scan is None:
                rospy.logwarn_throttle(2, "Đang chờ dữ liệu từ Camera và Lidar...")
                rate.sleep()
                continue
                
            # 1. Thu thập dữ liệu Lidar & Cập nhật FSM
            front_dist, closest_angle, side_clear = self.lidar.process_scan(self.latest_scan)
            
            self.fsm.update_from_lidar(front_dist, closest_angle, side_clear)
            current_offset_px = self.fsm.update_offset()
            state_name = self.fsm.get_state_name()
            
            # 2. Xử lý ảnh Camera (Line tracking)
            center_x = self.camera.process_frame(self.latest_image, self.fsm.dodge_direction)
                
            # 3. Tính toán góc lái (PID) & Ra lệnh động cơ
            steering = self.controller.calculate_steering(center_x, current_offset_px)
            self.controller.move(settings.BASE_SPEED, steering)
            
            # 4. Ghi log debug CSV
            self.debugger.log_csv(state_name, front_dist, closest_angle, front_dist, current_offset_px, steering)
            rospy.loginfo_throttle(1, f"[{state_name}] offset: {current_offset_px:.1f}, steer: {steering:.3f}, f_dist: {front_dist:.2f}")
            
            rate.sleep()

        # Đóng an toàn khi người dùng nhấn Ctrl+C
        rospy.loginfo("Dừng hệ thống, xả ga và tắt lưu log.")
        self.controller.stop()
        self.debugger.close()

if __name__ == '__main__':
    try:
        node = ROSSpeedTrackNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print(f"Lỗi ngoài ý muốn: {e}")
        try:
            r = RacerController()
            r.stop()
        except:
            pass
